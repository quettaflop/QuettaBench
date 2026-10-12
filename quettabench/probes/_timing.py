"""Probe device layer (backend, sync, names, event timing), CUDA-graph replay and NCU durations."""

from __future__ import annotations

import csv
import importlib.util
import io
import statistics as st
import subprocess
import sys
import time

import torch


def backend() -> str:
    """cuda, rocm, cpu, or tpu / neuron when their torch plugin is installed (probes refuse those)."""
    if torch.cuda.is_available():
        return "rocm" if torch.version.hip else "cuda"
    for name, module in (("tpu", "torch_xla"), ("neuron", "torch_neuronx")):
        if importlib.util.find_spec(module):
            return name
    return "cpu"


def require_backend(what: str, *supported: str) -> None:
    if backend() not in supported:
        raise SystemExit(f"{what} runs on {', '.join(supported)}, not {backend()}")


def require_ncu() -> None:
    if backend() != "cuda":
        raise SystemExit(f"--ncu needs cuda; the {backend()} kernel profiler (rocprofv3) is not wired yet")


def device() -> str:
    """ROCm builds of torch use the cuda device type."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def device_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def device_capability() -> tuple[int, int]:
    return torch.cuda.get_device_capability()


def device_count() -> int:
    return torch.cuda.device_count()


def set_device(index: int) -> None:
    if torch.cuda.is_available():
        torch.cuda.set_device(index)


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def time_once_ms(fn) -> float:
    """One synced call; the caller runs any barrier first."""
    if not torch.cuda.is_available():
        t0 = time.perf_counter()
        fn()
        return (time.perf_counter() - t0) * 1000.0
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e)


def event_time_us(fn, warmup: int, reps: int, reduce) -> float:
    """Warmup calls, then reduce over reps timed calls."""
    for _ in range(warmup):
        fn()
    sync()
    return reduce([time_once_ms(fn) * 1000.0 for _ in range(reps)])


def loop_time_us(call, iters: int, warmup: int) -> float:
    """Per-call time of back-to-back calls timed as one span."""
    for _ in range(warmup):
        call()
    sync()

    def span():
        for _ in range(iters):
            call()

    return time_once_ms(span) / iters * 1000.0


def graph_time_us(fn, *, warmup: int = 10, reps: int = 50, reduce=min) -> float:
    """fn must be shape-static with inputs pre-allocated; without a GPU it is eager timing."""
    if not torch.cuda.is_available():
        return event_time_us(fn, warmup, reps, reduce)
    for _ in range(3):
        fn()
    # Capture iters ops per graph to amortize the ~6us launch.
    torch.cuda.synchronize()
    s0, e0 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s0.record()
    fn()
    e0.record()
    torch.cuda.synchronize()
    est_us = s0.elapsed_time(e0) * 1000.0
    iters = 32 if est_us < 30.0 else (8 if est_us < 200.0 else 1)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)

    # No shared pool: one pool handle dies with the first graph and trips a CUDACachingAllocator assert.
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()

    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    del g
    return reduce(ts) / iters


def graph_median_us(fn, **kw) -> float:
    return graph_time_us(fn, reduce=st.median, **kw)


def ncu_kernel_durations(inner: str, ncu_bin: str) -> list[tuple[str, float]]:
    cmd = [
        ncu_bin,
        "--csv",
        "--metrics",
        "gpu__time_duration.sum",
        "--target-processes",
        "all",
        sys.executable,
        "-c",
        inner,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    lines = out.splitlines()
    hdr = next((i for i, line in enumerate(lines) if line.startswith('"ID"')), None)
    if hdr is None:
        return []
    res: list[tuple[str, float]] = []
    for r in csv.DictReader(io.StringIO("\n".join(lines[hdr:]))):
        if r.get("Metric Name") != "gpu__time_duration.sum":
            continue
        v = float(r["Metric Value"].replace(",", ""))
        us = v / 1000.0 if r.get("Metric Unit") == "ns" else v
        res.append((r.get("Kernel Name", ""), us))
    return res


def ncu_min_op_us(inner: str, ncu_bin: str, reps: int, exclude: tuple[str, ...] = ("at::",)) -> float | None:
    """Per-invocation sum of the op's kernels (minus `exclude`), min over reps."""
    durs = [us for kn, us in ncu_kernel_durations(inner, ncu_bin) if not any(t in kn for t in exclude)]
    if not durs or reps <= 0 or len(durs) % reps != 0:
        return None
    n = len(durs) // reps
    groups = [sum(durs[i * n : (i + 1) * n]) for i in range(reps)]
    return min(groups)
