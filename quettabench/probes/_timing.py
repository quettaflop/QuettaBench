"""Kernel timing shared by the probes: CUDA-graph replay, and NCU per-kernel durations.
Graph replay works where NCU counters are locked (RmProfilingAdminOnly=1).
"""

from __future__ import annotations

import csv
import io
import statistics as st
import subprocess
import sys

import torch


def event_time_us(fn, warmup: int, reps: int, reduce) -> float:
    """Eager CUDA-event time of one fn call in us: warmup calls, then reduce over reps timed calls."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return reduce(ts)


def graph_time_us(fn, *, warmup: int = 10, reps: int = 50, reduce=min) -> float:
    """Capture fn into a CUDA graph and time replay in us, reduced over reps; fn must be shape-static with inputs pre-allocated."""
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

    # No shared memory pool: one pool handle dies with the first graph and trips a CUDACachingAllocator use_count assert.
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
    """[(kernel_name, us)] for every profiled kernel launch, in launch order."""
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


def ncu_op_us(inner: str, ncu_bin: str, reps: int, exclude: tuple[str, ...] = ("at::",)) -> float | None:
    """Kernel us of one op call: the snippet calls the op `reps` times; the op's kernels (minus `exclude`) are summed per invocation and the min taken. None when NCU gave no table or the kernel count is not a multiple of reps (an autotune leak)."""
    durs = [us for kn, us in ncu_kernel_durations(inner, ncu_bin) if not any(t in kn for t in exclude)]
    if not durs or reps <= 0 or len(durs) % reps != 0:
        return None
    n = len(durs) // reps  # kernels per invocation
    groups = [sum(durs[i * n : (i + 1) * n]) for i in range(reps)]
    return min(groups)
