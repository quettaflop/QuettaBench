#!/usr/bin/env python3
"""NCCL collective microbenchmark under torchrun: every op LLM inference emits."""

from __future__ import annotations

import argparse

import torch
import torch.distributed as dist

from quettabench.probes import _timing
from quettabench.probes.base import ProbeRow, add_output_args, emit, run_recorded

_DTYPE = torch.bfloat16
_ITERS, _WARMUP = 50, 10


def _time_op(op: str, n_elems: int, world: int, iters: int, warmup: int) -> float:
    """Back-to-back loop, like graphed decode."""
    # n_elems is the full logical buffer, so ops compare at equal bytes.
    full = torch.ones(n_elems, dtype=_DTYPE, device=_timing.device())
    shard = torch.ones(max(1, n_elems // world), dtype=_DTYPE, device=_timing.device())
    if op == "all_reduce":
        call = lambda: dist.all_reduce(full)  # noqa: E731
    elif op == "all_gather":
        call = lambda: dist.all_gather_into_tensor(full, shard)  # noqa: E731
    elif op == "reduce_scatter":
        call = lambda: dist.reduce_scatter_tensor(shard, full)  # noqa: E731
    elif op == "all_to_all":
        out = torch.empty_like(full)
        call = lambda: dist.all_to_all_single(out, full)  # noqa: E731
    else:
        raise ValueError(f"unknown op {op!r}")

    return _timing.loop_time_us(call, iters, warmup) / 1e6


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_output_args(ap)
    ap.add_argument("--hidden", type=int, required=True, help="model hidden_size (for prefill rate)")
    ap.add_argument("--layers", type=int, required=True, help="model n_layers (for prefill rate)")
    ap.add_argument("--prefill-tokens", type=int, default=2048, help="chunk width for the per-token rate")
    ap.add_argument(
        "--ops",
        default="all_reduce,all_gather,reduce_scatter,all_to_all",
        help="collectives to sweep (comma-separated). all_reduce is required for the tp_comm scalars.",
    )
    a = ap.parse_args(argv)
    _timing.require_backend("collective", "cuda", "rocm")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    _timing.set_device(rank % _timing.device_count())

    # Small sizes give the latency floor, large ones the bandwidth.
    sizes = [2**k for k in range(10, 28)]  # 1K to 128M elements
    ops = [o.strip() for o in a.ops.split(",") if o.strip()]
    rows: list[tuple[str, int, int, float]] = []  # (op, world, bytes, us)
    lat_us, bw = [], []
    for op in ops:
        for n in sizes:
            try:
                t = _time_op(op, n, world, _ITERS, _WARMUP)
            except RuntimeError as e:  # OOM or unsupported shape: leave a gap
                if rank == 0:
                    print(f"  skip {op} n={n}: {e}")
                continue
            rows.append((op, world, n * 2, t * 1e6))
            if op == "all_reduce":
                lat_us.append(t * 1e6)
                # Ring all-reduce moves 2*(world-1)/world of the bytes over the link.
                bytes_moved = 2.0 * (world - 1) / world * n * 2  # bf16
                bw.append(bytes_moved / t)
    if not lat_us:
        raise SystemExit("all_reduce must be in --ops to derive the tp_comm scalars")

    # Per-token prefill rate: all-reduce a [T x hidden] activation once per layer.
    t_pref = _time_op("all_reduce", a.prefill_tokens * a.hidden, world, _ITERS, _WARMUP)
    prefill_ms_per_token = (t_pref / a.prefill_tokens) * a.layers * 1000.0

    latency_us_per_op = round(min(lat_us), 3)
    link_bw_bytes_per_s = round(max(bw), 1)
    prefill_ms_per_token = round(prefill_ms_per_token, 6)

    if rank == 0:
        if a.out_dir != "-":
            table = f"collectives/{a.gpu_label}_tp{world}.csv"
            for op, wd, nbytes, t_us in rows:
                emit(
                    ProbeRow(
                        "collective",
                        table,
                        {"op": op, "world": wd, "bytes": nbytes},
                        {"latency_us": round(t_us, 4), "dtype_bytes": 2},
                        "amortized_wallclock",
                    )
                )
            got = sorted({op for op, _, _, _ in rows})
            print(f"{len(rows)} rows of {table}  (world={world}, ops={got})\n")

        # The tp_comm scalars are tp-invariant; the grid is not.
        print(f"[{a.gpu_label}] all-reduce over {world} ranks, hidden={a.hidden} layers={a.layers}\n")
        print(f"# paste into device_spec/{a.gpu_label.lower()}.yaml (collective_probe.py):")
        print("tp_comm:")
        print(f"  prefill_ms_per_token: {prefill_ms_per_token}")
        print(f"  latency_us_per_op: {latency_us_per_op}")
        print(f"  link_bw_bytes_per_s: {link_bw_bytes_per_s:.3e}")
        print("  # source: measured   # <- uncomment to make this device use the grid above")
    dist.destroy_process_group()
    return rank == 0


if __name__ == "__main__":
    run_recorded(main, "collective")
