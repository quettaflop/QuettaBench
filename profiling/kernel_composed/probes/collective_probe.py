#!/usr/bin/env python3
"""NCCL collective microbenchmark -> the measured collective GRID (+ legacy scalars).

Method B for the kernel-composed backend. Sweeps message size across the ranks for
EVERY collective LLM inference emits, and writes the whole curve to

    kernel_data/ncu/collectives/{GPU}_tp{world}.csv   (op, world, bytes, latency_us, ...)

  all_reduce      TP: 2 per layer (after o-proj, after down-proj)
  reduce_scatter  TP + sequence parallelism: replaces half of each all-reduce
  all_gather      TP + SP (the other half); also the vocab-parallel lm_head gather
  all_to_all      EP: 2 per MoE layer (expert dispatch + combine)

Measuring each separately matters: their cost shapes genuinely differ (all-reduce
moves 2(n-1)/n of the buffer, the others (n-1)/n), and all-to-all's real cost is
routing-dependent in a way no closed form captures.

which `kernel_composed.collectives.CollectiveTable` reads under
``comm_source="measured"``. Measuring the curve rather than fitting it means we
capture whatever NCCL/vLLM actually selected -- custom-all-reduce thresholds,
LL/LL128/SIMPLE protocol switchover, NVLS on NVSwitch parts -- without having to
model any of it. That is the whole point of the measured source.

It ALSO still prints the legacy two-scalar `tp_comm:` block (small-message floor +
large-message bandwidth) so existing device YAMLs can be refreshed; those scalars
are tp-invariant by construction, which is the limitation the grid removes.

Launch with torchrun across the tp ranks (run_probe.sh does this when NPROC>1):

  torchrun --nproc_per_node=2 collective_probe.py --gpu-label A100 --hidden 4096 --layers 32

Run it once per world size you care about (2, 4, 8): each writes its own
`_tp{world}.csv`, and the table interpolates across world only for sizes you did
not measure. Rank 0 writes the CSV and prints the YAML block; like
derive_device_yaml.py it does NOT edit the YAML.
"""
from __future__ import annotations
import argparse, csv, os, statistics as st
from pathlib import Path

import torch
import torch.distributed as dist

from _paths import kernel_data_root

_DTYPE = torch.bfloat16
_ITERS, _WARMUP = 50, 10


def _time_op(op: str, n_elems: int, world: int, iters: int, warmup: int) -> float:
    """AMORTIZED wall-time (seconds) per ``op`` call over an n_elems bf16 buffer.

    Times the whole back-to-back loop and divides by iters, so per-op launch/dispatch
    overhead pipelines away and we get the MARGINAL cost -- the faithful measure for
    decode, which replays the all-reduce inside a CUDA graph (no per-launch dispatch).
    Per-op event timing instead would bake in launch overhead the graphed decode never
    pays and over-price tp_comm (README: decode is graphed -> measure amortized)."""
    # Buffers follow the table's convention: n_elems is the FULL LOGICAL BUFFER --
    # the input for all_reduce/reduce_scatter, the RESULT for all_gather, the total
    # each rank sends for all_to_all -- so every op is comparable at equal bytes.
    full = torch.ones(n_elems, dtype=_DTYPE, device="cuda")
    shard = torch.ones(max(1, n_elems // world), dtype=_DTYPE, device="cuda")
    if op == "all_reduce":
        call = lambda: dist.all_reduce(full)
    elif op == "all_gather":
        call = lambda: dist.all_gather_into_tensor(full, shard)
    elif op == "reduce_scatter":
        call = lambda: dist.reduce_scatter_tensor(shard, full)
    elif op == "all_to_all":
        out = torch.empty_like(full)
        call = lambda: dist.all_to_all_single(out, full)
    else:
        raise ValueError(f"unknown op {op!r}")

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        call()
    end.record()
    torch.cuda.synchronize()
    return (start.elapsed_time(end) / iters) / 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-label", required=True)
    ap.add_argument("--hidden", type=int, required=True, help="model hidden_size (for prefill rate)")
    ap.add_argument("--layers", type=int, required=True, help="model n_layers (for prefill rate)")
    ap.add_argument("--out-dir", default=None,
                    help="kernel_data root; the grid lands in <out-dir>/ncu/allreduce/ "
                         "(default: $KDATA, else data/kernel_data). '-' skips the write.")
    ap.add_argument("--prefill-tokens", type=int, default=2048, help="chunk width for the per-token rate")
    ap.add_argument("--ops", default="all_reduce,all_gather,reduce_scatter,all_to_all",
                    help="collectives to sweep (comma-separated). all_reduce is required "
                         "for the legacy tp_comm: scalars.")
    a = ap.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # message-size sweep per op: small end -> latency floor, large end -> bandwidth.
    sizes = [2 ** k for k in range(10, 28)]   # 1K .. 128M elements
    ops = [o.strip() for o in a.ops.split(",") if o.strip()]
    rows: list[tuple[str, int, int, float]] = []   # (op, world, bytes, us)
    lat_us, bw = [], []
    for op in ops:
        for n in sizes:
            try:
                t = _time_op(op, n, world, _ITERS, _WARMUP)
            except RuntimeError as e:            # OOM / unsupported shape: record a gap
                if rank == 0:
                    print(f"  skip {op} n={n}: {e}")
                continue
            rows.append((op, world, n * 2, t * 1e6))
            if op == "all_reduce":
                lat_us.append(t * 1e6)
                # ring all-reduce moves ~2*(world-1)/world * bytes over the link
                bytes_moved = 2.0 * (world - 1) / world * n * 2  # bf16 = 2 bytes
                bw.append(bytes_moved / t)
    if not lat_us:
        raise SystemExit("all_reduce must be in --ops to derive the legacy scalars")

    # per-token prefill rate: all-reduce a [T x hidden] activation, once per layer.
    t_pref = _time_op("all_reduce", a.prefill_tokens * a.hidden, world, _ITERS, _WARMUP)
    prefill_ms_per_token = (t_pref / a.prefill_tokens) * a.layers * 1000.0

    latency_us_per_op = round(min(lat_us), 3)       # small-message floor
    link_bw_bytes_per_s = round(max(bw), 1)         # bandwidth-bound tail
    prefill_ms_per_token = round(prefill_ms_per_token, 6)

    if rank == 0:
        # ── Method B: the whole curve, as a grid ──────────────────────────────
        if a.out_dir != "-":
            root = Path(a.out_dir or os.environ.get("KDATA") or kernel_data_root())
            out = root / "ncu" / "collectives" / f"{a.gpu_label}_tp{world}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["op", "world", "bytes", "latency_us", "dtype_bytes"])
                for op, wd, nbytes, t_us in rows:
                    w.writerow([op, wd, nbytes, round(t_us, 4), 2])
            got = sorted({op for op, _, _, _ in rows})
            print(f"wrote {out}  ({len(rows)} points, world={world}, ops={got})\n")

        # ── legacy two-scalar block (tp-invariant; kept for refreshes) ────────
        print(f"[{a.gpu_label}] all-reduce over {world} ranks, hidden={a.hidden} layers={a.layers}\n")
        print(f"# paste into device_spec/{a.gpu_label.lower()}.yaml (collective_probe.py):")
        print("tp_comm:")
        print(f"  prefill_ms_per_token: {prefill_ms_per_token}")
        print(f"  latency_us_per_op: {latency_us_per_op}")
        print(f"  link_bw_bytes_per_s: {link_bw_bytes_per_s:.3e}")
        print("  # source: measured   # <- uncomment to make this device use the grid above")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
