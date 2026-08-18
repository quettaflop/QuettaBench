#!/usr/bin/env python3
"""Elementwise kernel affine fits -> cuda_event/elementwise/{gpu}.json, matching the
kernel_composed ElementwiseTable model: latency_us = floor_us + bytes / eff_bw,
bytes = elements * dtype_bytes * (reads + writes). For each kernel we CUDA-event
time a torch equivalent across element counts and least-squares fit (floor_us,
eff_bw_gb_s). eff_bw is *effective* (may exceed peak HBM: cache/fusion).

  CUDA_VISIBLE_DEVICES=0 python elementwise_probe.py --gpu-label A100 --out-dir <dir>
"""
from __future__ import annotations
import argparse, json, statistics as st
from pathlib import Path
import torch
import torch.nn.functional as F

# (reads, writes) per kernel — MUST match kernel_composed/elementwise.py::_IO
IO = {"rmsnorm": (1, 1), "silu_mul": (2, 1), "rotary_emb": (1, 1), "residual_add": (2, 1)}
ELEMS = [2**e for e in range(18, 30)]   # ~262k .. ~537M elements
REPS, WARMUP = 50, 10
DT_BYTES = 2


def time_op(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)  # us
    return st.median(ts)


def op_for(kernel, n, dev, dt):
    if kernel == "rmsnorm":
        x = torch.randn(n, device=dev, dtype=dt); w = torch.ones(n, device=dev, dtype=dt)
        return lambda: (x * torch.rsqrt(x.float().pow(2).mean() + 1e-6).to(dt) * w)
    if kernel == "silu_mul":
        g = torch.randn(n, device=dev, dtype=dt); u = torch.randn(n, device=dev, dtype=dt)
        return lambda: F.silu(g) * u
    if kernel == "rotary_emb":
        x = torch.randn(n, device=dev, dtype=dt); c = torch.randn(n, device=dev, dtype=dt)
        return lambda: x * c + x  # rope is ~2 fused mul-add reads/writes of the q/k rows
    if kernel == "residual_add":
        x = torch.randn(n, device=dev, dtype=dt); r = torch.randn(n, device=dev, dtype=dt)
        return lambda: x + r
    raise ValueError(kernel)


def fit(bytes_list, us_list) -> tuple[float, float]:
    """Least-squares latency_us = floor + slope*bytes -> (floor_us, eff_bw_gb_s)."""
    nP = len(bytes_list); sx = sum(bytes_list); sy = sum(us_list)
    sxx = sum(b * b for b in bytes_list); sxy = sum(b * u for b, u in zip(bytes_list, us_list))
    denom = nP * sxx - sx * sx
    slope = (nP * sxy - sx * sy) / denom if denom else 0.0     # us/byte
    floor = (sy - slope * sx) / nP
    eff_bw_gb_s = (1e-3 / slope) if slope > 0 else 0.0          # (1/slope) bytes/us -> GB/s
    return round(max(0.0, floor), 3), round(eff_bw_gb_s, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-label", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--max-mem-gb", type=float, default=20.0)
    a = ap.parse_args()
    dev = torch.device("cuda"); dt = torch.bfloat16
    print(f"[elem] {torch.cuda.get_device_name(0)}", flush=True)
    result = {}
    for kernel, (r, w) in IO.items():
        bl, ul = [], []
        for n in ELEMS:
            if n * DT_BYTES * (r + w + 1) / 1e9 > a.max_mem_gb:
                continue
            us = time_op(op_for(kernel, n, dev, dt))
            bl.append(n * DT_BYTES * (r + w)); ul.append(us)
        floor, bw = fit(bl, ul)
        result[kernel] = {"floor_us": floor, "eff_bw_gb_s": bw}
        print(f"  {kernel:14} floor={floor} us  eff_bw={bw} GB/s", flush=True)
    # cuda_event-timed (no NCU path) -> cuda_event/elementwise/. The loader prefers
    # ncu/elementwise/ (NCU-derived H100/A100) and falls back here.
    out = Path(a.out_dir) / "cuda_event" / "elementwise"; out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.gpu_label}.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"[elem] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
