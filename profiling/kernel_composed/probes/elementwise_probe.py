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
# --vllm adds the kernels only vLLM ships fused: fused_add_rms_norm (one kernel for
# the residual-add + rmsnorm pair — the composition prefers it when calibrated) and
# reshape_and_cache_flash (the real KV-cache write, previously priced by roofline).
VLLM_IO = {**IO, "fused_add_rmsnorm": (2, 2), "kv_cache_write": (1, 1)}
ELEMS = [2**e for e in range(18, 30)]   # ~262k .. ~537M elements
REPS, WARMUP = 50, 10
DT_BYTES = 2
HIDDEN = 4096                            # row width for the 2-D vLLM ops


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


def op_for_vllm(kernel, n, dev, dt):
    """The REAL vLLM kernel for each composition op (vs the torch-chain proxies in
    ``op_for``, which launch several kernels where vLLM launches one — the H200
    torch-chain fit overpriced fused ops ~2-3 ms/step at 48 layers). ``n`` keeps the
    same meaning as the composition's ``elements`` for that kernel."""
    from vllm import _custom_ops as ops  # noqa: PLC0415
    if kernel == "rmsnorm":
        t = max(1, n // HIDDEN)
        x = torch.randn(t, HIDDEN, device=dev, dtype=dt)
        w = torch.ones(HIDDEN, device=dev, dtype=dt)
        out = torch.empty_like(x)
        return lambda: ops.rms_norm(out, x, w, 1e-6)
    if kernel == "fused_add_rmsnorm":
        t = max(1, n // HIDDEN)
        x = torch.randn(t, HIDDEN, device=dev, dtype=dt)
        r = torch.randn_like(x)
        w = torch.ones(HIDDEN, device=dev, dtype=dt)
        return lambda: ops.fused_add_rms_norm(x, r, w, 1e-6)
    if kernel == "silu_mul":
        # n = OUTPUT elements (composition passes pairs*inter); input is (t, 2H).
        t = max(1, n // HIDDEN)
        x = torch.randn(t, 2 * HIDDEN, device=dev, dtype=dt)
        out = torch.empty(t, HIDDEN, device=dev, dtype=dt)
        return lambda: torch.ops._C.silu_and_mul(out, x)
    if kernel == "rotary_emb":
        # n = q+k elements (composition passes m*(nq+nkv)*hd); Llama-8B head config.
        nq, nkv, hd = 32, 8, 128
        t = max(1, n // ((nq + nkv) * hd))
        pos = torch.arange(t, dtype=torch.int64, device=dev) % 4096
        q = torch.randn(t, nq * hd, device=dev, dtype=dt)
        k = torch.randn(t, nkv * hd, device=dev, dtype=dt)
        cache = torch.randn(4096, hd, device=dev, dtype=dt)
        return lambda: ops.rotary_embedding(pos, q, k, hd, cache, True)
    if kernel == "residual_add":
        x = torch.randn(n, device=dev, dtype=dt); r = torch.randn(n, device=dev, dtype=dt)
        return lambda: x + r
    if kernel == "kv_cache_write":
        # n = k+v elements (composition passes tokens*n_kv*hd*2).
        nkv, hd, block = 8, 128, 16
        t = max(1, n // (2 * nkv * hd))
        nblocks = (t + block - 1) // block
        key = torch.randn(t, nkv, hd, device=dev, dtype=dt)
        val = torch.randn_like(key)
        kc = torch.zeros(nblocks, block, nkv, hd, device=dev, dtype=dt)
        vc = torch.zeros_like(kc)
        slots = torch.arange(t, dtype=torch.int64, device=dev)
        ks = torch.ones(1, device=dev, dtype=torch.float32)
        vs = torch.ones_like(ks)
        return lambda: ops.reshape_and_cache_flash(key, val, kc, vc, slots, "auto", ks, vs)
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
    ap.add_argument("--graph", action="store_true",
                    help="CUDA-graph-replay timing (graphed-decode-faithful, no NCU "
                         "needed) -> ncu/elementwise/; default eager -> cuda_event/")
    ap.add_argument("--vllm", action="store_true",
                    help="time the REAL vLLM fused kernels (rms_norm, "
                         "fused_add_rms_norm, silu_and_mul, rotary_embedding, "
                         "reshape_and_cache_flash) instead of torch-chain proxies; "
                         "adds the fused_add_rmsnorm + kv_cache_write entries the "
                         "composition prefers when present")
    a = ap.parse_args()
    dev = torch.device("cuda"); dt = torch.bfloat16
    if a.graph:
        from _graph import graph_median_us  # noqa: PLC0415
    tag = ("/graph" if a.graph else "") + ("/vllm" if a.vllm else "")
    print(f"[elem{tag}] {torch.cuda.get_device_name(0)}", flush=True)
    io_map = VLLM_IO if a.vllm else IO
    result = {}
    for kernel, (r, w) in io_map.items():
        bl, ul = [], []
        for n in ELEMS:
            if n * DT_BYTES * (r + w + 1) / 1e9 > a.max_mem_gb:
                continue
            fn = op_for_vllm(kernel, n, dev, dt) if a.vllm else op_for(kernel, n, dev, dt)
            us = graph_median_us(fn) if a.graph else time_op(fn)
            bl.append(n * DT_BYTES * (r + w)); ul.append(us)
        floor, bw = fit(bl, ul)
        result[kernel] = {"floor_us": floor, "eff_bw_gb_s": bw}
        print(f"  {kernel:14} floor={floor} us  eff_bw={bw} GB/s", flush=True)
    # ncu/ = graphed-decode-faithful (NCU or --graph replay, both dispatch-free);
    # cuda_event/ = eager incl. dispatch. The loader prefers ncu/ and falls back.
    method = "ncu" if a.graph else "cuda_event"
    out = Path(a.out_dir) / method / "elementwise"; out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.gpu_label}.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"[elem] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
