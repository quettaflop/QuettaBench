#!/usr/bin/env python3
"""Elementwise kernel affine fits -> elementwise/{gpu}.json: us = floor_us + bytes / eff_bw."""

from __future__ import annotations

import argparse
import statistics as st

import torch
import torch.nn.functional as F

from quettabench.probes import _timing
from quettabench.probes._timing import event_time_us, graph_median_us
from quettabench.probes.base import Probe, ProbeRow, add_output_args, run_recorded

# (reads, writes) per kernel; must match kernel_composed/elementwise.py::_IO.
IO = {"rmsnorm": (1, 1), "silu_mul": (2, 1), "rotary_emb": (1, 1), "residual_add": (2, 1)}
# Only vLLM ships these fused: fused_add_rms_norm and reshape_and_cache_flash.
VLLM_IO = {**IO, "fused_add_rmsnorm": (2, 2), "kv_cache_write": (1, 1)}
ELEMS = [2**e for e in range(18, 30)]  # 262k to 537M elements
REPS, WARMUP = 50, 10
DT_BYTES = 2
HIDDEN = 4096  # row width for the 2-D vLLM ops


def time_op(fn) -> float:
    return event_time_us(fn, WARMUP, REPS, st.median)


def op_for(kernel, n, dev, dt):
    if kernel == "rmsnorm":
        x = torch.randn(n, device=dev, dtype=dt)
        w = torch.ones(n, device=dev, dtype=dt)
        return lambda: x * torch.rsqrt(x.float().pow(2).mean() + 1e-6).to(dt) * w
    if kernel == "silu_mul":
        g = torch.randn(n, device=dev, dtype=dt)
        u = torch.randn(n, device=dev, dtype=dt)
        return lambda: F.silu(g) * u
    if kernel == "rotary_emb":
        x = torch.randn(n, device=dev, dtype=dt)
        c = torch.randn(n, device=dev, dtype=dt)
        return lambda: x * c + x  # rope is ~2 fused mul-add reads/writes of the q/k rows
    if kernel == "residual_add":
        x = torch.randn(n, device=dev, dtype=dt)
        r = torch.randn(n, device=dev, dtype=dt)
        return lambda: x + r
    raise ValueError(kernel)


def op_for_vllm(kernel, n, dev, dt):
    """vllm is imported here: only --vllm needs it."""
    from vllm import _custom_ops as ops

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
        # n is output elements (composition passes pairs*inter); input is (t, 2H).
        t = max(1, n // HIDDEN)
        x = torch.randn(t, 2 * HIDDEN, device=dev, dtype=dt)
        out = torch.empty(t, HIDDEN, device=dev, dtype=dt)
        return lambda: torch.ops._C.silu_and_mul(out, x)
    if kernel == "rotary_emb":
        # n is q+k elements (composition passes m*(nq+nkv)*hd); Llama-8B head config.
        nq, nkv, hd = 32, 8, 128
        t = max(1, n // ((nq + nkv) * hd))
        pos = torch.arange(t, dtype=torch.int64, device=dev) % 4096
        q = torch.randn(t, nq * hd, device=dev, dtype=dt)
        k = torch.randn(t, nkv * hd, device=dev, dtype=dt)
        cache = torch.randn(4096, hd, device=dev, dtype=dt)
        return lambda: ops.rotary_embedding(pos, q, k, hd, cache, True)
    if kernel == "residual_add":
        x = torch.randn(n, device=dev, dtype=dt)
        r = torch.randn(n, device=dev, dtype=dt)
        return lambda: x + r
    if kernel == "kv_cache_write":
        # n is k+v elements (composition passes tokens*n_kv*hd*2).
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
    """Returns (floor_us, eff_bw_gb_s)."""
    nP = len(bytes_list)
    sx = sum(bytes_list)
    sy = sum(us_list)
    sxx = sum(b * b for b in bytes_list)
    sxy = sum(b * u for b, u in zip(bytes_list, us_list))
    denom = nP * sxx - sx * sx
    slope = (nP * sxy - sx * sy) / denom if denom else 0.0  # us/byte
    floor = (sy - slope * sx) / nP
    eff_bw_gb_s = (1e-3 / slope) if slope > 0 else 0.0
    return round(max(0.0, floor), 3), round(eff_bw_gb_s, 1)


class ElementwiseProbe(Probe):
    def __init__(self, a):
        self.a = a
        self.dev, self.dt = torch.device(_timing.device()), torch.bfloat16
        self.timing = "graph_replay" if a.graph else "eager"

    def grid(self):
        return (VLLM_IO if self.a.vllm else IO).items()

    def run_point(self, point):
        kernel, (r, w) = point
        a = self.a
        bl, ul = [], []
        for n in ELEMS:
            if n * DT_BYTES * (r + w + 1) / 1e9 > a.max_mem_gb:
                continue
            fn = op_for_vllm(kernel, n, self.dev, self.dt) if a.vllm else op_for(kernel, n, self.dev, self.dt)
            us = graph_median_us(fn) if a.graph else time_op(fn)
            bl.append(n * DT_BYTES * (r + w))
            ul.append(us)
        floor, bw = fit(bl, ul)
        print(f"  {kernel:14} floor={floor} us  eff_bw={bw} GB/s", flush=True)
        return ProbeRow(
            "elementwise",
            f"elementwise/{a.gpu_label}.json",
            {"entry": kernel},
            {"value": {"floor_us": floor, "eff_bw_gb_s": bw}},
            self.timing,
        )


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_output_args(ap)
    ap.add_argument("--max-mem-gb", type=float, default=20.0)
    ap.add_argument(
        "--graph",
        action="store_true",
        help="CUDA-graph-replay timing (graphed-decode-faithful, no NCU "
        "needed) -> graph/elementwise/; default eager -> eager/",
    )
    ap.add_argument(
        "--vllm",
        action="store_true",
        help="time the REAL vLLM fused kernels (rms_norm, "
        "fused_add_rms_norm, silu_and_mul, rotary_embedding, "
        "reshape_and_cache_flash) instead of torch-chain proxies; "
        "adds the fused_add_rmsnorm + kv_cache_write entries the "
        "composition prefers when present",
    )
    a = ap.parse_args(argv)
    _timing.require_backend("elementwise", "cuda", "rocm", "cpu")
    if a.vllm:
        _timing.require_backend("elementwise --vllm", "cuda", "rocm")
    tag = ("/graph" if a.graph else "") + ("/vllm" if a.vllm else "")
    print(f"[elem{tag}] {_timing.device_name()}", flush=True)
    n = ElementwiseProbe(a).run()
    print(f"[elem] {n} entries of elementwise/{a.gpu_label}.json", flush=True)


if __name__ == "__main__":
    run_recorded(main, "elementwise")
