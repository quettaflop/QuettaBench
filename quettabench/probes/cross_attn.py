#!/usr/bin/env python3
"""Chunked-prefill attention (q_len new tokens over resident cached tokens, one varlen call) -> fa3_cross/{gpu}.csv rows, per layer, eager.
Neither the decode (q=1) nor the full-causal prefill grid covers this shape, which dominates prefill attention on long-ISL traces; --paged times vLLM's paged call (block_table, seqused_k), whose split-KV path is much faster at q << kv.
"""
from __future__ import annotations

import argparse
import statistics as st

import torch
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

from quettabench.probes.base import Probe, ProbeRow, add_output_args, run_recorded

Q_AXIS = [512, 1024, 2048, 4096, 8192]
CTX_AXIS = [0, 1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768]
REPS, WARMUP = 30, 5


def time_call(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return st.median(ts)




class CrossAttnProbe(Probe):
    """One head config's (q_len x resident) grid, eager, a row of
    fa3_cross/{gpu}.csv per point."""

    def __init__(self, a):
        self.a = a
        self.dev, self.dt = torch.device("cuda"), torch.bfloat16

    def grid(self):
        q_axis = [int(x) for x in self.a.q_axis.split(",")]
        ctx_axis = [int(x) for x in self.a.ctx_axis.split(",")]
        return ((ql, ctx) for ql in q_axis for ctx in ctx_axis)

    def run_point(self, point):
        ql, ctx = point
        a, dev, dt = self.a, self.dev, self.dt
        kv = ctx + ql
        q = torch.randn(ql, a.n_heads, a.head_dim, device=dev, dtype=dt)
        cu_q = torch.tensor([0, ql], dtype=torch.int32, device=dev)
        if a.paged:
            # vLLM's actual call shape: paged KV cache + block_table + seqused_k.
            # This is load-bearing at q << kv: with a block_table the kernel's
            # split-KV heuristics engage (more CTAs than q_blocks x heads), while
            # the contiguous cu_seqlens_k path runs unsplit and measures up to
            # ~3x slower at q<=512 over long residents -- a shape a chunked
            # high-APC serving workload lives in.
            bs = 16
            nb = (kv + bs - 1) // bs
            k = torch.randn(nb, bs, a.n_kv_heads, a.head_dim, device=dev, dtype=dt)
            v = torch.randn_like(k)
            bt = torch.arange(nb, dtype=torch.int32, device=dev).unsqueeze(0)
            sk = torch.tensor([kv], dtype=torch.int32, device=dev)
            us = time_call(lambda: flash_attn_varlen_func(
                q, k, v, ql, cu_q, kv, seqused_k=sk, block_table=bt,
                causal=True, fa_version=a.fa_version))
        else:
            k = torch.randn(kv, a.n_kv_heads, a.head_dim, device=dev, dtype=dt)
            v = torch.randn_like(k)
            cu_k = torch.tensor([0, kv], dtype=torch.int32, device=dev)
            us = time_call(lambda: flash_attn_varlen_func(
                q, k, v, ql, cu_q, kv, cu_seqlens_k=cu_k, causal=True, fa_version=a.fa_version))
        del q, k, v
        print(f"  q={ql:5d} resident={ctx:6d} kv={kv:6d}: {us:8.1f} us/layer", flush=True)
        return ProbeRow("cross_attn", f"fa3_cross/{a.gpu_label}.csv",
                        {"n_heads": a.n_heads, "n_kv_heads": a.n_kv_heads, "head_dim": a.head_dim,
                         "q_len": ql, "resident_tokens": ctx, "kv_len": kv,
                         "fa_version": f"vllm-fa{a.fa_version}", "dtype": "bfloat16"},
                        {"latency_us": round(us, 3)}, "eager")


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_output_args(ap)
    ap.add_argument("--n-heads", type=int, required=True)
    ap.add_argument("--n-kv-heads", type=int, required=True)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--fa-version", type=int, default=0, help="0=auto (FA3 on sm90+, else FA2)")
    ap.add_argument("--paged", action="store_true",
                    help="paged KV + block_table + seqused_k (vLLM's real call; enables "
                         "the kernel's split-KV heuristics at q << kv)")
    ap.add_argument("--q-axis", default=",".join(map(str, Q_AXIS)))
    ap.add_argument("--ctx-axis", default=",".join(map(str, CTX_AXIS)))
    a = ap.parse_args(argv)
    if a.fa_version == 0:
        a.fa_version = 3 if torch.cuda.get_device_capability()[0] >= 9 else 2
    print(f"[cross] {torch.cuda.get_device_name(0)} heads={a.n_heads}/{a.n_kv_heads}/{a.head_dim} fa{a.fa_version}", flush=True)
    # This head config's rows only; kernel_tables unions every ok run on the head config.
    n = CrossAttnProbe(a).run()
    print(f"[cross] {n} rows of fa3_cross/{a.gpu_label}.csv", flush=True)


if __name__ == "__main__":
    run_recorded(main, "cross_attn")
