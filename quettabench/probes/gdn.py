#!/usr/bin/env python3
"""Gated-DeltaNet (Qwen3.5) kernel grid on vLLM's vendored FLA ops: prefill chunk_gated_delta_rule over seq -> gdn/{gpu}_{geom}_prefill.csv (always eager), decode fused_recurrent_gated_delta_rule over batch -> gdn/{gpu}_{geom}_decode.csv (NCU with --ncu, else eager).
Rows hold per-call median us and us x n_linear (all-GDN-layers convention).
"""
from __future__ import annotations

import argparse
import os
import statistics as st
import traceback

import torch
import triton
from vllm.model_executor.layers.fla.ops import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)

from quettabench.probes._timing import ncu_op_us
from quettabench.probes.base import Probe, ProbeRow, add_output_args, run_recorded

# chunk_gated_delta_rule's solve_tril merge kernel autotunes on a fresh host (no
# Triton disk cache) and needs a scratch allocator; vLLM sets one at model init, a
# bare probe must too. Without this the prefill loop crashes ("no allocator was
# set") on fresh H100/3090 and writes 0 prefill rows.
triton.set_allocator(lambda size, alignment, stream: torch.empty(size, dtype=torch.int8, device="cuda"))

REPS, WARMUP = 30, 5
# cap at 80: the generic fused_recurrent decode proxy hits an illegal-memory bug at
# B>=120 (ssm_state_indices path) that corrupts the CUDA context and would kill the
# prefill loop too. The faithful fused_sigmoid_gating_delta_rule_update decode (which
# lifts this cap) is a follow-up -> then re-profile decode via gdn-ncu.
BATCH_FULL = [1, 2, 4, 8, 16, 32, 40, 64, 80]
SEQ_FULL   = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
BATCH_SMOKE = [1, 32, 256]
SEQ_SMOKE   = [128, 2048]
# NCU is slow -> curated grids; the loader interpolates. Decode capped at 80 (the
# fused_recurrent proxy hits a synthetic-input bug at B>=120 -- same limit as the
# cuda_event grid). NCU_REPS invocations profiled per point; min = warm steady state.
BATCH_NCU = [1, 8, 32, 80]
NCU_REPS = 5


def _ncu_decode_inner(B, H, K, V) -> str:
    return (
        "import torch;"
        "from vllm.model_executor.layers.fla.ops import fused_recurrent_gated_delta_rule as R;"
        "d=torch.device('cuda');t=torch.bfloat16;"
        f"B,H,K,V={B},{H},{K},{V};s=1.0/(K**0.5);"
        "q=torch.randn(B,1,H,K,device=d,dtype=t);k=torch.randn(B,1,H,K,device=d,dtype=t);"
        "v=torch.randn(B,1,H,V,device=d,dtype=t);"
        "g=-torch.rand(B,1,H,device=d,dtype=torch.float32);beta=torch.rand(B,1,H,device=d,dtype=t);"
        "st=torch.randn(B,H,V,K,device=d,dtype=torch.float32);"
        "f=lambda:R(q,k,v,g,beta,scale=s,initial_state=st,inplace_final_state=False,use_qk_l2norm_in_kernel=True);"
        f"[f() for _ in range({NCU_REPS})];torch.cuda.synchronize()"
    )


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
        ts.append(s.elapsed_time(e) * 1000.0)  # ms -> us
    return st.median(ts)


def mk(B, T, H, K, V, dev, dt):
    q = torch.randn(B, T, H, K, device=dev, dtype=dt)
    k = torch.randn(B, T, H, K, device=dev, dtype=dt)
    v = torch.randn(B, T, H, V, device=dev, dtype=dt)
    g = -torch.rand(B, T, H, device=dev, dtype=torch.float32)   # log-space forget gate (<=0)
    beta = torch.rand(B, T, H, device=dev, dtype=dt)            # betas in (0,1)
    return q, k, v, g, beta


class GdnProbe(Probe):
    """The prefill grid (seq, always eager) of gdn/{gpu}_{geom}_prefill.csv,
    then the decode grid (batch, NCU or eager) of ..._decode.csv. A phase
    stops at its first failure: the decode proxy's illegal-memory bug
    corrupts the CUDA context."""

    def __init__(self, a):
        self.a = a
        self.dev, self.dt = torch.device("cuda"), torch.bfloat16
        self.scale = 1.0 / (a.dk ** 0.5)
        self.base = f"{a.gpu_label}_{a.geometry_tag}"
        self.failed = False

    def grid(self):
        a = self.a
        # PREFILL grid FIRST: chunk_gated_delta_rule, B=1, T=seq (eager in serving ->
        # always cuda_event). Deliberately before decode: the fragile fused_recurrent
        # decode proxy hits an illegal-memory bug (B=16 on 3090, B>=120 on A100/H100)
        # that corrupts the CUDA context; running prefill first saves the floor grid
        # regardless of where decode dies.
        for T in SEQ_SMOKE if a.mode == "smoke" else SEQ_FULL:
            if self.failed:
                break
            yield "prefill", T
        self.failed = False
        # DECODE grid: fused recurrent, T=1, initial recurrent state [B,H,V,K]
        for B in BATCH_NCU if a.ncu else (BATCH_SMOKE if a.mode == "smoke" else BATCH_FULL):
            if self.failed:
                break
            yield "decode", B

    def run_point(self, point):
        phase, n = point
        a = self.a
        try:
            us = self.prefill_us(n) if phase == "prefill" else self.decode_us(n)
        except Exception as ex:
            print(f"  {phase} {'T' if phase == 'prefill' else 'B'}={n} FAILED: {type(ex).__name__}: {ex}",
                  flush=True)
            traceback.print_exc()
            self.failed = True
            return None
        if us is None:
            return None
        if phase == "prefill":
            print(f"  prefill T={n:5d}: {us:8.2f} us/call  x{a.n_linear}L={us*a.n_linear:9.1f} us", flush=True)
            return ProbeRow("gdn", f"gdn/{self.base}_prefill.csv", {"seq_len": n},
                            {"us_per_call": round(us, 3), "us_x_nlinear": round(us * a.n_linear, 3)}, "eager")
        print(f"  {'ncu ' if a.ncu else ''}decode B={n:4d}: {us:8.2f} us/call  x{a.n_linear}L={us*a.n_linear:9.1f} us", flush=True)
        # Decode is CUDA-graphed in serving -> NCU (--ncu) is faithful.
        return ProbeRow("gdn", f"gdn/{self.base}_decode.csv", {"batch": n},
                        {"us_per_call": round(us, 3), "us_x_nlinear": round(us * a.n_linear, 3)},
                        "ncu" if a.ncu else "eager")

    def prefill_us(self, T):
        a = self.a
        q, k, v, g, beta = mk(1, T, a.hv, a.dk, a.dv, self.dev, self.dt)

        def call():
            return chunk_gated_delta_rule(
                q, k, v, g, beta, scale=self.scale, output_final_state=True,
                use_qk_l2norm_in_kernel=True)
        return time_call(call)

    def decode_us(self, B):
        a = self.a
        H, K, V = a.hv, a.dk, a.dv
        if a.ncu:
            us = ncu_op_us(_ncu_decode_inner(B, H, K, V), a.ncu_bin, NCU_REPS)
            if us is None:
                print(f"  ncu MISS decode B={B}", flush=True)
            return us
        q, k, v, g, beta = mk(B, 1, H, K, V, self.dev, self.dt)
        state = torch.randn(B, H, V, K, device=self.dev, dtype=torch.float32)
        # NOTE: representative GDN decode-core cost. The faithful serving decode
        # is fused_sigmoid_gating_delta_rule_update (fixed-param, no autotune);
        # the generic recurrent delta-rule is the same math and a valid proxy.

        def call():
            return fused_recurrent_gated_delta_rule(
                q, k, v, g, beta, scale=self.scale, initial_state=state,
                inplace_final_state=False, use_qk_l2norm_in_kernel=True)
        return time_call(call)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hv", type=int, default=32)     # linear_num_value_heads
    ap.add_argument("--dk", type=int, default=128)    # linear_key_head_dim
    ap.add_argument("--dv", type=int, default=128)    # linear_value_head_dim
    ap.add_argument("--n-linear", type=int, default=24)
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    add_output_args(ap)
    ap.add_argument("--geometry-tag", default="9b", help="GDN head geometry tag, e.g. 9b (H_v=32) / 27b (H_v=48)")
    ap.add_argument("--max-mem-gb", type=float, default=30.0)
    ap.add_argument("--ncu", action="store_true", help="NCU per-kernel timing (the standard); curated grid")
    ap.add_argument("--ncu-bin", default=os.environ.get("NCU_BIN", "ncu"))
    a = ap.parse_args(argv)
    print(f"[gdn{'/ncu' if a.ncu else ''}] dev={torch.cuda.get_device_name(0)} H={a.hv} K={a.dk} V={a.dv} n_linear={a.n_linear}", flush=True)
    n = GdnProbe(a).run()
    print(f"[gdn] {n} rows of gdn/{a.gpu_label}_{a.geometry_tag}_decode.csv and _prefill.csv", flush=True)


if __name__ == "__main__":
    run_recorded(main, "gdn")
