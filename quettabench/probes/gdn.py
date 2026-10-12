#!/usr/bin/env python3
"""Gated-DeltaNet kernel grid on vLLM's FLA ops: prefill and decode CSVs."""

from __future__ import annotations

import argparse
import functools
import os
import statistics as st
import traceback

import torch
import triton
from vllm.model_executor.layers.fla.ops import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)

from quettabench.probes import _timing
from quettabench.probes._timing import event_time_us, ncu_min_op_us
from quettabench.probes.base import Probe, ProbeRow, add_output_args, run_recorded

# solve_tril needs a scratch allocator (vLLM sets one at model init), else prefill fails.
triton.set_allocator(lambda size, alignment, stream: torch.empty(size, dtype=torch.int8, device=_timing.device()))

REPS, WARMUP = 30, 5
# Capped at 80: the decode proxy hits an illegal-memory bug at B>=120 that corrupts CUDA.
BATCH_FULL = [1, 2, 4, 8, 16, 32, 40, 64, 80]
SEQ_FULL = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
BATCH_SMOKE = [1, 32, 256]
SEQ_SMOKE = [128, 2048]
# NCU is slow, so a curated grid; the loader interpolates.
BATCH_NCU = [1, 8, 32, 80]
NCU_REPS = 5  # Launches per NCU capture; the min over per-invocation sums is kept.


def _ncu_decode_inner(B, H, K, V) -> str:
    return (
        "import torch;from quettabench.probes import _timing;"
        "from vllm.model_executor.layers.fla.ops import fused_recurrent_gated_delta_rule as R;"
        f"d=torch.device('{_timing.device()}');t=torch.bfloat16;"
        f"B,H,K,V={B},{H},{K},{V};s=1.0/(K**0.5);"
        "q=torch.randn(B,1,H,K,device=d,dtype=t);k=torch.randn(B,1,H,K,device=d,dtype=t);"
        "v=torch.randn(B,1,H,V,device=d,dtype=t);"
        "g=-torch.rand(B,1,H,device=d,dtype=torch.float32);beta=torch.rand(B,1,H,device=d,dtype=t);"
        "st=torch.randn(B,H,V,K,device=d,dtype=torch.float32);"
        "f=lambda:R(q,k,v,g,beta,scale=s,initial_state=st,inplace_final_state=False,use_qk_l2norm_in_kernel=True);"
        f"[f() for _ in range({NCU_REPS})];_timing.sync()"
    )


time_call = functools.partial(event_time_us, warmup=WARMUP, reps=REPS, reduce=st.median)


def mk(B, T, H, K, V, dev, dt):
    q = torch.randn(B, T, H, K, device=dev, dtype=dt)
    k = torch.randn(B, T, H, K, device=dev, dtype=dt)
    v = torch.randn(B, T, H, V, device=dev, dtype=dt)
    g = -torch.rand(B, T, H, device=dev, dtype=torch.float32)  # log-space forget gate, <= 0
    beta = torch.rand(B, T, H, device=dev, dtype=dt)
    return q, k, v, g, beta


class GdnProbe(Probe):
    """A phase stops at its first failure: the decode bug corrupts the CUDA context."""

    def __init__(self, a):
        self.a = a
        self.dev, self.dt = torch.device(_timing.device()), torch.bfloat16
        self.scale = 1.0 / (a.key_dim**0.5)
        self.base = f"{a.gpu_label}_{a.gdn_geometry}"
        self.failed = False

    def grid(self):
        a = self.a
        # Prefill first: the decode proxy's illegal-memory bug at large B would lose it.
        for T in SEQ_SMOKE if a.grid_size == "smoke" else SEQ_FULL:
            if self.failed:
                break
            yield "prefill", T
        self.failed = False
        for B in BATCH_NCU if a.ncu else (BATCH_SMOKE if a.grid_size == "smoke" else BATCH_FULL):
            if self.failed:
                break
            yield "decode", B

    def run_point(self, point):
        phase, n = point
        a = self.a
        try:
            us = self.prefill_us(n) if phase == "prefill" else self.decode_us(n)
        except Exception as ex:
            print(f"  {phase} {'T' if phase == 'prefill' else 'B'}={n} FAILED: {type(ex).__name__}: {ex}", flush=True)
            traceback.print_exc()
            self.failed = True
            return None
        if us is None:
            return None
        if phase == "prefill":
            print(f"  prefill T={n:5d}: {us:8.2f} us/call  x{a.gdn_layers}L={us * a.gdn_layers:9.1f} us", flush=True)
            return ProbeRow(
                "gdn",
                f"gdn/{self.base}_prefill.csv",
                {"seq_len": n},
                {"us_per_call": round(us, 3), "us_x_nlinear": round(us * a.gdn_layers, 3)},
                "eager",
            )
        print(
            f"  {'ncu ' if a.ncu else ''}decode B={n:4d}: {us:8.2f} us/call  x{a.gdn_layers}L={us * a.gdn_layers:9.1f} us",
            flush=True,
        )
        # Decode is CUDA-graphed in serving, so NCU timing is faithful.
        return ProbeRow(
            "gdn",
            f"gdn/{self.base}_decode.csv",
            {"batch": n},
            {"us_per_call": round(us, 3), "us_x_nlinear": round(us * a.gdn_layers, 3)},
            "ncu" if a.ncu else "eager",
        )

    def prefill_us(self, T):
        a = self.a
        q, k, v, g, beta = mk(1, T, a.value_heads, a.key_dim, a.value_dim, self.dev, self.dt)

        def call():
            return chunk_gated_delta_rule(
                q, k, v, g, beta, scale=self.scale, output_final_state=True, use_qk_l2norm_in_kernel=True
            )

        return time_call(call)

    def decode_us(self, B):
        a = self.a
        H, K, V = a.value_heads, a.key_dim, a.value_dim
        if a.ncu:
            us = ncu_min_op_us(_ncu_decode_inner(B, H, K, V), a.ncu_bin, NCU_REPS)
            if us is None:
                print(f"  ncu MISS decode B={B}", flush=True)
            return us
        q, k, v, g, beta = mk(B, 1, H, K, V, self.dev, self.dt)
        state = torch.randn(B, H, V, K, device=self.dev, dtype=torch.float32)
        # Serving runs fused_sigmoid_gating_delta_rule_update; the generic rule is the same math.

        def call():
            return fused_recurrent_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                scale=self.scale,
                initial_state=state,
                inplace_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )

        return time_call(call)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--value-heads", type=int, default=32)
    ap.add_argument("--key-dim", type=int, default=128)
    ap.add_argument("--value-dim", type=int, default=128)
    ap.add_argument("--gdn-layers", type=int, default=24)
    ap.add_argument("--grid-size", choices=["smoke", "full"], default="smoke")
    add_output_args(ap)
    ap.add_argument("--gdn-geometry", default="9b", help="GDN head geometry tag, e.g. 9b (H_v=32) / 27b (H_v=48)")
    ap.add_argument("--max-mem-gb", type=float, default=30.0)
    ap.add_argument("--ncu", action="store_true", help="NCU per-kernel timing (the standard); curated grid")
    ap.add_argument("--ncu-bin", default=os.environ.get("NCU_BIN", "ncu"))
    a = ap.parse_args(argv)
    _timing.require_backend("gdn", "cuda", "rocm")
    if a.ncu:
        _timing.require_ncu()
    print(
        f"[gdn{'/ncu' if a.ncu else ''}] dev={_timing.device_name()} H={a.value_heads} K={a.key_dim} V={a.value_dim} gdn_layers={a.gdn_layers}",
        flush=True,
    )
    n = GdnProbe(a).run()
    print(f"[gdn] {n} rows of gdn/{a.gpu_label}_{a.gdn_geometry}_decode.csv and _prefill.csv", flush=True)


if __name__ == "__main__":
    run_recorded(main, "gdn")
