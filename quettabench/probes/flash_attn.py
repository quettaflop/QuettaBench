#!/usr/bin/env python3
"""FA attention grids for one head config on vLLM's flash_attn_varlen_func: decode (kv_len x batch) -> flash_attn/{gpu}.csv rows, prefill (seq) -> fa3_prefill/{gpu}.csv rows, latency x --layers.

Decode is NCU (--ncu), graph replay (--graph) or eager; prefill is always eager, as serving runs it.
"""

from __future__ import annotations

import argparse
import os
import statistics as st

import torch
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

from quettabench.probes._timing import graph_median_us, ncu_op_us
from quettabench.probes.base import Probe, ProbeRow, add_output_args, run_recorded

KV_AXIS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
BATCH_AXIS = [1, 2, 4, 8, 16, 32, 40, 64, 80, 120, 160, 200, 256, 320]
SEQ_AXIS = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
REPS, WARMUP = 30, 5
# NCU is 100-1000x slower per kernel, hence curated grids; the loader interpolates.
KV_NCU = [512, 2048, 8192, 16384]
BATCH_NCU = [1, 8, 32, 128, 320]
NCU_REPS = 5  # invocations per point; the min is warm steady state


def _ncu_us(inner: str, ncu_bin: str) -> float | None:
    """Per-call kernel us via NCU, summing the flash kernels and dropping torch setup."""
    return ncu_op_us(inner, ncu_bin, NCU_REPS)


def _decode_inner(b, nh, nkv, hd, kv, fav) -> str:
    return (
        "import torch;"
        "from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func;"
        "d=torch.device('cuda');t=torch.bfloat16;"
        f"q=torch.randn({b},{nh},{hd},device=d,dtype=t);"
        f"k=torch.randn({b * kv},{nkv},{hd},device=d,dtype=t);v=torch.randn_like(k);"
        f"cq=torch.arange(0,{b + 1},dtype=torch.int32,device=d);"
        f"ck=torch.arange(0,{(b + 1) * kv},{kv},dtype=torch.int32,device=d);"
        f"f=lambda:flash_attn_varlen_func(q,k,v,1,cq,{kv},cu_seqlens_k=ck,causal=True,fa_version={fav});"
        f"[f() for _ in range({NCU_REPS})];torch.cuda.synchronize()"
    )


def time_call(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(REPS):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)
    return st.median(times)


class FlashAttnProbe(Probe):
    """Decode grid (kv_len x batch) of flash_attn/{gpu}.csv, then prefill grid (seq, always eager) of fa3_prefill/{gpu}.csv, for one head config."""

    def __init__(self, a):
        self.a = a
        self.dev, self.dt = torch.device("cuda"), torch.bfloat16
        self.decode_timing = "ncu" if a.ncu else "graph_replay" if a.graph else "eager"

    def grid(self):
        a = self.a
        kv_axis = [int(x) for x in a.kv_axis.split(",")] if a.kv_axis else (KV_NCU if a.ncu else KV_AXIS)
        batch_axis = [int(x) for x in a.batch_axis.split(",")] if a.batch_axis else (BATCH_NCU if a.ncu else BATCH_AXIS)
        for kv in kv_axis:
            for b in batch_axis:
                if 2 * b * kv * a.n_kv_heads * a.head_dim * 2 / 1e9 <= a.max_mem_gb:
                    yield "decode", kv, b
        # Prefill is not CUDA-graphed in serving, so its dispatch overhead is real and NCU would drop it.
        for seq in SEQ_AXIS:
            yield "prefill", seq, None

    def run_point(self, point):
        phase, n, b = point
        return self.decode(n, b) if phase == "decode" else self.prefill(n)

    def decode(self, kv, b):
        a, dev, dt = self.a, self.dev, self.dt
        if a.ncu:
            us = _ncu_us(_decode_inner(b, a.n_heads, a.n_kv_heads, a.head_dim, kv, a.fa_version), a.ncu_bin)
            if us is None:
                print(f"  ncu MISS decode kv={kv} b={b}", flush=True)
                return None
        else:
            q = torch.randn(b, a.n_heads, a.head_dim, device=dev, dtype=dt)
            k = torch.randn(b * kv, a.n_kv_heads, a.head_dim, device=dev, dtype=dt)
            v = torch.randn_like(k)
            cu_q = torch.arange(0, b + 1, dtype=torch.int32, device=dev)
            cu_k = torch.arange(0, (b + 1) * kv, kv, dtype=torch.int32, device=dev)
            fn = lambda: flash_attn_varlen_func(  # noqa: E731
                q, k, v, 1, cu_q, kv, cu_seqlens_k=cu_k, causal=True, fa_version=a.fa_version
            )
            if a.graph:
                us = graph_median_us(fn)
            else:
                us = time_call(fn)
            del q, k, v
            torch.cuda.empty_cache()
        print(f"{'ncu ' if a.ncu else ''}decode kv={kv} b={b}: {us:.1f}us/call", flush=True)
        return ProbeRow(
            "flash_attn",
            f"flash_attn/{a.gpu_label}.csv",
            {
                "q_len": 1,
                "kv_len": kv,
                "n_heads": a.n_heads,
                "n_kv_heads": a.n_kv_heads,
                "head_dim": a.head_dim,
                "batch": b,
                "causal": False,
                "phase": "decode",
            },
            {"layers": a.layers, "latency_us": round(us * a.layers, 3)},
            self.decode_timing,
        )

    def prefill(self, seq):
        a, dev, dt = self.a, self.dev, self.dt
        q = torch.randn(seq, a.n_heads, a.head_dim, device=dev, dtype=dt)
        k = torch.randn(seq, a.n_kv_heads, a.head_dim, device=dev, dtype=dt)
        v = torch.randn_like(k)
        cu = torch.tensor([0, seq], dtype=torch.int32, device=dev)
        us = time_call(
            lambda: flash_attn_varlen_func(q, k, v, seq, cu, seq, cu_seqlens_k=cu, causal=True, fa_version=a.fa_version)
        )
        del q, k, v
        torch.cuda.empty_cache()
        print(f"prefill seq={seq}: {us:.1f}us/call", flush=True)
        return ProbeRow(
            "flash_attn",
            f"fa3_prefill/{a.gpu_label}.csv",
            {
                "gpu": a.gpu_label,
                "prefill_tokens": seq,
                "q_len": seq,
                "kv_len": seq,
                "causal": True,
                "fa_version": f"vllm-fa{a.fa_version}",
                "n_heads": a.n_heads,
                "n_kv_heads": a.n_kv_heads,
                "head_dim": a.head_dim,
                "dtype": "bfloat16",
            },
            {
                "layers": a.layers,
                "flash_ms_median": round(us / 1000.0, 6),
                "flash_ms_mean": round(us / 1000.0, 6),
                "flash_full_model_ms": round(us / 1000.0 * a.layers, 6),
            },
            "eager",
        )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-heads", type=int, required=True)
    ap.add_argument("--n-kv-heads", type=int, required=True)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--fa-version", type=int, default=0, help="0=auto (FA3 on Hopper sm90+, else FA2)")
    add_output_args(ap)
    ap.add_argument("--tag", required=True, help="grid tag, e.g. tp1/tp2/tp4 (head config)")
    ap.add_argument("--max-mem-gb", type=float, default=40.0)
    ap.add_argument(
        "--kv-axis",
        default=None,
        help="comma-separated decode kv_len axis override (default tops out "
        "at 16384; a 64k-context workload needs 32768,65536 measured "
        "rather than linearly extrapolated)",
    )
    ap.add_argument("--batch-axis", default=None, help="comma-separated decode batch axis override")
    ap.add_argument("--ncu", action="store_true", help="NCU per-kernel timing (the standard); curated grid")
    ap.add_argument("--ncu-bin", default=os.environ.get("NCU_BIN", "ncu"))
    ap.add_argument(
        "--graph",
        action="store_true",
        help="CUDA-graph-replay DECODE timing (graphed-decode-faithful, "
        "no NCU needed) -> graph/flash_attn/; prefill stays eager",
    )
    a = ap.parse_args(argv)
    if a.fa_version == 0:  # auto: FA3 needs sm90+
        a.fa_version = 3 if torch.cuda.get_device_capability()[0] >= 9 else 2
        print(f"[flash] auto fa_version={a.fa_version} for {torch.cuda.get_device_name(0)}", flush=True)

    # kernel_tables keeps the last-finished run per (n_heads, n_kv_heads, head_dim).
    n = FlashAttnProbe(a).run()
    print(f"{n} rows of flash_attn/{a.gpu_label}.csv and fa3_prefill/{a.gpu_label}.csv")


if __name__ == "__main__":
    run_recorded(main, "flash_attn")
