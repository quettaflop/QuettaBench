#!/usr/bin/env python3
# profiling/probes/fa_grid_probe.py
"""FA attention grids at arbitrary head configs (tp-sharded), CUDA-event timed on
vLLM's production kernel (`vllm_flash_attn.flash_attn_varlen_func`, fa_version=3
on Hopper). Emits the kernel_floor loader schemas with the grids' ALL-LAYERS
convention (per-call median x --layers).

  python fa_grid_probe.py --n-heads 16 --n-kv-heads 4 --tag fa3-tp2 --out-dir <dir>

The unsharded config (32/8) doubles as cross-validation against the NCU grid.
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

import torch
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

KV_AXIS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
BATCH_AXIS = [1, 2, 4, 8, 16, 32, 40, 64, 80, 120, 160, 200, 256, 320]
SEQ_AXIS = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
REPS, WARMUP = 30, 5


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
        times.append(s.elapsed_time(e) * 1000.0)  # ms -> us
    return st.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-heads", type=int, required=True)
    ap.add_argument("--n-kv-heads", type=int, required=True)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--fa-version", type=int, default=3)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--max-mem-gb", type=float, default=40.0)
    a = ap.parse_args()

    dev = torch.device("cuda")
    dt = torch.bfloat16
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dec_rows = []
    for kv in KV_AXIS:
        for b in BATCH_AXIS:
            if 2 * b * kv * a.n_kv_heads * a.head_dim * 2 / 1e9 > a.max_mem_gb:
                continue
            q = torch.randn(b, a.n_heads, a.head_dim, device=dev, dtype=dt)
            k = torch.randn(b * kv, a.n_kv_heads, a.head_dim, device=dev, dtype=dt)
            v = torch.randn_like(k)
            cu_q = torch.arange(0, b + 1, dtype=torch.int32, device=dev)
            cu_k = torch.arange(0, (b + 1) * kv, kv, dtype=torch.int32, device=dev)
            us = time_call(lambda: flash_attn_varlen_func(
                q, k, v, 1, cu_q, kv, cu_seqlens_k=cu_k, causal=True,
                fa_version=a.fa_version))
            dec_rows.append({"q_len": 1, "kv_len": kv, "n_heads": a.n_heads,
                             "n_kv_heads": a.n_kv_heads, "head_dim": a.head_dim,
                             "batch": b, "causal": False, "phase": "decode",
                             "latency_us": round(us * a.layers, 3)})
            del q, k, v
            torch.cuda.empty_cache()
            print(f"decode kv={kv} b={b}: {us:.1f}us/call", flush=True)

    pf_rows = []
    for seq in SEQ_AXIS:
        q = torch.randn(seq, a.n_heads, a.head_dim, device=dev, dtype=dt)
        k = torch.randn(seq, a.n_kv_heads, a.head_dim, device=dev, dtype=dt)
        v = torch.randn_like(k)
        cu = torch.tensor([0, seq], dtype=torch.int32, device=dev)
        us = time_call(lambda: flash_attn_varlen_func(
            q, k, v, seq, cu, seq, cu_seqlens_k=cu, causal=True,
            fa_version=a.fa_version))
        pf_rows.append({"gpu": "H100", "prefill_tokens": seq, "q_len": seq,
                        "kv_len": seq, "causal": True,
                        "fa_version": f"vllm-fa{a.fa_version}",
                        "n_heads": a.n_heads, "n_kv_heads": a.n_kv_heads,
                        "head_dim": a.head_dim, "dtype": "bfloat16",
                        "layers": a.layers,
                        "flash_ms_median": round(us / 1000.0, 6),
                        "flash_ms_mean": round(us / 1000.0, 6),
                        "flash_full_model_ms": round(us / 1000.0 * a.layers, 6)})
        del q, k, v
        torch.cuda.empty_cache()
        print(f"prefill seq={seq}: {us:.1f}us/call", flush=True)

    dec_path = out / f"flash_attn_H100_{a.tag}.csv"
    with dec_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dec_rows[0].keys()))
        w.writeheader()
        w.writerows(dec_rows)
    pf_path = out / f"fa3_prefill_H100_{a.tag}.csv"
    with pf_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pf_rows[0].keys()))
        w.writeheader()
        w.writerows(pf_rows)
    print(f"wrote {dec_path} ({len(dec_rows)} cells) and {pf_path} ({len(pf_rows)} rows)")


if __name__ == "__main__":
    main()
