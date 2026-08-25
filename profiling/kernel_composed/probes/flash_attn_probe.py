#!/usr/bin/env python3
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
import os
import statistics as st
from pathlib import Path

import torch
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func
from _ncu import ncu_op_us

KV_AXIS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
BATCH_AXIS = [1, 2, 4, 8, 16, 32, 40, 64, 80, 120, 160, 200, 256, 320]
SEQ_AXIS = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
REPS, WARMUP = 30, 5
# NCU is ~100-1000x slower per kernel -> curated (kv,batch)/seq grids; the loader interpolates.
KV_NCU = [512, 2048, 8192, 16384]
BATCH_NCU = [1, 8, 32, 128, 320]
NCU_REPS = 5   # op invocations profiled per point; min = warm steady state


def _ncu_us(inner: str, ncu_bin: str) -> float | None:
    """Per-call pure-kernel us via NCU (sums the flash sub-kernels, drops torch setup)."""
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
        times.append(s.elapsed_time(e) * 1000.0)  # ms -> us
    return st.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-heads", type=int, required=True)
    ap.add_argument("--n-kv-heads", type=int, required=True)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--fa-version", type=int, default=0, help="0=auto (FA3 on Hopper sm90+, else FA2)")
    ap.add_argument("--gpu-label", required=True, help="device label matching device_spec YAML name, e.g. A100/H100/RTX3090")
    ap.add_argument("--tag", required=True, help="grid tag, e.g. tp1/tp2/tp4 (head config)")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--max-mem-gb", type=float, default=40.0)
    ap.add_argument("--ncu", action="store_true", help="NCU per-kernel timing (the standard); curated grid")
    ap.add_argument("--ncu-bin", default=os.environ.get("NCU_BIN", "ncu"))
    a = ap.parse_args()
    if a.fa_version == 0:   # auto: FA3 is Hopper-only (sm90+); Ampere/older need FA2
        a.fa_version = 3 if torch.cuda.get_device_capability()[0] >= 9 else 2
        print(f"[flash] auto fa_version={a.fa_version} for {torch.cuda.get_device_name(0)}", flush=True)

    dev = torch.device("cuda")
    dt = torch.bfloat16
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dec_rows = []
    for kv in (KV_NCU if a.ncu else KV_AXIS):
        for b in (BATCH_NCU if a.ncu else BATCH_AXIS):
            if 2 * b * kv * a.n_kv_heads * a.head_dim * 2 / 1e9 > a.max_mem_gb:
                continue
            if a.ncu:
                us = _ncu_us(_decode_inner(b, a.n_heads, a.n_kv_heads, a.head_dim, kv, a.fa_version), a.ncu_bin)
                if us is None:
                    print(f"  ncu MISS decode kv={kv} b={b}", flush=True)
                    continue
            else:
                q = torch.randn(b, a.n_heads, a.head_dim, device=dev, dtype=dt)
                k = torch.randn(b * kv, a.n_kv_heads, a.head_dim, device=dev, dtype=dt)
                v = torch.randn_like(k)
                cu_q = torch.arange(0, b + 1, dtype=torch.int32, device=dev)
                cu_k = torch.arange(0, (b + 1) * kv, kv, dtype=torch.int32, device=dev)
                us = time_call(lambda: flash_attn_varlen_func(
                    q, k, v, 1, cu_q, kv, cu_seqlens_k=cu_k, causal=True,
                    fa_version=a.fa_version))
                del q, k, v
                torch.cuda.empty_cache()
            dec_rows.append({"q_len": 1, "kv_len": kv, "n_heads": a.n_heads,
                             "n_kv_heads": a.n_kv_heads, "head_dim": a.head_dim,
                             "batch": b, "causal": False, "phase": "decode",
                             "latency_us": round(us * a.layers, 3)})
            print(f"{'ncu ' if a.ncu else ''}decode kv={kv} b={b}: {us:.1f}us/call", flush=True)

    # Prefill is EAGER in serving (dynamic shapes, not CUDA-graphed) so dispatch
    # overhead is real -> always cuda_event, never NCU (which would drop it).
    pf_rows = []
    for seq in SEQ_AXIS:
        q = torch.randn(seq, a.n_heads, a.head_dim, device=dev, dtype=dt)
        k = torch.randn(seq, a.n_kv_heads, a.head_dim, device=dev, dtype=dt)
        v = torch.randn_like(k)
        cu = torch.tensor([0, seq], dtype=torch.int32, device=dev)
        us = time_call(lambda: flash_attn_varlen_func(
            q, k, v, seq, cu, seq, cu_seqlens_k=cu, causal=True,
            fa_version=a.fa_version))
        del q, k, v
        torch.cuda.empty_cache()
        pf_rows.append({"gpu": a.gpu_label, "prefill_tokens": seq, "q_len": seq,
                        "kv_len": seq, "causal": True,
                        "fa_version": f"vllm-fa{a.fa_version}",
                        "n_heads": a.n_heads, "n_kv_heads": a.n_kv_heads,
                        "head_dim": a.head_dim, "dtype": "bfloat16",
                        "layers": a.layers,
                        "flash_ms_median": round(us / 1000.0, 6),
                        "flash_ms_mean": round(us / 1000.0, 6),
                        "flash_full_model_ms": round(us / 1000.0 * a.layers, 6)})
        print(f"prefill seq={seq}: {us:.1f}us/call", flush=True)

    # Method-explicit kernel_data layout the loader resolves: decode grids read
    # from ncu/ (CUDA-graphed decode -> NCU faithful), prefill from cuda_event/
    # (eager prefill -> dispatch real). Decode's method follows --ncu; prefill is
    # always cuda_event. Grids are CONSOLIDATED per (method, kind, GPU): one
    # {gpu_label}.csv holds every measured head config (rows carry the geometry;
    # the loader selects by (n_heads, n_kv_heads, head_dim)), so a probe run
    # UPSERTS: it replaces rows at the head config it just measured and keeps
    # every other head config's rows.
    def _upsert(path, rows):
        geom_cols = ("n_heads", "n_kv_heads", "head_dim")
        geom = tuple(str(rows[0][c]) for c in geom_cols)
        fields = list(rows[0].keys())
        kept = []
        if path.exists():
            with path.open(newline="") as fh:
                r = csv.DictReader(fh)
                kept = [row for row in r
                        if tuple(str(row.get(c)) for c in geom_cols) != geom]
                for c in (r.fieldnames or []):
                    if c not in fields:
                        fields.append(c)
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, restval="")
            w.writeheader()
            w.writerows(kept + rows)
        return len(kept)

    dec_method = "ncu" if a.ncu else "cuda_event"
    (out / dec_method / "flash_attn").mkdir(parents=True, exist_ok=True)
    dec_path = out / dec_method / "flash_attn" / f"{a.gpu_label}.csv"
    dec_kept = _upsert(dec_path, dec_rows)
    (out / "cuda_event" / "fa3_prefill").mkdir(parents=True, exist_ok=True)
    pf_path = out / "cuda_event" / "fa3_prefill" / f"{a.gpu_label}.csv"
    pf_kept = _upsert(pf_path, pf_rows)
    print(f"upserted {dec_path} ({len(dec_rows)} new cells, {dec_kept} kept) and "
          f"{pf_path} ({len(pf_rows)} new rows, {pf_kept} kept)")


if __name__ == "__main__":
    main()
