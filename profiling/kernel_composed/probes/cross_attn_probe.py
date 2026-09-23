#!/usr/bin/env python3
"""Rectangular (chunked-prefill) attention grid -> eager/fa3_cross/{gpu}.csv.

A chunk of ``q_len`` new tokens attending ``resident`` already-cached tokens plus
itself is ONE flash_attn_varlen_func call with q_len queries over kv_len =
resident + q_len keys (causal). Neither existing grid covers this shape: decode
grids are q=1, fa3_prefill grids are full-causal q == kv. On long-ISL traces
(Mooncake: 14k avg, 8192-token chunks) the resident part dominates prefill
attention FLOPs, so pricing it off the roofline left a ~20% under-estimate at
16-32k prompts (H200 tp2 sweep, 2026-08-25).

Eager wall-clock timing (prefill runs eager in vLLM). Rows are PER LAYER, one
call; the loader multiplies by the model's full-attention layer count.

  CUDA_VISIBLE_DEVICES=7 python cross_attn_probe.py --gpu-label H200 \
      --n-heads 16 --n-kv-heads 2 --head-dim 128 --out-dir <kernel_data>
"""
from __future__ import annotations
import argparse, csv
import os
from pathlib import Path
import torch
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

Q_AXIS = [512, 1024, 2048, 4096, 8192]
CTX_AXIS = [0, 1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768]
REPS, WARMUP = 30, 5


def time_call(fn) -> float:
    """Eager median (prefill runs eager in vLLM); see _timing.py."""
    from _timing import eager_time_us  # noqa: PLC0415
    return eager_time_us(fn, reps=REPS, warmup=WARMUP)


def _default_out_dir() -> str:
    """Where measured grids are written: the NFS data root, else the in-repo tree.

    Read straight from the environment rather than importing ``engine.paths`` so the
    probe stays standalone -- it runs under the vLLM interpreter, invoked by path,
    where the repo is not necessarily importable. Mirrors engine/paths.py: generated
    data belongs on the shared disk, not in the git working tree.
    """
    for env in ("KERNEL_DATA", "QUETTASIM_DATA"):
        raw = (os.environ.get(env) or "").strip()
        if raw:
            p = Path(raw).expanduser()
            # QUETTASIM_DATA is the data ROOT; the probes write under kernel_data/.
            return str(p if env == "KERNEL_DATA" else p / "kernel_data")
    return str(Path(__file__).resolve().parents[2] / "data" / "kernel_data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-label", required=True)
    ap.add_argument("--n-heads", type=int, required=True)
    ap.add_argument("--n-kv-heads", type=int, required=True)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--fa-version", type=int, default=0, help="0=auto (FA3 on sm90+, else FA2)")
    ap.add_argument("--out-dir", default=_default_out_dir(),
                    help="kernel_data root to write into; defaults to $KERNEL_DATA / "
                         "$QUETTASIM_DATA/kernel_data (the NFS disk), else the in-repo data/")
    ap.add_argument("--paged", action="store_true",
                    help="paged KV + block_table + seqused_k (vLLM's real call; enables "
                         "the kernel's split-KV heuristics at q << kv)")
    ap.add_argument("--q-axis", default=",".join(map(str, Q_AXIS)))
    ap.add_argument("--ctx-axis", default=",".join(map(str, CTX_AXIS)))
    a = ap.parse_args()
    if a.fa_version == 0:
        a.fa_version = 3 if torch.cuda.get_device_capability()[0] >= 9 else 2
    dev, dt = torch.device("cuda"), torch.bfloat16
    q_axis = [int(x) for x in a.q_axis.split(",")]
    ctx_axis = [int(x) for x in a.ctx_axis.split(",")]
    print(f"[cross] {torch.cuda.get_device_name(0)} heads={a.n_heads}/{a.n_kv_heads}/{a.head_dim} fa{a.fa_version}", flush=True)
    rows = []
    for ql in q_axis:
        for ctx in ctx_axis:
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
            rows.append({"n_heads": a.n_heads, "n_kv_heads": a.n_kv_heads, "head_dim": a.head_dim,
                         "q_len": ql, "resident_tokens": ctx, "kv_len": kv,
                         "fa_version": f"vllm-fa{a.fa_version}", "dtype": "bfloat16",
                         "latency_us": round(us, 3)})
            print(f"  q={ql:5d} resident={ctx:6d} kv={kv:6d}: {us:8.1f} us/layer", flush=True)
    out = Path(a.out_dir) / "eager" / "fa3_cross"; out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.gpu_label}.csv"
    geom = (str(a.n_heads), str(a.n_kv_heads), str(a.head_dim))
    kept = []
    if path.exists():
        with path.open(newline="") as fh:
            kept = [r for r in csv.DictReader(fh)
                    if (r["n_heads"], r["n_kv_heads"], r["head_dim"]) != geom]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(kept + rows)
    print(f"[cross] wrote {path} ({len(rows)} new rows, {len(kept)} kept)", flush=True)
    from _manifest import write_manifest  # noqa: PLC0415
    write_manifest(path, mode="eager", tool="cross_attn_probe.py", reduce="median", gpu_label=a.gpu_label,
                   reps=REPS, warmup=WARMUP, upsert=True,
                   notes=f"chunked-prefill attention PER LAYER over (q_len x resident); last upsert at "
                         f"{a.n_heads}q{a.n_kv_heads}kv{a.head_dim}")


if __name__ == "__main__":
    main()
