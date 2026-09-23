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
from pathlib import Path

import torch
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

KV_AXIS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
BATCH_AXIS = [1, 2, 4, 8, 16, 32, 40, 64, 80, 120, 160, 200, 256, 320]
SEQ_AXIS = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
REPS, WARMUP = 30, 5
# NCU is ~100-1000x slower per kernel -> curated (kv,batch)/seq grids; the loader interpolates.
KV_NCU = [512, 2048, 8192, 16384]
BATCH_NCU = [1, 8, 32, 128, 320]
NCU_REPS = 5   # op invocations profiled per point; min = warm steady state


def _ncu_us(inner: str, ncu_bin: str) -> float | None:
    """Per-call pure-kernel us via NCU (sums the flash sub-kernels, drops torch setup).

    ``_ncu`` is imported lazily: it is needed only on the --ncu path, and importing it
    at module scope made the whole probe unrunnable (including the default eager
    and --graph paths) once the helper stopped shipping alongside this file.
    """
    from _ncu import ncu_op_us  # noqa: PLC0415
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


def time_call(fn, *, graph: bool = False) -> float:
    """One cell (us): graph-replay min (decode under --graph) or eager median."""
    from _timing import time_us  # noqa: PLC0415
    return time_us(fn, graph=graph, reps=REPS, warmup=WARMUP)


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
    ap.add_argument("--n-heads", type=int, required=True)
    ap.add_argument("--n-kv-heads", type=int, required=True)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--fa-version", type=int, default=0, help="0=auto (FA3 on Hopper sm90+, else FA2)")
    ap.add_argument("--gpu-label", required=True, help="device label matching device_spec YAML name, e.g. A100/H100/RTX3090")
    ap.add_argument("--tag", required=True, help="grid tag, e.g. tp1/tp2/tp4 (head config)")
    ap.add_argument("--out-dir", default=_default_out_dir(),
                    help="kernel_data root to write into; defaults to $KERNEL_DATA / "
                         "$QUETTASIM_DATA/kernel_data (the NFS disk), else the in-repo data/")
    ap.add_argument("--max-mem-gb", type=float, default=40.0)
    ap.add_argument("--kv-axis", default=None,
                    help="comma-separated decode kv_len axis override (default tops out "
                         "at 16384; a 64k-context workload needs 32768,65536 measured "
                         "rather than linearly extrapolated)")
    ap.add_argument("--batch-axis", default=None, help="comma-separated decode batch axis override")
    ap.add_argument("--ncu", action="store_true", help="NCU per-kernel DECODE timing (cross-check of --graph); curated grid")
    ap.add_argument("--ncu-bin", default=os.environ.get("NCU_BIN", "ncu"))
    ap.add_argument("--graph", action="store_true",
                    help="CUDA-graph-replay DECODE timing (graphed-decode-faithful, "
                         "no NCU needed) -> graph/flash_attn/; prefill stays eager")
    a = ap.parse_args()
    if a.fa_version == 0:   # auto: FA3 is Hopper-only (sm90+); Ampere/older need FA2
        a.fa_version = 3 if torch.cuda.get_device_capability()[0] >= 9 else 2
        print(f"[flash] auto fa_version={a.fa_version} for {torch.cuda.get_device_name(0)}", flush=True)

    dev = torch.device("cuda")
    dt = torch.bfloat16
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    kv_axis = ([int(x) for x in a.kv_axis.split(",")] if a.kv_axis
               else (KV_NCU if a.ncu else KV_AXIS))
    batch_axis = ([int(x) for x in a.batch_axis.split(",")] if a.batch_axis
                  else (BATCH_NCU if a.ncu else BATCH_AXIS))
    dec_rows = []
    for kv in kv_axis:
        for b in batch_axis:
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
                fn = lambda: flash_attn_varlen_func(  # noqa: E731
                    q, k, v, 1, cu_q, kv, cu_seqlens_k=cu_k, causal=True,
                    fa_version=a.fa_version)
                us = time_call(fn, graph=a.graph)
                del q, k, v
                torch.cuda.empty_cache()
            dec_rows.append({"q_len": 1, "kv_len": kv, "n_heads": a.n_heads,
                             "n_kv_heads": a.n_kv_heads, "head_dim": a.head_dim,
                             "batch": b, "causal": False, "phase": "decode",
                             "layers": a.layers,
                             "latency_us": round(us * a.layers, 3)})
            print(f"{'ncu ' if a.ncu else ''}decode kv={kv} b={b}: {us:.1f}us/call", flush=True)

    # Prefill is EAGER in serving (dynamic shapes, not CUDA-graphed) so dispatch
    # overhead is real -> always eager, never NCU (which would drop it).
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
    # from graph/ (CUDA-graphed decode -> NCU faithful), prefill from eager/
    # (eager prefill -> dispatch real). Decode's method follows --ncu; prefill is
    # always eager. Grids are CONSOLIDATED per (method, kind, GPU): one
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

    dec_method = "graph" if (a.ncu or a.graph) else "eager"
    (out / dec_method / "flash_attn").mkdir(parents=True, exist_ok=True)
    dec_path = out / dec_method / "flash_attn" / f"{a.gpu_label}.csv"
    dec_kept = _upsert(dec_path, dec_rows)
    (out / "eager" / "fa3_prefill").mkdir(parents=True, exist_ok=True)
    pf_path = out / "eager" / "fa3_prefill" / f"{a.gpu_label}.csv"
    pf_kept = _upsert(pf_path, pf_rows)
    print(f"upserted {dec_path} ({len(dec_rows)} new cells, {dec_kept} kept) and "
          f"{pf_path} ({len(pf_rows)} new rows, {pf_kept} kept)")
    from _manifest import write_manifest  # noqa: PLC0415
    geom = f"{a.n_heads}q{a.n_kv_heads}kv{a.head_dim} fa{a.fa_version} layers={a.layers}"
    write_manifest(dec_path, mode="ncu" if a.ncu else ("graph_replay" if a.graph else "eager"),
                   tool="flash_attn_probe.py" + (" --ncu" if a.ncu else " --graph" if a.graph else ""),
                   reduce="min" if (a.ncu or a.graph) else "median", gpu_label=a.gpu_label,
                   reps=NCU_REPS if a.ncu else REPS, warmup=WARMUP, upsert=True,
                   notes=f"decode grid (q_len=1) over (kv_len x batch); last upsert at {geom}")
    write_manifest(pf_path, mode="eager", tool="flash_attn_probe.py", reduce="median",
                   gpu_label=a.gpu_label, reps=REPS, warmup=WARMUP, upsert=True,
                   notes=f"full-causal prefill over seq; last upsert at {geom}")


if __name__ == "__main__":
    main()
