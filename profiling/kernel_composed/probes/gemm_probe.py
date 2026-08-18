#!/usr/bin/env python3
"""GEMM latency grid (M, N, K) -> us for the kernel_composed GemmTable
(min-on-load semantics: a decode step runs its GEMMs warm). Two methods:

  default  CUDA-event timed torch.matmul (bf16)         -- portable, no ncu
  --ncu    NCU per-kernel gpu__time_duration (the STANDARD; matches the H100
           tables) -- slower, so a curated shape grid; needs a working ncu

Feeds {method}/gemm/{gpu}.csv (default) or {method}/gemm_wide/{gpu}.csv (--wide),
where method is ncu/ or cuda_event/. M is the token count; (N, K) are
(out_features, in_features) of each projection (qkv / o / gate_up / down /
lm_head / router). The table interpolates off-grid.

  CUDA_VISIBLE_DEVICES=0 python gemm_probe.py --gpu-label A100 --out-dir <dir>
  NCU_BIN=/opt/nvidia/nsight-compute/2024.3.2/ncu python gemm_probe.py --gpu-label A100 --ncu ...
"""
from __future__ import annotations
import argparse, csv, io, os, statistics as st, subprocess, sys
from pathlib import Path
import torch

M_AXIS = [1, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
# out/in feature dims covering Llama-3.1-8B/70B, Qwen2.5-72B, Qwen3.5-9B/27B
# projections (hidden, 2*inter, inter, qkv-fused, o); the loader interpolates.
DIMS = [1024, 2048, 4096, 5120, 6144, 8192, 10240, 11008, 12288,
        14336, 16384, 17408, 28672, 29568]
VOCAB = [128256, 152064, 248320]   # lm_head N (with K = hidden)
HIDDENS = [4096, 5120, 8192]
REPS, WARMUP = 50, 15
# NCU is ~100-1000x slower per kernel, so profile a curated grid, not the full sweep.
NCU_M = [1, 64, 512, 4096]
NCU_NK = [(4096, 4096), (8192, 4096), (4096, 8192), (14336, 4096), (4096, 14336),
          (6144, 4096), (5120, 5120), (11008, 4096), (28672, 8192), (8192, 8192),
          (152064, 8192), (28672, 4096)]
NCU_REPS = 3


def ncu_gemm_us(m, n, k, ncu_bin, reps=NCU_REPS) -> float | None:
    """NCU pure-kernel gpu__time_duration (us) for one GEMM, min over `reps` warm
    launches. Filters to the GEMM kernel (ignores the randn setup kernels)."""
    inner = (f"import torch;a=torch.randn({m},{k},device='cuda',dtype=torch.bfloat16);"
             f"b=torch.randn({k},{n},device='cuda',dtype=torch.bfloat16);"
             f"[torch.matmul(a,b) for _ in range({reps})];torch.cuda.synchronize()")
    cmd = [ncu_bin, "--csv", "--metrics", "gpu__time_duration.sum",
           "--target-processes", "all", sys.executable, "-c", inner]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    lines = out.splitlines()
    hdr = next((i for i, l in enumerate(lines) if l.startswith('"ID"')), None)
    if hdr is None:
        return None
    durs = []
    for r in csv.DictReader(io.StringIO("\n".join(lines[hdr:]))):
        if r.get("Metric Name") != "gpu__time_duration.sum":
            continue
        # GEMM kernel family across archs: Ampere/Ada cuBLAS name their bf16 GEMMs
        # "...gemm...", but Hopper (H100) dispatches to nvJet ("nvjet_...") and
        # cutlass3x/xmma/wgmma kernels that DON'T contain "gemm". Match the family so
        # the randn-setup kernel (distribution_elementwise...) is still excluded.
        kn = r.get("Kernel Name", "").lower()
        if not any(tok in kn for tok in ("gemm", "nvjet", "cutlass", "xmma", "wgmma")):
            continue
        val = float(r["Metric Value"].replace(",", ""))
        durs.append(val / 1000.0 if r.get("Metric Unit") == "ns" else val)
    return min(durs) if durs else None


def time_matmul(m, n, k, dev, dt) -> float:
    a = torch.randn(m, k, device=dev, dtype=dt)
    b = torch.randn(k, n, device=dev, dtype=dt)
    for _ in range(WARMUP):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); torch.matmul(a, b); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)  # ms -> us
    del a, b; torch.cuda.empty_cache()
    return min(ts)  # min-reduce, matching the table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-label", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--wide", action="store_true", help="emit {method}/gemm_wide/{gpu}.csv (wide MoE shapes) instead of {method}/gemm/{gpu}.csv")
    ap.add_argument("--dims", type=int, nargs="*", default=None, help="override the (N,K) dim set")
    ap.add_argument("--max-mem-gb", type=float, default=30.0)
    ap.add_argument("--ncu", action="store_true", help="NCU per-kernel timing (the standard); curated grid")
    ap.add_argument("--ncu-bin", default=os.environ.get("NCU_BIN", "ncu"))
    a = ap.parse_args()
    dev = torch.device("cuda"); dt = torch.bfloat16
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    rows = []
    if a.ncu:
        print(f"[gemm/ncu] {torch.cuda.get_device_name(0)} | {a.ncu_bin} | {len(NCU_NK)} (N,K) x {len(NCU_M)} M", flush=True)
        for (n, k) in NCU_NK:
            for m in NCU_M:
                if (m * k + k * n + m * n) * 2 / 1e9 > a.max_mem_gb:
                    continue
                us = ncu_gemm_us(m, n, k, a.ncu_bin)
                if us is None:
                    print(f"  ncu MISS M={m} N={n} K={k} (no gemm kernel parsed)", flush=True); continue
                rows.append({"M": m, "N": n, "K": k, "dtype_bytes": 2, "latency_us": round(us, 3)})
                print(f"  ncu M={m:5d} N={n:6d} K={k:6d}: {us:9.2f} us", flush=True)
    else:
        dims = a.dims if a.dims else DIMS
        pairs = {(n, k) for n in dims for k in dims}
        for v in VOCAB:
            for h in HIDDENS:
                pairs.add((v, h))          # lm_head
        print(f"[gemm] {torch.cuda.get_device_name(0)} | {len(pairs)} (N,K) x {len(M_AXIS)} M", flush=True)
        for (n, k) in sorted(pairs):
            for m in M_AXIS:
                if (m * k + k * n + m * n) * 2 / 1e9 > a.max_mem_gb:
                    continue
                try:
                    us = time_matmul(m, n, k, dev, dt)
                except Exception as ex:
                    print(f"  skip M={m} N={n} K={k}: {type(ex).__name__}", flush=True); continue
                rows.append({"M": m, "N": n, "K": k, "dtype_bytes": 2, "latency_us": round(us, 3)})
    # Method-explicit layout: ncu/ (per-kernel, the standard) vs cuda_event/;
    # gemm_wide/ for MoE wide-shape tables, gemm/ for the default narrow set.
    method = "ncu" if a.ncu else "cuda_event"
    sub = f"{method}/{'gemm_wide' if a.wide else 'gemm'}"
    dst = out / sub
    dst.mkdir(parents=True, exist_ok=True)
    path = dst / f"{a.gpu_label}.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["M", "N", "K", "dtype_bytes", "latency_us"])
        w.writeheader(); w.writerows(rows)
    print(f"[gemm] wrote {path} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
