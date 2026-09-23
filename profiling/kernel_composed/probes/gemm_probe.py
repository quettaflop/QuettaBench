#!/usr/bin/env python3
"""GEMM latency grid (M, N, K) -> us for the kernel_composed GemmTable
(min-on-load semantics: a decode step runs its GEMMs warm). Timing (see _timing.py):

  --graph  CUDA-graph-replay timing, dispatch-free -> graph/  (the STANDARD: how
           vLLM runs decode; min over replays)
  default  eager CUDA-event timing, dispatch included -> eager/ (median over calls)
  --ncu    NCU per-kernel gpu__time_duration -> graph/, mode "ncu" -- a CROSS-CHECK
           of --graph on a curated shape grid; needs a working ncu

PRECISION (--dtype). bf16 (default) times torch.matmul. fp8 times vLLM's OWN
block-scaled FP8 linear op (`W8A8BlockFp8LinearOp`, weight blocks 128x128 +
per-token-group activation scales) -- i.e. exactly the kernel a checkpoint with
`quant_method: fp8, weight_block_size: [128,128]` (Qwen3-235B-A22B-FP8) runs. That
op quantizes the activations itself, so the measured cell includes the runtime
activation-quant kernels, which are a real and NON-negligible part of an FP8 step:
at M=1 they make FP8 SLOWER than bf16 on H200 (30us vs 19us), while at M=4096 FP8
wins 1.45x (264us vs 382us). A "2x the bf16 peak" assumption is wrong by ~2x in
both directions, which is why this is measured rather than scaled.

FP8 rows carry dtype_bytes=1 and are written to a SEPARATE table
({method}/gemm/{gpu}_fp8.csv): a GemmTable is keyed on (M,N,K) only, so mixing
precisions in one file would silently collapse two different kernels into one cell.

Feeds {method}/gemm/{gpu}.csv (default) or {method}/gemm_wide/{gpu}.csv (--wide),
where method is graph/ or eager/. M is the token count; (N, K) are
(out_features, in_features) of each projection (qkv / o / gate_up / down /
lm_head / router). The table interpolates off-grid.

  CUDA_VISIBLE_DEVICES=0 python gemm_probe.py --gpu-label A100 --out-dir <dir>
  NCU_BIN=/opt/nvidia/nsight-compute/2024.3.2/ncu python gemm_probe.py --gpu-label A100 --ncu ...
"""
from __future__ import annotations
import argparse
import json, csv, io, os, statistics as st, subprocess, sys
from pathlib import Path
import torch

M_AXIS = [1, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
# out/in feature dims covering Llama-3.1-8B/70B, Qwen2.5-72B, Qwen3.5-9B/27B
# projections (hidden, 2*inter, inter, qkv-fused, o); the loader interpolates.
DIMS = [1024, 2048, 4096, 5120, 6144, 8192, 10240, 11008, 12288,
        14336, 16384, 17408, 28672, 29568]
VOCAB = [128256, 152064, 248320]   # lm_head N (with K = hidden)
HIDDENS = [4096, 5120, 8192]

# Exact (N, K) cells a model's projections query, so the table answers from a
# MEASURED cell instead of interpolating (or, off the K axis, falling to the
# roofline). Shapes are read off sum_kernels._block_linear_us:
#     qkv fused   N = (n_q/tp + 2*n_kv/kv_shards) * head_dim,  K = hidden
#     o_proj      N = hidden,                                  K = hidden/tp
#     MoE router  N = n_experts,                               K = hidden
#     lm_head     N = vocab/tp,                                K = hidden
# kv_shards = min(tp, kv_heads), so k/v stop shrinking past tp = kv_heads.
MODEL_SHAPES: dict[str, list[tuple[int, int]]] = {
    # Qwen3-235B-A22B: 94L, hidden 4096, GQA 64q/4kv, head_dim 128, 128 experts,
    # vocab 151936. tp in {1,2,4,8} (the tp that divide 64 heads).
    "qwen3-235b-a22b": [
        (9216, 4096), (4608, 4096), (2304, 4096), (1280, 4096),   # qkv @ tp 1/2/4/8
        (4096, 4096), (4096, 2048), (4096, 1024), (4096, 512),    # o_proj @ tp 1/2/4/8
        (128, 4096),                                              # MoE router gate
        (151936, 4096), (75968, 4096), (37984, 4096), (18992, 4096),  # lm_head @ tp 1/2/4/8
    ],
}
REPS, WARMUP = 50, 15
FP8_WEIGHT_BLOCK = (128, 128)   # checkpoint weight_block_size (Qwen3-235B-A22B-FP8)
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


class Fp8Op:
    """The two kernels vLLM 0.27.1 runs for one block-scaled FP8 linear on Hopper
    (model_executor/kernels/linear/scaled_mm/cutlass.py::CutlassFp8BlockScaledMMKernel):

        quant   QuantFP8(dynamic, group (1,128), column-major scales): bf16 -> fp8 + scales
        gemm    ops.cutlass_scaled_mm(A_fp8, W_fp8^T, scale_a, scale_b^T) -> bf16

    They are timed SEPARATELY. The grid stores the GEMM alone, because the server does
    not always run the quant: torch.compile's ``norm_quant`` pass fuses it into the
    RMSNorm that feeds qkv_proj (and lm_head), and the MoE experts quantize inside the
    fused_moe kernel (already in the MoE grid) -- only o_proj's input still pays a
    standalone quant. The quant kernel is fitted as an elementwise affine
    (``fp8_act_quant`` in elementwise/{gpu}.json) and the composition charges it per
    unfused GEMM. The retired W8A8BlockFp8LinearOp.apply timing folded quant + glue
    into every cell: 23.6 us at M=1 for qkv where quant+GEMM is 11.3 us.

    Needs an active vLLM config (QuantFP8 is a CustomOp that reads the compilation
    config at __init__); the context is kept open for the process lifetime. Raising
    when the import fails is correct rather than falling back to torch._scaled_mm:
    that is a PER-TENSOR-scaled GEMM, a different and faster kernel than the
    per-128x128-block one the checkpoint ships.
    """

    def __init__(self) -> None:
        from vllm.config import VllmConfig, set_current_vllm_config  # noqa: PLC0415
        ctx = set_current_vllm_config(VllmConfig())
        ctx.__enter__()                  # deliberately never exited; see above
        from vllm import _custom_ops as ops  # noqa: PLC0415
        from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8  # noqa: PLC0415
        from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape  # noqa: PLC0415
        self.ops = ops
        self.quant = QuantFP8(static=False, group_shape=GroupShape(1, FP8_WEIGHT_BLOCK[1]),
                              column_major_scales=True)
        self.quant_pts: list[tuple[int, int, float]] = []    # (m, k, us)
        print(f"[gemm/fp8] block {FP8_WEIGHT_BLOCK} | QuantFP8 + cutlass_scaled_mm "
              f"(CutlassFp8BlockScaledMMKernel path)", flush=True)

    def _time(self, run, graph: bool) -> float:
        from _timing import time_us  # noqa: PLC0415
        return time_us(run, graph=graph, reps=REPS, warmup=WARMUP)

    def gemm_us(self, m, n, k, dev, graph: bool = False) -> float:
        """GEMM kernel alone (us), input already quantized. Also times the quant kernel
        for this (m, k) once, for the elementwise fit."""
        x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        w = torch.randn(n, k, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        bn, bk = FP8_WEIGHT_BLOCK
        # Scales are positive and O(1e-2), the range a real per-block weight scale
        # takes; magnitude cannot change kernel time but a zero/NaN scale can trip a
        # fast path.
        ws = torch.rand((n + bn - 1) // bn, (k + bk - 1) // bk,
                        device=dev, dtype=torch.float32).mul_(0.01).add_(1e-4)
        try:
            if not any(pm == m and pk == k for pm, pk, _ in self.quant_pts):
                self.quant_pts.append((m, k, self._time(lambda: self.quant(x), graph)))
            xq, xs = self.quant(x)
            ops = self.ops
            return self._time(lambda: ops.cutlass_scaled_mm(
                xq, w.T, out_dtype=torch.bfloat16, scale_a=xs, scale_b=ws.T), graph)
        finally:
            del x, w, ws; torch.cuda.empty_cache()

    def quant_fit(self) -> tuple[float, float] | None:
        """(floor_us, eff_bw_gb_s) of the quant kernel: us = floor + bytes / bw with
        bytes = m*k*(2 read + 1 write) -- the elementwise table's convention."""
        pts = [(m * k * 3, us) for m, k, us in self.quant_pts]
        if len(pts) < 3:
            return None
        xs = [b for b, _ in pts]; ys = [u for _, u in pts]
        n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
        floor = my - slope * mx
        # slope is us per byte -> GB/s = 1e-3 / slope
        return max(0.0, floor), (1e-3 / slope if slope > 0 else float("inf"))


def fp8_linear_op():
    return Fp8Op()


def time_matmul_fp8(op, m, n, k, dev, graph: bool = False) -> float:
    return op.gemm_us(m, n, k, dev, graph=graph)


def time_matmul(m, n, k, dev, dt, graph: bool = False) -> float:
    a = torch.randn(m, k, device=dev, dtype=dt)
    b = torch.randn(k, n, device=dev, dtype=dt)
    from _timing import time_us  # noqa: PLC0415
    us = time_us(lambda: torch.matmul(a, b), graph=graph, reps=REPS, warmup=WARMUP)
    del a, b; torch.cuda.empty_cache()
    return us


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
    ap.add_argument("--out-dir", default=_default_out_dir(),
                    help="kernel_data root to write into; defaults to $KERNEL_DATA / "
                         "$QUETTASIM_DATA/kernel_data (the NFS disk), else the in-repo data/")
    ap.add_argument("--wide", action="store_true", help="emit {method}/gemm_wide/{gpu}.csv (wide MoE shapes) instead of {method}/gemm/{gpu}.csv")
    ap.add_argument("--dims", type=int, nargs="*", default=None, help="override the (N,K) dim set")
    ap.add_argument("--max-mem-gb", type=float, default=30.0)
    ap.add_argument("--ncu", action="store_true", help="NCU per-kernel timing (cross-check of --graph); curated grid")
    ap.add_argument("--ncu-bin", default=os.environ.get("NCU_BIN", "ncu"))
    ap.add_argument("--graph", action="store_true",
                    help="CUDA-graph-replay timing: NCU-equivalent (graphed-decode-"
                         "faithful) without needing GPU counter permissions; full grid")
    ap.add_argument("--dtype", choices=("bf16", "fp8"), default="bf16",
                    help="bf16: torch.matmul. fp8: vLLM's block-scaled FP8 linear "
                         "(128x128 weight blocks, per-token-group act scales) -> "
                         "{gpu}_fp8.csv with dtype_bytes=1")
    ap.add_argument("--model-shapes", default=None,
                    help=f"append a model's exact projection (N,K) cells: "
                         f"{','.join(sorted(MODEL_SHAPES))}")
    ap.add_argument("--pairs", nargs="*", default=None, metavar="N:K",
                    help="explicit (N,K) cells, e.g. --pairs 4608:4096 4096:2048")
    ap.add_argument("--m-axis", default=None,
                    help="comma-separated M (token) axis override. The default tops out at "
                         "8192 = the usual max_num_batched_tokens, but a MIXED step is "
                         "prefill_chunk + decode_batch tokens, so m exceeds the budget and "
                         "walks off the grid; sweep past it (e.g. 8448 = 8192 + 256 seqs)")
    ap.add_argument("--append", action="store_true",
                    help="merge into the target CSV instead of truncating it "
                         "(duplicate (M,N,K) rows collapse by min on load)")
    a = ap.parse_args()
    dev = torch.device("cuda"); dt = torch.bfloat16
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    if a.dtype == "fp8" and a.ncu:
        raise SystemExit("--dtype fp8 has no --ncu path (the op is a Python-level "
                         "dispatch, not one kernel); use --graph for dispatch-free timing.")
    fp8_op = fp8_linear_op() if a.dtype == "fp8" else None
    dtype_bytes = 1 if a.dtype == "fp8" else 2

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
        if a.pairs or a.model_shapes:
            # Targeted cell list: do NOT also sweep the full dims x dims cross
            # product (that is ~250 (N,K) x 9 M and would dwarf the cells asked for).
            pairs = set()
            for spec in (a.pairs or []):
                n_s, _, k_s = spec.partition(":")
                pairs.add((int(n_s), int(k_s)))
            for name in (a.model_shapes or "").split(","):
                if name.strip():
                    if name.strip() not in MODEL_SHAPES:
                        raise SystemExit(f"unknown --model-shapes {name!r}; "
                                         f"have {sorted(MODEL_SHAPES)}")
                    pairs.update(MODEL_SHAPES[name.strip()])
        else:
            dims = a.dims if a.dims else DIMS
            pairs = {(n, k) for n in dims for k in dims}
            for v in VOCAB:
                for h in HIDDENS:
                    pairs.add((v, h))          # lm_head
        m_axis = [int(x) for x in a.m_axis.split(",")] if a.m_axis else M_AXIS
        print(f"[gemm] {torch.cuda.get_device_name(0)} | dtype={a.dtype} | "
              f"{len(pairs)} (N,K) x {len(m_axis)} M", flush=True)
        for (n, k) in sorted(pairs):
            for m in m_axis:
                if (m * k + k * n + m * n) * 2 / 1e9 > a.max_mem_gb:
                    continue
                try:
                    us = (time_matmul_fp8(fp8_op, m, n, k, dev, graph=a.graph)
                          if fp8_op is not None else
                          time_matmul(m, n, k, dev, dt, graph=a.graph))
                except Exception as ex:
                    print(f"  skip M={m} N={n} K={k}: {type(ex).__name__}: {ex}", flush=True); continue
                rows.append({"M": m, "N": n, "K": k, "dtype_bytes": dtype_bytes,
                             "latency_us": round(us, 3)})
                print(f"  M={m:5d} N={n:6d} K={k:6d}: {us:9.2f} us", flush=True)
    # Method-explicit layout: graph/ = dispatch-free (CUDA-graph replay, or NCU as a
    # cross-check) vs eager/ (incl. dispatch); gemm_wide/ for MoE wide-shape tables,
    # gemm/ for the default narrow set. See _timing.py.
    method = "graph" if (a.ncu or a.graph) else "eager"
    mode = "ncu" if a.ncu else ("graph_replay" if a.graph else "eager")
    reduce = "min" if (a.ncu or a.graph) else "median"
    tool = "gemm_probe.py" + (" --ncu" if a.ncu else " --graph" if a.graph else "") \
        + (" --wide" if a.wide else "") + (f" --dtype {a.dtype}" if a.dtype != "bf16" else "")
    sub = f"{method}/{'gemm_wide' if a.wide else 'gemm'}"
    dst = out / sub
    dst.mkdir(parents=True, exist_ok=True)
    # FP8 lives in its own table: GemmTable keys on (M,N,K) only, so two precisions
    # in one file would collapse into one cell (min wins) and silently misprice both.
    stem = a.gpu_label if a.dtype == "bf16" else f"{a.gpu_label}_{a.dtype}"
    path = dst / f"{stem}.csv"
    if fp8_op is not None and fp8_op.quant_fit() is not None:
        # The standalone activation-quant kernel, as an elementwise affine the
        # composition charges per FP8 GEMM whose quant the server does not fuse.
        floor, bw = fp8_op.quant_fit()
        ej = out / method / "elementwise" / f"{a.gpu_label}.json"
        ej.parent.mkdir(parents=True, exist_ok=True)
        table = json.loads(ej.read_text()) if ej.exists() else {}
        table["fp8_act_quant"] = {"floor_us": round(floor, 3), "eff_bw_gb_s": round(bw, 1)}
        ej.write_text(json.dumps(table, indent=2) + "\n")
        print(f"[gemm/fp8] quant kernel: floor {floor:.2f} us, eff_bw {bw:.0f} GB/s "
              f"({len(fp8_op.quant_pts)} (M,K) points) -> {ej} [fp8_act_quant]", flush=True)
    n_new = len(rows)
    if a.append and path.exists():
        keep = {(int(r["M"]), int(r["N"]), int(r["K"])) for r in rows}
        with path.open() as f:
            prior = [r for r in csv.DictReader(f)
                     if (int(r["M"]), int(r["N"]), int(r["K"])) not in keep]
        rows = prior + rows
        print(f"[gemm] append: {len(prior)} prior rows kept, {n_new} re-measured", flush=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["M", "N", "K", "dtype_bytes", "latency_us"])
        w.writeheader(); w.writerows(rows)
    print(f"[gemm] wrote {path} ({len(rows)} rows, {n_new} new)", flush=True)
    from _manifest import write_manifest  # noqa: PLC0415
    write_manifest(path, mode=mode, tool=tool, reduce=reduce, gpu_label=a.gpu_label,
                   reps=NCU_REPS if a.ncu else REPS, warmup=WARMUP, upsert=bool(a.append),
                   notes="torch.matmul cuBLAS (bf16) / cutlass_scaled_mm (fp8); M = tokens, (N, K) = projection dims")


if __name__ == "__main__":
    main()
