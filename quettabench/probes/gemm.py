#!/usr/bin/env python3
"""GEMM latency grid (M, N, K) -> us for the kernel_composed GemmTable
(min-on-load semantics: a decode step runs its GEMMs warm). Two methods:

  default  CUDA-event timed torch.matmul (bf16)         -- portable, no ncu
  --ncu    NCU per-kernel gpu__time_duration (the STANDARD; matches the H100
           tables) -- slower, so a curated shape grid; needs a working ncu
  --graph  CUDA-graph-replay timing (dispatch-free, NCU-equivalent)

PRECISION (--dtype). bf16 (default) times torch.matmul. fp8 times vLLM's OWN
block-scaled FP8 linear op (`W8A8BlockFp8LinearOp`, weight blocks 128x128 +
per-token-group activation scales) -- i.e. exactly the kernel a checkpoint with
`quant_method: fp8, weight_block_size: [128,128]` (Qwen3-235B-A22B-FP8) runs. That
op quantizes the activations itself, so the measured cell includes the runtime
activation-quant kernels, which are a real and NON-negligible part of an FP8 step:
at M=1 they make FP8 SLOWER than bf16 on H200 (30us vs 19us), while at M=4096 FP8
wins 1.45x (264us vs 382us). A "2x the bf16 peak" assumption is wrong by ~2x in
both directions, which is why this is measured rather than scaled.

FP8 rows carry dtype_bytes=1 and go to a SEPARATE table (gemm/{gpu}_fp8.csv):
a GemmTable is keyed on (M,N,K) only, so mixing precisions in one file would
silently collapse two different kernels into one cell.

Emits rows of gemm/{gpu}.csv (default) or gemm_wide/{gpu}.csv (--wide), which
export puts under graph/ (--ncu, --graph) or eager/. M is the token count; (N, K) are
(out_features, in_features) of each projection (qkv / o / gate_up / down /
lm_head / router). The table interpolates off-grid.

  CUDA_VISIBLE_DEVICES=0 python3 -m quettabench.probes.gemm --gpu-label A100 --out <dir>
  NCU_BIN=/opt/nvidia/nsight-compute/2024.3.2/ncu python3 -m quettabench.probes.gemm --gpu-label A100 --ncu ...
"""
from __future__ import annotations

import argparse
import os

import torch

from quettabench.probes import _timing
from quettabench.probes._timing import graph_time_us
from quettabench.probes.base import Probe, ProbeRow, add_output_args, emit, run_recorded

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


# GEMM kernel family across archs: Ampere/Ada cuBLAS name their bf16 GEMMs
# "...gemm...", but Hopper (H100) dispatches to nvJet ("nvjet_...") and
# cutlass3x/xmma/wgmma kernels that DON'T contain "gemm". Matching the family
# still excludes the randn setup kernel (distribution_elementwise...).
GEMM_KERNELS = ("gemm", "nvjet", "cutlass", "xmma", "wgmma")


def ncu_gemm_us(m, n, k, ncu_bin, reps=NCU_REPS) -> float | None:
    """NCU pure-kernel gpu__time_duration (us) for one GEMM, min over `reps` warm
    launches of its GEMM-family kernels (_timing.ncu_kernel_durations parses the CSV)."""
    inner = (f"import torch;a=torch.randn({m},{k},device='cuda',dtype=torch.bfloat16);"
             f"b=torch.randn({k},{n},device='cuda',dtype=torch.bfloat16);"
             f"[torch.matmul(a,b) for _ in range({reps})];torch.cuda.synchronize()")
    durs = [us for name, us in _timing.ncu_kernel_durations(inner, ncu_bin)
            if any(tok in name.lower() for tok in GEMM_KERNELS)]
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
        """vllm is imported here: only --dtype fp8 needs it; bf16 runs without it."""
        from vllm.config import VllmConfig, set_current_vllm_config
        ctx = set_current_vllm_config(VllmConfig())
        ctx.__enter__()                  # deliberately never exited; see above
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
        from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
        self.ops = ops
        self.quant = QuantFP8(static=False, group_shape=GroupShape(1, FP8_WEIGHT_BLOCK[1]),
                              column_major_scales=True)
        self.quant_pts: list[tuple[int, int, float]] = []    # (m, k, us)
        print(f"[gemm/fp8] block {FP8_WEIGHT_BLOCK} | QuantFP8 + cutlass_scaled_mm "
              f"(CutlassFp8BlockScaledMMKernel path)", flush=True)

    def _time(self, run, graph: bool) -> float:
        if graph:
            return graph_time_us(run)
        for _ in range(WARMUP):
            run()
        torch.cuda.synchronize()
        ts = []
        for _ in range(REPS):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            run()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) * 1000.0)
        return min(ts)

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
            del x, w, ws
            torch.cuda.empty_cache()

    def quant_fit(self) -> tuple[float, float] | None:
        """(floor_us, eff_bw_gb_s) of the quant kernel: us = floor + bytes / bw with
        bytes = m*k*(2 read + 1 write) -- the elementwise table's convention."""
        pts = [(m * k * 3, us) for m, k, us in self.quant_pts]
        if len(pts) < 3:
            return None
        xs = [b for b, _ in pts]
        ys = [u for _, u in pts]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
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
    if graph:
        us = graph_time_us(lambda: torch.matmul(a, b))
        del a, b
        torch.cuda.empty_cache()
        return us
    for _ in range(WARMUP):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        torch.matmul(a, b)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)  # ms -> us
    del a, b
    torch.cuda.empty_cache()
    return min(ts)  # min-reduce, matching the table


class GemmProbe(Probe):
    """One run's (N, K) x M cells: NCU's curated grid with --ncu, else the
    swept or targeted pairs, each a row of gemm/ (gemm_wide/ with --wide)."""

    def __init__(self, a, fp8_op):
        self.a, self.fp8_op = a, fp8_op
        self.dev, self.dt = torch.device("cuda"), torch.bfloat16
        self.dtype_bytes = 1 if a.dtype == "fp8" else 2
        # ncu and graph replay are dispatch-free (graph/), eager includes dispatch (eager/).
        self.timing = "ncu" if a.ncu else "graph_replay" if a.graph else "eager"
        # FP8 lives in its own table: GemmTable keys on (M,N,K) only, so two precisions
        # in one file would collapse into one cell (min wins) and silently misprice both.
        stem = a.gpu_label if a.dtype == "bf16" else f"{a.gpu_label}_{a.dtype}"
        self.table = f"{'gemm_wide' if a.wide else 'gemm'}/{stem}.csv"

    def grid(self):
        a = self.a
        if a.ncu:
            print(f"[gemm/ncu] {torch.cuda.get_device_name(0)} | {a.ncu_bin} | {len(NCU_NK)} (N,K) x {len(NCU_M)} M", flush=True)
            cells = [(n, k, m) for (n, k) in NCU_NK for m in NCU_M]
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
            cells = [(n, k, m) for (n, k) in sorted(pairs) for m in m_axis]
        for n, k, m in cells:
            if (m * k + k * n + m * n) * 2 / 1e9 <= a.max_mem_gb:
                yield n, k, m

    def run_point(self, point):
        n, k, m = point
        a = self.a
        if a.ncu:
            us = ncu_gemm_us(m, n, k, a.ncu_bin)
            if us is None:
                print(f"  ncu MISS M={m} N={n} K={k} (no gemm kernel parsed)", flush=True)
                return None
            print(f"  ncu M={m:5d} N={n:6d} K={k:6d}: {us:9.2f} us", flush=True)
        else:
            try:
                us = (time_matmul_fp8(self.fp8_op, m, n, k, self.dev, graph=a.graph)
                      if self.fp8_op is not None else
                      time_matmul(m, n, k, self.dev, self.dt, graph=a.graph))
            except Exception as ex:
                print(f"  skip M={m} N={n} K={k}: {type(ex).__name__}: {ex}", flush=True)
                return None
            print(f"  M={m:5d} N={n:6d} K={k:6d}: {us:9.2f} us", flush=True)
        return ProbeRow("gemm", self.table, {"M": m, "N": n, "K": k},
                        {"dtype_bytes": self.dtype_bytes, "latency_us": round(us, 3)}, self.timing)


def main(argv=None):
    # No abbreviations: export reads --append from the argv as written.
    ap = argparse.ArgumentParser(allow_abbrev=False)
    add_output_args(ap)
    ap.add_argument("--wide", action="store_true", help="emit gemm_wide/{gpu}.csv rows (wide MoE shapes) instead of gemm/{gpu}.csv")
    ap.add_argument("--dims", type=int, nargs="*", default=None, help="override the (N,K) dim set")
    ap.add_argument("--max-mem-gb", type=float, default=30.0)
    ap.add_argument("--ncu", action="store_true", help="NCU per-kernel timing (the standard); curated grid")
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
                    help="at export, add this run's cells over earlier ok runs' tables; "
                         "without it the export drops the rows of runs that finished before it")
    a = ap.parse_args(argv)
    if a.dtype == "fp8" and a.ncu:
        raise SystemExit("--dtype fp8 has no --ncu path (the op is a Python-level "
                         "dispatch, not one kernel); use --graph for dispatch-free timing.")
    fp8_op = fp8_linear_op() if a.dtype == "fp8" else None
    probe = GemmProbe(a, fp8_op)
    n = probe.run()
    if fp8_op is not None and fp8_op.quant_fit() is not None:
        # The standalone activation-quant kernel, as an elementwise affine the
        # composition charges per FP8 GEMM whose quant the server does not fuse.
        floor, bw = fp8_op.quant_fit()
        # This entry only; kernel_tables merges it into the elementwise probe's table.
        emit(ProbeRow("gemm", f"elementwise/{a.gpu_label}.json", {"entry": "fp8_act_quant"},
                      {"value": {"floor_us": round(floor, 3), "eff_bw_gb_s": round(bw, 1)}}, probe.timing))
        print(f"[gemm/fp8] quant kernel: floor {floor:.2f} us, eff_bw {bw:.0f} GB/s "
              f"({len(fp8_op.quant_pts)} (M,K) points) -> elementwise/{a.gpu_label}.json [fp8_act_quant]",
              flush=True)
    print(f"[gemm] {n} rows of {probe.table}{', --append' if a.append else ''}", flush=True)


if __name__ == "__main__":
    run_recorded(main, "gemm")
