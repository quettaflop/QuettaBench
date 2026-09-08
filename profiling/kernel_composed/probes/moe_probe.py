#!/usr/bin/env python3
"""Grouped-MoE expert-kernel grid -> {method}/moe/{gpu}.csv, plus the routing
kernels -> {method}/moe/{gpu}_routing.json.

WHY THIS PROBE EXISTS. Every other kernel family in `kernel_composed` prices off a
measured table; the grouped MoE FFN does not. `sum_kernels._moe_ffn_us` is an
analytic roofline over touched bytes/FLOPs, scaled by `util_bw`/`util_flops` --
scalars fitted on DENSE Llama GEMMs. A roofline has no kernel-launch cost and no
tile quantization, so it is optimistic exactly where a grouped MoE kernel is
launch-bound: when 128 experts each receive ~1 token. Measured against the
lss-valid H100 numbers, the roofline is -51% at a 1-token decode step, -14.5% at
8 tokens, and converges to -4% by 64 (where each expert finally has enough rows
to look like the dense GEMM the roofline assumes).

That batch-dependent deficit is what the flat `_decode_host_floor_us` constant
(decode_overhead_ms_per_layer x n_layers) currently patches -- which is why it
cannot be right at more than one operating point. Measure the kernel and the
floor can go.

WHAT IT MEASURES

  1. expert kernel  vLLM's own ``fused_experts`` over (tokens, E, top_k,
                    intermediate, hidden) -> us. `intermediate` is the PER-EXPERT
                    SwiGLU width ALREADY SHARDED by tp, so a tp sweep is just more
                    intermediate values -- which is also what replaces the fitted
                    `moe.shard_half_width` thinning curve.
  2. routing        topk_softmax and the permute/unpermute pair, fitted to the same
                    affine `floor_us + bytes / eff_bw` model the ElementwiseTable
                    uses, so they drop straight into the `moe_topk` / `moe_permute`
                    keys that today fall back to a roofline.
  3. align floor    ``moe_align_block_size``, reported as a per-MoE-layer launch
                    constant for the device YAML's ``moe.routing_launch_us``
                    (currently commented out: "no NCU sweep of it exists yet").

Routing weights are drawn from a real router GEMM + top-k, not synthesised, so the
expert-row distribution (and therefore ``moe_align_block_size`` padding) is
realistic rather than perfectly balanced.

  CUDA_VISIBLE_DEVICES=0 python moe_probe.py --gpu-label H100 --out-dir <dir>
  ... --geometries qwen3-30b-a3b,mixtral-8x7b --tp 1,2,4
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import os
from pathlib import Path

import torch

# Per-expert geometry of every MoE model in engine/device_spec/models/.
# (n_experts, top_k, intermediate_per_expert, hidden). `intermediate` is sharded
# by tp at sweep time.
#
# gpt-oss is swept for its SHAPE coverage, but the composition deliberately will
# NOT price MXFP4-expert models off this bf16 grid: only the weight-read term
# scales with expert dtype, and a measured latency cannot be decomposed after the
# fact. Those models keep the analytic roofline until an MXFP4 grid is swept.
GEOMETRIES: dict[str, tuple[int, int, int, int]] = {
    "qwen3-30b-a3b": (128, 8, 768, 2048),
    "qwen3-235b-a22b": (128, 8, 1536, 4096),
    "mixtral-8x7b": (8, 2, 14336, 4096),
    "gpt-oss-20b": (32, 4, 2880, 2880),
    "gpt-oss-120b": (128, 4, 2880, 2880),
}
# Dense at the small end: that is the decode regime where the roofline fails and
# where every agentic / low-concurrency workload lives.
TOKENS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 2048]
REPS, WARMUP = 30, 10
DT = torch.bfloat16
DT_BYTES = 2
# --dtype fp8: measure vLLM's FP8 grouped-expert path instead of the bf16 one.
# Block-scaled (128x128 weight blocks), matching a `quant_method: fp8` checkpoint's
# expert tensors (Qwen3-235B-A22B-FP8). Set by main().
_FP8 = False
FP8_BLOCK = [128, 128]
# Element counts for the routing affine fits (bytes = elements * 2 * (r + w)).
ROUTE_ELEMS = [2**e for e in range(12, 26)]


def _fused_experts():
    """vLLM's grouped MoE entry point, across the versions that moved it.

    Returns (fused_experts, fused_topk). Raising here is correct: a torch
    re-implementation would measure a different kernel than the one vLLM runs,
    which is the entire thing this probe exists to capture.
    """
    paths = [
        "vllm.model_executor.layers.fused_moe.fused_moe",
        "vllm.model_executor.layers.fused_moe",
    ]
    last = None
    for mod in paths:
        try:
            m = __import__(mod, fromlist=["fused_experts", "fused_topk"])
            fe = getattr(m, "fused_experts", None)
            ft = getattr(m, "fused_topk", None)
            if fe is not None and ft is not None:
                return fe, ft
        except Exception as ex:  # noqa: BLE001 - probe over many vLLM versions
            last = ex
    raise RuntimeError(
        "could not import vllm fused_experts/fused_topk "
        f"(tried {paths}; last error: {last}). This probe must measure vLLM's own "
        "grouped kernel -- run it in the vLLM env (PYTHON_BIN)."
    )


_GRAPH = False  # set by --graph: CUDA-graph-replay timing (graphed-decode-faithful)
# --backend flashinfer: measure FlashInfer's CUTLASS grouped-expert kernel instead of
# the Triton fused_experts. vLLM >= 0.28 serves unquantized bf16 MoE through it
# ("Using FlashInfer CUTLASS Unquantized MoE backend" in the server log), and the two
# differ a lot: on RTXPRO6000/Qwen3-235B CUTLASS is ~2x SLOWER at batch 1 and several
# times FASTER at prefill chunk sizes, so a Triton grid misprices both ends.
_BACKEND = "triton"


def _flashinfer_moe():
    from flashinfer.fused_moe import cutlass_fused_moe  # noqa: PLC0415
    return cutlass_fused_moe


def _time(fn) -> float:
    """min-of-REPS CUDA-event time (us). MIN, not mean: it matches GemmTable's
    min-on-load reduce and the warm, back-to-back kernels of a real decode step."""
    if _GRAPH:
        from _graph import graph_time_us  # noqa: PLC0415
        return graph_time_us(fn)
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return min(ts)


def _fp8_quant_config(e, inter, hidden, dev):
    """vLLM's FusedMoEQuantConfig for block-scaled FP8 experts.

    Scales are positive and O(1e-2) -- the range a real per-block weight scale
    takes. Magnitude cannot change kernel time, but a zero/NaN scale can trip a
    different code path, so they are not left as raw randn.
    """
    from vllm.model_executor.layers.fused_moe.config import (  # noqa: PLC0415
        fp8_w8a8_moe_quant_config,
    )
    bn, bk = FP8_BLOCK
    nb = lambda x, b: (x + b - 1) // b  # noqa: E731
    s1 = torch.rand(e, nb(2 * inter, bn), nb(hidden, bk),
                    device=dev, dtype=torch.float32).mul_(0.01).add_(1e-4)
    s2 = torch.rand(e, nb(hidden, bn), nb(inter, bk),
                    device=dev, dtype=torch.float32).mul_(0.01).add_(1e-4)
    return fp8_w8a8_moe_quant_config(w1_scale=s1, w2_scale=s2, block_shape=list(FP8_BLOCK))


def sweep_expert_kernel(name, e, k, inter, hidden, tokens_axis, dev, max_mem_gb):
    """One geometry's (tokens -> us) curve for the grouped expert kernel."""
    fused_experts, fused_topk = _fused_experts()
    w_bytes = 1 if _FP8 else DT_BYTES
    w_gb = (e * 2 * inter * hidden + e * hidden * inter) * w_bytes / 1e9
    if w_gb > max_mem_gb:
        print(f"  skip {name} E={e} inter={inter}: weights {w_gb:.1f} GB > {max_mem_gb}",
              flush=True)
        return []
    w1 = torch.randn(e, 2 * inter, hidden, device=dev, dtype=DT) * 0.02
    w2 = torch.randn(e, hidden, inter, device=dev, dtype=DT) * 0.02
    qc = None
    if _FP8:
        w1 = w1.to(torch.float8_e4m3fn)
        w2 = w2.to(torch.float8_e4m3fn)
        qc = _fp8_quant_config(e, inter, hidden, dev)
    rows = []
    for m in tokens_axis:
        try:
            x = torch.randn(m, hidden, device=dev, dtype=DT)
            # Real router logits -> real top-k -> realistic (imbalanced) expert
            # occupancy, so moe_align_block_size padding is representative.
            gate = torch.randn(m, e, device=dev, dtype=DT)
            # fused_topk returns (tw, tid) through vllm ~0.10 and
            # (tw, tid, token_expert_indices) from ~0.11 on.
            tw, tid, *_rest = fused_topk(x, gate, k, True)
            if _BACKEND == "flashinfer":
                if _FP8:
                    raise RuntimeError("--backend flashinfer supports bf16 only here "
                                       "(the served CUTLASS Unquantized path)")
                cutlass_fused_moe = _flashinfer_moe()
                tid_i = tid.to(torch.int)
                tw_f = tw.to(torch.float32)
                out = torch.empty_like(x)
                us = _time(lambda: cutlass_fused_moe(
                    input=x, token_selected_experts=tid_i, token_final_scales=tw_f,
                    fc1_expert_weights=w1, fc2_expert_weights=w2,
                    output_dtype=DT, quant_scales=[], output=out))
            elif qc is not None:
                us = _time(lambda: fused_experts(x, w1, w2, tw, tid, quant_config=qc))
            else:
                try:
                    us = _time(lambda: fused_experts(x, w1, w2, tw, tid, inplace=False))
                except TypeError:  # vllm >= 0.27 dropped the inplace kwarg
                    us = _time(lambda: fused_experts(x, w1, w2, tw, tid))
        except Exception as ex:  # noqa: BLE001
            print(f"  skip {name} tokens={m}: {type(ex).__name__}: {ex}", flush=True)
            continue
        rows.append({"tokens": m, "n_experts": e, "top_k": k, "intermediate": inter,
                     "hidden": hidden, "dtype_bytes": (1 if _FP8 else DT_BYTES),
                     "latency_us": round(us, 3)})
        print(f"  {name:16s} E={e:4d} k={k} inter={inter:6d} h={hidden:5d} "
              f"tok={m:5d}: {us:9.2f} us", flush=True)
    del w1, w2
    torch.cuda.empty_cache()
    return rows


def fit_affine(points) -> tuple[float, float] | None:
    """Least-squares (floor_us, eff_bw_gb_s) for latency = floor + bytes/eff_bw.
    Same model as kernel_composed.ElementwiseTable, so the output drops in."""
    pts = [(b, t) for b, t in points if b > 0 and t > 0]
    if len(pts) < 3:
        return None
    n = len(pts)
    sx = sum(b for b, _ in pts)
    sy = sum(t for _, t in pts)
    sxx = sum(b * b for b, _ in pts)
    sxy = sum(b * t for b, t in pts)
    den = n * sxx - sx * sx
    if den == 0:
        return None
    slope = (n * sxy - sx * sy) / den          # us per byte
    floor = (sy - slope * sx) / n
    if slope <= 0:
        return None
    return max(0.0, floor), 1e3 / slope        # us/byte -> GB/s


def sweep_routing(dev, e_default=128, k_default=8):
    """Affine fits for moe_topk / moe_permute + the moe_align launch floor.

    Byte counts MUST match kernel_composed/elementwise.py::_IO:
      moe_topk    (1, 0) -> elements * dtype * 1
      moe_permute (1, 1) -> elements * dtype * 2
    """
    out: dict[str, dict] = {}

    topk_pts, perm_pts = [], []
    for n in ROUTE_ELEMS:
        rows = max(1, n // e_default)
        try:
            logits = torch.randn(rows, e_default, device=dev, dtype=DT)
            t = _time(lambda: torch.topk(logits.float(), k_default, dim=-1))
            topk_pts.append((rows * e_default * DT_BYTES * 1, t))
        except Exception:  # noqa: BLE001
            pass
        try:
            hidden = 2048
            src = torch.randn(max(1, n // hidden), hidden, device=dev, dtype=DT)
            idx = torch.randint(0, src.shape[0], (src.shape[0],), device=dev)
            t = _time(lambda: torch.index_select(src, 0, idx))
            perm_pts.append((src.numel() * DT_BYTES * 2, t))
        except Exception:  # noqa: BLE001
            pass

    for key, pts in (("moe_topk", topk_pts), ("moe_permute", perm_pts)):
        fit = fit_affine(pts)
        if fit:
            out[key] = {"floor_us": round(fit[0], 4), "eff_bw_gb_s": round(fit[1], 1),
                        "n_points": len(pts)}

    # moe_align_block_size: launch-bound, no meaningful byte model -> report the
    # median as a flat per-MoE-layer constant for the device YAML.
    aligns = []
    try:
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: PLC0415
            moe_align_block_size,
        )
        for m in (8, 64, 512):
            tid = torch.randint(0, e_default, (m, k_default), device=dev, dtype=torch.int32)
            aligns.append(_time(lambda: moe_align_block_size(tid, 128, e_default)))
    except Exception as ex:  # noqa: BLE001
        print(f"  moe_align_block_size unavailable ({type(ex).__name__}); "
              f"routing_launch_us not measured", flush=True)
    if aligns:
        out["_moe_align_launch_us"] = round(st.median(aligns), 3)
    return out


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu-label", required=True)
    ap.add_argument("--out-dir", default=_default_out_dir(),
                    help="kernel_data root to write into; defaults to $KERNEL_DATA / "
                         "$QUETTASIM_DATA/kernel_data (the NFS disk), else the in-repo data/")
    ap.add_argument("--geometries", default=",".join(GEOMETRIES),
                    help="comma-separated model keys to sweep")
    ap.add_argument("--geometry-spec", default=None,
                    help="explicit geometries E:top_k:intermediate:hidden (comma-separated), "
                         "for shapes that are not a model's tp shard -- notably an "
                         "EXPERT-PARALLEL rank, which owns n_experts/ep WHOLE experts at "
                         "the FULL d_ff (e.g. 16:8:1536:4096 = Qwen3-235B at ep8). Swept "
                         "verbatim; --tp is ignored for these.")
    ap.add_argument("--tp", default="1,2,4",
                    help="tp shards to sweep (each divides `intermediate`)")
    ap.add_argument("--tokens", default=None, help="comma-separated token axis override")
    ap.add_argument("--max-mem-gb", type=float, default=40.0)
    ap.add_argument("--skip-routing", action="store_true")
    ap.add_argument("--dtype", choices=("bf16", "fp8"), default="bf16",
                    help="bf16: vLLM's unquantized fused_experts. fp8: the SAME entry "
                         "point with a block-scaled FP8 quant_config (128x128 weight "
                         "blocks), i.e. what a `quant_method: fp8` checkpoint runs. "
                         "Writes {gpu}_fp8.csv with dtype_bytes=1.")
    ap.add_argument("--append", action="store_true",
                    help="merge into the target CSV instead of truncating it")
    ap.add_argument("--graph", action="store_true",
                    help="CUDA-graph-replay timing -> ncu/moe/. Faithful for vLLM's "
                         "CUDA-graphed MoE DECODE (the eager rationale below only "
                         "holds for eager execution); slight underprice of small "
                         "eager prefill chunks, negligible at real chunk sizes.")
    ap.add_argument("--backend", choices=("triton", "flashinfer"), default="triton",
                    help="triton: vLLM's fused_experts (Triton grouped GEMM). "
                         "flashinfer: FlashInfer's CUTLASS grouped kernel -- what "
                         "vLLM >= 0.28 actually serves for unquantized bf16 MoE.")
    a = ap.parse_args()
    global _GRAPH, _FP8, _BACKEND
    _GRAPH = a.graph
    _FP8 = (a.dtype == "fp8")
    _BACKEND = a.backend
    if _FP8:
        # fused_experts' FP8 path builds a CustomOp internally, which needs a live
        # vLLM config (same reason as gemm_probe.fp8_linear_op).
        from vllm.config import VllmConfig, set_current_vllm_config  # noqa: PLC0415
        set_current_vllm_config(VllmConfig()).__enter__()

    dev = torch.device("cuda")
    tokens_axis = ([int(x) for x in a.tokens.split(",")] if a.tokens else TOKENS)
    tps = [int(x) for x in a.tp.split(",")]
    print(f"[moe] {torch.cuda.get_device_name(0)} | dtype={a.dtype} "
          f"| geometries={a.geometries} | tp={tps} | {len(tokens_axis)} token points",
          flush=True)

    rows = []
    seen: set[tuple[int, int, int, int]] = set()
    for spec in (a.geometry_spec or "").split(","):
        if not spec.strip():
            continue
        e, k, inter, hidden = (int(x) for x in spec.strip().split(":"))
        seen.add((e, k, inter, hidden))
        rows += sweep_expert_kernel(f"ep/{spec.strip()}", e, k, inter, hidden,
                                    tokens_axis, dev, a.max_mem_gb)
    for key in ("" if a.geometry_spec else a.geometries).split(","):
        key = key.strip()
        if not key:
            continue
        if key not in GEOMETRIES:
            print(f"  unknown geometry {key!r}, skipping", flush=True)
            continue
        e, k, inter, hidden = GEOMETRIES[key]
        for tp in tps:
            if inter % tp:
                continue
            shard = inter // tp
            if (e, k, shard, hidden) in seen:      # tp shards can collide across models
                continue
            seen.add((e, k, shard, hidden))
            rows += sweep_expert_kernel(f"{key}/tp{tp}", e, k, shard, hidden,
                                        tokens_axis, dev, a.max_mem_gb)

    # Default cuda_event: the grouped kernel's LAUNCH cost is the term the roofline
    # misses -- but that is only faithful for EAGER execution. vLLM CUDA-graphs MoE
    # decode, so --graph (dispatch-free, ncu/-equivalent) is the decode-faithful
    # method; H200 GT showed the eager grid overprices graphed decode 2.3x.
    dst = Path(a.out_dir) / ("ncu" if _GRAPH else "cuda_event") / "moe"
    dst.mkdir(parents=True, exist_ok=True)
    path = dst / f"{a.gpu_label}{'' if a.dtype == 'bf16' else '_' + a.dtype}.csv"
    n_new = len(rows)
    if a.append and path.exists():
        key = lambda r: (int(r["tokens"]), int(r["n_experts"]), int(r["top_k"]),  # noqa: E731
                         int(r["intermediate"]), int(r["hidden"]))
        fresh = {key(r) for r in rows}
        with path.open() as f:
            prior = [r for r in csv.DictReader(f) if key(r) not in fresh]
        rows = prior + rows
        print(f"[moe] append: {len(prior)} prior rows kept, {n_new} re-measured", flush=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tokens", "n_experts", "top_k", "intermediate",
                                          "hidden", "dtype_bytes", "latency_us"])
        w.writeheader()
        w.writerows(rows)
    print(f"[moe] wrote {path} ({len(rows)} rows, {n_new} new)", flush=True)

    if not a.skip_routing and a.dtype == "bf16":
        # Routing (topk / permute / align) is dtype-agnostic -- it moves indices and
        # bf16 hidden states either way -- so an fp8 run must not rewrite the fit.
        routing = sweep_routing(dev)
        rp = dst / f"{a.gpu_label}_routing.json"
        rp.write_text(json.dumps(routing, indent=1) + "\n")
        print(f"[moe] wrote {rp}", flush=True)
        if "_moe_align_launch_us" in routing:
            print(f"[moe] set `moe.routing_launch_us: "
                  f"{routing['_moe_align_launch_us']}` in the {a.gpu_label} device YAML",
                  flush=True)


if __name__ == "__main__":
    main()
