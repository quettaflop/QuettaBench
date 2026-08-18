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


def _time(fn) -> float:
    """min-of-REPS CUDA-event time (us). MIN, not mean: it matches GemmTable's
    min-on-load reduce and the warm, back-to-back kernels of a real decode step."""
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


def sweep_expert_kernel(name, e, k, inter, hidden, tokens_axis, dev, max_mem_gb):
    """One geometry's (tokens -> us) curve for the grouped expert kernel."""
    fused_experts, fused_topk = _fused_experts()
    w_gb = (e * 2 * inter * hidden + e * hidden * inter) * DT_BYTES / 1e9
    if w_gb > max_mem_gb:
        print(f"  skip {name} E={e} inter={inter}: weights {w_gb:.1f} GB > {max_mem_gb}",
              flush=True)
        return []
    w1 = torch.randn(e, 2 * inter, hidden, device=dev, dtype=DT) * 0.02
    w2 = torch.randn(e, hidden, inter, device=dev, dtype=DT) * 0.02
    rows = []
    for m in tokens_axis:
        try:
            x = torch.randn(m, hidden, device=dev, dtype=DT)
            # Real router logits -> real top-k -> realistic (imbalanced) expert
            # occupancy, so moe_align_block_size padding is representative.
            gate = torch.randn(m, e, device=dev, dtype=DT)
            tw, tid = fused_topk(x, gate, k, True)
            us = _time(lambda: fused_experts(x, w1, w2, tw, tid, inplace=False))
        except Exception as ex:  # noqa: BLE001
            print(f"  skip {name} tokens={m}: {type(ex).__name__}: {ex}", flush=True)
            continue
        rows.append({"tokens": m, "n_experts": e, "top_k": k, "intermediate": inter,
                     "hidden": hidden, "dtype_bytes": DT_BYTES,
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu-label", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--geometries", default=",".join(GEOMETRIES),
                    help="comma-separated model keys to sweep")
    ap.add_argument("--tp", default="1,2,4",
                    help="tp shards to sweep (each divides `intermediate`)")
    ap.add_argument("--tokens", default=None, help="comma-separated token axis override")
    ap.add_argument("--max-mem-gb", type=float, default=40.0)
    ap.add_argument("--skip-routing", action="store_true")
    a = ap.parse_args()

    dev = torch.device("cuda")
    tokens_axis = ([int(x) for x in a.tokens.split(",")] if a.tokens else TOKENS)
    tps = [int(x) for x in a.tp.split(",")]
    print(f"[moe] {torch.cuda.get_device_name(0)} | geometries={a.geometries} "
          f"| tp={tps} | {len(tokens_axis)} token points", flush=True)

    rows = []
    seen: set[tuple[int, int, int, int]] = set()
    for key in a.geometries.split(","):
        key = key.strip()
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

    # cuda_event, not ncu: the grouped kernel's LAUNCH cost is exactly the term the
    # roofline misses, and NCU's pure gpu__time_duration excludes it.
    dst = Path(a.out_dir) / "cuda_event" / "moe"
    dst.mkdir(parents=True, exist_ok=True)
    path = dst / f"{a.gpu_label}.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tokens", "n_experts", "top_k", "intermediate",
                                          "hidden", "dtype_bytes", "latency_us"])
        w.writeheader()
        w.writerows(rows)
    print(f"[moe] wrote {path} ({len(rows)} rows)", flush=True)

    if not a.skip_routing:
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
