#!/usr/bin/env python3
# profiling/emit/build_roofline_utils.py
"""Build the roofline-utils artifact from the pinned serving-wall step traces (gitignored working-set on this host — see the de-fit entry).

Audit-v2 G4: ``util_bw = 0.93`` reproduces from a real trace byte-wise but matches NO
documented computation (the documented recipe gives 0.945; real step walls give
0.90-0.92; the cited step 17 is a 290 ms outlier). This builder replaces the hand-read
anchor with a DETERMINISTIC, PRE-REGISTERED recipe — the recipe text was committed to
``profiling/docs/defit_log_entries/L6-utils.md`` BEFORE any number was computed, and this
script implements exactly that text. No RNG, no fitting; medians over rule-selected steps.

Inputs (pinned, audit-verified; gitignored working-set artifacts, see L6-utils.md):
    profile_data/_archive/vllm_engine_step_trace_{swe_c40_t12,swe_c80_t12,swe_c320_t2,
    terminal_c80_t16}_benchmark_serving_wall.jsonl

Outputs:
    profile_data/kernels/roofline_utils_<GPU>.json

Conventions re-derived (each follows the constant's OWN pinned documentation):
  - util_bw:    median of bw_roofline_ms / engine_step_wall_ms over steady-state,
                bandwidth-dominated, outlier-trimmed decode-only steps
                (closed_form_tpot prices tpot = bw/util_bw with NO separate sched term).
  - util_flops: median of compute_roofline_ms / model_submit_wall_ms over pure-prefill
                steps >= 1024 scheduled tokens (the pinned anchor's own wall convention).
  - sched:      median of engine_step_wall_ms - model_submit_wall_ms over decode_batch==1
                steps (the pinned comment's lowest-work-decode population).

Usage:
    python3 -m profiling.emit.build_roofline_utils [--gpu H100] [--archive DIR]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRACES = [
    "vllm_engine_step_trace_swe_c40_t12_benchmark_serving_wall.jsonl",
    "vllm_engine_step_trace_swe_c80_t12_benchmark_serving_wall.jsonl",
    "vllm_engine_step_trace_swe_c320_t2_benchmark_serving_wall.jsonl",
    "vllm_engine_step_trace_terminal_c80_t16_benchmark_serving_wall.jsonl",
]

# Pre-registered thresholds (L6-utils.md). SCHED_PRIOR is an eligibility gate only —
# it never enters the output arithmetic.
SCHED_PRIOR_MS = 5.7
FULL_BATCH_FRAC = 0.9
WARMUP_STEP_ID = 5
BW_DOMINANCE = 2.0          # bw_roofline_ms >= BW_DOMINANCE * SCHED_PRIOR_MS
WALL_TRIM = (0.5, 2.0)      # vs per-trace median of the candidate set
PREFILL_MIN_TOKENS = 1024
UTIL_FLOPS_MIN_N = 5


def _f(v) -> float:
    return float(v) if v not in ("", None) else 0.0


def _i(v) -> int:
    return int(float(v)) if v not in ("", None) else 0


def load_trace(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def reconstruct_contexts(steps: list[dict]) -> list[dict | None]:
    """Per decode-only step: total context tokens across the decode batch.

    prompt_tokens[id] from engine_cache_truth.requests[*].prompt_tokens (first sighting
    wins); generated[id] = # prior steps where id appeared in decode_request_ids.
    Returns one entry per step: None for non-decode-only steps or steps with an
    unknown-prompt request; else {"ctx_total": int, "batch": int}.
    """
    prompt_tokens: dict[str, int] = {}
    generated: dict[str, int] = {}
    out: list[dict | None] = []
    for s in steps:
        ect = s.get("engine_cache_truth")
        if ect:
            try:
                truth = json.loads(ect)
            except (TypeError, ValueError):
                truth = {}
            for req in truth.get("requests", []):
                rid, pt = req.get("request_id"), req.get("prompt_tokens")
                if rid and pt and rid not in prompt_tokens:
                    prompt_tokens[rid] = int(pt)
        decode_ids = [r for r in (s.get("decode_request_ids") or "").split() if r]
        is_decode_only = (
            _i(s.get("decode_batch")) > 0
            and _i(s.get("prefill_tokens")) == 0
            and str(s.get("model_executed")) == "true"
        )
        if is_decode_only:
            if any(r not in prompt_tokens for r in decode_ids):
                out.append(None)  # unknown-context request -> step excluded
            else:
                ctx = sum(prompt_tokens[r] + generated.get(r, 0) for r in decode_ids)
                out.append({"ctx_total": ctx, "batch": len(decode_ids)})
        else:
            out.append(None)
        for r in decode_ids:  # this step generates 1 token per decode request
            generated[r] = generated.get(r, 0) + 1
    return out


def quartile_medians(pairs: list[tuple[float, float]]) -> list[dict]:
    """pairs = (bw_roofline_ms, ratio); medians per byte-scale quartile."""
    if not pairs:
        return []
    pairs = sorted(pairs)
    out = []
    n = len(pairs)
    for q in range(4):
        chunk = pairs[q * n // 4: (q + 1) * n // 4] or pairs[-1:]
        out.append({
            "bw_roofline_ms_range": [round(chunk[0][0], 2), round(chunk[-1][0], 2)],
            "n": len(chunk),
            "util_bw_median": round(statistics.median(r for _, r in chunk), 4),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="H100")
    ap.add_argument("--archive", default=str(REPO_ROOT / "profile_data" / "_archive"))
    args = ap.parse_args()

    params_path = (REPO_ROOT / "profile_data" / "kernels"
                   / f"roofline_params_{args.gpu}_llama31_8b.json")
    params = json.loads(params_path.read_text())
    n_params = float(params["n_params"])
    bpp = float(params["bytes_per_param"])
    kv_bpt = float(params["kv_bytes_per_token"])
    peak_flops = float(params["peak_flops_per_s"])
    peak_bw = float(params["peak_bw_bytes_per_s"])

    archive = Path(args.archive)
    per_trace: dict[str, dict] = {}
    bw_pool: list[tuple[float, float]] = []      # (bw_roofline_ms, ratio_wall)
    bw_pool_net: list[float] = []                # ratio vs (wall - SCHED_PRIOR)
    flops_pool: list[float] = []
    flops_pool_engine: list[float] = []
    sched_pool: list[float] = []

    for name in TRACES:
        path = archive / name
        if not path.exists():
            raise SystemExit(f"missing pinned trace: {path} (see L6-utils.md provenance)")
        steps = load_trace(path)
        ctxs = reconstruct_contexts(steps)
        key = name.split("_benchmark")[0].replace("vllm_engine_step_trace_", "")
        tr: dict = {"steps": len(steps)}

        # ---------------- util_bw candidates (rules 1-3), then trim (rule 4)
        max_batch = max((c["batch"] for c in ctxs if c), default=0)
        candidates = []
        excluded_unknown_ctx = 0
        for s, c in zip(steps, ctxs):
            if (_i(s.get("decode_batch")) > 0 and _i(s.get("prefill_tokens")) == 0
                    and str(s.get("model_executed")) == "true" and c is None):
                excluded_unknown_ctx += 1
            if c is None:
                continue
            if c["batch"] < FULL_BATCH_FRAC * max_batch:
                continue
            if _i(s.get("step_id")) <= WARMUP_STEP_ID:
                continue
            bw_ms = (bpp * n_params + c["ctx_total"] * kv_bpt + c["batch"] * kv_bpt) \
                / peak_bw * 1e3
            if bw_ms < BW_DOMINANCE * SCHED_PRIOR_MS:
                continue
            wall = _f(s.get("engine_step_wall_ms"))
            if wall <= 0:
                continue
            candidates.append((bw_ms, wall))
        trimmed = []
        if candidates:
            med_wall = statistics.median(w for _, w in candidates)
            lo, hi = WALL_TRIM[0] * med_wall, WALL_TRIM[1] * med_wall
            for bw_ms, wall in candidates:
                if lo <= wall <= hi:
                    trimmed.append((bw_ms, wall))
        ratios = [(bw, bw / w) for bw, w in trimmed]
        bw_pool.extend(ratios)
        bw_pool_net.extend(bw / max(w - SCHED_PRIOR_MS, 1e-9) for bw, w in trimmed)
        tr["util_bw"] = {
            "n_candidates": len(candidates),
            "n_after_trim": len(trimmed),
            "n_outliers_trimmed": len(candidates) - len(trimmed),
            "n_decode_steps_unknown_ctx": excluded_unknown_ctx,
            "max_decode_batch": max_batch,
            "median": round(statistics.median(r for _, r in ratios), 4) if ratios else None,
        }

        # ---------------- util_flops candidates
        fl_cand = []
        for s in steps:
            if not (_i(s.get("prefill_tokens")) >= PREFILL_MIN_TOKENS
                    and _i(s.get("decode_batch")) == 0
                    and str(s.get("model_executed")) == "true"):
                continue
            m = _i(s.get("prefill_tokens"))
            submit = _f(s.get("model_submit_wall_ms"))
            engine = _f(s.get("engine_step_wall_ms"))
            if submit <= 0:
                continue
            fl_cand.append((m, submit, engine))
        fl_trim = []
        if fl_cand:
            med_pt = statistics.median(sub / m for m, sub, _ in fl_cand)
            lo, hi = WALL_TRIM[0] * med_pt, WALL_TRIM[1] * med_pt
            fl_trim = [(m, sub, eng) for m, sub, eng in fl_cand if lo <= sub / m <= hi]
        fl_ratios = [2.0 * n_params * m / peak_flops * 1e3 / sub for m, sub, _ in fl_trim]
        flops_pool.extend(fl_ratios)
        flops_pool_engine.extend(
            2.0 * n_params * m / peak_flops * 1e3 / eng for m, _, eng in fl_trim if eng > 0)
        tr["util_flops"] = {
            "n_candidates": len(fl_cand),
            "n_after_trim": len(fl_trim),
            "median": round(statistics.median(fl_ratios), 4) if fl_ratios else None,
        }

        # ---------------- scheduler overhead
        gaps = []
        for s in steps:
            if (_i(s.get("decode_batch")) == 1 and _i(s.get("prefill_tokens")) == 0
                    and str(s.get("model_executed")) == "true"):
                gaps.append(_f(s.get("engine_step_wall_ms")) - _f(s.get("model_submit_wall_ms")))
        sched_pool.extend(gaps)
        tr["sched"] = {
            "n": len(gaps),
            "median": round(statistics.median(gaps), 4) if gaps else None,
        }
        per_trace[key] = tr

    util_bw = statistics.median(r for _, r in bw_pool) if bw_pool else None
    util_flops = (statistics.median(flops_pool)
                  if len(flops_pool) >= UTIL_FLOPS_MIN_N else None)
    sched = statistics.median(sched_pool) if sched_pool else None

    out = {
        "gpu": args.gpu,
        "recipe": "profiling/docs/defit_log_entries/L6-utils.md (pre-registered 2026-06-10; "
                  "implemented by profiling/emit/build_roofline_utils.py)",
        "params_source": str(params_path.relative_to(REPO_ROOT)),
        "util_bw": round(util_bw, 4) if util_bw is not None else None,
        "util_flops": round(util_flops, 4) if util_flops is not None else None,
        "scheduler_overhead_ms_per_step": round(sched, 4) if sched is not None else None,
        "n_steps": {
            "util_bw": len(bw_pool),
            "util_flops": len(flops_pool),
            "sched": len(sched_pool),
        },
        "diagnostics": {
            "util_bw_net_of_sched_prior": round(statistics.median(bw_pool_net), 4)
            if bw_pool_net else None,
            "util_bw_net_note": "same steps, ratio vs (engine_wall - 5.7ms): the "
            "ttft_queue_sim.py:762 convention (sched added separately). Headline uses the "
            "full engine wall: closed_form_tpot adds no sched term to tpot.",
            "util_flops_vs_engine_wall": round(statistics.median(flops_pool_engine), 4)
            if flops_pool_engine else None,
            "util_flops_thinness": "pure-prefill steps are rare in decode-heavy serving "
            "traces; prefill_gemm_util_H100.json (R1 measured curve) remains the better "
            "prefill-util source.",
            "util_bw_by_byte_quartile": quartile_medians(bw_pool),
        },
        "per_trace": per_trace,
    }
    out_path = REPO_ROOT / "profile_data" / "kernels" / f"roofline_utils_{args.gpu}.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {out_path}")
    print(json.dumps({k: out[k] for k in
                      ("util_bw", "util_flops", "scheduler_overhead_ms_per_step", "n_steps")},
                     indent=2))


if __name__ == "__main__":
    main()
