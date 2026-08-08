#!/usr/bin/env python3
"""Tier-1 ablation: does the synthetic generator match REALIZED workload shape
at each concurrency?

Motivation
----------
The 44-cell LSS validation (docs/lss-validation-results.md) certifies the
SIMULATOR, and it does so via trace replay -- the synthetic generator is not in
that loop at all. Meanwhile `DistributionalSampler` takes no concurrency
argument, so it is concurrency-invariant by construction, and the "short"
distributions were fit from a single concurrency (conc5) capture. Whether a
conc5-fit, concurrency-invariant generator reproduces realized workload shape at
conc 1..320 has never been measured.

This script measures exactly that, with no simulator in the loop, so it is cheap
and can run before committing to any GPU/sim time.

Method
------
For each (benchmark, hardware, concurrency):
  realized  = by_concurrency[c].trajectory_pool, a list of sessions, each a list
              of [cached, new, output] per turn. So
                  total_context = cached + new
                  new_prefill   = new
                  cache_hit     = cached / (cached + new)
  synthetic = DistributionalSampler over the matching *synthetic* distribution,
              sampling the same number of sessions with the same seed.

Reports the signed relative error of synthetic vs realized on the p50 and p90 of
each quantity. Positive = synthetic overstates.

Usage:
    python scripts/ablate_synthetic_generator.py
    python scripts/ablate_synthetic_generator.py --hw h100 --benchmark swebench
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DIST = ROOT / "data" / "distributions"
sys.path.insert(0, str(ROOT))

from src.workloads.distributional import DistributionalSampler  # noqa: E402
from src.workloads.trace_distributions import load_trace_distribution  # noqa: E402

# synthetic distribution  ->  realized file stem
PAIRS = {
    "swebench": (
        "swebench_multiturn_short_tracereplay_filtered-mse",
        "swebench_multiturn_short_tracereplay_filtered-mse_realized",
    ),
    "terminalbench": (
        "terminalbench_multiturn_short_tracereplay_filtered-mse",
        "terminalbench_multiturn_short_tracereplay_filtered-mse_realized",
    ),
}


def realized_turns(pool: list) -> dict[str, list[float]]:
    ctx, new, out, hit, tcount = [], [], [], [], []
    for sess in pool:
        tcount.append(len(sess))
        for turn in sess:
            if not isinstance(turn, (list, tuple)) or len(turn) < 3:
                continue
            cached, nnew, o = float(turn[0]), float(turn[1]), float(turn[2])
            total = cached + nnew
            if total <= 0:
                continue
            ctx.append(total)
            new.append(nnew)
            out.append(o)
            hit.append(cached / total)
    return {"context": ctx, "prefill": new, "output": out, "cache_hit": hit,
            "turns": [float(t) for t in tcount]}


def _draw_scale(quantiles: list[float], u: float) -> float:
    """Inverse-CDF sample from a 101-point quantile vector (index = percentile)."""
    if not quantiles:
        return 1.0
    pos = max(0.0, min(1.0, u)) * (len(quantiles) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(quantiles) - 1)
    return quantiles[lo] + (quantiles[hi] - quantiles[lo]) * (pos - lo)


def synthetic_turns(dist_name: str, n_sessions: int, seed: int,
                    max_turns: int,
                    context_scale_quantiles: list[float] | None = None,
                    renormalize: bool = False,
                    ) -> dict[str, list[float]]:
    """Sample synthetic sessions.

    G1 (context_scale_quantiles=None): the shipping generator, unmodified.

    G2 (quantiles supplied): additionally draw ONE multiplier per session from
    the concurrency's realized `context_scale_quantiles` and scale that session's
    contexts by it. Semantics come from `context_scale_source`:
    "per-session median(total_context / per-(conc,turn)-median)", i.e. a
    session-level size factor relative to the typical session at that
    (concurrency, turn). Its median is ~1.0 at every concurrency, so this adds
    DISPERSION rather than shifting the level -- which is the right shape of fix
    for a p90 bias with an unbiased p50.

    Scaling is uniform within a session, so cache_hit_rate is invariant (both
    cached and new prefill scale together) and only the magnitudes move.
    """
    import random

    d = load_trace_distribution(dist_name)
    s = DistributionalSampler(d, seed=seed, min_turns=1, max_turns=max_turns)
    rng = random.Random(seed)

    # G2b needs the base distribution's own per-turn-index median context, so a
    # session's existing size factor can be divided out before the realized one
    # is applied. Without this, G2 stacks two dispersions and overshoots.
    base_median: dict[int, float] = {}
    if renormalize:
        for idx, samples in d.turns_by_index.items():
            vals = sorted(float(t.total_context_tokens) for t in samples)
            if vals:
                base_median[idx] = vals[len(vals) // 2]

    ctx, new, out, hit, tcount = [], [], [], [], []
    for _ in range(n_sessions):
        scale = (_draw_scale(context_scale_quantiles, rng.random())
                 if context_scale_quantiles else 1.0)
        sess = s.sample_session()
        # Read token accounting off `.specs` (SyntheticTurnSpec), not `.turns`
        # (BenchmarkRequest). The latter carries rendered prompt TEXT; only the
        # specs carry the generator's own token bookkeeping, which is what we
        # want to compare against realized [cached, new, output].
        specs = getattr(sess, "specs", []) or []
        tcount.append(len(specs))

        # G2b: replace the session's OWN size factor with the realized one,
        # instead of multiplying the two. own = median_t(sampled_t / base_median_t).
        if renormalize and base_median and specs:
            ratios = [
                float(sp.total_context_tokens) / base_median[sp.turn_index]
                for sp in specs
                if base_median.get(sp.turn_index)
            ]
            if ratios:
                own = sorted(ratios)[len(ratios) // 2]
                if own > 0:
                    scale = scale / own

        for sp in specs:
            total = float(sp.total_context_tokens)
            if total <= 0:
                continue
            ctx.append(total * scale)
            new.append(float(sp.actual_new_prefill_tokens) * scale)
            out.append(float(sp.output_tokens))
            hit.append(float(sp.cache_hit_rate))
    return {"context": ctx, "prefill": new, "output": out, "cache_hit": hit,
            "turns": [float(t) for t in tcount]}


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def relerr(syn: float, real: float) -> float:
    if real == 0 or real != real or syn != syn:
        return float("nan")
    return (syn - real) / real * 100.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default=None, choices=sorted(PAIRS))
    ap.add_argument("--hw", default="h100", help="realized suffix, e.g. h100, a100, h100x4")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variant", choices=("g1", "g2", "g2b", "all"), default="all",
                    help="g1=shipping; g2=+context scaling; g2b=+scaling, renormalized")
    args = ap.parse_args()

    benches = [args.benchmark] if args.benchmark else sorted(PAIRS)
    for bench in benches:
        syn_name, real_stem = PAIRS[bench]
        real_path = DIST / f"{real_stem}_{args.hw}.json"
        if not real_path.exists():
            print(f"[skip] {bench}/{args.hw}: missing {real_path.name}")
            continue
        rd = json.loads(real_path.read_text())
        bc = rd.get("by_concurrency") or {}

        variants = ["g1", "g2", "g2b"] if args.variant == "all" else [args.variant]
        print(f"\n=== {bench} / {args.hw} : synthetic vs realized "
              f"(signed % error, + = synthetic overstates) ===")
        hdr = f"{'conc':>5} {'nsess':>6} |"
        for v in variants:
            hdr += f" {v+' ctx p50':>11} {v+' ctx p90':>11} |"
        hdr += f" {'hit p50 g1':>10}"
        print(hdr)

        acc = {v: {"p50": [], "p90": []} for v in variants}
        for c in sorted(bc, key=int):
            pool = bc[c].get("trajectory_pool") or []
            if not pool:
                continue
            R = realized_turns(pool)
            if not R["context"]:
                continue
            max_turns = int(max(R["turns"])) if R["turns"] else 30
            csq = bc[c].get("context_scale_quantiles") or []

            row = f"{c:>5} {len(pool):>6} |"
            hit_err = float("nan")
            for v in variants:
                S = synthetic_turns(syn_name, len(pool), args.seed, max_turns,
                                    context_scale_quantiles=(csq if v in ("g2", "g2b") else None),
                                    renormalize=(v == "g2b"))
                if not S["context"]:
                    row += f" {'n/a':>11} {'n/a':>11} |"
                    continue
                e50 = relerr(pct(S["context"], .5), pct(R["context"], .5))
                e90 = relerr(pct(S["context"], .9), pct(R["context"], .9))
                acc[v]["p50"].append(e50)
                acc[v]["p90"].append(e90)
                row += f" {e50:>10.1f}% {e90:>10.1f}% |"
                if v == "g1":
                    hit_err = relerr(pct(S["cache_hit"], .5), pct(R["cache_hit"], .5))
            row += f" {hit_err:>9.1f}%"
            print(row)

        print(f"{'':>5} {'MEAN|err|':>6} |", end="")
        for v in variants:
            m50 = statistics.mean(abs(x) for x in acc[v]["p50"]) if acc[v]["p50"] else float("nan")
            m90 = statistics.mean(abs(x) for x in acc[v]["p90"]) if acc[v]["p90"] else float("nan")
            print(f" {m50:>10.1f}% {m90:>10.1f}% |", end="")
        print()
        # Sign test on p90: systematic bias shows up as one sign dominating.
        for v in variants:
            xs = acc[v]["p90"]
            if xs:
                neg = sum(1 for x in xs if x < 0)
                print(f"{'':>7}{v} p90 sign: {neg}/{len(xs)} negative "
                      f"(mean {statistics.mean(xs):+.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
