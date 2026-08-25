#!/usr/bin/env python3
"""Held-out test of the synthetic generator using Claw-Eval sessions.

Why this is worth doing
-----------------------
The Tier-1 ablation (ablate_synthetic_generator.py) is limited two ways:

1. Statistical power. Realized pools have `n_sessions == concurrency`, so the
   low-concurrency cells compare against 1-20 sessions. Nothing conclusive can
   be said there.
2. Train-on-test. `swebench_..._filtered-mse` was FIT on the conc5 capture it is
   then compared against, so a good conc5 score is partly circular.

The Claw-Eval capture fixes both: 300 real sessions at one concurrency
(--parallel 16), which is ~15x the sessions in the conc-20 pool, and enough to
split. We fit the distribution on TRAIN sessions only and score the generator
against held-out TEST sessions it has never seen.

This measures generator fidelity in the *unsaturated* regime specifically. The
claw-eval run had 8 idle H100s at parallel=16 and was not queue-bound, so there
is little load-induced truncation -- which is exactly the regime the existing GT
cannot speak to.

Usage:
    python scripts/holdout_claweval_generator.py
    python scripts/holdout_claweval_generator.py --splits 5 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from build_trace_distributions import build_trajectory_distribution  # noqa: E402
from src.workloads.distributional import DistributionalSampler  # noqa: E402
from src.workloads.trace_distributions import parse_trace_distribution  # noqa: E402

TRAJ = ROOT / "data" / "claweval_trajectories.jsonl"


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def relerr(a: float, b: float) -> float:
    if not b or b != b or a != a:
        return float("nan")
    return (a - b) / b * 100.0


def observed(rows: list[dict]) -> dict[str, list[float]]:
    """Per-turn quantities straight from the real held-out sessions."""
    ctx, new, out, hit, turns = [], [], [], [], []
    for r in rows:
        ts = r.get("turns") or []
        turns.append(float(len(ts)))
        prev = 0.0
        for t in ts:
            total = float(t["input_tokens"])
            npf = max(1.0, total - prev)
            ctx.append(total)
            new.append(npf)
            out.append(float(t["output_tokens"]))
            hit.append(max(0.0, 1.0 - npf / total))
            prev = total
    return {"context": ctx, "prefill": new, "output": out,
            "cache_hit": hit, "turns": turns}


def generated(train_rows: list[dict], n: int, seed: int,
              tmp: Path) -> dict[str, list[float]]:
    """Fit a distribution on TRAIN only, then sample n synthetic sessions."""
    tmp.write_text("".join(json.dumps(r) + "\n" for r in train_rows))
    payload = build_trajectory_distribution("holdout_train", tmp)
    dist = parse_trace_distribution(payload, path=tmp)
    max_turns = max(int(max(len(r["turns"]) for r in train_rows)), 1)
    s = DistributionalSampler(dist, seed=seed, min_turns=1, max_turns=max_turns)

    ctx, new, out, hit, turns = [], [], [], [], []
    for _ in range(n):
        specs = getattr(s.sample_session(), "specs", []) or []
        turns.append(float(len(specs)))
        for sp in specs:
            total = float(sp.total_context_tokens)
            if total <= 0:
                continue
            ctx.append(total)
            new.append(float(sp.actual_new_prefill_tokens))
            out.append(float(sp.output_tokens))
            hit.append(float(sp.cache_hit_rate))
    return {"context": ctx, "prefill": new, "output": out,
            "cache_hit": hit, "turns": turns}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", type=Path, default=TRAJ)
    ap.add_argument("--splits", type=int, default=5, help="random 50/50 splits")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.traj.exists():
        print(f"[error] missing {args.traj}; run build_claweval_trajectories.py first")
        return 1
    rows = [json.loads(l) for l in args.traj.read_text().splitlines() if l.strip()]
    print(f"  sessions: {len(rows)}")

    tmp = ROOT / "data" / "_holdout_train.jsonl"
    metrics = ("context", "prefill", "output", "cache_hit", "turns")
    acc: dict[tuple[str, str], list[float]] = {(m, q): [] for m in metrics
                                               for q in ("p50", "p90")}
    try:
        for i in range(args.splits):
            rng = random.Random(args.seed + i)
            shuffled = rows[:]
            rng.shuffle(shuffled)
            mid = len(shuffled) // 2
            train, test = shuffled[:mid], shuffled[mid:]
            T = observed(test)
            G = generated(train, len(test), args.seed + i, tmp)
            for m in metrics:
                for q, qq in (("p50", .5), ("p90", .9)):
                    acc[(m, q)].append(relerr(pct(G[m], qq), pct(T[m], qq)))
    finally:
        tmp.unlink(missing_ok=True)

    print(f"\n  Held-out generator error over {args.splits} random 50/50 splits")
    print(f"  (fit on ~{len(rows)//2} train sessions, scored on ~{len(rows)-len(rows)//2} unseen)")
    print(f"\n  {'metric':<12} {'p50 mean':>10} {'p50 sd':>8} | {'p90 mean':>10} {'p90 sd':>8}")
    for m in metrics:
        a, b = acc[(m, "p50")], acc[(m, "p90")]
        print(f"  {m:<12} {statistics.mean(a):>9.1f}% "
              f"{(statistics.stdev(a) if len(a) > 1 else 0):>7.1f}% | "
              f"{statistics.mean(b):>9.1f}% "
              f"{(statistics.stdev(b) if len(b) > 1 else 0):>7.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
