#!/usr/bin/env python3
"""A/B the two DistributionalSampler paths on any trajectory dataset.

`DistributionalSampler.sample_session()` replays a WHOLE recorded session when
turn samples carry `source_session_id`, and otherwise rebuilds one from
per-turn-index marginals. `build_trajectory_distribution()` never set that field,
so every trajectory-built distribution silently used the marginal path.

This measures what that cost, per dataset, on held-out data:

  observed  = distribution built from the TEST half (real per-turn values, using
              the builder's own estimator so both sides are measured identically)
  marginal  = fit on TRAIN, `source_session_id` stripped -> marginal path
  session   = fit on TRAIN, `source_session_id` kept     -> whole-session path

Reporting the same quantity for both variants isolates the path, since the fit
data, seed, and estimator are identical.

Usage:
    python scripts/holdout_marginal_vs_session.py --traj data/swebench_trajectories.jsonl
    python scripts/holdout_marginal_vs_session.py --all
"""

from __future__ import annotations

import argparse
import copy
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

DATA = ROOT / "data"
DATASETS = [
    "swebench_trajectories.jsonl",
    "terminalbench_trajectories.jsonl",
    "osworld_trajectories.jsonl",
    "claweval_trajectories.jsonl",
]
METRICS = ("context", "output", "cache_hit", "turns")


def pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def relerr(a, b):
    if not b or b != b or a != a:
        return float("nan")
    return (a - b) / b * 100.0


def payload_for(rows, tmp: Path):
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return build_trajectory_distribution("holdout", tmp)


def observed_from_payload(payload) -> dict:
    """Real per-turn values, straight out of the builder's own sample rows."""
    turns = payload["samples"]["turns"]
    by_sess: dict[str, int] = {}
    ctx, out, hit = [], [], []
    for r in turns:
        ctx.append(float(r["total_context_tokens"]))
        out.append(float(r["output_tokens"]))
        hit.append(float(r["cache_hit_rate"]))
        sid = r.get("source_session_id")
        if sid is not None:
            by_sess[sid] = by_sess.get(sid, 0) + 1
    tc = [float(v) for v in by_sess.values()] or \
         [float(v) for v in payload["samples"]["turn_count"]]
    return {"context": ctx, "output": out, "cache_hit": hit, "turns": tc}


def generated(payload, n, seed, marginal: bool, tmp: Path) -> dict:
    p = copy.deepcopy(payload)
    if marginal:
        for r in p["samples"]["turns"]:
            r.pop("source_session_id", None)
    dist = parse_trace_distribution(p, path=tmp)
    max_turns = max(p["samples"]["turn_count"])
    s = DistributionalSampler(dist, seed=seed, min_turns=1, max_turns=max_turns)
    ctx, out, hit, tc = [], [], [], []
    for _ in range(n):
        specs = getattr(s.sample_session(), "specs", []) or []
        tc.append(float(len(specs)))
        for sp in specs:
            if sp.total_context_tokens <= 0:
                continue
            ctx.append(float(sp.total_context_tokens))
            out.append(float(sp.output_tokens))
            hit.append(float(sp.cache_hit_rate))
    return {"context": ctx, "output": out, "cache_hit": hit, "turns": tc}


def run(path: Path, splits: int, seed: int) -> None:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    tmp = DATA / "_holdout_tmp.jsonl"
    acc = {(v, m): [] for v in ("marginal", "session") for m in METRICS}
    try:
        for i in range(splits):
            rng = random.Random(seed + i)
            sh = rows[:]
            rng.shuffle(sh)
            mid = len(sh) // 2
            train, test = sh[:mid], sh[mid:]
            if not train or not test:
                return
            obs = observed_from_payload(payload_for(test, tmp))
            tp = payload_for(train, tmp)
            for variant, marg in (("marginal", True), ("session", False)):
                gen = generated(tp, len(test), seed + i, marg, tmp)
                for m in METRICS:
                    acc[(variant, m)].append(relerr(pct(gen[m], .5), pct(obs[m], .5)))
    finally:
        tmp.unlink(missing_ok=True)

    print(f"\n=== {path.name}  ({len(rows)} sessions, {splits} splits) ===")
    print(f"  {'metric':<11} {'marginal p50 err':>18} {'session p50 err':>18}   verdict")
    for m in METRICS:
        a = [x for x in acc[("marginal", m)] if x == x]
        b = [x for x in acc[("session", m)] if x == x]
        if not a or not b:
            continue
        ma, mb = statistics.mean(a), statistics.mean(b)
        better = "session better" if abs(mb) < abs(ma) - 1 else (
            "marginal better" if abs(ma) < abs(mb) - 1 else "~same")
        print(f"  {m:<11} {ma:>17.1f}% {mb:>17.1f}%   {better}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", type=Path)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    targets = ([DATA / d for d in DATASETS] if args.all
               else [args.traj] if args.traj else [])
    if not targets:
        ap.error("pass --traj or --all")
    for t in targets:
        if not t.exists():
            print(f"[skip] missing {t.name}")
            continue
        run(t, args.splits, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
