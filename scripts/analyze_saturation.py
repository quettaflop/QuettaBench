#!/usr/bin/env python3
"""Locate the knee and the throughput plateau in an existing concurrency ladder.

The canonical sweeps already walk a closed-loop concurrency ladder
(sweep_all_profiles.sh: 1 10 20 40 80 160 256 320), which traces the whole
throughput-vs-load curve. Nothing read that curve back out. This does, offline,
from result JSONs already on disk -- no GPU time.

Definitions (closed loop, so the system cannot become unstable; it saturates):

  mu    service rate: max sustainable completions/sec for THIS workload. Read as
        the plateau the throughput curve flattens onto. mu is workload-specific
        -- a 150-token prompt and a 17k-token prompt have different mu on the
        same GPU.
  R0    unloaded latency: E2EL at the lowest concurrency in the ladder, i.e.
        service time with no queueing.
  N*    the knee: concurrency where the pipe is exactly full, N* = mu * R0
        (Little's Law). Below it, added concurrency buys throughput. Above it,
        throughput is pinned at mu and added concurrency only buys latency.

A ladder that never flattens gives a LOWER BOUND on mu, not mu. That case is
reported explicitly rather than fitting a plateau that was never observed.

Usage:
    python3 scripts/analyze_saturation.py results/            # a sweep dir
    python3 scripts/analyze_saturation.py results/ --json     # machine-readable
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Throughput within this fraction of the ladder max counts as "on the plateau".
PLATEAU_TOL = 0.05
# Need at least this many plateau points to claim saturation was observed.
MIN_PLATEAU_POINTS = 2


def load_cells(root: Path) -> list[dict]:
    """Read every result JSON under root into flat per-cell records."""
    cells = []
    for path in sorted(root.rglob("*.json")):
        if path.name.endswith("_per_turn.json"):
            continue
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        summary = data.get("summary") or {}
        config = data.get("config") or {}
        if not summary or summary.get("successful_requests", 0) <= 0:
            continue
        conc = summary.get("concurrency") or config.get("concurrency")
        thr = summary.get("request_throughput")
        e2el = summary.get("median_e2el_ms")
        if not conc or not thr or not e2el:
            continue
        cells.append({
            "file": path.name,
            "model": config.get("model", "?").split("/")[-1],
            "tp": config.get("tensor_parallel_size"),
            "backend": config.get("backend", "?"),
            "profile": summary.get("profile") or config.get("profile", "?"),
            "load_mode": config.get("load_mode", "closed-loop"),
            "concurrency": int(conc),
            "req_per_s": float(thr),
            "out_tok_per_s": float(summary.get("output_token_throughput") or 0.0),
            "e2el_ms": float(e2el),
            "mean_inflight": float(summary.get("mean_inflight_requests") or 0.0),
        })
    return cells


def analyse(ladder: list[dict]) -> dict:
    """Estimate mu, R0 and the knee from one (model, tp, backend, profile) ladder."""
    ladder = sorted(ladder, key=lambda c: c["concurrency"])
    peak = max(c["req_per_s"] for c in ladder)
    plateau = [c for c in ladder if c["req_per_s"] >= peak * (1 - PLATEAU_TOL)]
    saturated = (
        len(plateau) >= MIN_PLATEAU_POINTS
        # the ladder must actually turn over: the top point must not still be
        # the sole maximum, or we only ever climbed.
        and plateau[0]["concurrency"] < ladder[-1]["concurrency"]
    )

    mu = sum(c["req_per_s"] for c in plateau) / len(plateau) if saturated else peak
    r0_s = ladder[0]["e2el_ms"] / 1000.0
    knee = mu * r0_s if r0_s > 0 else 0.0

    return {
        "mu_req_per_s": mu,
        "mu_is_lower_bound": not saturated,
        "r0_s": r0_s,
        "r0_from_concurrency": ladder[0]["concurrency"],
        "knee_concurrency": knee,
        "plateau_points": [c["concurrency"] for c in plateau] if saturated else [],
        "ladder": ladder,
    }


def format_group(key: tuple, res: dict) -> str:
    model, tp, backend, profile = key
    out = []
    out.append(f"\n{'=' * 78}")
    out.append(f" {model} | tp={tp} | {backend} | {profile}")
    out.append(f"{'=' * 78}")

    mu, knee = res["mu_req_per_s"], res["knee_concurrency"]
    if res["mu_is_lower_bound"]:
        out.append(f" mu  >= {mu:.2f} req/s   (LOWER BOUND -- throughput was still")
        out.append(f"                          climbing at the top of the ladder;")
        out.append(f"                          extend it to find the real plateau)")
    else:
        out.append(f" mu   = {mu:.2f} req/s   (plateau over concurrency "
                   f"{res['plateau_points']})")
    out.append(f" R0   = {res['r0_s'] * 1000:.0f} ms      "
               f"(E2EL p50 at concurrency {res['r0_from_concurrency']})")
    out.append(f" knee = {knee:.1f} concurrent requests   (N* = mu x R0)")
    out.append("")
    out.append(f" {'conc':>6} {'req/s':>8} {'out tok/s':>10} {'E2EL p50':>10} "
               f"{'% of mu':>8}  regime")
    out.append(f" {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*8}  {'-'*22}")
    for c in res["ladder"]:
        pct = 100.0 * c["req_per_s"] / mu if mu > 0 else 0.0
        if knee <= 0:
            regime = "?"
        elif c["concurrency"] < knee * 0.75:
            regime = "pre-knee (latency-lim)"
        elif c["concurrency"] > knee * 1.5:
            regime = "post-knee (queue-lim)"
        else:
            regime = "at the knee"
        out.append(f" {c['concurrency']:>6} {c['req_per_s']:>8.2f} "
                   f"{c['out_tok_per_s']:>10.0f} {c['e2el_ms']:>9.0f}ms "
                   f"{pct:>7.0f}%  {regime}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    ap.add_argument("--min-points", type=int, default=3,
                    help="skip ladders with fewer points (default 3)")
    args = ap.parse_args()

    if not args.results_dir.is_dir():
        print(f"Not a directory: {args.results_dir}", file=sys.stderr)
        return 1

    cells = load_cells(args.results_dir)
    if not cells:
        print(f"No usable result JSONs under {args.results_dir}", file=sys.stderr)
        return 1

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for c in cells:
        # Open-loop cells are not a concurrency ladder; mixing them in would put
        # points from a different independent variable on the same curve.
        if c["load_mode"] == "open-loop":
            continue
        groups[(c["model"], c["tp"], c["backend"], c["profile"])].append(c)

    ladders = {k: v for k, v in groups.items() if len(v) >= args.min_points}
    skipped = len(groups) - len(ladders)

    if not ladders:
        print(f"No closed-loop ladder had >= {args.min_points} concurrency points "
              f"({len(groups)} group(s) found).", file=sys.stderr)
        return 1

    results = {k: analyse(v) for k, v in sorted(ladders.items(), key=lambda kv: str(kv[0]))}

    if args.json:
        print(json.dumps([
            {"model": k[0], "tp": k[1], "backend": k[2], "profile": k[3],
             **{kk: vv for kk, vv in r.items() if kk != "ladder"}}
            for k, r in results.items()
        ], indent=2))
    else:
        for k, r in results.items():
            print(format_group(k, r))
        print()
        if skipped:
            print(f"({skipped} group(s) skipped: fewer than {args.min_points} "
                  f"concurrency points)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
