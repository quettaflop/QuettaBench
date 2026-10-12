#!/usr/bin/env python3
"""Fit the sim's missing per-request first-token constant (the median TTFT residual) from a live run.

Joins a live bench_dir to the sim's per-request CSV by request index, reports TTFT error before and after the correction, and can write first_token_overhead_us into an exported meta.yaml.
An index or input-count mismatch above 5 percent aborts; the residual spread is printed because a wrong --sim-time-unit shows up as spread, not a crash.
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

BUCKETS = ((0, 512, "lt512"), (512, 2048, "512-2047"), (2048, 10 ** 9, "ge2048"))
UNIT_MS = {"ns": 1e-6, "us": 1e-3, "ms": 1.0}


def read_live(bench_dir):
    """{index: (ttft_ms, input_toks)} from requests.jsonl; TTFT is first token minus queue entry."""
    out = {}
    for line in open(Path(bench_dir) / "requests.jsonl"):
        r = json.loads(line)
        i = int(str(r["request_id"]).rsplit("-", 1)[-1])
        out[i] = ((r["first_token_ts"] - r["queued_ts"]) * 1e3, int(r["input_toks"]))
    return out


def read_sim(path, unit):
    """{index: (ttft_ms, input_toks)} from the sim's per-request CSV."""
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out[int(r["request id"])] = (float(r["TTFT"]) * UNIT_MS[unit], int(r["input"]))
    return out


def fit_overhead(pairs):
    """pairs: [(live_ms, sim_ms, input_toks)]. Returns the fit with mean absolute percentage TTFT error before and after."""
    res = sorted(live - sim for live, sim, _ in pairs)
    overhead = statistics.median(res)
    before = statistics.mean(abs(sim - live) / live for live, sim, _ in pairs)
    after = statistics.mean(abs(sim + overhead - live) / live for live, sim, _ in pairs)
    buckets = {}
    for lo, hi, label in BUCKETS:
        b = [live - sim for live, sim, toks in pairs if lo <= toks < hi]
        if b:
            buckets[label] = (statistics.median(b), len(b))
    return {"overhead_ms": overhead, "p10_ms": res[len(res) // 10],
            "p90_ms": res[(len(res) * 9) // 10], "before_mape": before,
            "after_mape": after, "buckets": buckets, "n": len(pairs)}


def join(live, sim):
    """Intersect on index; exits when input token counts disagree on more than 5 percent of rows."""
    idx = sorted(set(live) & set(sim))
    if not idx:
        sys.exit("no joinable request indices between live and sim")
    bad = sum(1 for i in idx if live[i][1] != sim[i][1])
    if bad > 0.05 * len(idx):
        sys.exit(f"{bad}/{len(idx)} joined rows disagree on input tokens; "
                 "live and sim are not the same workload")
    return [(live[i][0], sim[i][0], live[i][1]) for i in idx]


def main():
    ap = argparse.ArgumentParser(description="fit the sim's first-token constant")
    ap.add_argument("--live", required=True, help="live bench_dir with requests.jsonl")
    ap.add_argument("--sim", required=True, help="sim per-request CSV")
    ap.add_argument("--sim-time-unit", choices=tuple(UNIT_MS), default="ns",
                    help="unit of the sim CSV times; a wrong unit gives a widely spread residual")
    ap.add_argument("--meta", help="exported meta.yaml to update in place")
    args = ap.parse_args()

    fit = fit_overhead(join(read_live(args.live), read_sim(args.sim, args.sim_time_unit)))
    print(f"FIRSTTOKEN n={fit['n']} overhead_ms={fit['overhead_ms']:.3f} "
          f"p10_ms={fit['p10_ms']:.3f} p90_ms={fit['p90_ms']:.3f}")
    for label, (med, n) in fit["buckets"].items():
        print(f"FIRSTTOKEN bucket={label} n={n} median_ms={med:.3f}")
    print(f"TTFTERR before_mape={fit['before_mape'] * 100:.1f}% "
          f"after_mape={fit['after_mape'] * 100:.1f}%")
    if args.meta:
        import yaml
        meta = yaml.safe_load(Path(args.meta).read_text())
        meta["first_token_overhead_us"] = round(fit["overhead_ms"] * 1e3, 1)
        Path(args.meta).write_text(yaml.safe_dump(meta, sort_keys=False))
        print(f"wrote first_token_overhead_us={meta['first_token_overhead_us']} to {args.meta}")


if __name__ == "__main__":
    main()
