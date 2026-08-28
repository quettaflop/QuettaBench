#!/usr/bin/env python3
"""Score two replay result files under one metric definition.

Reads per request records from QuettaBench result JSON (schema with a
per_request list) and compares TTFT, TPOT and E2EL between the two
sides: mean, median, p90, p99, and the relative difference of each.
Applies the acceptance gates: TPOT and E2EL means within 5 percent.
TTFT is reported but not gated, it is queueing sensitive. If traces
are supplied, also compares the ISL and OSL distributions with a
Kolmogorov Smirnov statistic (KS below).

Exit code 0 when every gate passes, 1 otherwise.

Usage:
    python scripts/score_replay_records.py --a results/qb.json --b results/ref.json \
        [--trace-a real.jsonl --trace-b synth.jsonl] [--gate-pct 5.0]
"""

import argparse
import json
import sys


def load_records(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "per_request" in data:
        rows = [r for r in data["per_request"] if r.get("success")]
        return {
            "ttft": [r["ttft_ms"] for r in rows if r.get("ttft_ms") is not None],
            "tpot": [r["tpot_ms"] for r in rows if r.get("tpot_ms") is not None],
            "e2el": [r["e2el_ms"] for r in rows if r.get("e2el_ms") is not None],
        }
    raise ValueError(
        f"{path}: unrecognized result schema, expected a per_request list"
    )


def stat(values, q):
    s = sorted(values)
    if not s:
        return float("nan")
    if q == "mean":
        return sum(s) / len(s)
    idx = min(len(s) - 1, int(len(s) * {"p50": 0.5, "p90": 0.9, "p99": 0.99}[q]))
    return s[idx]


def ks_statistic(a, b):
    """Kolmogorov Smirnov statistic: the maximum distance between the
    two empirical cumulative distribution functions."""
    if not a or not b:
        return 1.0
    xs = sorted(set(a) | set(b))
    sa, sb = sorted(a), sorted(b)
    import bisect
    d = 0.0
    for x in xs:
        fa = bisect.bisect_right(sa, x) / len(sa)
        fb = bisect.bisect_right(sb, x) / len(sb)
        d = max(d, abs(fa - fb))
    return d


def lengths(path):
    isl, osl = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                isl.append(row["input_length"])
                osl.append(row["output_length"])
    return isl, osl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, help="result JSON, side A")
    ap.add_argument("--b", required=True, help="result JSON, side B")
    ap.add_argument("--trace-a", help="trace JSONL for ISL OSL comparison")
    ap.add_argument("--trace-b", help="trace JSONL for ISL OSL comparison")
    ap.add_argument("--gate-pct", type=float, default=5.0)
    args = ap.parse_args()

    a = load_records(args.a)
    b = load_records(args.b)

    failed = []
    print(f"{'metric':8} {'stat':5} {'A':>10} {'B':>10} {'diff%':>8}")
    for metric in ("ttft", "tpot", "e2el"):
        for q in ("mean", "p50", "p90", "p99"):
            va, vb = stat(a[metric], q), stat(b[metric], q)
            diff = (vb - va) / va * 100 if va else float("nan")
            print(f"{metric:8} {q:5} {va:10.2f} {vb:10.2f} {diff:+8.2f}")
            gated = metric in ("tpot", "e2el") and q == "mean"
            # nan (no successful requests) must fail the gate.
            if gated and not abs(diff) <= args.gate_pct:
                failed.append(f"{metric} mean diff {diff:+.2f}% exceeds "
                              f"{args.gate_pct}% or is undefined")

    if args.trace_a and args.trace_b:
        isl_a, osl_a = lengths(args.trace_a)
        isl_b, osl_b = lengths(args.trace_b)
        for name, xa, xb in (("ISL", isl_a, isl_b), ("OSL", osl_a, osl_b)):
            ks = ks_statistic(xa, xb)
            print(f"{name} KS statistic: {ks:.4f}")
            if ks > 0.1:
                failed.append(f"{name} KS {ks:.4f} exceeds 0.10")

    if failed:
        print("GATES FAILED:")
        for f in failed:
            print(f"  {f}")
        return 1
    print("GATES PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
