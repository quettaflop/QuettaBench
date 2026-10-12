#!/usr/bin/env python3
"""Audit a live replay result against its source trace.

Server-reported input_tokens and output_tokens must equal the trace fields per request; mismatches, missing usage or dropped requests fail.
"""

import argparse
import json
import sys


def trace_lengths(path):
    """(input_length, output_length) per non-blank Mooncake line, in file order."""
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [(int(r["input_length"]), int(r["output_length"])) for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result", required=True)
    ap.add_argument("--trace", required=True)
    args = ap.parse_args()

    with open(args.result) as f:
        rows = json.load(f)["per_request"]
    records = trace_lengths(args.trace)

    ok = fail = usage_missing = 0
    mismatches = []
    for r in rows:
        idx = r.get("request_index")
        rec = records[idx] if isinstance(idx, int) and 0 <= idx < len(records) else None
        if not r.get("success"):
            fail += 1
        elif rec is None:
            mismatches.append(f"row without matching trace record: {idx}")
        elif not r.get("usage_reported"):
            usage_missing += 1
        else:
            if r["input_tokens"] != rec[0]:
                mismatches.append(f"req {idx}: input {r['input_tokens']} != trace {rec[0]}")
            if r["output_tokens"] != rec[1]:
                mismatches.append(f"req {idx}: output {r['output_tokens']} != trace {rec[1]}")
            ok += 1

    print(f"requests: {len(rows)}  audited: {ok}  failed: {fail}  "
          f"usage_missing: {usage_missing}")
    ttft = sorted(r["ttft_ms"] for r in rows if r.get("ttft_ms"))
    tpot = sorted(r["tpot_ms"] for r in rows if r.get("tpot_ms"))
    if ttft and tpot:
        print(f"ttft ms p50 {ttft[len(ttft) // 2]:.1f} max {ttft[-1]:.1f}  "
              f"tpot ms p50 {tpot[len(tpot) // 2]:.2f}")
    if mismatches:
        print(f"EXACTNESS FAILED ({len(mismatches)} mismatches):")
        for m in mismatches[:10]:
            print(f"  {m}")
        return 1
    if fail > len(rows) * 0.01:
        print("EXACTNESS FAILED: more than 1 percent of requests failed")
        return 1
    if ok == 0:
        print("EXACTNESS FAILED: zero requests audited, nothing was proven")
        return 1
    if usage_missing > 0:
        print(f"EXACTNESS FAILED: {usage_missing} requests lacked server usage")
        return 1
    idxs = sorted(r.get("request_index") for r in rows
                  if r.get("request_index") is not None)
    if idxs != list(range(len(rows))):
        print("EXACTNESS FAILED: request_index coverage has gaps or duplicates")
        return 1
    print("EXACTNESS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
