#!/usr/bin/env python3
"""Audit a live replay result against its source trace.

Per request, server reported input_tokens and output_tokens must equal
the trace fields. Mismatches, missing usage, or dropped requests fail.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.workloads.mooncake import parse_mooncake_trace


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result", required=True)
    ap.add_argument("--trace", required=True)
    args = ap.parse_args()

    with open(args.result) as f:
        rows = json.load(f)["per_request"]
    records = parse_mooncake_trace(args.trace)

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
            if r["input_tokens"] != rec.input_length:
                mismatches.append(
                    f"req {idx}: input {r['input_tokens']} != trace {rec.input_length}")
            if r["output_tokens"] != rec.output_length:
                mismatches.append(
                    f"req {idx}: output {r['output_tokens']} != trace {rec.output_length}")
            ok += 1

    print(f"requests: {len(rows)}  audited: {ok}  failed: {fail}  "
          f"usage_missing: {usage_missing}")
    ttft = sorted(r["ttft_ms"] for r in rows if r.get("ttft_ms"))
    tpot = sorted(r["tpot_ms"] for r in rows if r.get("tpot_ms"))
    if ttft and tpot:
        mid = lambda v: v[len(v) // 2]
        print(f"ttft ms p50 {mid(ttft):.1f} max {ttft[-1]:.1f}  "
              f"tpot ms p50 {mid(tpot):.2f}")
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
