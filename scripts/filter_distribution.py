"""Filter a trace distribution JSON to match legacy ISL filter semantics.

Legacy profiles filter entire sessions by max-turn ISL: if any turn exceeds
`isl_limit`, the session is excluded. This applies the same filter to a
distribution JSON so distributional + legacy draw from the same population.

Usage:
  python scripts/filter_distribution.py \\
      data/distributions/swebench_multiturn.json \\
      --isl-limit 32768 \\
      --out data/distributions/swebench_multiturn_filtered.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def filter_distribution(input_path: Path, output_path: Path, isl_limit: int) -> None:
    with open(input_path) as f:
        data = json.load(f)

    if data.get("schema_version") != 1:
        raise ValueError(f"Unsupported schema version: {data.get('schema_version')}")

    turns = data.get("samples", {}).get("turns", [])
    turn_counts = data.get("samples", {}).get("turn_count", [])

    filtered_turn_counts = []
    filtered_turns = []
    kept = 0
    total = len(turn_counts)
    turn_idx = 0

    for tc in turn_counts:
        session_turns = []
        ok = True
        for _ in range(int(tc) if tc is not None else 0):
            if turn_idx >= len(turns):
                break
            turn = turns[turn_idx]
            session_turns.append(turn)
            if turn.get("total_context_tokens", 0) > isl_limit:
                ok = False
            turn_idx += 1

        if ok:
            filtered_turn_counts.append(tc)
            filtered_turns.extend(session_turns)
            kept += 1

    data["samples"]["turn_count"] = filtered_turn_counts
    data["samples"]["turns"] = filtered_turns
    data.setdefault("_filter", {})
    data["_filter"]["isl_limit"] = isl_limit
    data["_filter"]["original_sessions"] = total
    data["_filter"]["kept_sessions"] = kept
    data["_filter"]["discarded"] = total - kept

    # Invalidate aggregates based on unfiltered data
    for k in ("summary", "by_turn_index", "histograms", "diagnostics"):
        data.pop(k, None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    print(f"Filtered {input_path.name} at ISL <= {isl_limit}:")
    print(f"  {total} -> {kept} sessions kept ({total - kept} discarded)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Filter trace distribution by ISL limit"
    )
    ap.add_argument("input", type=Path)
    ap.add_argument("--isl-limit", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    filter_distribution(args.input, args.out, args.isl_limit)


if __name__ == "__main__":
    main()
