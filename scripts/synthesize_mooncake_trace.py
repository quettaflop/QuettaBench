#!/usr/bin/env python3
"""Sample a synthetic Mooncake style trace from a real one.

Pools per record lengths and prefix sharing from the source trace,
then samples a rotated, relabeled copy in the same JSONL schema, so
the same dataset class replays real and synthetic traces identically.
Writes the pooled stats next to the output for auditing.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.workloads.mooncake import parse_mooncake_trace


def build_record_pools(records):
    """Extract the per record distributions and sharing structure.

    Sharing is modeled as leading prefix reuse only. Interior or suffix
    block reuse in the source trace is not reproduced.
    """
    seen: set[int] = set()
    share_flags = []
    reuse_depths = []
    novel_counts = []
    record_pool = []
    hash_pool = []
    for r in records:
        reused = 0
        for h in r.hash_ids:
            if h in seen:
                reused += 1
            else:
                break
        share_flags.append(1 if reused > 0 else 0)
        if reused > 0:
            reuse_depths.append(reused)
        novel_counts.append(len(r.hash_ids) - reused)
        record_pool.append([r.input_length, r.output_length,
                            len(r.hash_ids), reused])
        hash_pool.append(list(r.hash_ids))
        seen.update(r.hash_ids)

    return {
        "num_requests": len(records),
        "isl_pool": [r.input_length for r in records],
        "osl_pool": [r.output_length for r in records],
        "block_count_pool": [len(r.hash_ids) for r in records],
        "p_share": sum(share_flags) / len(records),
        "reuse_depth_pool": reuse_depths or [0],
        "novel_count_pool": novel_counts,
        "record_pool": record_pool,
        "hash_pool": hash_pool,
    }


def sample_rotated_trace(stats, num_requests, seed):
    """Circular block bootstrap over the pooled records.

    Rotate the source order at a seeded cut and assign fresh block
    ids. Sizes and sharing keep their real per record coupling.
    """
    rng = random.Random(seed)
    record_pool = stats["record_pool"]
    hash_pool = stats["hash_pool"]

    n = len(record_pool)
    cut = rng.randrange(1, n) if n > 1 else 0
    rotation = list(range(cut, n)) + list(range(cut))
    order: list[int] = []
    while len(order) < num_requests:
        order.extend(rotation)
    order = order[:num_requests]

    relabel: dict[int, int] = {}

    def fresh_block_id(h):
        if h not in relabel:
            relabel[h] = 1_000_000 + len(relabel)
        return relabel[h]

    rows = []
    for i in order:
        isl, osl, _, _ = record_pool[i]
        rows.append({
            "input_length": isl,
            "output_length": osl,
            "hash_ids": [fresh_block_id(h) for h in hash_pool[i]],
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-requests", type=int, default=0,
                    help="0 means match the source trace size")
    ap.add_argument("--fit-limit", type=int, default=0,
                    help="pool only the first N records; 0 uses the whole trace")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    records = parse_mooncake_trace(args.trace, args.fit_limit)
    stats = build_record_pools(records)
    n = args.num_requests or stats["num_requests"]
    rows = sample_rotated_trace(stats, n, args.seed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    audit = {k: v for k, v in stats.items()
             if not k.endswith("_pool")}
    audit["synthetic_requests"] = n
    audit["seed"] = args.seed
    audit["fit_limit"] = args.fit_limit
    audit_path = args.out + ".fit.json"
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit, indent=2))
    print(f"synthetic trace: {args.out}")
    print(f"fit audit: {audit_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
