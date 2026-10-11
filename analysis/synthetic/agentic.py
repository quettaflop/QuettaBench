#!/usr/bin/env python3
"""Synthetic multi-turn agentic traces (terminal_bench, deep_research, ...).

Calibratable shapes, not measured captures. Writes the replay JSONL schema
that quettabench/workloads/live.py reads.
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from analysis.synthetic.synth_bench import lcg
from quettabench.workloads.base import synth_ids, workload_hash
from quettabench.workloads.grid import stamp_synthetic


# Agentic workload shapes: prefix_frac = cache-hit fraction, growth = context per
# turn, gap = inter-turn think/tool seconds, burst = session-arrival pattern.
# Calibratable defaults, not measured captures; real traces override via load_trace.
AGENTIC = {
    "terminal_bench": {"turns": (5, 30),  "prompt": (512, 4096),   "output": (32, 256),    "prefix_frac": 0.4, "growth": 256,  "image": 0,    "rate": 1.0,  "gap": (0.2, 1.5),  "burst": "steady"},
    "deep_research":  {"turns": (10, 40), "prompt": (2048, 8192),  "output": (512, 2048),  "prefix_frac": 0.3, "growth": 2048, "image": 0,    "rate": 0.5,  "gap": (5.0, 30.0), "burst": "poisson"},
    "deep_search":    {"turns": (8, 25),  "prompt": (1024, 4096),  "output": (128, 512),   "prefix_frac": 0.3, "growth": 1024, "image": 0,    "rate": 0.5,  "gap": (2.0, 8.0),  "burst": "poisson"},
    "osworld":        {"turns": (10, 50), "prompt": (512, 2048),   "output": (16, 128),    "prefix_frac": 0.2, "growth": 256,  "image": 1536, "rate": 0.5,  "gap": (0.3, 2.0),  "burst": "steady"},
    "rl_cf":          {"turns": (1, 3),   "prompt": (1024, 4096),  "output": (4096, 16384), "prefix_frac": 0.1, "growth": 0,   "image": 0,    "rate": 0.25, "gap": (2.0, 10.0), "burst": "bursty"},
}


def agentic_requests(profile, count, seed=0, groups=8, rate=None):
    """Multi-turn agentic sessions for `profile`: a carried prefix, context growing
    per turn, one output per turn; turns carry session_id, turn, arrival_ts and
    output_length. Deterministic from the seed."""
    spec = AGENTIC[profile]
    r = spec["rate"] if rate is None else rate
    def span(lo, hi, s):
        return lo + s % max(1, hi - lo + 1)
    reqs, state, sess, t0 = [], (seed * 2_654_435_761 + 1) % (1 << 31), 0, 0.0
    while len(reqs) < count:
        g = sess % groups
        state = lcg(state); turns = span(spec["turns"][0], spec["turns"][1], state)
        ctx = synth_ids(int(spec["prompt"][0] * spec["prefix_frac"]), seed * 7_919 + sess)
        t = t0
        for turn in range(turns):
            if len(reqs) >= count:
                break
            state = lcg(state); inp = span(spec["prompt"][0], spec["prompt"][1], state)
            state = lcg(state); out = span(spec["output"][0], spec["output"][1], state)
            prompt = ctx + synth_ids(inp + spec["image"], seed * 104_729 + len(reqs))
            req = {
                "prompt_token_ids": prompt,
                "prompt_len": len(prompt),
                "output_length": out,
                "arrival_ts": round(t, 6),
                "session_id": f"{profile}-{g}",
                "turn": turn,
            }
            if spec["image"]:
                req["image_tokens"] = spec["image"]
            reqs.append(req)
            # Grow the carried context for the next turn (prior io + retrieved docs).
            ctx = prompt + synth_ids(spec["growth"], seed * 13 + len(reqs)) if spec["growth"] else prompt
            # inter-turn gap: the agent's think and tool time between turns
            state = lcg(state); u = (state % 10000) / 10000.0
            glo, ghi = spec["gap"]; t += glo + u * (ghi - glo)
        sess += 1
        # session arrival pattern, distinct per profile
        if spec["burst"] == "poisson":
            state = lcg(state); u = (state % 9999 + 1) / 10000.0
            t0 += -math.log(u) / max(r, 1e-6)
        elif spec["burst"] == "bursty":
            t0 += (1.0 / max(r, 1e-6)) if sess % 4 == 0 else (0.1 / max(r, 1e-6))
        else:
            t0 += 1.0 / max(r, 1e-6)
    return stamp_synthetic(reqs[:count], seed)


def main():
    ap = argparse.ArgumentParser(description="synthetic agentic multi-turn traces")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen", help="write a synthetic agentic trace")
    g.add_argument("--profile", choices=tuple(AGENTIC), required=True)
    g.add_argument("--n", type=int, required=True)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--groups", type=int, default=8, help="shared-prefix sessions")
    g.add_argument("--rate", type=float, default=None, help="mean session rate (default: the profile's)")
    g.add_argument("--out", required=True)
    a = ap.parse_args()
    reqs = agentic_requests(a.profile, a.n, a.seed, a.groups, rate=a.rate)
    with open(a.out, "w") as fh:
        for r in reqs:
            fh.write(json.dumps(r) + "\n")
    lens = [r["prompt_len"] for r in reqs]
    print(f"wrote {len(reqs)} SYNTHETIC {a.profile} reqs (seed {a.seed}) to {a.out}: "
          f"len min/max {min(lens)}/{max(lens)} hash {workload_hash(reqs)}")


if __name__ == "__main__":
    main()
