#!/usr/bin/env python3
"""Synthetic and trace-driven request streams for the cross-validator.

Builds distinct length-varied streams (engine benches clone one sequence into
every slot, the MoE-routing worst case) and loads JSONL traces for replay.
Profiles: uniform and swebench; the multi-turn agentic set lives in
analysis/synthetic/agentic.py.
Token ids and lengths only, no model or GPU.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from quettabench.workloads.base import synth_ids, workload_hash
from quettabench.workloads.live import RealTrace


def synth_requests(count, ctx, seed=0, length_jitter=0.0):
    """`count` requests of ~`ctx` tokens. length_jitter in [0,1) varies each length
    deterministically; 0.0 keeps them uniform for a clean grid cell."""
    reqs = []
    for j in range(count):
        s = seed * 1_000_003 + j
        if length_jitter > 0.0:
            span = int(ctx * length_jitter)
            length = max(1, ctx - span + (s * 2654435761) % (2 * span + 1))
        else:
            length = ctx
        reqs.append({"prompt_token_ids": synth_ids(length, s), "prompt_len": length})
    return stamp_synthetic(reqs, seed)


def lcg(state):
    """Deterministic 31-bit LCG step, the module's only randomness source."""
    return (1103515245 * state + 12345) % (1 << 31)


def stamp_synthetic(reqs, seed):
    """Stamp seed, order and the SYNTHETIC label on generated requests; seed and
    order are hash inputs, so a stream is self-identifying and reproducible."""
    for i, r in enumerate(reqs):
        r["seed"] = seed
        r["order"] = i
        r["source"] = "SYNTHETIC"
    return reqs


def swebench_requests(count, seed=0, groups=8, prefix_frac=0.5,
                      min_ctx=4096, max_ctx=24576,
                      min_out=128, max_out=1024, rate=1.0):
    """SWE-bench-shaped stream: long repo-context prompts, `groups` sessions
    sharing a prefix, bursty arrivals at ~`rate` req/s. Deterministic from the seed."""
    group_prefix = {}
    prefix_len = int(min_ctx * prefix_frac)
    for g in range(groups):
        group_prefix[g] = synth_ids(prefix_len, seed * 7_919 + g)
    reqs, state, t = [], (seed * 2_654_435_761 + 1) % (1 << 31), 0.0
    for j in range(count):
        g = j % groups
        state = lcg(state)
        length = min_ctx + state % max(1, max_ctx - min_ctx + 1)
        state = lcg(state)
        out = min_out + state % max(1, max_out - min_out + 1)
        # Bursty: turns within a session cluster, bursts a full mean gap apart.
        state = lcg(state)
        gap = (state % 1000) / 1000.0 / max(rate, 1e-6)
        t += gap / 8.0 if j % groups else gap
        suffix = synth_ids(length - prefix_len, seed * 104_729 + j)
        reqs.append({
            "prompt_token_ids": group_prefix[g] + suffix,
            "prompt_len": length,
            "output_length": out,
            "arrival_ts": round(t, 6),
            "session_id": f"g{g}",
        })
    return stamp_synthetic(reqs, seed)


def load_trace(path):
    """Read a JSONL trace into the common request dicts (live.RealTrace.from_jsonl)."""
    return [r.to_dict() for r in RealTrace.from_jsonl(path).requests()]


def take(reqs, count, ctx):
    """Pick `count` requests for a (ctx, bs) cell, trimming/padding ids to ctx and
    cycling the trace if short. Returns (ids, cycled)."""
    usable = [r for r in reqs if r["prompt_len"] >= ctx] or reqs
    out, cycled = [], False
    for j in range(count):
        if j >= len(usable):
            cycled = True
        r = usable[j % len(usable)]
        ids = r["prompt_token_ids"]
        ids = ids[:ctx] if len(ids) >= ctx else ids + synth_ids(ctx - len(ids), j)
        out.append(list(ids))
    return out, cycled


def main():
    ap = argparse.ArgumentParser(description="synthetic / trace request streams")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen", help="write a synthetic trace")
    g.add_argument("--profile", choices=("uniform", "swebench"), default="uniform")
    g.add_argument("--n", type=int, required=True)
    g.add_argument("--ctx", type=int, help="uniform profile: prompt length")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--jitter", type=float, default=0.0)
    g.add_argument("--groups", type=int, default=8, help="swebench: shared-prefix sessions")
    g.add_argument("--prefix-frac", type=float, default=0.5, help="swebench: shared fraction of min-ctx")
    g.add_argument("--rate", type=float, default=None, help="swebench: mean arrival rate req/s")
    g.add_argument("--out", required=True)
    i = sub.add_parser("inspect", help="summarize a trace")
    i.add_argument("path")
    h = sub.add_parser("hash", help="print the workload-identity hash of a trace")
    h.add_argument("path")
    a = ap.parse_args()
    if a.cmd == "hash":
        print(workload_hash(load_trace(a.path)))
        return
    if a.cmd == "gen":
        if a.profile == "uniform":
            if not a.ctx:
                ap.error("--ctx is required for --profile uniform")
            reqs = synth_requests(a.n, a.ctx, a.seed, a.jitter)
        else:
            reqs = swebench_requests(a.n, a.seed, a.groups, a.prefix_frac, rate=a.rate or 1.0)
        with open(a.out, "w") as fh:
            for r in reqs:
                fh.write(json.dumps(r) + "\n")
        lens = [r["prompt_len"] for r in reqs]
        print(f"wrote {len(reqs)} SYNTHETIC {a.profile} reqs (seed {a.seed}) to {a.out}: "
              f"len min/max {min(lens)}/{max(lens)} hash {workload_hash(reqs)}")
    else:
        reqs = load_trace(a.path)
        lens = sorted(r["prompt_len"] for r in reqs)
        arrivals = [r["arrival_ts"] for r in reqs if "arrival_ts" in r]
        sessions = {r.get("session_id") for r in reqs if r.get("session_id") is not None}
        turns = [r["turn"] for r in reqs if "turn" in r]
        extra = ""
        if arrivals:
            extra += f", arrivals {min(arrivals):.2f}..{max(arrivals):.2f}s"
        if sessions:
            extra += f", sessions {len(sessions)}"
        if turns:
            extra += f", max_turn {max(turns)}"
        srcs = ",".join(sorted({str(r.get("source", "?")) for r in reqs}))
        extra += f", source {srcs}, hash {workload_hash(reqs)}"
        print(f"{a.path}: {len(reqs)} reqs, len p50 {lens[len(lens)//2]} "
              f"min/max {lens[0]}/{lens[-1]}, "
              f"with-ids {sum(1 for r in reqs if r['prompt_token_ids'])}{extra}")


if __name__ == "__main__":
    main()
