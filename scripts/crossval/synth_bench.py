#!/usr/bin/env python3
"""Synthetic and trace-driven request streams for the cross-validator.

Engine benches clone one sequence into every slot, the worst case for MoE
routing; this builds distinct length-varied streams instead and loads JSONL
traces for replay. Token ids and lengths only, no model or GPU.

Trace lines hold prompt_token_ids, prompt_len, or Mooncake
input_length/output_length/timestamp; arrival_ts and session_id carry
through, unknown keys are ignored.
"""

import argparse
import json
import sys

VOCAB = 100_000  # safe id ceiling for the checkpoints under test


def synth_ids(n, seed):
    """Distinct pseudo-random token stream; different seed gives different routing."""
    a = 1103515245 * (seed + 1) + 12345
    return [((a + i * (2 * seed + 7)) % (VOCAB - 1)) + 1 for i in range(n)]


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
    return reqs


def _lcg(state):
    """Deterministic 31-bit LCG step, the module's only randomness source."""
    return (1103515245 * state + 12345) % (1 << 31)


def swebench_requests(count, seed=0, groups=8, prefix_frac=0.5,
                      min_ctx=4096, max_ctx=24576,
                      min_out=128, max_out=1024, rate=1.0):
    """SWE-bench-shaped stream: long repo-context prompts, short-medium outputs,
    `groups` sessions sharing a prefix (what a prefix cache would hit), bursty
    arrivals at ~`rate` req/s. Deterministic from the seed."""
    group_prefix = {}
    prefix_len = int(min_ctx * prefix_frac)
    for g in range(groups):
        group_prefix[g] = synth_ids(prefix_len, seed * 7_919 + g)
    reqs, state, t = [], (seed * 2_654_435_761 + 1) % (1 << 31), 0.0
    for j in range(count):
        g = j % groups
        state = _lcg(state)
        length = min_ctx + state % max(1, max_ctx - min_ctx + 1)
        state = _lcg(state)
        out = min_out + state % max(1, max_out - min_out + 1)
        # Bursty: turns within a session cluster, bursts a full mean gap apart.
        state = _lcg(state)
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
    return reqs


def _carry_meta(req, d):
    """Carry arrival_ts (seconds; Mooncake writes timestamp in ms) and session_id."""
    if d.get("arrival_ts") is not None:
        req["arrival_ts"] = float(d["arrival_ts"])
    elif d.get("timestamp") is not None:
        req["arrival_ts"] = float(d["timestamp"]) / 1e3
    if d.get("session_id") is not None:
        req["session_id"] = d["session_id"]
    return req


def load_trace(path):
    """Read a JSONL trace into the common request shape; missing ids are synthesized."""
    reqs = []
    for i, line in enumerate(open(path)):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("prompt_token_ids"):
            ids = list(d["prompt_token_ids"])
            req = {"prompt_token_ids": ids, "prompt_len": len(ids)}
            if d.get("output_length"):
                req["output_length"] = int(d["output_length"])
            reqs.append(_carry_meta(req, d))
            continue
        length = d.get("prompt_len") or d.get("input_length")
        if not length:
            raise ValueError(f"{path}:{i+1}: need prompt_token_ids, prompt_len, or input_length")
        req = {"prompt_token_ids": synth_ids(int(length), i), "prompt_len": int(length)}
        if d.get("output_length"):
            req["output_length"] = int(d["output_length"])
        reqs.append(_carry_meta(req, d))
    if not reqs:
        raise ValueError(f"{path}: no requests")
    return reqs


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
    g.add_argument("--rate", type=float, default=1.0, help="swebench: mean arrival rate req/s")
    g.add_argument("--out", required=True)
    i = sub.add_parser("inspect", help="summarize a trace")
    i.add_argument("path")
    a = ap.parse_args()
    if a.cmd == "gen":
        if a.profile == "swebench":
            reqs = swebench_requests(a.n, a.seed, a.groups, a.prefix_frac, rate=a.rate)
        else:
            if not a.ctx:
                ap.error("--ctx is required for --profile uniform")
            reqs = synth_requests(a.n, a.ctx, a.seed, a.jitter)
        with open(a.out, "w") as fh:
            for r in reqs:
                fh.write(json.dumps(r) + "\n")
        lens = [r["prompt_len"] for r in reqs]
        print(f"wrote {len(reqs)} {a.profile} reqs to {a.out}: len min/max {min(lens)}/{max(lens)}")
    else:
        reqs = load_trace(a.path)
        lens = sorted(r["prompt_len"] for r in reqs)
        arrivals = [r["arrival_ts"] for r in reqs if "arrival_ts" in r]
        sessions = {r.get("session_id") for r in reqs if r.get("session_id") is not None}
        extra = ""
        if arrivals:
            extra += f", arrivals {min(arrivals):.2f}..{max(arrivals):.2f}s"
        if sessions:
            extra += f", sessions {len(sessions)}"
        print(f"{a.path}: {len(reqs)} reqs, len p50 {lens[len(lens)//2]} "
              f"min/max {lens[0]}/{lens[-1]}, "
              f"with-ids {sum(1 for r in reqs if r['prompt_token_ids'])}{extra}")


if __name__ == "__main__":
    main()
