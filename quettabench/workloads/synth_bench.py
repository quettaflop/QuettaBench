#!/usr/bin/env python3
"""Synthetic and trace-driven request streams for the cross-validator.

Builds distinct length-varied streams (engine benches clone one sequence into
every slot, the MoE-routing worst case) and loads JSONL traces for replay.
Profiles: uniform, swebench, and the multi-turn agentic set in AGENTIC.
Token ids and lengths only, no model or GPU.
"""

import argparse
import hashlib
import json
import math
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
    return _stamp(reqs, seed)


def _lcg(state):
    """Deterministic 31-bit LCG step, the module's only randomness source."""
    return (1103515245 * state + 12345) % (1 << 31)


def _stamp(reqs, seed):
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
    return _stamp(reqs, seed)


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
        state = _lcg(state); turns = span(spec["turns"][0], spec["turns"][1], state)
        ctx = synth_ids(int(spec["prompt"][0] * spec["prefix_frac"]), seed * 7_919 + sess)
        t = t0
        for turn in range(turns):
            if len(reqs) >= count:
                break
            state = _lcg(state); inp = span(spec["prompt"][0], spec["prompt"][1], state)
            state = _lcg(state); out = span(spec["output"][0], spec["output"][1], state)
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
            state = _lcg(state); u = (state % 10000) / 10000.0
            glo, ghi = spec["gap"]; t += glo + u * (ghi - glo)
        sess += 1
        # session arrival pattern, distinct per profile
        if spec["burst"] == "poisson":
            state = _lcg(state); u = (state % 9999 + 1) / 10000.0
            t0 += -math.log(u) / max(r, 1e-6)
        elif spec["burst"] == "bursty":
            t0 += (1.0 / max(r, 1e-6)) if sess % 4 == 0 else (0.1 / max(r, 1e-6))
        else:
            t0 += 1.0 / max(r, 1e-6)
    return _stamp(reqs[:count], seed)


def _carry_meta(req, d):
    """Carry scheduling and provenance metadata; Mooncake timestamp is ms."""
    if d.get("arrival_ts") is not None:
        req["arrival_ts"] = float(d["arrival_ts"])
    elif d.get("timestamp") is not None:
        req["arrival_ts"] = float(d["timestamp"]) / 1e3
    for k in ("session_id", "turn", "image_tokens", "seed", "source"):
        if d.get(k) is not None:
            req[k] = d[k]
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
            # Reference tokens for acceptance scoring (SPECSUM); must survive load
            # or trace_serve has nothing to compare the engine's output against.
            if d.get("output_token_ids"):
                req["output_token_ids"] = list(d["output_token_ids"])
            req["order"] = i  # submission order is the file line index, authoritative
            reqs.append(_carry_meta(req, d))
            continue
        length = d.get("prompt_len") or d.get("input_length")
        if not length:
            raise ValueError(f"{path}:{i+1}: need prompt_token_ids, prompt_len, or input_length")
        req = {"prompt_token_ids": synth_ids(int(length), i), "prompt_len": int(length)}
        if d.get("output_length"):
            req["output_length"] = int(d["output_length"])
        req["order"] = i
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


def _req_digest(r):
    """Identity of one request over deterministic inputs only (seed, order, arrival,
    prompt content, requested output), never generated tokens or timings, so two
    engines doing identical work hash the same."""
    ids = r.get("prompt_token_ids") or []
    parts = [
        str(r.get("seed", "")),
        str(r.get("order", "")),
        f"{float(r.get('arrival_ts', 0.0)):.6f}",
        str(r.get("prompt_len", len(ids))),
        ",".join(map(str, ids)),
        str(r.get("output_length", "")),
        str(r.get("session_id", "")),
        str(r.get("turn", "")),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def workload_hash(reqs):
    """Order-independent identity of the whole workload: same request set gives the
    same hash regardless of completion order; a dropped or altered request changes
    it. This is the check that every system saw identical work."""
    digs = sorted(_req_digest(r) for r in reqs)
    h = hashlib.sha256()
    h.update(str(len(digs)).encode())
    for d in digs:
        h.update(b"\n")
        h.update(d.encode())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description="synthetic / trace request streams")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen", help="write a synthetic trace")
    g.add_argument("--profile", choices=("uniform", "swebench", *AGENTIC), default="uniform")
    g.add_argument("--n", type=int, required=True)
    g.add_argument("--ctx", type=int, help="uniform profile: prompt length")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--jitter", type=float, default=0.0)
    g.add_argument("--groups", type=int, default=8, help="agentic/swebench: shared-prefix sessions")
    g.add_argument("--prefix-frac", type=float, default=0.5, help="swebench: shared fraction of min-ctx")
    g.add_argument("--rate", type=float, default=None, help="mean arrival rate req/s (agentic: spec default)")
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
        elif a.profile == "swebench":
            reqs = swebench_requests(a.n, a.seed, a.groups, a.prefix_frac, rate=a.rate or 1.0)
        else:
            reqs = agentic_requests(a.profile, a.n, a.seed, a.groups, rate=a.rate)
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
