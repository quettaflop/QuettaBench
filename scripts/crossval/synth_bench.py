#!/usr/bin/env python3
"""Synthetic request-stream generator for the cross-validator.

Why this exists: the engine batch benches CLONE one sequence into every slot,
so all tokens route to the same experts. On an expert-parallel MoE that is the
worst case (a few ranks do all the work, the rest idle) and it is not how real
traffic behaves. This module builds DISTINCT, length-varied request streams so
routing spreads like production, and it reads a request trace (JSONL) so a real
capture -- e.g. a Mooncake trace -- can be replayed through the same path.

It is transport-only: it emits token-id lists (and lengths). vmin_fit.py turns
them into vLLM prompts; an engine-side reader can consume the same JSONL. No
model or GPU is touched here.

Trace schema (JSONL, one request per line), any of:
    {"prompt_token_ids": [int, ...]}          # exact tokens (replayed verbatim)
    {"prompt_len": int}                        # length only (ids synthesized)
    {"input_length": int, "output_length": int}  # Mooncake-style; output_length
                                                   # advises decode steps
Unknown keys are ignored, so richer traces (timestamps, session ids) pass
through untouched for a future arrival-time scheduler.
"""

import argparse
import json
import sys

VOCAB = 100_000  # safe id ceiling for the checkpoints under test


def synth_ids(n, seed):
    """A distinct pseudo-random token stream. Different seed => different tokens
    => different expert routing. Coprime stride keeps it non-repeating within a
    prompt so the stream is not a single hot token."""
    a = 1103515245 * (seed + 1) + 12345
    return [((a + i * (2 * seed + 7)) % (VOCAB - 1)) + 1 for i in range(n)]


def synth_requests(count, ctx, seed=0, length_jitter=0.0):
    """`count` distinct requests of ~`ctx` tokens. length_jitter in [0,1) varies
    each length by up to +/-jitter*ctx (deterministic from the seed) so the
    batch mixes lengths like real traffic; 0.0 keeps them uniform for a clean
    (ctx, bs) grid cell."""
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


def load_trace(path):
    """Read a JSONL request trace into the common request shape. Lengths without
    ids are synthesized (routing-diverse); ids are replayed verbatim."""
    reqs = []
    for i, line in enumerate(open(path)):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("prompt_token_ids"):
            ids = list(d["prompt_token_ids"])
            reqs.append({"prompt_token_ids": ids, "prompt_len": len(ids)})
            continue
        length = d.get("prompt_len") or d.get("input_length")
        if not length:
            raise ValueError(f"{path}:{i+1}: need prompt_token_ids, prompt_len, or input_length")
        req = {"prompt_token_ids": synth_ids(int(length), i), "prompt_len": int(length)}
        if d.get("output_length"):
            req["output_length"] = int(d["output_length"])
        reqs.append(req)
    if not reqs:
        raise ValueError(f"{path}: no requests")
    return reqs


def take(reqs, count, ctx):
    """Pick `count` requests for a (ctx, bs) cell: prefer trace entries long
    enough for the context, cycle if the trace is short, and trim/pad ids to
    exactly ctx so the grid stays rectangular. Reports if it had to cycle."""
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
    g.add_argument("--n", type=int, required=True)
    g.add_argument("--ctx", type=int, required=True)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--jitter", type=float, default=0.0)
    g.add_argument("--out", required=True)
    i = sub.add_parser("inspect", help="summarize a trace")
    i.add_argument("path")
    a = ap.parse_args()
    if a.cmd == "gen":
        reqs = synth_requests(a.n, a.ctx, a.seed, a.jitter)
        with open(a.out, "w") as fh:
            for r in reqs:
                fh.write(json.dumps(r) + "\n")
        lens = [r["prompt_len"] for r in reqs]
        print(f"wrote {len(reqs)} reqs to {a.out}: len min/max {min(lens)}/{max(lens)}")
    else:
        reqs = load_trace(a.path)
        lens = sorted(r["prompt_len"] for r in reqs)
        print(f"{a.path}: {len(reqs)} reqs, len p50 {lens[len(lens)//2]} "
              f"min/max {lens[0]}/{lens[-1]}, "
              f"with-ids {sum(1 for r in reqs if r['prompt_token_ids'])}")


if __name__ == "__main__":
    main()
