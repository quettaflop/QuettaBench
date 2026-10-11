"""Synthetic and trace request streams for the cross-validator, as JSONL.

gen writes a uniform (grid synth mode) or a SWE-bench-shaped stream, inspect
summarizes a trace, hash prints its workload hash. Token ids and lengths
only, no model or GPU. Run from the repo root (or with quettabench
installed): python3 -m analysis.synthetic.synth_bench gen|inspect|hash ...
"""

import argparse
import json

from quettabench.workloads.base import synth_ids, workload_hash
from quettabench.workloads.grid import stamp_synthetic, synth_requests, take
from quettabench.workloads.live import RealTrace

__all__ = ["lcg", "load_trace", "swebench_requests", "synth_requests", "take", "workload_hash"]


def lcg(state):
    """Deterministic 31-bit LCG step, the module's only randomness source."""
    return (1103515245 * state + 12345) % (1 << 31)


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
