#!/usr/bin/env python3
"""Step 1: measure ACTUAL prefix reuse, instead of inferring it from size deltas.

`build_trace_distributions.py` derives cache_hit_rate as

    new_prefill   = max(1, total_context - previous_context)
    cache_hit     = 1 - new_prefill / total_context

i.e. it assumes that if turn N+1 is only slightly larger than turn N, the
overlap is a shared prefix. That holds for append-style (growing-history)
agents. It does NOT hold for agents that REPLACE part of the context each turn:
two same-sized prompts with completely different content score as a ~100% cache
hit.

This script ignores sizes and measures the literal shared prefix between
consecutive turns' prompts, which is what a prefix cache can actually reuse.

Prompts are approximated by concatenating message contents in order. That is not
the exact chat template, but the question here is 0.5% vs 99%, which is far
outside any template overhead.

Usage:
    python scripts/measure_true_prefix_reuse.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
sys.path.insert(0, str(HERE))

DATASETS = ["swebench", "terminalbench", "osworld"]


def prompt_text(messages) -> str:
    parts = []
    for m in messages or []:
        parts.append(str(m.get("role", "")))
        parts.append(str(m.get("content", "")))
    return "\n".join(parts)


def measure(name: str, max_sessions: int = 40) -> None:
    path = DATA / f"{name}_trajectories.jsonl"
    if not path.exists():
        print(f"  [skip] {name}: missing {path.name}")
        return

    measured: list[float] = []   # true shared-prefix fraction
    inferred: list[float] = []   # what the builder's delta heuristic would say
    n_sessions = 0

    with path.open() as f:
        for line in f:
            if n_sessions >= max_sessions:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            turns = row.get("turns") or []
            if len(turns) < 2:
                continue
            n_sessions += 1
            prev_txt = None
            prev_len = 0
            for t in turns:
                txt = prompt_text(t.get("messages"))
                if not txt:
                    continue
                if prev_txt is not None:
                    shared = len(os.path.commonprefix([prev_txt, txt]))
                    measured.append(shared / len(txt) if txt else 0.0)
                    # builder's heuristic, in the same char units
                    new_prefill = max(1, len(txt) - prev_len)
                    inferred.append(max(0.0, 1.0 - new_prefill / len(txt)))
                prev_txt = txt
                prev_len = len(txt)

    if not measured:
        print(f"  [skip] {name}: no comparable turns")
        return

    def q(xs, p):
        xs = sorted(xs)
        return xs[min(int(p * (len(xs) - 1)), len(xs) - 1)]

    print(f"  {name:<14} sessions={n_sessions:>3} turnpairs={len(measured):>5}")
    print(f"    {'':>16}{'p10':>8}{'p50':>8}{'p90':>8}{'mean':>8}")
    print(f"    {'TRUE prefix':>16}{q(measured,.1):>8.3f}{q(measured,.5):>8.3f}"
          f"{q(measured,.9):>8.3f}{statistics.mean(measured):>8.3f}")
    print(f"    {'delta heuristic':>16}{q(inferred,.1):>8.3f}{q(inferred,.5):>8.3f}"
          f"{q(inferred,.9):>8.3f}{statistics.mean(inferred):>8.3f}")
    gap = statistics.mean(inferred) - statistics.mean(measured)
    verdict = "OK" if abs(gap) < 0.10 else "*** HEURISTIC WRONG ***"
    print(f"    overstatement by heuristic: {gap:+.3f}   {verdict}\n")


def main() -> int:
    print("\n=== True shared-prefix reuse vs the builder's delta heuristic ===")
    print("(fraction of each turn's prompt that literally repeats the previous turn's)\n")
    for d in DATASETS:
        measure(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
