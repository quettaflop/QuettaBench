#!/usr/bin/env python3
"""Measure the literal shared prefix between consecutive turn prompts of Claw-Eval traces, with and without reasoning kept in the prompt.

Prompts are rebuilt from the raw trace messages by concatenating parts in order.
Usage: python analysis/prefix_reuse/measure_claweval_prefix_reuse.py --traces <run-dir>
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path


def render_block(b) -> str:
    if not isinstance(b, dict):
        return str(b)
    t = b.get("type")
    if t == "text":
        return str(b.get("text", ""))
    if t == "tool_use":
        return f"{b.get('name', '')}({json.dumps(b.get('input'), sort_keys=True)})"
    if t == "tool_result":
        return str(b.get("content", ""))
    return json.dumps(b, sort_keys=True)


def render_message(msg: dict, include_reasoning: bool) -> str:
    parts = [str(msg.get("role", ""))]
    if include_reasoning and msg.get("reasoning_content"):
        parts.append(str(msg["reasoning_content"]))
    c = msg.get("content")
    if isinstance(c, list):
        parts.extend(render_block(b) for b in c)
    elif c:
        parts.append(str(c))
    return "\n".join(parts)


def prompts_for_trace(path: Path, include_reasoning: bool) -> list[str]:
    """Prompt text sent before each assistant turn."""
    history: list[dict] = []
    prompts: list[str] = []
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") != "message":
            continue
        m = d.get("message") or {}
        if m.get("role") == "assistant":
            prompts.append("\n".join(
                render_message(h, include_reasoning) for h in history))
        history.append(m)
    return [p for p in prompts if p]


def score(trace_dir: Path, include_reasoning: bool, max_files: int) -> dict:
    measured: list[float] = []
    inferred: list[float] = []
    files = sorted(p for p in trace_dir.rglob("*.jsonl")
                   if not p.name.startswith("batch_"))[:max_files]
    for f in files:
        ps = prompts_for_trace(f, include_reasoning)
        for a, b in zip(ps, ps[1:]):
            if not b:
                continue
            shared = len(os.path.commonprefix([a, b]))
            measured.append(shared / len(b))
            new_prefill = max(1, len(b) - len(a))
            inferred.append(max(0.0, 1.0 - new_prefill / len(b)))
    return {"measured": measured, "inferred": inferred, "files": len(files)}


def q(xs, p):
    xs = sorted(xs)
    return xs[min(int(p * (len(xs) - 1)), len(xs) - 1)] if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=Path,
                    default=Path.home() / "traces/DeepSeek-V4-Flash-0731_26-08-05-17-13")
    ap.add_argument("--max-files", type=int, default=300)
    args = ap.parse_args()

    if not args.traces.is_dir():
        print(f"[error] not a directory: {args.traces}")
        return 1

    print("\n=== Claw-Eval true prefix reuse (Think Max capture) ===\n")
    for label, inc in (("drop_thinking=True  (as served)", False),
                       ("drop_thinking=False (counterfactual)", True)):
        r = score(args.traces, inc, args.max_files)
        m, i = r["measured"], r["inferred"]
        if not m:
            print(f"  {label}: no turn pairs")
            continue
        print(f"  {label}   files={r['files']} turnpairs={len(m)}")
        print(f"    {'':>16}{'p10':>8}{'p50':>8}{'p90':>8}{'mean':>8}")
        print(f"    {'TRUE prefix':>16}{q(m,.1):>8.3f}{q(m,.5):>8.3f}{q(m,.9):>8.3f}"
              f"{statistics.mean(m):>8.3f}")
        print(f"    {'delta heuristic':>16}{q(i,.1):>8.3f}{q(i,.5):>8.3f}{q(i,.9):>8.3f}"
              f"{statistics.mean(i):>8.3f}")
        gap = statistics.mean(i) - statistics.mean(m)
        print(f"    overstatement by heuristic: {gap:+.3f}"
              f"   {'OK' if abs(gap) < 0.10 else '*** HEURISTIC WRONG ***'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
