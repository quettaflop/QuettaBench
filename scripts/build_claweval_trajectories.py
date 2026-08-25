#!/usr/bin/env python3
"""Convert Claw-Eval agent traces into QuettaBench trajectory JSONL.

Claw-Eval (github.com/claw-eval/claw-eval) writes one JSONL *event stream* per
task: a `trace_start` record, then interleaved `message` / `tool_dispatch` /
`audit_snapshot` records, then `trace_end`. QuettaBench's distribution builder
instead wants one JSON object per *session* with a flat `turns` list, so the two
formats do not line up without this step.

What makes these traces worth ingesting: every assistant turn carries a real
`usage` block from the serving engine, so `total_context_tokens` comes from the
server tokenizer rather than the coarse word-ratio estimate that
`build_trace_distributions.message_tokens()` falls back to. That is the same
fidelity class as the captured-vLLM sources, not the estimated ones.

Output rows match what `build_trajectory_distribution()` reads:

    {"session_id": ..., "num_turns": N,
     "turns": [{"turn_idx": i, "input_tokens": ctx, "output_tokens": out}, ...]}

Usage:
    python scripts/build_claweval_trajectories.py \
        --traces ~/traces/DeepSeek-V4-Flash-0731_26-08-05-17-13 \
        --out data/claweval_trajectories.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_OUT = ROOT / "data" / "claweval_trajectories.jsonl"


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON records, skipping blank and malformed lines."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def convert_trace(path: Path) -> dict[str, Any] | None:
    """Convert one Claw-Eval trace file into a single trajectory session row."""
    task_id: str | None = None
    model: str | None = None
    tool_time_s = 0.0
    started_at: str | None = None
    turns: list[dict[str, Any]] = []
    thinking_turns = 0
    thinking_chars = 0
    # tool_dispatch records appear AFTER the assistant turn that issued them and
    # before the next one, so they measure the gap between LLM calls. Verified
    # against trace_end.tool_time_s: summing (not max-ing) reproduces it exactly,
    # i.e. Claw-Eval executes a turn's tool calls serially.
    pending_tool_ms = 0.0

    for rec in iter_records(path):
        kind = rec.get("type")

        if kind == "trace_start":
            task_id = rec.get("task_id")
            model = rec.get("model")
            started_at = rec.get("timestamp")
            continue

        if kind == "trace_end":
            # Session-level wall/tool timing; useful for the agentic formats that
            # model think-time between LLM calls.
            tool_time_s = float(rec.get("tool_time_s") or 0.0)
            continue

        if kind == "tool_dispatch":
            pending_tool_ms += float(rec.get("latency_ms") or 0.0)
            continue

        if kind != "message":
            continue

        msg = rec.get("message") or {}
        if msg.get("role") != "assistant":
            # Only assistant turns correspond to an LLM call. user/tool records
            # are context that is already priced into the next call's usage.
            continue

        usage = rec.get("usage") or {}
        in_tok = usage.get("input_tokens")
        out_tok = usage.get("output_tokens")
        if in_tok is None or out_tok is None:
            # No server-reported usage -> we would have to estimate, which would
            # silently mix fidelity classes within one distribution. Skip loudly
            # instead (counted by the caller).
            continue

        in_tok = int(in_tok)
        out_tok = int(out_tok)
        if in_tok <= 0:
            continue

        # Think Max reasoning is billed inside output_tokens by the engine, so it
        # is already reflected above. Tracked separately only for reporting.
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        if reasoning:
            thinking_turns += 1
            thinking_chars += len(reasoning)

        # Attribute tools observed since the PREVIOUS assistant turn to that turn.
        if turns:
            turns[-1]["tool_ms"] = round(pending_tool_ms, 3)
        pending_tool_ms = 0.0

        turns.append(
            {
                "turn_idx": len(turns),
                # `input_tokens` is read by build_trajectory_distribution() as the
                # TOTAL context at this turn, which is exactly OpenAI/vLLM prompt
                # token semantics. It derives new_prefill and cache_hit_rate from
                # successive deltas, so do not pre-subtract anything here.
                "input_tokens": in_tok,
                "output_tokens": max(1, out_tok),
                "tool_ms": 0.0,
            }
        )

    if not turns:
        return None

    # Trailing tools (after the final assistant turn) belong to that turn.
    turns[-1]["tool_ms"] = round(turns[-1].get("tool_ms", 0.0) + pending_tool_ms, 3)

    return {
        "session_id": task_id or path.stem,
        "num_turns": len(turns),
        "turns": turns,
        "source": "claw-eval",
        "model": model,
        "tool_time_s": round(tool_time_s, 3),
        "started_at": started_at,
        "thinking_turns": thinking_turns,
        "thinking_chars": thinking_chars,
    }


def write_llmservingsim(sessions: list[dict[str, Any]], out: Path,
                        arrival: str, rate: float, seed: int) -> None:
    """Emit LLMServingSim *agentic* JSONL.

        {"session_id", "arrival_time_ns",
         "sub_requests": [{"input_toks", "output_toks", "tool_duration_ns"}]}

    Each sub-request's arrival is derived by the simulator from the previous
    call's completion plus `tool_duration_ns`, so only the SESSION needs an
    arrival time.

    Not emitted: `input_tok_ids`. LLMServingSim uses real token IDs to hash
    prefixes for its cache model; Claw-Eval traces store text, not IDs. Without
    them the simulator falls back to count-only prefix handling. To get true
    prefix-cache fidelity you would re-tokenize the stored text with the
    DeepSeek V4 tokenizer and attach IDs here.
    """
    import random

    rng = random.Random(seed)

    # Real arrivals preserve the capture's own pacing. Note the capture ran at a
    # fixed 16-way parallelism, so "real" reproduces THAT schedule, not a natural
    # arrival process -- use poisson for open-loop rate ladders.
    base_ns: int | None = None
    if arrival == "real":
        stamps = []
        for s in sessions:
            ts = s.get("started_at")
            stamps.append(_parse_ts_ns(ts) if ts else None)
        known = [t for t in stamps if t is not None]
        base_ns = min(known) if known else 0
    else:
        stamps = [None] * len(sessions)

    clock_ns = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for s, ts in zip(sessions, stamps):
            if arrival == "real" and ts is not None:
                arrival_ns = max(0, ts - (base_ns or 0))
            else:
                # Exponential inter-arrival = Poisson process at `rate`.
                clock_ns += int(rng.expovariate(rate) * 1e9)
                arrival_ns = clock_ns
            f.write(json.dumps({
                "session_id": s["session_id"],
                "arrival_time_ns": arrival_ns,
                "sub_requests": [
                    {
                        "input_toks": t["input_tokens"],
                        "output_toks": t["output_tokens"],
                        "tool_duration_ns": int(round(t.get("tool_ms", 0.0) * 1e6)),
                    }
                    for t in s["turns"]
                ],
            }) + "\n")
    print(f"[write] {out}  (LLMServingSim agentic, arrival={arrival})")


def _parse_ts_ns(ts: str) -> int | None:
    from datetime import datetime
    try:
        return int(datetime.fromisoformat(ts).timestamp() * 1e9)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True, type=Path,
                    help="Claw-Eval run directory containing *.jsonl traces")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"output QuettaBench trajectory JSONL (default: {DEFAULT_OUT})")
    ap.add_argument("--lss-out", type=Path, default=None,
                    help="also emit LLMServingSim *agentic* JSONL to this path")
    ap.add_argument("--lss-arrival", choices=("real", "poisson"), default="real",
                    help="LLMServingSim session arrival model (default: real, from trace_start)")
    ap.add_argument("--lss-rate", type=float, default=1.0,
                    help="sessions/sec when --lss-arrival=poisson")
    ap.add_argument("--lss-seed", type=int, default=42)
    args = ap.parse_args()

    trace_dir: Path = args.traces.expanduser()
    if not trace_dir.is_dir():
        print(f"[error] not a directory: {trace_dir}")
        return 1

    # batch_results/batch_summary are run metadata, not traces.
    files = sorted(p for p in trace_dir.rglob("*.jsonl") if not p.name.startswith("batch_"))
    if not files:
        print(f"[error] no *.jsonl traces under {trace_dir}")
        return 1

    sessions: list[dict[str, Any]] = []
    empty = 0
    for p in files:
        row = convert_trace(p)
        if row is None:
            empty += 1
            continue
        sessions.append(row)

    if not sessions:
        print("[error] no sessions with usable per-turn usage")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for s in sessions:
            f.write(json.dumps(s) + "\n")

    if args.lss_out:
        write_llmservingsim(sessions, args.lss_out, args.lss_arrival,
                            args.lss_rate, args.lss_seed)

    turn_counts = [s["num_turns"] for s in sessions]
    all_ctx = [t["input_tokens"] for s in sessions for t in s["turns"]]
    all_out = [t["output_tokens"] for s in sessions for t in s["turns"]]
    thinking = sum(s["thinking_turns"] for s in sessions)

    print(f"[write] {args.out}")
    print(f"  trace files      : {len(files)}  (skipped, no usable turns: {empty})")
    print(f"  sessions         : {len(sessions)}")
    print(f"  turns            : {sum(turn_counts)}  "
          f"(min={min(turn_counts)} p50={statistics.median(turn_counts):.0f} max={max(turn_counts)})")
    print(f"  context tokens   : p50={statistics.median(all_ctx):,.0f} max={max(all_ctx):,}")
    print(f"  output tokens    : p50={statistics.median(all_out):,.0f} max={max(all_out):,}")
    print(f"  turns w/ thinking: {thinking} ({thinking * 100 // max(1, sum(turn_counts))}%)")
    print("  turn buckets:")
    for lo, hi, label in [(1, 5, "short (1-5)"), (5, 10, "medium (5-10)"),
                          (10, 20, "long (10-20)"), (20, 1000, "xl (20+)")]:
        n = sum(1 for t in turn_counts if lo <= t < hi)
        print(f"    {label}: {n} sessions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
