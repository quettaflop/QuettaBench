#!/usr/bin/env python3
"""Write the trajectory JSONL the session traces read: one session per line,
{session_id, source, num_turns, turns[{turn_idx, input_tokens, output_tokens, ...}]}.

chat reads ShareGPT and counts tokens with --tokenizer; each turn's `messages`
is only the new user message, and input_tokens is the full replayed context.
claweval converts Claw-Eval event-stream traces (github.com/claw-eval/claw-eval),
one file per task, taking each assistant turn's server-reported usage and the
tool time between LLM calls; --lss-out also writes LLMServingSim agentic JSONL.

    python3 data/build_trajectories.py chat \
      --tokenizer /path/to/Llama-3.1-8B-Instruct \
      --max-sessions 4000 --out data/chat_trajectories.jsonl
    python3 data/build_trajectories.py claweval \
        --traces ~/traces/DeepSeek-V4-Flash-0731_26-08-05-17-13 \
        --out data/claweval_trajectories.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CHAT_OUT = ROOT / "data" / "chat_trajectories.jsonl"
CLAWEVAL_OUT = ROOT / "data" / "claweval_trajectories.jsonl"
HF_DATASET = "Aeala/ShareGPT_Vicuna_unfiltered"   # same id ShareGPTMultiTurnDataset uses


def _load_tokenizer(name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


def _count(tok, messages: list[dict[str, str]]) -> int:
    """Token length of `messages` as the server would see them: chat template if the
    tokenizer has one, else a plain role-prefixed concatenation.

    NB: render the template to TEXT and tokenize that, rather than passing
    tokenize=True. On transformers 5.x the tokenize=True form returns a BatchEncoding,
    so len() silently yields 2 (its key count) and every context measures 2 tokens.
    add_special_tokens=False because the template already emits BOS/headers."""
    try:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return len(tok(text, add_special_tokens=False)["input_ids"])
    except Exception:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return len(tok(text, add_special_tokens=True)["input_ids"])


def _pairs(convs: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """ShareGPT 'conversations' -> [(human, assistant)] pairs, tolerant of the
    from-label variants in the dataset and of leading assistant turns."""
    human_keys = {"human", "user"}
    out: list[tuple[str, str]] = []
    pending: str | None = None
    for msg in convs:
        role = str(msg.get("from") or msg.get("role") or "").strip().lower()
        text = (msg.get("value") or msg.get("content") or "").strip()
        if not text:
            continue
        if role in human_keys:
            pending = text
        elif pending is not None:            # assistant reply closes the pair
            out.append((pending, text))
            pending = None
    return out


def build_chat(args: argparse.Namespace) -> int:
    import datasets as hf_datasets

    print(f"[chat-traj] loading {HF_DATASET} (split=train)", flush=True)
    ds = hf_datasets.load_dataset(HF_DATASET, split="train")
    print(f"[chat-traj] {len(ds)} raw conversations", flush=True)

    print(f"[chat-traj] tokenizer: {args.tokenizer}", flush=True)
    tok = _load_tokenizer(args.tokenizer)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sessions = turns_total = skipped = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for i, item in enumerate(ds):
            if sessions >= args.max_sessions:
                break
            pairs = _pairs(item.get("conversations") or [])
            if len(pairs) < args.min_turns:
                skipped += 1
                continue
            pairs = pairs[: args.max_turns]

            history: list[dict[str, str]] = []
            turns: list[dict[str, Any]] = []
            for idx, (human, assistant) in enumerate(pairs):
                new_msgs = [{"role": "user", "content": human}]
                # input_tokens is the FULL context this turn: replayed history + new user
                # message. That growing total is what the builder differences into
                # new_prefill / cache_hit_rate, so it must include the history.
                input_tokens = _count(tok, history + new_msgs)
                output_tokens = len(tok(assistant, add_special_tokens=False)["input_ids"])
                if input_tokens <= 0 or output_tokens <= 0:
                    continue
                if args.max_context_tokens and input_tokens > args.max_context_tokens:
                    break               # stop the session at the context cap
                turns.append({
                    "turn_idx": idx,
                    "messages": new_msgs,
                    "input_tokens": int(input_tokens),
                    "output_tokens": int(output_tokens),
                })
                history = history + new_msgs + [{"role": "assistant", "content": assistant}]

            if len(turns) < args.min_turns:
                skipped += 1
                continue
            fh.write(json.dumps({
                "session_id": str(item.get("id") or f"sharegpt_{i}"),
                "source": "sharegpt",
                "num_turns": len(turns),
                "turns": turns,
            }, ensure_ascii=False) + "\n")
            sessions += 1
            turns_total += len(turns)
            if sessions % 500 == 0:
                print(f"[chat-traj] {sessions} sessions, {turns_total} turns", flush=True)

    size_mb = out_path.stat().st_size / 1e6
    print(f"[chat-traj] WROTE {out_path}", flush=True)
    print(f"[chat-traj] sessions={sessions} turns={turns_total} "
          f"skipped={skipped} size={size_mb:.1f} MB", flush=True)
    print(f"[chat-traj] dataset={HF_DATASET} tokenizer={args.tokenizer}", flush=True)
    if not sessions:
        print("[chat-traj] FAIL: no sessions written", file=sys.stderr)
        return 1

    # Self-check: input_tokens MUST grow across turns, since it is the full replayed
    # context, and a reader differences it into new_prefill and cache_hit_rate. A flat series means the token counter is broken (this caught
    # apply_chat_template(tokenize=True) returning a BatchEncoding, which made every
    # context measure 2 tokens) and would yield garbage deltas.
    grew = flat = 0
    with out_path.open(encoding="utf-8") as fh:
        for line in fh:
            toks = [t["input_tokens"] for t in json.loads(line)["turns"]]
            if len(toks) < 2:
                continue
            if all(b > a for a, b in zip(toks, toks[1:])):
                grew += 1
            elif len(set(toks)) == 1:
                flat += 1
    multi = grew + flat
    print(f"[chat-traj] self-check: strictly-growing context {grew}/{multi} "
          f"multi-turn sessions (flat: {flat})", flush=True)
    if multi and grew / multi < 0.9:
        print("[chat-traj] FAIL: context is not growing across turns -- the token "
              "counter is wrong; refusing to ship this artifact", file=sys.stderr)
        return 1
    return 0


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
            # silently mix fidelity classes within one trajectory set. Skip loudly
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
                # TOTAL context at this turn; readers difference successive turns, so never pre-subtract.
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
    try:
        return int(datetime.fromisoformat(ts).timestamp() * 1e9)
    except Exception:
        return None


def build_claweval(args: argparse.Namespace) -> int:
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="source", required=True)
    chat = sub.add_parser("chat", help="ShareGPT sessions, tokenized with --tokenizer")
    chat.add_argument("--out", type=Path, default=CHAT_OUT)
    chat.add_argument("--tokenizer", required=True,
                      help="local model dir or HF id; counts are recorded with the artifact")
    chat.add_argument("--max-sessions", type=int, default=4000)
    chat.add_argument("--min-turns", type=int, default=2)
    chat.add_argument("--max-turns", type=int, default=20,
                      help="matches the chat-multiturn profile's max_turns")
    chat.add_argument("--max-context-tokens", type=int, default=32768,
                      help="truncate a session once its context exceeds this (0 = no cap)")
    chat.set_defaults(build=build_chat)
    claw = sub.add_parser("claweval", help="Claw-Eval traces, with their server-reported usage")
    claw.add_argument("--traces", required=True, type=Path,
                      help="Claw-Eval run directory containing *.jsonl traces")
    claw.add_argument("--out", type=Path, default=CLAWEVAL_OUT,
                      help=f"output QuettaBench trajectory JSONL (default: {CLAWEVAL_OUT})")
    claw.add_argument("--lss-out", type=Path, default=None,
                      help="also emit LLMServingSim *agentic* JSONL to this path")
    claw.add_argument("--lss-arrival", choices=("real", "poisson"), default="real",
                      help="LLMServingSim session arrival model (default: real, from trace_start)")
    claw.add_argument("--lss-rate", type=float, default=1.0,
                      help="sessions/sec when --lss-arrival=poisson")
    claw.add_argument("--lss-seed", type=int, default=42)
    claw.set_defaults(build=build_claweval)
    args = ap.parse_args()
    return args.build(args)


if __name__ == "__main__":
    sys.exit(main())
