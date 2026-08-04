#!/usr/bin/env python3
"""Capture a real chat trajectory source from ShareGPT into data/chat_trajectories.jsonl.

WHY THIS EXISTS
---------------
swebench / terminalbench / osworld each have a captured trajectory JSONL on R2, and
their synthetic distributions declare honest provenance back to it:

    source: {"kind": "trajectory_jsonl", "path": "data/terminalbench_trajectories.jsonl",
             "sessions": 267, "turns": 20042}

chat did not. `data/distributions/chat_multiturn.json` was instead derived from our own
prior benchmark output (`source.kind == "dashboard_per_turn_summary"`), because -- per
that file's own note -- "ShareGPT raw multi-turn source is not stored locally". That
makes the chat workload circular: it reproduces the token-length shape of runs we
already did, so a chat MAPE cannot validate anything about real chat traffic.

This script closes that gap by materialising the source the code already knows how to
read (`src/workloads/dataset.py::ShareGPTMultiTurnDataset` loads the same HF dataset),
so `build_trace_distributions.py` can treat chat exactly like the other three.

SCHEMA (matches the existing trajectory JSONLs, one session per line)
--------------------------------------------------------------------
    {"session_id": str, "source": "sharegpt", "num_turns": int,
     "turns": [{"turn_idx": int,
                "messages": [{"role": ..., "content": ...}],   # NEW messages this turn
                "input_tokens": int,      # FULL growing context at this turn
                "output_tokens": int}]}

`messages` holds only the messages *added* at that turn, not the replayed history:
the full context is already captured numerically in `input_tokens`, and storing the
cumulative history per turn would blow the file up quadratically (that is why the
agentic captures are GB-scale). The history is still reconstructable by concatenating
earlier turns. `build_trajectory_distribution()` prefers explicit `input_tokens` and
only falls back to a coarse word-ratio estimate over `messages`, so supplying real
tokenizer counts here is strictly more accurate than letting it guess.

Token counts use a real tokenizer with the model's chat template when available, which
is what the serving stack actually sees.

  python3 scripts/build_chat_trajectories.py \
      --tokenizer /home/kevinlau/models/Llama-3.1-8B-Instruct \
      --max-sessions 4000 --out data/chat_trajectories.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_OUT = ROOT / "data" / "chat_trajectories.jsonl"
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


def build(args: argparse.Namespace) -> int:
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
    # context and build_trajectory_distribution() differences it into new_prefill and
    # cache_hit_rate. A flat series means the token counter is broken (this caught
    # apply_chat_template(tokenize=True) returning a BatchEncoding, which made every
    # context measure 2 tokens) and would yield a garbage distribution.
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tokenizer", default="/home/kevinlau/models/Llama-3.1-8B-Instruct",
                    help="local model dir or HF id; counts are recorded with the artifact")
    ap.add_argument("--max-sessions", type=int, default=4000)
    ap.add_argument("--min-turns", type=int, default=2)
    ap.add_argument("--max-turns", type=int, default=20,
                    help="matches the chat-multiturn profile's max_turns")
    ap.add_argument("--max-context-tokens", type=int, default=32768,
                    help="truncate a session once its context exceeds this (0 = no cap)")
    return build(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
