#!/usr/bin/env python3
"""Build the session-replay JSONL files from recorded agent runs; each step keeps the token counts the engine reported.
just fetch calls the four builders; each one keeps only steps whose request and reply were both recorded."""

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
DEEPRESEARCH_OUT = ROOT / "data" / "deepresearch_trajectories.jsonl"
DEEPSEARCH_OUT = ROOT / "data" / "deepsearch_trajectories.jsonl"
SWEBENCH_NATIVE_OUT = ROOT / "data" / "swebench_native_trajectories.jsonl"
TERMINALBENCH_ATIF_OUT = ROOT / "data" / "terminalbench_qwen_trajectories.jsonl"
HF_DATASET = "Aeala/ShareGPT_Vicuna_unfiltered"


def _load_tokenizer(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def _count(tok, messages: list[dict[str, str]]) -> int:
    """Via the chat template, else role-prefixed concatenation."""
    # tokenize=True returns a BatchEncoding on transformers 5.x, so render to text first.
    try:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return len(tok(text, add_special_tokens=False)["input_ids"])
    except Exception:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return len(tok(text, add_special_tokens=True)["input_ids"])


def _pairs(convs: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Leading assistant turns are dropped."""
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
        elif pending is not None:
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
                # Full replayed context; readers difference it per turn.
                input_tokens = _count(tok, history + new_msgs)
                output_tokens = len(tok(assistant, add_special_tokens=False)["input_ids"])
                if input_tokens <= 0 or output_tokens <= 0:
                    continue
                if args.max_context_tokens and input_tokens > args.max_context_tokens:
                    break
                turns.append(
                    {
                        "turn_idx": idx,
                        "messages": new_msgs,
                        "input_tokens": int(input_tokens),
                        "output_tokens": int(output_tokens),
                    }
                )
                history = history + new_msgs + [{"role": "assistant", "content": assistant}]

            if len(turns) < args.min_turns:
                skipped += 1
                continue
            fh.write(
                json.dumps(
                    {
                        "session_id": str(item.get("id") or f"sharegpt_{i}"),
                        "source": "sharegpt",
                        "num_turns": len(turns),
                        "turns": turns,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            sessions += 1
            turns_total += len(turns)
            if sessions % 500 == 0:
                print(f"[chat-traj] {sessions} sessions, {turns_total} turns", flush=True)

    size_mb = out_path.stat().st_size / 1e6
    print(f"[chat-traj] WROTE {out_path}", flush=True)
    print(f"[chat-traj] sessions={sessions} turns={turns_total} skipped={skipped} size={size_mb:.1f} MB", flush=True)
    print(f"[chat-traj] dataset={HF_DATASET} tokenizer={args.tokenizer}", flush=True)
    if not sessions:
        print("[chat-traj] FAIL: no sessions written", file=sys.stderr)
        return 1

    # A flat input_tokens series means the token counter is broken.
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
    print(
        f"[chat-traj] self-check: strictly-growing context {grew}/{multi} multi-turn sessions (flat: {flat})",
        flush=True,
    )
    if multi and grew / multi < 0.9:
        print(
            "[chat-traj] FAIL: context is not growing across turns -- the token "
            "counter is wrong; refusing to ship this artifact",
            file=sys.stderr,
        )
        return 1
    return 0


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
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
    """None if no assistant turn has usage."""
    task_id: str | None = None
    model: str | None = None
    tool_time_s = 0.0
    started_at: str | None = None
    turns: list[dict[str, Any]] = []
    thinking_turns = 0
    thinking_chars = 0
    # Tool calls run serially, so latencies are summed (matches trace_end.tool_time_s).
    pending_tool_ms = 0.0

    for rec in iter_records(path):
        kind = rec.get("type")

        if kind == "trace_start":
            task_id = rec.get("task_id")
            model = rec.get("model")
            started_at = rec.get("timestamp")
            continue

        if kind == "trace_end":
            tool_time_s = float(rec.get("tool_time_s") or 0.0)
            continue

        if kind == "tool_dispatch":
            pending_tool_ms += float(rec.get("latency_ms") or 0.0)
            continue

        if kind != "message":
            continue

        msg = rec.get("message") or {}
        if msg.get("role") != "assistant":
            continue

        usage = rec.get("usage") or {}
        in_tok = usage.get("input_tokens")
        out_tok = usage.get("output_tokens")
        if in_tok is None or out_tok is None:
            # Estimating would mix measured and guessed counts in one set.
            continue

        in_tok = int(in_tok)
        out_tok = int(out_tok)
        if in_tok <= 0:
            continue

        # Reasoning is already inside output_tokens.
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        if reasoning:
            thinking_turns += 1
            thinking_chars += len(reasoning)

        # Tool time since the previous assistant turn belongs to that turn.
        if turns:
            turns[-1]["tool_ms"] = round(pending_tool_ms, 3)
        pending_tool_ms = 0.0

        turns.append(
            {
                "turn_idx": len(turns),
                # Total context, not a delta.
                "input_tokens": in_tok,
                "output_tokens": max(1, out_tok),
                "tool_ms": 0.0,
            }
        )

    if not turns:
        return None

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


def write_llmservingsim(sessions: list[dict[str, Any]], out: Path, arrival: str, rate: float, seed: int) -> None:
    """Only the session carries an arrival time; sub-requests chain on tool_duration_ns."""
    rng = random.Random(seed)

    # The capture ran at fixed 16-way parallelism: "real" is not an open-loop arrival process.
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
                clock_ns += int(rng.expovariate(rate) * 1e9)
                arrival_ns = clock_ns
            f.write(
                json.dumps(
                    {
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
                    }
                )
                + "\n"
            )
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

    # batch_* files are run metadata.
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
        write_llmservingsim(sessions, args.lss_out, args.lss_arrival, args.lss_rate, args.lss_seed)

    turn_counts = [s["num_turns"] for s in sessions]
    all_ctx = [t["input_tokens"] for s in sessions for t in s["turns"]]
    all_out = [t["output_tokens"] for s in sessions for t in s["turns"]]
    thinking = sum(s["thinking_turns"] for s in sessions)

    print(f"[write] {args.out}")
    print(f"  trace files      : {len(files)}  (skipped, no usable turns: {empty})")
    print(f"  sessions         : {len(sessions)}")
    print(
        f"  turns            : {sum(turn_counts)}  "
        f"(min={min(turn_counts)} p50={statistics.median(turn_counts):.0f} max={max(turn_counts)})"
    )
    print(f"  context tokens   : p50={statistics.median(all_ctx):,.0f} max={max(all_ctx):,}")
    print(f"  output tokens    : p50={statistics.median(all_out):,.0f} max={max(all_out):,}")
    print(f"  turns w/ thinking: {thinking} ({thinking * 100 // max(1, sum(turn_counts))}%)")
    print("  turn buckets:")
    for lo, hi, label in [
        (1, 5, "short (1-5)"),
        (5, 10, "medium (5-10)"),
        (10, 20, "long (10-20)"),
        (20, 1000, "xl (20+)"),
    ]:
        n = sum(1 for t in turn_counts if lo <= t < hi)
        print(f"    {label}: {n} sessions")
    return 0


def deepresearch_session(row: dict[str, Any], source: str = "deepresearch") -> dict[str, Any]:
    """Each turn's prompt_text already holds the whole context; its search time (retrieval_s, or tool_s in the
    DeepSearchQA traces) is the tool wait after it."""
    turns = [
        {
            "turn_idx": t["turn_index"],
            "messages": [{"role": "user", "content": t["prompt_text"]}],
            "prompt_tokens": t["prompt_tokens"],
            "osl_tokens": max(1, t["completion_tokens"]),
            "tool_ms": 1000.0 * (t.get("retrieval_s") or t.get("tool_s") or 0.0),
        }
        for t in row["turns"]
    ]
    return {"session_id": row["session_id"], "source": source, "num_turns": len(turns), "turns": turns}


def write_sessions(sessions: list[dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for s in sessions:
            f.write(json.dumps(s) + "\n")
    steps = sorted(s["num_turns"] for s in sessions)
    print(
        f"[write] {out}: {len(sessions)} sessions, {sum(steps)} turns, "
        f"steps {steps[0]}-{steps[-1]} (median {steps[len(steps) // 2]})"
    )


def build_deepresearch(args: argparse.Namespace) -> int:
    sessions = [deepresearch_session(r, args.source) for r in iter_records(args.trace) if r.get("turns")]
    if not sessions:
        print(f"[error] no sessions in {args.trace}")
        return 1
    write_sessions(sessions, args.out)
    return 0


# mini-swe-agent 2.4.6 sends this schema with every request (minisweagent/models/utils/actions_toolcall.py BASH_TOOL);
# the server counts it in each step's prompt_tokens.
MINI_SWE_AGENT_BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to execute"}},
            "required": ["command"],
        },
    },
}


def wait_after_ms(reqs: list[dict[str, Any]], k: int) -> float:
    """Tool wait to replay after step k, from the recorded send times."""
    if k + 1 >= len(reqs):
        return 0.0
    return max(0.0, 1000.0 * (float(reqs[k + 1]["t_request_sent"]) - float(reqs[k]["t_response_done"])))


def swebench_native_session(run_dir: Path) -> dict[str, Any] | None:
    """Turn one mini-swe-agent run into a replay session; skipped when its request log and trajectory disagree."""
    trajs = sorted(run_dir.glob("*.traj.json"))
    sidecar = run_dir / "requests.jsonl"
    if not trajs or not sidecar.exists():
        return None
    reqs = sorted((json.loads(line) for line in sidecar.open()), key=lambda r: int(r["turn_idx"]))
    msgs = json.loads(trajs[0].read_text())["messages"]
    if sum(m["role"] == "assistant" for m in msgs) != len(reqs):
        return None
    history: list[dict[str, Any]] = []
    turns = []
    for m in msgs:
        if m["role"] == "assistant":
            r = reqs[len(turns)]
            turns.append(
                {
                    "turn_idx": len(turns),
                    "messages": [dict(h) for h in history],
                    "prompt_tokens": int(r["prompt_tokens"]),
                    "osl_tokens": max(1, int(r["completion_tokens"])),
                    "tool_ms": wait_after_ms(reqs, len(turns)),
                }
            )
        history.append({k: v for k, v in m.items() if k != "extra" and v is not None})
    return {
        "session_id": run_dir.name,
        "source": "swebench",
        "run": reqs[0].get("run_id"),
        "tools": [MINI_SWE_AGENT_BASH_TOOL],
        "num_turns": len(turns),
        "turns": turns,
    }


def build_swebench_native(args: argparse.Namespace) -> int:
    sessions, skipped = [], []
    for d in sorted(p for p in args.runs.iterdir() if p.is_dir()):
        s = swebench_native_session(d)
        if s:
            sessions.append(s)
        else:
            skipped.append(d.name)
    if not sessions:
        print(f"[error] no usable sessions in {args.runs}")
        return 1
    write_sessions(sessions, args.out)
    print(f"[skip] {len(skipped)} sessions where sidecar and trajectory disagree on model calls: {' '.join(skipped)}")
    return 0


def terminus_reply(step: dict[str, Any]) -> str:
    """The JSON reply terminus-2 asked the model for, rebuilt from the step's parsed fields; the model's own text is
    not recorded."""
    analysis, _, plan = (step.get("message") or "").partition("\nPlan: ")
    calls = step.get("tool_calls") or []
    body: dict[str, Any] = {
        "analysis": analysis.removeprefix("Analysis: "),
        "plan": plan,
        "commands": [c.get("arguments") for c in calls if c.get("function_name") == "bash_command"],
    }
    if any(c.get("function_name") == "mark_task_complete" for c in calls):
        body["task_complete"] = True
    return json.dumps(body, indent=2, ensure_ascii=False)


def telemetry_gaps_ms(path: Path) -> list[float]:
    """Tool waits between model calls, from Harbor's telemetry."""
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.open() if line.strip()]
    calls = sorted((r for r in rows if r.get("kind") == "model"), key=lambda r: (int(r["turn_id"]), r["model_start"]))
    gaps = [max(0.0, 1000.0 * (b["model_start"] - a["model_end"])) for a, b in zip(calls, calls[1:])]
    return gaps + [0.0] if calls else []


def atif_session(path: Path, source: str) -> dict[str, Any] | None:
    """Turn one Terminal-Bench trajectory into a replay session; skipped when a step lacks token metrics."""
    traj = json.loads(path.read_text())
    history: list[dict[str, str]] = []
    turns = []
    for step in traj.get("steps") or []:
        if step.get("source") != "agent":
            if step.get("message"):
                role = "system" if step.get("source") == "system" else "user"
                history.append({"role": role, "content": step["message"]})
            continue
        metrics = step.get("metrics") or {}
        if not metrics.get("completion_tokens"):
            return None
        turns.append(
            {
                "turn_idx": len(turns),
                "messages": [dict(h) for h in history],
                "prompt_tokens": metrics.get("prompt_tokens"),
                "osl_tokens": max(1, metrics["completion_tokens"]),
            }
        )
        history.append({"role": "assistant", "content": terminus_reply(step)})
        results = (step.get("observation") or {}).get("results") or []
        output = "\n".join(r.get("content", "") for r in results if isinstance(r, dict))
        if output:
            history.append({"role": "user", "content": output})
    if not turns:
        return None
    gaps = telemetry_gaps_ms(path.with_name("telemetry.jsonl"))
    recorded_waits = len(gaps) == len(turns)
    if recorded_waits:
        for turn, gap in zip(turns, gaps):
            turn["tool_ms"] = gap
    return {
        "session_id": traj.get("session_id") or path.parent.parent.name,
        "source": source,
        "run": path.parts[-4],
        "recorded_waits": recorded_waits,
        "num_turns": len(turns),
        "turns": turns,
    }


def build_terminalbench_atif(args: argparse.Namespace) -> int:
    """One session per task: a task tried in several Harbor jobs keeps its latest trial."""
    by_task: dict[str, tuple[str, Path]] = {}
    for p in sorted(args.jobs.glob("*/*/agent/trajectory.json")):
        task, job = p.parts[-3].split("__")[0], p.parts[-4]
        if task not in by_task or job > by_task[task][0]:
            by_task[task] = (job, p)
    sessions, skipped = [], []
    for task, (_, p) in sorted(by_task.items()):
        s = atif_session(p, "terminalbench")
        if s:
            sessions.append(s)
        else:
            skipped.append(task)
    if not sessions:
        print(f"[error] no usable sessions in {args.jobs}")
        return 1
    write_sessions(sessions, args.out)
    print(f"[skip] {len(skipped)} tasks without per-step metrics: {' '.join(skipped)}")
    print(f"[waits] {sum(s['recorded_waits'] for s in sessions)} of {len(sessions)} sessions carry recorded waits")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="source", required=True)
    chat = sub.add_parser("chat", help="ShareGPT sessions, tokenized with --tokenizer")
    chat.add_argument("--out", type=Path, default=CHAT_OUT)
    chat.add_argument(
        "--tokenizer", required=True, help="local model dir or HF id; counts are recorded with the artifact"
    )
    chat.add_argument("--max-sessions", type=int, default=4000)
    chat.add_argument("--min-turns", type=int, default=2)
    chat.add_argument("--max-turns", type=int, default=20, help="matches the chat-multiturn profile's max_turns")
    chat.add_argument(
        "--max-context-tokens",
        type=int,
        default=32768,
        help="truncate a session once its context exceeds this (0 = no cap)",
    )
    chat.set_defaults(build=build_chat)
    claw = sub.add_parser("claweval", help="Claw-Eval traces, with their server-reported usage")
    claw.add_argument("--traces", required=True, type=Path, help="Claw-Eval run directory containing *.jsonl traces")
    claw.add_argument(
        "--out", type=Path, default=CLAWEVAL_OUT, help=f"output QuettaBench trajectory JSONL (default: {CLAWEVAL_OUT})"
    )
    claw.add_argument("--lss-out", type=Path, default=None, help="also emit LLMServingSim *agentic* JSONL to this path")
    claw.add_argument(
        "--lss-arrival",
        choices=("real", "poisson"),
        default="real",
        help="LLMServingSim session arrival model (default: real, from trace_start)",
    )
    claw.add_argument("--lss-rate", type=float, default=1.0, help="sessions/sec when --lss-arrival=poisson")
    claw.add_argument("--lss-seed", type=int, default=42)
    claw.set_defaults(build=build_claweval)
    dr = sub.add_parser("deepresearch", help="DeepResearch-9k traces (dr9k-v4flash.jsonl), with their recorded usage")
    dr.add_argument("--trace", required=True, type=Path, help="the DeepResearch-9k trace JSONL")
    dr.add_argument("--out", type=Path, default=DEEPRESEARCH_OUT)
    dr.set_defaults(build=build_deepresearch, source="deepresearch")
    ds = sub.add_parser("deepsearch", help="DeepSearchQA traces (dsqa-v4flash.jsonl), the same session shape")
    ds.add_argument("--trace", required=True, type=Path, help="the DeepSearchQA trace JSONL")
    ds.add_argument("--out", type=Path, default=DEEPSEARCH_OUT)
    ds.set_defaults(build=build_deepresearch, source="deepsearch")
    sw = sub.add_parser("swebench-native", help="mini-swe-agent SWE-bench runs with per-request sidecars")
    sw.add_argument(
        "--runs",
        required=True,
        type=Path,
        help="run directory: one <instance>/ with *.traj.json and requests.jsonl each",
    )
    sw.add_argument("--out", type=Path, default=SWEBENCH_NATIVE_OUT)
    sw.set_defaults(build=build_swebench_native)
    tb = sub.add_parser("terminalbench-atif", help="Harbor Terminal-Bench ATIF trajectories with per-step metrics")
    tb.add_argument(
        "--jobs", required=True, type=Path, help="Harbor jobs directory: <job>/<trial>/agent/trajectory.json"
    )
    tb.add_argument("--out", type=Path, default=TERMINALBENCH_ATIF_OUT)
    tb.set_defaults(build=build_terminalbench_atif)
    args = ap.parse_args()
    return args.build(args)


if __name__ == "__main__":
    sys.exit(main())
