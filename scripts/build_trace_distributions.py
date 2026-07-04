#!/usr/bin/env python3
"""Build compact synthetic-workload distributions from existing traces.

This is intentionally a no-op with respect to benchmark behavior: it only reads
existing trace/result files and writes JSON summaries under data/distributions/.
The runner/profile/dashboard wiring should consume these artifacts in a later
phase after the distributions have been inspected.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
DEFAULT_OUT_DIR = DATA_DIR / "distributions"

TRAJECTORY_SOURCES = {
    "swebench_multiturn": DATA_DIR / "swebench_trajectories.jsonl",
    "terminalbench_multiturn": DATA_DIR / "terminalbench_trajectories.jsonl",
    "osworld_multiturn": DATA_DIR / "osworld_trajectories.jsonl",
}

CHAT_RESULT_PROFILES = {
    "chat-multiturn-short",
    "chat-multiturn-medium",
    "chat-multiturn-long",
}

QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


@dataclass
class TurnSample:
    turn_index: int
    total_context_tokens: int
    new_prefill_tokens: int
    output_tokens: int
    cache_hit_rate: float
    source_session_id: str | None = None
    token_source: str | None = None


@dataclass
class SessionSample:
    session_id: str
    turn_count: int
    turns: list[TurnSample]
    context_decrease_turns: int = 0
    context_non_growth_turns: int = 0
    estimated_context_turns: int = 0


def estimate_tokens(value: Any) -> int:
    """Estimate tokens using the same coarse word ratio used by datasets.py."""
    if value is None:
        return 0
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False)
    words = text.split()
    if not words:
        return 0
    return max(1, int(len(words) * 1.35))


def message_tokens(messages: Iterable[dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.get("content", ""))
    return total


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    xs = sorted(values)
    pos = q * (len(xs) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(xs[lo])
    frac = pos - lo
    return float(xs[lo] * (1 - frac) + xs[hi] * frac)


def stats(values: Iterable[float], *, round_digits: int = 3) -> dict[str, float | int]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"count": 0}
    out: dict[str, float | int] = {
        "count": len(vals),
        "min": round(min(vals), round_digits),
        "mean": round(statistics.fmean(vals), round_digits),
        "max": round(max(vals), round_digits),
    }
    for q in QUANTILES:
        key = f"p{int(q * 100):02d}"
        out[key] = round(percentile(vals, q), round_digits)
    return out


def int_histogram(values: Iterable[int]) -> dict[str, int]:
    return {str(k): v for k, v in sorted(Counter(values).items())}


def round_turn_sample(sample: TurnSample) -> dict[str, int | float | str]:
    row: dict[str, int | float | str] = {
        "turn_index": sample.turn_index,
        "total_context_tokens": sample.total_context_tokens,
        "new_prefill_tokens": sample.new_prefill_tokens,
        "output_tokens": sample.output_tokens,
        "cache_hit_rate": round(sample.cache_hit_rate, 4),
    }
    if sample.source_session_id:
        row["source_session_id"] = sample.source_session_id
    if sample.token_source:
        row["token_source"] = sample.token_source
    return row


CAPTURED_MSE_SHORT_SOURCES = {
    "swebench_multiturn_short_tracereplay_filtered-mse": {
        "filename": "h100_real_swebench-short_conc5.json",
        "min_turns": 13,
        "max_turns": 30,
    },
    "terminalbench_multiturn_short_tracereplay_filtered-mse": {
        "filename": "h100_real_terminalbench-short_conc5.json",
        "min_turns": 2,
        "max_turns": 30,
    },
}


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def build_trajectory_distribution(name: str, path: Path) -> dict[str, Any]:
    sessions: list[SessionSample] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            raw_turns = row.get("turns") or []
            turns: list[TurnSample] = []
            previous_context = 0
            context_decrease_turns = 0
            context_non_growth_turns = 0
            estimated_context_turns = 0
            for idx, turn in enumerate(raw_turns):
                messages = turn.get("messages") or []
                total_context_raw = turn.get("input_tokens")
                if total_context_raw is None:
                    estimated_context_turns += 1
                total_context = int(total_context_raw or message_tokens(messages))
                if total_context <= 0:
                    continue
                output_tokens = int(
                    turn.get("output_tokens")
                    or turn.get("osl_tokens")
                    or turn.get("max_tokens")
                    or 1
                )
                if turns and total_context < previous_context:
                    context_decrease_turns += 1
                if turns and total_context <= previous_context:
                    context_non_growth_turns += 1
                new_prefill = max(1, total_context - previous_context)
                cache_hit_rate = max(0.0, min(1.0, 1.0 - new_prefill / total_context))
                turns.append(
                    TurnSample(
                        turn_index=int(turn.get("turn_idx", idx)),
                        total_context_tokens=total_context,
                        new_prefill_tokens=new_prefill,
                        output_tokens=max(1, output_tokens),
                        cache_hit_rate=cache_hit_rate,
                    )
                )
                previous_context = total_context

            if not turns:
                skipped += 1
                continue
            sessions.append(
                SessionSample(
                    session_id=str(row.get("session_id", len(sessions))),
                    turn_count=len(turns),
                    turns=turns,
                    context_decrease_turns=context_decrease_turns,
                    context_non_growth_turns=context_non_growth_turns,
                    estimated_context_turns=estimated_context_turns,
                )
            )

    return build_distribution_json(
        name=name,
        source_kind="trajectory_jsonl",
        source_path=path,
        sessions=sessions,
        skipped_sessions=skipped,
    )


def build_chat_distribution_from_results(name: str, path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    sessions: list[SessionSample] = []
    skipped = 0
    for idx, row in enumerate(rows):
        config = row.get("config") or {}
        profile = str(config.get("profile") or "")
        if profile not in CHAT_RESULT_PROFILES:
            continue
        per_turn = row.get("perTurn")
        if not per_turn:
            skipped += 1
            continue

        turns: list[TurnSample] = []
        previous_context = 0
        for turn in per_turn:
            total_context = int(round(
                turn.get("median_input_tokens")
                or turn.get("avg_input_tokens")
                or 0
            ))
            if total_context <= 0:
                continue
            output_tokens = int(round(
                turn.get("median_output_tokens")
                or turn.get("avg_output_tokens")
                or 1
            ))
            new_prefill = int(round(
                turn.get("median_new_prefill_tokens")
                or max(1, total_context - previous_context)
            ))
            cache_hit_rate = float(
                turn.get("median_cache_hit_rate")
                if turn.get("median_cache_hit_rate") is not None
                else max(0.0, min(1.0, 1.0 - new_prefill / total_context))
            )
            turns.append(
                TurnSample(
                    turn_index=int(turn.get("turn_index", len(turns))),
                    total_context_tokens=total_context,
                    new_prefill_tokens=max(1, new_prefill),
                    output_tokens=max(1, output_tokens),
                    cache_hit_rate=max(0.0, min(1.0, cache_hit_rate)),
                )
            )
            previous_context = total_context

        if not turns:
            skipped += 1
            continue
        session_id = f"{profile}:{config.get('backend', 'unknown')}:{config.get('concurrency', 'unknown')}:{idx}"
        sessions.append(
            SessionSample(
                session_id=session_id,
                turn_count=len(turns),
                turns=turns,
            )
        )

    if not sessions:
        return None
    return build_distribution_json(
        name=name,
        source_kind="dashboard_per_turn_summary",
        source_path=path,
        sessions=sessions,
        skipped_sessions=skipped,
        note=(
            "ShareGPT raw multi-turn source is not stored locally, so this "
            "artifact is derived from existing per-turn benchmark summaries."
        ),
    )


def build_captured_real_distribution(
    *,
    name: str,
    path: Path,
    min_turns: int,
    max_turns: int,
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    per_request = payload.get("per_request")
    if not isinstance(per_request, list):
        raise ValueError(f"{path} does not contain a per_request list")

    grouped: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    skipped_requests = 0
    for row in per_request:
        if not isinstance(row, dict) or not row.get("success"):
            skipped_requests += 1
            continue
        total_context = int(row.get("input_tokens") or 0)
        if total_context <= 0:
            skipped_requests += 1
            continue
        metadata = row.get("request_metadata") or {}
        source_session_id = metadata.get("source_session_id")
        source_turn_index = metadata.get("source_turn_index", row.get("turn_index"))
        if source_session_id is None or source_turn_index is None:
            skipped_requests += 1
            continue
        turn_index = int(source_turn_index)
        if turn_index >= max_turns:
            continue
        grouped[str(source_session_id)].setdefault(turn_index, row)

    sessions: list[SessionSample] = []
    skipped_sessions = 0
    for source_session_id, rows_by_turn in grouped.items():
        turns: list[TurnSample] = []
        previous_context = 0
        context_decrease_turns = 0
        context_non_growth_turns = 0
        for expected_turn_index in range(max_turns):
            row = rows_by_turn.get(expected_turn_index)
            if row is None:
                break
            total_context = int(row.get("input_tokens") or 0)
            if total_context <= 0:
                break
            raw_new_prefill = row.get("new_prefill_tokens")
            new_prefill = int(raw_new_prefill) if raw_new_prefill else max(1, total_context - previous_context)
            if turns and total_context < previous_context:
                context_decrease_turns += 1
            if turns and total_context <= previous_context:
                context_non_growth_turns += 1
            output_tokens = int(
                row.get("output_tokens")
                or (row.get("request_metadata") or {}).get("planned_output_tokens")
                or 1
            )
            cache_hit_rate = float(
                row.get("cache_hit_rate")
                if row.get("cache_hit_rate") is not None
                else max(0.0, min(1.0, 1.0 - new_prefill / total_context))
            )
            turns.append(
                TurnSample(
                    turn_index=expected_turn_index,
                    total_context_tokens=total_context,
                    new_prefill_tokens=max(1, new_prefill),
                    output_tokens=max(1, output_tokens),
                    cache_hit_rate=max(0.0, min(1.0, cache_hit_rate)),
                    source_session_id=source_session_id,
                    token_source="captured_vllm_input_tokens",
                )
            )
            previous_context = total_context

        if len(turns) < min_turns:
            skipped_sessions += 1
            continue
        sessions.append(
            SessionSample(
                session_id=source_session_id,
                turn_count=len(turns),
                turns=turns,
                context_decrease_turns=context_decrease_turns,
                context_non_growth_turns=context_non_growth_turns,
                estimated_context_turns=0,
            )
        )

    if not sessions:
        raise ValueError(f"No captured REAL sessions in {path} passed min_turns={min_turns}")

    return build_distribution_json(
        name=name,
        source_kind="captured_vllm_real_per_request",
        source_path=path,
        sessions=sessions,
        skipped_sessions=skipped_sessions,
        skipped_requests=skipped_requests,
        token_estimator="captured_vllm_input_tokens",
        note=(
            "Built from successful REAL validation per_request rows. "
            "total_context_tokens uses vLLM-reported input_tokens, preserving "
            "the server tokenizer and chat-template accounting."
        ),
    )


def build_distribution_json(
    *,
    name: str,
    source_kind: str,
    source_path: Path,
    sessions: list[SessionSample],
    skipped_sessions: int,
    skipped_requests: int = 0,
    token_estimator: str = "estimated_tokens = int(word_count * 1.35) when raw input_tokens are absent",
    note: str | None = None,
) -> dict[str, Any]:
    turns = [turn for session in sessions for turn in session.turns]
    by_turn: dict[int, list[TurnSample]] = defaultdict(list)
    for turn in turns:
        by_turn[turn.turn_index].append(turn)
    context_decrease_turns = sum(s.context_decrease_turns for s in sessions)
    context_non_growth_turns = sum(s.context_non_growth_turns for s in sessions)
    estimated_context_turns = sum(s.estimated_context_turns for s in sessions)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "name": name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "kind": source_kind,
            "path": _display_path(source_path),
            "sessions": len(sessions),
            "turns": len(turns),
            "skipped_sessions": skipped_sessions,
            "skipped_requests": skipped_requests,
        },
        "token_estimator": token_estimator,
        "summary": {
            "turn_count": stats([s.turn_count for s in sessions], round_digits=2),
            "total_context_tokens": stats([t.total_context_tokens for t in turns], round_digits=2),
            "new_prefill_tokens": stats([t.new_prefill_tokens for t in turns], round_digits=2),
            "output_tokens": stats([t.output_tokens for t in turns], round_digits=2),
            "cache_hit_rate": stats([t.cache_hit_rate for t in turns], round_digits=4),
        },
        "diagnostics": {
            "context_decrease_turns": context_decrease_turns,
            "context_non_growth_turns": context_non_growth_turns,
            "estimated_context_turns": estimated_context_turns,
            "estimated_context_turn_fraction": (
                round(estimated_context_turns / len(turns), 4) if turns else 0.0
            ),
            "note": (
                "Context deltas are token-count estimates. Non-growth or decreases "
                "mean the source rows may not be literal prefix-growing prompts; "
                "future sampling should treat those deltas as approximate."
            ),
        },
        "histograms": {
            "turn_count": int_histogram(s.turn_count for s in sessions),
        },
        "samples": {
            "turn_count": [s.turn_count for s in sessions],
            "turns": [round_turn_sample(t) for t in turns],
        },
        "by_turn_index": [
            {
                "turn_index": idx,
                "num_samples": len(samples),
                "total_context_tokens": stats([t.total_context_tokens for t in samples], round_digits=2),
                "new_prefill_tokens": stats([t.new_prefill_tokens for t in samples], round_digits=2),
                "output_tokens": stats([t.output_tokens for t in samples], round_digits=2),
                "cache_hit_rate": stats([t.cache_hit_rate for t in samples], round_digits=4),
            }
            for idx, samples in sorted(by_turn.items())
        ],
    }
    if note:
        payload["note"] = note
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for distribution JSON files.",
    )
    parser.add_argument(
        "--dashboard-data",
        type=Path,
        default=ROOT / "dashboard" / "public" / "data.json",
        help="Dashboard data.json used only to derive chat_multiturn when perTurn summaries exist.",
    )
    parser.add_argument(
        "--skip-chat",
        action="store_true",
        help="Do not derive chat_multiturn from dashboard per-turn summaries.",
    )
    parser.add_argument(
        "--skip-trajectory",
        action="store_true",
        help="Do not rebuild the standard trajectory-derived distribution files.",
    )
    parser.add_argument(
        "--captured-real-results-dir",
        type=Path,
        default=None,
        help="Directory containing REAL validation result JSONs used to rebuild short MSE distributions.",
    )
    parser.add_argument(
        "--captured-real-only",
        action="store_true",
        help="Only write captured REAL short MSE distributions.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wrote: list[Path] = []

    if args.captured_real_results_dir is not None:
        for name, spec in CAPTURED_MSE_SHORT_SOURCES.items():
            path = args.captured_real_results_dir / str(spec["filename"])
            if not path.exists():
                print(f"[skip] {name}: missing {path}")
                continue
            payload = build_captured_real_distribution(
                name=name,
                path=path,
                min_turns=int(spec["min_turns"]),
                max_turns=int(spec["max_turns"]),
            )
            out_path = args.out_dir / f"{name}.json"
            write_json(out_path, payload)
            wrote.append(out_path)
            print(
                f"[write] {out_path} "
                f"({payload['source']['sessions']} sessions, {payload['source']['turns']} turns, "
                f"token_source={payload['token_estimator']})"
            )

    if not args.skip_trajectory and not args.captured_real_only:
        for name, path in TRAJECTORY_SOURCES.items():
            if not path.exists():
                print(f"[skip] {name}: missing {path}")
                continue
            payload = build_trajectory_distribution(name, path)
            out_path = args.out_dir / f"{name}.json"
            write_json(out_path, payload)
            wrote.append(out_path)
            print(
                f"[write] {out_path} "
                f"({payload['source']['sessions']} sessions, {payload['source']['turns']} turns)"
            )

    if not args.skip_chat and not args.captured_real_only:
        payload = build_chat_distribution_from_results("chat_multiturn", args.dashboard_data)
        if payload is None:
            print(f"[skip] chat_multiturn: no usable perTurn summaries in {args.dashboard_data}")
        else:
            out_path = args.out_dir / "chat_multiturn.json"
            write_json(out_path, payload)
            wrote.append(out_path)
            print(
                f"[write] {out_path} "
                f"({payload['source']['sessions']} summary rows, {payload['source']['turns']} turns)"
            )

    print(f"[done] wrote {len(wrote)} distribution files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
