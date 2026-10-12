"""Multi-turn traces: sessions of ordered turns built from ShareGPT conversations or agent trajectories, plus the per-turn wait policy."""

import asyncio
import itertools
import json
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from .live import RealTrace, Request, est_tokens, load_sharegpt


@dataclass(frozen=True)
class TurnPacing:
    """per-session runs a session's turns back to back; interleaved waits at a barrier after every turn index. Tool and human waits follow each turn, plus recorded gaps when use_recorded_waits; use_recorded_arrivals starts each session at its recorded arrival."""

    pacing: str = "per-session"
    tool_wait_ms: float = 0.0
    human_wait_ms: float = 0.0
    use_recorded_waits: bool = False
    use_recorded_arrivals: bool = False


class SessionTrace(RealTrace):
    """A RealTrace whose requests are turns; sessions() groups them by session_id in first-appearance order, each ordered by turn.

    replies maps a session id to its assistant placeholder dicts, which later turns' messages share; empty when recorded replies are kept."""

    pacing = TurnPacing()
    replies: Mapping = MappingProxyType({})

    def sessions(self):
        groups = {}
        for r in self.requests():
            groups.setdefault(r.session_id, []).append(r)
        return [sorted(turns, key=lambda r: (r.turn is None, r.turn or 0, r.order))
                for turns in groups.values()]

    def source_meta(self):
        return {**super().source_meta(), "sessions": len(self.sessions()),
                "pacing": self.pacing.pacing}


def turn_wait_metadata(turn: dict) -> dict:
    """The inter-turn wait fields of a trajectory turn (tool_wait_after_ms, human_wait_after_ms), accepting the `*_ms` and `*_wait_ms` aliases; missing is 0."""
    def _ms(*keys: str) -> float:
        for key in keys:
            if key in turn and turn[key] is not None:
                try:
                    return max(0.0, float(turn[key]))
                except (TypeError, ValueError):
                    pass
        return 0.0

    return {
        "tool_wait_after_ms": _ms("tool_wait_after_ms", "tool_ms", "tool_wait_ms"),
        "human_wait_after_ms": _ms(
            "human_wait_after_ms", "human_ms", "human_wait_ms"
        ),
    }


def session_arrival_ms(entry: dict) -> float:
    """Session start offset in milliseconds; missing or invalid is 0."""
    for key in ("arrival_time_ms", "arrival_time"):
        if key in entry and entry[key] is not None:
            try:
                return max(0.0, float(entry[key]))
            except (TypeError, ValueError):
                pass
    return 0.0


@dataclass
class MultiTurnSession:
    """One conversation as Requests with growing history; assistant_messages are the dicts later turns share, empty for a trajectory."""

    session_id: int
    turns: list
    assistant_messages: list = field(default_factory=list)
    arrival_time_ms: float = 0.0


class LoadedSessionTrace(SessionTrace):
    """Sessions shuffled with the seed and cut to num_sessions, turns up to max_turn_index, turn 0 carrying the recorded arrival. max_turns is the longest session before the cut; warmup draws cycle the sessions turn-major."""

    def __init__(self, sessions, path, reader, seed=42, num_sessions=100, warmup=0,
                 max_turn_index=None, pacing=None, profile=None):
        random.Random(seed).shuffle(sessions)
        sessions = sessions[:num_sessions]
        if not sessions:
            raise ValueError("No multi-turn sessions loaded -- check ShareGPT dataset and filter bounds")
        self.max_turns = max(len(s.turns) for s in sessions)
        print(f"Loaded {len(sessions)} sessions, max {self.max_turns} turns per session")
        reqs = []
        for s in sessions:
            for t, req in enumerate(s.turns):
                if max_turn_index is not None and t > max_turn_index:
                    break
                arrival = s.arrival_time_ms / 1000.0 if t == 0 else None
                reqs.append(replace(req, order=len(reqs), session_id=s.session_id, turn=t,
                                    arrival_ts=arrival))
        if profile is not None:
            path, reader = profile.file_path or profile.name, f"profile:{profile.dataset}"
        super().__init__(reqs, path, reader)
        self.profile = profile
        self.pacing = pacing or TurnPacing()
        self.replies = {s.session_id: s.assistant_messages for s in sessions}
        flat = [s.turns[t] for t in range(self.max_turns) for s in sessions if t < len(s.turns)]
        self._warmup = [replace(r, order=i) for i, r in zip(range(warmup), itertools.cycle(flat))]

    def warmup_requests(self, n):
        return self._warmup[:n]

    def source_meta(self):
        if self.profile is None:
            return super().source_meta()
        p = self.profile
        return {"profile": p.name, "dataset": p.dataset, "path": p.file_path,
                "reqs": len(self.requests()), "sessions": len(self.replies),
                "pacing": self.pacing.pacing}


class ShareGPTSessionTrace(LoadedSessionTrace):
    """ShareGPT conversations as growing-history turns, loading up to 3x num_sessions.

    Pairs stop at the first reply outside the output length bounds and turns at the first that overflows max_isl_tokens; conversations under min_turns are skipped, longer ones cut at max_turns."""

    def __init__(self, min_turns=3, max_turns=10, num_sessions=200, seed=42,
                 system_prompt="You are a helpful assistant.", max_isl_tokens=8192,
                 max_osl_tokens=2048, min_osl_tokens=50, **trace_kw):
        sessions = []
        for item in load_sharegpt():
            convs = item["conversations"]
            if len(convs) < min_turns * 2:
                continue
            pairs, human_msg = [], None
            for turn in convs:
                if turn.get("from") == "human":
                    human_msg = turn.get("value", "")
                elif turn.get("from") == "gpt" and human_msg is not None:
                    assistant_msg = turn.get("value", "")
                    osl_est = est_tokens(assistant_msg)
                    if osl_est > max_osl_tokens or osl_est < min_osl_tokens:
                        break
                    pairs.append((human_msg, assistant_msg, osl_est))
                    human_msg = None
            if len(pairs) < min_turns:
                continue
            turns, assistant_messages = [], []
            history = [{"role": "system", "content": system_prompt}] if system_prompt else []
            total = 0
            for human_msg, assistant_msg, osl_est in pairs[:max_turns]:
                history.append({"role": "user", "content": human_msg})
                total += est_tokens(human_msg)
                if total > max_isl_tokens:
                    break
                turns.append(Request([], 0, output_length=osl_est, messages=list(history),
                                     metadata={}))
                assistant_message = {"role": "assistant", "content": assistant_msg}
                history.append(assistant_message)
                assistant_messages.append(assistant_message)
                total += osl_est
            if len(turns) < min_turns:
                continue
            sessions.append(MultiTurnSession(len(sessions), turns, assistant_messages))
            if len(sessions) >= num_sessions * 3:
                break
        super().__init__(sessions, "sharegpt", "sharegpt-multi-turn", seed=seed,
                         num_sessions=num_sessions, **trace_kw)


class TrajectorySessionTrace(LoadedSessionTrace):
    """Agent trajectories, one session per JSONL line: {"session_id", "turns": [{"messages", "osl_tokens"}, ...]}.

    Sessions under min_turns are skipped, longer ones cut at max_turns; a session stops at the first turn whose estimated prompt exceeds max_isl_tokens less its output and the safety margin."""

    def __init__(self, path, min_turns=3, max_turns=30, num_sessions=100, seed=42,
                 max_isl_tokens=131072, max_osl_tokens=2000, context_safety_margin_tokens=0,
                 **trace_kw):
        sessions = []
        with open(path) as f:
            lines = [json.loads(line) for line in (raw.strip() for raw in f) if line]
        for entry in lines:
            raw_turns = entry.get("turns", [])
            if len(raw_turns) < min_turns:
                continue
            turns = []
            for t in raw_turns[:max_turns]:
                messages = t["messages"]
                osl = min(t.get("osl_tokens", 500), max_osl_tokens)
                isl_est = sum(est_tokens(m.get("content", "")) for m in messages)
                prompt_budget = max_isl_tokens - osl - context_safety_margin_tokens
                if prompt_budget < 1 or isl_est > prompt_budget:
                    break
                turns.append(Request([], 0, output_length=osl, messages=messages, metadata={
                    "source_session_id": entry.get("session_id"),
                    "source_turn_index": t.get("turn_idx", t.get("turn_index", len(turns))),
                    "trace_content_source": entry.get("source"),
                    "planned_total_context_tokens": isl_est,
                    "planned_output_tokens": osl,
                    "planned_total_with_output_tokens": isl_est + osl,
                    "context_window_tokens": max_isl_tokens,
                    "context_safety_margin_tokens": context_safety_margin_tokens,
                    "prompt_token_budget": prompt_budget,
                    "truncated_by_context_limit": False,
                    **turn_wait_metadata(t),
                }))
            if len(turns) < min_turns:
                continue
            sessions.append(MultiTurnSession(len(sessions), turns,
                                             arrival_time_ms=session_arrival_ms(entry)))
        super().__init__(sessions, path, "trajectory", seed=seed, num_sessions=num_sessions,
                         **trace_kw)


def inter_turn_wait_s(
    request,
    *,
    tool_wait_ms: float = 0.0,
    human_wait_ms: float = 0.0,
    use_recorded_waits: bool = False,
) -> float:
    """Seconds to sleep after a turn: the fixed tool and human waits, plus the recorded waits in the turn metadata when use_recorded_waits."""
    recorded_tool = 0.0
    recorded_human = 0.0
    if use_recorded_waits and request is not None:
        metadata = getattr(request, "metadata", None) or {}
        recorded_tool = float(metadata.get("tool_wait_after_ms") or 0.0)
        recorded_human = float(metadata.get("human_wait_after_ms") or 0.0)
    total_ms = tool_wait_ms + human_wait_ms + recorded_tool + recorded_human
    return max(0.0, total_ms / 1000.0)


async def _sleep_until_elapsed(benchmark_start: float, target_s: float) -> None:
    delay = target_s - (time.perf_counter() - benchmark_start)
    if delay > 0:
        await asyncio.sleep(delay)


def resolve_multi_turn_num_sessions(
    profile,
    concurrency: int,
    override: int | None = None,
) -> tuple[int, str]:
    """Multi-turn runs need enough sessions to saturate the requested concurrency."""
    if profile.mode != "multi-turn":
        return profile.num_sessions, "profile_default"

    if override is not None:
        effective_num_sessions = max(override, concurrency)
        if effective_num_sessions == override:
            return effective_num_sessions, "cli"
        return effective_num_sessions, "cli_concurrency_floor"

    effective_num_sessions = max(profile.num_sessions, concurrency)
    if effective_num_sessions == profile.num_sessions:
        return effective_num_sessions, "profile_default"
    return effective_num_sessions, "concurrency_floor"
