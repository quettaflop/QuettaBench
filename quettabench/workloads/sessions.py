"""Multi-turn traces: sessions of ordered turns, plus the per-turn wait policy."""

import asyncio
import collections
import hashlib
import itertools
import json
import os
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from .base import sha256_file
from .live import RealTrace, Request, est_tokens, load_sharegpt


@dataclass(frozen=True)
class TurnPacing:
    """per-session runs turns back to back; interleaved waits after every turn index."""

    pacing: str = "per-session"
    tool_wait_ms: float = 0.0
    human_wait_ms: float = 0.0
    use_recorded_waits: bool = False
    use_recorded_arrivals: bool = False


class SessionTrace(RealTrace):
    """Requests are turns; sessions() groups them by session_id."""

    pacing = TurnPacing()
    replies: Mapping = MappingProxyType({})

    def sessions(self):
        groups: dict[object, list[Request]] = {}
        for r in self.requests():
            groups.setdefault(r.session_id, []).append(r)
        return [sorted(turns, key=lambda r: (r.turn is None, r.turn or 0, r.order)) for turns in groups.values()]

    def source_meta(self):
        return {**super().source_meta(), "sessions": len(self.sessions()), "pacing": self.pacing.pacing}


def turn_wait_metadata(turn: dict) -> dict:
    """Accepts the `*_ms` and `*_wait_ms` aliases; missing is 0."""

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
        "human_wait_after_ms": _ms("human_wait_after_ms", "human_ms", "human_wait_ms"),
    }


def session_arrival_ms(entry: dict) -> float:
    """Missing or invalid is 0."""
    for key in ("arrival_time_ms", "arrival_time"):
        if key in entry and entry[key] is not None:
            try:
                return max(0.0, float(entry[key]))
            except (TypeError, ValueError):
                pass
    return 0.0


@dataclass
class MultiTurnSession:
    """assistant_messages are the dicts later turns share; empty for a trajectory."""

    session_id: int
    turns: list
    assistant_messages: list = field(default_factory=list)
    arrival_time_ms: float = 0.0


def quartile(values: list, v) -> int:
    """1 to 4: which quarter of values v falls in."""
    ordered = sorted(values)
    cuts = [ordered[min(len(ordered) - 1, int(q * len(ordered)))] for q in (0.25, 0.5, 0.75)]
    return 1 + sum(v > c for c in cuts)


def prompt_size(session) -> int:
    last = session.turns[-1]
    return int((last.metadata or {}).get("planned_total_context_tokens") or last.prompt_len or 0)


def stratify(sessions: list, take: int | None) -> tuple[list, dict]:
    """Order sessions round-robin across turn-count and prompt-size quartile strata; with take, each stratum gives
    its share. Returns the order and each session's stratum."""
    turns, sizes = [len(s.turns) for s in sessions], [prompt_size(s) for s in sessions]
    strata: dict[str, list] = {}
    for s, t, z in zip(sessions, turns, sizes):
        strata.setdefault(f"turns_q{quartile(turns, t)}.prompt_q{quartile(sizes, z)}", []).append(s)
    if take is not None:
        shares = {k: take * len(v) / len(sessions) for k, v in strata.items()}
        counts = {k: int(x) for k, x in shares.items()}
        for k in sorted(shares, key=lambda k: shares[k] - counts[k], reverse=True)[: take - sum(counts.values())]:
            counts[k] += 1
        strata = {k: v[: counts[k]] for k, v in strata.items() if counts[k]}
    order = [s for group in itertools.zip_longest(*strata.values()) for s in group if s is not None]
    return order, {s.session_id: k for k, v in strata.items() for s in v}


# How play() salts a play under each workload.salt; part of a recycled run's plan hash, so change one when its
# salt changes.
PLAY_RULES = {
    "text": "rid-v1: [rid:<sha256(seed:session:play)[:12]>] and a blank line before the first user message",
    "cache_salt": "rid-v1: vLLM's cache_salt field set to sha256(seed:session:play)[:12]; the prompt as recorded",
}


def play(turns: list, seed: int, number: int, salt: str = "text") -> list:
    """One salted play of a session for a recycled replay, so the play misses the prefix cache its earlier plays
    left. salt "text" puts a [rid:...] tag before the first user message, as AgentX does; "cache_salt" leaves the
    prompt as recorded and sets vLLM's cache_salt field on every turn. session_id becomes "<session>#<number>"."""
    tag = hashlib.sha256(f"{seed}:{turns[0].session_id}:{number}".encode()).hexdigest()[:12]
    sid = f"{turns[0].session_id}#{number}"
    out = []
    for r in turns:
        messages = [dict(m) for m in r.messages or []]
        for m in messages if salt == "text" else ():
            if m.get("role") == "user":
                m["content"] = f"[rid:{tag}]\n\n{m.get('content') or ''}"
                break
        out.append(
            replace(
                r,
                session_id=sid,
                messages=messages,
                metadata=dict(r.metadata or {}, play=number),
                cache_salt=tag if salt == "cache_salt" else None,
            )
        )
    return out


def sample_digest(ids):
    """Names the sample: the sessions replayed, in order."""
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()[:16]


class LoadedSessionTrace(SessionTrace):
    """Turn 0 carries the recorded arrival. The sample is every eligible session, or the first num_sessions of a
    seeded shuffle; it is the same at every concurrency, and a file with fewer sessions than asked for is refused."""

    def __init__(
        self,
        sessions,
        path,
        reader,
        seed=42,
        num_sessions=None,
        warmup=0,
        max_turn_index=None,
        pacing=None,
        profile=None,
        stratified=False,
    ):
        # The replayed file's digest, the one data/MANIFEST.json lists; None when the source is not a file.
        self.file_sha256 = sha256_file(path) if isinstance(path, str) and os.path.isfile(path) else None
        random.Random(seed).shuffle(sessions)
        if num_sessions is not None and len(sessions) < num_sessions:
            raise ValueError(
                f"{path}: {len(sessions)} eligible sessions, fewer than the {num_sessions} asked for; "
                f"ask for at most {len(sessions)} so every point replays the same sample"
            )
        # session_id -> stratum, for a stratified sample; the summary counts sessions and plays by it.
        self.stratum_of: dict | None = None
        if stratified and sessions:
            sessions, self.stratum_of = stratify(sessions, num_sessions)
        elif num_sessions is not None:
            sessions = sessions[:num_sessions]
        if not sessions:
            raise ValueError("No multi-turn sessions loaded -- check ShareGPT dataset and filter bounds")
        self.sample = [
            str(s.turns[0].metadata.get("source_session_id", s.session_id) if s.turns[0].metadata else s.session_id)
            for s in sessions
        ]
        self.live_replies = any(s.assistant_messages for s in sessions)
        self.max_turns = max(len(s.turns) for s in sessions)
        print(f"Loaded {len(sessions)} sessions, max {self.max_turns} turns per session")
        reqs: list[Request] = []
        for s in sessions:
            for t, req in enumerate(s.turns):
                if max_turn_index is not None and t > max_turn_index:
                    break
                arrival = s.arrival_time_ms / 1000.0 if t == 0 else None
                reqs.append(replace(req, order=len(reqs), session_id=s.session_id, turn=t, arrival_ts=arrival))
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
            return {**super().source_meta(), "file_sha256": self.file_sha256}
        p = self.profile
        return {
            "profile": p.name,
            "dataset": p.dataset,
            "path": p.file_path,
            "file_sha256": self.file_sha256,
            "reqs": len(self.requests()),
            "sessions": len(self.replies),
            "sample_digest": sample_digest(self.sample),
            "strata": dict(collections.Counter(self.stratum_of.values())) if self.stratum_of else None,
            "pacing": self.pacing.pacing,
        }


class ShareGPTSessionTrace(LoadedSessionTrace):
    """Loads up to 3x num_sessions."""

    def __init__(
        self,
        min_turns=3,
        max_turns=10,
        num_sessions=200,
        seed=42,
        system_prompt="You are a helpful assistant.",
        max_isl_tokens=8192,
        max_osl_tokens=2048,
        min_osl_tokens=50,
        **trace_kw,
    ):
        sessions: list[MultiTurnSession] = []
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
                turns.append(Request([], 0, output_length=osl_est, messages=list(history), metadata={}))
                assistant_message = {"role": "assistant", "content": assistant_msg}
                history.append(assistant_message)
                assistant_messages.append(assistant_message)
                total += osl_est
            if len(turns) < min_turns:
                continue
            sessions.append(MultiTurnSession(len(sessions), turns, assistant_messages))
            if num_sessions is not None and len(sessions) >= num_sessions * 3:
                break
        super().__init__(sessions, "sharegpt", "sharegpt-multi-turn", seed=seed, num_sessions=num_sessions, **trace_kw)


def recorded_counts(path, entry, turn, index):
    """The prompt and completion token counts the engine reported when the turn was recorded."""
    try:
        return int(turn["prompt_tokens"]), int(turn["osl_tokens"])
    except KeyError as e:
        raise ValueError(
            f"{path}: session {entry.get('session_id')} turn {index} has no recorded {e.args[0]}; "
            "a replay file carries the counts the serving engine reported, never an estimate"
        ) from None


class TrajectorySessionTrace(LoadedSessionTrace):
    """Sessions from a trajectory JSONL: {"session_id", "tools"?, "turns": [{"messages", "prompt_tokens", "osl_tokens"}]}.
    A session is cut at the first turn past the window or the output cap; the last turn kept says so in
    truncated_by_context_limit."""

    def __init__(
        self,
        path,
        min_turns=3,
        max_turns=30,
        num_sessions=100,
        seed=42,
        max_isl_tokens=131072,
        max_osl_tokens=2000,
        context_safety_margin_tokens=0,
        **trace_kw,
    ):
        sessions: list[MultiTurnSession] = []
        cut_sessions = dropped = no_waits = 0
        pacing = trace_kw.get("pacing")
        waits_on = pacing is not None and pacing.use_recorded_waits
        with open(path) as f:
            lines = [json.loads(line) for line in (raw.strip() for raw in f) if line]
        for entry in lines:
            # The builder marks a session whose waits it could not recover; under recorded waits it would run
            # back to back while the run says waits were on.
            if waits_on and entry.get("recorded_waits") is False:
                no_waits += 1
                continue
            raw_turns = entry.get("turns", [])
            if len(raw_turns) < min_turns:
                dropped += 1
                continue
            turns: list[Request] = []
            cut = False
            last: dict = {}
            for t in raw_turns[:max_turns]:
                prompt_tokens, osl = recorded_counts(path, entry, t, len(turns))
                prompt_budget = max_isl_tokens - osl - context_safety_margin_tokens
                if osl > max_osl_tokens or prompt_budget < 1 or prompt_tokens > prompt_budget:
                    cut = True
                    break
                last = {
                    "source_session_id": entry.get("session_id"),
                    "source_turn_index": t.get("turn_idx", t.get("turn_index", len(turns))),
                    "trace_content_source": entry.get("source"),
                    "planned_total_context_tokens": prompt_tokens,
                    "planned_output_tokens": osl,
                    "planned_total_with_output_tokens": prompt_tokens + osl,
                    "context_window_tokens": max_isl_tokens,
                    "context_safety_margin_tokens": context_safety_margin_tokens,
                    "prompt_token_budget": prompt_budget,
                    "truncated_by_context_limit": False,
                    **turn_wait_metadata(t),
                }
                tools = t.get("tools", entry.get("tools"))
                turns.append(Request([], 0, output_length=osl, messages=t["messages"], tools=tools, metadata=last))
            if cut and turns:
                last["truncated_by_context_limit"] = True
            if len(turns) < min_turns:
                dropped += 1
                continue
            cut_sessions += cut
            sessions.append(MultiTurnSession(len(sessions), turns, arrival_time_ms=session_arrival_ms(entry)))
        self.sessions_cut, self.sessions_dropped, self.sessions_dropped_no_waits = cut_sessions, dropped, no_waits
        super().__init__(sessions, path, "trajectory", seed=seed, num_sessions=num_sessions, **trace_kw)

    def source_meta(self):
        return {
            **super().source_meta(),
            "sessions_cut": self.sessions_cut,
            "sessions_dropped": self.sessions_dropped,
            "sessions_dropped_no_waits": self.sessions_dropped_no_waits,
        }


def inter_turn_wait_s(
    request,
    *,
    tool_wait_ms: float = 0.0,
    human_wait_ms: float = 0.0,
    use_recorded_waits: bool = False,
) -> float:
    """Plus the recorded waits in the turn metadata when use_recorded_waits."""
    recorded_tool = 0.0
    recorded_human = 0.0
    if use_recorded_waits and request is not None:
        metadata = getattr(request, "metadata", None) or {}
        recorded_tool = float(metadata.get("tool_wait_after_ms") or 0.0)
        recorded_human = float(metadata.get("human_wait_after_ms") or 0.0)
    total_ms = tool_wait_ms + human_wait_ms + recorded_tool + recorded_human
    return max(0.0, total_ms / 1000.0)


async def sleep_until_elapsed(benchmark_start: float, target_s: float) -> None:
    delay = target_s - (time.perf_counter() - benchmark_start)
    if delay > 0:
        await asyncio.sleep(delay)


def resolve_multi_turn_num_sessions(
    profile,
    override: int | None = None,
) -> tuple[int | None, str]:
    """The sample size is fixed before the sweep: the run's override, else the profile's; None is every session."""
    if override is not None:
        return override, "cli"
    return profile.num_sessions, "all" if profile.num_sessions is None else "profile"
