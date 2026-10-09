"""Multi-turn traces: sessions of ordered turns and how the next turn waits.

SessionTrace groups a RealTrace's requests into sessions and carries the
per-turn pacing policy as data. The dispatch that feeds engine replies into
later turns stays in modes/serve.py.
"""

import asyncio
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .live import RealTrace


@dataclass(frozen=True)
class TurnPacing:
    """per-session runs a session's turns back to back; interleaved puts a
    barrier after every turn index across sessions. Waits are added after
    each turn: fixed tool and human gaps, plus each turn's recorded gaps when
    use_recorded_waits. use_recorded_arrivals starts each session at its
    recorded arrival instead of t=0 or a generated schedule."""

    pacing: str = "per-session"
    tool_wait_ms: float = 0.0
    human_wait_ms: float = 0.0
    use_recorded_waits: bool = False
    use_recorded_arrivals: bool = False


class SessionTrace(RealTrace):
    """A RealTrace whose requests are turns: sessions() groups them by
    session_id in first-appearance order, each ordered by turn.

    replies maps a session id to its assistant placeholder dicts: serve
    writes each turn's engine reply into them, and later turns' messages
    hold the same dicts. Empty for a trace that keeps its recorded replies.
    """

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


def inter_turn_wait_s(
    request,
    *,
    tool_wait_ms: float = 0.0,
    human_wait_ms: float = 0.0,
    use_recorded_waits: bool = False,
) -> float:
    """Seconds to sleep after a turn before the next turn in the same session.

    Defaults are zero (auto-mode: no human approval pause, ignore recorded
    tool gaps). ``--use-recorded-waits`` adds ``tool_wait_after_ms`` and
    ``human_wait_after_ms`` from the turn metadata when present.
    """
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


def _load_source_session_ids(path: str | None) -> list[str] | None:
    if not path:
        return None
    source_path = Path(path)
    ids = []
    seen = set()
    for line in source_path.read_text(encoding="utf-8").splitlines():
        source_session_id = line.strip()
        if not source_session_id or source_session_id.startswith("#"):
            continue
        if source_session_id in seen:
            continue
        ids.append(source_session_id)
        seen.add(source_session_id)
    if not ids:
        print(f"Error: --source-session-ids-file had no usable IDs: {path}")
        sys.exit(1)
    return ids


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
