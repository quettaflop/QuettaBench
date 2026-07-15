"""Load compact trace distributions for synthetic multi-turn workloads."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DISTRIBUTION_DIR = ROOT / "data" / "distributions"
SUPPORTED_SCHEMA_VERSION = 1


class TraceDistributionError(ValueError):
    """Raised when a distribution artifact is missing or malformed."""


@dataclass(frozen=True)
class TraceTurnSample:
    turn_index: int
    total_context_tokens: int
    new_prefill_tokens: int
    output_tokens: int
    cache_hit_rate: float
    source_session_id: str | None = None
    token_source: str | None = None


@dataclass(frozen=True)
class TraceDistribution:
    name: str
    path: Path
    source: dict[str, Any]
    summary: dict[str, Any]
    diagnostics: dict[str, Any]
    turn_counts: tuple[int, ...]
    turns: tuple[TraceTurnSample, ...]

    @property
    def turns_by_index(self) -> dict[int, tuple[TraceTurnSample, ...]]:
        grouped: dict[int, list[TraceTurnSample]] = {}
        for turn in self.turns:
            grouped.setdefault(turn.turn_index, []).append(turn)
        return {idx: tuple(samples) for idx, samples in grouped.items()}


def distribution_path(name_or_path: str | Path, *, root: Path = DEFAULT_DISTRIBUTION_DIR) -> Path:
    path = Path(name_or_path)
    if path.suffix == ".json" or path.parent != Path("."):
        if path.is_absolute() or path.exists():
            return path
        return ROOT / path
    if path.is_absolute() or path.exists():
        return path
    return root / f"{path.name}.json"


def load_trace_distribution(
    name_or_path: str | Path,
    *,
    root: Path = DEFAULT_DISTRIBUTION_DIR,
) -> TraceDistribution:
    path = distribution_path(name_or_path, root=root)
    if not path.exists():
        raise TraceDistributionError(f"Distribution file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    return parse_trace_distribution(payload, path=path)


def parse_trace_distribution(payload: dict[str, Any], *, path: Path) -> TraceDistribution:
    schema_version = payload.get("schema_version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise TraceDistributionError(
            f"Unsupported distribution schema_version={schema_version!r}; "
            f"expected {SUPPORTED_SCHEMA_VERSION}"
        )

    name = payload.get("name")
    if not isinstance(name, str) or not name:
        raise TraceDistributionError("Distribution must include a non-empty string name")

    samples = payload.get("samples")
    if not isinstance(samples, dict):
        raise TraceDistributionError("Distribution must include a samples object")

    raw_turn_counts = samples.get("turn_count")
    raw_turns = samples.get("turns")
    if not isinstance(raw_turn_counts, list) or not raw_turn_counts:
        raise TraceDistributionError("samples.turn_count must be a non-empty list")
    if not isinstance(raw_turns, list) or not raw_turns:
        raise TraceDistributionError("samples.turns must be a non-empty list")

    turn_counts = tuple(_positive_int(v, "turn_count") for v in raw_turn_counts)
    turns = tuple(_parse_turn_sample(row) for row in raw_turns)

    return TraceDistribution(
        name=name,
        path=path,
        source=dict(payload.get("source") or {}),
        summary=dict(payload.get("summary") or {}),
        diagnostics=dict(payload.get("diagnostics") or {}),
        turn_counts=turn_counts,
        turns=turns,
    )


def load_all_trace_distributions(
    *,
    root: Path = DEFAULT_DISTRIBUTION_DIR,
) -> dict[str, TraceDistribution]:
    if not root.exists():
        return {}
    distributions = {}
    for path in sorted(root.glob("*.json")):
        dist = load_trace_distribution(path)
        distributions[dist.name] = dist
    return distributions


def _parse_turn_sample(row: Any) -> TraceTurnSample:
    if not isinstance(row, dict):
        raise TraceDistributionError("Each turn sample must be an object")
    return TraceTurnSample(
        turn_index=_non_negative_int(row.get("turn_index"), "turn_index"),
        total_context_tokens=_positive_int(row.get("total_context_tokens"), "total_context_tokens"),
        new_prefill_tokens=_positive_int(row.get("new_prefill_tokens"), "new_prefill_tokens"),
        output_tokens=_positive_int(row.get("output_tokens"), "output_tokens"),
        cache_hit_rate=_bounded_float(row.get("cache_hit_rate"), "cache_hit_rate"),
        source_session_id=_optional_str(row.get("source_session_id"), "source_session_id"),
        token_source=_optional_str(row.get("token_source"), "token_source"),
    )


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise TraceDistributionError(f"{field} must be a positive integer")
    return value


def _non_negative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise TraceDistributionError(f"{field} must be a non-negative integer")
    return value


def _bounded_float(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)):
        raise TraceDistributionError(f"{field} must be a number")
    out = float(value)
    if out < 0.0 or out > 1.0:
        raise TraceDistributionError(f"{field} must be in [0, 1]")
    return out


def _optional_str(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TraceDistributionError(f"{field} must be a non-empty string when provided")
    return value
