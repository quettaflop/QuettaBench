# profiling/emit/extract_benchmark_per_request.py
"""Extract per-request OSL/prefill/cache distributions from bench JSONs.

Each multi-turn benchmark JSON at
``/mnt/100g/agent-bench/results/synthetic_distributional/h100_Llama-3.1-8B_tp1_vllm/*.json``
has a ``per_request`` list with one entry per (session, turn).  We emit two
top-level views into the same JSON sidecar:

1. Per-(profile, concurrency, turn_index) distributions for the simple
   uniform-arrival simulator path (legacy keying — back-compat).
2. Per-(profile, concurrency) **session timelines** for the wall-clock
   multi-turn replay simulator.  Each session is an ordered list of turns
   with arrival/completion offsets relative to cohort start.  This is what
   reproduces real client-side inter-turn timing.

Output schema (JSON):

    {
      "per_turn": {
        "<profile>__<concurrency>__<turn_index>": {
          "profile": "<profile>",
          "concurrency": <int>,
          "turn_index": <int>,
          "request_count": <int>,
          "output_tokens": [<int>, ...],
          "new_prefill_tokens": [<int>, ...],
          "cached_context_tokens": [<int>, ...],
        },
        ...
      },
      "session_timelines": {
        "<profile>__<concurrency>": {
          "profile": "<profile>",
          "concurrency": <int>,
          "cohort_t0_ms": <float>,
          "sessions": [
            [  # one session
              {
                "turn_index": <int>,
                "arrival_offset_ms": <float>,
                "completion_offset_ms": <float>,
                "new_prefill_tokens": <int>,
                "cached_context_tokens": <int>,
                "output_tokens": <int>,
              },
              ...
            ],
            ...
          ],
        },
        ...
      }
    }

Only successful requests are included.  Multi-turn bench JSONs at the
non-``_per_turn`` variant carry the per_request field; the ``_per_turn``
files are pre-aggregated and skipped automatically.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _iter_bench_jsons(root: Path) -> Iterable[Path]:
    """Yield raw bench JSONs (skipping ``*_per_turn.json`` aggregates).

    Searches both the top level and one directory deep so callers can pass a
    parent directory containing per-hardware subdirs.
    """
    if not root.exists():
        return
    for path in sorted(root.glob("*.json")):
        if path.name.endswith("_per_turn.json"):
            continue
        yield path


def _to_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _load_bench(path: Path) -> dict | None:
    try:
        with path.open() as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if "per_request" not in data or not data["per_request"]:
        return None
    return data


def _to_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def collect_session_timelines(
    bench_roots: Path | list[Path],
    profile_alias: dict[str, str] | None = None,
) -> dict[str, dict]:
    """Build per-(profile, concurrency) wall-clock session timelines.

    Each session is an ordered list of turns with arrival/completion offsets
    relative to the cohort's earliest dispatch.  Later roots win on key
    collision so paired same-run bench JSONs override broader distributional
    samples.
    """
    if isinstance(bench_roots, Path):
        bench_roots = [bench_roots]
    profile_alias = profile_alias or {}

    out: dict[str, dict] = {}
    for root in bench_roots:
        per_key_rows: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for path in _iter_bench_jsons(root):
            data = _load_bench(path)
            if data is None:
                continue
            config = data.get("config", {}) or {}
            profile = str(config.get("profile", "")).strip()
            profile = profile_alias.get(profile, profile)
            concurrency = _to_int(config.get("concurrency"), 0)
            if not profile or concurrency <= 0:
                continue
            for row in data["per_request"]:
                if not row.get("success"):
                    continue
                if row.get("turn_index") is None:
                    continue
                if row.get("session_id") is None:
                    continue
                if row.get("dispatch_started_at_ms") is None:
                    continue
                per_key_rows[(profile, concurrency)].append(row)

        for (profile, concurrency), rows in per_key_rows.items():
            if not rows:
                continue
            cohort_t0 = min(_to_float(r["dispatch_started_at_ms"]) for r in rows)
            sessions_dict: dict[Any, list[dict]] = defaultdict(list)
            for r in rows:
                turn = {
                    "turn_index": _to_int(r["turn_index"], 0),
                    "arrival_offset_ms": _to_float(r["dispatch_started_at_ms"]) - cohort_t0,
                    "completion_offset_ms": _to_float(r.get("completed_at_ms")) - cohort_t0,
                    "new_prefill_tokens": _to_int(r.get("new_prefill_tokens"), 0),
                    "cached_context_tokens": _to_int(r.get("cached_context_tokens"), 0),
                    "output_tokens": _to_int(r.get("output_tokens"), 0),
                }
                sessions_dict[r["session_id"]].append(turn)
            # Sort each session's turns by turn_index (== arrival order).
            sessions = []
            for sid in sorted(sessions_dict.keys(), key=lambda x: str(x)):
                turns = sorted(sessions_dict[sid], key=lambda t: t["turn_index"])
                sessions.append(turns)
            slug = f"{profile}__{concurrency}"
            out[slug] = {
                "profile": profile,
                "concurrency": concurrency,
                "cohort_t0_ms": cohort_t0,
                "sessions": sessions,
            }
    return out


def collect_per_request_distributions(
    bench_roots: Path | list[Path],
    profile_alias: dict[str, str] | None = None,
) -> dict[str, dict]:
    """Walk one or more bench-JSON dirs, group ``per_request`` by turn.

    When multiple roots are passed, **later roots win on key collision** — so
    pass the most-trusted source last (e.g. same-run paired traces).

    ``profile_alias`` lets capture-time names (``swebench-multiturn``)
    reconcile with predictor-time names (``swebench-multiturn-synth``).
    """
    if isinstance(bench_roots, Path):
        bench_roots = [bench_roots]
    profile_alias = profile_alias or {}

    out: dict[str, dict] = {}
    for root in bench_roots:
        buckets: dict[
            tuple[str, int, int], dict[str, list[int]]
        ] = defaultdict(lambda: {
            "output_tokens": [],
            "new_prefill_tokens": [],
            "cached_context_tokens": [],
        })
        for path in _iter_bench_jsons(root):
            data = _load_bench(path)
            if data is None:
                continue
            config = data.get("config", {}) or {}
            profile = str(config.get("profile", "")).strip()
            profile = profile_alias.get(profile, profile)
            concurrency = _to_int(config.get("concurrency"), 0)
            if not profile or concurrency <= 0:
                continue
            for row in data["per_request"]:
                if not row.get("success"):
                    continue
                turn_index_raw = row.get("turn_index")
                if turn_index_raw is None:
                    continue
                turn_index = _to_int(turn_index_raw, -1)
                if turn_index < 0:
                    continue
                key = (profile, concurrency, turn_index)
                bucket = buckets[key]
                bucket["output_tokens"].append(
                    _to_int(row.get("output_tokens"), 0)
                )
                bucket["new_prefill_tokens"].append(
                    _to_int(row.get("new_prefill_tokens"), 0)
                )
                bucket["cached_context_tokens"].append(
                    _to_int(row.get("cached_context_tokens"), 0)
                )

        for (profile, concurrency, turn_index), bucket in sorted(buckets.items()):
            if not bucket["output_tokens"]:
                continue
            slug = f"{profile}__{concurrency}__{turn_index}"
            out[slug] = {
                "profile": profile,
                "concurrency": concurrency,
                "turn_index": turn_index,
                "request_count": len(bucket["output_tokens"]),
                "output_tokens": bucket["output_tokens"],
                "new_prefill_tokens": bucket["new_prefill_tokens"],
                "cached_context_tokens": bucket["cached_context_tokens"],
            }
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bench-root",
        type=Path,
        action="append",
        default=None,
        help=(
            "Directory of raw bench JSONs to scan.  Repeat to merge "
            "multiple sources; later --bench-root entries win on collision. "
            "Default scans the May-11 distributional set then overlays the "
            "May-21 same-run paired bench JSONs from profile_data/results."
        ),
    )
    p.add_argument(
        "--profile-alias",
        action="append",
        default=[],
        help=(
            "Map capture-time profile name to canonical name "
            "(e.g. swebench-multiturn=swebench-multiturn-synth).  Repeat for "
            "multiple aliases.  Useful because same-run traces use the bare "
            "name while the predictor expects the -synth suffix."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "profile_data/results/benchmark_per_request_llama31_8b_h100_vllm.json"
        ),
        help="Destination JSON sidecar.",
    )
    return p.parse_args()


def _parse_alias_arg(alias_args: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in alias_args:
        if "=" not in entry:
            raise SystemExit(f"--profile-alias expects from=to, got {entry!r}")
        src, dst = entry.split("=", 1)
        out[src.strip()] = dst.strip()
    return out


def main() -> None:
    args = parse_args()
    bench_roots = args.bench_root or [
        # Broader distributional set (older runs — fallback for keys not in
        # the paired same-run set).
        Path(
            "/mnt/100g/agent-bench/results/synthetic_distributional/"
            "h100_Llama-3.1-8B_tp1_vllm"
        ),
        # Same-run paired bench JSONs captured alongside the engine traces —
        # these win on key collision because they're the ground-truth match.
        Path("profile_data/results"),
    ]
    # If no aliases supplied, default to mapping the capture-time profile names
    # (without the ``-synth`` suffix) to the predictor's canonical names.
    aliases = _parse_alias_arg(args.profile_alias) or {
        "swebench-multiturn": "swebench-multiturn-synth",
        "terminalbench-multiturn": "terminalbench-multiturn-synth",
        "chat-multiturn": "chat-multiturn-synth",
        "osworld-multiturn": "osworld-multiturn-synth",
    }
    distributions = collect_per_request_distributions(
        bench_roots, profile_alias=aliases
    )
    if not distributions:
        raise SystemExit(
            f"no per_request entries found under {bench_roots}"
        )
    session_timelines = collect_session_timelines(
        bench_roots, profile_alias=aliases
    )
    payload = {
        "per_turn": distributions,
        "session_timelines": session_timelines,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(payload, handle)
    print(
        f"Wrote {args.output} with {len(distributions)} per-turn buckets "
        f"and {len(session_timelines)} session-timeline (profile, c) groups"
    )


if __name__ == "__main__":
    main()
