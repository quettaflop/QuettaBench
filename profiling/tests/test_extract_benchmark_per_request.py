# profiling/tests/test_extract_benchmark_per_request.py
from __future__ import annotations

import json
from pathlib import Path

from profiling.emit.extract_benchmark_per_request import (
    collect_per_request_distributions,
)


def _write_bench(path: Path, profile: str, concurrency: int, rows: list[dict]) -> None:
    payload = {
        "config": {"profile": profile, "concurrency": concurrency},
        "per_request": rows,
    }
    path.write_text(json.dumps(payload))


def test_collect_groups_per_request_by_turn_index(tmp_path: Path) -> None:
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    _write_bench(
        bench_dir / "chat-multiturn-synth_conc4.json",
        profile="chat-multiturn-synth",
        concurrency=4,
        rows=[
            # session 0, two turns
            {"success": True, "turn_index": 0, "output_tokens": 10, "new_prefill_tokens": 100, "cached_context_tokens": 0},
            {"success": True, "turn_index": 1, "output_tokens": 20, "new_prefill_tokens": 50, "cached_context_tokens": 100},
            # session 1, two turns
            {"success": True, "turn_index": 0, "output_tokens": 30, "new_prefill_tokens": 110, "cached_context_tokens": 0},
            {"success": True, "turn_index": 1, "output_tokens": 40, "new_prefill_tokens": 55, "cached_context_tokens": 110},
            # session 2, turn 0 only (request that failed turn 1)
            {"success": True, "turn_index": 0, "output_tokens": 50, "new_prefill_tokens": 120, "cached_context_tokens": 0},
            {"success": False, "turn_index": 1, "output_tokens": 0, "new_prefill_tokens": 0, "cached_context_tokens": 0},
        ],
    )

    out = collect_per_request_distributions(bench_dir)

    assert set(out.keys()) == {
        "chat-multiturn-synth__4__0",
        "chat-multiturn-synth__4__1",
    }
    t0 = out["chat-multiturn-synth__4__0"]
    assert t0["request_count"] == 3
    assert sorted(t0["output_tokens"]) == [10, 30, 50]
    assert sorted(t0["new_prefill_tokens"]) == [100, 110, 120]

    t1 = out["chat-multiturn-synth__4__1"]
    assert t1["request_count"] == 2  # failed row dropped
    assert sorted(t1["output_tokens"]) == [20, 40]
    assert sorted(t1["cached_context_tokens"]) == [100, 110]


def test_collect_skips_per_turn_aggregates(tmp_path: Path) -> None:
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    # Aggregate file — should be skipped even if it has a per_request field.
    (bench_dir / "chat_conc1_per_turn.json").write_text(
        json.dumps({"config": {"profile": "x", "concurrency": 1}, "per_request": [
            {"success": True, "turn_index": 0, "output_tokens": 999, "new_prefill_tokens": 0, "cached_context_tokens": 0},
        ]})
    )
    out = collect_per_request_distributions(bench_dir)
    assert out == {}


def test_collect_skips_files_without_per_request(tmp_path: Path) -> None:
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    (bench_dir / "summary_only.json").write_text(
        json.dumps({"config": {"profile": "x", "concurrency": 1}, "summary": {"x": 1}})
    )
    out = collect_per_request_distributions(bench_dir)
    assert out == {}


def test_collect_later_root_overlays_earlier_on_key_collision(tmp_path: Path) -> None:
    # Earlier root: distributional bench from a stale run (long OSL tail).
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    _write_bench(
        earlier / "swebench_conc2.json",
        profile="swebench-multiturn",
        concurrency=2,
        rows=[
            {"success": True, "turn_index": 0, "output_tokens": 999, "new_prefill_tokens": 100, "cached_context_tokens": 0},
            {"success": True, "turn_index": 0, "output_tokens": 888, "new_prefill_tokens": 100, "cached_context_tokens": 0},
        ],
    )
    # Later root: paired same-run trace bench (the source of truth).
    later = tmp_path / "later"
    later.mkdir()
    _write_bench(
        later / "benchmark_serving_swe_c2.json",
        profile="swebench-multiturn",
        concurrency=2,
        rows=[
            {"success": True, "turn_index": 0, "output_tokens": 10, "new_prefill_tokens": 100, "cached_context_tokens": 0},
            {"success": True, "turn_index": 0, "output_tokens": 20, "new_prefill_tokens": 100, "cached_context_tokens": 0},
        ],
    )
    out = collect_per_request_distributions([earlier, later])
    assert out["swebench-multiturn__2__0"]["output_tokens"] == [10, 20]


def test_collect_applies_profile_alias(tmp_path: Path) -> None:
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    _write_bench(
        bench_dir / "swe.json",
        profile="swebench-multiturn",
        concurrency=4,
        rows=[
            {"success": True, "turn_index": 0, "output_tokens": 5, "new_prefill_tokens": 8, "cached_context_tokens": 0},
        ],
    )
    out = collect_per_request_distributions(
        [bench_dir],
        profile_alias={"swebench-multiturn": "swebench-multiturn-synth"},
    )
    assert "swebench-multiturn-synth__4__0" in out
    assert "swebench-multiturn__4__0" not in out
