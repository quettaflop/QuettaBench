# profiling/tests/test_build_serving_decode_grid.py
"""Deterministic-side tests for the serving-context decode grid tool (L11).

Covers the PRE-REGISTERED arithmetic that must not drift: the 26-cell tp4 lattice and its
prompt/osl fixed point + pool cap, the steady-window / per-request-p50 summary, the
validation flags, and the builder's raw->CSV merge (latest run wins per cell).
The live-SSE side runs only on the GPU host (aiohttp lazily imported there).
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from profiling.emit.build_serving_decode_grid import (  # noqa: E402
    OUT_FIELDS, _client, build,
)

solve_cell = _client.solve_cell
lattice_cells = _client.lattice_cells
summarize_cell = _client.summarize_cell
LATTICE_TP4 = _client.LATTICE_TP4
KV_POOL = _client.KV_POOL_TOKENS_TP4


def test_lattice_is_the_26_preregistered_cells() -> None:
    cells = lattice_cells()
    assert len(cells) == 26
    keys = {(c["batch_size"], c["nominal_T"]) for c in cells}
    assert {(b, t) for b in (1, 8, 32, 80) for t in (512, 2048, 8192, 16384)} <= keys
    assert {(160, t) for t in (512, 2048, 8192, 12288)} <= keys
    assert {(256, 512), (256, 2048), (256, 6144)} <= keys
    assert {(320, 512), (320, 2048), (320, 4096)} <= keys


def test_lattice_cap_rule() -> None:
    # every cell within 0.95*pool except the directive-named (160, 12288) at 95.2%
    for c in lattice_cells():
        if (c["batch_size"], c["nominal_T"]) == (160, 12288):
            assert c["exceeds_cap"] and c["kv_frac_of_pool"] < 0.96
            assert c["kv_final_tokens"] < KV_POOL  # still inside the LIVE pool: no preemption
        else:
            assert not c["exceeds_cap"], c


def test_solve_cell_fixed_point() -> None:
    for b in LATTICE_TP4:
        for t in LATTICE_TP4[b]:
            prompt, osl = solve_cell(b, t)
            assert prompt == t - osl // 2
            assert osl == 384 + math.ceil(b * prompt / 8192)
    prompt, osl = solve_cell(1, 512)
    assert osl == 385 and prompt == 512 - 385 // 2


def test_prompt_token_ids_deterministic_and_unique() -> None:
    a = _client.prompt_token_ids(320, 2048, 0, 1820)
    b = _client.prompt_token_ids(320, 2048, 0, 1820)
    c = _client.prompt_token_ids(320, 2048, 1, 1820)
    assert a == b and a != c and len(a) == 1820
    assert all(_client.TOKEN_ID_LO <= t < _client.TOKEN_ID_HI for t in a)


def _mk_record(req: int, first_wall: float, n_events: int, itl_ms: float,
               prompt: int = 1820, osl: int = 456, nominal_t: int = 2048,
               lag: float = 0.1) -> dict:
    return {"req": req, "shard": 0, "prompt_tokens": prompt, "osl": osl,
            "nominal_T": nominal_t, "t_first_wall": first_wall,
            "deltas_ms": [itl_ms] * (n_events - 1), "n_events": n_events,
            "lag_p99_ms": lag}


def test_summarize_cell_constant_itl() -> None:
    # 4 requests, 7 ms ITL, staggered ramp: request i emits its first token at i*0.1 s.
    recs = [_mk_record(i, first_wall=100.0 + 0.1 * i, n_events=456, itl_ms=7.0)
            for i in range(4)]
    row = summarize_cell(recs)
    assert row["batch_size"] == 4
    assert row["decode_step_ms"] == 7.0
    assert row["validation_status"] == "ok"
    assert row["n_samples"] >= 64
    # effective context = prompt + median in-window progress, near nominal by construction
    assert abs(row["context_len"] - (1820 + row["median_inwindow_progress"])) <= 1
    assert row["nominal_T"] == 2048 and row["osl"] == 456


def test_summarize_cell_flags() -> None:
    # too few in-window samples -> check
    short = [_mk_record(i, 100.0, n_events=40, itl_ms=7.0) for i in range(2)]
    assert summarize_cell(short)["validation_status"] == "check"
    # loop lag p99 over 2 ms -> check
    lag = [_mk_record(i, 100.0, n_events=456, itl_ms=7.0, lag=3.5) for i in range(2)]
    assert summarize_cell(lag)["validation_status"] == "check"
    # disjoint steady window (one request ends before another starts) -> check
    disjoint = [_mk_record(0, 100.0, n_events=100, itl_ms=1.0),
                _mk_record(1, 300.0, n_events=100, itl_ms=1.0)]
    assert summarize_cell(disjoint)["validation_status"] == "check"


def test_builder_merge_latest_wins(tmp_path: Path) -> None:
    run1 = tmp_path / "run1.jsonl.gz"
    run2 = tmp_path / "run2.jsonl.gz"
    cell = [4, 2048]
    with gzip.open(run1, "wt") as f:
        f.write(json.dumps({"_meta": True, "tool": "t"}) + "\n")
        for i in range(4):
            f.write(json.dumps({"cell": cell, **_mk_record(i, 100.0 + 0.1 * i, 456, 9.0)}) + "\n")
    with gzip.open(run2, "wt") as f:
        for i in range(4):
            f.write(json.dumps({"cell": cell, **_mk_record(i, 100.0 + 0.1 * i, 456, 7.0)}) + "\n")
    rows = build([run1, run2])
    assert len(rows) == 1
    assert rows[0]["decode_step_ms"] == 7.0          # run2 (latest) wins
    assert rows[0]["source_file"] == "run2.jsonl.gz"
    # snap rule: effective (prompt 1820 + ~progress) lands within 2% of nominal 2048 -> snapped,
    # measured effective preserved in the diagnostic column
    assert rows[0]["context_len"] == 2048
    assert abs(rows[0]["effective_context_len"] - 2048) / 2048 <= 0.02


def test_builder_snap_flags_large_deviation(tmp_path: Path) -> None:
    # an effective context >2% off nominal (asymmetric steady window) keeps the measured
    # effective and is flagged check
    run = tmp_path / "run.jsonl.gz"
    with gzip.open(run, "wt") as f:
        for i in range(2):
            f.write(json.dumps({"cell": [2, 2048],
                                **_mk_record(i, 100.0, 456, 7.0, prompt=1500)}) + "\n")
    rows = build([run])
    assert rows[0]["validation_status"] == "check"
    assert rows[0]["context_len"] == rows[0]["effective_context_len"] != 2048


def test_builder_csv_is_load_grid_compatible(tmp_path: Path) -> None:
    run = tmp_path / "run.jsonl.gz"
    with gzip.open(run, "wt") as f:
        for b, t, itl in ((1, 512, 3.0), (8, 512, 3.4)):
            prompt, osl = solve_cell(b, t)
            for i in range(b):
                f.write(json.dumps({"cell": [b, t], **_mk_record(
                    i, 100.0 + 0.05 * i, osl, itl, prompt=prompt, osl=osl, nominal_t=t)}) + "\n")
    out = tmp_path / "grid.csv"
    subprocess.run([sys.executable, "-m", "profiling.emit.build_serving_decode_grid",
                    "--inputs", str(run), "--out", str(out)],
                   cwd=REPO_ROOT, check=True, capture_output=True)
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert [r["batch_size"] for r in rows] == ["1", "8"]
    assert all(set(OUT_FIELDS) <= set(r.keys()) for r in rows)

    import pytest
    load_grid = pytest.importorskip("simulator.kernel_step_cost").load_grid  # consumer-only dep
    g = load_grid(out)
    assert len(g.cells) == 2 and g.fixed_floor_ms == 3.0


def _write_run(path: Path, cell: list[int], itl: float, lag: float, n: int = 4,
               n_events: int = 456) -> None:
    with gzip.open(path, "wt") as f:
        f.write(json.dumps({"_meta": True, "tool": "t"}) + "\n")
        for i in range(n):
            f.write(json.dumps({"cell": cell, **_mk_record(
                i, 100.0 + 0.1 * i, n_events, itl, lag=lag)}) + "\n")


def test_lag_upgrade_default_off_keeps_check(tmp_path: Path) -> None:
    # L13 amendment is OPT-IN: without --lag-upgrade a reproducible lag-only cell stays check
    # (pre-L13 artifacts regenerate byte-identically).
    r1, r2 = tmp_path / "r1.jsonl.gz", tmp_path / "r2.jsonl.gz"
    _write_run(r1, [4, 2048], itl=7.0, lag=2.1)
    _write_run(r2, [4, 2048], itl=7.0, lag=2.05)
    rows = build([r1, r2])
    assert rows[0]["validation_status"] == "check"
    assert "lag_reproducibility_upgrade" not in rows[0]


def test_lag_upgrade_reproducible_marginal_lag_to_ok(tmp_path: Path) -> None:
    # >=2 hygiene-clean passes, value spread <=0.5%, latest lag <=1.5x limit -> ok + evidence
    r1, r2 = tmp_path / "r1.jsonl.gz", tmp_path / "r2.jsonl.gz"
    _write_run(r1, [4, 2048], itl=7.0, lag=2.1)
    _write_run(r2, [4, 2048], itl=7.02, lag=2.05)   # 0.29% spread
    rows = build([r1, r2], lag_upgrade=True)
    assert rows[0]["validation_status"] == "ok"
    assert rows[0]["lag_reproducibility_upgrade"].startswith("passes=2")
    assert rows[0]["decode_step_ms"] == 7.02         # latest pass still wins


def test_lag_upgrade_refuses_drift_high_lag_single_pass_and_nonlag(tmp_path: Path) -> None:
    # value drift >0.5% across passes -> stays check (the honest exclusion)
    r1, r2 = tmp_path / "d1.jsonl.gz", tmp_path / "d2.jsonl.gz"
    _write_run(r1, [4, 2048], itl=7.0, lag=2.1)
    _write_run(r2, [4, 2048], itl=7.1, lag=2.05)     # 1.4% spread
    assert build([r1, r2], lag_upgrade=True)[0]["validation_status"] == "check"
    # latest-pass lag beyond 1.5x the limit -> stays check even if values agree
    r3, r4 = tmp_path / "h1.jsonl.gz", tmp_path / "h2.jsonl.gz"
    _write_run(r3, [4, 2048], itl=7.0, lag=2.1)
    _write_run(r4, [4, 2048], itl=7.0, lag=3.5)
    assert build([r3, r4], lag_upgrade=True)[0]["validation_status"] == "check"
    # single pass -> no corroboration -> stays check
    r5 = tmp_path / "s1.jsonl.gz"
    _write_run(r5, [4, 2048], itl=7.0, lag=2.1)
    assert build([r5], lag_upgrade=True)[0]["validation_status"] == "check"
    # non-lag hygiene flag (too few in-window samples) is NOT upgradeable
    r6, r7 = tmp_path / "n1.jsonl.gz", tmp_path / "n2.jsonl.gz"
    _write_run(r6, [4, 2048], itl=7.0, lag=0.1, n_events=40)
    _write_run(r7, [4, 2048], itl=7.0, lag=0.1, n_events=40)
    assert build([r6, r7], lag_upgrade=True)[0]["validation_status"] == "check"
