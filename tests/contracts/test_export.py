"""export/quettasim.py: the builders, the kernel tables merge and the manifests."""
from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import gzip
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest

from quettabench.export import quettasim
from quettabench.export.quettasim import (
    OUT_FIELDS,
    PROD_FA3,
    PROD_HOST_SUM,
    _client,
    _ols,
    build_decode_grid,
    export,
    main,
    run_serving_decode_grid,
)
from quettabench.probes.base import add_output_args, out_root, run_id, run_recorded


def write_leg(path: Path, scale: float) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["new", "cached", "prefill_span_ms", "ttft_ms"])
        for new in (8, 128, 512, 1024, 2048):
            for cached in (0, 2000, 8000, 16000):
                span = scale * (5 + 0.02 * new + 0.004 * cached + 3e-7 * new * (cached + new / 2))
                w.writerow([new, cached, span, span + 2])


def test_tp_comm_reads_results_and_writes_dest(tmp_path: Path) -> None:
    results, dest = tmp_path / "results", tmp_path / "out"
    results.mkdir()
    write_leg(results / "serving_stage_split_H100.csv", 1.0)
    write_leg(results / "serving_stage_split_H100_tp2.csv", 0.6)
    written = export([results], dest, only=["tp_comm"])
    assert written == [dest / "kernels" / "prefill_tp_comm_H100.json"]
    payload = json.loads(written[0].read_text())
    assert payload["constants"]["PREFILL_TP_COMM_MS_PER_TOKEN"] > 0


def test_manifest_names_sources_sha_and_time(tmp_path: Path) -> None:
    results, dest = tmp_path / "results", tmp_path / "out"
    results.mkdir()
    export([results], dest, only=["tp_comm"])
    manifest = json.loads((dest / "manifest.json").read_text())
    assert manifest["source_dirs"] == [str(results)]
    assert set(manifest) == {"source_dirs", "runs", "bench_sha", "exported_at"}
    assert manifest["runs"] == []
    assert manifest["exported_at"].endswith("+00:00")


def test_manifest_lists_the_runs_in_its_sources(tmp_path: Path) -> None:
    results, dest = tmp_path / "results", tmp_path / "out"
    run = results / "H100" / "gemm"
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({
        "config": {"run_name": "probe-gemm"}, "run_status": {"status": "ok", "reason": None},
        "provenance": {"started": "2026-10-11T00:00:00+00:00"}}))
    export([results], dest, only=["tp_comm"])
    manifest = json.loads((dest / "manifest.json").read_text())
    assert manifest["runs"] == [{"run": str(run), "run_name": "probe-gemm", "status": "ok",
                                 "started": "2026-10-11T00:00:00+00:00"}]


def test_unfinished_unreadable_and_failed_runs_are_skipped(tmp_path: Path, capsys) -> None:
    results, dest = tmp_path / "results", tmp_path / "out"
    good = results / "z-good"  # sorts after the bad runs
    good.mkdir(parents=True)
    write_leg(good / "serving_stage_split_H100.csv", 1.0)
    write_leg(good / "serving_stage_split_H100_tp2.csv", 0.6)
    for name, run_json in (("crashed", None), ("garbled", "{not json"),
                           ("failed", json.dumps({"run_status": {"status": "failed"}}))):
        run = results / name
        run.mkdir()
        (run / "requests.jsonl").write_text("")
        if run_json is not None:
            (run / "run.json").write_text(run_json)
        write_leg(run / "serving_stage_split_H100.csv", 9.0)  # sorts first; must not be read
    written = export([results], dest, only=["tp_comm"])
    out = capsys.readouterr().out
    assert f"skip {results / 'crashed'}: no run.json" in out
    assert f"skip {results / 'garbled'}: run.json unreadable (JSONDecodeError)" in out
    assert f"skip {results / 'failed'}: run_status failed" in out
    assert json.loads((dest / "manifest.json").read_text())["runs"] == []
    reference = export([good], tmp_path / "ref", only=["tp_comm"])
    assert written[0].read_text() == reference[0].read_text()


@pytest.mark.parametrize("fails", [False, True])
def test_a_failed_probe_run_keeps_its_tables_out(tmp_path: Path, fails: bool) -> None:
    """`just probe serving_stage_split` writes its legs and run.json in one dir;
    a run that raised after writing them exports nothing."""
    results = tmp_path / "results"
    leg = results / "H100" / "serving_stage_split" / "serving_stage_split_H100.csv"

    def main(argv):
        leg.parent.mkdir(parents=True, exist_ok=True)
        write_leg(leg, 1.0)
        write_leg(leg.with_name("serving_stage_split_H100_tp2.csv"), 0.6)
        if fails:
            raise RuntimeError("server died")

    with mock.patch.object(sys, "argv", ["probe", "--out", str(leg)]), \
            pytest.raises(RuntimeError) if fails else contextlib.nullcontext():
        run_recorded(main, "serving_stage_split", file_flag="--out")
    assert (leg.parent / "run.json").exists()
    written = export([results], tmp_path / "out", only=["tp_comm"])
    assert written == ([] if fails else [tmp_path / "out" / "kernels" / "prefill_tp_comm_H100.json"])


def test_a_killed_probe_run_leaves_its_tables_out(tmp_path: Path) -> None:
    """A probe killed before run_recorded's finally (SIGKILL, OOM) leaves its
    tables beside requests.jsonl and no run.json: export skips them."""
    leg = tmp_path / "results" / "H100" / "serving_stage_split" / "serving_stage_split_H100.csv"
    seen = []

    def main(argv):
        seen.append(sorted(p.name for p in leg.parent.iterdir()))
        write_leg(leg, 1.0)
        write_leg(leg.with_name("serving_stage_split_H100_tp2.csv"), 0.6)
        raise KeyboardInterrupt

    with mock.patch.object(sys, "argv", ["probe", "--out", str(leg)]), \
            pytest.raises(KeyboardInterrupt):
        run_recorded(main, "serving_stage_split", file_flag="--out")
    assert seen == [["requests.jsonl"]]
    (leg.parent / "run.json").unlink()  # what a kill, which skips the finally, leaves
    quettasim.run_dirs.cache_clear()
    assert quettasim.skipped_runs(tmp_path / "results") == {leg.parent: "no run.json"}
    assert export([tmp_path / "results"], tmp_path / "out", only=["tp_comm"]) == []


def test_a_second_run_into_one_dir_is_refused(tmp_path: Path) -> None:
    """A probe that names its output file records the dir that file is in; a
    second run there would make one run.json stand for both, so it is refused
    and the first run's record is kept."""
    out = tmp_path / "h100" / "raw.jsonl.gz"
    with mock.patch.object(sys, "argv", ["probe", "--out-raw", str(out)]):
        run_recorded(lambda argv: None, "serving_decode_grid", file_flag="--out-raw")
        first = (out.parent / "run.json").read_text()
        main = mock.Mock()
        with pytest.raises(SystemExit, match="already exists"):
            run_recorded(main, "serving_decode_grid", file_flag="--out-raw")
    main.assert_not_called()
    assert (out.parent / "run.json").read_text() == first


def test_a_file_belongs_to_its_nearest_run_dir(tmp_path: Path) -> None:
    """A failed run dir outside a table's own run dir does not drop it, and a
    table under a failed run's subdirs is dropped."""
    for run, status in (("H100/gemm", "ok"), ("H100/flash_attn", "failed"), ("H100", "failed")):
        (tmp_path / run).mkdir(parents=True, exist_ok=True)
        (tmp_path / run / "run.json").write_text(json.dumps({"run_status": {"status": status}}))
    for table in ("H100/gemm/ncu/gemm/H100.csv", "H100/flash_attn/ncu/flash_attn/H100.csv"):
        (tmp_path / table).parent.mkdir(parents=True)
        (tmp_path / table).write_text("")
    quettasim.run_dirs.cache_clear()
    kept = [p.relative_to(tmp_path).as_posix() for p in quettasim.files(tmp_path, "*.csv")]
    assert kept == ["H100/gemm/ncu/gemm/H100.csv"]


def test_a_launched_servers_dir_belongs_to_its_run(tmp_path: Path, capsys) -> None:
    """A live-server probe launches into <run>/server/ (engine.json there):
    that dir is part of the run, not a run of its own without a run.json."""
    run = tmp_path / "results" / "H100" / "pool_capacity" / "20261011T120000.000000Z"
    (run / "server").mkdir(parents=True)
    (run / "server" / "engine.json").write_text("{}")
    (run / "run.json").write_text(json.dumps({
        "config": {"run_name": "probe-pool_capacity"}, "run_status": {"status": "ok"},
        "provenance": {"started": "2026-10-11T12:00:00+00:00"}}))
    export([tmp_path / "results"], tmp_path / "out", only=["tp_comm"])
    assert "skip" not in capsys.readouterr().out
    assert quettasim.skipped_runs(tmp_path / "results") == {}
    assert quettasim.owner(run / "server" / "engine.json", tmp_path / "results",
                         quettasim.run_dirs(tmp_path / "results")) == run


def test_each_builder_reads_run_status_fresh(tmp_path: Path, capsys) -> None:
    """A run whose run.json changes while export runs is judged by what is on
    disk when each builder (and the manifest) reads it."""
    results = tmp_path / "results"
    run = results / "r"
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({
        "config": {"run_name": "r"}, "run_status": {"status": "ok"}, "provenance": {}}))
    seen = []

    def first(dirs, dest):
        seen.append(quettasim.skipped_runs(results))
        (run / "run.json").write_text(json.dumps({"run_status": {"status": "failed"}}))
        return []

    def second(dirs, dest):
        seen.append(quettasim.skipped_runs(results))
        return []

    with mock.patch.dict(quettasim.BUILDERS, {"first": first, "second": second}):
        export([results], tmp_path / "out", only=["first", "second"])
    assert seen == [{}, {run: "run_status failed"}]
    assert json.loads((tmp_path / "out" / "manifest.json").read_text())["runs"] == []


def stage_split_run(root: Path, run_id: str, finished: str, status: str, scale: float) -> Path:
    """One serving_stage_split run dir holding both legs and its run.json."""
    run = root / "H100" / "serving_stage_split" / run_id
    run.mkdir(parents=True)
    write_leg(run / "serving_stage_split_H100.csv", scale)
    write_leg(run / "serving_stage_split_H100_tp2.csv", scale * 0.6)
    (run / "run.json").write_text(json.dumps({
        "config": {"run_name": "probe-serving_stage_split"},
        "run_status": {"status": status, "reason": None},
        "provenance": {"started": run_id, "finished": finished}}))
    return run


def test_stage_split_legs_come_from_the_latest_finished_ok_run(tmp_path: Path) -> None:
    """Run dirs sort by start; the run that finished last is read, a newer
    failed run never is, and both legs follow the one rule."""
    results = tmp_path / "results"
    old = stage_split_run(results, "20261001", "2026-10-01T02:00:00+00:00", "ok", 1.0)
    new = stage_split_run(results, "20261002", "2026-10-02T02:00:00+00:00", "ok", 2.0)
    stage_split_run(results, "20261003", "2026-10-03T02:00:00+00:00", "failed", 3.0)
    quettasim.run_dirs.cache_clear()
    assert quettasim.find([results], "serving_stage_split_H100.csv") == new / "serving_stage_split_H100.csv"
    assert quettasim.stage_split_legs([results], "H100") == [
        (2, new / "serving_stage_split_H100.csv", new / "serving_stage_split_H100_tp2.csv")]
    # A run that started first but finished last wins over one that started later.
    (old / "run.json").write_text(json.dumps({
        "config": {}, "run_status": {"status": "ok", "reason": None},
        "provenance": {"started": "20261001", "finished": "2026-10-09T00:00:00+00:00"}}))
    quettasim.run_dirs.cache_clear()
    assert quettasim.find([results], "serving_stage_split_H100_tp2.csv") == old / "serving_stage_split_H100_tp2.csv"


def test_files_and_find_skip_a_clearing_dir(tmp_path: Path) -> None:
    """A run dir clear_run_dir renamed to .clearing-* and did not finish
    removing: its files are no run's, and export reads none of them."""
    results = tmp_path / "results"
    run = stage_split_run(results, "20261001", "2026-10-01T02:00:00+00:00", "ok", 1.0)
    run.rename(run.with_name(".clearing-abc123"))
    quettasim.run_dirs.cache_clear()
    assert quettasim.files(results, "*.csv") == []
    assert quettasim.find([results], "serving_stage_split_H100.csv") is None


def test_a_pulled_run_is_ordered_by_its_pulled_time(tmp_path: Path) -> None:
    """A remote run's finished is on the remote clock; its pulled, on this
    one, orders it when present."""
    results = tmp_path / "results"
    local = stage_split_run(results, "20261001", "2026-10-05T00:00:00+00:00", "ok", 1.0)
    pulled = stage_split_run(results, "20261002", "2099-01-01T00:00:00+00:00", "ok", 2.0)
    doc = json.loads((pulled / "run.json").read_text())
    doc["provenance"]["pulled"] = "2026-10-04T00:00:00+00:00"
    (pulled / "run.json").write_text(json.dumps(doc))
    quettasim.run_dirs.cache_clear()
    assert quettasim.finished(pulled) == datetime(2026, 10, 4, tzinfo=timezone.utc)
    assert quettasim.find([results], "serving_stage_split_H100.csv") == local / "serving_stage_split_H100.csv"


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
ART = RESULTS / "export" / "kernels" / "prefill_stage_rates_H100x4.json"
TP1_CSV = RESULTS / "serving_stage_split_H100.csv"
TP4_CSV = RESULTS / "serving_stage_split_H100_tp4.csv"


def test_ols_recovers_exact_synthetic_coefficients():
    # y = 7 + 0.02*new + 0.005*cached + 3e-7*new*(cached+new/2), exactly linear in the
    # design -> the closed-form OLS must recover the coefficients to float precision.
    rows = []
    for new in (8.0, 128.0, 512.0, 1024.0, 2048.0):
        for cached in (0.0, 2000.0, 8000.0, 16000.0):
            y = 7.0 + 0.02 * new + 0.005 * cached + 3e-7 * new * (cached + 0.5 * new)
            rows.append({"new": new, "cached": cached, "y": y})
    beta = _ols(rows, "y", ["1", "new", "cached", "fa3"])
    assert abs(beta["1"] - 7.0) < 1e-9
    assert abs(beta["new"] - 0.02) < 1e-12
    assert abs(beta["cached"] - 0.005) < 1e-12
    assert abs(beta["fa3"] - 3e-7) < 1e-15
    assert beta["_fit_mape_pct"] < 1e-9
    # 3-param family (the production host estimator) on data without the cross term
    rows3 = [{"new": r["new"], "cached": r["cached"],
              "y": 5.0 + 0.01 * r["new"] + 0.004 * r["cached"]} for r in rows]
    beta3 = _ols(rows3, "y", ["1", "new", "cached"])
    assert abs(beta3["cached"] - 0.004) < 1e-12


@pytest.mark.skipif(not (TP1_CSV.exists() and TP4_CSV.exists() and ART.exists()),
                    reason="stage-split CSVs and the exported artifact are not on this checkout")
def test_artifact_regenerates_from_the_csvs():
    rows1 = list(csv.DictReader(TP1_CSV.open()))
    rows4 = list(csv.DictReader(TP4_CSV.open()))
    host1 = _ols(rows1, "ttft_ms", ["1", "new", "cached"])
    host4 = _ols(rows4, "ttft_ms", ["1", "new", "cached"])
    fa3_1 = _ols(rows1, "prefill_span_ms", ["1", "new", "cached", "fa3"])
    fa3_4 = _ols(rows4, "prefill_span_ms", ["1", "new", "cached", "fa3"])
    art = json.loads(ART.read_text())
    assert art["constants"]["prefill_host_cached_ms_per_token"] == host4["cached"]
    assert art["constants"]["prefill_fa3_ms_per_token2"] == PROD_FA3 * (fa3_4["fa3"] / fa3_1["fa3"])
    # the pre-registered validations hold on the real data
    assert abs(host1["cached"] - PROD_HOST_SUM) / PROD_HOST_SUM < 0.05   # tp1 leg ~ production
    assert 0.0 < fa3_4["fa3"] / fa3_1["fa3"] < 1.0                       # sharding signal


def test_builders_stop_in_one_line_without_ttft_queue_sim(tmp_path):
    """QuettaSim main has no simulator.ttft_queue_sim; the builders name the
    missing constant instead of raising ImportError."""
    x = tmp_path / "x.csv"
    for call in (lambda: quettasim.build_stage_rates(4, x, x, x),
                 lambda: quettasim.build_consumer("H200", 2, x, x, x, x)):
        with pytest.raises(SystemExit, match="QuettaSim main has no "
                           "simulator.ttft_queue_sim.PREFILL_HOST_SHARED_MS_PER_TOKEN"):
            call()
    write_leg(tmp_path / "tp1.csv", 1.0)
    write_leg(tmp_path / "tp4.csv", 0.4)
    quettasim.build_tpn(4, tmp_path / "tp4.csv", tmp_path / "out.json", tmp_path / "tp1.csv")
    out = json.loads((tmp_path / "out.json").read_text())
    assert out["fallback_it_replaces_ms_per_token"] is None




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
    rows = build_decode_grid([run1, run2])
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
    run = tmp_path / "serving_decode_grid_H100x4_2026-10-01.jsonl.gz"
    with gzip.open(run, "wt") as f:
        for i in range(2):
            f.write(json.dumps({"cell": [2, 2048],
                                **_mk_record(i, 100.0, 456, 7.0, prompt=1500)}) + "\n")
    rows = build_decode_grid([run])
    assert rows[0]["validation_status"] == "check"
    assert rows[0]["context_len"] == rows[0]["effective_context_len"] != 2048


def test_builder_csv_is_load_grid_compatible(tmp_path: Path) -> None:
    run = tmp_path / "serving_decode_grid_H100x4_2026-10-01.jsonl.gz"
    with gzip.open(run, "wt") as f:
        for b, t, itl in ((1, 512, 3.0), (8, 512, 3.4)):
            prompt, osl = solve_cell(b, t)
            for i in range(b):
                f.write(json.dumps({"cell": [b, t], **_mk_record(
                    i, 100.0 + 0.05 * i, osl, itl, prompt=prompt, osl=osl, nominal_t=t)}) + "\n")
    [out] = run_serving_decode_grid([tmp_path], tmp_path / "exported")
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
    rows = build_decode_grid([r1, r2])
    assert rows[0]["validation_status"] == "check"
    assert "lag_reproducibility_upgrade" not in rows[0]


def test_lag_upgrade_reproducible_marginal_lag_to_ok(tmp_path: Path) -> None:
    # >=2 hygiene-clean passes, value spread <=0.5%, latest lag <=1.5x limit -> ok + evidence
    r1, r2 = tmp_path / "r1.jsonl.gz", tmp_path / "r2.jsonl.gz"
    _write_run(r1, [4, 2048], itl=7.0, lag=2.1)
    _write_run(r2, [4, 2048], itl=7.02, lag=2.05)   # 0.29% spread
    rows = build_decode_grid([r1, r2], lag_upgrade=True)
    assert rows[0]["validation_status"] == "ok"
    assert rows[0]["lag_reproducibility_upgrade"].startswith("passes=2")
    assert rows[0]["decode_step_ms"] == 7.02         # latest pass still wins


def test_lag_upgrade_refuses_drift_high_lag_single_pass_and_nonlag(tmp_path: Path) -> None:
    # value drift >0.5% across passes -> stays check (the honest exclusion)
    r1, r2 = tmp_path / "d1.jsonl.gz", tmp_path / "d2.jsonl.gz"
    _write_run(r1, [4, 2048], itl=7.0, lag=2.1)
    _write_run(r2, [4, 2048], itl=7.1, lag=2.05)     # 1.4% spread
    assert build_decode_grid([r1, r2], lag_upgrade=True)[0]["validation_status"] == "check"
    # latest-pass lag beyond 1.5x the limit -> stays check even if values agree
    r3, r4 = tmp_path / "h1.jsonl.gz", tmp_path / "h2.jsonl.gz"
    _write_run(r3, [4, 2048], itl=7.0, lag=2.1)
    _write_run(r4, [4, 2048], itl=7.0, lag=3.5)
    assert build_decode_grid([r3, r4], lag_upgrade=True)[0]["validation_status"] == "check"
    # single pass -> no corroboration -> stays check
    r5 = tmp_path / "s1.jsonl.gz"
    _write_run(r5, [4, 2048], itl=7.0, lag=2.1)
    assert build_decode_grid([r5], lag_upgrade=True)[0]["validation_status"] == "check"
    # non-lag hygiene flag (too few in-window samples) is NOT upgradeable
    r6, r7 = tmp_path / "n1.jsonl.gz", tmp_path / "n2.jsonl.gz"
    _write_run(r6, [4, 2048], itl=7.0, lag=0.1, n_events=40)
    _write_run(r7, [4, 2048], itl=7.0, lag=0.1, n_events=40)
    assert build_decode_grid([r6, r7], lag_upgrade=True)[0]["validation_status"] == "check"


T0 = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)


def probe_run(results: Path, probe: str, tables: dict[str, str], at: int, fails=False,
              gpu: str = "H100", finished: int | None = None, args=()) -> None:
    """One `python -m quettabench.probes.<probe>` run through run_recorded:
    writes tables {path under the run dir: text} where the probe would.
    It starts at T0+at and, given finished, finishes at T0+finished."""
    def main(argv):
        ap = argparse.ArgumentParser()
        add_output_args(ap)
        root = out_root(ap.parse_known_args(argv)[0], probe)
        for rel, text in tables.items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(text)
        if fails:
            raise RuntimeError("probe died")

    argv = ["probe", "--gpu-label", gpu, "--out", str(results / "probes" / gpu), *args]
    with mock.patch.object(sys, "argv", argv), \
            pytest.raises(RuntimeError) if fails else contextlib.nullcontext():
        run_recorded(main, probe, now=T0 + timedelta(seconds=at))
    if finished is not None:
        record = results / "probes" / gpu / probe / run_id(T0 + timedelta(seconds=at)) / "run.json"
        doc = json.loads(record.read_text())
        doc["provenance"]["finished"] = (T0 + timedelta(seconds=finished)).isoformat()
        record.write_text(json.dumps(doc))


def tree(root: Path) -> dict[str, str]:
    """{path: text} of every file under root but the export's manifests."""
    return {p.relative_to(root).as_posix(): p.read_text()
            for p in sorted(root.rglob("*")) if p.is_file() and p.name != "qbench_manifest.json"
            and not p.name.endswith(".meta.json")}


def csv_text(rows):
    return "".join(",".join(map(str, r)) + "\r\n" for r in rows)


COLLECTIVE = csv_text([("op", "world", "bytes", "latency_us", "dtype_bytes"),
                       ("all_gather", 2, 4096, 15.2, 2), ("all_reduce", 2, 4096, 20.3, 2),
                       ("reduce_scatter", 2, 4096, 14.1, 2)])
VLLM = csv_text([("op", "world", "bytes", "latency_us", "dtype_bytes", "path"),
                 ("all_reduce", 2, 4096, 9.1, 2, "vllm"), ("all_reduce", 2, 8192, 9.4, 2, "vllm")])


def test_probe_runs_export_to_the_quettasim_layout(tmp_path: Path) -> None:
    """Every table lands where kernel_artifact_paths looks for it; tables
    outside ncu/ and cuda_event/ stay home. ncu/ goes to graph/, cuda_event/ to eager/."""
    results, dest = tmp_path / "results", tmp_path / "out"
    probe_run(results, "gemm", {"ncu/gemm/H100.csv": "M,N,K\n", "ncu/gemm_wide/H100.csv": "w\n",
                                "ncu/elementwise/H100.json": '{"fp8_act_quant": {"floor_us": 2}}'}, 0)
    probe_run(results, "elementwise", {"ncu/elementwise/H100.json": '{"rms_norm": {"floor_us": 1}}'}, 1)
    probe_run(results, "flash_attn", {"ncu/flash_attn/H100.csv": "fa\n",
                                      "cuda_event/fa3_prefill/H100.csv": "pf\n"}, 2)
    probe_run(results, "cross_attn", {"cuda_event/fa3_cross/H100.csv": "x\n"}, 3)
    probe_run(results, "moe", {"ncu/moe/H100.csv": "moe\n", "ncu/moe/H100_routing.json": "{}"}, 4)
    probe_run(results, "gdn", {"cuda_event/gdn/H100_9b_decode.csv": "g\n"}, 5)
    probe_run(results, "reparallel", {"cuda_event/reparallel/H100_reshard.csv": "r\n"}, 6)
    probe_run(results, "collective", {"ncu/collectives/H100_tp2.csv": COLLECTIVE}, 7)
    probe_run(results, "vllm_allreduce", {"ncu/collectives/H100_tp2.csv": VLLM,
                                          "ncu/collectives/H100_tp2_nccl_vs_vllm.csv": "b\n"}, 8)
    probe_run(results, "fused_block", {"validation/fused_block_profile_m_tp1_H100.csv": "v\n"}, 9)
    probe_run(results, "moe_routing", {"m_layer1.safetensors": "s"}, 10)
    written = export([results], kernel_dest=dest, only=["kernel_tables"])
    got = tree(dest / "kernel_data")
    assert sorted(got) == [
        "eager/fa3_cross/H100.csv",               # cross_attn_path
        "eager/fa3_prefill/H100.csv",             # prefill_attn_path
        "eager/gdn/H100_9b_decode.csv",           # gdn_data_dir
        "eager/reparallel/H100_reshard.csv",
        "graph/collectives/H100_tp2.csv",         # allreduce_path
        "graph/collectives/H100_tp2_nccl_vs_vllm.csv",
        "graph/elementwise/H100.json",            # elementwise_path
        "graph/flash_attn/H100.csv",              # flash_attn_path
        "graph/gemm/H100.csv",                    # gemm_path, router_gemm_path
        "graph/gemm_wide/H100.csv",
        "graph/moe/H100.csv",                     # moe_path
        "graph/moe/H100_routing.json",            # moe_routing_path
    ]
    sidecars = [dest / "kernel_data" / f"{rel}.meta.json" for rel in got]
    assert sorted(written) == sorted([dest / "kernel_data" / rel for rel in got] + sidecars)
    assert got["graph/gemm/H100.csv"] == "M,N,K\n"
    assert json.loads(got["graph/elementwise/H100.json"]) == {"fp8_act_quant": {"floor_us": 2},
                                                           "rms_norm": {"floor_us": 1}}
    assert not (dest / "manifest.json").exists()
    manifest = json.loads((dest / "kernel_data" / "qbench_manifest.json").read_text())
    runs = manifest["kernel_tables"]
    assert set(runs) == set(got)
    assert [Path(r).parent.name for r in runs["graph/collectives/H100_tp2.csv"]["runs"]] == [
        "collective", "vllm_allreduce"]


@pytest.mark.parametrize("vllm_first", [False, True])
def test_vllm_all_reduce_rows_merge_over_the_collective_ones(tmp_path: Path, vllm_first) -> None:
    """vllm_allreduce's all_reduce rows replace the collective probe's, and
    the other ops keep their NCCL rows with path nccl, whichever ran last."""
    results, dest = tmp_path / "results", tmp_path / "out"
    probe_run(results, "collective", {"ncu/collectives/H100_tp2.csv": COLLECTIVE}, 1 - vllm_first)
    probe_run(results, "vllm_allreduce", {"ncu/collectives/H100_tp2.csv": VLLM}, vllm_first)
    export([results], kernel_dest=dest, only=["kernel_tables"])
    with (dest / "kernel_data" / "graph" / "collectives" / "H100_tp2.csv").open(newline="") as f:
        rows = list(csv.reader(f))
    assert rows == [["op", "world", "bytes", "latency_us", "dtype_bytes", "path"],
                    ["all_gather", "2", "4096", "15.2", "2", "nccl"],
                    ["reduce_scatter", "2", "4096", "14.1", "2", "nccl"],
                    ["all_reduce", "2", "4096", "9.1", "2", "vllm"],
                    ["all_reduce", "2", "8192", "9.4", "2", "vllm"]]


def test_the_newest_ok_run_wins_and_failed_or_unfinished_runs_never_do(tmp_path: Path) -> None:
    results, dest = tmp_path / "results", tmp_path / "out"
    probe_run(results, "gemm", {"ncu/gemm/H100.csv": "old\n"}, 0)
    probe_run(results, "gemm", {"ncu/gemm/H100.csv": "good\n"}, 1)
    probe_run(results, "gemm", {"ncu/gemm/H100.csv": "partial\n"}, 2, fails=True)
    probe_run(results, "gemm", {"ncu/gemm/H100.csv": "killed\n"}, 3)
    killed = results / "probes" / "H100" / "gemm" / "20261011T120003.000000Z"
    (killed / "run.json").unlink()   # what a kill before run_recorded's finally leaves
    probe_run(results, "flash_attn", {"ncu/flash_attn/H100.csv": "bad\n"}, 4, fails=True)
    export([results], kernel_dest=dest, only=["kernel_tables"])
    assert tree(dest / "kernel_data") == {"graph/gemm/H100.csv": "good\n"}


def test_a_qbench_run_probe_exports_too(tmp_path: Path) -> None:
    """A `mode: probe` run writes results/<run>/<probe>/<method>/... under the
    run's own run.json."""
    run = tmp_path / "results" / "gemm-h100"
    (run / "gemm" / "ncu" / "gemm").mkdir(parents=True)
    (run / "gemm" / "ncu" / "gemm" / "H100.csv").write_text("t\n")
    (run / "run.json").write_text(json.dumps({
        "config": {"mode": {"mode": "probe", "probe": "gemm"}},
        "provenance": {"started": "2026-10-11T00:00:00+00:00"},
        "run_status": {"status": "ok", "reason": None}}))
    export([tmp_path / "results"], kernel_dest=tmp_path / "out", only=["kernel_tables"])
    assert tree(tmp_path / "out" / "kernel_data") == {"graph/gemm/H100.csv": "t\n"}


def test_two_probes_writing_one_file_stop_the_export(tmp_path: Path) -> None:
    results = tmp_path / "results"
    probe_run(results, "gemm", {"ncu/moe/H100.csv": "a\n"}, 0)
    probe_run(results, "moe", {"ncu/moe/H100.csv": "b\n"}, 1)
    with pytest.raises(SystemExit, match="graph/moe/H100.csv.*gemm.*moe"):
        quettasim.run_kernel_tables([results], tmp_path / "out")


def rows(path: Path) -> list[list[str]]:
    with path.open(newline="") as f:
        return list(csv.reader(f))


FA_HEAD = ("n_heads", "n_kv_heads", "head_dim", "q_len", "latency_us")


def test_overlapping_runs_union_their_rows(tmp_path: Path) -> None:
    """Run A starts first and finishes last, run B starts later and finishes
    first; each measured its own head config, so the export holds both, and
    the head config both measured comes from A, the run that finished last."""
    results, dest = tmp_path / "results", tmp_path / "out"
    a = csv_text([FA_HEAD, (32, 8, 128, 1, 10.0), (32, 8, 128, 2, 11.0), (16, 2, 128, 1, 7.0)])
    b = csv_text([FA_HEAD, (64, 8, 128, 1, 20.0), (16, 2, 128, 1, 9.0), (16, 2, 128, 2, 9.5)])
    probe_run(results, "flash_attn", {"ncu/flash_attn/H100.csv": a}, at=0, finished=100)
    probe_run(results, "flash_attn", {"ncu/flash_attn/H100.csv": b}, at=1, finished=50)
    export([results], kernel_dest=dest, only=["kernel_tables"])
    assert rows(dest / "kernel_data" / "graph" / "flash_attn" / "H100.csv") == [
        list(FA_HEAD), ["64", "8", "128", "1", "20.0"],
        ["32", "8", "128", "1", "10.0"], ["32", "8", "128", "2", "11.0"],
        ["16", "2", "128", "1", "7.0"]]
    runs = json.loads((dest / "kernel_data" / "qbench_manifest.json").read_text())["kernel_tables"]
    assert [Path(r).name for r in runs["graph/flash_attn/H100.csv"]["runs"]] == [
        "20261011T120001.000000Z", "20261011T120000.000000Z"]


GEMM_HEAD = ("M", "N", "K", "dtype_bytes", "latency_us")


def test_gemm_rows_union_with_append_and_reset_without(tmp_path: Path) -> None:
    """A gemm run without --append replaces the table, as it did when it
    wrote the whole file; a run with --append adds its cells over it."""
    results, dest = tmp_path / "results", tmp_path / "out"
    table = "ncu/gemm/H100.csv"
    probe_run(results, "gemm", {table: csv_text([GEMM_HEAD, (1, 2, 3, 2, 9.0)])}, 0)
    probe_run(results, "gemm", {table: csv_text([GEMM_HEAD, (1, 4, 4, 2, 5.0), (8, 4, 4, 2, 6.0)])}, 1)
    probe_run(results, "gemm", {table: csv_text([GEMM_HEAD, (8, 4, 4, 2, 7.0)])}, 2,
              args=["--append"])
    export([results], kernel_dest=dest, only=["kernel_tables"])
    assert rows(dest / "kernel_data" / "graph/gemm/H100.csv") == [
        list(GEMM_HEAD), ["1", "4", "4", "2", "5.0"], ["8", "4", "4", "2", "7.0"]]


def test_reparallel_rows_upsert_on_their_key_and_sort(tmp_path: Path) -> None:
    head = ("gpu", "world", "tp_from", "tp_to", "total_weight_gb", "comm_ms")
    results, dest = tmp_path / "results", tmp_path / "out"
    table = "cuda_event/reparallel/H100_reshard.csv"
    probe_run(results, "reparallel", {table: csv_text([head, ("H100", 8, 4, 2, 16.0, 3.0),
                                                       ("H100", 8, 2, 1, 16.0, 2.0)])}, 0)
    probe_run(results, "reparallel", {table: csv_text([head, ("H100", 8, 2, 1, 16.0, 1.5)])}, 1)
    export([results], kernel_dest=dest, only=["kernel_tables"])
    assert rows(dest / "kernel_data" / "eager/reparallel/H100_reshard.csv") == [
        list(head), ["H100", "8", "2", "1", "16.0", "1.5"], ["H100", "8", "4", "2", "16.0", "3.0"]]


def test_a_whole_table_comes_from_the_run_that_finished_last(tmp_path: Path) -> None:
    """gdn writes its whole table each run (no row key): the latest finished
    run's file, byte for byte, even when it started first."""
    results, dest = tmp_path / "results", tmp_path / "out"
    table = "cuda_event/gdn/H100_9b_decode.csv"
    probe_run(results, "gdn", {table: "batch,us\n1,5\n"}, at=0, finished=100)
    probe_run(results, "gdn", {table: "batch,us\n2,6\n"}, at=1, finished=50)
    export([results], kernel_dest=dest, only=["kernel_tables"])
    assert tree(dest / "kernel_data") == {"eager/gdn/H100_9b_decode.csv": "batch,us\n1,5\n"}


def test_elementwise_entries_union_with_gemm_fp8_quant(tmp_path: Path) -> None:
    """The elementwise probe's latest run gives its entries whole; every gemm
    run's fp8_act_quant goes over them, the later finished run winning a key."""
    results, dest = tmp_path / "results", tmp_path / "out"
    table = "ncu/elementwise/H100.json"
    probe_run(results, "elementwise", {table: '{"rms_norm": 1, "silu": 1}'}, 0)
    probe_run(results, "gemm", {table: '{"fp8_act_quant": 1}'}, 1)
    probe_run(results, "elementwise", {table: '{"rms_norm": 2}'}, 2)
    probe_run(results, "gemm", {table: '{"fp8_act_quant": 3}'}, 3)
    export([results], kernel_dest=dest, only=["kernel_tables"])
    assert json.loads((dest / "kernel_data" / "graph/elementwise/H100.json").read_text()) == {
        "rms_norm": 2, "fp8_act_quant": 3}


def test_an_unmergeable_file_stops_the_export_before_anything_is_written(tmp_path: Path) -> None:
    results, dest = tmp_path / "results", tmp_path / "out"
    probe_run(results, "gdn", {"cuda_event/gdn/H100_9b_decode.csv": "g\n"}, 0)
    probe_run(results, "gemm", {"ncu/moe/H100.csv": "a\n"}, 1)
    probe_run(results, "moe", {"ncu/moe/H100.csv": "b\n"}, 2)
    with pytest.raises(SystemExit, match="graph/moe/H100.csv"):
        export([results], kernel_dest=dest, only=["kernel_tables"])
    assert not dest.exists()


def test_the_manifest_names_only_runs_whose_rows_were_copied(tmp_path: Path) -> None:
    """A run every one of whose rows a later run replaced, and a collective
    run with only all_reduce rows, put nothing in the export."""
    results, dest = tmp_path / "results", tmp_path / "out"
    fa = "ncu/flash_attn/H100.csv"
    probe_run(results, "flash_attn", {fa: csv_text([FA_HEAD, (32, 8, 128, 1, 9.0)])}, 0)
    probe_run(results, "flash_attn", {fa: csv_text([FA_HEAD, (32, 8, 128, 1, 8.0)])}, 1)
    only_ar = csv_text([("op", "world", "bytes", "latency_us", "dtype_bytes"),
                        ("all_reduce", 2, 4096, 20.3, 2)])
    probe_run(results, "collective", {"ncu/collectives/H100_tp2.csv": only_ar}, 2)
    probe_run(results, "vllm_allreduce", {"ncu/collectives/H100_tp2.csv": VLLM}, 3)
    written, used = quettasim.run_kernel_tables([results], dest)
    assert sorted(used) == ["graph/collectives/H100_tp2.csv", "graph/flash_attn/H100.csv"]
    assert [Path(r).parent.name for r in used["graph/collectives/H100_tp2.csv"]] == ["vllm_allreduce"]
    assert [Path(r).name for r in used["graph/flash_attn/H100.csv"]] == ["20261011T120001.000000Z"]
    export([results], kernel_dest=dest, only=["kernel_tables"])
    manifest = json.loads((dest / "kernel_data" / "qbench_manifest.json").read_text())
    assert {rel: entry["runs"] for rel, entry in manifest["kernel_tables"].items()} == used


def test_kernel_tables_go_to_their_own_dest_and_only_when_given(tmp_path: Path, capsys) -> None:
    """--dest takes the other builders' files; the kernel tables go to the
    QuettaSim data root given as --kernel-dest, and without it they are not built."""
    results, dest, data = tmp_path / "results", tmp_path / "out", tmp_path / "QuettaSim" / "data"
    probe_run(results, "gdn", {"cuda_event/gdn/H100_9b_decode.csv": "g\n"}, 0)
    with mock.patch.dict(quettasim.BUILDERS, {"tp_comm": quettasim.run_tp_comm,
                                              "kernel_tables": quettasim.run_kernel_tables}, clear=True):
        main(["--results", str(results), "--dest", str(dest)])
    assert not (dest / "kernel_data").exists() and not data.exists()
    assert "kernel_tables" not in json.loads((dest / "manifest.json").read_text())
    main(["--results", str(results), "--kernel-dest", str(data), "--only", "kernel_tables"])
    assert tree(data / "kernel_data") == {"eager/gdn/H100_9b_decode.csv": "g\n"}
    assert "kernel_tables" in json.loads(
        (data / "kernel_data" / "qbench_manifest.json").read_text())
    for argv in (["--only", "kernel_tables", "--dest", str(dest)], ["--only", "tp_comm"],
                 ["--kernel-dest", str(data), "--only", "tp_comm"]):
        with pytest.raises(SystemExit):
            main(["--results", str(results), *argv])


GEMM = "ncu/gemm/H100.csv"


@pytest.mark.parametrize("bad", ["M,N,K,us\r\n2,2,2,6,99\r\n", "M,N,K,us\r\n2,2\r\n",
                                 "M,N,K,us\r\n,2,2,6\r\n", "M,N,K,us\r\nx,2,2,6\r\n",
                                 "M,N,K,us\r\n1.0,1,1,9\r\n"])
def test_a_malformed_row_stops_the_export_and_leaves_dest_as_it_was(tmp_path: Path, bad) -> None:
    """An extra column, a truncated row, an empty or unparsable key (gemm's
    M is an int, as the old upsert's int() read it): nothing is replaced."""
    results, dest = tmp_path / "results", tmp_path / "out"
    probe_run(results, "gdn", {"cuda_event/gdn/H100_9b_decode.csv": "g\n"}, 0)
    probe_run(results, "gemm", {GEMM: "M,N,K,us\r\n1,1,1,5\r\n"}, 1)
    export([results], kernel_dest=dest, only=["kernel_tables"])
    before = tree(dest)
    probe_run(results, "gdn", {"cuda_event/gdn/H100_9b_decode.csv": "g2\n"}, 2)
    probe_run(results, "gemm", {GEMM: bad}, 3, args=["--append"])
    with pytest.raises(SystemExit, match=r"graph/gemm/H100.csv: .*20261011T120003.000000Z row 1 malformed"):
        export([results], kernel_dest=dest, only=["kernel_tables"], replace_tables=True)
    assert tree(dest) == before


@pytest.mark.parametrize("bad", ['{"y":', "[1]"])
def test_bad_or_non_object_json_stops_the_export(tmp_path: Path, bad) -> None:
    results, dest = tmp_path / "results", tmp_path / "out"
    table = "ncu/elementwise/H100.json"
    probe_run(results, "gdn", {"cuda_event/gdn/H100_9b_decode.csv": "g\n"}, 0)
    probe_run(results, "elementwise", {table: '{"x": 1}'}, 1)
    probe_run(results, "gemm", {table: bad}, 2)
    with pytest.raises(SystemExit, match="graph/elementwise/H100.json: .*malformed"):
        export([results], kernel_dest=dest, only=["kernel_tables"])
    assert not dest.exists()


def test_numeric_keys_compare_as_numbers_and_a_runs_last_row_wins(tmp_path: Path) -> None:
    """total_weight_gb 16 and 16.0 are one key; within one run's table the
    last row on a key wins."""
    head = ("gpu", "world", "tp_from", "tp_to", "total_weight_gb", "comm_ms")
    results, dest = tmp_path / "results", tmp_path / "out"
    table = "cuda_event/reparallel/H100_reshard.csv"
    probe_run(results, "reparallel", {table: csv_text([head, ("H100", 2, 1, 2, 16, 5.0)])}, 0)
    probe_run(results, "reparallel", {table: csv_text([head, ("H100", 2, 1, 2, 16.0, 9.0),
                                                       ("H100", 2, 1, 2, 16.0, 8.0)])}, 1)
    export([results], kernel_dest=dest, only=["kernel_tables"])
    assert rows(dest / "kernel_data" / "eager/reparallel/H100_reshard.csv") == [
        list(head), ["H100", "2", "1", "2", "16.0", "8.0"]]


def test_a_gemm_csv_beside_the_elementwise_json_is_not_unioned(tmp_path: Path) -> None:
    """Only gemm's elementwise .json unions across runs; any other file there
    is the latest run's."""
    results, dest = tmp_path / "results", tmp_path / "out"
    probe_run(results, "gemm", {"ncu/elementwise/H100.csv": "a\n1\n"}, 0)
    probe_run(results, "gemm", {"ncu/elementwise/H100.csv": "a\n2\n"}, 1)
    export([results], kernel_dest=dest, only=["kernel_tables"])
    assert tree(dest / "kernel_data") == {"graph/elementwise/H100.csv": "a\n2\n"}


def test_finish_times_compare_as_instants(tmp_path: Path) -> None:
    """01:00+01:00 is 00:00Z, before 00:30Z; a missing time sorts oldest."""
    results, dest = tmp_path / "results", tmp_path / "out"
    probe_run(results, "gdn", {"cuda_event/gdn/H100_9b_decode.csv": "later\n"}, 0)
    probe_run(results, "gdn", {"cuda_event/gdn/H100_9b_decode.csv": "earlier\n"}, 1)
    probe_run(results, "gdn", {"cuda_event/gdn/H100_9b_decode.csv": "none\n"}, 2)
    gdn = results / "probes" / "H100" / "gdn"
    for name, finished in (("20261011T120000.000000Z", "2026-01-02T00:30:00+00:00"),
                           ("20261011T120001.000000Z", "2026-01-02T01:00:00+01:00"),
                           ("20261011T120002.000000Z", "not a time")):
        doc = json.loads((gdn / name / "run.json").read_text())
        doc["provenance"]["finished"] = finished
        (gdn / name / "run.json").write_text(json.dumps(doc))
    export([results], kernel_dest=dest, only=["kernel_tables"])
    assert tree(dest / "kernel_data")["eager/gdn/H100_9b_decode.csv"] == "later\n"


# QuettaSim origin/main's kernel_data tables and mode vocabulary, captured once.
QUETTASIM = json.loads((Path(__file__).parent / "quettasim_kernel_data.json").read_text())
FA = "n_heads,n_kv_heads,head_dim,q_len,us\r\n32,8,128,1,5\r\n"
MOE = "tokens,n_experts,top_k,intermediate,hidden,us\r\n8,128,8,768,2048,5\r\n"
RESHARD = "gpu,world,tp_from,tp_to,total_weight_gb,ms\r\nH200,8,4,2,16,3\r\n"
GEMM_ROW = "M,N,K,us\r\n1,1,1,5\r\n"


@pytest.mark.parametrize("probe,gpu,args,tables,want", [
    ("gemm", "H200", ["--graph"], {"ncu/gemm/H200.csv": GEMM_ROW},
     {"graph/gemm/H200.csv": "graph_replay"}),
    ("gemm", "H200", ["--graph", "--dtype", "fp8"], {"ncu/gemm/H200_fp8.csv": GEMM_ROW},
     {"graph/gemm/H200_fp8.csv": "graph_replay"}),
    ("gemm", "H200", ["--graph", "--wide"], {"ncu/gemm_wide/H200.csv": GEMM_ROW},
     {"graph/gemm_wide/H200.csv": "graph_replay"}),
    ("gemm", "H100", ["--ncu"], {"ncu/gemm/H100.csv": GEMM_ROW}, {"graph/gemm/H100.csv": "ncu"}),
    ("gemm", "H200", [], {"cuda_event/gemm/H200.csv": GEMM_ROW}, {"eager/gemm/H200.csv": "eager"}),
    ("elementwise", "H200", ["--graph", "--vllm"], {"ncu/elementwise/H200.json": "{}"},
     {"graph/elementwise/H200.json": "graph_replay"}),
    ("elementwise", "H200", ["--vllm"], {"cuda_event/elementwise/H200.json": "{}"},
     {"eager/elementwise/H200.json": "eager"}),
    ("flash_attn", "H200", ["--graph"],
     {"ncu/flash_attn/H200.csv": FA, "cuda_event/fa3_prefill/H200.csv": FA},
     {"graph/flash_attn/H200.csv": "graph_replay", "eager/fa3_prefill/H200.csv": "eager"}),
    ("flash_attn", "H200", [], {"cuda_event/flash_attn/H200.csv": FA},
     {"eager/flash_attn/H200.csv": "eager"}),
    ("cross_attn", "H200", [], {"cuda_event/fa3_cross/H200.csv": FA}, {"eager/fa3_cross/H200.csv": "eager"}),
    ("moe", "H200", ["--graph"], {"ncu/moe/H200.csv": MOE, "ncu/moe/H200_routing.json": "{}"},
     {"graph/moe/H200.csv": "graph_replay", "graph/moe/H200_routing.json": "graph_replay"}),
    ("moe", "H200", [], {"cuda_event/moe/H200.csv": MOE, "cuda_event/moe/H200_routing.json": "{}"},
     {"eager/moe/H200.csv": "eager", "eager/moe/H200_routing.json": "eager"}),
    ("gdn", "A100", [], {"cuda_event/gdn/A100_9b_decode.csv": "g\n", "cuda_event/gdn/A100_9b_prefill.csv": "g\n"},
     {"eager/gdn/A100_9b_decode.csv": "eager", "eager/gdn/A100_9b_prefill.csv": "eager"}),
    ("collective", "H200", [], {"ncu/collectives/H200_tp2.csv": COLLECTIVE},
     {"graph/collectives/H200_tp2.csv": "amortized_wallclock"}),
    ("vllm_allreduce", "H200", [], {"ncu/collectives/H200_tp2_nccl_vs_vllm.csv": "b\n"},
     {"graph/collectives/H200_tp2_nccl_vs_vllm.csv": "reference"}),
    ("reparallel", "H200", [], {"cuda_event/reparallel/H200_reshard.csv": RESHARD},
     {"eager/reparallel/H200_reshard.csv": "host_wallclock"}),
])
def test_each_probe_exports_to_a_quettasim_table_path(tmp_path: Path, probe, gpu, args, tables, want) -> None:
    """Every exported table is a path QuettaSim main holds, with a sidecar
    whose mode is in its vocabulary and belongs in the table's dir."""
    results, dest = tmp_path / "results", tmp_path / "out"
    probe_run(results, probe, tables, 0, gpu=gpu, args=args, finished=0)
    export([results], kernel_dest=dest, only=["kernel_tables"])
    assert sorted(tree(dest / "kernel_data")) == sorted(want)
    for rel, mode in want.items():
        assert rel in QUETTASIM["tables"]
        meta = json.loads((dest / "kernel_data" / f"{rel}.meta.json").read_text())
        assert meta["mode"] == mode
        assert QUETTASIM["modes"][mode] in (None, rel.split("/")[0])
        assert (meta["schema"], meta["table"], meta["dir"], meta["gpu"]) == (1, rel, rel.split("/")[0], gpu)
        assert meta["tool"] == f"quettabench.probes.{probe}"
        assert meta["measured"] == "2026-10-11"


KV_HEAD = ("gpu", "kv_heads_per_rank", "head_dim", "block_size", "layers", "layout", "tokens",
           "n_descs", "bytes", "prep_ms", "xfer_ms", "total_ms", "eff_gbps")
KV_TABLE = "cuda_event/kv_transfer/H200_nixl.csv"


def test_kv_transfer_run_exports_to_eager_with_a_host_wallclock_sidecar(tmp_path: Path) -> None:
    results, dest = tmp_path / "results", tmp_path / "out"
    probe_run(results, "kv_transfer", {KV_TABLE: csv_text([KV_HEAD, ("H200", 1, 128, 16, 94, "contig", 16,
                                                                     94, 770048, 0.008, 0.491, 0.499, 1.54)])},
              0, gpu="H200", finished=0)
    export([results], kernel_dest=dest, only=["kernel_tables"])
    rel = "eager/kv_transfer/H200_nixl.csv"
    assert sorted(tree(dest / "kernel_data")) == [rel]
    assert rel in QUETTASIM["tables"]
    meta = json.loads((dest / "kernel_data" / f"{rel}.meta.json").read_text())
    assert (meta["mode"], meta["dir"], meta["gpu"]) == ("host_wallclock", "eager", "H200")
    assert QUETTASIM["modes"]["host_wallclock"] == "eager"
    assert meta["tool"] == "quettabench.profiling.kv_transfer"


def test_kv_transfer_rows_union_on_the_old_upsert_key(tmp_path: Path) -> None:
    """The row key is kv_transfer.py's KEY, the columns its upsert replaced on;
    layout stays a string, and the union is sorted on the key."""
    source = (Path(quettasim.__file__).parents[1] / "profiling" / "kv_transfer.py").read_text()
    (key,) = [ast.literal_eval(n.value) for n in ast.parse(source).body
              if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "KEY"]
    assert quettasim.ROW_KEYS[("kv_transfer", "kv_transfer")] == key
    results, dest = tmp_path / "results", tmp_path / "out"
    row = ("H200", 1, 128, 16, 94)
    probe_run(results, "kv_transfer", {KV_TABLE: csv_text([KV_HEAD, (*row, "runs8", 16, 0, 0, 0, 0, 2.0, 0),
                                                         (*row, "contig", 1024, 0, 0, 0, 0, 1.0, 0)])},
              0, gpu="H200")
    probe_run(results, "kv_transfer", {KV_TABLE: csv_text([KV_HEAD, (*row, "contig", 1024, 0, 0, 0, 0, 0.5, 0),
                                                         (*row, "contig", 16, 0, 0, 0, 0, 0.4, 0)])},
              1, gpu="H200")
    export([results], kernel_dest=dest, only=["kernel_tables"])
    got = rows(dest / "kernel_data" / "eager/kv_transfer/H200_nixl.csv")
    assert [(r[5], r[6], r[11]) for r in got[1:]] == [("contig", "16", "0.4"), ("contig", "1024", "0.5"),
                                                      ("runs8", "16", "2.0")]


def test_a_table_from_runs_timed_two_ways_has_mode_unknown(tmp_path: Path) -> None:
    results, dest = tmp_path / "results", tmp_path / "out"
    probe_run(results, "gemm", {"ncu/gemm/H100.csv": GEMM_ROW}, 0, args=["--ncu"])
    probe_run(results, "gemm", {"ncu/gemm/H100.csv": "M,N,K,us\r\n2,2,2,5\r\n"}, 1,
              args=["--graph", "--append"])
    export([results], kernel_dest=dest, only=["kernel_tables"])
    meta = json.loads((dest / "kernel_data" / "graph/gemm/H100.csv.meta.json").read_text())
    assert meta["mode"] == "unknown"


def test_a_stopped_export_leaves_no_kernel_dest_it_made(tmp_path: Path) -> None:
    results, dest = tmp_path / "results", tmp_path / "new" / "data"
    probe_run(results, "gdn", {"cuda_event/gdn/H100_9b_decode.csv": "g\n"}, 0)
    probe_run(results, "gemm", {GEMM: "M,N,K,us\r\nx,1,1,5\r\n"}, 1)
    with pytest.raises(SystemExit, match="malformed"):
        export([results], kernel_dest=dest, only=["kernel_tables"])
    assert not (tmp_path / "new").exists()


def test_no_tables_found_writes_no_manifest(tmp_path: Path) -> None:
    results, dest = tmp_path / "results", tmp_path / "data"
    probe_run(results, "fused_block", {"validation/v.csv": "v\n"}, 0)
    assert export([results], kernel_dest=dest, only=["kernel_tables"]) == []
    assert not dest.exists()


def test_an_empty_csv_stops_the_export_naming_its_run(tmp_path: Path) -> None:
    results = tmp_path / "results"
    probe_run(results, "gemm", {GEMM: ""}, 0)
    with pytest.raises(SystemExit, match=r"graph/gemm/H100.csv: .*20261011T120000.000000Z .*no header"):
        quettasim.run_kernel_tables([results], tmp_path / "out")


def test_unparsable_probe_args_stop_the_export_naming_the_run(tmp_path: Path) -> None:
    results = tmp_path / "results"
    probe_run(results, "gemm", {GEMM: GEMM_ROW}, 0)
    record = results / "probes" / "H100" / "gemm" / "20261011T120000.000000Z" / "run.json"
    doc = json.loads(record.read_text())
    doc["config"]["mode"]["probe_args"] = 'a "b'
    record.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match=r"20261011T120000.000000Z.*probe_args"):
        quettasim.run_kernel_tables([results], tmp_path / "out")


def seed(dest: Path, rel: str, text: str, meta: dict | None = None) -> Path:
    """A table QuettaSim already holds at dest/kernel_data/rel, with its sidecar."""
    path = dest / "kernel_data" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if meta is not None:
        path.with_name(path.name + ".meta.json").write_text(json.dumps(meta))
    return path


def test_a_dest_table_keeps_its_rows_and_a_newer_run_wins_a_key(tmp_path: Path) -> None:
    """The dest table is the oldest source: its rows stay unless a run holds
    their key, and a run's row replaces the dest row on the same key."""
    results, dest = tmp_path / "results", tmp_path / "data"
    table = seed(dest, "graph/gemm/H100.csv", csv_text([GEMM_HEAD, (1, 1, 1, 2, 3.0), (2, 2, 2, 2, 4.0)]))
    probe_run(results, "gemm", {GEMM: csv_text([GEMM_HEAD, (2, 2, 2, 2, 9.0), (3, 3, 3, 2, 5.0)])}, 0,
              args=["--append"])
    export([results], kernel_dest=dest, only=["kernel_tables"])
    assert rows(table) == [list(GEMM_HEAD), ["1", "1", "1", "2", "3.0"], ["2", "2", "2", "2", "9.0"],
                           ["3", "3", "3", "2", "5.0"]]
    runs = json.loads((dest / "kernel_data" / "qbench_manifest.json").read_text())["kernel_tables"]
    assert [Path(r).name for r in runs["graph/gemm/H100.csv"]["runs"]] == [
        "adopted from dest", "20261011T120000.000000Z"]


def test_a_dest_table_with_other_columns_unions_them(tmp_path: Path) -> None:
    results, dest = tmp_path / "results", tmp_path / "data"
    table = seed(dest, "graph/gemm/H100.csv", csv_text([("M", "N", "K", "us", "note"), (1, 1, 1, 3, "x")]))
    probe_run(results, "gemm", {GEMM: csv_text([("M", "N", "K", "us", "dtype_bytes"), (2, 2, 2, 4, 2)])}, 0,
              args=["--append"])
    export([results], kernel_dest=dest, only=["kernel_tables"])
    assert rows(table) == [["M", "N", "K", "us", "note", "dtype_bytes"], ["1", "1", "1", "3", "x", ""],
                           ["2", "2", "2", "4", "", "2"]]


def test_a_dest_json_table_is_the_oldest_source(tmp_path: Path) -> None:
    results, dest = tmp_path / "results", tmp_path / "data"
    table = seed(dest, "graph/elementwise/H100.json", '{"rms_norm": 1, "silu": 1}')
    probe_run(results, "elementwise", {"ncu/elementwise/H100.json": '{"rms_norm": 2}'}, 0)
    export([results], kernel_dest=dest, only=["kernel_tables"])
    assert json.loads(table.read_text()) == {"rms_norm": 2, "silu": 1}


@pytest.mark.parametrize("probe,rel,old,new", [
    ("gemm", "graph/gemm/H100.csv", csv_text([GEMM_HEAD, (1, 1, 1, 2, 3.0), (2, 2, 2, 2, 4.0)]),
     {GEMM: csv_text([GEMM_HEAD, (2, 2, 2, 2, 9.0)])}),
    ("gdn", "eager/gdn/H100_9b_decode.csv", "batch,us\n1,5\n2,6\n",
     {"cuda_event/gdn/H100_9b_decode.csv": "batch,us\n1,5\n"}),
])
def test_dropping_dest_rows_stops_unless_replace_tables(tmp_path: Path, capsys, probe, rel, old, new) -> None:
    """A gemm run without --append, or a table with no row key, would drop
    rows dest holds: the export stops naming the file and the row counts,
    and --replace-tables (through the qbench export door) replaces it."""
    from quettabench import cli
    results, dest = tmp_path / "results", tmp_path / "data"
    table = seed(dest, rel, old)
    probe_run(results, probe, new, 0)
    argv = ["export", "--results", str(results), "--kernel-dest", str(dest), "--only", "kernel_tables"]
    with pytest.raises(SystemExit, match=rf"{rel}: .* keep 1 of its 2 rows.*--replace-tables"):
        cli.main(argv)
    assert table.read_bytes() == old.encode()
    cli.main([*argv, "--replace-tables"])
    assert table.read_bytes() == next(iter(new.values())).encode()


def test_an_export_over_its_own_tables_changes_nothing(tmp_path: Path) -> None:
    results, dest = tmp_path / "results", tmp_path / "data"
    probe_run(results, "gdn", {"cuda_event/gdn/H100_9b_decode.csv": "batch,us\n1,5\n"}, 0)
    probe_run(results, "gemm", {GEMM: csv_text([GEMM_HEAD, (1, 1, 1, 2, 3.0)])}, 1)
    export([results], kernel_dest=dest, only=["kernel_tables"])
    before = tree(dest)
    export([results], kernel_dest=dest, only=["kernel_tables"])
    assert tree(dest) == before


def test_the_dest_sidecar_keeps_its_notes(tmp_path: Path) -> None:
    """Fields qbench does not own (notes, and any QuettaSim adds) survive;
    the ones it owns follow the export."""
    results, dest = tmp_path / "results", tmp_path / "data"
    rel = "graph/gemm/H100.csv"
    table = seed(dest, rel, csv_text([GEMM_HEAD, (1, 1, 1, 2, 3.0)]),
                 {"schema": 1, "table": rel, "dir": "graph", "gpu": "H100", "mode": "graph_replay",
                  "tool": "gemm_probe.py --graph", "reduce": "min", "measured": "2026-08-20",
                  "notes": "NCU counters locked", "reviewed_by": "kw", "provenance": "backfilled"})
    probe_run(results, "gemm", {GEMM: csv_text([GEMM_HEAD, (2, 2, 2, 2, 4.0)])}, 0,
              args=["--graph", "--append"], finished=0)
    export([results], kernel_dest=dest, only=["kernel_tables"])
    meta = json.loads(table.with_name(table.name + ".meta.json").read_text())
    assert (meta["notes"], meta["reviewed_by"]) == ("NCU counters locked", "kw")
    assert (meta["tool"], meta["measured"], meta["mode"]) == (
        "gemm_probe.py --graph + quettabench.probes.gemm", "2026-08-20 .. 2026-10-11", "graph_replay")


def test_kept_dest_rows_timed_another_way_make_mode_unknown(tmp_path: Path) -> None:
    results, dest = tmp_path / "results", tmp_path / "data"
    rel = "graph/gemm/H100.csv"
    table = seed(dest, rel, csv_text([GEMM_HEAD, (1, 1, 1, 2, 3.0)]), {"mode": "ncu", "notes": ""})
    probe_run(results, "gemm", {GEMM: csv_text([GEMM_HEAD, (2, 2, 2, 2, 4.0)])}, 0,
              args=["--graph", "--append"])
    export([results], kernel_dest=dest, only=["kernel_tables"])
    assert json.loads(table.with_name(table.name + ".meta.json").read_text())["mode"] == "unknown"


def test_the_manifest_merges_per_table(tmp_path: Path) -> None:
    """A second export of other tables keeps the first export's entries; each
    entry names its runs, bench commit and export time."""
    first, second, dest = tmp_path / "a", tmp_path / "b", tmp_path / "data"
    probe_run(first, "gemm", {GEMM: GEMM_ROW}, 0)
    export([first], kernel_dest=dest, only=["kernel_tables"])
    probe_run(second, "flash_attn", {"ncu/flash_attn/H100.csv": FA}, 1)
    export([second], kernel_dest=dest, only=["kernel_tables"])
    manifest = json.loads((dest / "kernel_data" / "qbench_manifest.json").read_text())
    assert sorted(manifest["kernel_tables"]) == ["graph/flash_attn/H100.csv", "graph/gemm/H100.csv"]
    gemm = manifest["kernel_tables"]["graph/gemm/H100.csv"]
    assert sorted(gemm) == ["bench_sha", "exported_at", "runs"]
    assert [Path(r).parents[3].name for r in gemm["runs"]] == ["a"]


QS_FA_HEAD = ("q_len", "kv_len", "n_heads", "n_kv_heads", "head_dim", "batch", "causal", "phase",
              "layers", "latency_us")


def test_a_flash_attn_run_over_quettasim_keeps_the_other_head_configs(tmp_path: Path) -> None:
    """QuettaSim's graph/flash_attn holds several head configs; a run of one
    replaces that config's rows only."""
    results, dest = tmp_path / "results", tmp_path / "data"
    heads = [(32, 4, 128), (32, 8, 128), (64, 8, 128)]
    table = seed(dest, "graph/flash_attn/H200.csv", csv_text(
        [QS_FA_HEAD, *[(1, kv, *h, 1, False, "decode", "", 9.0) for h in heads for kv in (128, 256)]]))
    probe_run(results, "flash_attn", {"ncu/flash_attn/H200.csv": csv_text(
        [QS_FA_HEAD, (1, 512, 32, 8, 128, 1, False, "decode", "", 5.0)])}, 0, gpu="H200", args=["--graph"])
    export([results], kernel_dest=dest, only=["kernel_tables"])
    got = rows(table)[1:]
    assert sorted({tuple(r[2:5]) for r in got}) == [("32", "4", "128"), ("32", "8", "128"), ("64", "8", "128")]
    assert [r[1] for r in got if r[2:5] == ["32", "8", "128"]] == ["512"]
    assert len(got) == 5


def test_a_vllm_allreduce_run_over_quettasim_collectives_keeps_the_nccl_ops(tmp_path: Path) -> None:
    """QuettaSim's <gpu>_tp<N>.csv holds every op; a vllm_allreduce run's
    all_reduce rows replace every all_reduce row of its world, as its upsert
    did, and the other ops stay."""
    results, dest = tmp_path / "results", tmp_path / "data"
    head = ("op", "world", "bytes", "latency_us", "dtype_bytes", "path")
    old = [(op, 2, b, 10.0, 2, "nccl") for op in ("all_gather", "all_to_all", "reduce_scatter")
           for b in (2048, 4096)] + [("all_reduce", 2, b, 8.0, 2, "vllm") for b in (2048, 4096)]
    table = seed(dest, "graph/collectives/H200_tp2.csv", csv_text([head, *old]))
    probe_run(results, "vllm_allreduce", {"ncu/collectives/H200_tp2.csv": csv_text(
        [head, ("all_reduce", 2, 4096, 7.0, 2, "vllm")])}, 0, gpu="H200")
    export([results], kernel_dest=dest, only=["kernel_tables"])
    got = rows(table)
    assert got[0] == list(head)
    assert sorted({r[0] for r in got[1:]}) == ["all_gather", "all_reduce", "all_to_all", "reduce_scatter"]
    assert [r for r in got if r[0] == "all_reduce"] == [["all_reduce", "2", "4096", "7.0", "2", "vllm"]]
    assert len(got) == 8


def test_a_gemm_append_run_over_quettasim_keeps_its_rows(tmp_path: Path) -> None:
    results, dest = tmp_path / "results", tmp_path / "data"
    old = [(m, 4096, 4096, 2, 1.0 + m) for m in range(1, 200)]
    table = seed(dest, "graph/gemm/H200.csv", csv_text([GEMM_HEAD, *old]))
    probe_run(results, "gemm", {"ncu/gemm/H200.csv": csv_text([GEMM_HEAD, (1, 4096, 4096, 2, 5.0)])}, 0,
              gpu="H200", args=["--graph", "--append"])
    export([results], kernel_dest=dest, only=["kernel_tables"])
    got = rows(table)[1:]
    assert len(got) == 199 and got[-1] == ["1", "4096", "4096", "2", "5.0"]


def test_a_stopped_merge_leaves_the_manifest_as_it_was(tmp_path: Path) -> None:
    """The manifest is checked before any table moves, and a stop writes neither."""
    results, dest = tmp_path / "results", tmp_path / "data"
    probe_run(results, "gemm", {GEMM: GEMM_ROW}, 0)
    export([results], kernel_dest=dest, only=["kernel_tables"])
    before = (dest / "kernel_data" / "qbench_manifest.json").read_text()
    seed(dest, "eager/gdn/H100_9b_decode.csv", "batch,us\n1,5\n")
    probe_run(results, "gdn", {"cuda_event/gdn/H100_9b_decode.csv": "batch,us\n2,6\n"}, 1)
    with pytest.raises(SystemExit, match="keep 0 of its 1 rows"):
        export([results], kernel_dest=dest, only=["kernel_tables"])
    assert (dest / "kernel_data" / "qbench_manifest.json").read_text() == before


def test_an_unreadable_source_run_stops_before_a_table_moves(tmp_path: Path) -> None:
    """The kernel manifest reads the runs before kernel_tables builds, so its
    sidecars never point at a manifest that was not written."""
    results, dest = tmp_path / "results", tmp_path / "data"
    probe_run(results, "gemm", {GEMM: GEMM_ROW}, 0)
    (results / "other").mkdir()
    (results / "other" / "run.json").write_text(json.dumps({"run_status": {"status": "ok"}, "config": {}}))
    with pytest.raises(KeyError):
        export([results], kernel_dest=dest, only=["kernel_tables"])
    assert not dest.exists()


def test_repeated_dest_samples_survive_an_append_on_another_key(tmp_path: Path) -> None:
    """QuettaSim's gemm tables repeat (M, N, K) samples, which its loader
    reduces by min: an --append run on another key keeps every one."""
    results, dest = tmp_path / "results", tmp_path / "data"
    old = [GEMM_HEAD, (1, 1, 1, 2, 3.0), (1, 1, 1, 2, 2.5), (2, 2, 2, 2, 4.0)]
    table = seed(dest, "graph/gemm/H100.csv", csv_text(old))
    probe_run(results, "gemm", {GEMM: csv_text([GEMM_HEAD, (3, 3, 3, 2, 5.0)])}, 0, args=["--append"])
    export([results], kernel_dest=dest, only=["kernel_tables"])
    assert rows(table) == [[str(c) for c in r] for r in [*old, (3, 3, 3, 2, 5.0)]]


def test_a_run_on_a_repeated_dest_key_replaces_every_sample(tmp_path: Path) -> None:
    results, dest = tmp_path / "results", tmp_path / "data"
    table = seed(dest, "graph/gemm/H100.csv",
                 csv_text([GEMM_HEAD, (1, 1, 1, 2, 3.0), (2, 2, 2, 2, 4.0), (1, 1, 1, 2, 2.5)]))
    probe_run(results, "gemm", {GEMM: csv_text([GEMM_HEAD, (1, 1, 1, 2, 9.0)])}, 0, args=["--append"])
    export([results], kernel_dest=dest, only=["kernel_tables"])
    assert rows(table) == [list(GEMM_HEAD), ["2", "2", "2", "2", "4.0"], ["1", "1", "1", "2", "9.0"]]


def test_many_repeated_dest_samples_all_stay(tmp_path: Path) -> None:
    """63 dest rows over 38 keys, as QuettaSim's graph/gemm/H100.csv: an
    --append run on a new key keeps all 63 and adds its one."""
    results, dest = tmp_path / "results", tmp_path / "data"
    old = [(m, 4096, 4096, 2, 1.0 + i) for i, m in enumerate([*range(1, 39), *range(1, 26)])]
    table = seed(dest, "graph/gemm/H100.csv", csv_text([GEMM_HEAD, *old]))
    probe_run(results, "gemm", {GEMM: csv_text([GEMM_HEAD, (8192, 8192, 8192, 2, 999.0)])}, 0,
              args=["--ncu", "--append"])
    export([results], kernel_dest=dest, only=["kernel_tables"])
    got = rows(table)[1:]
    assert len(got) == 64 and got[:63] == [[str(c) for c in r] for r in old]


def test_a_dict_upserted_dest_table_keeps_its_last_row_on_a_key(tmp_path: Path) -> None:
    """reparallel's old upsert held its rows in a dict, so a dest key repeated
    keeps its last row, as the run's own table does."""
    head = ("gpu", "world", "tp_from", "tp_to", "total_weight_gb", "comm_ms")
    results, dest = tmp_path / "results", tmp_path / "data"
    table = seed(dest, "eager/reparallel/H100_reshard.csv",
                 csv_text([head, ("H100", 2, 1, 2, 16, 5.0), ("H100", 2, 1, 2, 16, 6.0)]))
    probe_run(results, "reparallel", {"cuda_event/reparallel/H100_reshard.csv":
                                      csv_text([head, ("H100", 4, 1, 4, 16, 7.0)])}, 0)
    export([results], kernel_dest=dest, only=["kernel_tables"])
    assert rows(table) == [list(head), ["H100", "2", "1", "2", "16", "6.0"], ["H100", "4", "1", "4", "16", "7.0"]]


def test_dropping_a_dest_column_stops_unless_replace_tables(tmp_path: Path) -> None:
    results, dest = tmp_path / "results", tmp_path / "data"
    old = csv_text([("M", "N", "K", "us", "note"), (1, 1, 1, 3, "x")])
    table = seed(dest, "graph/gemm/H100.csv", old)
    probe_run(results, "gemm", {GEMM: csv_text([("M", "N", "K", "us"), (1, 1, 1, 4)])}, 0)
    with pytest.raises(SystemExit, match=r"graph/gemm/H100.csv: .*drop column note.*--replace-tables"):
        export([results], kernel_dest=dest, only=["kernel_tables"])
    assert table.read_bytes() == old.encode()
    export([results], kernel_dest=dest, only=["kernel_tables"], replace_tables=True)
    assert rows(table) == [["M", "N", "K", "us"], ["1", "1", "1", "4"]]


def test_replace_tables_replaces_only_the_tables_that_would_stop(tmp_path: Path) -> None:
    """A gemm --append merge stays a merge under --replace-tables; a gdn
    table that would lose rows is replaced."""
    results, dest = tmp_path / "results", tmp_path / "data"
    gemm = seed(dest, "graph/gemm/H100.csv", csv_text([GEMM_HEAD, (1, 1, 1, 2, 3.0), (2, 2, 2, 2, 4.0)]))
    gdn = seed(dest, "eager/gdn/H100_9b_decode.csv", "batch,us\n1,5\n2,6\n")
    probe_run(results, "gemm", {GEMM: csv_text([GEMM_HEAD, (3, 3, 3, 2, 5.0)])}, 0, args=["--append"])
    probe_run(results, "gdn", {"cuda_event/gdn/H100_9b_decode.csv": "batch,us\n1,7\n"}, 1)
    with pytest.raises(SystemExit, match="keep 0 of its 2 rows"):
        export([results], kernel_dest=dest, only=["kernel_tables"])
    export([results], kernel_dest=dest, only=["kernel_tables"], replace_tables=True)
    assert len(rows(gemm)) == 4
    assert gdn.read_text() == "batch,us\n1,7\n"


TP4_HEAD = ("op", "world", "bytes", "latency_us", "dtype_bytes")


def test_a_collective_rerun_over_dest_updates_its_all_reduce_rows(tmp_path: Path) -> None:
    """With no vllm_allreduce run beside it, a collective run's all_reduce
    rows are its own measurement and replace dest's; a dest table with no
    path column gets none."""
    results, dest = tmp_path / "results", tmp_path / "data"
    old = [TP4_HEAD, ("all_reduce", 4, 2048, 26.0, 2), ("all_gather", 4, 2048, 13.0, 2)]
    table = seed(dest, "graph/collectives/H200_tp4.csv", csv_text(old))
    probe_run(results, "collective", {"ncu/collectives/H200_tp4.csv": csv_text(
        [TP4_HEAD, ("all_reduce", 4, 2048, 99.0, 2), ("all_gather", 4, 2048, 98.0, 2)])}, 0, gpu="H200")
    export([results], kernel_dest=dest, only=["kernel_tables"])
    assert rows(table) == [list(TP4_HEAD), ["all_reduce", "4", "2048", "99.0", "2"],
                           ["all_gather", "4", "2048", "98.0", "2"]]


def test_collectives_over_dest_match_a_fresh_export(tmp_path: Path) -> None:
    fresh, over = tmp_path / "fresh", tmp_path / "over"
    results = tmp_path / "results"
    probe_run(results, "collective", {"ncu/collectives/H100_tp2.csv": COLLECTIVE}, 0)
    probe_run(results, "vllm_allreduce", {"ncu/collectives/H100_tp2.csv": VLLM}, 1)
    export([results], kernel_dest=fresh, only=["kernel_tables"])
    seed(over, "graph/collectives/H100_tp2.csv", csv_text([
        ("op", "world", "bytes", "latency_us", "dtype_bytes", "path"), ("all_gather", 2, 4096, 1.0, 2, "nccl"),
        ("all_reduce", 2, 2048, 1.0, 2, "vllm"), ("", 2, 1024, 1.0, 2, "nccl")]))
    export([results], kernel_dest=over, only=["kernel_tables"])
    assert tree(over / "kernel_data") == tree(fresh / "kernel_data")


def test_two_fresh_exports_in_a_row_give_the_same_tree(tmp_path: Path) -> None:
    results, first, second = tmp_path / "results", tmp_path / "a", tmp_path / "b"
    probe_run(results, "gemm", {GEMM: GEMM_ROW, "ncu/elementwise/H100.json": '{"fp8_act_quant": 1}'}, 0)
    probe_run(results, "elementwise", {"ncu/elementwise/H100.json": '{"rms_norm": 1}'}, 1)
    probe_run(results, "moe", {"ncu/moe/H100_routing.json": '{\n "a": 1\n}\n'}, 2)
    probe_run(results, "collective", {"ncu/collectives/H100_tp2.csv": COLLECTIVE}, 3)
    probe_run(results, "vllm_allreduce", {"ncu/collectives/H100_tp2.csv": VLLM}, 4)
    export([results], kernel_dest=first, only=["kernel_tables"])
    export([results], kernel_dest=second, only=["kernel_tables"])
    once = tree(first / "kernel_data")
    export([results], kernel_dest=second, only=["kernel_tables"])
    assert tree(second / "kernel_data") == once


def test_an_export_of_unchanged_tables_leaves_dest_bytes_alone(tmp_path: Path) -> None:
    """Runs that hold what dest holds rewrite nothing: JSON keeps its
    indent and final newline (or none), CSV its line ending."""
    results, dest = tmp_path / "results", tmp_path / "data"
    tables = {"graph/elementwise/H100.json": '{\n    "rms_norm": {\n        "floor_us": 1.5\n    }\n}',
              "graph/moe/H100_routing.json": '{\n "a": {\n  "floor_us": 2.0\n }\n}\n',
              "graph/gemm/H100.csv": "M,N,K,dtype_bytes,latency_us\n1,1,1,2,3.0\n"}
    for rel, text in tables.items():
        seed(dest, rel, text)
    probe_run(results, "elementwise", {"ncu/elementwise/H100.json": tables["graph/elementwise/H100.json"]}, 0)
    probe_run(results, "moe", {"ncu/moe/H100_routing.json": tables["graph/moe/H100_routing.json"]}, 1)
    probe_run(results, "gemm", {GEMM: tables["graph/gemm/H100.csv"]}, 2, args=["--append"])
    export([results], kernel_dest=dest, only=["kernel_tables"])
    assert {rel: (dest / "kernel_data" / rel).read_text() for rel in tables} == tables


def test_the_manifest_chains_runs_across_exports(tmp_path: Path) -> None:
    """Dest rows a later export keeps carry the runs an earlier export
    recorded for them, then the new runs."""
    first, second, dest = tmp_path / "a", tmp_path / "b", tmp_path / "data"
    probe_run(first, "gemm", {GEMM: csv_text([GEMM_HEAD, (1, 1, 1, 2, 3.0)])}, 0)
    export([first], kernel_dest=dest, only=["kernel_tables"])
    probe_run(second, "gemm", {GEMM: csv_text([GEMM_HEAD, (2, 2, 2, 2, 3.0)])}, 1, args=["--append"])
    export([second], kernel_dest=dest, only=["kernel_tables"])
    runs = json.loads((dest / "kernel_data" / "qbench_manifest.json").read_text())["kernel_tables"]
    assert [Path(r).parents[3].name for r in runs["graph/gemm/H100.csv"]["runs"]] == ["a", "b"]


@pytest.mark.parametrize("bad", ["{", "[]", '{"kernel_tables": []}', "{}"])
def test_a_malformed_kernel_manifest_stops_before_a_table_moves(tmp_path: Path, bad) -> None:
    results, dest = tmp_path / "results", tmp_path / "data"
    manifest = seed(dest, "qbench_manifest.json", bad)
    probe_run(results, "gemm", {GEMM: GEMM_ROW}, 0)
    with pytest.raises(SystemExit, match="qbench_manifest.json: malformed manifest"):
        export([results], kernel_dest=dest, only=["kernel_tables"])
    assert manifest.read_text() == bad and not (dest / "kernel_data" / "graph").exists()


def test_the_kernel_manifest_keeps_its_other_keys(tmp_path: Path) -> None:
    results, dest = tmp_path / "results", tmp_path / "data"
    manifest = seed(dest, "qbench_manifest.json", '{"kernel_tables": {}, "reviewed_by": "kw"}')
    probe_run(results, "gemm", {GEMM: GEMM_ROW}, 0)
    export([results], kernel_dest=dest, only=["kernel_tables"])
    assert json.loads(manifest.read_text())["reviewed_by"] == "kw"


def test_a_kept_dest_sidecar_joins_notes_and_keeps_known_fields(tmp_path: Path) -> None:
    """qbench's note is appended to dest's; a field qbench leaves null keeps
    the value dest's sidecar had."""
    results, dest = tmp_path / "results", tmp_path / "data"
    rel = "graph/gemm/H100.csv"
    table = seed(dest, rel, csv_text([GEMM_HEAD, (1, 1, 1, 2, 3.0)]),
                 {"notes": "locked clocks", "reduce": "min", "gpu": "H100"})
    probe_run(results, "gemm", {GEMM: csv_text([GEMM_HEAD, (2, 2, 2, 2, 4.0)])}, 0, args=["--ncu", "--append"])
    probe_run(results, "gemm", {GEMM: csv_text([GEMM_HEAD, (3, 3, 3, 2, 4.0)])}, 1,
              args=["--graph", "--append"])
    export([results], kernel_dest=dest, only=["kernel_tables"])
    meta = json.loads(table.with_name(table.name + ".meta.json").read_text())
    assert meta["notes"] == "locked clocks; rows timed as graph_replay, ncu"
    tp2 = seed(dest, "graph/collectives/H100_tp2.csv", COLLECTIVE, {"reduce": "median"})
    probe_run(results, "vllm_allreduce", {"ncu/collectives/H100_tp2.csv": VLLM}, 2)
    export([results], kernel_dest=dest, only=["kernel_tables"])
    assert json.loads(tp2.with_name(tp2.name + ".meta.json").read_text())["reduce"] == "median"
