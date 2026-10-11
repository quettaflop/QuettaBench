"""export() routes results dirs to QuettaSim's files and records where they came from."""
from __future__ import annotations

import contextlib
import csv
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

from quettabench.export import quettasim
from quettabench.export.quettasim import _inputs, export
from quettabench.probes.base import run_recorded


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
    _inputs.run_dirs.cache_clear()
    assert _inputs.skipped_runs(tmp_path / "results") == {leg.parent: "no run.json"}
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
    _inputs.run_dirs.cache_clear()
    kept = [p.relative_to(tmp_path).as_posix() for p in _inputs.files(tmp_path, "*.csv")]
    assert kept == ["H100/gemm/ncu/gemm/H100.csv"]


def test_find_dir_skips_a_dir_that_is_itself_a_failed_run(tmp_path: Path) -> None:
    """A bench dir that is a run dir is judged by its own run.json, not only
    by its ancestors'."""
    for root, status in (("a", "failed"), ("b", "ok")):
        run = tmp_path / root / "bench"
        run.mkdir(parents=True)
        (run / "run.json").write_text(json.dumps({"run_status": {"status": status}}))
    (tmp_path / "c" / "bench").mkdir(parents=True)
    (tmp_path / "c" / "bench" / "requests.jsonl").write_text("")
    _inputs.run_dirs.cache_clear()
    roots = [tmp_path / "a", tmp_path / "c", tmp_path / "b"]
    assert _inputs.find_dir(roots, "bench") == tmp_path / "b" / "bench"
    assert _inputs.find_dir(roots[:2], "bench") is None


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
    assert _inputs.skipped_runs(tmp_path / "results") == {}
    assert _inputs.owner(run / "server" / "engine.json", tmp_path / "results",
                         _inputs.run_dirs(tmp_path / "results")) == run


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
        seen.append(_inputs.skipped_runs(results))
        (run / "run.json").write_text(json.dumps({"run_status": {"status": "failed"}}))
        return []

    def second(dirs, dest):
        seen.append(_inputs.skipped_runs(results))
        return []

    with mock.patch.dict(quettasim.BUILDERS, {"first": mock.Mock(run=first),
                                              "second": mock.Mock(run=second)}):
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
    _inputs.run_dirs.cache_clear()
    assert _inputs.find([results], "serving_stage_split_H100.csv") == new / "serving_stage_split_H100.csv"
    assert _inputs.stage_split_legs([results], "H100") == [
        (2, new / "serving_stage_split_H100.csv", new / "serving_stage_split_H100_tp2.csv")]
    # A run that started first but finished last wins over one that started later.
    (old / "run.json").write_text(json.dumps({
        "config": {}, "run_status": {"status": "ok", "reason": None},
        "provenance": {"started": "20261001", "finished": "2026-10-09T00:00:00+00:00"}}))
    _inputs.run_dirs.cache_clear()
    assert _inputs.find([results], "serving_stage_split_H100_tp2.csv") == old / "serving_stage_split_H100_tp2.csv"
