"""export() routes results dirs to QuettaSim's files and records where they came from."""
from __future__ import annotations

import contextlib
import csv
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

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

    def main():
        leg.parent.mkdir(parents=True)
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
