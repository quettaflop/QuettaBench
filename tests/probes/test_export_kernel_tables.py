"""kernel_tables: probe runs' tables -> dest/kernel_data/<method>/<kind>/<file>,
the tree QuettaSim's kernel_composed backend reads (kernel_composed_cost.py
kernel_artifact_paths, under its data root's kernel_data/)."""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest

from quettabench.export.quettasim import export, kernel_tables
from quettabench.probes.base import add_output_args, out_root, run_id, run_recorded

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
    return {p.relative_to(root).as_posix(): p.read_text()
            for p in sorted(root.rglob("*")) if p.is_file()}


def csv_text(rows):
    return "".join(",".join(map(str, r)) + "\r\n" for r in rows)


COLLECTIVE = csv_text([("op", "world", "bytes", "latency_us", "dtype_bytes"),
                       ("all_gather", 2, 4096, 15.2, 2), ("all_reduce", 2, 4096, 20.3, 2),
                       ("reduce_scatter", 2, 4096, 14.1, 2)])
VLLM = csv_text([("op", "world", "bytes", "latency_us", "dtype_bytes", "path"),
                 ("all_reduce", 2, 4096, 9.1, 2, "vllm"), ("all_reduce", 2, 8192, 9.4, 2, "vllm")])


def test_probe_runs_export_to_the_quettasim_layout(tmp_path: Path) -> None:
    """Every table lands where kernel_artifact_paths looks for it; tables
    outside ncu/ and cuda_event/ stay home."""
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
    written = export([results], dest, only=["kernel_tables"])
    got = tree(dest / "kernel_data")
    assert sorted(got) == [
        "cuda_event/fa3_cross/H100.csv",          # cross_attn_path
        "cuda_event/fa3_prefill/H100.csv",        # prefill_attn_path
        "cuda_event/gdn/H100_9b_decode.csv",      # gdn_data_dir
        "cuda_event/reparallel/H100_reshard.csv",
        "ncu/collectives/H100_tp2.csv",           # allreduce_path
        "ncu/collectives/H100_tp2_nccl_vs_vllm.csv",
        "ncu/elementwise/H100.json",              # elementwise_path
        "ncu/flash_attn/H100.csv",                # flash_attn_path
        "ncu/gemm/H100.csv",                      # gemm_path, router_gemm_path
        "ncu/gemm_wide/H100.csv",
        "ncu/moe/H100.csv",                       # moe_path
        "ncu/moe/H100_routing.json",              # moe_routing_path
    ]
    assert sorted(written) == sorted(dest / "kernel_data" / rel for rel in got)
    assert got["ncu/gemm/H100.csv"] == "M,N,K\n"
    assert json.loads(got["ncu/elementwise/H100.json"]) == {"fp8_act_quant": {"floor_us": 2},
                                                           "rms_norm": {"floor_us": 1}}
    manifest = json.loads((dest / "manifest.json").read_text())
    runs = manifest["kernel_tables"]
    assert set(runs) == set(got)
    assert [Path(r).parent.name for r in runs["ncu/collectives/H100_tp2.csv"]] == [
        "collective", "vllm_allreduce"]


@pytest.mark.parametrize("vllm_first", [False, True])
def test_vllm_all_reduce_rows_merge_over_the_collective_ones(tmp_path: Path, vllm_first) -> None:
    """vllm_allreduce's all_reduce rows replace the collective probe's, and
    the other ops keep their NCCL rows with path nccl, whichever ran last."""
    results, dest = tmp_path / "results", tmp_path / "out"
    probe_run(results, "collective", {"ncu/collectives/H100_tp2.csv": COLLECTIVE}, 1 - vllm_first)
    probe_run(results, "vllm_allreduce", {"ncu/collectives/H100_tp2.csv": VLLM}, vllm_first)
    export([results], dest, only=["kernel_tables"])
    with (dest / "kernel_data" / "ncu" / "collectives" / "H100_tp2.csv").open(newline="") as f:
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
    export([results], dest, only=["kernel_tables"])
    assert tree(dest / "kernel_data") == {"ncu/gemm/H100.csv": "good\n"}


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
    export([tmp_path / "results"], tmp_path / "out", only=["kernel_tables"])
    assert tree(tmp_path / "out" / "kernel_data") == {"ncu/gemm/H100.csv": "t\n"}


def test_two_probes_writing_one_file_stop_the_export(tmp_path: Path) -> None:
    results = tmp_path / "results"
    probe_run(results, "gemm", {"ncu/moe/H100.csv": "a\n"}, 0)
    probe_run(results, "moe", {"ncu/moe/H100.csv": "b\n"}, 1)
    with pytest.raises(SystemExit, match="ncu/moe/H100.csv.*gemm.*moe"):
        kernel_tables.run([results], tmp_path / "out")


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
    export([results], dest, only=["kernel_tables"])
    assert rows(dest / "kernel_data" / "ncu" / "flash_attn" / "H100.csv") == [
        list(FA_HEAD), ["64", "8", "128", "1", "20.0"],
        ["32", "8", "128", "1", "10.0"], ["32", "8", "128", "2", "11.0"],
        ["16", "2", "128", "1", "7.0"]]
    runs = json.loads((dest / "manifest.json").read_text())["kernel_tables"]
    assert [Path(r).name for r in runs["ncu/flash_attn/H100.csv"]] == [
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
    export([results], dest, only=["kernel_tables"])
    assert rows(dest / "kernel_data" / table) == [
        list(GEMM_HEAD), ["1", "4", "4", "2", "5.0"], ["8", "4", "4", "2", "7.0"]]


def test_reparallel_rows_upsert_on_their_key_and_sort(tmp_path: Path) -> None:
    head = ("gpu", "world", "tp_from", "tp_to", "total_weight_gb", "comm_ms")
    results, dest = tmp_path / "results", tmp_path / "out"
    table = "cuda_event/reparallel/H100_reshard.csv"
    probe_run(results, "reparallel", {table: csv_text([head, ("H100", 8, 4, 2, 16.0, 3.0),
                                                       ("H100", 8, 2, 1, 16.0, 2.0)])}, 0)
    probe_run(results, "reparallel", {table: csv_text([head, ("H100", 8, 2, 1, 16.0, 1.5)])}, 1)
    export([results], dest, only=["kernel_tables"])
    assert rows(dest / "kernel_data" / table) == [
        list(head), ["H100", "8", "2", "1", "16.0", "1.5"], ["H100", "8", "4", "2", "16.0", "3.0"]]


def test_a_whole_table_comes_from_the_run_that_finished_last(tmp_path: Path) -> None:
    """gdn writes its whole table each run (no row key): the latest finished
    run's file, byte for byte, even when it started first."""
    results, dest = tmp_path / "results", tmp_path / "out"
    table = "cuda_event/gdn/H100_9b_decode.csv"
    probe_run(results, "gdn", {table: "batch,us\n1,5\n"}, at=0, finished=100)
    probe_run(results, "gdn", {table: "batch,us\n2,6\n"}, at=1, finished=50)
    export([results], dest, only=["kernel_tables"])
    assert tree(dest / "kernel_data") == {table: "batch,us\n1,5\n"}


def test_elementwise_entries_union_with_gemm_fp8_quant(tmp_path: Path) -> None:
    """The elementwise probe's latest run gives its entries whole; every gemm
    run's fp8_act_quant goes over them, the later finished run winning a key."""
    results, dest = tmp_path / "results", tmp_path / "out"
    table = "ncu/elementwise/H100.json"
    probe_run(results, "elementwise", {table: '{"rms_norm": 1, "silu": 1}'}, 0)
    probe_run(results, "gemm", {table: '{"fp8_act_quant": 1}'}, 1)
    probe_run(results, "elementwise", {table: '{"rms_norm": 2}'}, 2)
    probe_run(results, "gemm", {table: '{"fp8_act_quant": 3}'}, 3)
    export([results], dest, only=["kernel_tables"])
    assert json.loads((dest / "kernel_data" / table).read_text()) == {
        "rms_norm": 2, "fp8_act_quant": 3}


def test_an_unmergeable_file_stops_the_export_before_anything_is_written(tmp_path: Path) -> None:
    results, dest = tmp_path / "results", tmp_path / "out"
    probe_run(results, "gdn", {"cuda_event/gdn/H100_9b_decode.csv": "g\n"}, 0)
    probe_run(results, "gemm", {"ncu/moe/H100.csv": "a\n"}, 1)
    probe_run(results, "moe", {"ncu/moe/H100.csv": "b\n"}, 2)
    with pytest.raises(SystemExit, match="ncu/moe/H100.csv"):
        export([results], dest, only=["kernel_tables"])
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
    written, used = kernel_tables.run([results], dest)
    assert sorted(used) == ["ncu/collectives/H100_tp2.csv", fa]
    assert [Path(r).parent.name for r in used["ncu/collectives/H100_tp2.csv"]] == ["vllm_allreduce"]
    assert [Path(r).name for r in used[fa]] == ["20261011T120001.000000Z"]
    export([results], dest, only=["kernel_tables"])
    assert json.loads((dest / "manifest.json").read_text())["kernel_tables"] == used
