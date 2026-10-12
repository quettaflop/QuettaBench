"""kernel_tables: probe runs' tables -> dest/kernel_data/{graph,eager}/<kind>/<file>
with a <file>.meta.json beside each, the tree QuettaSim's kernel_composed
backend reads (kernel_composed_cost.py kernel_artifact_paths, under its data
root's kernel_data/)."""
from __future__ import annotations

import argparse
import ast
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
    assert [Path(r).parent.name for r in runs["graph/collectives/H100_tp2.csv"]] == [
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
    export([results], kernel_dest=dest, only=["kernel_tables"])
    assert rows(dest / "kernel_data" / "graph" / "flash_attn" / "H100.csv") == [
        list(FA_HEAD), ["64", "8", "128", "1", "20.0"],
        ["32", "8", "128", "1", "10.0"], ["32", "8", "128", "2", "11.0"],
        ["16", "2", "128", "1", "7.0"]]
    runs = json.loads((dest / "kernel_data" / "qbench_manifest.json").read_text())["kernel_tables"]
    assert [Path(r).name for r in runs["graph/flash_attn/H100.csv"]] == [
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
    written, used = kernel_tables.run([results], dest)
    assert sorted(used) == ["graph/collectives/H100_tp2.csv", "graph/flash_attn/H100.csv"]
    assert [Path(r).parent.name for r in used["graph/collectives/H100_tp2.csv"]] == ["vllm_allreduce"]
    assert [Path(r).name for r in used["graph/flash_attn/H100.csv"]] == ["20261011T120001.000000Z"]
    export([results], kernel_dest=dest, only=["kernel_tables"])
    assert json.loads((dest / "kernel_data" / "qbench_manifest.json").read_text())["kernel_tables"] == used


def test_kernel_tables_go_to_their_own_dest_and_only_when_given(tmp_path: Path, capsys) -> None:
    """--dest takes the other builders' files; the kernel tables go to the
    QuettaSim data root given as --kernel-dest, and without it they are not built."""
    from quettabench.export.quettasim.__main__ import main
    results, dest, data = tmp_path / "results", tmp_path / "out", tmp_path / "QuettaSim" / "data"
    probe_run(results, "gdn", {"cuda_event/gdn/H100_9b_decode.csv": "g\n"}, 0)
    from quettabench.export import quettasim
    with mock.patch.dict(quettasim.BUILDERS, {"tp_comm": quettasim.tp_comm,
                                              "kernel_tables": kernel_tables}, clear=True):
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
        export([results], kernel_dest=dest, only=["kernel_tables"])
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
    source = (Path(kernel_tables.__file__).parents[2] / "profiling" / "kv_transfer.py").read_text()
    (key,) = [ast.literal_eval(n.value) for n in ast.parse(source).body
              if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "KEY"]
    assert kernel_tables.ROW_KEYS[("kv_transfer", "kv_transfer")] == key
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
        kernel_tables.run([results], tmp_path / "out")


def test_unparsable_probe_args_stop_the_export_naming_the_run(tmp_path: Path) -> None:
    results = tmp_path / "results"
    probe_run(results, "gemm", {GEMM: GEMM_ROW}, 0)
    record = results / "probes" / "H100" / "gemm" / "20261011T120000.000000Z" / "run.json"
    doc = json.loads(record.read_text())
    doc["config"]["mode"]["probe_args"] = 'a "b'
    record.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match=r"20261011T120000.000000Z.*probe_args"):
        kernel_tables.run([results], tmp_path / "out")
