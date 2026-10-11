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
from quettabench.probes.base import add_output_args, out_root, run_recorded

T0 = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)


def probe_run(results: Path, probe: str, tables: dict[str, str], at: int, fails=False,
              gpu: str = "H100") -> None:
    """One `python -m quettabench.probes.<probe>` run through run_recorded:
    writes tables {path under the run dir: text} where the probe would."""
    def main(argv):
        ap = argparse.ArgumentParser()
        add_output_args(ap)
        root = out_root(ap.parse_known_args(argv)[0], probe)
        for rel, text in tables.items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(text)
        if fails:
            raise RuntimeError("probe died")

    argv = ["probe", "--gpu-label", gpu, "--out", str(results / "probes" / gpu)]
    with mock.patch.object(sys, "argv", argv), \
            pytest.raises(RuntimeError) if fails else contextlib.nullcontext():
        run_recorded(main, probe, now=T0 + timedelta(seconds=at))


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
