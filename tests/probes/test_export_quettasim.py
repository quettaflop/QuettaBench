"""export() routes results dirs to QuettaSim's files and records where they came from."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from quettabench.export.quettasim import export


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
    assert set(manifest) == {"source_dirs", "bench_sha", "exported_at"}
    assert manifest["exported_at"].endswith("+00:00")
