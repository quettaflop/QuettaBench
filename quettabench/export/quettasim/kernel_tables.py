"""Measured kernel tables -> dest/kernel_data/<method>/<kind>/<file>.

That is the tree QuettaSim's kernel_composed backend reads
(engine/backends/kernel_composed_cost.py kernel_artifact_paths, under the
data root $QUETTASIM_DATA, default the checkout's data/), so --dest is that
data root. A probe run writes <method>/<kind>/<file> (method ncu or
cuda_event) under its run dir: results/probes/<gpu>/<probe>/<run-id>/, or
results/<run>/<probe>/ for a `qbench run` probe. Per (probe, method, kind,
file) the newest ok run that wrote it (run.json provenance.started) is
taken; _inputs.files leaves out failed and unfinished runs, and an earlier
results dir wins over a later one. Two probes write one dest file in two
cases, and they merge:
- collective and vllm_allreduce, ncu/collectives/<gpu>_tp<N>.csv: the
  collective probe's rows other than all_reduce (path nccl), then
  vllm_allreduce's all_reduce rows, as vllm_allreduce's upsert did;
- gemm and elementwise, <method>/elementwise/<gpu>.json: the entries of
  both, the newer run's entry winning (gemm adds fp8_act_quant).
Any other file two probes write stops the export.
"""
from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Iterable
from pathlib import Path

from quettabench.export.quettasim import _inputs

METHODS = ("ncu", "cuda_event")
COLLECTIVE_FIELDS = ("op", "world", "bytes", "latency_us", "dtype_bytes", "path")


def _record(run_dir: Path) -> tuple[str | None, str]:
    """(probe, started) from run_dir/run.json; probe None for a run that is no probe run."""
    doc = json.loads((run_dir / "run.json").read_text())
    mode = (doc.get("config") or {}).get("mode") or {}
    probe = mode.get("probe") if mode.get("mode") == "probe" else None
    return probe, (doc.get("provenance") or {}).get("started") or ""


Source = tuple[str, Path, Path]  # (started, table, its run dir)


def pick(results_dirs: Iterable[Path]) -> dict[Path, dict[str, Source]]:
    """{<method>/<kind>/<file>: {probe: (started, table, run dir)}}, one
    source per probe: its newest ok run's table."""
    found: dict[tuple[str, Path], Source] = {}
    for root in map(Path, results_dirs):
        runs = _inputs.run_dirs(root)
        here: dict[tuple[str, Path], Source] = {}
        for method in METHODS:
            for path in _inputs.files(root, f"{method}/*/*"):
                run = _inputs.owner(path, root, runs)
                if not path.is_file() or run is None or not (run / "run.json").is_file():
                    continue
                rel = path.relative_to(run).parts
                probe, started = _record(run)
                if probe is None or rel[-3] != method or rel[:-3] not in ((), (probe,)):
                    continue
                key = (probe, Path(*rel[-3:]))
                if key not in found and (key not in here or started > here[key][0]):
                    here[key] = (started, path, run)
        found.update(here)
    tables: dict[Path, dict[str, Source]] = {}
    for (probe, rel), source in sorted(found.items()):
        tables.setdefault(rel, {})[probe] = source
    return tables


def sources(results_dirs: Iterable[Path]) -> dict[str, list[str]]:
    """{<method>/<kind>/<file>: the run dirs its table comes from}, for the manifest."""
    return {rel.as_posix(): [str(run) for _, _, run in sorted(by_probe.values())]
            for rel, by_probe in pick(results_dirs).items()}


def _merge_collectives(collective: Path, vllm: Path, out: Path) -> None:
    with collective.open(newline="") as f:
        kept = [r for r in csv.DictReader(f) if (r.get("op") or "all_reduce") != "all_reduce"]
    with vllm.open(newline="") as f:
        ours = [r for r in csv.DictReader(f) if (r.get("op") or "all_reduce") == "all_reduce"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLLECTIVE_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in kept:
            w.writerow({**r, "dtype_bytes": r.get("dtype_bytes") or 2, "path": r.get("path") or "nccl"})
        w.writerows(ours)


def _merge_json(tables: list[Path], out: Path) -> None:
    merged: dict = {}
    for table in tables:
        merged.update(json.loads(table.read_text()))
    out.write_text(json.dumps(merged, indent=2) + "\n")


def run(results_dirs: Iterable[Path], dest: Path) -> list[Path]:
    written = []
    for rel, by_probe in pick(results_dirs).items():
        out = Path(dest) / "kernel_data" / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        probes = sorted(by_probe)
        if len(probes) == 1:
            shutil.copyfile(by_probe[probes[0]][1], out)
        elif probes == ["collective", "vllm_allreduce"] and rel.suffix == ".csv":
            _merge_collectives(by_probe["collective"][1], by_probe["vllm_allreduce"][1], out)
        elif probes == ["elementwise", "gemm"] and rel.suffix == ".json":
            _merge_json([src for _, src, _ in sorted(by_probe.values())], out)
        else:
            raise SystemExit(f"{rel.as_posix()}: written by probes {probes}; kernel_tables "
                             "merges only collective+vllm_allreduce and elementwise+gemm")
        print(f"{rel.as_posix()} <- {', '.join(probes)}")
        written.append(out)
    return written
