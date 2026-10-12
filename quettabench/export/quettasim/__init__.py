"""The one route out of QuettaBench: results dirs in, QuettaSim's files out."""
from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from quettabench.export.quettasim import (
    _inputs,
    extract_benchmark_per_request,
    kernel_tables,
    prefill_floor,
    roofline_utils,
    saturated_ceiling,
    serving_decode_grid,
    stage_split_rates,
    tp_comm,
)

BUILDERS = {
    "tp_comm": tp_comm,
    "stage_split_rates": stage_split_rates,
    "roofline_utils": roofline_utils,
    "serving_decode_grid": serving_decode_grid,
    "extract_benchmark_per_request": extract_benchmark_per_request,
    "prefill_floor": prefill_floor,
    "saturated_ceiling": saturated_ceiling,
    "kernel_tables": kernel_tables,
}
REPO_ROOT = Path(__file__).resolve().parents[3]


def bench_sha() -> str:
    """HEAD of this repo, empty when git cannot say."""
    try:
        done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
                              capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return ""
    return done.stdout.strip()


def source_runs(results_dirs: list[Path]) -> list[dict]:
    """Every exported run.json under the results dirs: its run dir (the run
    id), run name, status and start, so an exported table names the runs
    behind it. A run dir _inputs.skipped_runs leaves out is printed instead."""
    runs = []
    for root in results_dirs:
        skipped = _inputs.skipped_runs(root)
        for run_dir, why in skipped.items():
            print(f"skip {run_dir}: {why}")
        for path in _inputs.files(root, "run.json"):
            doc = json.loads(path.read_text())
            runs.append({"run": str(path.parent), "run_name": doc["config"].get("run_name"),
                         "status": doc["run_status"]["status"],
                         "started": doc["provenance"].get("started")})
    return runs


def export(results_dirs: Iterable[Path], dest: Path | None = None, only: Iterable[str] | None = None,
           kernel_dest: Path | None = None) -> list[Path]:
    """Run the builders (all, or the named ones) over the results dirs; return the files written.

    kernel_tables writes under kernel_dest (QuettaSim's data root) and runs
    only when it is given; the other builders write under dest and run only
    when it is given. A builder named in only without its dest stops the
    export. dest gets manifest.json, kernel_dest kernel_data/qbench_manifest.json
    when kernel_tables found a table (not at QuettaSim's data root): where the inputs came from, the runs
    recorded there (run.json), which bench commit, when, and for kernel_dest
    the run dirs whose rows each kernel table holds.
    """
    results_dirs = [Path(d) for d in results_dirs]
    names = list(only) if only else list(BUILDERS)
    unknown = [n for n in names if n not in BUILDERS]
    if unknown:
        raise SystemExit(f"unknown builder {unknown}; choose from {sorted(BUILDERS)}")
    if only:
        if "kernel_tables" in names and kernel_dest is None:
            raise SystemExit("kernel_tables writes under --kernel-dest (QuettaSim's data root); give it")
        if any(n != "kernel_tables" for n in names) and dest is None:
            raise SystemExit("the builders other than kernel_tables write under --dest; give it")
    names = [n for n in names if (kernel_dest if n == "kernel_tables" else dest) is not None]
    if not names:
        raise SystemExit("give --dest, --kernel-dest or both")
    written: list[Path] = []
    tables = None
    for name in names:
        _inputs.run_dirs.cache_clear()
        if name == "kernel_tables":
            paths, tables = kernel_tables.run(results_dirs, Path(kernel_dest))
            written.extend(paths)
        else:
            written.extend(BUILDERS[name].run(results_dirs, Path(dest)))
    _inputs.run_dirs.cache_clear()
    manifest = {
        "source_dirs": [str(d) for d in results_dirs],
        "runs": source_runs(results_dirs),
        "bench_sha": bench_sha(),
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if any(n != "kernel_tables" for n in names):
        Path(dest).mkdir(parents=True, exist_ok=True)
        (Path(dest) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if tables:
        (Path(kernel_dest) / "kernel_data").mkdir(parents=True, exist_ok=True)
        (Path(kernel_dest) / "kernel_data" / "qbench_manifest.json").write_text(
            json.dumps({**manifest, "kernel_tables": tables}, indent=2) + "\n")
    return written
