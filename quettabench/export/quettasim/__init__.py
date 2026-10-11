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


def export(results_dirs: Iterable[Path], dest: Path, only: Iterable[str] | None = None) -> list[Path]:
    """Run the builders (all, or the named ones) over the results dirs; return the files written.

    Also writes dest/manifest.json: where the inputs came from, the runs
    recorded there (run.json), which bench commit, when.
    """
    results_dirs = [Path(d) for d in results_dirs]
    dest = Path(dest)
    names = list(only) if only else list(BUILDERS)
    unknown = [n for n in names if n not in BUILDERS]
    if unknown:
        raise SystemExit(f"unknown builder {unknown}; choose from {sorted(BUILDERS)}")
    written: list[Path] = []
    for name in names:
        written.extend(BUILDERS[name].run(results_dirs, dest))
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_dirs": [str(d) for d in results_dirs],
        "runs": source_runs(results_dirs),
        "bench_sha": bench_sha(),
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return written
