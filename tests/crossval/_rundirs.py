"""Run dirs built from the golden data, the way serve writes them.

grid_run_dir is a real grid run on the stubs (qbench run --results-dir),
git rev-parse frozen to the golden baseline's bench_sha, plus the two things
no run produces: the pre-normalization RAW cell (no ms_at_eval)
the golden baseline carries as its last point, and the finish time set to
the baseline's recorded_utc. bench_run_dir is a QuettaServe run: the bench
log where BenchDriver keeps it, its rows appended through RunWriter, and
run.json with the step section fit_run derives from those rows.
"""

import json
import shutil
import types
from pathlib import Path
from unittest import mock

from quettabench.config.schema import (
    Config,
    EngineCfg,
    HardwareCfg,
    ModeCfg,
    WorkloadCfg,
)
from quettabench.engines.drivers import BenchDriver
from quettabench.metrics.results import RunWriter
from quettabench.metrics.step_fit import fit_run
from tests.golden._harness import grid_argv, run_grid_cli

RECORDED = "2026-09-21T14:13:20+00:00"
FAKE_BENCH_SHA = "0123abc"
RAW_STEP = {"ctx": 4096, "bs": 64, "tp": 1, "ms_per_step": 19.874, "ms_at_eval": None,
            "eval_pos": 0, "r2": 0.9995, "prefill_s": 1.1, "raw": [2.29, 5.67, 9.05]}


def _edit_run_json(path, edit):
    run = json.loads(path.read_text())
    edit(run)
    path.write_text(json.dumps(run, indent=2, default=str) + "\n")


def grid_run_dir(out):
    """results/<run>/ of the llama grid on the stubs, as the golden baseline's run."""
    out = Path(out)
    fake_git = types.SimpleNamespace(stdout=FAKE_BENCH_SHA + "\n")
    with mock.patch("subprocess.run", return_value=fake_git):
        run = run_grid_cli(grid_argv("llama", "--set", f"run_name={out.name}",
                                     "--results-dir", str(out.parent)), {})
    if run["exit"] is not None:
        raise RuntimeError(f"grid run exited: {run['exit']}")

    def edit(run_json):
        run_json["steps"].append(dict(RAW_STEP))
        run_json["provenance"]["finished"] = RECORDED
    _edit_run_json(out / "run.json", edit)
    return out


def bench_run_dir(out, bench_log):
    """results/<run>/ of a QuettaServe bench run whose log is bench_log."""
    out = Path(out)
    out.mkdir(parents=True)
    shutil.copy(bench_log, out / "bench.log")
    rows = list(BenchDriver(out / "bench.log").rows())
    writer = RunWriter(out, bench_config(out.name))
    for r in rows:
        writer.append(r)
    writer.finish(None, steps=fit_run(rows, 0, 1, 1.0))
    return out


def bench_config(run_name):
    return Config(engine=EngineCfg(name="quettaserve"), workload=WorkloadCfg(), mode=ModeCfg(),
                  hardware=HardwareCfg(), run_name=run_name)
