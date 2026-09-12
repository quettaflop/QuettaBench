"""Contract tests for the crossval harness moved from QuettaServe.

The crossval scripts run on a GPU box (they drive cargo benches and a vLLM
baseline), so CI cannot execute them. What CI can pin is the wiring that
otherwise only fails once someone has booked GPU time:

  1. A script stops parsing -- a bad edit lands and the whole `just crossval`
     recipe dies at its first invocation on the GPU host, not here.
  2. workloads.json loses a grid that a workload names -- vmin_fit.py reads
     CFG["workloads"][crate]["grid"] and indexes CFG["grids"] with it, so a
     dangling reference is a KeyError mid-run with the GPU already allocated.
  3. A shell helper loses its executable bit or gains a syntax error --
     with_gpu.sh and vllm.sh are exec'd directly, so either one silently
     breaks the baseline refresh.
  4. A grid cell that cannot run at its workload's maxlen -- vmin_fit prints
     a SKIP line and moves on, so the config looks covered while the cell is
     never measured.
  5. A tensor-parallel workload whose name and tp field disagree -- the
     baseline would be captured at the wrong degree and paired with the
     wrong bench rows.

The scripts are parsed, never imported: vmin_fit.py imports vllm, a GPU-only
dependency absent from CI. The glob is recursive so the ported slice_tf
harness is parse-covered too.
"""

import ast
import json
import os
import re
import subprocess
import unittest
from pathlib import Path

CROSSVAL = Path(__file__).resolve().parents[1] / "scripts" / "crossval"
VERIFIED_CRATE = "llama"


def _scripts(suffix):
    return sorted(p for p in CROSSVAL.rglob(f"*{suffix}") if "__pycache__" not in p.parts)


def _load_workloads():
    return json.loads((CROSSVAL / "workloads.json").read_text())


class CrossvalScripts(unittest.TestCase):
    def test_python_scripts_parse(self):
        scripts = _scripts(".py")
        self.assertTrue(scripts, "no python scripts under scripts/crossval")
        for script in scripts:
            ast.parse(script.read_text(), filename=str(script))

    def test_shell_scripts_have_valid_syntax(self):
        scripts = _scripts(".sh")
        self.assertTrue(scripts, "no shell scripts under scripts/crossval")
        for script in scripts:
            subprocess.run(["bash", "-n", str(script)], check=True)

    def test_shell_scripts_are_executable(self):
        for script in _scripts(".sh"):
            self.assertTrue(
                os.access(script, os.X_OK),
                f"{script.name} is exec'd directly and must keep its +x bit",
            )


class WorkloadsConfig(unittest.TestCase):
    def test_top_level_shape(self):
        cfg = _load_workloads()
        self.assertIsInstance(cfg.get("step_points"), list)
        self.assertTrue(cfg["step_points"])
        self.assertIsInstance(cfg.get("grids"), dict)
        self.assertIsInstance(cfg.get("workloads"), dict)

    def test_every_workload_names_an_existing_grid(self):
        cfg = _load_workloads()
        grids = cfg["grids"]
        for name, workload in cfg["workloads"].items():
            grid = workload.get("grid")
            self.assertIn(
                grid, grids,
                f"workload {name} names grid {grid!r}, absent from grids",
            )

    def test_verified_crate_grid_is_context_batch_pairs(self):
        cfg = _load_workloads()
        grid_name = cfg["workloads"][VERIFIED_CRATE]["grid"]
        grid = cfg["grids"][grid_name]
        self.assertTrue(grid, f"{VERIFIED_CRATE} grid {grid_name} is empty")
        for cell in grid:
            self.assertEqual(len(cell), 2, f"grid cell {cell} is not (context, batch)")
            self.assertTrue(all(isinstance(x, int) for x in cell))

    def test_tp_suffixed_workloads_declare_matching_tp(self):
        cfg = _load_workloads()
        for name, workload in cfg["workloads"].items():
            m = re.search(r"-tp(\d+)$", name)
            if m:
                self.assertEqual(
                    workload.get("tp"), int(m.group(1)),
                    f"workload {name} must declare tp={m.group(1)}",
                )

    def test_grid_cells_fit_workload_maxlen(self):
        cfg = _load_workloads()
        top = max(cfg["step_points"])
        for name, workload in cfg["workloads"].items():
            for ctx, bs in cfg["grids"][workload["grid"]]:
                self.assertLessEqual(
                    ctx + top + 2, workload["maxlen"],
                    f"workload {name} cell ({ctx}, {bs}) exceeds maxlen "
                    f"{workload['maxlen']}; vmin_fit would SKIP it",
                )


class EngineBenchContract(unittest.TestCase):
    """Cells frozen from QuettaServe llama/benches/latency.rs, identical on
    the sm80, kev/xval and supp_tp branches: the batch groups sweep ctx
    {100, 1024, 4096} x bs {1, 8, 16, 32, 64}; the tensor-parallel groups
    sweep ctx {100, 1024, 4096, 8192} at bs 1. Every cell the engine bench
    times needs a vLLM cell, or the comparison table silently drops the row."""

    def test_llama_grid_covers_the_batch_bench_cells(self):
        grid = {tuple(c) for c in _load_workloads()["grids"]["llama"]}
        expected = {(ctx, bs) for ctx in (100, 1024, 4096) for bs in (1, 8, 16, 32, 64)}
        self.assertTrue(expected <= grid, f"missing cells: {sorted(expected - grid)}")

    def test_llama_single_grid_covers_the_tp_bench_cells(self):
        grid = {tuple(c) for c in _load_workloads()["grids"]["llama_single"]}
        expected = {(ctx, 1) for ctx in (100, 1024, 4096, 8192)}
        self.assertTrue(expected <= grid, f"missing cells: {sorted(expected - grid)}")


if __name__ == "__main__":
    unittest.main()
