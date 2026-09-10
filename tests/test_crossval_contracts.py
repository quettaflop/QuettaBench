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

The scripts are parsed, never imported: vmin_fit.py imports vllm, a GPU-only
dependency absent from CI.
"""

import ast
import json
import os
import subprocess
import unittest
from pathlib import Path

CROSSVAL = Path(__file__).resolve().parents[1] / "scripts" / "crossval"
VERIFIED_CRATE = "llama"


def _scripts(suffix):
    return sorted(CROSSVAL.glob(f"*{suffix}"))


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


if __name__ == "__main__":
    unittest.main()
