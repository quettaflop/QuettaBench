"""qbench's doors hand argv to their module's main, and qbench run refuses bad values before anything starts."""

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from quettabench import cli
from quettabench.workloads.base import uniform_len_bounds
from tests.golden._harness import REPO


class Doors(unittest.TestCase):
    def test_each_door_gets_the_rest_of_argv(self):
        for door, module in cli.DOORS.items():
            with self.subTest(door=door), mock.patch("importlib.import_module") as imp:
                imp.return_value.main.return_value = 0
                self.assertEqual(cli.main([door, "--x", "y"]), 0)
                imp.assert_called_once_with(module)
                imp.return_value.main.assert_called_once_with(["--x", "y"])

    def test_runs_without_installing(self):
        proc = subprocess.run(
            [sys.executable, "-m", "quettabench.cli", "export", "--help"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(proc.stdout.startswith("usage: qbench export"))

    def test_no_door_prints_the_doors(self):
        proc = subprocess.run(
            [sys.executable, "-m", "quettabench.cli"], cwd=REPO, capture_output=True, text=True, check=False
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("{run,profile,xval,export}", proc.stdout)


ROOT = Path(__file__).resolve().parents[2]

BASE = [
    sys.executable,
    "-m",
    "quettabench.cli",
    "run",
    "configs/runs/llama.yaml",
    "--dry-run",
    "--set",
    "engine.model=m",
    "--set",
    "mode.grid=null",
]


def run_cli(*sets: str) -> subprocess.CompletedProcess:
    argv = [a for s in sets for a in ("--set", s)]
    return subprocess.run(BASE + argv, cwd=ROOT, capture_output=True, text=True, timeout=30)


class TestFlagValidation(unittest.TestCase):
    def assert_rejected(self, value, needle):
        proc = run_cli(value)
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn(needle, proc.stderr)

    def test_accepted(self):
        self.assertEqual(run_cli().returncode, 0)

    def test_concurrency_zero_rejected(self):
        self.assert_rejected("mode.concurrency=0", "mode.concurrency must be in [1, inf], got 0")

    def test_concurrency_negative_rejected(self):
        self.assert_rejected("mode.concurrency=-2", "mode.concurrency must be in [1, inf]")

    def test_num_requests_zero_rejected(self):
        self.assert_rejected("workload.num_requests=0", "workload.num_requests must be in [1, inf]")

    def test_warmup_negative_rejected(self):
        self.assert_rejected("mode.warmup=-1", "mode.warmup must be in [0, inf]")

    def test_turn_waits_negative_rejected(self):
        self.assert_rejected("workload.tool_wait_ms=-1", "workload.tool_wait_ms must be in [0.0, inf]")

    def test_min_success_rate_out_of_range_rejected(self):
        self.assert_rejected("mode.min_success_rate=1.5", "mode.min_success_rate must be in [0.0, 1.0]")

    def test_bad_serving_style_is_one_line_not_a_traceback(self):
        self.assert_rejected("engine.serving_style=disagg:2p", "unknown serving_style 'disagg:2p'")
        self.assertNotIn("Traceback", run_cli("engine.serving_style=disagg:2p").stderr)

    def test_afd_on_vllm_is_one_line_exit_1(self):
        self.assert_rejected(
            "engine.serving_style=afd:1a1f", "vllm: serving_style=afd:1a1f has no launch flag in use today"
        )


class TestUniformLenBounds(unittest.TestCase):
    def test_default_ratio_is_exact(self):
        self.assertEqual(uniform_len_bounds(1024, 1.0), (1024, 1024))

    def test_partial_ratio(self):
        self.assertEqual(uniform_len_bounds(1024, 0.5), (512, 1024))

    def test_tiny_input_floors_at_one(self):
        self.assertEqual(uniform_len_bounds(1, 0.1), (1, 1))


if __name__ == "__main__":
    unittest.main()
