"""CLI validation in runner's __main__ block, plus path/bounds helpers."""

import subprocess
import sys
import unittest
from pathlib import Path

from src.benchmark.runner import per_turn_output_path
from src.workloads.dataset import _uniform_len_bounds

ROOT = Path(__file__).resolve().parent.parent

BASE = [
    sys.executable, "-m", "src.benchmark.runner",
    "--url", "http://127.0.0.1:1", "--model", "m", "--tensor-parallel-size", "1",
]


def run_cli(*extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        BASE + list(extra), cwd=ROOT, capture_output=True, text=True, timeout=30
    )


class TestFlagValidation(unittest.TestCase):
    def assert_rejected(self, args, needle):
        proc = run_cli(*args)
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn(needle, proc.stdout)

    def test_concurrency_zero_rejected(self):
        self.assert_rejected(["--concurrency", "0"], "--concurrency must be >= 1")

    def test_concurrency_negative_rejected(self):
        self.assert_rejected(["--concurrency", "-2"], "--concurrency must be >= 1")

    def test_num_requests_zero_rejected(self):
        self.assert_rejected(["--num-requests", "0"], "--num-requests must be >= 1")

    def test_warmup_negative_rejected(self):
        self.assert_rejected(["--warmup", "-1"], "--warmup must be >= 0")

    def test_max_turn_index_negative_rejected(self):
        self.assert_rejected(["--max-turn-index", "-1"], "--max-turn-index must be >= 0")

    def test_min_success_rate_out_of_range_rejected(self):
        self.assert_rejected(["--min-success-rate", "1.5"],
                             "--min-success-rate must be in [0, 1]")


class TestPerTurnOutputPath(unittest.TestCase):
    def test_json_suffix(self):
        self.assertEqual(per_turn_output_path("results/run.json"),
                         "results/run_per_turn.json")

    def test_no_suffix_never_clobbers(self):
        out = per_turn_output_path("results/run1")
        self.assertEqual(out, "results/run1_per_turn.json")
        self.assertNotEqual(out, "results/run1")

    def test_compound_suffix(self):
        self.assertEqual(per_turn_output_path("a.b.json"), "a.b_per_turn.json")


class TestUniformLenBounds(unittest.TestCase):
    def test_default_ratio_is_exact(self):
        self.assertEqual(_uniform_len_bounds(1024, 1.0), (1024, 1024))

    def test_partial_ratio(self):
        self.assertEqual(_uniform_len_bounds(1024, 0.5), (512, 1024))

    def test_tiny_input_floors_at_one(self):
        self.assertEqual(_uniform_len_bounds(1, 0.1), (1, 1))


if __name__ == "__main__":
    unittest.main()
