"""BenchDriver's rows are table.py's parse of the same log, and its cell loop
fails where bench.sh fails."""

import sys
import tempfile
import unittest
from pathlib import Path

from quettabench.crossval import table
from quettabench.engines.base import LaunchSpec
from quettabench.engines.drivers import BenchDriver, BenchFailed
from tests.golden._harness import FIXTURES

FAKE = Path(__file__).with_name("fake_bench.py")


def cell_spec(ctx, bs, **env):
    return LaunchSpec(
        adapter="quettaserve", bench_argv=(sys.executable, str(FAKE)), cwd=".",
        env={"QW_BENCH_PROMPT": str(ctx), "QW_BATCH": str(bs), "QW_WORLD": "1", **env},
        ok_pattern=r"^LOOP ", log_header=f">>> ctx={ctx} bs={bs} world=1")


def as_points(rows):
    return {(r["ctx"], r["bs"], r["tp"]): (r["ms_per_step"], r["group"], r["kv"]) for r in rows}


class GoldenLogs(unittest.TestCase):
    def test_rows_equal_table_parsers(self):
        for name in ("bench_loop.log", "bench_criterion.log"):
            with self.subTest(log=name):
                path = FIXTURES / name
                rows = list(BenchDriver(path).rows())
                points, method = table.ours_points(path)
                self.assertTrue(points)
                self.assertEqual(as_points(rows), points)
                self.assertEqual({r["method"] for r in rows}, {method})


class CellLoop(unittest.TestCase):
    def setUp(self):
        self.log = Path(tempfile.mkdtemp()) / "bench.log"

    def test_two_cells_log_and_rows(self):
        rows = list(BenchDriver(self.log).run([(1024, 1, cell_spec(1024, 1)),
                                               (1024, 16, cell_spec(1024, 16))]))
        self.assertEqual(as_points(rows), {(1024, 1, 1): (3.5, "decode_loop", "fp16"),
                                           (1024, 16, 1): (3.5, "decode_loop", "fp16")})
        lines = self.log.read_text().splitlines()
        self.assertEqual(lines[0], ">>> ctx=1024 bs=1 world=1")
        self.assertIn("bench chatter", lines)

    def test_missing_result_line_fails_after_all_cells(self):
        with self.assertRaisesRegex(BenchFailed, r"1 cell\(s\) failed"):
            list(BenchDriver(self.log).run([(1, 1, cell_spec(1, 1, FAKE_BENCH_SILENT="1")),
                                            (2, 1, cell_spec(2, 1))]))
        text = self.log.read_text()
        self.assertIn("FAIL ctx=1 bs=1 (no bench result line", text)
        self.assertIn("LOOP ctx=2 bs=1", text)

    def test_nonzero_exit_stops_the_sweep(self):
        with self.assertRaisesRegex(BenchFailed, "exited 3"):
            list(BenchDriver(self.log).run([(1, 1, cell_spec(1, 1, FAKE_BENCH_EXIT="3")),
                                            (2, 1, cell_spec(2, 1))]))
        self.assertNotIn("ctx=2", self.log.read_text())


if __name__ == "__main__":
    unittest.main()
