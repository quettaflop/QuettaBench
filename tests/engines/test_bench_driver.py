"""BenchDriver's rows are crossval's parse of the same log, and its cell loop
fails where bench.sh fails."""

import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from quettabench.crossval import reader
from quettabench.engines import drivers
from quettabench.engines.base import LaunchSpec
from quettabench.engines.drivers import BenchDriver, BenchFailed
from tests.golden._harness import FIXTURES

FAKE = Path(__file__).with_name("fake_bench.py")


def cell_spec(ctx, bs, **env):
    return LaunchSpec(
        adapter="quettaserve", bench_argv=(sys.executable, str(FAKE)), cwd=".",
        env={"QW_BENCH_PROMPT": str(ctx), "QW_BATCH": str(bs), "QW_WORLD": "1", **env},
        ok_pattern=r"^LOOP ", log_header=f">>> ctx={ctx} bs={bs} world=1")


def alive(pid):
    """Whether pid runs: a zombie waiting for its reaper counts as gone."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ")[1][0] != "Z"
    except (OSError, IndexError):
        return False


def as_points(rows):
    return {(r["cell"]["ctx"], r["cell"]["bs"], r["cell"]["tp"]):
            (r["measured_ms_per_step"], r["group"], r["kv"]) for r in rows}


class GoldenLogs(unittest.TestCase):
    def test_rows_equal_the_reader_parse(self):
        for name in ("bench_loop.log", "bench_criterion.log"):
            with self.subTest(log=name):
                path = FIXTURES / name
                rows = list(BenchDriver(path).rows())
                points, method, _ = reader.bench_log_points(path)
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

    def test_an_interrupted_cell_kills_its_bench(self):
        started, real = [], drivers.subprocess.Popen

        def popen(*args, **kwargs):
            started.append(real(*args, **kwargs))
            return started[-1]

        with mock.patch.object(drivers.subprocess, "Popen", side_effect=popen), \
                mock.patch.object(drivers.sys, "stdout") as out, \
                self.assertRaises(KeyboardInterrupt):
            out.write.side_effect = KeyboardInterrupt
            list(BenchDriver(self.log).run([(1, 1, cell_spec(1, 1, FAKE_BENCH_SLEEP="60"))]))
        (proc,) = started
        self.assertIsNotNone(proc.poll())  # killed and reaped, not left sleeping

    def test_an_interrupted_cell_kills_the_bench_children(self):
        pid_file = self.log.with_name("child.pid")
        spec = cell_spec(1, 1, FAKE_BENCH_SLEEP="60", FAKE_BENCH_CHILD=str(pid_file))
        with mock.patch.object(drivers.sys, "stdout") as out, self.assertRaises(KeyboardInterrupt):
            out.write.side_effect = KeyboardInterrupt
            list(BenchDriver(self.log).run([(1, 1, spec)]))
        pid = int(pid_file.read_text())
        deadline = time.monotonic() + 5
        while alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(alive(pid))

    def test_a_finished_cell_leaves_no_bench_child(self):
        """A bench that exits on its own can leave a child (cargo's bench
        binary) holding the GPUs; its group is killed all the same."""
        pid_file = self.log.with_name("child.pid")
        rows = list(BenchDriver(self.log).run(
            [(1, 1, cell_spec(1, 1, FAKE_BENCH_CHILD=str(pid_file)))]))
        self.assertEqual(len(rows), 1)
        pid = int(pid_file.read_text())
        deadline = time.monotonic() + 5
        while alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(alive(pid))


if __name__ == "__main__":
    unittest.main()
