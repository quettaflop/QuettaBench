"""crossval/reader.py: each input kind read into a Side, on the golden data."""

import contextlib
import io
import math
import tempfile
import unittest
from pathlib import Path

from quettabench.crossval import reader, table
from tests.crossval._rundirs import bench_run_dir, grid_run_dir
from tests.golden._harness import FIXTURES


def by_cell(side):
    return {(r["ctx"], r["bs"], r["tp"]): r for r in side.rows}


class BenchLog(unittest.TestCase):
    def test_loop_log(self):
        side = reader.read_bench_log(FIXTURES / "bench_loop.log")
        self.assertEqual(side.header["method"], "loop")
        self.assertEqual(side.notes, [
            "ignored 1 truncated-model batch_bench line(s) (layers != all)",
            "engine gdn kernel(s): Tiled"])
        cells = by_cell(side)
        self.assertEqual(len(cells), 8)
        self.assertEqual(cells[(100, 1, 1)]["group"], "batch_bench")
        self.assertEqual(cells[(100, 1, 1)]["kv"], "?")
        self.assertEqual(cells[(100, 8, 1)]["mode"], "eager")
        self.assertEqual(cells[(2048, 4, 2)]["ms_per_step"], 4.1)
        self.assertTrue(all(r["measured"] for r in side.rows))

    def test_criterion_log(self):
        side = reader.read_bench_log(FIXTURES / "bench_criterion.log")
        self.assertEqual(side.header["method"], "criterion")
        cells = by_cell(side)
        # decode_graph outranks plain decode on the same cell
        self.assertEqual(cells[(100, 1, 1)]["group"], "decode_graph")
        self.assertEqual(cells[(1024, 1, 2)]["group"], "decode_tp2")
        self.assertIsNone(cells[(1024, 1, 1)]["kv"])

    def test_parse_is_table_py_parse(self):
        for log, method in (("bench_loop.log", "loop"), ("bench_criterion.log", "step")):
            with self.subTest(log=log):
                with contextlib.redirect_stdout(io.StringIO()) as printed:
                    old = table.ours_points(FIXTURES / log)
                points, _, notes = reader.bench_log_points(FIXTURES / log)
                self.assertEqual(old, (points, method))
                self.assertEqual(printed.getvalue(), "".join(n + "\n" for n in notes))

    def test_engine_facts_fill_the_header(self):
        side = reader.read_bench_log(FIXTURES / "bench_loop.log",
                                     {"quant": "nvfp4", "nccl_algo": "Ring"})
        self.assertEqual((side.header["quant"], side.header["nccl_algo"]), ("nvfp4", "Ring"))


class Baseline(unittest.TestCase):
    def test_golden_baseline(self):
        side = reader.read_baseline(FIXTURES / "baseline_llama.json")
        self.assertEqual(len(side.rows), 15)
        h = side.header
        self.assertEqual((h["engine"], h["version"], h["method"], h["timestamp"]),
                         ("vllm", "0.0.0+stub", "slope", "2026-09-21T14:13:20Z"))
        self.assertEqual((h["quant"], h["serving_style"], h["comm_bound_bs"]),
                         ("float16", "aggregated", None))
        raw = by_cell(side)[(4096, 64, 1)]
        self.assertIsNone(raw["ms_at_eval"])
        self.assertFalse(raw["measured"])

    def test_age(self):
        side = reader.read_baseline(FIXTURES / "baseline_llama.json")
        self.assertTrue(math.isclose(reader.age_days(side, 1790000000 + 86400), 1.0))


class RunDir(unittest.TestCase):
    def test_grid_run_reads_as_the_baseline_packed_from_its_log(self):
        with tempfile.TemporaryDirectory() as td:
            run = reader.read_run(grid_run_dir(Path(td) / "vllm-llama"))
        base = reader.read_baseline(FIXTURES / "baseline_llama.json")
        self.assertEqual(run.header, base.header)
        self.assertEqual(run.rows, base.rows)

    def test_bench_run_reads_as_its_log(self):
        for log in ("bench_loop.log", "bench_criterion.log"):
            with self.subTest(log=log), tempfile.TemporaryDirectory() as td:
                run = reader.read_run(bench_run_dir(Path(td) / "qs", FIXTURES / log))
                side = reader.read_bench_log(FIXTURES / log)
                self.assertEqual(run.rows, side.rows)
                self.assertEqual(run.notes, side.notes)
                self.assertEqual(run.header["method"], side.header["method"])


if __name__ == "__main__":
    unittest.main()
