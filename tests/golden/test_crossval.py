"""Golden: baseline cache packing and the crossval table text."""

import calendar
import contextlib
import io
import os
import re
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from quettabench import crossval
from tests.contracts.helpers import bench_run_dir, grid_run_dir
from tests.golden._harness import FIXTURES, check_text, crossval_path, fresh_import, grid_argv, run_grid_cli

VMIN_FIT_LOG = "vmin_fit_llama.log"
BASELINE = "baseline_llama.json"
LEGACY_RESULT = ("RESULT mode=FULL tp=1 kv=float16 ctx=4096 bs=64 ms_per_step=19.874 "
                 "r2=0.99950 prefill_s=1.10 raw=[2.29, 5.67, 9.05]")
VLLM_SH_FILTER = re.compile(r"^(META|RESULT|SKIP|CAPACITY)")


class VminFitLogGolden(unittest.TestCase):
    def test_log_is_what_a_grid_run_prints(self):
        run = run_grid_cli(grid_argv("llama"), {})
        self.assertIsNone(run["exit"])
        log = "\n".join(line for line in run["stdout"] if VLLM_SH_FILTER.match(line)) + "\n"
        check_text(self, VMIN_FIT_LOG, log + LEGACY_RESULT + "\n")


class CompatReaderGolden(unittest.TestCase):
    def test_old_baseline_and_new_run_dir_are_one_side(self):
        with tempfile.TemporaryDirectory() as td:
            run = crossval.read_run(grid_run_dir(Path(td) / "vllm-llama"))
        old = crossval.read_baseline(FIXTURES / BASELINE)
        self.assertEqual(run.header, old.header)
        self.assertEqual(run.rows, old.rows)
        self.assertEqual(len(run.rows), 15)


NOW = calendar.timegm(time.strptime("2026-09-21T14:13:20Z", "%Y-%m-%dT%H:%M:%SZ")) + 2.5 * 86400


def run_crossval(ours, theirs):
    """crossval.main stdout, plus the SystemExit message when it exits."""
    out = io.StringIO()
    with tempfile.TemporaryDirectory() as td, crossval_path(), \
            mock.patch.dict(os.environ, {}, clear=True):
        crossval = fresh_import("quettabench.crossval")
        with mock.patch.object(crossval.time, "time", return_value=NOW), \
                contextlib.redirect_stdout(out):
            try:
                crossval.main([str(ours), str(theirs), "--json", str(Path(td) / "xval.json")])
            except SystemExit as e:
                print(f"<exit {e.code}>")
    return out.getvalue()


def run_dirs_text(bench_log, td):
    """The text from run dirs built from bench_log and the golden baseline's data."""
    with contextlib.redirect_stdout(io.StringIO()):
        ours = bench_run_dir(Path(td) / "qs-llama", FIXTURES / bench_log)
        theirs = grid_run_dir(Path(td) / "vllm-llama")
    return run_crossval(ours, theirs)


class TableTextGolden(unittest.TestCase):
    def test_loop_log(self):
        text = run_crossval(FIXTURES / "bench_loop.log", FIXTURES / "baseline_llama.json")
        for marker in ("LOW R2", "RAW", "KV bf16/fp16", "EAGER", "ours-only", "vLLM-only"):
            self.assertIn(marker, text)
        check_text(self, "table_loop.txt", text)

    def test_criterion_log_withholds_every_ratio(self):
        text = run_crossval(FIXTURES / "bench_criterion.log", FIXTURES / "baseline_llama.json")
        self.assertIn("ratios are withheld", text)
        check_text(self, "table_criterion.txt", text)

    def test_run_dirs_give_the_same_text(self):
        for bench, golden in (("bench_loop.log", "table_loop.txt"),
                              ("bench_criterion.log", "table_criterion.txt")):
            with self.subTest(log=bench), tempfile.TemporaryDirectory() as td:
                self.assertEqual(run_dirs_text(bench, td), (FIXTURES / golden).read_text())


if __name__ == "__main__":
    unittest.main()
