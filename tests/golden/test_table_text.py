"""Golden: the crossval comparison text, exact, through both input paths.

The text was pinned from the old quettabench/crossval/table.py. Baseline is
fixtures/baseline_llama.json (an old cache.py baseline, read by the compat
reader). No real bench log exists in the repo or the 2026-09-28 backup, so
both bench logs are synthesized in the formats the bench parser reads:

- bench_loop.log: LOOP and [batch_bench] lines (bench.sh, batch_bench). Rows
  cover a plain ratio, LOW R2, RAW, a KV-mismatch withhold, EAGER, a
  batch_bench row, a truncated batch_bench line, ours-only cells (one at tp 2)
  and, by omission, vLLM-only cells.
- bench_criterion.log: criterion decode groups, a method mismatch, so every
  ratio is withheld.

Each text comes out of crossval.main twice: from the bench log against the
old baseline json, and from a QuettaServe run dir holding that log's rows
against a stub grid run dir holding the baseline's cells (tests/crossval/
_rundirs.py). The clock is frozen 2.5 days after the baseline's
recorded_utc and os.environ is empty, so the header and the NCCL, serving
and quant rules are fixed.
"""

import calendar
import contextlib
import io
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from tests.crossval._rundirs import bench_run_dir, grid_run_dir
from tests.golden._harness import FIXTURES, check_text, crossval_path, fresh_import

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
