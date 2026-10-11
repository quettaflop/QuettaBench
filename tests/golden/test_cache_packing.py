"""Golden: a grid run and the old baseline it replaces read as the same Side.

No real vmin_fit log exists in the repo or in the 2026-09-28 bench backup, so
fixtures/vmin_fit_llama.log is what vmin_fit.py printed for the llama
workload on the stubs (fake clock, 48 GiB stub GPU), filtered the way
vllm.sh filtered it (retired in af0e9b8), plus one hand-written pre-normalization RESULT line (no
ms_at_eval) as the last line. cache.py (retired in dc833f9) packed that log into
fixtures/baseline_llama.json, the old baseline layout crossval still reads.

The first test proves the log is still what a grid run (qbench run, which
replaced vmin_fit.py) prints. The second reads that grid run's results dir,
with the hand-written RAW cell added back, and the old baseline through the
compat reader: same header, same rows. Grid runs no longer write a
baseline; that equality is what lets the old files stay readable.
"""

import re
import tempfile
import unittest
from pathlib import Path

from quettabench.crossval import reader
from tests.crossval._rundirs import grid_run_dir
from tests.golden._harness import FIXTURES, check_text, grid_argv, run_grid_cli

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
            run = reader.read_run(grid_run_dir(Path(td) / "vllm-llama"))
        old = reader.read_baseline(FIXTURES / BASELINE)
        self.assertEqual(run.header, old.header)
        self.assertEqual(run.rows, old.rows)
        self.assertEqual(len(run.rows), 15)


if __name__ == "__main__":
    unittest.main()
