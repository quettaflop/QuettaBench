"""Golden: quettabench/crossval/cache.py packing a vmin_fit log into a baseline.

No real vmin_fit log exists in the repo or in the 2026-09-28 bench backup, so
fixtures/vmin_fit_llama.log is what vmin_fit.py printed for the llama
workload on the stubs (fake clock, 48 GiB stub GPU), filtered the way
vllm.sh filters it, plus one hand-written pre-normalization RESULT line (no
ms_at_eval) as the last line so the baseline carries a RAW point.
The first test proves the log is still what a grid run (qbench run, which
replaced vmin_fit.py) prints; the second pins the baseline json cache.py
writes from it, byte for byte. The third holds the compat baseline a grid
run writes itself (results.RunWriter.write_baseline) to cache.py's bytes on
that run's log, and, with the hand-written RAW point added back, to
baseline_llama.json and so to table.py's golden text.
"""

import contextlib
import io
import json
import os
import re
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

from tests.golden._harness import (
    FIXTURES,
    check_text,
    crossval_path,
    fresh_import,
    grid_argv,
    run_grid_cli,
)
from tests.golden.test_table_text import run_table

VMIN_FIT_LOG = "vmin_fit_llama.log"
BASELINE = "baseline_llama.json"
LEGACY_RESULT = ("RESULT mode=FULL tp=1 kv=float16 ctx=4096 bs=64 ms_per_step=19.874 "
                 "r2=0.99950 prefill_s=1.10 raw=[2.29, 5.67, 9.05]")
VLLM_SH_FILTER = re.compile(r"^(META|RESULT|SKIP|CAPACITY)")
RECORDED_EPOCH = 1790000000  # 2026-09-21T14:13:20Z, frozen recorded_utc
FAKE_BENCH_SHA = "0123abc"


@contextlib.contextmanager
def frozen_capture():
    """git rev-parse and the UTC clock frozen as pack_baseline freezes them."""
    fake_git = types.SimpleNamespace(stdout=FAKE_BENCH_SHA + "\n")
    recorded = time.gmtime(RECORDED_EPOCH)
    with mock.patch("subprocess.run", return_value=fake_git), \
            mock.patch("time.gmtime", return_value=recorded):
        yield


def pack_baseline(log_path, out_path):
    """cache.main() with git and the clock frozen and os.environ empty."""
    fake_git = types.SimpleNamespace(stdout=FAKE_BENCH_SHA + "\n")
    stdout = io.StringIO()
    with crossval_path(), mock.patch.dict(os.environ, {}, clear=True):
        cache = fresh_import("quettabench.crossval.cache")
        with mock.patch.object(cache.subprocess, "run", return_value=fake_git), \
                mock.patch.object(cache.time, "gmtime",
                                  return_value=time.gmtime(RECORDED_EPOCH)), \
                mock.patch("sys.argv", ["cache.py", str(log_path), str(out_path)]), \
                contextlib.redirect_stdout(stdout):
            cache.main()
    return stdout.getvalue().replace(str(out_path), "<OUT>")


class VminFitLogGolden(unittest.TestCase):
    def test_log_is_what_a_grid_run_prints(self):
        run = run_grid_cli(grid_argv("llama"), {})
        self.assertIsNone(run["exit"])
        log = "\n".join(l for l in run["stdout"] if VLLM_SH_FILTER.match(l)) + "\n"
        check_text(self, VMIN_FIT_LOG, log + LEGACY_RESULT + "\n")


class CachePackingGolden(unittest.TestCase):
    def test_baseline_json(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "vllm-llama.json"
            printed = pack_baseline(FIXTURES / VMIN_FIT_LOG, out)
            self.assertEqual(printed, "wrote <OUT>: 15 points, vllm 0.0.0+stub on STUB-GPU-48GiB\n")
            check_text(self, BASELINE, out.read_text())


class CompatBaselineGolden(unittest.TestCase):
    def test_grid_run_writes_cache_py_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            with frozen_capture():
                run = run_grid_cli(grid_argv("llama"), {}, baselines=td)
            compat = (Path(td) / "vllm-llama.json").read_text()
            log = Path(td) / "raw.txt"
            log.write_text("\n".join(l for l in run["stdout"] if VLLM_SH_FILTER.match(l)) + "\n")
            pack_baseline(log, Path(td) / "cache.json")
            self.assertEqual(compat, (Path(td) / "cache.json").read_text())

            # No run prints the pre-normalization RESULT line, so it is put back
            # by hand; then the file is the golden baseline and table.py's text.
            baseline = json.loads(compat)
            golden = (FIXTURES / BASELINE).read_text()
            baseline["points"].append(json.loads(golden)["points"][-1])
            with_raw = Path(td) / "vllm-llama-raw.json"
            with_raw.write_text(json.dumps(baseline, indent=2))
            self.assertEqual(with_raw.read_text(), golden)
            for bench, text in (("bench_loop.log", "table_loop.txt"),
                                ("bench_criterion.log", "table_criterion.txt")):
                self.assertEqual(run_table(FIXTURES / bench, with_raw),
                                 (FIXTURES / text).read_text())


if __name__ == "__main__":
    unittest.main()
