"""Golden: scripts/crossval/cache.py packing a vmin_fit log into a baseline.

No real vmin_fit log exists in the repo or in the 2026-09-28 bench backup, so
fixtures/vmin_fit_llama.log is what vmin_fit.py itself prints for the llama
workload on the stubs (fake clock, 48 GiB stub GPU), filtered the way
vllm.sh filters it, plus one hand-written pre-normalization RESULT line (no
ms_at_eval) as the last line so the baseline carries a RAW point.
The first test proves the log is still what vmin_fit prints; the second pins
the baseline json cache.py writes from it, byte for byte.
"""

import contextlib
import io
import os
import re
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

from tests.golden._harness import FIXTURES, check_text, crossval_path, fresh_import, run_vmin_fit

VMIN_FIT_LOG = "vmin_fit_llama.log"
BASELINE = "baseline_llama.json"
LEGACY_RESULT = ("RESULT mode=FULL tp=1 kv=float16 ctx=4096 bs=64 ms_per_step=19.874 "
                 "r2=0.99950 prefill_s=1.10 raw=[2.29, 5.67, 9.05]")
VLLM_SH_FILTER = re.compile(r"^(META|RESULT|SKIP|CAPACITY)")
RECORDED_EPOCH = 1790000000  # 2026-09-21T14:13:20Z, frozen recorded_utc
FAKE_BENCH_SHA = "0123abc"


def pack_baseline(log_path, out_path):
    """cache.main() with git and the clock frozen and os.environ empty."""
    fake_git = types.SimpleNamespace(stdout=FAKE_BENCH_SHA + "\n")
    stdout = io.StringIO()
    with crossval_path(), mock.patch.dict(os.environ, {}, clear=True):
        cache = fresh_import("cache")
        with mock.patch.object(cache.subprocess, "run", return_value=fake_git), \
                mock.patch.object(cache.time, "gmtime",
                                  return_value=time.gmtime(RECORDED_EPOCH)), \
                mock.patch("sys.argv", ["cache.py", str(log_path), str(out_path)]), \
                contextlib.redirect_stdout(stdout):
            cache.main()
    return stdout.getvalue().replace(str(out_path), "<OUT>")


class VminFitLogGolden(unittest.TestCase):
    def test_log_is_what_vmin_fit_prints(self):
        run = run_vmin_fit(["llama", "--model", "<MODEL>"], {})
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


if __name__ == "__main__":
    unittest.main()
