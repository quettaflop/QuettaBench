"""Golden: scripts/crossval/table.py stdout, exact text.

Baseline is fixtures/baseline_llama.json (cache.py output, see
test_cache_packing). No real bench log exists in the repo or the 2026-09-28
backup, so both bench logs are synthesized in the formats table.py parses:

- bench_loop.log: LOOP and [batch_bench] lines (bench.sh, batch_bench). Rows
  cover a plain ratio, LOW R2, RAW, a KV-mismatch withhold, EAGER, a
  batch_bench row, a truncated batch_bench line, ours-only cells (one at tp 2)
  and, by omission, vLLM-only cells.
- bench_criterion.log: criterion decode groups, a method mismatch, so every
  ratio is withheld.

The clock is frozen 2.5 days after the baseline's recorded_utc and
os.environ is empty, so the header and the NCCL/serving/quant rules are fixed.
"""

import calendar
import contextlib
import io
import os
import time
import unittest
from unittest import mock

from tests.golden._harness import FIXTURES, check_text, crossval_path, fresh_import

NOW = calendar.timegm(time.strptime("2026-09-21T14:13:20Z", "%Y-%m-%dT%H:%M:%SZ")) + 2.5 * 86400


def run_table(bench_log, baseline):
    """table.main() stdout, plus the SystemExit message when it exits."""
    out = io.StringIO()
    with crossval_path(), mock.patch.dict(os.environ, {}, clear=True):
        table = fresh_import("table")
        with mock.patch.object(table.time, "time", return_value=NOW), \
                mock.patch("sys.argv", ["table.py", str(bench_log), str(baseline)]), \
                contextlib.redirect_stdout(out):
            try:
                table.main()
            except SystemExit as e:
                print(f"<exit {e.code}>")
    return out.getvalue()


class TableTextGolden(unittest.TestCase):
    def test_loop_log(self):
        text = run_table(FIXTURES / "bench_loop.log", FIXTURES / "baseline_llama.json")
        for marker in ("LOW R2", "RAW", "KV bf16/fp16", "EAGER", "ours-only", "vLLM-only"):
            self.assertIn(marker, text)
        check_text(self, "table_loop.txt", text)

    def test_criterion_log_withholds_every_ratio(self):
        text = run_table(FIXTURES / "bench_criterion.log", FIXTURES / "baseline_llama.json")
        self.assertIn("ratios are withheld", text)
        check_text(self, "table_criterion.txt", text)


if __name__ == "__main__":
    unittest.main()
