"""crossval.main: input kinds told apart by content, the json report, and
--allow-method-mismatch."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from quettabench import crossval
from tests.golden._harness import FIXTURES

BASELINE = FIXTURES / "baseline_llama.json"


def run(argv, env=None):
    out = io.StringIO()
    with mock.patch.dict(os.environ, env or {}, clear=True), contextlib.redirect_stdout(out):
        crossval.main([str(a) for a in argv])
    return out.getvalue()


class Door(unittest.TestCase):
    def test_json_report(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "r.json"
            run([FIXTURES / "bench_loop.log", BASELINE, "--json", out])
            report = json.loads(out.read_text())
        self.assertEqual(report["theirs"]["header"]["engine"], "vllm")
        self.assertEqual(report["ours"]["header"]["method"], "loop")
        rows = {(r["ctx"], r["bs"]): r for r in report["rows"]}
        self.assertIsNone(rows[(4096, 8)]["ratio"])
        self.assertEqual(rows[(4096, 8)]["withheld"], ["kv"])
        self.assertEqual(report["ours_only"], [[2048, 4, 2], [8192, 1, 1]])

    def test_default_report_path(self):
        with tempfile.TemporaryDirectory() as td, \
                mock.patch.object(crossval, "XVAL_DIR", Path(td)):
            run([FIXTURES / "bench_loop.log", BASELINE])
            self.assertTrue((Path(td) / "bench_loop_vs_baseline_llama.json").is_file())

    def test_method_mismatch_forced_by_flag_or_env(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [FIXTURES / "bench_criterion.log", BASELINE, "--json", Path(td) / "r.json"]
            self.assertIn("ratios are withheld", run(argv))
            for forced in (run(argv + ["--allow-method-mismatch"]),
                           run(argv, {"XVAL_ALLOW_METHOD_MISMATCH": "1"})):
                self.assertNotIn("ratios are withheld", forced)
                self.assertIn("1.04x", forced)

    def test_engine_facts_from_env(self):
        with tempfile.TemporaryDirectory() as td:
            text = run([FIXTURES / "bench_loop.log", BASELINE, "--json", Path(td) / "r.json"],
                       {"NCCL_ALGO": "Ring", "XVAL_QUANT": "nvfp4"})
        self.assertIn("NCCL MISMATCH engine=Ring/None vllm=auto/auto", text)
        self.assertIn("QUANT MISMATCH engine=nvfp4 vllm=float16; ratios withheld", text)

    def test_fidelity_or_band_fail_exits_1_after_the_json(self):
        for kind in ("logit_summary", "band"):
            with self.subTest(kind), tempfile.TemporaryDirectory() as td, \
                    mock.patch.object(crossval, "read_side"), \
                    mock.patch.object(crossval, "read_rows"), \
                    mock.patch.object(crossval, "render", return_value=""), \
                    mock.patch.object(crossval, "pair",
                                      return_value=mock.Mock(rows=[], agreement=[])), \
                    mock.patch.object(crossval, "report_json", return_value={}), \
                    mock.patch.object(crossval, "compare_runs",
                                      return_value=[{"kind": kind, "verdict": "FAIL"}]):
                with self.assertRaises(SystemExit) as e:
                    run([td, td, "--json", Path(td) / "r.json"])
                self.assertEqual(e.exception.code, 1)
                self.assertTrue((Path(td) / "r.json").is_file())

    def test_missing_input(self):
        with self.assertRaises(SystemExit) as e:
            run(["/nonexistent/run", BASELINE])
        self.assertEqual(e.exception.code, "no results at /nonexistent/run")


if __name__ == "__main__":
    unittest.main()
