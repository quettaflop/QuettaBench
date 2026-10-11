"""probes.base.run_recorded: every probe run leaves <out>/<probe>/run.json
through RunWriter, with its argv and run_status."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from quettabench.probes.base import run_recorded


def recorded(argv, main, name="gemm", **kw):
    """(raised exception or None, run.json dict or None, the out dir's files)."""
    with tempfile.TemporaryDirectory() as td:
        argv = [a.replace("<TD>", td) for a in argv]
        raised = None
        with mock.patch.object(sys, "argv", ["probe", *argv]):
            try:
                run_recorded(main, name, **kw)
            except BaseException as e:  # noqa: BLE001 -- the test inspects what escaped
                raised = e
        found = sorted(Path(td).rglob("run.json"))
        doc = json.loads(found[0].read_text()) if found else None
        return raised, doc, [str(p.relative_to(td)) for p in found]


class RunRecorded(unittest.TestCase):
    def test_ok_run_records_probe_and_argv(self):
        argv = ["--gpu-label", "H100", "--out", "<TD>", "--wide"]
        raised, doc, files = recorded(argv, lambda: None)
        self.assertIsNone(raised)
        self.assertEqual(files, ["gemm/run.json"])
        self.assertEqual(doc["run_status"], {"status": "ok", "reason": None})
        self.assertEqual(doc["config"]["mode"]["mode"], "probe")
        self.assertEqual(doc["config"]["mode"]["probe"], "gemm")
        self.assertEqual(doc["summary"]["argv"][-1], "--wide")
        self.assertIn("bench_sha", doc["provenance"])

    def test_failure_is_recorded_and_raised(self):
        def boom():
            raise RuntimeError("no GPU")
        raised, doc, _ = recorded(["--gpu-label", "H100", "--out", "<TD>"], boom)
        self.assertIsInstance(raised, RuntimeError)
        self.assertEqual(doc["run_status"], {"status": "failed", "reason": "RuntimeError: no GPU"})

    def test_non_zero_rank_and_no_output_write_nothing(self):
        self.assertEqual(recorded(["--gpu-label", "H100", "--out", "<TD>"], lambda: False)[2], [])
        self.assertEqual(recorded(["--gpu-label", "H100", "--out", "-"], lambda: None)[2], [])

    def test_a_probe_that_names_its_file(self):
        _, _, files = recorded(["--out-raw", "<TD>/h100/raw.jsonl.gz"], lambda: None,
                               name="serving_decode_grid", file_flag="--out-raw")
        self.assertEqual(files, ["h100/serving_decode_grid/run.json"])


if __name__ == "__main__":
    unittest.main()
