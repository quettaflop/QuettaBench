"""metrics/results.py: run.json + requests.jsonl through one writer."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from quettabench.config.schema import (
    Config,
    EngineCfg,
    HardwareCfg,
    ModeCfg,
    WorkloadCfg,
)
from quettabench.metrics import results
from quettabench.metrics.core import RequestResult

ROOT = Path(__file__).resolve().parent.parent


def config():
    return Config(engine=EngineCfg(model="m", kv_dtype="fp8"), workload=WorkloadCfg(),
                  mode=ModeCfg(), hardware=HardwareCfg(collective={"NCCL_ALGO": "Ring"}),
                  provenance={"engine_sha": "abc"}, origins={"engine.model": "cli"})


def result(i, **kw):
    return RequestResult(success=True, ttft=0.01, e2el=0.05, input_tokens=10 + i,
                         output_tokens=4, request_index=i, **kw)


class RoundTrip(unittest.TestCase):
    def test_rows_and_run_json(self):
        with tempfile.TemporaryDirectory() as td:
            writer = results.RunWriter(Path(td) / "run1", config(), launch_env={"A": "1"})
            for i in range(3):
                writer.append(result(i, ttft_basis="slot"))
            run = writer.finish({"n": 3}, status="partial", reason="1 of 3 failed")
            on_disk = json.loads((Path(td) / "run1" / "run.json").read_text())
            rows = [json.loads(line) for line in
                    (Path(td) / "run1" / "requests.jsonl").read_text().splitlines()]
        self.assertEqual(on_disk, json.loads(json.dumps(run)))
        self.assertEqual(on_disk["schema_version"], 1)
        self.assertEqual(on_disk["run_status"], {"status": "partial", "reason": "1 of 3 failed"})
        self.assertEqual(on_disk["summary"], {"n": 3})
        self.assertIsNone(on_disk["steps"])
        self.assertEqual(on_disk["origins"], {"engine.model": "cli"})
        self.assertNotIn("origins", on_disk["config"])
        self.assertEqual(on_disk["config"]["engine"]["model"], "m")
        prov = on_disk["provenance"]
        self.assertEqual((prov["engine"], prov["engine_sha"], prov["nccl_algo"], prov["kv_dtype"]),
                         ("vllm", "abc", "Ring", "fp8"))
        self.assertTrue(prov["started"] <= prov["finished"])
        self.assertEqual([r["request_index"] for r in rows], [0, 1, 2])
        self.assertEqual(rows[0]["ttft_basis"], "slot")
        self.assertEqual(rows[0], results.jsonl_row(result(0, ttft_basis="slot")))

    def test_new_fields_stay_out_of_the_old_layout(self):
        r = result(0, ttft_basis="arrival", accepted_tokens=3)
        self.assertNotIn("ttft_basis", results.request_row(r))
        self.assertEqual(results.jsonl_row(r, probe=2)["accepted_tokens"], 3)
        self.assertEqual(results.jsonl_row(r, probe=2)["probe"], 2)

    def test_no_dir_writes_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            writer = results.RunWriter(None, config())
            writer.append(result(0))
            run = writer.finish({})
            self.assertEqual(list(Path(td).iterdir()), [])
        self.assertEqual(run["run_status"]["status"], "ok")
        self.assertEqual(writer.rows, 1)

    def test_unknown_status_is_refused(self):
        with self.assertRaises(ValueError):
            results.RunWriter(None, config()).finish({}, status="done")


class Redaction(unittest.TestCase):
    def test_secret_keys_are_redacted(self):
        env = {"HF_TOKEN": "t", "HF_HOME": "/h", "OPENAI_API_KEY": "k", "AWS_SECRET": "s",
               "NCCL_ALGO": "Ring", "CUDA_VISIBLE_DEVICES": "0,1"}
        self.assertEqual(results.redact(env), {
            "HF_TOKEN": "REDACTED", "HF_HOME": "REDACTED", "OPENAI_API_KEY": "REDACTED",
            "AWS_SECRET": "REDACTED", "NCCL_ALGO": "Ring", "CUDA_VISIBLE_DEVICES": "0,1"})

    def test_run_json_never_holds_a_secret_value(self):
        with tempfile.TemporaryDirectory() as td:
            writer = results.RunWriter(td, config(), launch_env={"HF_TOKEN": "hf_live_secret"})
            writer.finish({})
            self.assertNotIn("hf_live_secret", (Path(td) / "run.json").read_text())


CRASH = """
import os, sys
from quettabench.metrics.results import RunWriter
from quettabench.metrics.core import RequestResult
from tests.test_results_writer import config
w = RunWriter(sys.argv[1], config())
for i in range(int(sys.argv[2])):
    w.append(RequestResult(success=True, request_index=i))
os._exit(9)
"""


class Crash(unittest.TestCase):
    def test_rows_written_before_a_crash_survive(self):
        with tempfile.TemporaryDirectory() as td:
            proc = subprocess.run([sys.executable, "-c", CRASH, td, "5"], cwd=ROOT,
                                  capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 9, proc.stderr)
            lines = (Path(td) / "requests.jsonl").read_text().splitlines()
            self.assertFalse((Path(td) / "run.json").exists())
        self.assertEqual([json.loads(line)["request_index"] for line in lines], list(range(5)))


class BenchSha(unittest.TestCase):
    def test_this_checkout_has_a_sha(self):
        self.assertRegex(results.bench_sha(), r"^[0-9a-f]{40}$")

    def test_not_a_repo_is_empty(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(results.bench_sha(td), "")

    def test_no_git_is_empty(self):
        with mock.patch.object(results.subprocess, "run", side_effect=OSError("no git")):
            self.assertEqual(results.bench_sha(), "")


class LegacyLayout(unittest.TestCase):
    def test_per_turn_file_sits_beside_the_output(self):
        ts = mock.Mock(to_dict=lambda: {"turn_index": 0})
        with tempfile.TemporaryDirectory() as td, mock.patch("builtins.print"):
            results.write_legacy_per_turn([ts], str(Path(td) / "r.json"), {"a": 1})
            data = json.loads((Path(td) / "r_per_turn.json").read_text())
        self.assertEqual(data, {"config": {"a": 1}, "per_turn": [{"turn_index": 0}]})


if __name__ == "__main__":
    unittest.main()
