"""metrics/results.py: run.json + requests.jsonl through one writer."""

import dataclasses
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from quettabench.config import loader
from quettabench.config.schema import (
    Config,
    EngineCfg,
    HardwareCfg,
    ModeCfg,
    WorkloadCfg,
)
from quettabench.engines import vllm
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


class AtomicWrite(unittest.TestCase):
    def test_run_json_goes_through_a_replace(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "run1"
            writer = results.RunWriter(out, config())
            with mock.patch.object(results.os, "replace", wraps=results.os.replace) as replace:
                writer.finish({"n": 0})
            self.assertEqual(replace.call_args.args, (out / "run.json.tmp", out / "run.json"))
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["requests.jsonl", "run.json"])

    def test_a_failed_write_leaves_the_old_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run.json"
            path.write_text("old\n")
            with mock.patch.object(results.os, "replace", side_effect=OSError("disk")), \
                    self.assertRaises(OSError):
                results.write_atomic(path, "new\n")
            self.assertEqual(path.read_text(), "old\n")


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

    def test_set_fields_and_extra_are_written(self):
        r = result(0, ttft_basis="arrival", accepted_tokens=3)
        self.assertNotIn("cell", results.jsonl_row(r))
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
            "HF_TOKEN": "REDACTED", "HF_HOME": "/h", "OPENAI_API_KEY": "REDACTED",
            "AWS_SECRET": "REDACTED", "NCCL_ALGO": "Ring", "CUDA_VISIBLE_DEVICES": "0,1"})

    def test_names_match_without_case_and_only_as_words(self):
        kwargs = {"api_key": "k", "hf_token": "t", "Password": "p", "max_num_batched_tokens": 4096,
                  "kv_cache_dtype": "fp8", "keyless": 1}
        self.assertEqual(results.redact(kwargs), {
            "api_key": "REDACTED", "hf_token": "REDACTED", "Password": "REDACTED",
            "max_num_batched_tokens": 4096, "kv_cache_dtype": "fp8", "keyless": 1})

    def test_word_rules(self):
        """token/key/apikey/pat only as the last word; secret, password,
        credential, bearer and auth anywhere as a word."""
        names = {"num_key_value_heads": False, "prompt_token_ids": False, "HF_HOME": False,
                 "hf_home": False, "path": False, "author": False, "GITHUB_PAT": True,
                 "auth_header": True, "GOOGLE_CREDENTIALS_FILE": True, "bearer": True,
                 "x-credential": True, "apiKey": True, "AWS_SECRET_ACCESS_KEY": True}
        self.assertEqual({n: results.secret(n) for n in names}, names)

    def test_a_secret_named_list_is_redacted_item_by_item(self):
        self.assertEqual(results.redact({"api_key": ["k1", 2, {"token": "t", "n": 1}]}),
                         {"api_key": ["REDACTED", "REDACTED", {"token": "REDACTED", "n": 1}]})

    def test_an_argv_with_non_string_items(self):
        self.assertEqual(results.redact(["--port", 8000, "--api-key", 123, "--tp", 2]),
                         ["--port", 8000, "--api-key", "REDACTED", "--tp", 2])

    def test_name_equals_value_items(self):
        argv = ["docker", "run", "-e", "HF_TOKEN=hf_x", "-e", "HF_HOME=/h", "-e", "HF_TOKEN",
                "engine.extra.vllm.api_key=sk"]
        self.assertEqual(results.redact(argv), [
            "docker", "run", "-e", "HF_TOKEN=REDACTED", "-e", "HF_HOME=/h", "-e", "HF_TOKEN",
            "engine.extra.vllm.api_key=REDACTED"])

    def test_argv_values_are_redacted_in_both_forms(self):
        argv = ["vllm", "serve", "/w", "--api-key", "k1", "--hf-token", "t1", "--x-token=t2",
                "--max-num-batched-tokens", "4096", "--trust-remote-code"]
        self.assertEqual(results.redact(argv), [
            "vllm", "serve", "/w", "--api-key", "REDACTED", "--hf-token", "REDACTED",
            "--x-token=REDACTED", "--max-num-batched-tokens", "4096", "--trust-remote-code"])
        self.assertEqual(results.redact({"probe_args": "--gpu A100 --api-key k1"}),
                         {"probe_args": "--gpu A100 --api-key REDACTED"})

    def test_extra_vllm_api_key_reaches_no_results_file(self):
        (cfg,) = loader.load(ROOT / "configs" / "runs" / "llama.yaml",
                             ["engine.model=/w", "mode.grid=null"], env={})
        cfg = dataclasses.replace(cfg, engine=dataclasses.replace(
            cfg.engine, extra={"vllm": {"api_key": "sk-live"}}))
        spec = vllm.ADAPTER.translate(cfg)
        self.assertEqual(spec.env["VLLM_API_KEY"], "sk-live")
        self.assertNotIn("sk-live", spec.server_argv)
        self.assertNotIn("sk-live", json.dumps(results.redact(dataclasses.asdict(spec))))
        with tempfile.TemporaryDirectory() as td:
            results.RunWriter(td, cfg, launch_env=spec.env).finish({})
            self.assertNotIn("sk-live", (Path(td) / "run.json").read_text())

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


if __name__ == "__main__":
    unittest.main()
