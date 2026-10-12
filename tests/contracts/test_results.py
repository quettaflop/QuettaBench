"""metrics/results.py: run.json and requests.jsonl through one writer."""

import dataclasses
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from quettabench.config import loader
from quettabench.config.schema import Config, EngineCfg, HardwareCfg, ModeCfg, WorkloadCfg
from quettabench.engines import vllm
from quettabench.metrics import results
from quettabench.metrics.core import RequestResult

ROOT = Path(__file__).resolve().parents[2]


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

    def test_probe_rows_append_to_rows_jsonl_and_survive_a_discard(self):
        @dataclasses.dataclass
        class Row:
            probe: str
            key: dict
        with tempfile.TemporaryDirectory() as td:
            writer = results.RunWriter(Path(td, "run"), config())
            writer.probe_row(Row("gemm", {"M": 1}))
            writer.probe_row(Row("gemm", {"M": 2}))
            self.assertEqual((Path(td, "run", "rows.jsonl")).read_text(),
                             '{"probe": "gemm", "key": {"M": 1}}\n{"probe": "gemm", "key": {"M": 2}}\n')
            writer.discard()
            self.assertEqual(sorted(p.name for p in Path(td, "run").iterdir()), ["rows.jsonl"])
            results.RunWriter(None, config()).probe_row(Row("gemm", {}))

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

    def test_each_listed_name_is_pinned(self):
        """camelCase splits into words; authorization and oauth are secret
        words; a tokenizer's special-token names are not secrets; key or keys
        as the last word is a secret only after api, secret, private, access
        or auth, so cache_key and routing_key are not."""
        names = {"api_key": True, "hf_token": True, "HF_HOME": False,
                 "num_key_value_heads": False, "prompt_token_ids": False,
                 "authorization": True, "Authorization": True, "bearer_token": True,
                 "my_pat": True, "PATH": False, "patch_size": False, "authToken": True,
                 "hfToken": True, "HFToken": True, "OAUTH": True, "api-keys": True,
                 "eos_token": False, "pad_token": False, "--pad-token": False, "eosToken": False,
                 "bos_token": False, "unk_token": False, "sep_token": False, "cls_token": False,
                 "mask_token": False, "cache_key": False, "routing_key": False,
                 "maxTokens": False, "max_num_batched_tokens": False}
        self.assertEqual({n: results.secret(n) for n in names}, names)

    def test_authorization_header_values_are_redacted(self):
        argv = ["curl", "-H", "Authorization: Bearer sk-live", "--header",
                "authorization:Bearer sk-2", "-H", "Content-Type: application/json",
                "http://h:8000/v1"]
        self.assertEqual(results.redact(argv), [
            "curl", "-H", "Authorization: REDACTED", "--header", "authorization: REDACTED",
            "-H", "Content-Type: application/json", "http://h:8000/v1"])
        self.assertEqual(results.redact({"headers": {"Authorization": "Bearer sk-live",
                                                     "Content-Type": "application/json"}}),
                         {"headers": {"Authorization": "REDACTED",
                                      "Content-Type": "application/json"}})

    def test_a_secret_named_list_or_mapping_is_redacted_whole(self):
        self.assertEqual(results.redact({"api_key": ["k1", 2, {"token": "t", "n": 1}],
                                         "auth": {"user": "u", "n": 1}, "token": [], "secret": {}}),
                         {"api_key": "REDACTED", "auth": "REDACTED", "token": [], "secret": {}})

    def test_an_assignment_after_a_comma_is_redacted(self):
        self.assertEqual(results.redact({"env": "A=1,HF_TOKEN=sk"}), {"env": "A=1,HF_TOKEN=REDACTED"})
        self.assertEqual(results.redact(["-e", "A=1,HF_TOKEN=sk,B=2"]), ["-e", "A=1,HF_TOKEN=REDACTED,B=2"])

    def test_a_quoted_secret_assignment_is_redacted_whole(self):
        self.assertEqual(results.redact({"cmd": "export HF_TOKEN='sk x'"}), {"cmd": "export HF_TOKEN=REDACTED"})
        self.assertEqual(results.redact(["sh", "-c", "export HF_TOKEN='sk x' ; vllm serve m"]),
                         ["sh", "-c", "export HF_TOKEN=REDACTED ; vllm serve m"])

    def test_an_argv_with_non_string_items(self):
        self.assertEqual(results.redact(["--port", 8000, "--api-key", 123, "--tp", 2]),
                         ["--port", 8000, "--api-key", "REDACTED", "--tp", 2])

    def test_name_equals_value_items(self):
        argv = ["docker", "run", "-e", "HF_TOKEN=hf_x", "-e", "HF_TOKEN", "-e", "HF_HOME=/h",
                "engine.extra.vllm.api_key=sk"]
        self.assertEqual(results.redact(argv), [
            "docker", "run", "-e", "HF_TOKEN=REDACTED", "-e", "HF_TOKEN", "-e", "HF_HOME=/h",
            "engine.extra.vllm.api_key=REDACTED"])

    def test_argv_values_are_redacted_in_both_forms(self):
        argv = ["vllm", "serve", "/w", "--api-key", "k1", "--hf-token", "t1", "--x-token=t2",
                "--max-num-batched-tokens", "4096", "--trust-remote-code"]
        self.assertEqual(results.redact(argv), [
            "vllm", "serve", "/w", "--api-key", "REDACTED", "--hf-token", "REDACTED",
            "--x-token=REDACTED", "--max-num-batched-tokens", "4096", "--trust-remote-code"])
        self.assertEqual(results.redact({"probe_args": "--gpu A100 --api-key k1"}),
                         {"probe_args": "--gpu A100 --api-key REDACTED"})

    def test_key_names_are_secret_only_after_a_secret_word(self):
        names = {"--num-keys": False, "cache_keys": False, "primary_keys": False,
                 "api_key": True, "private_key": True, "access_key": True, "secret_keys": True,
                 "auth-key": True, "OPENAI_API_KEY": True, "key": False}
        self.assertEqual({n: results.secret(n) for n in names}, names)

    def test_cookie_and_pass_are_secret_words(self):
        names = {"cookie": True, "Set-Cookie": True, "--pass": True, "db_pass": True,
                 "dbPass": True, "passes": False, "bypass": False, "compass_dir": False}
        self.assertEqual({n: results.secret(n) for n in names}, names)

    def test_the_value_after_a_bare_secret_name_is_hidden(self):
        self.assertEqual(results.redact(["login", "password", "hunter2", "-v"]),
                         ["login", "password", "REDACTED", "-v"])
        self.assertEqual(results.redact(["-e", "HF_TOKEN", "-e", "X=1"]),
                         ["-e", "HF_TOKEN", "-e", "X=1"])

    def test_a_header_after_equals_is_redacted(self):
        self.assertEqual(results.redact(["curl", "--header=Authorization: Bearer sk-live",
                                         "-H=Cookie: sid=abc", "--header=Accept: */*"]),
                         ["curl", "--header=Authorization: REDACTED", "-H=Cookie: REDACTED",
                          "--header=Accept: */*"])

    def test_header_pairs_as_lists_and_tuples(self):
        got = results.redact({"headers": [["Authorization", "Bearer sk-live"],
                                          ("Accept", "*/*"), ("Cookie", "sid=abc")]})
        self.assertEqual(got, {"headers": [["Authorization", "REDACTED"], ["Accept", "*/*"],
                                           ["Cookie", "REDACTED"]]})

    def test_a_command_line_in_any_mapping_value_is_redacted(self):
        doc = {"cmd": "vllm serve /w --api-key sk-live --port 8000",
               "note": "it's  plain -- text", "launch": {"argv": "x --hf-token=t1"}}
        self.assertEqual(results.redact(doc), {
            "cmd": "vllm serve /w --api-key REDACTED --port 8000",
            "note": "it's  plain -- text", "launch": {"argv": "x --hf-token=REDACTED"}})
        self.assertEqual(results.redact({"cmd": "x --api-key 'unbalanced"}),
                         {"cmd": "x --api-key REDACTED"})

    def test_a_command_line_without_a_leading_space_is_redacted(self):
        doc = {"a": "--api-key sk-A", "b": "--api-key=sk-B", "c": "-H 'Authorization: Bearer x'",
               "d": "x\n--api-key y"}
        self.assertEqual(results.redact(doc), {
            "a": "--api-key REDACTED", "b": "--api-key=REDACTED",
            "c": "-H 'Authorization: REDACTED'", "d": "x --api-key REDACTED"})

    def test_an_argv_item_holding_a_command_is_redacted(self):
        self.assertEqual(results.redact(["sh", "-c", "vllm serve --api-key sk-1 --port 1"]),
                         ["sh", "-c", "vllm serve --api-key REDACTED --port 1"])
        self.assertEqual(results.redact(["-H", "Content-Type: application/json"]),
                         ["-H", "Content-Type: application/json"])

    def test_a_bare_word_in_a_command_line_hides_nothing(self):
        """In free text only a flag hides the next word: "pass" is a secret
        word, but "pass the salt" is no secret."""
        for text in ("Please - pass the salt", "You are a pass-through bot"):
            self.assertEqual(results.redact({"note": text}), {"note": text})
            self.assertEqual(results.redact(["--system-prompt", text]), ["--system-prompt", text])
        self.assertEqual(results.redact({"cmd": "vllm serve m --api-key x"}),
                         {"cmd": "vllm serve m --api-key REDACTED"})

    def test_a_secret_assignment_without_a_flag_is_redacted(self):
        self.assertEqual(results.redact({"cmd": "HF_TOKEN=sk vllm serve m"}),
                         {"cmd": "HF_TOKEN=REDACTED vllm serve m"})
        self.assertEqual(results.redact({"cmd": "export HF_TOKEN=sk; vllm serve"}),
                         {"cmd": "export HF_TOKEN=REDACTED ; vllm serve"})
        self.assertEqual(results.redact({"cmd": "HF_HOME=/h vllm serve m"}),
                         {"cmd": "HF_HOME=/h vllm serve m"})

    def test_a_json_value_in_an_argv_is_redacted_as_json(self):
        self.assertEqual(results.redact(["vllm", "--hf-overrides", '{"api_key": "sk", "n": 1}']),
                         ["vllm", "--hf-overrides", '{"api_key": "REDACTED", "n": 1}'])
        self.assertEqual(results.redact({"cmd": """vllm serve --hf-overrides '{"api_key": "sk"}'"""}),
                         {"cmd": """vllm serve --hf-overrides '{"api_key": "REDACTED"}'"""})
        self.assertEqual(results.redact(["--x", '{"n":  1}']), ["--x", '{"n":  1}'])

    def test_bearer_and_authorization_hide_the_next_word_in_a_command(self):
        self.assertEqual(results.redact(["Bearer sk"]), ["Bearer REDACTED"])
        self.assertEqual(results.redact({"cmd": "curl -H authorization sk http://h"}),
                         {"cmd": "curl -H authorization REDACTED http://h"})

    def test_an_assignment_in_a_command_item_keeps_the_rest(self):
        self.assertEqual(results.redact(["sh", "-c", "export HF_TOKEN=sk; vllm serve m"]),
                         ["sh", "-c", "export HF_TOKEN=REDACTED ; vllm serve m"])
        self.assertEqual(results.redact(["sh", "-c", "OPENAI_API_KEY=sk python -m x"]),
                         ["sh", "-c", "OPENAI_API_KEY=REDACTED python -m x"])
        self.assertEqual(results.redact({"cmd": "sh -c 'OPENAI_API_KEY=sk python -m x'"}),
                         {"cmd": "sh -c 'OPENAI_API_KEY=REDACTED python -m x'"})
        self.assertEqual(results.redact(["--api-key=a b"]), ["--api-key=REDACTED"])

    def test_shell_punctuation_stays_its_own_token(self):
        self.assertEqual(results.redact({"cmd": "--api-key sk;echo hi"}),
                         {"cmd": "--api-key REDACTED ; echo hi"})
        self.assertEqual(results.redact({"cmd": "--token sk && ls | wc -l & x"}),
                         {"cmd": "--token REDACTED && ls | wc -l & x"})

    def test_redacted_json_keeps_unicode_and_numbers(self):
        self.assertEqual(results.redact(["--hf-overrides", '{"name": "café", "api_key": "sk", "f": 4.0}']),
                         ["--hf-overrides", '{"name": "café", "api_key": "REDACTED", "f": 4.0}'])

    def test_a_header_or_json_mapping_value_is_redacted(self):
        doc = {"header": "Authorization: Bearer sk", "obj": '{"api_key": "sk"}',
               "list": '["--api-key", "sk"]', "plain": '{"n": 1}', "text": "[draft] notes"}
        self.assertEqual(results.redact(doc), {
            "header": "Authorization: REDACTED", "obj": '{"api_key": "REDACTED"}',
            "list": '["--api-key", "REDACTED"]', "plain": '{"n": 1}', "text": "[draft] notes"})

    def test_url_credentials_are_redacted(self):
        self.assertEqual(results.redact({"db": "postgres://user:hunter2@db/x"}),
                         {"db": "postgres://user:REDACTED@db/x"})
        self.assertEqual(results.redact(["--db", "redis://:pw@h:6379/0", "http://h:8000/v1"]),
                         ["--db", "redis://:REDACTED@h:6379/0", "http://h:8000/v1"])
        self.assertEqual(results.redact({"cmd": "psql postgres://u:pw@h/x --port 1"}),
                         {"cmd": "psql postgres://u:REDACTED@h/x --port 1"})

    def test_passphrase_and_ssh_encryption_signing_keys_are_secret(self):
        names = {"passphrase": True, "SSH_KEY": True, "encryption_key": True,
                 "signing-key": True, "routing_key": False}
        self.assertEqual({n: results.secret(n) for n in names}, names)

    def test_probe_args_and_extra_cmd_reach_no_run_json(self):
        for args in ("--api-key sk-AAA", "--api-key=sk-BBB", "-H 'Authorization: Bearer sk-CCC'",
                     "x\n--api-key sk-DDD"):
            cfg = Config(EngineCfg(extra={"vllm": {"cmd": "-H 'Authorization: Bearer sk-FFF'"}}),
                         WorkloadCfg(), ModeCfg(mode="probe", probe="gemm", probe_args=args),
                         HardwareCfg(), run_name="x")
            with tempfile.TemporaryDirectory() as td:
                results.RunWriter(td, cfg).finish({})
                self.assertNotIn("sk-", (Path(td) / "run.json").read_text(), args)

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
from tests.contracts.test_results import config
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
        with mock.patch.object(results.subprocess, "run", side_effect=OSError("no git")), \
                mock.patch.object(results.env_map, "read_env", return_value={}):
            self.assertEqual(results.bench_sha(), "")

    def test_without_a_repo_the_images_qbench_sha_stands_in(self):
        with tempfile.TemporaryDirectory() as td, \
                mock.patch.object(results.env_map, "read_env", return_value={"detect.bench_sha": "abc123"}) as read:
            self.assertEqual(results.bench_sha(td), "abc123")
        read.assert_called_once_with(only=("QBENCH_SHA",))
        self.assertRegex(results.bench_sha(), r"^[0-9a-f]{40}$")

    def test_without_git_run_json_records_the_images_qbench_sha(self):
        with tempfile.TemporaryDirectory() as td, \
                mock.patch.object(results.subprocess, "run", side_effect=OSError("no git")), \
                mock.patch.dict(results.os.environ, {"QBENCH_SHA": "7eea8ff"}):
            results.RunWriter(Path(td) / "run1", config()).finish({"n": 0})
            doc = json.loads((Path(td) / "run1" / "run.json").read_text())
        self.assertEqual(doc["provenance"]["bench_sha"], "7eea8ff")


if __name__ == "__main__":
    unittest.main()
