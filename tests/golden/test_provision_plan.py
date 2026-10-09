"""Golden: provision.py's plan under --dry-run (it replaced bootstrap.sh;
fixtures/bootstrap_plan.json keeps bootstrap.sh's plan content in
provision's words).

The vLLM python is a stand-in that answers `-m pip show vllm` with
success and the GPU check sees no GPU, so the plan depends on neither the
host's vLLM nor its GPUs. The env file goes to the scratch dir, so a dry
run that wrote anything would leave it behind there. Scratch paths read
<TMP>.
"""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from quettabench import provision
from tests.golden._harness import RUNS, check_json


def run_dry_provision(names, scoped):
    with tempfile.TemporaryDirectory() as tmp:
        fake_py = Path(tmp, "vllm-python")
        fake_py.write_text("#!/bin/sh\nexit 0\n")
        fake_py.chmod(0o755)
        argv = [*(str(RUNS / f"{n}.yaml") for n in names), "--dry-run",
                "--env-file", f"{tmp}/qbench.env",
                "--set", f"hardware.provision.cache_root={tmp}/cache",
                "--set", f"hardware.provision.py={fake_py}"]
        argv += [a for item in scoped for a in ("--set", item.replace("<TMP>", tmp))]
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(provision, "compute_capability", return_value=None), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = provision.main(argv)
        leftovers = sorted(p.name for p in Path(tmp).iterdir() if p != fake_py)
    return {"rc": rc, "stdout": out.getvalue(),
            "stderr": err.getvalue().replace(tmp, "<TMP>").splitlines(),
            "files_left_behind": leftovers}


class ProvisionPlanGolden(unittest.TestCase):
    def test_dry_run_plans(self):
        record = {
            "gaps": run_dry_provision(["llama", "qwen3", "deepseek"], [
                "deepseek:hardware.provision.qs_ckpt=/ckpt/ds-0731-tp8-fp4",
                "qwen3:hardware.provision.weights=/weights/qwen3",
                "deepseek:hardware.provision.hf_id=deepseek-ai/DeepSeek-V4-Flash",
            ]),
            "complete": run_dry_provision(["qwen3"], [
                "qwen3:hardware.provision.bench_bin=<TMP>/vllm-python",
                "qwen3:hardware.provision.qs_ckpt=/ckpt/qwen3",
                "qwen3:hardware.provision.weights=/weights/qwen3",
            ]),
        }
        self.assertEqual(record["gaps"]["rc"], 1)
        self.assertEqual(record["complete"]["rc"], 0)
        check_json(self, "bootstrap_plan", record)

    def test_a_refused_run_yaml_is_missing(self):
        record = run_dry_provision(["qwen"], [])
        self.assertEqual(record["rc"], 1)
        self.assertIn("attention=hybrid needs kv_bytes", record["stderr"][0])
        self.assertTrue(record["stderr"][0].startswith("[provision] MISSING: "))


if __name__ == "__main__":
    unittest.main()
