"""Golden: scripts/crossval/bootstrap.sh plan text under XVAL_DRYRUN=1.

bootstrap.sh writes .xval_env.tmp next to itself even on a dry run, so it
runs from a scratch copy of scripts/crossval, never from the tree. The
vLLM python is a stand-in that answers `-m pip show vllm` with success, so
the plan does not depend on whether the host has vLLM. Scratch paths are
replaced with <HERE> and <TMP> before comparing.
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.golden._harness import CROSSVAL, REPO, check_json

# Scratch tree mirrors the repo layout: bootstrap.sh finds xval_config.py and
# configs/ relative to itself.
COPIED = (
    (CROSSVAL / "bootstrap.sh", "scripts/crossval/bootstrap.sh"),
    (REPO / "quettabench/config/xval_config.py", "quettabench/config/xval_config.py"),
    (REPO / "configs/workloads.json", "configs/workloads.json"),
    (REPO / "configs/xval.yaml", "configs/xval.yaml"),
)


def run_dry_bootstrap(workloads, extra_env):
    with tempfile.TemporaryDirectory() as tmp:
        here = Path(tmp, "scripts", "crossval")
        for src, dst in COPIED:
            Path(tmp, dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, Path(tmp, dst))
        fake_py = Path(tmp, "vllm-python")
        fake_py.write_text("#!/bin/sh\nexit 0\n")
        fake_py.chmod(0o755)
        env = {"PATH": os.environ["PATH"], "XVAL_DRYRUN": "1", "XVAL_PY": str(fake_py),
               "XVAL_CACHE_ROOT": f"{tmp}/cache"}
        env.update({k: v.replace("<TMP>", tmp) for k, v in extra_env.items()})
        proc = subprocess.run(["bash", str(here / "bootstrap.sh"), *workloads],
                              capture_output=True, text=True, env=env, check=False)
        leftovers = sorted(p.name for p in here.iterdir() if p.name != "bootstrap.sh")

    def scrub(text):
        return text.replace(str(here), "<HERE>").replace(tmp, "<TMP>")

    return {"rc": proc.returncode, "stdout": scrub(proc.stdout),
            "stderr": scrub(proc.stderr).splitlines(), "files_left_behind": leftovers}


@unittest.skipUnless(shutil.which("bash"), "bootstrap.sh needs bash")
class BootstrapPlanGolden(unittest.TestCase):
    def test_dry_run_plans(self):
        record = {
            "gaps": run_dry_bootstrap(["llama", "qwen3", "deepseek"], {
                "XVAL_CKPT_deepseek": "/ckpt/ds-0731-tp8-fp4",
                "XVAL_WEIGHTS_qwen3": "/weights/qwen3",
                "XVAL_HF_deepseek": "deepseek-ai/DeepSeek-V4-Flash",
            }),
            "complete": run_dry_bootstrap(["qwen3"], {
                "QW_BENCH_BIN": "<TMP>/vllm-python",
                "XVAL_CKPT_qwen3": "/ckpt/qwen3",
                "XVAL_WEIGHTS_qwen3": "/weights/qwen3",
            }),
        }
        self.assertEqual(record["gaps"]["rc"], 1)
        self.assertEqual(record["complete"]["rc"], 0)
        check_json(self, "bootstrap_plan", record)


if __name__ == "__main__":
    unittest.main()
