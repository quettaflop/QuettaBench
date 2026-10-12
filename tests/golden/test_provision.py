"""Golden: the provision plan text for the run yamls."""

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
        argv = [
            *(str(RUNS / f"{n}.yaml") for n in names),
            "--dry-run",
            "--env-file",
            f"{tmp}/qbench.env",
            "--set",
            f"hardware.provision.cache_root={tmp}/cache",
            "--set",
            f"hardware.provision.py={fake_py}",
        ]
        argv += [a for item in scoped for a in ("--set", item.replace("<TMP>", tmp))]
        out, err = io.StringIO(), io.StringIO()
        with (
            mock.patch.object(provision, "compute_capability", return_value=None),
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
        ):
            rc = provision.main(argv)
        leftovers = sorted(p.name for p in Path(tmp).iterdir() if p != fake_py)
    return {
        "rc": rc,
        "stdout": out.getvalue(),
        "stderr": err.getvalue().replace(tmp, "<TMP>").splitlines(),
        "files_left_behind": leftovers,
    }


class ProvisionPlanGolden(unittest.TestCase):
    def test_dry_run_plans(self):
        record = {
            "gaps": run_dry_provision(
                ["llama", "qwen3", "deepseek"],
                [
                    "deepseek:hardware.provision.quettaserve_checkpoint=/ckpt/ds-0731-tp8-fp4",
                    "qwen3:hardware.provision.weights=/weights/qwen3",
                    "deepseek:hardware.provision.hf_id=deepseek-ai/DeepSeek-V4-Flash",
                ],
            ),
            "complete": run_dry_provision(
                ["qwen3"],
                [
                    "qwen3:hardware.provision.bench_bin=<TMP>/vllm-python",
                    "qwen3:hardware.provision.quettaserve_checkpoint=/ckpt/qwen3",
                    "qwen3:hardware.provision.weights=/weights/qwen3",
                ],
            ),
        }
        self.assertEqual(record["gaps"]["rc"], 1)
        self.assertEqual(record["complete"]["rc"], 0)
        check_json(self, "bootstrap_plan", record)

    def test_a_model_without_quettaserve_provisions_the_vllm_side_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_py = Path(tmp, "vllm-python")
            fake_py.write_text("#!/bin/sh\nexit 0\n")
            fake_py.chmod(0o755)
            out, err = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(provision, "compute_capability", return_value=None),
                contextlib.redirect_stdout(out),
                contextlib.redirect_stderr(err),
            ):
                rc = provision.main(
                    [
                        str(RUNS / "llama.yaml"),
                        "--env-file",
                        f"{tmp}/qbench.env",
                        "--set",
                        f"hardware.provision.cache_root={tmp}/cache",
                        "--set",
                        f"hardware.provision.py={fake_py}",
                        "--set",
                        "llama:hardware.provision.weights=/weights/llama",
                    ]
                )
        self.assertEqual(rc, 0, err.getvalue())
        self.assertNotIn("MISSING", err.getvalue())
        self.assertEqual(
            out.getvalue().splitlines(), ["qbench run configs/runs/llama.yaml --set engine.model=/weights/llama"]
        )

    def test_a_refused_run_yaml_is_missing(self):
        record = run_dry_provision(["qwen"], [])
        self.assertEqual(record["rc"], 1)
        self.assertIn("attention=hybrid needs kv_bytes", record["stderr"][0])
        self.assertTrue(record["stderr"][0].startswith("[provision] MISSING: "))

    def test_a_matrix_run_yaml_is_missing(self):
        with mock.patch.object(provision.loader, "load", return_value=[mock.Mock()] * 2):
            record = run_dry_provision(["llama"], [])
        self.assertEqual(record["rc"], 1)
        self.assertIn("a matrix of 2 points", record["stderr"][0])


class DockerCheck(unittest.TestCase):
    def provision(self, run, *extra, docker="/usr/bin/docker", inspect_ok=False):
        calls = []

        def succeeds(argv):
            calls.append(argv)
            return inspect_ok if argv[:3] == ["docker", "image", "inspect"] else True

        err = io.StringIO()
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(provision.shutil, "which", return_value=docker),
            mock.patch.object(provision, "succeeds", side_effect=succeeds),
            mock.patch.object(provision.subprocess, "run") as run_cmd,
            mock.patch.object(provision, "compute_capability", return_value=None),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(err),
        ):
            provision.main(
                [
                    str(RUNS / f"{run}.yaml"),
                    "--env-file",
                    f"{tmp}/qbench.env",
                    "--set",
                    f"hardware.provision.cache_root={tmp}/cache",
                    "--set",
                    "hardware.backend=cuda",
                    "--set",
                    "hardware.link_profile=nvlink",
                    *extra,
                ]
            )
        self.assertFalse(any("pull" in c.args[0] for c in run_cmd.call_args_list))
        return err.getvalue().splitlines(), calls

    def test_missing_image_prints_the_pull_command(self):
        lines, calls = self.provision("llama-tp1-docker")
        self.assertIn(
            "[provision] MISSING: docker image vllm/vllm-openai:v0.26.0; "
            "pull it with: docker pull vllm/vllm-openai:v0.26.0",
            lines,
        )
        self.assertIn(["docker", "image", "inspect", "vllm/vllm-openai:v0.26.0"], calls)
        self.assertEqual([c for c in calls if "pip" in c], [])
        self.assertIn("[provision] vllm: in the docker image", lines)

    def test_present_image_and_no_docker(self):
        lines, _ = self.provision("llama-tp1-docker", inspect_ok=True)
        self.assertIn("[provision] llama-tp1-docker: docker image ok: vllm/vllm-openai:v0.26.0", lines)
        lines, calls = self.provision("llama-tp1-docker", docker=None)
        self.assertIn(
            "[provision] MISSING: docker is not on PATH; hardware.launcher=docker runs the server with it", lines
        )
        self.assertEqual([c for c in calls if c[0] == "docker"], [])

    def test_dry_run_plans_only_and_backend_flag_sets_the_launcher(self):
        lines, calls = self.provision("llama-tp1-docker", "--dry-run")
        self.assertIn("[provision] would: docker image inspect vllm/vllm-openai:v0.26.0", lines)
        self.assertEqual([c for c in calls if c[0] == "docker"], [])
        lines, _ = self.provision("llama-tp1", "--backend", "docker", "--dry-run")
        self.assertIn("[provision] MISSING: llama-tp1: hardware.launcher=docker needs engine.image", lines)


if __name__ == "__main__":
    unittest.main()
