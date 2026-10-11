"""qbench run's sequence without a GPU: --dry-run on every run yaml prints
its Config and LaunchSpec and writes nothing, a matrix runs once per point,
the engine gets the caller's GPUs or the first idle tp * pp, and mode:
probe hands the probe its flags plus --out results/<run>/."""

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from quettabench import cli, runner
from quettabench.config import loader
from quettabench.config.schema import ConfigError
from quettabench.engines.base import LaunchSpec
from tests.golden._harness import REPO

RUNS = sorted((REPO / "configs" / "runs").glob("*.yaml"))
# Hybrid attention without kv_bytes: the loader refuses on purpose.
REFUSED = {"qwen": "attention=hybrid needs kv_bytes"}


def qbench(*argv):
    """(stdout, SystemExit message or None) of cli.main(argv)."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out), mock.patch("shutil.which", return_value=None):
        try:
            cli.main(list(argv))
        except SystemExit as e:
            return out.getvalue(), str(e.code)
    return out.getvalue(), None


@contextlib.contextmanager
def scratch_configs(name, run):
    """configs/ copied to a scratch dir with one more run yaml."""
    with tempfile.TemporaryDirectory() as td:
        configs = Path(td, "configs")
        shutil.copytree(REPO / "configs", configs)
        path = configs / "runs" / f"{name}.yaml"
        path.write_text(yaml.safe_dump(run))
        yield path, Path(td)


class DryRun(unittest.TestCase):
    def test_every_run_yaml(self):
        for path in RUNS:
            with self.subTest(path.stem), tempfile.TemporaryDirectory() as td:
                out, exit_msg = qbench("run", str(path), "--dry-run", "--allow-unverified",
                                       "--set", "engine.model=/w", "--results-dir", td)
                self.assertEqual(list(Path(td).iterdir()), [])
                if path.stem in REFUSED:
                    self.assertIn(REFUSED[path.stem], exit_msg)
                    continue
                self.assertIsNone(exit_msg)
                lines = out.splitlines()
                self.assertEqual(lines[0], f"== {path.stem}")
                self.assertIn("engine.model = '/w'  (cli)", lines)
                self.assertIn("spec.adapter = 'vllm'", lines)

    def test_matrix_is_one_run_per_point(self):
        run = {"model": "llama31_8b", "workload": {"kind": "grid"}, "mode": {"grid": "smoke"},
               "matrix": {"engine.tp": [1, 2]}}
        with scratch_configs("sweep", run) as (path, _):
            out, exit_msg = qbench("run", str(path), "--dry-run", "--set", "engine.model=/w")
        self.assertIsNone(exit_msg)
        self.assertEqual([line for line in out.splitlines() if line.startswith("==")],
                         ["== sweep-tp1", "== sweep-tp2"])
        self.assertIn("launch = none, the driver runs it", out)  # grid runs launch nothing

    def test_refusal_is_the_exit_message(self):
        _, exit_msg = qbench("run", str(RUNS[0].with_name("llama.yaml")), "--dry-run",
                             "--set", "mode.grid=nope")
        self.assertIn("unknown grid 'nope'", exit_msg)


class GpuPick(unittest.TestCase):
    def child_env(self, *sets):
        """The env run_one gives the child, with the engine stubbed out."""
        (cfg,) = loader.load(RUNS[0].with_name("llama-tp2.yaml"),
                             ["engine.model=/w", "hardware.backend=cuda", *sets], env={})
        adapter = mock.Mock(**{"needs_server.return_value": False,
                               "translate.return_value": LaunchSpec(adapter="x",
                                                                    bench_argv=("b",))})
        with mock.patch.object(runner.base, "get", return_value=adapter), \
                mock.patch.object(runner.launcher, "idle_gpus", return_value="3,5") as idle, \
                mock.patch.object(runner.env_map, "export") as export, \
                mock.patch.object(runner.workloads, "build"), \
                mock.patch.object(runner, "RunWriter"), mock.patch.object(runner.serve, "run"), \
                tempfile.TemporaryDirectory() as td:
            runner.run_one(cfg, Path(td))
        export.assert_not_called()  # a bench child gets its env, this process keeps its own
        return adapter.driver.call_args.args[0].env["CUDA_VISIBLE_DEVICES"], idle.call_args

    def test_idle_gpus_for_tp_times_pp(self):
        gpus, idle = self.child_env("engine.pp=2")
        self.assertEqual((gpus, idle.args), ("3,5", (4, "cuda")))

    def test_caller_devices_win(self):
        gpus, idle = self.child_env("hardware.visible_devices=6,7")
        self.assertEqual((gpus, idle), ("6,7", None))


class MatrixEnv(unittest.TestCase):
    def test_each_point_gets_its_own_gpus_and_pins(self):
        run = {"model": "llama31_8b", "mode": {"grid": None},
               "hardware": {"backend": "cuda", "link_profile": "pcie"},
               "matrix": {"engine.tp": [1, 2]}}
        idle = {1: "4", 2: "6,7"}
        with scratch_configs("sweep", run) as (path, td), \
                mock.patch.object(runner.launcher, "idle_gpus",
                                  side_effect=lambda n, backend: idle[n]), \
                mock.patch.object(runner.launcher, "start") as start, \
                mock.patch.object(runner.env_map, "export") as export, \
                mock.patch.object(runner.workloads, "build"), \
                mock.patch.object(runner, "RunWriter"), mock.patch.object(runner.serve, "run"):
            runner.main(path, ["engine.model=/w"], results_dir=td, allow_unverified=True)
        envs = [c.args[0].env for c in start.call_args_list]
        self.assertEqual([e["CUDA_VISIBLE_DEVICES"] for e in envs], ["4", "6,7"])
        self.assertEqual([e["NCCL_PROTO"] for e in envs], ["Simple", "Simple"])
        export.assert_not_called()

    def test_in_process_points_that_differ_are_refused(self):
        run = {"model": "llama31_8b", "workload": {"kind": "grid"}, "mode": {"grid": "smoke"},
               "matrix": {"engine.tp": [1, 2]}}
        with scratch_configs("sweep", run) as (path, td), \
                self.assertRaisesRegex(ConfigError, "one process cannot change them"):
            runner.main(path, ["engine.model=/w"], results_dir=td, allow_unverified=True)


class ReadyTimeout(unittest.TestCase):
    def test_hardware_ready_timeout_reaches_the_launcher(self):
        (cfg,) = loader.load(RUNS[0].with_name("llama.yaml"),
                             ["engine.model=/w", "hardware.visible_devices=0",
                              "mode.grid=null", "hardware.ready_timeout_s=10800"], env={})
        with mock.patch.object(runner.launcher, "start") as start, \
                mock.patch.object(runner.env_map, "export"), \
                mock.patch.object(runner.workloads, "build"), \
                mock.patch.object(runner, "RunWriter"), mock.patch.object(runner.serve, "run"), \
                tempfile.TemporaryDirectory() as td:
            runner.run_one(cfg, Path(td))
        self.assertEqual(start.call_args.kwargs["timeout_s"], 10800)
        self.assertEqual(start.call_args.kwargs["backend"], "local")

    def test_trace_is_built_before_the_server_launches(self):
        (cfg,) = loader.load(RUNS[0].with_name("llama.yaml"),
                             ["engine.model=/w", "hardware.visible_devices=0",
                              "mode.grid=null"], env={})
        with mock.patch.object(runner.launcher, "start") as start, \
                mock.patch.object(runner.workloads, "build", side_effect=ValueError("bad")), \
                tempfile.TemporaryDirectory() as td, self.assertRaisesRegex(ValueError, "bad"):
            runner.run_one(cfg, Path(td))
        start.assert_not_called()

    def test_dry_run_shows_it(self):
        out, _ = qbench("run", str(RUNS[0].with_name("llama.yaml")), "--dry-run",
                        "--set", "engine.model=/w", "--set", "hardware.ready_timeout_s=3600")
        self.assertIn("hardware.ready_timeout_s = 3600  (cli)", out.splitlines())


class DockerDryRun(unittest.TestCase):
    def test_dry_run_prints_the_docker_command(self):
        out, exit_msg = qbench("run", str(RUNS[0].with_name("llama-tp1-docker.yaml")),
                               "--dry-run", "--set", "engine.model=/w",
                               "--set", "hardware.hf_home=/hf")
        self.assertIsNone(exit_msg)
        self.assertIn("hardware.launcher = 'docker'  (run)", out.splitlines())
        (line,) = [x for x in out.splitlines() if x.startswith("docker = ")]
        self.assertTrue(line.startswith("docker = docker run -d --pull=never --init --name "
                                        "qbench-llama-tp1-docker --gpus '\"device=0\"' "))
        self.assertTrue(line.endswith("vllm/vllm-openai:v0.30.0 serve /w --port 8000 "
                                      "--tensor-parallel-size 1 --dtype float16 --max-model-len "
                                      "9216 --gpu-memory-utilization 0.9 "
                                      "--no-enable-prefix-caching --enable-chunked-prefill "
                                      "--trust-remote-code"))

    def test_run_hands_the_launcher_docker_and_the_gpus(self):
        (cfg,) = loader.load(RUNS[0].with_name("llama-tp1-docker.yaml"),
                             ["engine.model=/w", "hardware.visible_devices=3,5"], env={})
        with mock.patch.object(runner.launcher, "start") as start, \
                mock.patch.object(runner.env_map, "export"), \
                mock.patch.object(runner.workloads, "build"), \
                mock.patch.object(runner, "RunWriter"), mock.patch.object(runner.serve, "run"), \
                tempfile.TemporaryDirectory() as td:
            runner.run_one(cfg, Path(td))
        kw = start.call_args.kwargs
        self.assertEqual((kw["backend"], kw["cfg"], kw["gpus"]), ("docker", cfg, "3,5"))

    def test_docker_without_an_image_is_refused(self):
        _, exit_msg = qbench("run", str(RUNS[0].with_name("llama-tp1-docker.yaml")),
                             "--dry-run", "--set", "engine.model=/w", "--set", "engine.image=null")
        self.assertIn("hardware.launcher=docker needs engine.image", exit_msg)

    def test_docker_needs_a_launched_server(self):
        _, exit_msg = qbench("run", str(RUNS[0].with_name("llama-tp1.yaml")), "--dry-run",
                             "--set", "engine.model=/w", "--set", "hardware.launcher=docker")
        self.assertIn("needs a launched server", exit_msg)


class ProbeMode(unittest.TestCase):
    def test_probe_gets_its_flags_and_the_run_dir(self):
        run = {"mode": {"mode": "probe", "probe": "fake", "probe_args": "--gpu-label A100 --wide"}}
        probe = mock.Mock()
        with scratch_configs("gemm-a100", run) as (path, td), \
                mock.patch.dict(sys.modules, {"quettabench.probes.fake": probe}):
            _, exit_msg = qbench("run", str(path), "--allow-unverified",
                                 "--results-dir", str(td / "results"))
            doc = json.loads((td / "results" / "gemm-a100" / "run.json").read_text())
        self.assertIsNone(exit_msg)
        argv = ["--gpu-label", "A100", "--wide", "--out", str(td / "results" / "gemm-a100")]
        probe.main.assert_called_once_with(argv)
        self.assertEqual(doc["summary"], {"probe": "fake", "argv": argv})

    def test_probe_mode_needs_a_probe(self):
        with scratch_configs("bare", {"mode": {"mode": "probe"}}) as (path, _):
            _, exit_msg = qbench("run", str(path), "--dry-run", "--allow-unverified")
        self.assertIn("mode=probe needs mode.probe", exit_msg)


if __name__ == "__main__":
    unittest.main()
