"""qbench run's sequence without a GPU: --dry-run on every run yaml prints
its Config and LaunchSpec and writes nothing, a matrix runs once per point,
the engine gets the caller's GPUs or the first idle tp * pp, and mode:
probe hands the probe its flags plus --out results/<run>/."""

import contextlib
import functools
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from quettabench import cli, profiler, runner
from quettabench.config import loader
from quettabench.config.schema import ConfigError
from quettabench.engines import launcher
from quettabench.engines.base import LaunchSpec
from quettabench.workloads.sessions import TrajectorySessionTrace
from tests.golden._harness import REPO
from tests.profiling import fake_endpoint


def setUpModule():
    """The loader reads os.environ when no env is given; run on an empty one,
    so a caller's NCCL_*, GPU_UTIL or CUDA_VISIBLE_DEVICES never moves a result."""
    patch = mock.patch.dict(os.environ, {}, clear=True)
    patch.start()
    unittest.addModuleCleanup(patch.stop)


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


class InProcessEnv(unittest.TestCase):
    def test_export_replaces_what_the_process_holds(self):
        with mock.patch.dict(os.environ, {"NCCL_PROTO": "LL"}, clear=True):
            names = runner.env_map.export({"NCCL_PROTO": "Simple", "NCCL_ALGO": ""})
            self.assertEqual((os.environ["NCCL_PROTO"], names), ("Simple", ["NCCL_PROTO"]))
            self.assertNotIn("NCCL_ALGO", os.environ)

    def test_a_server_point_after_an_in_process_one_drops_its_exports(self):
        load = functools.partial(loader.load, RUNS[0].with_name("llama.yaml"), env={})
        (grid,) = load(["engine.model=/w", "hardware.visible_devices=0",
                        "hardware.link_profile=pcie"])
        (served,) = load(["engine.model=/w", "hardware.visible_devices=1", "mode.grid=null"])
        exported = set()
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(runner.launcher, "start") as start, \
                mock.patch.object(runner.workloads, "build"), \
                mock.patch.object(runner, "RunWriter"), mock.patch.object(runner.serve, "run"):
            runner.run_one(grid, Path(td, "a"), exported)
            in_process = {k: os.environ[k] for k in exported}
            runner.run_one(served, Path(td, "b"), exported)
        spec = start.call_args.args[0]
        self.assertEqual(in_process["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(in_process["NCCL_PROTO"], "Simple")
        self.assertEqual(spec.env["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(set(spec.env_drop), exported - set(spec.env))
        self.assertIn("NCCL_PROTO", spec.env_drop)
        self.assertEqual(spec.child_argv(["x"])[:3], ["env", "-u", spec.env_drop[0]])

    def test_a_multi_point_in_process_matrix_names_its_gpus(self):
        run = {"model": "llama31_8b", "workload": {"kind": "grid"}, "mode": {"grid": "smoke"},
               "matrix": {"engine.gpu_util": [0.8, 0.9]}}
        with scratch_configs("sweep", run) as (path, td):
            with self.assertRaisesRegex(ConfigError, "set hardware.visible_devices"):
                runner.main(path, ["engine.model=/w"], results_dir=td, allow_unverified=True)
            with mock.patch.object(runner, "run_one") as run_one:
                runner.main(path, ["engine.model=/w", "hardware.visible_devices=0"],
                            results_dir=td, allow_unverified=True)
        self.assertEqual(run_one.call_count, 2)


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
        self.assertRegex(line, r"^docker = docker run -d --pull=never --init --name "
                               r"qbench-llama-tp1-docker-\d+-[0-9a-f]{6} "
                               r"--gpus '\"device=0\"' ")
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

    def test_dry_run_prints_no_secret(self):
        run = yaml.safe_load(RUNS[0].with_name("llama-tp1-docker.yaml").read_text())
        run["engine"]["extra"] = {"vllm": {"api_key": "sk-live"}}
        with scratch_configs("secret", run) as (path, _):
            out, exit_msg = qbench("run", str(path), "--dry-run", "--set", "engine.model=/w")
        self.assertIsNone(exit_msg)
        self.assertNotIn("sk-live", out)
        self.assertIn("engine.extra.vllm = {'api_key': 'REDACTED'}  (run)", out.splitlines())
        self.assertIn("-e VLLM_API_KEY=REDACTED", out)

    def test_docker_without_an_image_is_refused(self):
        _, exit_msg = qbench("run", str(RUNS[0].with_name("llama-tp1-docker.yaml")),
                             "--dry-run", "--set", "engine.model=/w", "--set", "engine.image=null")
        self.assertIn("hardware.launcher=docker needs engine.image", exit_msg)

    def test_docker_needs_a_launched_server(self):
        _, exit_msg = qbench("run", str(RUNS[0].with_name("llama-tp1.yaml")), "--dry-run",
                             "--set", "engine.model=/w", "--set", "hardware.launcher=docker")
        self.assertIn("needs a launched server", exit_msg)


def session_trace(td):
    """Two sessions of three turns, each turn's messages the whole conversation."""
    sessions = []
    for s in range(2):
        history, turns = [], []
        for t in range(3):
            history = [*history, {"role": "user", "content": f"s{s} t{t}"}]
            turns.append({"turn_idx": t, "messages": history, "osl_tokens": 3})
            history = [*history, {"role": "assistant", "content": f"reply {t}"}]
        sessions.append({"session_id": f"S{s}", "turns": turns})
    path = Path(td, "t.jsonl")
    path.write_text("".join(json.dumps(x) + "\n" for x in sessions))
    with contextlib.redirect_stdout(io.StringIO()):
        return TrajectorySessionTrace(str(path), min_turns=3, num_sessions=2)


class ChatOverHttp(unittest.TestCase):
    def test_sessions_on_vllm_go_over_chat_with_the_whole_conversation(self):
        (cfg,) = loader.load(RUNS[0].with_name("llama.yaml"),
                             ["engine.model=/w", "hardware.visible_devices=0", "mode.grid=null",
                              "mode.warmup=0", "mode.concurrency=2"], env={})
        with tempfile.TemporaryDirectory() as td, fake_endpoint.serve() as (endpoint, posts):
            trace = session_trace(td)
            handle = launcher.Handle("vllm", endpoint, 0, Path(td), mock.Mock())
            with mock.patch.object(runner.launcher, "start", return_value=handle), \
                    mock.patch.object(handle, "stop"), \
                    mock.patch.object(runner.workloads, "build", return_value=trace), \
                    contextlib.redirect_stdout(io.StringIO()):
                runner.run_one(cfg, Path(td, "run"))
            rows = [json.loads(x) for x in Path(td, "run", "requests.jsonl").read_text().splitlines()]
        self.assertEqual({path for path, _ in posts}, {"/v1/chat/completions"})
        sent = sorted(json.dumps(body["messages"]) for _, body in posts)
        self.assertEqual(sent, sorted(json.dumps(r.messages) for r in trace.requests()))
        self.assertEqual(max(len(body["messages"]) for _, body in posts), 5)
        self.assertEqual([r["success"] for r in rows], [True] * 6)


class FailureRecord(unittest.TestCase):
    """A run that raises anywhere after its Config resolved leaves run.json."""

    def setUp(self):
        """run_probe exports CUDA_VISIBLE_DEVICES into os.environ; undo it per test."""
        patch = mock.patch.dict(os.environ, {}, clear=True)
        patch.start()
        self.addCleanup(patch.stop)

    def run_raising(self, sets=("hardware.visible_devices=0",), **patches):
        (cfg,) = loader.load(RUNS[0].with_name("llama.yaml"),
                             ["engine.model=/w", "mode.grid=null", *sets], env={})
        with tempfile.TemporaryDirectory() as td, \
                mock.patch.object(runner.workloads, "build"), \
                mock.patch.object(runner.launcher, "start", **patches.get("start", {})), \
                mock.patch.object(runner.serve, "run", **patches.get("serve", {})):
            out = Path(td, "run")
            with self.assertRaises(BaseException) as raised:
                runner.run_one(cfg, out)
            doc = json.loads((out / "run.json").read_text())
            files = sorted(p.name for p in out.iterdir())
        return raised.exception, doc, files

    def test_a_failed_launch_is_recorded(self):
        exc, doc, files = self.run_raising(start={"side_effect": RuntimeError("no health")})
        self.assertIsInstance(exc, RuntimeError)
        self.assertEqual(doc["run_status"], {"status": "failed",
                                             "reason": "RuntimeError: no health"})
        self.assertEqual(files, ["requests.jsonl", "run.json"])

    def test_translate_gpu_pick_and_env_export_failures_are_recorded(self):
        targets = {"translate": (runner.base.get("vllm"), "translate", ()),
                   "idle_gpus": (runner.launcher, "idle_gpus", ("hardware.visible_devices=null",)),
                   "export": (runner.env_map, "export", ("mode.grid=llama_single",))}
        for name, (obj, attr, sets) in targets.items():
            with self.subTest(name), mock.patch.object(obj, attr, side_effect=OSError(name)):
                exc, doc, files = self.run_raising(sets=("hardware.visible_devices=0", *sets))
            self.assertIsInstance(exc, OSError)
            self.assertEqual(doc["run_status"], {"status": "failed", "reason": f"OSError: {name}"})
            self.assertEqual(files, ["requests.jsonl", "run.json"])

    def test_ctrl_c_mid_run_is_aborted_and_the_server_stops(self):
        handle = mock.Mock(gpu="H200", endpoint="http://127.0.0.1:1")
        exc, doc, _ = self.run_raising(start={"return_value": handle},
                                       serve={"side_effect": KeyboardInterrupt()})
        self.assertIsInstance(exc, KeyboardInterrupt)
        self.assertEqual(doc["run_status"]["status"], "aborted")
        self.assertEqual(doc["provenance"]["gpu"], "H200")
        handle.stop.assert_called_once_with()

    def test_a_probe_whose_gpu_pick_fails_is_recorded(self):
        with scratch_configs("p", {"mode": {"mode": "probe", "probe": "fake"}}) as (path, td), \
                mock.patch.dict(sys.modules, {"quettabench.probes.fake": mock.Mock()}), \
                mock.patch.object(runner.launcher, "idle_gpus", side_effect=RuntimeError("busy")):
            with self.assertRaises(RuntimeError):
                runner.main(path, results_dir=td / "results", allow_unverified=True)
            doc = json.loads((td / "results" / "p" / "run.json").read_text())
        self.assertEqual(doc["run_status"], {"status": "failed", "reason": "RuntimeError: busy"})

    def test_a_probe_that_raises_is_recorded(self):
        probe = mock.Mock(**{"main.side_effect": ValueError("bad shape")})
        with scratch_configs("p", {"mode": {"mode": "probe", "probe": "fake"}}) as (path, td), \
                mock.patch.dict(sys.modules, {"quettabench.probes.fake": probe}):
            with self.assertRaises(ValueError):
                runner.main(path, results_dir=td / "results", allow_unverified=True)
            doc = json.loads((td / "results" / "p" / "run.json").read_text())
        self.assertEqual(doc["run_status"], {"status": "failed", "reason": "ValueError: bad shape"})


class ProbeMode(unittest.TestCase):
    def setUp(self):
        """run_probe exports CUDA_VISIBLE_DEVICES into os.environ; undo it per test."""
        patch = mock.patch.dict(os.environ, {}, clear=True)
        patch.start()
        self.addCleanup(patch.stop)

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

    def test_probe_runs_on_the_runs_gpus(self):
        """CUDA_VISIBLE_DEVICES is exported before the probe module is imported,
        from hardware.visible_devices or else the idle GPUs, and recorded."""
        seen = []

        def import_probe(name):
            seen.append(os.environ.get("CUDA_VISIBLE_DEVICES"))
            return mock.Mock()

        for hardware, idle, gpus in (({"visible_devices": "2,3"}, None, "2,3"), ({}, "5", "5")):
            run = {"mode": {"mode": "probe", "probe": "fake"}, "hardware": hardware}
            with self.subTest(gpus=gpus), scratch_configs("p", run) as (path, td), \
                    mock.patch.dict(os.environ, {}, clear=True), \
                    mock.patch.dict(sys.modules, {"quettabench.probes.fake": mock.Mock()}), \
                    mock.patch.object(runner.importlib, "import_module", side_effect=import_probe), \
                    mock.patch.object(launcher, "idle_gpus", return_value=idle) as idle_gpus:
                runner.main(path, results_dir=td / "results", allow_unverified=True)
                doc = json.loads((td / "results" / "p" / "run.json").read_text())
            self.assertEqual(seen.pop(), gpus)
            self.assertEqual(doc["provenance"]["launch_env"], {"CUDA_VISIBLE_DEVICES": gpus})
            self.assertEqual(idle_gpus.called, idle is not None)

    def test_a_run_dir_with_a_run_is_refused_without_overwrite(self):
        run = {"mode": {"mode": "probe", "probe": "fake"}}
        old_files = ("run.json", "requests.jsonl", "engine.json", "server.log", "bench.log",
                     "run.json.tmp", "nsys.sqlite", "nsys-512x8.nsys-rep")
        with scratch_configs("p", run) as (path, td), \
                mock.patch.dict(sys.modules, {"quettabench.probes.fake": mock.Mock()}):
            out = td / "results" / "p"
            (out / "gemm").mkdir(parents=True)
            for name in old_files:
                (out / name).write_text("old\n")
            (out / "gemm" / "table.csv").write_text("old\n")
            (out / "profile.json").write_text(json.dumps({"metrics": {"old": 1}}))
            argv = ("run", str(path), "--allow-unverified", "--results-dir", str(td / "results"))
            _, refused = qbench(*argv)
            self.assertEqual((out / "requests.jsonl").read_text(), "old\n")
            _, exit_msg = qbench(*argv, "--overwrite")
            left = sorted(p.name for p in out.iterdir())
            rows = (out / "requests.jsonl").read_text()
            doc = json.loads((out / "run.json").read_text())
            prof = profiler.profile(out, ["cost"], gpu_cost_hr=1.0)
        self.assertIn("already holds a run; pass --overwrite", refused)
        self.assertIsNone(exit_msg)
        self.assertEqual(left, ["requests.jsonl", "run.json"])
        self.assertEqual(rows, "")
        self.assertEqual(doc["summary"]["probe"], "fake")
        self.assertNotIn("metrics", prof)
        self.assertEqual(prof["sources_run"], ["cost"])

    def test_a_name_that_is_no_probe_is_refused(self):
        for name in ("nope", "base", "_graph", "../cli"):
            with self.subTest(name), scratch_configs("p", {"mode": {"mode": "probe", "probe": name}}) \
                    as (path, td):
                _, exit_msg = qbench("run", str(path), "--allow-unverified",
                                     "--results-dir", str(td / "results"))
                self.assertFalse((td / "results").exists())
            self.assertIn("is not a module under quettabench/probes/ with a main", exit_msg)
        self.assertTrue(runner.is_probe("gemm"))

    def test_probe_mode_needs_a_probe(self):
        with scratch_configs("bare", {"mode": {"mode": "probe"}}) as (path, _):
            _, exit_msg = qbench("run", str(path), "--dry-run", "--allow-unverified")
        self.assertIn("mode=probe needs mode.probe", exit_msg)


if __name__ == "__main__":
    unittest.main()
