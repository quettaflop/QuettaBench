"""qbench run without a GPU: dry runs, sweeps, probe runs, run dir rules."""

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
from quettabench.engines import drivers, launcher
from quettabench.engines.base import LaunchSpec
from quettabench.workloads.sessions import TrajectorySessionTrace
from tests.contracts import helpers
from tests.golden._harness import REPO


def setUpModule():
    """An empty env so a caller's NCCL_*, GPU_UTIL or CUDA_VISIBLE_DEVICES never moves a result."""
    patch = mock.patch.dict(os.environ, {}, clear=True)
    patch.start()
    unittest.addModuleCleanup(patch.stop)


RUNS = sorted((REPO / "configs" / "runs").glob("*.yaml"))
REFUSED = {"qwen": "attention=hybrid needs kv_bytes_per_token"}


def qbench(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), mock.patch("shutil.which", return_value=None):
        try:
            cli.main(list(argv))
        except SystemExit as e:
            return out.getvalue(), str(e.code)
    return out.getvalue(), None


@contextlib.contextmanager
def scratch_configs(name, run):
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
                out, exit_msg = qbench(
                    "run", str(path), "--dry-run", "--allow-unverified", "--set", "engine.model=/w", "--results-dir", td
                )
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
        run = {
            "model": "llama31_8b",
            "workload": {"kind": "grid"},
            "mode": {"grid": "smoke"},
            "matrix": {"engine.tp": [1, 2]},
        }
        with scratch_configs("sweep", run) as (path, _):
            out, exit_msg = qbench("run", str(path), "--dry-run", "--set", "engine.model=/w")
        self.assertIsNone(exit_msg)
        self.assertEqual([line for line in out.splitlines() if line.startswith("==")], ["== sweep-tp1", "== sweep-tp2"])
        self.assertIn("launch = server", out)

    def test_refusal_is_the_exit_message(self):
        _, exit_msg = qbench("run", str(RUNS[0].with_name("llama.yaml")), "--dry-run", "--set", "mode.grid=nope")
        self.assertIn("unknown grid 'nope'", exit_msg)


class GpuPick(unittest.TestCase):
    def child_env(self, *sets):
        (cfg,) = loader.load(
            RUNS[0].with_name("llama-tp2.yaml"), ["engine.model=/w", "hardware.backend=cuda", *sets], env={}
        )
        adapter = mock.Mock(
            **{"needs_server.return_value": False, "translate.return_value": LaunchSpec(adapter="x", bench_argv=("b",))}
        )
        with (
            mock.patch.object(runner.base, "get", return_value=adapter),
            mock.patch.object(runner.launcher, "idle_gpus", return_value="3,5") as idle,
            mock.patch.object(runner.env_map, "export") as export,
            mock.patch.object(runner, "build_trace"),
            mock.patch.object(runner, "RunWriter"),
            mock.patch.object(runner.serve, "run"),
            tempfile.TemporaryDirectory() as td,
        ):
            runner.run_one(cfg, Path(td))
        export.assert_not_called()
        return adapter.driver.call_args.args[0].env["CUDA_VISIBLE_DEVICES"], idle.call_args

    def test_idle_gpus_for_tp_times_pp(self):
        gpus, idle = self.child_env("engine.pp=2")
        self.assertEqual((gpus, idle.args), ("3,5", (4, "cuda")))

    def test_caller_devices_win(self):
        gpus, idle = self.child_env("hardware.visible_devices=6,7")
        self.assertEqual((gpus, idle), ("6,7", None))


class ReuseServer(unittest.TestCase):
    def run_matrix(self, matrix, reset=True, reuse=True):
        run = {
            "model": "llama31_8b",
            "workload": {"source": "chat-singleturn"},
            "mode": {"reset_prefix_cache": reset},
            "hardware": {"visible_devices": "0"},
            "matrix": matrix,
        }

        def start(spec, out, **kw):
            Path(out).mkdir(parents=True, exist_ok=True)
            (Path(out) / "engine.json").write_text("{}")
            return launcher.Handle("vllm", "http://127.0.0.1:1", 1, Path(out), mock.Mock())

        with (
            scratch_configs("sweep", run) as (path, td),
            mock.patch.object(runner.launcher, "start", side_effect=start) as started,
            mock.patch.object(launcher.Handle, "stop") as stop,
            mock.patch.object(runner, "build_trace"),
            mock.patch.object(runner, "RunWriter") as writer,
            mock.patch.object(runner.serve, "run") as served,
        ):
            runner.main(path, ["engine.model=/w"], results_dir=td, allow_unverified=True, reuse_server=reuse)
            copied = sorted(p.parent.name for p in Path(td).glob("*/engine.json"))
        return started, stop, served, writer, copied

    def test_points_that_differ_only_in_mode_share_one_server(self):
        started, stop, served, writer, copied = self.run_matrix({"mode.concurrency": [1, 4, 16]})
        self.assertEqual((started.call_count, served.call_count, stop.call_count), (1, 3, 1))
        self.assertEqual(copied, ["sweep-concurrency1", "sweep-concurrency16", "sweep-concurrency4"])
        self.assertIn(
            mock.call("server_started_by", "sweep-concurrency1"),
            writer.return_value.provenance.__setitem__.call_args_list,
        )

    def test_each_engine_config_gets_its_own_server(self):
        started, stop, served, _, _ = self.run_matrix({"engine.gpu_util": [0.8, 0.9], "mode.concurrency": [1, 4]})
        self.assertEqual((started.call_count, served.call_count, stop.call_count), (2, 4, 2))

    def test_without_the_flag_every_point_starts_a_server(self):
        started, stop, _, _, _ = self.run_matrix({"mode.concurrency": [1, 4]}, reuse=False)
        self.assertEqual((started.call_count, stop.call_count), (2, 2))

    def test_reuse_needs_a_prefix_cache_reset(self):
        with self.assertRaisesRegex(ConfigError, "set mode.reset_prefix_cache=true"):
            self.run_matrix({"mode.concurrency": [1, 4]}, reset=False)


class MatrixEnv(unittest.TestCase):
    def test_each_point_gets_its_own_gpus_and_pins(self):
        run = {
            "model": "llama31_8b",
            "mode": {"grid": None},
            "hardware": {"backend": "cuda", "link_profile": "pcie"},
            "matrix": {"engine.tp": [1, 2]},
        }
        idle = {1: "4", 2: "6,7"}
        with (
            scratch_configs("sweep", run) as (path, td),
            mock.patch.object(runner.launcher, "idle_gpus", side_effect=lambda n, backend: idle[n]),
            mock.patch.object(runner.launcher, "start") as start,
            mock.patch.object(runner.env_map, "export") as export,
            mock.patch.object(runner, "build_trace"),
            mock.patch.object(runner, "RunWriter"),
            mock.patch.object(runner.serve, "run"),
        ):
            runner.main(path, ["engine.model=/w"], results_dir=td, allow_unverified=True)
        envs = [c.args[0].env for c in start.call_args_list]
        self.assertEqual([e["CUDA_VISIBLE_DEVICES"] for e in envs], ["4", "6,7"])
        self.assertEqual([e["NCCL_PROTO"] for e in envs], ["Simple", "Simple"])
        export.assert_not_called()

    def test_in_process_points_that_differ_are_refused(self):
        run = {
            "model": "llama31_8b",
            "workload": {"kind": "grid"},
            "mode": {"grid": "smoke", "return_outputs": "text"},
            "matrix": {"engine.tp": [1, 2]},
        }
        with (
            scratch_configs("sweep", run) as (path, td),
            self.assertRaisesRegex(ConfigError, "one process cannot change them"),
        ):
            runner.main(path, ["engine.model=/w"], results_dir=td, allow_unverified=True)


class InProcessEnv(unittest.TestCase):
    def test_export_replaces_what_the_process_holds(self):
        with mock.patch.dict(os.environ, {"NCCL_PROTO": "LL"}, clear=True):
            names = runner.env_map.export({"NCCL_PROTO": "Simple", "NCCL_ALGO": ""})
            self.assertEqual((os.environ["NCCL_PROTO"], names), ("Simple", ["NCCL_PROTO"]))
            self.assertNotIn("NCCL_ALGO", os.environ)

    def test_a_server_point_after_an_in_process_one_drops_its_exports(self):
        load = functools.partial(loader.load, RUNS[0].with_name("llama.yaml"), env={})
        (grid,) = load(
            ["engine.model=/w", "hardware.visible_devices=0", "hardware.link_profile=pcie", "mode.return_outputs=text"]
        )
        (served,) = load(["engine.model=/w", "hardware.visible_devices=1", "mode.grid=null"])
        exported = set()
        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(runner.launcher, "start") as start,
            mock.patch.object(runner, "build_trace"),
            mock.patch.object(runner, "RunWriter"),
            mock.patch.object(runner.serve, "run"),
        ):
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
        run = {
            "model": "llama31_8b",
            "workload": {"kind": "grid"},
            "mode": {"grid": "smoke", "return_outputs": "text"},
            "matrix": {"engine.gpu_util": [0.8, 0.9]},
        }
        with scratch_configs("sweep", run) as (path, td):
            with self.assertRaisesRegex(ConfigError, "set hardware.visible_devices"):
                runner.main(path, ["engine.model=/w"], results_dir=td, allow_unverified=True)
            with mock.patch.object(runner, "run_one") as run_one:
                runner.main(
                    path, ["engine.model=/w", "hardware.visible_devices=0"], results_dir=td, allow_unverified=True
                )
        self.assertEqual(run_one.call_count, 2)

    def test_probe_points_count_as_in_process(self):
        probe = {"mode": "probe", "probe": "gemm", "probe_args": "--gpu-label A100"}
        for matrix, sets, refused in (
            ({"mode.probe_args": ["--gpu-label A", "--gpu-label B"]}, [], "set hardware.visible_devices"),
            ({"mode.probe_args": ["--gpu-label A", "--gpu-label B"]}, ["hardware.visible_devices=0"], None),
            ({"hardware.visible_devices": ["0", "1"]}, [], "one process cannot change"),
        ):
            with (
                self.subTest(sets=sets, matrix=matrix),
                scratch_configs("p", {"mode": probe, "matrix": matrix}) as (path, td),
                mock.patch.object(runner, "run_one") as run_one,
            ):
                if refused:
                    with self.assertRaisesRegex(ConfigError, refused):
                        runner.main(path, sets, results_dir=td, allow_unverified=True)
                else:
                    runner.main(path, sets, results_dir=td, allow_unverified=True)
                self.assertEqual(run_one.call_count, 0 if refused else 2)
        (grid,) = loader.load(
            RUNS[0].with_name("llama.yaml"),
            ["engine.model=/w", "hardware.visible_devices=0", "mode.return_outputs=text"],
            env={},
        )
        (gemm,) = loader.load(
            RUNS[0].with_name("llama.yaml"),
            ["hardware.visible_devices=1", "mode.mode=probe", "mode.probe=gemm", "mode.grid=null"],
            env={},
        )
        with self.assertRaisesRegex(ConfigError, "one process cannot change"):
            runner.refuse_mixed_in_process([grid, gemm])


class ReadyTimeout(unittest.TestCase):
    def test_hardware_ready_timeout_reaches_the_launcher(self):
        (cfg,) = loader.load(
            RUNS[0].with_name("llama.yaml"),
            ["engine.model=/w", "hardware.visible_devices=0", "mode.grid=null", "hardware.ready_timeout_s=10800"],
            env={},
        )
        with (
            mock.patch.object(runner.launcher, "start") as start,
            mock.patch.object(runner.env_map, "export"),
            mock.patch.object(runner, "build_trace"),
            mock.patch.object(runner, "RunWriter"),
            mock.patch.object(runner.serve, "run"),
            tempfile.TemporaryDirectory() as td,
        ):
            runner.run_one(cfg, Path(td))
        self.assertEqual(start.call_args.kwargs["timeout_s"], 10800)
        self.assertEqual(start.call_args.kwargs["backend"], "local")

    def test_trace_is_built_before_the_server_launches(self):
        (cfg,) = loader.load(
            RUNS[0].with_name("llama.yaml"), ["engine.model=/w", "hardware.visible_devices=0", "mode.grid=null"], env={}
        )
        with (
            mock.patch.object(runner.launcher, "start") as start,
            mock.patch.object(runner, "build_trace", side_effect=ValueError("bad")),
            tempfile.TemporaryDirectory() as td,
            self.assertRaisesRegex(ValueError, "bad"),
        ):
            runner.run_one(cfg, Path(td))
        start.assert_not_called()

    def test_dry_run_shows_it(self):
        out, _ = qbench(
            "run",
            str(RUNS[0].with_name("llama.yaml")),
            "--dry-run",
            "--set",
            "engine.model=/w",
            "--set",
            "hardware.ready_timeout_s=3600",
        )
        self.assertIn("hardware.ready_timeout_s = 3600  (cli)", out.splitlines())


class DockerDryRun(unittest.TestCase):
    def test_dry_run_prints_the_docker_command(self):
        out, exit_msg = qbench(
            "run",
            str(RUNS[0].with_name("llama-tp1-docker.yaml")),
            "--dry-run",
            "--set",
            "engine.model=/w",
            "--set",
            "hardware.hf_home=/hf",
        )
        self.assertIsNone(exit_msg)
        self.assertIn("hardware.launcher = 'docker'  (run)", out.splitlines())
        (line,) = [x for x in out.splitlines() if x.startswith("docker = ")]
        self.assertRegex(
            line,
            r"^docker = docker run -d --pull=never --init --name "
            r"qbench-llama-tp1-docker-\d+-[0-9a-f]{6} "
            r"--gpus '\"device=0\"' ",
        )
        self.assertTrue(
            line.endswith(
                "vllm/vllm-openai:v0.26.0 serve /w --port 8000 "
                "--tensor-parallel-size 1 --dtype float16 --max-model-len "
                "9216 --gpu-memory-utilization 0.9 "
                "--no-enable-prefix-caching --enable-chunked-prefill "
                "--trust-remote-code"
            )
        )

    def test_run_hands_the_launcher_docker_and_the_gpus(self):
        (cfg,) = loader.load(
            RUNS[0].with_name("llama-tp1-docker.yaml"), ["engine.model=/w", "hardware.visible_devices=3,5"], env={}
        )
        with (
            mock.patch.object(runner.launcher, "start") as start,
            mock.patch.object(runner.env_map, "export"),
            mock.patch.object(runner, "build_trace"),
            mock.patch.object(runner, "RunWriter"),
            mock.patch.object(runner.serve, "run"),
            tempfile.TemporaryDirectory() as td,
        ):
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
        docker = next(ln for ln in out.splitlines() if ln.startswith("docker = ")).split()
        self.assertEqual(docker[docker.index("VLLM_API_KEY") - 1], "-e")

    def test_docker_without_an_image_is_refused(self):
        _, exit_msg = qbench(
            "run",
            str(RUNS[0].with_name("llama-tp1-docker.yaml")),
            "--dry-run",
            "--set",
            "engine.model=/w",
            "--set",
            "engine.image=null",
        )
        self.assertIn("hardware.launcher=docker needs engine.image", exit_msg)

    def test_docker_needs_a_launched_server(self):
        _, exit_msg = qbench(
            "run",
            str(RUNS[0].with_name("llama-tp1.yaml")),
            "--dry-run",
            "--set",
            "engine.model=/w",
            "--set",
            "hardware.launcher=docker",
            "--set",
            "mode.return_outputs=text",
        )
        self.assertIn("needs a launched server", exit_msg)


def session_trace(td):
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
        (cfg,) = loader.load(
            RUNS[0].with_name("llama.yaml"),
            ["engine.model=/w", "hardware.visible_devices=0", "mode.grid=null", "mode.warmup=0", "mode.concurrency=2"],
            env={},
        )
        with tempfile.TemporaryDirectory() as td, helpers.serve() as (endpoint, posts):
            trace = session_trace(td)
            handle = launcher.Handle("vllm", endpoint, 0, Path(td), mock.Mock())
            with (
                mock.patch.object(runner.launcher, "start", return_value=handle),
                mock.patch.object(handle, "stop"),
                mock.patch.object(runner, "build_trace", return_value=trace),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                runner.run_one(cfg, Path(td, "run"))
            rows = [json.loads(x) for x in Path(td, "run", "requests.jsonl").read_text().splitlines()]
        self.assertEqual({path for path, _ in posts}, {"/v1/chat/completions"})
        sent = sorted(json.dumps(body["messages"]) for _, body in posts)
        self.assertEqual(sent, sorted(json.dumps(r.messages) for r in trace.requests()))
        self.assertEqual(max(len(body["messages"]) for _, body in posts), 5)
        self.assertEqual([r["success"] for r in rows], [True] * 6)


class RecipeResolved(unittest.TestCase):
    """A recipe's silent defaults are read from the live server before the first request."""

    def run_recipe(self, metrics, sets=("mode.serve_profile=single-turn",), models=None, flags=""):
        recipe = f"vllm serve /w --tensor-parallel-size 2 {flags}"
        (cfg,) = loader.load(
            RUNS[0].with_name("llama.yaml"),
            ["engine.model=/w", f"engine.recipe={recipe}", "hardware.visible_devices=0,1", "mode.grid=null", *sets],
            env={},
        )
        handle = mock.Mock(gpu="H200", endpoint="http://127.0.0.1:1")
        replies = {"/metrics": metrics, "/v1/models": models or '{"data": [{"id": "/w", "max_model_len": 4096}]}'}

        def urlopen(url, timeout):
            reply = mock.MagicMock()
            reply.__enter__.return_value = io.BytesIO(replies[url.removeprefix("http://127.0.0.1:1")].encode())
            return reply

        def measure(trace, driver, mode, writer, workload):
            writer.finish({})

        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch.object(runner, "build_trace"),
            mock.patch.object(runner.launcher, "start", return_value=handle),
            mock.patch.object(runner.launcher, "idle_gpus", return_value="0,1"),
            mock.patch.object(drivers.launcher, "urlopen", side_effect=urlopen),
            mock.patch.object(runner.serve, "run", side_effect=measure) as served,
        ):
            out = Path(td, "run")
            try:
                runner.run_one(cfg, out)
            except RuntimeError as e:
                error = e
            else:
                error = None
            doc = json.loads((out / "run.json").read_text())
        return error, doc, served

    def test_resolved_values_land_in_run_json(self):
        error, doc, served = self.run_recipe(helpers.cache_config_text())
        self.assertIsNone(error)
        self.assertEqual(
            doc["provenance"]["engine_resolved"],
            {
                "enable_prefix_caching": True,
                "gpu_memory_utilization": 0.85,
                "kv_cache_size_tokens": 82777364,
                "block_size": 128,
                "cache_dtype": "auto",
                "max_model_len": 4096,
            },
        )

    def test_a_pinned_kv_cache_size_replaces_the_utilization_fraction(self):
        _, doc, _ = self.run_recipe(helpers.cache_config_text(), flags="--kv-cache-memory-bytes 4G")
        resolved = doc["provenance"]["engine_resolved"]
        self.assertNotIn("gpu_memory_utilization", resolved)
        self.assertEqual(resolved["kv_cache_memory_bytes"], "4G")

    def test_engines_that_differ_are_recorded_one_by_one(self):
        _, doc, _ = self.run_recipe(helpers.cache_config_text(engines=2))
        self.assertEqual(sorted(doc["provenance"]["engine_resolved"]["per_engine"]), ["0", "1"])

    def test_a_server_without_the_prefix_caching_the_profile_needs_fails_before_measuring(self):
        error, doc, served = self.run_recipe(helpers.cache_config_text(prefix_caching=False))
        self.assertRegex(str(error), "enable_prefix_caching=False, but this run needs True")
        served.assert_not_called()
        self.assertEqual(doc["run_status"]["status"], "failed")

    def test_engines_that_disagree_on_prefix_caching_fail(self):
        text = helpers.cache_config_text(prefix_caching=True) + helpers.cache_config_text(prefix_caching=False).replace(
            'engine="0"', 'engine="1"'
        )
        error, _, served = self.run_recipe(text)
        self.assertRegex(str(error), "engines disagree")
        served.assert_not_called()

    def test_a_server_that_cannot_say_fails(self):
        error, _, served = self.run_recipe(helpers.METRICS)
        self.assertRegex(str(error), "no vllm:cache_config_info")
        served.assert_not_called()

    def test_a_profile_that_does_not_care_still_records_what_ran(self):
        error, doc, _ = self.run_recipe(helpers.cache_config_text(prefix_caching=False), sets=())
        self.assertIsNone(error)
        self.assertFalse(doc["provenance"]["engine_resolved"]["enable_prefix_caching"])

    def test_a_run_without_a_recipe_asks_the_server_nothing(self):
        (cfg,) = loader.load(RUNS[0].with_name("llama.yaml"), ["engine.model=/w", "mode.grid=null"], env={})
        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch.object(runner, "build_trace"),
            mock.patch.object(runner.launcher, "start"),
            mock.patch.object(runner.launcher, "idle_gpus", return_value="0"),
            mock.patch.object(drivers.HttpDriver, "engine_resolved") as resolved,
            mock.patch.object(runner.serve, "run", side_effect=lambda t, d, m, w, wl: w.finish({})),
        ):
            runner.run_one(cfg, Path(td, "run"))
            doc = json.loads(Path(td, "run", "run.json").read_text())
        resolved.assert_not_called()
        self.assertNotIn("engine_resolved", doc["provenance"])


class FailureRecord(unittest.TestCase):
    def setUp(self):
        """run_probe exports CUDA_VISIBLE_DEVICES into os.environ; undo it per test."""
        patch = mock.patch.dict(os.environ, {}, clear=True)
        patch.start()
        self.addCleanup(patch.stop)

    def run_raising(self, sets=("hardware.visible_devices=0",), **patches):
        (cfg,) = loader.load(RUNS[0].with_name("llama.yaml"), ["engine.model=/w", "mode.grid=null", *sets], env={})
        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch.object(runner, "build_trace"),
            mock.patch.object(runner.launcher, "start", **patches.get("start", {})),
            mock.patch.object(runner.serve, "run", **patches.get("serve", {})),
        ):
            out = Path(td, "run")
            with self.assertRaises(BaseException) as raised:
                runner.run_one(cfg, out)
            doc = json.loads((out / "run.json").read_text())
            files = sorted(p.name for p in out.iterdir())
        return raised.exception, doc, files

    def test_a_failed_launch_is_recorded(self):
        exc, doc, files = self.run_raising(start={"side_effect": RuntimeError("no health")})
        self.assertIsInstance(exc, RuntimeError)
        self.assertEqual(doc["run_status"], {"status": "failed", "reason": "RuntimeError: no health"})
        self.assertEqual(files, ["requests.jsonl", "run.json"])

    def test_translate_gpu_pick_and_env_export_failures_are_recorded(self):
        targets = {
            "translate": (runner.base.get("vllm"), "translate", ()),
            "idle_gpus": (runner.launcher, "idle_gpus", ("hardware.visible_devices=null",)),
            "export": (runner.env_map, "export", ("mode.grid=llama_single", "mode.return_outputs=text")),
        }
        for name, (obj, attr, sets) in targets.items():
            with self.subTest(name), mock.patch.object(obj, attr, side_effect=OSError(name)):
                exc, doc, files = self.run_raising(sets=("hardware.visible_devices=0", *sets))
            self.assertIsInstance(exc, OSError)
            self.assertEqual(doc["run_status"], {"status": "failed", "reason": f"OSError: {name}"})
            self.assertEqual(files, ["requests.jsonl", "run.json"])

    def test_ctrl_c_mid_run_is_aborted_and_the_server_stops(self):
        handle = mock.Mock(gpu="H200", endpoint="http://127.0.0.1:1")
        exc, doc, _ = self.run_raising(start={"return_value": handle}, serve={"side_effect": KeyboardInterrupt()})
        self.assertIsInstance(exc, KeyboardInterrupt)
        self.assertEqual(doc["run_status"]["status"], "aborted")
        self.assertEqual(doc["provenance"]["gpu"], "H200")
        handle.stop.assert_called_once_with()

    def test_a_probe_whose_gpu_pick_fails_is_recorded(self):
        with (
            scratch_configs("p", {"mode": {"mode": "probe", "probe": "fake"}}) as (path, td),
            mock.patch.dict(sys.modules, {"quettabench.probes.fake": mock.Mock()}),
            mock.patch.object(runner.launcher, "idle_gpus", side_effect=RuntimeError("busy")),
        ):
            with self.assertRaises(RuntimeError):
                runner.main(path, results_dir=td / "results", allow_unverified=True)
            doc = json.loads((td / "results" / "p" / "run.json").read_text())
        self.assertEqual(doc["run_status"], {"status": "failed", "reason": "RuntimeError: busy"})

    def test_a_probe_that_raises_is_recorded(self):
        probe = mock.Mock(**{"main.side_effect": ValueError("bad shape")})
        with (
            scratch_configs("p", {"mode": {"mode": "probe", "probe": "fake"}}) as (path, td),
            mock.patch.dict(sys.modules, {"quettabench.probes.fake": probe}),
        ):
            with self.assertRaises(ValueError):
                runner.main(path, results_dir=td / "results", allow_unverified=True)
            doc = json.loads((td / "results" / "p" / "run.json").read_text())
        self.assertEqual(doc["run_status"], {"status": "failed", "reason": "ValueError: bad shape"})


class ProbeMode(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.dict(os.environ, {}, clear=True)
        patch.start()
        self.addCleanup(patch.stop)

    def test_probe_gets_its_flags_and_the_run_dir(self):
        run = {"mode": {"mode": "probe", "probe": "fake", "probe_args": "--gpu-label A100 --wide"}}
        probe = mock.Mock()
        with (
            scratch_configs("gemm-a100", run) as (path, td),
            mock.patch.dict(sys.modules, {"quettabench.probes.fake": probe}),
        ):
            _, exit_msg = qbench("run", str(path), "--allow-unverified", "--results-dir", str(td / "results"))
            doc = json.loads((td / "results" / "gemm-a100" / "run.json").read_text())
        self.assertIsNone(exit_msg)
        argv = ["--gpu-label", "A100", "--wide", "--out", str(td / "results" / "gemm-a100")]
        probe.main.assert_called_once_with(argv)
        self.assertEqual(doc["summary"], {"probe": "fake", "argv": argv})

    def test_probe_runs_on_the_runs_gpus(self):
        seen = []

        def import_probe(name):
            seen.append(os.environ.get("CUDA_VISIBLE_DEVICES"))
            return mock.Mock()

        for hardware, idle, gpus in (({"visible_devices": "2,3"}, None, "2,3"), ({}, "5", "5")):
            run = {"mode": {"mode": "probe", "probe": "fake"}, "hardware": hardware}
            with (
                self.subTest(gpus=gpus),
                scratch_configs("p", run) as (path, td),
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.dict(sys.modules, {"quettabench.probes.fake": mock.Mock()}),
                mock.patch.object(runner.importlib, "import_module", side_effect=import_probe),
                mock.patch.object(launcher, "idle_gpus", return_value=idle) as idle_gpus,
            ):
                runner.main(path, results_dir=td / "results", allow_unverified=True)
                doc = json.loads((td / "results" / "p" / "run.json").read_text())
            self.assertEqual(seen.pop(), gpus)
            self.assertEqual(doc["provenance"]["launch_env"], {"CUDA_VISIBLE_DEVICES": gpus})
            self.assertEqual(idle_gpus.called, idle is not None)

    def test_a_run_dir_with_a_run_is_refused_without_overwrite(self):
        run = {"mode": {"mode": "probe", "probe": "fake"}}
        old_files = (
            "run.json",
            "requests.jsonl",
            "engine.json",
            "server.log",
            "bench.log",
            "run.json.tmp",
            "nsys.sqlite",
            "nsys-512x8.nsys-rep",
        )
        with (
            scratch_configs("p", run) as (path, td),
            mock.patch.dict(sys.modules, {"quettabench.probes.fake": mock.Mock()}),
        ):
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

    def test_overwrite_never_empties_a_dir_qbench_did_not_make(self):
        run = {"mode": {"mode": "probe", "probe": "fake"}}
        with (
            scratch_configs("p", run) as (path, td),
            mock.patch.dict(sys.modules, {"quettabench.probes.fake": mock.Mock()}),
        ):
            results, elsewhere = td / "results", td / "elsewhere"
            elsewhere.mkdir()
            (elsewhere / "run.json").write_text("{}")
            (elsewhere / "keep.txt").write_text("mine\n")
            results.mkdir()
            (results / "p").symlink_to(elsewhere)
            argv = ("run", str(path), "--allow-unverified", "--results-dir", str(results), "--overwrite")
            _, linked = qbench(*argv)
            self.assertEqual(sorted(p.name for p in elsewhere.iterdir()), ["keep.txt", "run.json"])
            (results / "p").unlink()
            (results / "p").mkdir()
            (results / "p" / "notes.txt").write_text("mine\n")
            _, foreign = qbench(*argv)
            self.assertEqual([p.name for p in (results / "p").iterdir()], ["notes.txt"])
        self.assertIn("is a symlink", linked)
        self.assertIn("holds no run.json, requests.jsonl or engine.json", foreign)

    def test_clear_run_dir_stays_directly_under_results(self):
        with tempfile.TemporaryDirectory() as td:
            results = Path(td, "results")
            for out in (Path(td, "x"), results, results / "a" / "b", results / ".." / "x"):
                out.mkdir(parents=True, exist_ok=True)
                (out / "run.json").write_text("{}")
                with self.subTest(out=str(out)), self.assertRaisesRegex(ConfigError, "not a run dir directly under"):
                    runner.clear_run_dir(out, True, results)
                self.assertTrue((out / "run.json").exists())
                (out / "run.json").unlink()

    def test_clearing_never_follows_a_dir_swapped_for_a_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            results, elsewhere = Path(td, "results"), Path(td, "elsewhere")
            out = results / "p"
            out.mkdir(parents=True)
            (out / "run.json").write_text(json.dumps({"run_status": {"status": "failed"}}))
            elsewhere.mkdir()
            (elsewhere / "keep.txt").write_text("mine\n")

            def swap(path):
                shutil.rmtree(out)
                out.symlink_to(elsewhere)
                return True

            with (
                mock.patch.object(runner, "run_failed", side_effect=swap),
                self.assertRaisesRegex(ConfigError, "symlink"),
            ):
                runner.clear_run_dir(out, False, results)
            self.assertEqual([p.name for p in elsewhere.iterdir()], ["keep.txt"])
            self.assertTrue(out.is_symlink())
            self.assertEqual(sorted(p.name for p in results.iterdir()), ["p"])

    def test_clearing_leaves_no_dir_behind(self):
        with tempfile.TemporaryDirectory() as td:
            results = Path(td, "results")
            (results / "p" / "sub").mkdir(parents=True)
            (results / "p" / "run.json").write_text("{}")
            (results / "p" / "sub" / "x").write_text("1")
            runner.clear_run_dir(results / "p", True, results)
            self.assertEqual(list(results.iterdir()), [])

    def test_an_interrupted_clear_leaves_no_run_export_reads(self):
        from quettabench.export import quettasim

        with tempfile.TemporaryDirectory() as td:
            results = Path(td, "results")
            (results / "p" / "ncu" / "gemm").mkdir(parents=True)
            (results / "p" / "ncu" / "gemm" / "H100.csv").write_text("M,N,K\n")
            (results / "p" / "run.json").write_text(json.dumps({"run_status": {"status": "ok"}}))
            (results / "p" / "requests.jsonl").write_text("")
            with (
                mock.patch.object(runner.shutil, "rmtree", side_effect=KeyboardInterrupt),
                self.assertRaises(KeyboardInterrupt),
            ):
                runner.clear_run_dir(results / "p", True, results)
            (aside,) = results.iterdir()
            self.assertTrue(aside.name.startswith(".clearing-"))
            self.assertEqual([p.name for p in aside.iterdir()], ["ncu"])
            (aside / "run.json").write_text(json.dumps({"run_status": {"status": "ok"}}))
            quettasim.run_dirs.cache_clear()
            self.assertEqual(quettasim.run_dirs(results), {})
            quettasim.run_dirs.cache_clear()

    def test_a_failed_run_is_replaced_without_overwrite(self):
        run = {"mode": {"mode": "probe", "probe": "fake"}}
        with (
            scratch_configs("p", run) as (path, td),
            mock.patch.dict(sys.modules, {"quettabench.probes.fake": mock.Mock()}),
        ):
            out = td / "results" / "p"
            out.mkdir(parents=True)
            argv = ("run", str(path), "--allow-unverified", "--results-dir", str(td / "results"))
            for status, refused in (("failed", False), ("ok", True), ("aborted", True)):
                (out / "run.json").write_text(json.dumps({"run_status": {"status": status}}))
                (out / "requests.jsonl").write_text("old\n")
                _, exit_msg = qbench(*argv)
                with self.subTest(status):
                    self.assertEqual("already holds a run" in (exit_msg or ""), refused)
                    self.assertEqual((out / "requests.jsonl").read_text(), "old\n" if refused else "")

    def test_a_name_that_is_no_probe_is_refused(self):
        for name in ("nope", "base", "_timing", "../cli"):
            with self.subTest(name), scratch_configs("p", {"mode": {"mode": "probe", "probe": name}}) as (path, td):
                _, exit_msg = qbench("run", str(path), "--allow-unverified", "--results-dir", str(td / "results"))
                self.assertFalse((td / "results").exists())
            self.assertIn("is not a module under quettabench/probes/ with a main", exit_msg)
        self.assertTrue(runner.is_probe("gemm"))

    def test_probe_mode_needs_a_probe(self):
        with scratch_configs("bare", {"mode": {"mode": "probe"}}) as (path, _):
            _, exit_msg = qbench("run", str(path), "--dry-run", "--allow-unverified")
        self.assertIn("mode=probe needs mode.probe", exit_msg)


if __name__ == "__main__":
    unittest.main()
