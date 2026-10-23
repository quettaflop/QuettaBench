"""start/health/engine.json/stop on a fake server; no child survives stop."""

import dataclasses
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

from quettabench import provision, runner
from quettabench.config import loader
from quettabench.config.schema import ConfigError
from quettabench.engines import drivers, launcher, observe, vllm
from quettabench.engines.base import LaunchSpec
from quettabench.profiling import engine_side
from tests.contracts import helpers

FAKE = Path(__file__).with_name("fake_server.py")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A reparented zombie no longer runs; count it dead.
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return not stat.split(") ")[1].startswith("Z")


class Foreign(BaseHTTPRequestHandler):
    """A server some other run started: healthy, and its models are not ours."""

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"data": [{"id": "other"}]}')

    def log_message(self, *args):
        pass


class Keyed(Foreign):
    """A server started with --api-key: /health is public, /v1/models and /version want the bearer key."""

    def do_GET(self):
        if self.path != "/health" and self.headers.get("Authorization") != "Bearer k":
            self.send_response(401)
            self.end_headers()
            return
        super().do_GET()


class LocalLauncher(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.state = self.dir / "state.json"
        self.port = free_port()

    def start(self, **kw):
        """A port taken before the bind (exit 98) is retried on a fresh one."""
        for _ in range(5):
            try:
                handle = launcher.start(self.spec(**kw), self.dir / "run", timeout_s=30, poll_s=0.1)
            except RuntimeError as e:
                if "exited with 98" not in str(e):
                    raise
                self.port = free_port()
                continue
            self.assertEqual(json.loads(self.state.read_text())["port"], self.port)
            return handle
        self.fail("no free port in 5 tries")

    def spec(self, **kw):
        return LaunchSpec(
            adapter="fake",
            port=self.port,
            health_path="/health",
            server_argv=(sys.executable, str(FAKE), str(self.port), str(self.state)),
            env={"FAKE_MARK": "from-spec", "HF_TOKEN": "secret", "DB_SECRET": "secret"},
            **kw,
        )

    def test_engine_json_holds_no_engine_kwarg_or_argv_secret(self):
        handle = self.start(engine_kwargs={"api_key": "sk-live"})
        handle.stop(grace_s=5)
        record = json.loads((self.dir / "run" / "engine.json").read_text())
        self.assertEqual(record["launch_spec"]["engine_kwargs"], {"api_key": "REDACTED"})
        self.assertNotIn("sk-live", json.dumps(record))

    def test_start_health_engine_json_stop(self):
        handle = self.start()
        state = json.loads(self.state.read_text())
        self.assertEqual(state["mark"], "from-spec")
        record = json.loads((self.dir / "run" / "engine.json").read_text())
        self.assertEqual(record["adapter"], "fake")
        self.assertEqual(record["endpoint"], f"http://127.0.0.1:{self.port}")
        self.assertEqual(record["pid"], handle.pid)
        self.assertEqual(record["instrumentation"], "none")
        self.assertEqual(
            record["launch_spec"]["env"], {"FAKE_MARK": "from-spec", "HF_TOKEN": "REDACTED", "DB_SECRET": "REDACTED"}
        )
        self.assertNotIn("secret", (self.dir / "run" / "engine.json").read_text())
        self.assertTrue(alive(state["child_pid"]))
        handle.stop(grace_s=5)
        self.assertFalse(alive(handle.pid))
        self.assertFalse(alive(state["child_pid"]))

    def test_server_that_dies_fails_fast(self):
        spec = LaunchSpec(adapter="fake", port=self.port, server_argv=(sys.executable, "-c", "raise SystemExit(4)"))
        with self.assertRaisesRegex(RuntimeError, "exited with 4"):
            launcher.start(spec, self.dir / "dead", timeout_s=30, poll_s=0.1)

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "no server"):
            launcher.start(LaunchSpec(adapter="quettaserve"), self.dir)

    def foreign_server(self):
        server = HTTPServer(("127.0.0.1", self.port), Foreign)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

    def test_a_server_already_on_the_port_is_refused_before_launch(self):
        self.foreign_server()
        with mock.patch.object(launcher.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, f"port {self.port} already answers"):
                launcher.start(self.spec(served_name="m"), self.dir / "run", timeout_s=30, poll_s=0.1)
        popen.assert_not_called()

    def test_a_foreign_server_that_binds_while_ours_loads_is_refused_and_ours_stopped(self):
        pidfile = self.dir / "pid"
        sleeper = f"import os, time; open({str(pidfile)!r}, 'w').write(str(os.getpid())); time.sleep(60)"
        spec = LaunchSpec(adapter="fake", port=self.port, served_name="m", server_argv=(sys.executable, "-c", sleeper))
        self.foreign_server()
        with (
            mock.patch.object(launcher, "port_taken", return_value=False),
            mock.patch.object(launcher, "FOREIGN_GRACE_S", 0.3),
            self.assertRaisesRegex(RuntimeError, "another server answers on .*still starting"),
        ):
            launcher.start(spec, self.dir / "run", timeout_s=30, poll_s=0.1)
        self.assertFalse(alive(int(pidfile.read_text())))

    def test_our_server_passes_once_it_lists_the_token(self):
        handle = self.start(served_name="llama")
        argv = json.loads((self.dir / "run" / "engine.json").read_text())["launch_spec"]["server_argv"]
        self.assertEqual(argv[-3], "--served-model-name")
        self.assertEqual(argv[-2], "llama")
        self.assertRegex(argv[-1], "^qbench-[0-9a-f]{16}$")
        handle.stop(grace_s=5)

    def test_the_token_follows_a_served_name_the_argv_already_sets(self):
        spec = LaunchSpec(
            adapter="vllm",
            served_name="llama",
            server_argv=("vllm", "serve", "--served-model-name", "llama", "--port", "1"),
        )
        self.assertEqual(
            launcher.tagged(spec, "tok").server_argv,
            ("vllm", "serve", "--served-model-name", "llama", "tok", "--port", "1"),
        )


DOCKER_YAML = launcher.REPO / "configs" / "runs" / "llama-tp1-docker.yaml"


def docker_spec(*sets):
    (cfg,) = loader.load(
        DOCKER_YAML,
        ["engine.model=/hf/llama", "hardware.backend=cuda", "hardware.link_profile=nvlink", *sets],
        env={"HF_HOME": "/hf"},
    )
    return cfg, vllm.ADAPTER.translate(cfg)


def keyed_docker_spec(key):
    cfg, _ = docker_spec()
    extra = {**cfg.engine.extra, "vllm": {**cfg.engine.extra.get("vllm", {}), "api_key": key}}
    cfg = dataclasses.replace(cfg, engine=dataclasses.replace(cfg.engine, extra=extra))
    return cfg, vllm.ADAPTER.translate(cfg)


NAME = "qbench-llama-tp1-docker-77-abcdef"


def fixed_name(case):
    for name, value in (("getpid", 77), ("urandom", b"\xab\xcd\xef")):
        patch = mock.patch.object(launcher.os, name, return_value=value)
        patch.start()
        case.addCleanup(patch.stop)


class Attach(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.port = free_port()

    def spec(self, served_name="other"):
        return LaunchSpec(
            adapter="fake", port=self.port, health_path="/health", served_name=served_name, env={"HF_TOKEN": "secret"}
        )

    def serve(self):
        server = HTTPServer(("127.0.0.1", self.port), Foreign)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server

    def test_attach_measures_the_running_server_and_leaves_it_running(self):
        self.serve()
        handle = launcher.attach(self.spec(), self.dir / "run", timeout_s=5, poll_s=0.1)
        self.assertEqual(handle.endpoint, f"http://127.0.0.1:{self.port}")
        handle.stop()
        self.assertTrue(launcher.port_taken(self.port))
        record = json.loads((self.dir / "run" / "engine.json").read_text())
        self.assertTrue(record["attached"])
        self.assertEqual(record["served_models"], ["other"])
        self.assertEqual((handle.served_models, handle.server_version), (["other"], record["server_version"]))
        self.assertEqual(record["expected_launch_spec"]["env"], {"HF_TOKEN": "REDACTED"})

    def test_attach_refuses_a_server_that_does_not_serve_the_model(self):
        self.serve()
        with self.assertRaisesRegex(RuntimeError, r"serves \['other'\], not mine"):
            launcher.attach(self.spec("mine"), self.dir / "run", timeout_s=5, poll_s=0.1)

    def test_attach_refuses_an_empty_port(self):
        with self.assertRaisesRegex(RuntimeError, "nothing healthy on"):
            launcher.attach(self.spec(), self.dir / "run", timeout_s=0.3, poll_s=0.1)

    def test_attach_sends_the_servers_key_on_discovery(self):
        # A server started with --api-key answers /health to anyone and /v1/models only with the key: discovery
        # must carry the key the yaml configured, or a healthy keyed server reads as serving nothing.
        server = HTTPServer(("127.0.0.1", self.port), Keyed)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        keyed = LaunchSpec(
            adapter="fake", port=self.port, served_name="other", env={"VLLM_API_KEY": "k"}, api_key_env="VLLM_API_KEY"
        )
        handle = launcher.attach(keyed, self.dir / "run", timeout_s=5, poll_s=0.1)
        self.assertEqual(handle.served_models, ["other"])
        self.assertEqual(
            json.loads((self.dir / "run" / "engine.json").read_text())["expected_launch_spec"]["env"],
            {"VLLM_API_KEY": "REDACTED"},
        )
        with self.assertRaisesRegex(RuntimeError, r"serves \[\], not other"):
            launcher.attach(self.spec(), self.dir / "run2", timeout_s=5, poll_s=0.1)


class DockerArgv(unittest.TestCase):
    def setUp(self):
        fixed_name(self)

    def test_a_containers_env_is_its_e_set_and_run_json_records_that(self):
        cfg, spec = keyed_docker_spec("sk-1")
        argv = launcher.docker_argv(spec, cfg, "2", "/r/run")
        given = {argv[i + 1].split("=", 1)[0] for i, a in enumerate(argv) if a == "-e"}
        self.assertEqual(given, set(launcher.docker_env(spec, cfg)))
        self.assertEqual(
            runner.server_env_record(cfg, spec, "docker"),
            {
                "server_env": "docker -e",
                "launch_env": runner.redact(launcher.docker_env(spec, cfg)),
                "inherited_env": None,
            },
        )
        self.assertEqual(runner.server_env_record(cfg, spec, "docker")["launch_env"]["VLLM_API_KEY"], "REDACTED")

    def test_example_yaml_argv(self):
        cfg, spec = docker_spec()
        repo = str(launcher.REPO)
        self.assertEqual(
            launcher.docker_argv(spec, cfg, "2", "/r/run"),
            [
                "docker",
                "run",
                "-d",
                "--pull=never",
                "--init",
                "--name",
                NAME,
                "--gpus",
                '"device=2"',
                "--ipc=host",
                "--shm-size=16g",
                "-p",
                "8000:8000",
                "-v",
                "/hf:/hf:ro",
                "-v",
                f"{repo}:{repo}:ro",
                "-v",
                "/r/run:/r/run",
                "-e",
                "VLLM_SERVER_DEV_MODE=1",
                "-e",
                "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1",
                "-e",
                "NCCL_IB_DISABLE=1",
                "-e",
                "HF_HOME=/hf",
                "--entrypoint",
                "vllm",
                "vllm/vllm-openai:v0.26.0",
                "serve",
                "/hf/llama",
                "--port",
                "8000",
                "--tensor-parallel-size",
                "1",
                "--dtype",
                "float16",
                "--max-model-len",
                "9216",
                "--gpu-memory-utilization",
                "0.9",
                "--no-enable-prefix-caching",
                "--enable-chunked-prefill",
                "--trust-remote-code",
            ],
        )

    def test_shm_size_and_collective_pins_come_from_hardware(self):
        cfg, spec = docker_spec("hardware.shm_size=64g", "hardware.link_profile=pcie")
        argv = launcher.docker_argv(spec, cfg, "0,1", "/r/run")
        self.assertIn("--shm-size=64g", argv)
        self.assertIn('"device=0,1"', argv)
        self.assertIn("NCCL_PROTO=Simple", argv)

    def test_provisioned_model_dir_is_mounted(self):
        cfg, spec = docker_spec("engine.model=/cache/models/llama")
        argv = launcher.docker_argv(spec, cfg, "0", "/r/run")
        self.assertIn(
            "/cache/models/llama:/cache/models/llama:ro", [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
        )
        cfg, spec = docker_spec("engine.model=meta-llama/Llama-3.1-8B")
        self.assertNotIn(
            "meta-llama/Llama-3.1-8B:meta-llama/Llama-3.1-8B:ro", launcher.docker_argv(spec, cfg, "0", "/r/run")
        )

    def test_a_secret_env_goes_by_name_only(self):
        """docker run -e NAME takes NAME from the client's env: the key is never in argv."""
        cfg, spec = keyed_docker_spec("sk-live")
        argv = launcher.docker_argv(spec, cfg, "0", "/r/run")
        self.assertEqual(argv[argv.index("VLLM_API_KEY") - 1], "-e")
        self.assertNotIn("sk-live", " ".join(argv))
        self.assertEqual(launcher.docker_secrets(spec), {"VLLM_API_KEY": "sk-live"})

    def test_missing_image_is_a_config_error(self):
        cfg, spec = docker_spec("engine.image=null")
        with self.assertRaisesRegex(ConfigError, "needs engine.image"):
            launcher.docker_argv(spec, cfg, "0", "/r/run")


class DockerStart(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        fixed_name(self)
        patch = mock.patch.object(launcher, "port_taken", return_value=False)
        patch.start()
        self.addCleanup(patch.stop)

    def run_mock(self, docker_rc=0, created="c0ffee\n"):
        def run(argv, **kw):
            if argv[:2] == ["docker", "run"]:
                return mock.Mock(returncode=docker_rc, stdout=created, stderr="no such image")
            return mock.Mock(returncode=0, stdout="", stderr="")

        return mock.patch.object(launcher.subprocess, "run", side_effect=run)

    @staticmethod
    def docker_calls(run):
        return [c.args[0] for c in run.call_args_list if c.args[0][0] == "docker"][1:]

    def start(self, cfg, spec):
        return launcher.start(spec, self.dir / "run", backend="docker", cfg=cfg, gpus="0")

    def test_start_logs_engine_json_and_stop(self):
        cfg, spec = docker_spec()
        proc = mock.Mock(pid=4242, **{"poll.return_value": None})
        with (
            self.run_mock() as run,
            mock.patch.object(launcher.subprocess, "Popen", return_value=proc) as popen,
            mock.patch.object(launcher, "wait_healthy") as healthy,
        ):
            handle = self.start(cfg, spec)
            ours = launcher.tagged(spec, "qbench-abcdef")
            self.assertEqual(run.call_args_list[0].args[0], launcher.docker_argv(ours, cfg, "0", self.dir / "run"))
            self.assertEqual(popen.call_args.args[0], ["docker", "logs", "-f", NAME])
            self.assertEqual(popen.call_args.kwargs["stdout"].name, str(self.dir / "run" / "server.log"))
            healthy.assert_called_once()
            self.assertEqual(healthy.call_args.args[:4], (proc, "http://127.0.0.1:8000/health", 1800, 1.0))
            record = json.loads((self.dir / "run" / "engine.json").read_text())
            self.assertEqual(
                (record["pid"], record["container"], record["container_id"], record["image"]),
                (proc.pid, NAME, "c0ffee", "vllm/vllm-openai:v0.26.0"),
            )
            self.assertEqual((handle.pid, handle.container_id), (proc.pid, "c0ffee"))
            handle.stop(grace_s=5)
            self.assertEqual(
                self.docker_calls(run), [["docker", "stop", "-t", "5", NAME], ["docker", "rm", "-f", NAME]]
            )
            proc.wait.assert_called_once_with(timeout=15)

    def test_docker_run_gets_the_secret_in_its_env(self):
        cfg, spec = keyed_docker_spec("sk-live")
        proc = mock.Mock(pid=4242, **{"poll.return_value": None})
        with (
            self.run_mock() as run,
            mock.patch.object(launcher.subprocess, "Popen", return_value=proc),
            mock.patch.object(launcher, "wait_healthy"),
        ):
            self.start(cfg, spec)
        docker_run = run.call_args_list[0]
        self.assertEqual(docker_run.kwargs["env"]["VLLM_API_KEY"], "sk-live")
        self.assertNotIn("sk-live", " ".join(docker_run.args[0]))
        self.assertNotIn("sk-live", (self.dir / "run" / "engine.json").read_text())

    def test_stop_kills_a_log_stream_that_does_not_end(self):
        cfg, spec = docker_spec()
        proc = mock.Mock(pid=4242, **{"wait.side_effect": [launcher.subprocess.TimeoutExpired("logs", 15), 0]})
        with (
            self.run_mock() as run,
            mock.patch.object(launcher.subprocess, "Popen", return_value=proc),
            mock.patch.object(launcher, "wait_healthy"),
        ):
            self.start(cfg, spec).stop(grace_s=5)
        proc.kill.assert_called_once_with()
        self.assertEqual(run.call_args.args[0], ["docker", "rm", "-f", NAME])

    def test_container_name_is_made_docker_safe_and_unique(self):
        self.assertEqual(launcher.container_name("a/b c:d_e.f-g"), "qbench-a-b-c-d_e.f-g-77-abcdef")
        mock.patch.stopall()
        self.assertNotEqual(launcher.container_name("r"), launcher.container_name("r"))

    def test_unhealthy_container_is_stopped(self):
        cfg, spec = docker_spec()
        with (
            self.run_mock() as run,
            mock.patch.object(launcher.subprocess, "Popen"),
            mock.patch.object(launcher, "wait_healthy", side_effect=TimeoutError("late")),
        ):
            with self.assertRaises(TimeoutError):
                self.start(cfg, spec)
        self.assertEqual([argv[:2] for argv in self.docker_calls(run)], [["docker", "stop"], ["docker", "rm"]])

    def test_failed_docker_run_and_missing_image(self):
        cfg, spec = docker_spec()
        with self.run_mock(docker_rc=125) as run, mock.patch.object(launcher.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "docker run exited with 125: no such image"):
                self.start(cfg, spec)
        popen.assert_not_called()
        self.assertEqual(self.docker_calls(run), [["docker", "logs", NAME], ["docker", "rm", "-f", NAME]])
        self.assertEqual(run.call_args_list[1].kwargs["stdout"].name, str(self.dir / "run" / "server.log"))
        self.assertEqual((self.dir / "run" / "server.log").read_text(), "no such image")

    def test_a_docker_run_that_printed_no_id_is_still_cleaned_by_name(self):
        cfg, spec = docker_spec()
        with (
            self.run_mock(docker_rc=125, created="") as run,
            self.assertRaisesRegex(RuntimeError, "docker run exited with 125"),
        ):
            self.start(cfg, spec)
        self.assertEqual(self.docker_calls(run), [["docker", "logs", NAME], ["docker", "rm", "-f", NAME]])
        cfg, spec = docker_spec("engine.image=null")
        with self.run_mock() as run, self.assertRaisesRegex(ConfigError, "needs engine.image"):
            self.start(cfg, spec)
        run.assert_not_called()


class NoProxy(unittest.TestCase):
    def test_endpoints_are_reached_directly(self):
        dead_proxy = {
            "http_proxy": "http://127.0.0.1:9",
            "HTTP_PROXY": "http://127.0.0.1:9",
            "no_proxy": "",
            "NO_PROXY": "",
        }
        with helpers.serve() as (endpoint, _), mock.patch.dict(os.environ, dead_proxy):
            self.assertIn("process_open_fds", engine_side.scrape_metrics(endpoint))
            with launcher.urlopen(endpoint + "/metrics", timeout=5) as resp:
                self.assertEqual(resp.status, 200)


class IdleGpus(unittest.TestCase):
    def test_non_cuda_takes_the_first_n(self):
        self.assertEqual(launcher.idle_gpus(3, "rocm"), "0,1,2")
        self.assertEqual(launcher.idle_gpus(2, "cpu"), "0,1")

    def test_cuda_picks_gpus_under_1000_mib(self):
        smi = "0, 70000 MiB\n1, 3 MiB\n2, 999 MiB\n3, 1000 MiB\n"
        with mock.patch.object(launcher.subprocess, "run", return_value=mock.Mock(stdout=smi)):
            self.assertEqual(launcher.idle_gpus(2, "cuda"), "1,2")
            with self.assertRaisesRegex(RuntimeError, "NEED 3 FREE GPUS, have: 1 2"):
                launcher.idle_gpus(3, "cuda")


def tools_on_path(td, **outputs):
    for tool, out in outputs.items():
        script = Path(td, tool)
        script.write_text(f"#!/bin/sh\nprintf '{out}'\n")
        script.chmod(0o755)
    return mock.patch.dict(os.environ, {"PATH": td})


class Backends(unittest.TestCase):
    def test_visible_env_per_backend(self):
        self.assertEqual(launcher.visible_env("cuda", "0,1"), ({"CUDA_VISIBLE_DEVICES": "0,1"}, ()))
        self.assertEqual(
            launcher.visible_env("rocm", "2"), ({"HIP_VISIBLE_DEVICES": "2", "CUDA_VISIBLE_DEVICES": "2"}, ())
        )
        self.assertEqual(
            launcher.visible_env("tpu", "0,1,2,3"),
            (
                {
                    "TPU_VISIBLE_CHIPS": "0,1,2,3",
                    "TPU_CHIPS_PER_PROCESS_BOUNDS": "1,4,1",
                    "TPU_PROCESS_BOUNDS": "1,1,1",
                },
                (),
            ),
        )
        self.assertEqual(
            launcher.visible_env("neuron", "0,1"), ({"NEURON_VISIBLE_DEVICES": "0,1"}, ("NEURON_RT_VISIBLE_CORES",))
        )

    def test_rocm_is_detected_from_rocm_smi_or_amd_smi(self):
        for tool in ("rocm-smi", "amd-smi"):
            with self.subTest(tool=tool), tempfile.TemporaryDirectory() as td, tools_on_path(td, **{tool: ""}):
                self.assertEqual(loader.detect_backend(None), "rocm")

    def child_spec(self, backend, *sets):
        (cfg,) = loader.load(
            DOCKER_YAML,
            [
                "engine.model=/w",
                f"hardware.backend={backend}",
                "engine.dtype=bfloat16",
                "mode.grid=null",
                "hardware.visible_devices=0,1",
                *sets,
            ],
            env={},
        )
        with (
            mock.patch.object(runner.launcher, "start") as start,
            mock.patch.object(runner, "build_trace"),
            mock.patch.object(runner.serve, "run"),
            tempfile.TemporaryDirectory() as td,
        ):
            runner.serve_one(cfg, Path(td), set(), mock.Mock(provenance={}))
        return start.call_args.args[0]

    def test_the_server_gets_its_backends_devices(self):
        for backend, launch in (("rocm", "local"), ("tpu", "docker"), ("neuron", "local")):
            with self.subTest(backend=backend):
                spec = self.child_spec(backend, f"hardware.launcher={launch}", "hardware.server_env=inherit")
                want, unset = launcher.visible_env(backend, "0,1")
                self.assertEqual({k: spec.env[k] for k in want}, want)
                self.assertEqual(set(unset) <= set(spec.env_drop), True)
        # Controlled, a name to unset is simply not in the empty environment, even when the shell has it.
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "7"}):
            spec = self.child_spec("neuron", "hardware.launcher=local")
        want, unset = launcher.visible_env("neuron", "0,1")
        self.assertTrue(spec.env_clear)
        self.assertEqual({k: spec.env[k] for k in want}, want)
        self.assertFalse(set(unset) & set(spec.env) - set(want))
        self.assertNotIn("CUDA_VISIBLE_DEVICES", self.child_spec("cuda", "hardware.launcher=docker").env)

    def test_docker_device_flags_per_backend(self):
        fixed_name(self)
        flags = {
            "rocm": [
                "--group-add=video",
                "--cap-add=SYS_PTRACE",
                "--security-opt",
                "seccomp=unconfined",
                "--device",
                "/dev/kfd",
                "--device",
                "/dev/dri",
            ],
            "tpu": ["--privileged", "--net=host"],
            "neuron": [
                "--device=/dev/neuron0",
                "--device=/dev/neuron1",
                "--device=/dev/neuron10",
                "--cap-add",
                "SYS_ADMIN",
                "--cap-add",
                "IPC_LOCK",
            ],
        }
        devices = ["/dev/neuron10", "/dev/neuron1", "/dev/neuron0"]
        for backend, want in flags.items():
            with self.subTest(backend=backend), mock.patch.object(launcher.glob, "glob", return_value=devices):
                cfg, spec = docker_spec(f"hardware.backend={backend}", "engine.dtype=bfloat16")
                argv = launcher.docker_argv(spec, cfg, "0,1", "/r/run")
                self.assertEqual(argv[7 : 7 + len(want)], want)
                self.assertNotIn("--gpus", argv)
                self.assertEqual("-p" in argv, backend != "tpu")
        cfg, spec = docker_spec("hardware.backend=neuron")
        with (
            mock.patch.object(launcher.glob, "glob", return_value=[]),
            self.assertRaisesRegex(ConfigError, "no /dev/neuron"),
        ):
            launcher.docker_argv(spec, cfg, "0", "/r/run")

    def test_accelerator_identity_per_backend(self):
        with (
            tempfile.TemporaryDirectory() as td,
            tools_on_path(
                td,
                **{
                    "nvidia-smi": "NVIDIA H200, 143771\n",
                    "amd-smi": "ASIC:\n    MARKET_NAME: AMD Instinct MI300X\n",
                    "neuron-ls": "instance-type: trn2.48xlarge\ninstance-id: i-0\n",
                },
            ),
        ):
            self.assertEqual(launcher.accelerator("cuda"), ("NVIDIA H200", 143771 << 20))
            self.assertEqual(launcher.accelerator("rocm"), ("AMD Instinct MI300X", None))
            self.assertEqual(launcher.accelerator("neuron"), ("trn2.48xlarge", None))
            self.assertEqual(launcher.accelerator("tpu"), ("", None))
        with tempfile.TemporaryDirectory() as td, tools_on_path(td):
            for backend in ("cuda", "rocm", "neuron", "tpu", "cpu"):
                self.assertEqual(launcher.accelerator(backend), ("", None))
        handle = launcher.NoServer(Path("r"), endpoint="http://127.0.0.1:1", backend="rocm")
        (cfg,) = loader.load(DOCKER_YAML, ["engine.model=/w", "hardware.backend=rocm", "mode.grid=null"], env={})
        driver = vllm.ADAPTER.driver(vllm.ADAPTER.translate(cfg), handle)
        with mock.patch.object(drivers.launcher, "accelerator", return_value=("MI300X", None)) as acc:
            self.assertEqual(driver.device(), ("MI300X", None))
        acc.assert_called_once_with("rocm")

    def test_cuda_only_refusals(self):
        def load(*sets):
            return loader.load(DOCKER_YAML.with_name("llama.yaml"), ["engine.model=/w", *sets], env={})

        with self.assertRaisesRegex(ConfigError, "TPU has no fp16; set engine.dtype=bfloat16"):
            load("hardware.backend=tpu")
        for backend in ("rocm", "neuron", "tpu"):
            with self.subTest(backend=backend):
                hw = [f"hardware.backend={backend}", "engine.dtype=bfloat16"]
                with self.assertRaisesRegex(ConfigError, f"nsys traces CUDA; hardware.backend={backend}"):
                    load(*hw, "engine.instrumentation=nsys")
                (probe,) = load(*hw, "mode.mode=probe", "mode.probe=gemm", "mode.grid=null")
                with self.assertRaisesRegex(ConfigError, "the kernel probes time CUDA kernels"):
                    runner.check_probe(probe)
                (ref,) = load(*hw, "engine.name=reference")
                with self.assertRaisesRegex(ConfigError, "reference: runs transformers on cuda:0"):
                    drivers.REFERENCE.translate(ref)
                plan = provision.Plan(True)
                with self.assertRaisesRegex(ConfigError, "nccl_lib is CUDA's NCCL"):
                    provision.machine(plan, {"cache_root": "/tmp/q", "nccl_lib": "/n"}, True, backend)


VLLM_LOG = (
    "INFO 10-19 02:00:00 [core.py:69] Initializing a V1 LLM engine (v0.31.0) with config: model='m', "
    "speculative_config=SpeculativeConfig(method='dspark', num_spec_tokens=5), tokenizer='m', "
    "tensor_parallel_size=8, pipeline_parallel_size=1, data_parallel_size=1, enable_prefix_caching=True, "
    "chunked_prefill_enabled=True, compilation_config={'level': 3, 'cudagraph_capture_sizes': [64, 32, 16, 8], "
    "'max_capture_size': 64}\n"
    "INFO 10-19 02:05:00 [kv_cache_utils.py:1087] GPU KV cache size: 1,234,567 tokens\n"
    "INFO 10-19 02:05:00 [kv_cache_utils.py:1091] Maximum concurrency for 1,048,576 tokens per request: 1.18x\n"
)


class Observed(BaseHTTPRequestHandler):
    """A server that answers /version and /v1/models."""

    def do_GET(self):
        body = {"version": "0.31.0"} if self.path == "/version" else {"data": [{"id": "m"}]}
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def log_message(self, *args):
        pass


class ServerObserved(unittest.TestCase):
    def test_the_startup_log_gives_the_settings_the_server_runs_with(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "server.log"
            log.write_text(VLLM_LOG)
            got = observe.log_settings(log)
            missing = observe.log_settings(Path(td) / "gone.log")
        self.assertEqual(
            got,
            {
                "tensor_parallel_size": 8,
                "pipeline_parallel_size": 1,
                "data_parallel_size": 1,
                "expert_parallel": None,  # not in this line: unknown, not False
                "chunked_prefill": True,
                "prefix_caching": True,
                "speculative_config": "SpeculativeConfig(method='dspark', num_spec_tokens=5)",
                "cudagraph_capture_sizes": [64, 32, 16, 8],
                "kv_cache_tokens": [1234567],
                "max_concurrency": [{"tokens_per_request": 1048576, "x": 1.18}],
            },
        )
        self.assertIn("no server log", missing["why"])

    def test_the_process_on_the_port_gives_its_argv_and_environment_with_secrets_redacted(self):
        port = free_port()
        code = f"import socket, time; s = socket.socket(); s.bind(('127.0.0.1', {port})); s.listen(); time.sleep(30)"
        env = {**os.environ, "QB_TEST_TOKEN": "secret", "QB_PLAIN": "x"}
        proc = subprocess.Popen([sys.executable, "-c", code], env=env)
        self.addCleanup(proc.wait)
        self.addCleanup(proc.kill)
        for _ in range(100):
            if observe.listening_pid(port):
                break
            time.sleep(0.05)
        self.assertEqual(observe.listening_pid(port), proc.pid)
        got = observe.process_argv_and_env(proc.pid)
        self.assertEqual(got["argv"][:2], [sys.executable, "-c"])
        self.assertEqual((got["env"]["QB_TEST_TOKEN"], got["env"]["QB_PLAIN"]), ("REDACTED", "x"))

    def test_an_attached_server_records_what_it_answers_and_its_listening_process(self):
        port = free_port()
        server = HTTPServer(("127.0.0.1", port), Observed)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        got = observe.observe_server(f"http://127.0.0.1:{port}")
        self.assertEqual((got["version"], got["models"]), ("0.31.0", ["m"]))
        self.assertEqual(got["process"]["pid"], os.getpid())  # the stub listens in this test process
        self.assertIn("qbench did not start the server", got["log"]["why"])
        boxed = observe.observe_server(f"http://127.0.0.1:{port}", container="qbench-x")
        self.assertIn("container qbench-x", boxed["process"]["why"])


if __name__ == "__main__":
    unittest.main()
