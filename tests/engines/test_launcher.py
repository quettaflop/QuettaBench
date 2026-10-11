"""start/health/engine.json/stop on a fake server, and no child survives stop."""

import json
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from quettabench.config import loader
from quettabench.config.schema import ConfigError
from quettabench.engines import launcher, vllm
from quettabench.engines.base import LaunchSpec, UnsupportedField
from quettabench.profiling import native
from tests.profiling import fake_endpoint

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
    stat = Path(f"/proc/{pid}/stat")
    return not (stat.exists() and stat.read_text().split(") ")[1].startswith("Z"))


class LocalLauncher(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.state = self.dir / "state.json"
        self.port = free_port()

    def spec(self, **kw):
        return LaunchSpec(adapter="fake", port=self.port, health_path="/health",
                          server_argv=(sys.executable, str(FAKE), str(self.port), str(self.state)),
                          env={"FAKE_MARK": "from-spec", "HF_TOKEN": "secret",
                               "DB_SECRET": "secret"}, **kw)

    def test_start_health_engine_json_stop(self):
        handle = launcher.start(self.spec(), self.dir / "run", timeout_s=30, poll_s=0.1)
        state = json.loads(self.state.read_text())
        self.assertEqual(state["mark"], "from-spec")
        record = json.loads((self.dir / "run" / "engine.json").read_text())
        self.assertEqual(record["adapter"], "fake")
        self.assertEqual(record["endpoint"], f"http://127.0.0.1:{self.port}")
        self.assertEqual(record["pid"], handle.pid)
        self.assertEqual(record["instrumentation"], "none")
        self.assertEqual(record["launch_spec"]["env"],
                         {"FAKE_MARK": "from-spec", "HF_TOKEN": "REDACTED",
                          "DB_SECRET": "REDACTED"})  # the results.redact rule
        self.assertNotIn("secret", (self.dir / "run" / "engine.json").read_text())
        self.assertTrue(alive(state["child_pid"]))
        handle.stop(grace_s=5)
        self.assertFalse(alive(handle.pid))
        self.assertFalse(alive(state["child_pid"]))

    def test_server_that_dies_fails_fast(self):
        spec = LaunchSpec(adapter="fake", port=self.port,
                          server_argv=(sys.executable, "-c", "raise SystemExit(4)"))
        with self.assertRaisesRegex(RuntimeError, "exited with 4"):
            launcher.start(spec, self.dir / "dead", timeout_s=30, poll_s=0.1)

    def test_refusals(self):
        with self.assertRaises(UnsupportedField):
            launcher.start(LaunchSpec(adapter="vllm", server_unsupported=("pp=2",)), self.dir)
        with self.assertRaisesRegex(ValueError, "no server"):
            launcher.start(LaunchSpec(adapter="quettaserve"), self.dir)


DOCKER_YAML = launcher.REPO / "configs" / "runs" / "llama-tp1-docker.yaml"


def docker_spec(*sets):
    """(cfg, spec) for the example docker run yaml on a pinned host."""
    (cfg,) = loader.load(DOCKER_YAML, ["engine.model=/hf/llama", "hardware.backend=cuda",
                                       "hardware.link_profile=nvlink", *sets],
                         env={"HF_HOME": "/hf"})
    return cfg, vllm.ADAPTER.translate(cfg)


class DockerArgv(unittest.TestCase):
    def test_example_yaml_argv(self):
        cfg, spec = docker_spec()
        repo = str(launcher.REPO)
        self.assertEqual(launcher.docker_argv(spec, cfg, "2"), [
            "docker", "run", "-d", "--pull=never", "--init", "--rm", "--name", "qbench-llama-tp1-docker",
            "--gpus", '"device=2"', "--ipc=host", "--shm-size=16g", "-p", "8000:8000",
            "-v", "/hf:/hf:ro", "-v", f"{repo}:{repo}:ro",
            "-e", "VLLM_SERVER_DEV_MODE=1", "-e", "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1",
            "-e", "NCCL_IB_DISABLE=1", "-e", "HF_HOME=/hf",
            "--entrypoint", "vllm", "vllm/vllm-openai:v0.30.0",
            "serve", "/hf/llama", "--port", "8000", "--tensor-parallel-size", "1",
            "--dtype", "float16", "--max-model-len", "9216", "--gpu-memory-utilization", "0.9",
            "--no-enable-prefix-caching", "--enable-chunked-prefill", "--trust-remote-code"])

    def test_shm_size_and_collective_pins_come_from_hardware(self):
        cfg, spec = docker_spec("hardware.shm_size=64g", "hardware.link_profile=pcie")
        argv = launcher.docker_argv(spec, cfg, "0,1")
        self.assertIn("--shm-size=64g", argv)
        self.assertIn('"device=0,1"', argv)
        self.assertIn("NCCL_PROTO=Simple", argv)

    def test_provisioned_model_dir_is_mounted(self):
        cfg, spec = docker_spec("engine.model=/cache/models/llama")
        argv = launcher.docker_argv(spec, cfg, "0")
        self.assertIn("/cache/models/llama:/cache/models/llama:ro",
                      [argv[i + 1] for i, a in enumerate(argv) if a == "-v"])
        cfg, spec = docker_spec("engine.model=meta-llama/Llama-3.1-8B")
        self.assertNotIn("meta-llama/Llama-3.1-8B:meta-llama/Llama-3.1-8B:ro",
                         launcher.docker_argv(spec, cfg, "0"))

    def test_missing_image_is_a_config_error(self):
        cfg, spec = docker_spec("engine.image=null")
        with self.assertRaisesRegex(ConfigError, "needs engine.image"):
            launcher.docker_argv(spec, cfg, "0")


class DockerStart(unittest.TestCase):
    """start/stop with backend docker; subprocess is mocked, no daemon runs."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def run_mock(self, docker_rc=0):
        def run(argv, **kw):
            if argv[:2] == ["docker", "run"]:
                return mock.Mock(returncode=docker_rc, stdout="c0ffee\n", stderr="no such image")
            return mock.Mock(returncode=0, stdout="", stderr="")
        return mock.patch.object(launcher.subprocess, "run", side_effect=run)

    def start(self, cfg, spec):
        return launcher.start(spec, self.dir / "run", backend="docker", cfg=cfg, gpus="0")

    def test_start_logs_engine_json_and_stop(self):
        cfg, spec = docker_spec()
        proc = mock.Mock(**{"poll.return_value": None})
        with self.run_mock() as run, \
                mock.patch.object(launcher.subprocess, "Popen", return_value=proc) as popen, \
                mock.patch.object(launcher, "wait_healthy") as healthy:
            handle = self.start(cfg, spec)
            self.assertEqual(run.call_args_list[0].args[0], launcher.docker_argv(spec, cfg, "0"))
            self.assertEqual(popen.call_args.args[0],
                             ["docker", "logs", "-f", "qbench-llama-tp1-docker"])
            self.assertEqual(popen.call_args.kwargs["stdout"].name,
                             str(self.dir / "run" / "server.log"))
            healthy.assert_called_once_with(proc, "http://127.0.0.1:8000/health", 1800, 1.0)
            record = json.loads((self.dir / "run" / "engine.json").read_text())
            self.assertEqual((record["pid"], record["container"], record["image"]),
                             ("c0ffee", "qbench-llama-tp1-docker", "vllm/vllm-openai:v0.30.0"))
            self.assertEqual(handle.pid, "c0ffee")
            handle.stop(grace_s=5)
            run.assert_called_with(["docker", "stop", "-t", "5", "qbench-llama-tp1-docker"],
                                   capture_output=True, check=False)
            proc.wait.assert_called_once_with()

    def test_unhealthy_container_is_stopped(self):
        cfg, spec = docker_spec()
        with self.run_mock() as run, mock.patch.object(launcher.subprocess, "Popen"), \
                mock.patch.object(launcher, "wait_healthy", side_effect=TimeoutError("late")):
            with self.assertRaises(TimeoutError):
                self.start(cfg, spec)
        self.assertEqual(run.call_args.args[0][:2], ["docker", "stop"])

    def test_failed_docker_run_and_missing_image(self):
        cfg, spec = docker_spec()
        with self.run_mock(docker_rc=125), mock.patch.object(launcher.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "docker run exited with 125: no such image"):
                self.start(cfg, spec)
        popen.assert_not_called()
        cfg, spec = docker_spec("engine.image=null")
        with self.run_mock() as run, self.assertRaisesRegex(ConfigError, "needs engine.image"):
            self.start(cfg, spec)
        run.assert_not_called()


class NoProxy(unittest.TestCase):
    """An http_proxy in the environment never carries engine traffic."""

    def test_endpoints_are_reached_directly(self):
        dead_proxy = {"http_proxy": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9",
                      "no_proxy": "", "NO_PROXY": ""}
        with fake_endpoint.serve() as (endpoint, _), \
                mock.patch.dict(os.environ, dead_proxy):
            self.assertIn("process_open_fds", native.scrape_metrics(endpoint))
            with launcher.urlopen(endpoint + "/metrics", timeout=5) as resp:
                self.assertEqual(resp.status, 200)


class IdleGpus(unittest.TestCase):
    def test_non_cuda_takes_visible_devices_or_the_first_n(self):
        self.assertEqual(launcher.idle_gpus(3, "rocm"), "0,1,2")
        self.assertEqual(launcher.idle_gpus(2, "cpu", "4,5"), "4,5")

    def test_cuda_picks_gpus_under_1000_mib(self):
        smi = "0, 70000 MiB\n1, 3 MiB\n2, 999 MiB\n3, 1000 MiB\n"
        with mock.patch.object(launcher.subprocess, "run",
                               return_value=mock.Mock(stdout=smi)):
            self.assertEqual(launcher.idle_gpus(2, "cuda"), "1,2")
            with self.assertRaisesRegex(RuntimeError, "NEED 3 FREE GPUS, have: 1 2"):
                launcher.idle_gpus(3, "cuda")


if __name__ == "__main__":
    unittest.main()
