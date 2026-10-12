"""start/health/engine.json/stop on a fake server, and no child survives stop."""

import dataclasses
import json
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from quettabench import provision, runner
from quettabench.config import loader
from quettabench.config.schema import ConfigError
from quettabench.engines import drivers, launcher, vllm
from quettabench.engines.base import LaunchSpec, UnsupportedField
from quettabench.profiling import native
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
    except FileNotFoundError:  # reaped between the kill and the read
        return False
    return not stat.split(") ")[1].startswith("Z")


class LocalLauncher(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.state = self.dir / "state.json"
        self.port = free_port()

    def start(self, **kw):
        """launcher.start of the fake server; a port taken between free_port
        and the bind (the server exits 98) is retried on a fresh one, and the
        server that answered must be ours."""
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
        return LaunchSpec(adapter="fake", port=self.port, health_path="/health",
                          server_argv=(sys.executable, str(FAKE), str(self.port), str(self.state)),
                          env={"FAKE_MARK": "from-spec", "HF_TOKEN": "secret",
                               "DB_SECRET": "secret"}, **kw)

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


def keyed_docker_spec(key):
    """docker_spec with engine.extra.vllm.api_key set."""
    cfg, _ = docker_spec()
    extra = {**cfg.engine.extra, "vllm": {**cfg.engine.extra.get("vllm", {}), "api_key": key}}
    cfg = dataclasses.replace(cfg, engine=dataclasses.replace(cfg.engine, extra=extra))
    return cfg, vllm.ADAPTER.translate(cfg)


NAME = "qbench-llama-tp1-docker-77-abcdef"


def fixed_name(case):
    """Pin container_name's pid and random part for the rest of the test."""
    for name, value in (("getpid", 77), ("urandom", b"\xab\xcd\xef")):
        patch = mock.patch.object(launcher.os, name, return_value=value)
        patch.start()
        case.addCleanup(patch.stop)


class DockerArgv(unittest.TestCase):
    def setUp(self):
        fixed_name(self)

    def test_example_yaml_argv(self):
        cfg, spec = docker_spec()
        repo = str(launcher.REPO)
        self.assertEqual(launcher.docker_argv(spec, cfg, "2", "/r/run"), [
            "docker", "run", "-d", "--pull=never", "--init", "--name", NAME,
            "--gpus", '"device=2"', "--ipc=host", "--shm-size=16g", "-p", "8000:8000",
            "-v", "/hf:/hf:ro", "-v", f"{repo}:{repo}:ro", "-v", "/r/run:/r/run",
            "-e", "VLLM_SERVER_DEV_MODE=1", "-e", "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1",
            "-e", "NCCL_IB_DISABLE=1", "-e", "HF_HOME=/hf",
            "--entrypoint", "vllm", "vllm/vllm-openai:v0.26.0",
            "serve", "/hf/llama", "--port", "8000", "--tensor-parallel-size", "1",
            "--dtype", "float16", "--max-model-len", "9216", "--gpu-memory-utilization", "0.9",
            "--no-enable-prefix-caching", "--enable-chunked-prefill", "--trust-remote-code"])

    def test_shm_size_and_collective_pins_come_from_hardware(self):
        cfg, spec = docker_spec("hardware.shm_size=64g", "hardware.link_profile=pcie")
        argv = launcher.docker_argv(spec, cfg, "0,1", "/r/run")
        self.assertIn("--shm-size=64g", argv)
        self.assertIn('"device=0,1"', argv)
        self.assertIn("NCCL_PROTO=Simple", argv)

    def test_provisioned_model_dir_is_mounted(self):
        cfg, spec = docker_spec("engine.model=/cache/models/llama")
        argv = launcher.docker_argv(spec, cfg, "0", "/r/run")
        self.assertIn("/cache/models/llama:/cache/models/llama:ro",
                      [argv[i + 1] for i, a in enumerate(argv) if a == "-v"])
        cfg, spec = docker_spec("engine.model=meta-llama/Llama-3.1-8B")
        self.assertNotIn("meta-llama/Llama-3.1-8B:meta-llama/Llama-3.1-8B:ro",
                         launcher.docker_argv(spec, cfg, "0", "/r/run"))

    def test_a_secret_env_goes_by_name_only(self):
        """docker run -e NAME (no value) takes NAME from the docker client's
        env, so the key is never in an argv that ps or the dry run shows."""
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
    """start/stop with backend docker; subprocess is mocked, no daemon runs."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        fixed_name(self)

    def run_mock(self, docker_rc=0, created="c0ffee\n"):
        def run(argv, **kw):
            if argv[:2] == ["docker", "run"]:
                return mock.Mock(returncode=docker_rc, stdout=created, stderr="no such image")
            return mock.Mock(returncode=0, stdout="", stderr="")
        return mock.patch.object(launcher.subprocess, "run", side_effect=run)

    @staticmethod
    def docker_calls(run):
        """The docker commands run after `docker run`."""
        return [c.args[0] for c in run.call_args_list if c.args[0][0] == "docker"][1:]

    def start(self, cfg, spec):
        return launcher.start(spec, self.dir / "run", backend="docker", cfg=cfg, gpus="0")

    def test_start_logs_engine_json_and_stop(self):
        cfg, spec = docker_spec()
        proc = mock.Mock(pid=4242, **{"poll.return_value": None})
        with self.run_mock() as run, \
                mock.patch.object(launcher.subprocess, "Popen", return_value=proc) as popen, \
                mock.patch.object(launcher, "wait_healthy") as healthy:
            handle = self.start(cfg, spec)
            self.assertEqual(run.call_args_list[0].args[0],
                             launcher.docker_argv(spec, cfg, "0", self.dir / "run"))
            self.assertEqual(popen.call_args.args[0],
                             ["docker", "logs", "-f", NAME])
            self.assertEqual(popen.call_args.kwargs["stdout"].name,
                             str(self.dir / "run" / "server.log"))
            healthy.assert_called_once_with(proc, "http://127.0.0.1:8000/health", 1800, 1.0)
            record = json.loads((self.dir / "run" / "engine.json").read_text())
            self.assertEqual((record["pid"], record["container"], record["container_id"],
                              record["image"]),
                             (proc.pid, NAME, "c0ffee",
                              "vllm/vllm-openai:v0.26.0"))
            self.assertEqual((handle.pid, handle.container_id), (proc.pid, "c0ffee"))
            handle.stop(grace_s=5)
            self.assertEqual(self.docker_calls(run),
                             [["docker", "stop", "-t", "5", NAME],
                              ["docker", "rm", "-f", NAME]])
            proc.wait.assert_called_once_with(timeout=15)

    def test_docker_run_gets_the_secret_in_its_env(self):
        cfg, spec = keyed_docker_spec("sk-live")
        proc = mock.Mock(pid=4242, **{"poll.return_value": None})
        with self.run_mock() as run, mock.patch.object(launcher.subprocess, "Popen",
                                                       return_value=proc), \
                mock.patch.object(launcher, "wait_healthy"):
            self.start(cfg, spec)
        docker_run = run.call_args_list[0]
        self.assertEqual(docker_run.kwargs["env"]["VLLM_API_KEY"], "sk-live")
        self.assertNotIn("sk-live", " ".join(docker_run.args[0]))
        self.assertNotIn("sk-live", (self.dir / "run" / "engine.json").read_text())

    def test_stop_kills_a_log_stream_that_does_not_end(self):
        cfg, spec = docker_spec()
        proc = mock.Mock(pid=4242, **{"wait.side_effect": [launcher.subprocess.TimeoutExpired("logs", 15),
                                                 0]})
        with self.run_mock() as run, \
                mock.patch.object(launcher.subprocess, "Popen", return_value=proc), \
                mock.patch.object(launcher, "wait_healthy"):
            self.start(cfg, spec).stop(grace_s=5)
        proc.kill.assert_called_once_with()
        self.assertEqual(run.call_args.args[0], ["docker", "rm", "-f", NAME])

    def test_container_name_is_made_docker_safe_and_unique(self):
        self.assertEqual(launcher.container_name("a/b c:d_e.f-g"), "qbench-a-b-c-d_e.f-g-77-abcdef")
        mock.patch.stopall()
        self.assertNotEqual(launcher.container_name("r"), launcher.container_name("r"))

    def test_unhealthy_container_is_stopped(self):
        cfg, spec = docker_spec()
        with self.run_mock() as run, mock.patch.object(launcher.subprocess, "Popen"), \
                mock.patch.object(launcher, "wait_healthy", side_effect=TimeoutError("late")):
            with self.assertRaises(TimeoutError):
                self.start(cfg, spec)
        self.assertEqual([argv[:2] for argv in self.docker_calls(run)],
                         [["docker", "stop"], ["docker", "rm"]])

    def test_failed_docker_run_and_missing_image(self):
        cfg, spec = docker_spec()
        with self.run_mock(docker_rc=125) as run, \
                mock.patch.object(launcher.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "docker run exited with 125: no such image"):
                self.start(cfg, spec)
        popen.assert_not_called()
        self.assertEqual(self.docker_calls(run),
                         [["docker", "logs", NAME], ["docker", "rm", "-f", NAME]])
        self.assertEqual(run.call_args_list[1].kwargs["stdout"].name,
                         str(self.dir / "run" / "server.log"))
        self.assertEqual((self.dir / "run" / "server.log").read_text(), "no such image")

    def test_a_docker_run_that_printed_no_id_is_still_cleaned_by_name(self):
        cfg, spec = docker_spec()
        with self.run_mock(docker_rc=125, created="") as run, \
                self.assertRaisesRegex(RuntimeError, "docker run exited with 125"):
            self.start(cfg, spec)
        self.assertEqual(self.docker_calls(run),
                         [["docker", "logs", NAME], ["docker", "rm", "-f", NAME]])
        cfg, spec = docker_spec("engine.image=null")
        with self.run_mock() as run, self.assertRaisesRegex(ConfigError, "needs engine.image"):
            self.start(cfg, spec)
        run.assert_not_called()


class NoProxy(unittest.TestCase):
    """An http_proxy in the environment never carries engine traffic."""

    def test_endpoints_are_reached_directly(self):
        dead_proxy = {"http_proxy": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9",
                      "no_proxy": "", "NO_PROXY": ""}
        with helpers.serve() as (endpoint, _), \
                mock.patch.dict(os.environ, dead_proxy):
            self.assertIn("process_open_fds", native.scrape_metrics(endpoint))
            with launcher.urlopen(endpoint + "/metrics", timeout=5) as resp:
                self.assertEqual(resp.status, 200)


class IdleGpus(unittest.TestCase):
    def test_non_cuda_takes_the_first_n(self):
        self.assertEqual(launcher.idle_gpus(3, "rocm"), "0,1,2")
        self.assertEqual(launcher.idle_gpus(2, "cpu"), "0,1")

    def test_cuda_picks_gpus_under_1000_mib(self):
        smi = "0, 70000 MiB\n1, 3 MiB\n2, 999 MiB\n3, 1000 MiB\n"
        with mock.patch.object(launcher.subprocess, "run",
                               return_value=mock.Mock(stdout=smi)):
            self.assertEqual(launcher.idle_gpus(2, "cuda"), "1,2")
            with self.assertRaisesRegex(RuntimeError, "NEED 3 FREE GPUS, have: 1 2"):
                launcher.idle_gpus(3, "cuda")


def tools_on_path(td, **outputs):
    """A PATH holding one stand-in script per tool name that prints its output."""
    for tool, out in outputs.items():
        script = Path(td, tool)
        script.write_text(f"#!/bin/sh\nprintf '{out}'\n")
        script.chmod(0o755)
    return mock.patch.dict(os.environ, {"PATH": td})


class Backends(unittest.TestCase):
    def test_visible_env_per_backend(self):
        self.assertEqual(launcher.visible_env("cuda", "0,1"), ({"CUDA_VISIBLE_DEVICES": "0,1"}, ()))
        self.assertEqual(launcher.visible_env("rocm", "2"),
                         ({"HIP_VISIBLE_DEVICES": "2", "CUDA_VISIBLE_DEVICES": "2"}, ()))
        self.assertEqual(launcher.visible_env("tpu", "0,1,2,3"),
                         ({"TPU_VISIBLE_CHIPS": "0,1,2,3", "TPU_CHIPS_PER_PROCESS_BOUNDS": "1,4,1",
                           "TPU_PROCESS_BOUNDS": "1,1,1"}, ()))
        self.assertEqual(launcher.visible_env("neuron", "0,1"),
                         ({"NEURON_VISIBLE_DEVICES": "0,1"}, ("NEURON_RT_VISIBLE_CORES",)))

    def test_rocm_is_detected_from_rocm_smi_or_amd_smi(self):
        for tool in ("rocm-smi", "amd-smi"):
            with self.subTest(tool=tool), tempfile.TemporaryDirectory() as td, \
                    tools_on_path(td, **{tool: ""}):
                self.assertEqual(loader.detect_backend(None), "rocm")

    def child_spec(self, backend, *sets):
        """The spec runner.serve_one hands launcher.start for a serve run on backend."""
        (cfg,) = loader.load(DOCKER_YAML, ["engine.model=/w", f"hardware.backend={backend}",
                                           "engine.dtype=bfloat16", "mode.grid=null",
                                           "hardware.visible_devices=0,1", *sets], env={})
        with mock.patch.object(runner.launcher, "start") as start, \
                mock.patch.object(runner, "build_trace"), mock.patch.object(runner.serve, "run"), \
                tempfile.TemporaryDirectory() as td:
            runner.serve_one(cfg, Path(td), set(), mock.Mock(provenance={}))
        return start.call_args.args[0]

    def test_the_server_gets_its_backends_devices(self):
        for backend, launch in (("rocm", "local"), ("tpu", "docker"), ("neuron", "local")):
            with self.subTest(backend=backend):
                spec = self.child_spec(backend, f"hardware.launcher={launch}")
                want, unset = launcher.visible_env(backend, "0,1")
                self.assertEqual({k: spec.env[k] for k in want}, want)
                self.assertEqual(set(unset) <= set(spec.env_drop), True)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", self.child_spec("cuda", "hardware.launcher=docker").env)

    def test_docker_device_flags_per_backend(self):
        fixed_name(self)
        flags = {"rocm": ["--group-add=video", "--cap-add=SYS_PTRACE", "--security-opt",
                          "seccomp=unconfined", "--device", "/dev/kfd", "--device", "/dev/dri"],
                 "tpu": ["--privileged", "--net=host"],
                 "neuron": ["--device=/dev/neuron0", "--device=/dev/neuron1", "--device=/dev/neuron10",
                            "--cap-add", "SYS_ADMIN", "--cap-add", "IPC_LOCK"]}
        devices = ["/dev/neuron10", "/dev/neuron1", "/dev/neuron0"]
        for backend, want in flags.items():
            with self.subTest(backend=backend), \
                    mock.patch.object(launcher.glob, "glob", return_value=devices):
                cfg, spec = docker_spec(f"hardware.backend={backend}", "engine.dtype=bfloat16")
                argv = launcher.docker_argv(spec, cfg, "0,1", "/r/run")
                self.assertEqual(argv[7:7 + len(want)], want)
                self.assertNotIn("--gpus", argv)
                self.assertEqual("-p" in argv, backend != "tpu")
        cfg, spec = docker_spec("hardware.backend=neuron")
        with mock.patch.object(launcher.glob, "glob", return_value=[]), \
                self.assertRaisesRegex(ConfigError, "no /dev/neuron"):
            launcher.docker_argv(spec, cfg, "0", "/r/run")

    def test_accelerator_identity_per_backend(self):
        with tempfile.TemporaryDirectory() as td, tools_on_path(
                td, **{"nvidia-smi": "NVIDIA H200, 143771\n",
                       "amd-smi": "ASIC:\n    MARKET_NAME: AMD Instinct MI300X\n",
                       "neuron-ls": "instance-type: trn2.48xlarge\ninstance-id: i-0\n"}):
            self.assertEqual(launcher.accelerator("cuda"), ("NVIDIA H200", 143771 << 20))
            self.assertEqual(launcher.accelerator("rocm"), ("AMD Instinct MI300X", None))
            self.assertEqual(launcher.accelerator("neuron"), ("trn2.48xlarge", None))
            self.assertEqual(launcher.accelerator("tpu"), ("", None))
        with tempfile.TemporaryDirectory() as td, tools_on_path(td):
            for backend in ("cuda", "rocm", "neuron", "tpu", "cpu"):
                self.assertEqual(launcher.accelerator(backend), ("", None))
        handle = launcher.NoServer(Path("r"), endpoint="http://127.0.0.1:1", backend="rocm")
        (cfg,) = loader.load(DOCKER_YAML, ["engine.model=/w", "hardware.backend=rocm",
                                           "mode.grid=null"], env={})
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


if __name__ == "__main__":
    unittest.main()
