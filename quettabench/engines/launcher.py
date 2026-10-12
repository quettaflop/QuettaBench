"""Start an engine server from a LaunchSpec, locally in its own process group or detached in docker, wait for its health check, write results/<run>/engine.json, and stop it with every child.

A container is not --rm, so a crashed engine's log stays readable. The docker path has not run against a real daemon.
"""

import glob
import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from quettabench.config import env_map
from quettabench.config.schema import ConfigError
from quettabench.metrics.results import redact, secret

REPO = Path(__file__).resolve().parents[2]

# Engine endpoints are local or cluster-private; an http_proxy would route them through the proxy, which answers 403.
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def urlopen(url, **kwargs):
    """urllib.request.urlopen(url, **kwargs) without any proxy."""
    return _DIRECT.open(url, **kwargs)


@dataclass
class Handle:
    adapter: str
    endpoint: str
    pid: int  # the server, or for docker the `docker logs -f` stream
    results_dir: Path
    proc: subprocess.Popen
    gpu: str = ""
    container: str = ""
    container_id: str = ""
    backend: str = "cuda"

    def stop(self, grace_s=30):
        """SIGTERM the process group and SIGKILL what is left after grace_s; a container gets `docker stop`, then its log stream is ended and the container removed."""
        if self.container:
            subprocess.run(["docker", "stop", "-t", str(grace_s), self.container],
                           capture_output=True, check=False)
            try:
                self.proc.wait(timeout=grace_s + 10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            remove(self.container)
            return
        deadline = time.monotonic() + grace_s
        if self._signal_group(signal.SIGTERM):
            while time.monotonic() < deadline and self._signal_group(0):
                self.proc.poll()  # a zombie leader would still count
                time.sleep(0.1)
            self._signal_group(signal.SIGKILL)
        self.proc.wait()

    def _signal_group(self, sig):
        """Send sig to the group; False once no member is left."""
        try:
            os.killpg(self.pid, sig)
        except ProcessLookupError:
            return False
        return True


@dataclass
class NoServer:
    """Handle for a driver that starts the engine itself (in-process vLLM, the cargo bench): no endpoint, nothing to stop."""
    results_dir: Path
    endpoint: str | None = None
    backend: str = "cuda"

    def stop(self):
        pass


def visible_env(backend, devices):
    """The env that limits a vLLM server to devices, and the inherited names to unset."""
    if backend == "rocm":
        # vLLM reads CUDA_VISIBLE_DEVICES on ROCm and refuses a HIP value that differs (vllm/platforms/rocm.py:127-145,496)
        return {"HIP_VISIBLE_DEVICES": devices, "CUDA_VISIBLE_DEVICES": devices}, ()
    if backend == "tpu":
        # tpu-inference examples/external_lb/run_dp_external_lb.sh:203-205 (v0.30.0)
        chips = len(devices.split(","))
        return {"TPU_VISIBLE_CHIPS": devices, "TPU_CHIPS_PER_PROCESS_BOUNDS": f"1,{chips},1",
                "TPU_PROCESS_BOUNDS": "1,1,1"}, ()
    if backend == "neuron":
        # vllm-neuron picks cores from NEURON_VISIBLE_DEVICES and refuses NEURON_RT_VISIBLE_CORES (neuron_worker.py:527-532)
        return {"NEURON_VISIBLE_DEVICES": devices}, ("NEURON_RT_VISIBLE_CORES",)
    return {"CUDA_VISIBLE_DEVICES": devices}, ()


def device_flags(backend, gpus):
    """docker run flags that give the container the accelerator, from each vendor's vLLM container docs."""
    if backend == "rocm":
        # vllm docs/getting_started/installation/gpu.rocm.inc.md:359-370 (v0.30.0)
        return ["--group-add=video", "--cap-add=SYS_PTRACE", "--security-opt", "seccomp=unconfined",
                "--device", "/dev/kfd", "--device", "/dev/dri"]
    if backend == "tpu":
        # tpu-inference docs/mkdocs/javascripts/interactive.js:26 (v0.30.0)
        return ["--privileged", "--net=host"]
    if backend == "neuron":
        # awsdocs-neuron v2.32.0 deploy/environments/quickstart-deploy-dlc.rst:66-84
        devices = sorted(glob.glob("/dev/neuron[0-9]*"), key=lambda d: int(d[len("/dev/neuron"):]))
        if not devices:
            raise ConfigError("hardware.backend=neuron: no /dev/neuron* device on this host")
        return [*(f"--device={d}" for d in devices), "--cap-add", "SYS_ADMIN", "--cap-add", "IPC_LOCK"]
    return ["--gpus", f'"device={gpus}"']


def idle_gpus(n, backend):
    """On cuda the first n GPUs using under 1000 MiB; other backends have no busy check and take 0..n-1."""
    if backend != "cuda":
        return ",".join(str(i) for i in range(n))
    rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
        capture_output=True, text=True, check=True).stdout.splitlines()
    ids = [index for index, used in (row.split(", ") for row in rows if row)
           if float(used.split()[0]) < 1000]
    if len(ids) < n:
        raise RuntimeError(f"NEED {n} FREE GPUS, have: {' '.join(ids) or 'none'}")
    return ",".join(ids[:n])


NVIDIA_SMI = ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
# backend -> (tool argv, the line prefix that carries the name)
IDENTITY = {
    "rocm": (["amd-smi", "static", "-g", "0", "-a"], "MARKET_NAME:"),  # vllm .buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh:44
    "neuron": (["neuron-ls"], "instance-type:"),  # awsdocs-neuron tools/neuron-sys-tools/neuron-ls.rst:45
}


def tool_lines(argv):
    """The non-empty stdout lines of a tool; none when it is missing or fails."""
    try:
        out = subprocess.run(argv, capture_output=True, text=True, check=False).stdout
    except OSError:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def accelerator(backend="cuda"):
    """(name, total bytes) of device 0: nvidia-smi on cuda, the name only from amd-smi or neuron-ls; ("", None) when unavailable, never raises."""
    if backend in IDENTITY:
        argv, prefix = IDENTITY[backend]
        names = [line[len(prefix):].strip() for line in tool_lines(argv) if line.startswith(prefix)]
        return (names[0] if names else ""), None
    if backend in ("tpu", "cpu"):
        return "", None
    lines = tool_lines(NVIDIA_SMI)
    if not lines or ", " not in lines[0]:
        return "", None
    name, mib = lines[0].rsplit(", ", 1)
    return name, int(float(mib)) * 2**20


def wait_healthy(proc, url, timeout_s, poll_s):
    """Poll url until it answers 200; fail fast if the server exits first."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode} before {url} answered")
        try:
            with urlopen(url, timeout=poll_s) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(poll_s)
    raise TimeoutError(f"{url} not healthy after {timeout_s}s")


def remove(container):
    """`docker rm -f`: the container is gone whatever state it was in."""
    subprocess.run(["docker", "rm", "-f", container], capture_output=True, check=False)


def container_name(run_name):
    """qbench-<run_name>-<pid>-<6 random hex> with characters docker refuses replaced by -, so two runs of one yaml never collide."""
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "-", run_name)
    return f"qbench-{safe}-{os.getpid()}-{os.urandom(3).hex()}"


def docker_argv(spec, cfg, gpus, results_dir):
    """`docker run -d` of spec.server_argv in spec.image on the given GPUs.

    --pull=never makes a missing image fail here instead of downloading; --init keeps the engine off PID 1 so its workers are reaped. The repo, HF cache and model dir mount read-only at their host paths so model paths resolve unchanged; the results dir mounts read-write so nsys output outlives the container.
    --entrypoint is the argv's first word, so the image's ENTRYPOINT does not prefix it. A secret-named env var is passed as `-e NAME` alone, so docker reads its value from the env start() gives docker run.
    """
    hw, run_dir = cfg.hardware, Path(results_dir).resolve()
    if not spec.image:
        raise ConfigError(f"{cfg.run_name}: hardware.launcher=docker needs engine.image")
    mounts = [hw.hf_home] if hw.hf_home else []
    model = spec.engine_kwargs.get("model") or cfg.engine.model
    if model and Path(model).is_absolute() and not any(
            Path(model).is_relative_to(m) for m in [*mounts, REPO]):
        mounts.append(model)
    env = {**spec.env, **{k: v for k, v in hw.collective.items() if v},
           **({"HF_HOME": hw.hf_home} if hw.hf_home else {})}
    hidden = docker_secrets(spec)
    devices = device_flags(hw.backend, gpus)
    ports = [] if "--net=host" in devices else ["-p", f"{spec.port}:{spec.port}"]
    return ["docker", "run", "-d", "--pull=never", "--init",
            "--name", container_name(cfg.run_name),
            *devices, "--ipc=host", f"--shm-size={hw.shm_size}", *ports,
            *(a for path in [*mounts, REPO] for a in ("-v", f"{path}:{path}:ro")),
            "-v", f"{run_dir}:{run_dir}",
            *(a for k, v in env.items() for a in ("-e", k if k in hidden else f"{k}={v}")),
            "--entrypoint", spec.server_argv[0], spec.image, *spec.server_argv[1:]]


def docker_secrets(spec):
    """The secret-named env of a docker launch (VLLM_API_KEY), passed by name only."""
    return {k: v for k, v in spec.env.items() if secret(k)}


def start(spec, results_dir, *, backend="local", cfg=None, gpus="", timeout_s=1800,
          poll_s=1.0):
    """Launch spec.server_argv, wait for health, write engine.json; return a Handle. backend docker runs it in a container built by docker_argv(spec, cfg, gpus, results_dir)."""
    if not spec.server_argv:
        raise ValueError(f"{spec.adapter}: the spec has no server to launch")
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    container = container_id = ""
    if backend == "docker":
        argv = docker_argv(spec, cfg, gpus, results_dir)
        container = argv[argv.index("--name") + 1]
        secrets = docker_secrets(spec)
        run = subprocess.run(argv, capture_output=True, text=True, check=False,
                             env=env_map.with_env(secrets) if secrets else None)
        if run.returncode != 0:
            # The name is this run's own (container_name), so whatever docker left under it is logged and removed.
            with (results_dir / "server.log").open("w") as log:
                log.write(run.stderr)
                log.flush()
                subprocess.run(["docker", "logs", container], stdout=log,
                               stderr=subprocess.STDOUT, check=False)
            remove(container)
            raise RuntimeError(f"docker run exited with {run.returncode}: {run.stderr.strip()}")
        container_id, server = run.stdout.strip(), ["docker", "logs", "-f", container]
    else:
        server = spec.child_argv(spec.server_argv)
    try:
        with (results_dir / "server.log").open("w") as log:
            proc = subprocess.Popen(server, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
    except BaseException:
        if container:
            remove(container)
        raise
    endpoint = f"http://127.0.0.1:{spec.port}"
    accel = cfg.hardware.backend if cfg else "cuda"
    handle = Handle(spec.adapter, endpoint, proc.pid, results_dir, proc, accelerator(accel)[0],
                    container, container_id, accel)
    try:
        wait_healthy(proc, endpoint + spec.health_path, timeout_s, poll_s)
        record = {
            "adapter": spec.adapter,
            "endpoint": endpoint,
            "pid": proc.pid,
            "launch_spec": redact(asdict(spec)),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "instrumentation": spec.instrumentation,
        }
        if container:
            record.update(container=container, container_id=container_id, image=spec.image)
        (results_dir / "engine.json").write_text(json.dumps(record, indent=2, sort_keys=True)
                                                 + "\n")
    except BaseException:
        handle.stop()
        raise
    return handle
