"""Start an engine server from a LaunchSpec, locally or in docker; stop it with its children."""

import glob
import json
import os
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from quettabench.config import env_map
from quettabench.config.schema import ConfigError
from quettabench.metrics.results import redact, secret

REPO = Path(__file__).resolve().parents[2]
SERVED_NAME_FLAG = "--served-model-name"
FOREIGN_GRACE_S = 10

# http_proxy would route local or cluster-private endpoints through a proxy that answers 403.
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def urlopen(url, **kwargs):
    return _DIRECT.open(url, **kwargs)


@dataclass
class Handle:
    adapter: str
    endpoint: str
    pid: int
    results_dir: Path
    proc: subprocess.Popen
    gpu: str = ""
    container: str = ""
    container_id: str = ""
    backend: str = "cuda"

    def stop(self, grace_s=30):
        """SIGKILL what is left after grace_s; a container gets `docker stop` and removal."""
        if self.container:
            subprocess.run(["docker", "stop", "-t", str(grace_s), self.container], capture_output=True, check=False)
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
                self.proc.poll()  # A zombie leader would still count.
                time.sleep(0.1)
            self._signal_group(signal.SIGKILL)
        self.proc.wait()

    def _signal_group(self, sig):
        try:
            os.killpg(self.pid, sig)
        except ProcessLookupError:
            return False
        return True


@dataclass
class NoServer:
    """Handle for a driver that starts the engine itself: nothing to stop."""

    results_dir: Path
    endpoint: str | None = None
    backend: str = "cuda"

    def stop(self):
        pass


def visible_env(backend, devices):
    """The env that limits a server to devices, and the inherited names to unset."""
    if backend == "rocm":
        # vLLM reads CUDA_VISIBLE_DEVICES on ROCm and refuses a differing HIP value.
        return {"HIP_VISIBLE_DEVICES": devices, "CUDA_VISIBLE_DEVICES": devices}, ()
    if backend == "tpu":
        # One process per chip group, like tpu-inference's external-LB example.
        chips = len(devices.split(","))
        return {
            "TPU_VISIBLE_CHIPS": devices,
            "TPU_CHIPS_PER_PROCESS_BOUNDS": f"1,{chips},1",
            "TPU_PROCESS_BOUNDS": "1,1,1",
        }, ()
    if backend == "neuron":
        # vllm-neuron picks cores from NEURON_VISIBLE_DEVICES and refuses NEURON_RT_VISIBLE_CORES.
        return {"NEURON_VISIBLE_DEVICES": devices}, ("NEURON_RT_VISIBLE_CORES",)
    return {"CUDA_VISIBLE_DEVICES": devices}, ()


def device_flags(backend, gpus):
    """From each vendor's vLLM container docs."""
    if backend == "rocm":
        # vLLM's ROCm install guide (v0.30.0)
        return [
            "--group-add=video",
            "--cap-add=SYS_PTRACE",
            "--security-opt",
            "seccomp=unconfined",
            "--device",
            "/dev/kfd",
            "--device",
            "/dev/dri",
        ]
    if backend == "tpu":
        # tpu-inference's container quickstart (v0.30.0)
        return ["--privileged", "--net=host"]
    if backend == "neuron":
        # the Neuron DLC quickstart (SDK 2.32)
        devices = sorted(glob.glob("/dev/neuron[0-9]*"), key=lambda d: int(d[len("/dev/neuron") :]))
        if not devices:
            raise ConfigError("hardware.backend=neuron: no /dev/neuron* device on this host")
        return [*(f"--device={d}" for d in devices), "--cap-add", "SYS_ADMIN", "--cap-add", "IPC_LOCK"]
    return ["--gpus", f'"device={gpus}"']


def idle_gpus(n, backend):
    """Other backends have no busy check and take 0..n-1."""
    if backend != "cuda":
        return ",".join(str(i) for i in range(n))
    rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    ids = [index for index, used in (row.split(", ") for row in rows if row) if float(used.split()[0]) < 1000]
    if len(ids) < n:
        raise RuntimeError(f"NEED {n} FREE GPUS, have: {' '.join(ids) or 'none'}")
    return ",".join(ids[:n])


NVIDIA_SMI = ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
# backend -> (tool argv, the line prefix that carries the name)
IDENTITY = {
    "rocm": (
        ["amd-smi", "static", "-g", "0", "-a"],
        "MARKET_NAME:",
    ),  # how vLLM's own benchmark scripts read the card name
    "neuron": (["neuron-ls"], "instance-type:"),
}


def tool_lines(argv):
    """The non-empty stdout lines of a tool; none when it is missing or fails."""
    try:
        out = subprocess.run(argv, capture_output=True, text=True, check=False).stdout
    except OSError:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def accelerator(backend="cuda"):
    """("", None) when unavailable; never raises."""
    if backend in IDENTITY:
        argv, prefix = IDENTITY[backend]
        names = [line[len(prefix) :].strip() for line in tool_lines(argv) if line.startswith(prefix)]
        return (names[0] if names else ""), None
    if backend in ("tpu", "cpu"):
        return "", None
    lines = tool_lines(NVIDIA_SMI)
    if not lines or ", " not in lines[0]:
        return "", None
    name, mib = lines[0].rsplit(", ", 1)
    return name, int(float(mib)) * 2**20


def port_taken(port):
    """True when something already accepts TCP connections where qbench will connect."""
    try:
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
    except OSError:
        return False
    return True


def tagged(spec, token):
    """The spec whose server also answers to token as a model name; vLLM keeps the first name for requests."""
    argv = list(spec.server_argv)
    if SERVED_NAME_FLAG in argv:
        argv.insert(argv.index(SERVED_NAME_FLAG) + 2, token)
    else:
        argv += [SERVED_NAME_FLAG, spec.served_name, token]
    return replace(spec, server_argv=tuple(argv))


def lists_token(models_url, token, headers, timeout):
    """True when /v1/models names token: only the server this launch started does."""
    try:
        with urlopen(urllib.request.Request(models_url, headers=headers), timeout=timeout) as resp:
            return token in [m.get("id") for m in json.load(resp).get("data", [])]
    except (urllib.error.URLError, ConnectionError, TimeoutError, ValueError, AttributeError):
        return False


def wait_healthy(proc, url, timeout_s, poll_s, owns=None):
    """With owns, a 200 only counts once owns() is true. A server that answers but is not ours is waited on
    for FOREIGN_GRACE_S (our own port bind may just be late), then refused instead of waiting out timeout_s."""
    deadline = time.monotonic() + timeout_s
    foreign_since = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode} before {url} answered")
        try:
            with urlopen(url, timeout=poll_s) as resp:
                if resp.status == 200:
                    if owns is None or owns():
                        return
                    foreign_since = foreign_since or time.monotonic()
                    if time.monotonic() - foreign_since > FOREIGN_GRACE_S:
                        raise RuntimeError(f"another server answers on {url}; ours is still starting")
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(poll_s)
    raise TimeoutError(f"{url} not healthy after {timeout_s}s")


def remove(container):
    subprocess.run(["docker", "rm", "-f", container], capture_output=True, check=False)


def container_name(run_name):
    """Random suffix so two runs of one yaml never collide."""
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "-", run_name)
    return f"qbench-{safe}-{os.getpid()}-{os.urandom(3).hex()}"


def docker_argv(spec, cfg, gpus, results_dir):
    hw, run_dir = cfg.hardware, Path(results_dir).resolve()
    if not spec.image:
        raise ConfigError(f"{cfg.run_name}: hardware.launcher=docker needs engine.image")
    mounts = [hw.hf_home] if hw.hf_home else []
    model = spec.engine_kwargs.get("model") or cfg.engine.model
    if model and Path(model).is_absolute() and not any(Path(model).is_relative_to(m) for m in [*mounts, REPO]):
        mounts.append(model)
    env = {
        **spec.env,
        **{k: v for k, v in hw.collective.items() if v},
        **({"HF_HOME": hw.hf_home} if hw.hf_home else {}),
    }
    hidden = docker_secrets(spec)
    devices = device_flags(hw.backend, gpus)
    ports = [] if "--net=host" in devices else ["-p", f"{spec.port}:{spec.port}"]
    return [
        "docker",
        "run",
        "-d",
        "--pull=never",  # A missing image fails here instead of downloading.
        "--init",
        "--name",
        container_name(cfg.run_name),
        *devices,
        "--ipc=host",
        f"--shm-size={hw.shm_size}",
        *ports,
        *(a for path in [*mounts, REPO] for a in ("-v", f"{path}:{path}:ro")),
        "-v",
        f"{run_dir}:{run_dir}",  # Read-write so nsys output outlives the container.
        *(a for k, v in env.items() for a in ("-e", k if k in hidden else f"{k}={v}")),
        "--entrypoint",  # The argv's first word, so the image ENTRYPOINT does not prefix it.
        spec.server_argv[0],
        spec.image,
        *spec.server_argv[1:],
    ]


def docker_secrets(spec):
    return {k: v for k, v in spec.env.items() if secret(k)}


def start(spec, results_dir, *, backend="local", cfg=None, gpus="", timeout_s=1800, poll_s=1.0):
    if not spec.server_argv:
        raise ValueError(f"{spec.adapter}: the spec has no server to launch")
    if port_taken(spec.port):
        raise RuntimeError(f"port {spec.port} already answers; another server is running there")
    token = ""
    if spec.served_name:
        token = f"qbench-{os.urandom(8).hex()}"
        spec = tagged(spec, token)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    container = container_id = ""
    if backend == "docker":
        argv = docker_argv(spec, cfg, gpus, results_dir)
        container = argv[argv.index("--name") + 1]
        secrets = docker_secrets(spec)
        run = subprocess.run(
            argv, capture_output=True, text=True, check=False, env=env_map.with_env(secrets) if secrets else None
        )
        if run.returncode != 0:
            with (results_dir / "server.log").open("w") as log:
                log.write(run.stderr)
                log.flush()
                subprocess.run(["docker", "logs", container], stdout=log, stderr=subprocess.STDOUT, check=False)
            remove(container)
            raise RuntimeError(f"docker run exited with {run.returncode}: {run.stderr.strip()}")
        container_id, server = run.stdout.strip(), ["docker", "logs", "-f", container]
    else:
        server = spec.child_argv(spec.server_argv)
    try:
        with (results_dir / "server.log").open("w") as log:
            proc = subprocess.Popen(server, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    except BaseException:
        if container:
            remove(container)
        raise
    endpoint = f"http://127.0.0.1:{spec.port}"
    accel = cfg.hardware.backend if cfg else "cuda"
    handle = Handle(
        spec.adapter, endpoint, proc.pid, results_dir, proc, accelerator(accel)[0], container, container_id, accel
    )
    try:
        models = endpoint + "/v1/models"
        # A keyed vLLM refuses /v1/models without its key.
        headers = {"Authorization": f"Bearer {spec.env['VLLM_API_KEY']}"} if "VLLM_API_KEY" in spec.env else {}
        owns = (lambda: lists_token(models, token, headers, poll_s)) if token else None
        wait_healthy(proc, endpoint + spec.health_path, timeout_s, poll_s, owns)
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
        (results_dir / "engine.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    except BaseException:
        handle.stop()
        raise
    return handle
