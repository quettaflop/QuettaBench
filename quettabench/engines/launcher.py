"""Launcher: start an engine server from a LaunchSpec, wait for its health
check, record results/<run>/engine.json, and stop it without leaving orphans.

local: the server runs in its own process group and Handle.stop() signals the
whole group, so tensor-parallel workers die with it; this replaces
scripts/clean_orphan_gpus.py. docker: the server runs detached in engine.image
(docker_argv), `docker logs -f` streams into server.log, and Handle.stop()
runs `docker stop`. The docker path has not been run against a real daemon.
"""

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from quettabench.config.schema import ConfigError
from quettabench.engines.base import UnsupportedField
from quettabench.metrics.results import redact

REPO = Path(__file__).resolve().parents[2]

# Engine endpoints are local or on cluster-private addresses; an http_proxy in
# the environment would send those requests through the proxy, which answers 403.
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def urlopen(url, **kwargs):
    """urllib.request.urlopen(url, **kwargs) without any proxy."""
    return _DIRECT.open(url, **kwargs)


@dataclass
class Handle:
    adapter: str
    endpoint: str
    pid: int
    results_dir: Path
    proc: subprocess.Popen
    gpu: str = ""
    container: str = ""   # docker: the container name; pid is then its id

    def stop(self, grace_s=30):
        """SIGTERM the process group; SIGKILL whatever is left after grace_s.
        A container gets `docker stop`, and its log stream then ends."""
        if self.container:
            subprocess.run(["docker", "stop", "-t", str(grace_s), self.container],
                           capture_output=True, check=False)
            self.proc.wait()
            return
        deadline = time.monotonic() + grace_s
        if self._signal_group(signal.SIGTERM):
            while time.monotonic() < deadline and self._signal_group(0):
                self.proc.poll()  # reap the leader so its zombie does not count
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
    """The handle of a run whose driver starts the engine itself (in-process
    vLLM, the cargo bench): the results dir, no endpoint, nothing to stop."""
    results_dir: Path
    endpoint: str | None = None

    def stop(self):
        pass


def idle_gpus(n, backend, visible_devices=None):
    """On cuda the first n GPUs using under 1000 MiB (free_gpu.sh's rule);
    other backends take the caller's CUDA_VISIBLE_DEVICES or 0..n-1."""
    if backend != "cuda":
        return visible_devices or ",".join(str(i) for i in range(n))
    rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
        capture_output=True, text=True, check=True).stdout.splitlines()
    ids = [index for index, used in (row.split(", ") for row in rows if row)
           if float(used.split()[0]) < 1000]
    if len(ids) < n:
        raise RuntimeError(f"NEED {n} FREE GPUS, have: {' '.join(ids) or 'none'}")
    return ",".join(ids[:n])


def gpu_name():
    """First GPU's name from nvidia-smi; empty when there is none, never a crash."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, check=False).stdout
    except OSError:
        return ""
    return out.splitlines()[0].strip() if out.strip() else ""


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


def docker_argv(spec, cfg, gpus):
    """`docker run -d` of spec.server_argv in spec.image on the given GPUs.

    The flags follow the H200 campaign notes (no launch script ever used
    docker): detached, --gpus, --ipc=host, --shm-size, the port published,
    and the repo and the HF cache (hardware.hf_home, when set) mounted
    read-only at their host paths so model paths resolve unchanged inside.
    --entrypoint is the argv's first word, so the image's own ENTRYPOINT does
    not prefix the command. env is spec.env, the non-empty collective pins
    and HF_HOME.
    """
    hw = cfg.hardware
    if not spec.image:
        raise ConfigError(f"{cfg.run_name}: hardware.launcher=docker needs engine.image")
    mounts = [hw.hf_home] if hw.hf_home else []
    env = {**spec.env, **{k: v for k, v in hw.collective.items() if v},
           **({"HF_HOME": hw.hf_home} if hw.hf_home else {})}
    return ["docker", "run", "-d", "--rm", "--name", f"qbench-{cfg.run_name}",
            "--gpus", f'"device={gpus}"', "--ipc=host", f"--shm-size={hw.shm_size}",
            "-p", f"{spec.port}:{spec.port}",
            *(a for path in [*mounts, REPO] for a in ("-v", f"{path}:{path}:ro")),
            *(a for k, v in env.items() for a in ("-e", f"{k}={v}")),
            "--entrypoint", spec.server_argv[0], spec.image, *spec.server_argv[1:]]


def start(spec, results_dir, *, backend="local", cfg=None, gpus="", timeout_s=1800,
          poll_s=1.0):
    """Launch spec.server_argv, wait for health, write engine.json; return a Handle.
    backend docker runs it in a container built by docker_argv(spec, cfg, gpus)."""
    if spec.server_unsupported:
        raise UnsupportedField(spec.adapter, ", ".join(spec.server_unsupported))
    if not spec.server_argv:
        raise ValueError(f"{spec.adapter}: the spec has no server to launch")
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    container = ""
    if backend == "docker":
        argv = docker_argv(spec, cfg, gpus)
        container = argv[argv.index("--name") + 1]
        run = subprocess.run(argv, capture_output=True, text=True, check=False)
        if run.returncode != 0:
            raise RuntimeError(f"docker run exited with {run.returncode}: {run.stderr.strip()}")
        pid, server = run.stdout.strip(), ["docker", "logs", "-f", container]
    else:
        pid, server = None, spec.child_argv(spec.server_argv)
    with (results_dir / "server.log").open("w") as log:
        proc = subprocess.Popen(server, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    pid = pid or proc.pid
    endpoint = f"http://127.0.0.1:{spec.port}"
    handle = Handle(spec.adapter, endpoint, pid, results_dir, proc, gpu_name(), container)
    try:
        wait_healthy(proc, endpoint + spec.health_path, timeout_s, poll_s)
    except BaseException:
        handle.stop()
        raise
    launch = {**asdict(spec), "env": redact(spec.env)}
    record = {
        "adapter": spec.adapter,
        "endpoint": endpoint,
        "pid": pid,
        "launch_spec": launch,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "instrumentation": spec.instrumentation,
    }
    if container:
        record.update(container=container, image=spec.image)
    (results_dir / "engine.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return handle
