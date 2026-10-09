"""Local launcher: start an engine server from a LaunchSpec, wait for its health
check, record results/<run>/engine.json, and stop it without leaving orphans.

The server runs in its own process group and Handle.stop() signals the whole
group, so tensor-parallel workers die with it; this replaces
scripts/clean_orphan_gpus.py. The docker backend is not implemented until a
GPU box validates it.
"""

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

from quettabench.engines.base import UnsupportedField

SECRET_NAME = re.compile(r"TOKEN|KEY|^HF_")


@dataclass
class Handle:
    adapter: str
    endpoint: str
    pid: int
    results_dir: Path
    proc: subprocess.Popen
    gpu: str = ""

    def stop(self, grace_s=30):
        """SIGTERM the process group; SIGKILL whatever is left after grace_s."""
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
    """scripts/crossval/free_gpu.sh: on cuda the first n GPUs using under
    1000 MiB; other backends take the caller's CUDA_VISIBLE_DEVICES or 0..n-1."""
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


def redacted(env):
    """env with every *TOKEN*, *KEY* and HF_* value replaced, for results/."""
    return {k: "<redacted>" if SECRET_NAME.search(k) else v for k, v in env.items()}


def wait_healthy(proc, url, timeout_s, poll_s):
    """Poll url until it answers 200; fail fast if the server exits first."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode} before {url} answered")
        try:
            with urllib.request.urlopen(url, timeout=poll_s) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(poll_s)
    raise TimeoutError(f"{url} not healthy after {timeout_s}s")


def start(spec, results_dir, *, backend="local", timeout_s=1800, poll_s=1.0):
    """Launch spec.server_argv, wait for health, write engine.json; return a Handle."""
    if backend == "docker":
        raise NotImplementedError("docker launcher: not shipped until a GPU box validates it")
    if spec.server_unsupported:
        raise UnsupportedField(spec.adapter, ", ".join(spec.server_unsupported))
    if not spec.server_argv:
        raise ValueError(f"{spec.adapter}: the spec has no server to launch")
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "server.log").open("w") as log:
        proc = subprocess.Popen(spec.child_argv(spec.server_argv), stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
    endpoint = f"http://127.0.0.1:{spec.port}"
    handle = Handle(spec.adapter, endpoint, proc.pid, results_dir, proc, gpu_name())
    try:
        wait_healthy(proc, endpoint + spec.health_path, timeout_s, poll_s)
    except BaseException:
        handle.stop()
        raise
    launch = {**asdict(spec), "env": redacted(spec.env)}
    (results_dir / "engine.json").write_text(json.dumps({
        "adapter": spec.adapter,
        "endpoint": endpoint,
        "pid": proc.pid,
        "launch_spec": launch,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "instrumentation": spec.instrumentation,
    }, indent=2, sort_keys=True) + "\n")
    return handle
