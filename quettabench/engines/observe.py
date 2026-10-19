"""What a running server actually got, as opposed to what the yaml asked for: its process's argv and environment,
what it answers on /version and /v1/models, and the settings its startup log reports. Call server() once the
server is healthy; run.json keeps the result as provenance.server_observed."""

import glob
import os
import re
from collections.abc import Callable
from pathlib import Path

from quettabench.engines.launcher import served_models, server_version
from quettabench.metrics.results import redact, redact_argv

# The startup log's settings are in its first lines; a long run's log can run to gigabytes.
LOG_HEAD_BYTES = 4 << 20

# vLLM startup log lines (0.31): one value each, the first match.
LOG_SETTINGS: dict[str, tuple[str, Callable]] = {
    "tensor_parallel_size": (r"tensor_parallel_size=(\d+)", int),
    "pipeline_parallel_size": (r"pipeline_parallel_size=(\d+)", int),
    "data_parallel_size": (r"data_parallel_size=(\d+)", int),
    "expert_parallel": (r"enable_expert_parallel=(True|False)", lambda v: v == "True"),
    "chunked_prefill": (r"(?:enable_chunked_prefill|chunked_prefill_enabled)=(True|False)", lambda v: v == "True"),
    "prefix_caching": (r"enable_prefix_caching=(True|False)", lambda v: v == "True"),
    "speculative_config": (r"speculative_config=(None|\w+\([^()]*\)|\{[^{}]*\})", str),
    "cudagraph_capture_sizes": (
        r"""['"]cudagraph_capture_sizes['"]: \[([\d, ]*)\]""",
        lambda v: [int(x) for x in v.split(",") if x.strip()],
    ),
}
# Lines a server prints once per engine (per data-parallel rank): every match, in order.
KV_TOKENS = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
MAX_CONCURRENCY = re.compile(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x")


def log_settings(path) -> dict:
    """The settings a vLLM server's startup log reports; a setting the log does not show is None."""
    try:
        with open(path, "rb") as f:
            text = f.read(LOG_HEAD_BYTES).decode("utf-8", "replace")
    except OSError as e:
        return {"why": f"no server log: {e.strerror}"}
    out = {}
    for name, (pattern, parse) in LOG_SETTINGS.items():
        m = re.search(pattern, text)
        out[name] = parse(m.group(1)) if m else None
    out["kv_cache_tokens"] = [int(m.group(1).replace(",", "")) for m in KV_TOKENS.finditer(text)] or None
    out["max_concurrency"] = [
        {"tokens_per_request": int(m.group(1).replace(",", "")), "x": float(m.group(2))}
        for m in MAX_CONCURRENCY.finditer(text)
    ] or None
    return out


def listening_pid(port) -> int | None:
    """This user's process listening on port (any address), from /proc/net/tcp and each process's open sockets; the
    lowest pid when several share the socket. None when none of this user's processes holds it."""
    inodes = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            rows = Path(table).read_text().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            cols = row.split()
            # 0A is TCP_LISTEN.
            if len(cols) > 9 and cols[3] == "0A" and int(cols[1].rsplit(":", 1)[1], 16) == port:
                inodes.add(f"socket:[{cols[9]}]")
    pids = []
    for fd_dir in glob.glob("/proc/[0-9]*/fd") if inodes else ():
        try:
            if any(os.readlink(os.path.join(fd_dir, fd)) in inodes for fd in os.listdir(fd_dir)):
                pids.append(int(fd_dir.split("/")[2]))
        except OSError:
            continue
    return min(pids) if pids else None


def process(pid) -> dict:
    """pid's argv and environment, from /proc, secrets redacted."""
    try:
        argv = [a.decode() for a in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if a]
        pairs = [e.decode().partition("=") for e in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0") if e]
    except OSError as e:
        return {"pid": pid, "why": f"/proc/{pid} unreadable: {e.strerror}"}
    return {"pid": pid, "argv": redact_argv(argv), "env": redact({k: v for k, _, v in pairs})}


def server(endpoint, pid=None, log=None, container="") -> dict:
    """What the server at endpoint got. pid is the server's process when qbench started it locally, else the one
    listening on endpoint's port is looked up; log is its server.log, when qbench has one."""
    out: dict = {
        "version": server_version(endpoint + "/version", 10),
        "models": served_models(endpoint + "/v1/models", 10),
    }
    if container:
        out["process"] = {"why": f"the server runs in container {container}; its argv and env are the container's"}
    else:
        pid = pid or listening_pid(int(endpoint.rsplit(":", 1)[1]))
        out["process"] = process(pid) if pid else {"why": "no process of this user listens on the server's port"}
    out["log"] = log_settings(log) if log else {"why": "no server log: qbench did not start the server"}
    return out
