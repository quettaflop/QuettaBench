#!/usr/bin/env python3
"""Read-only sweep progress and GPU occupancy reporter.

This script complements bench_orchestrator.sh. It does not dispatch jobs,
publish data, or edit sweep state. It reads local durable benchmark state,
polls each GPU host with nvidia-smi over SSH, and emits a compact Markdown
snapshot suitable for a long-running monitor loop plus optional dashboard JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable

import compile_sweep


HERE = Path(__file__).resolve().parent
BENCH_ROOT = HERE.parent
DEFAULT_JOBS_CONFIG = HERE / "sweep.yaml"
DEFAULT_STATE_DIR = Path("/mnt/100g/agent-bench/state")
DEFAULT_HOSTS = ("a100", "3090", "2080ti", "h100", "h100-2")
PORTS = tuple(range(8089, 8097))
IDLE_MEMORY_USED_THRESHOLD_MIB = 512
LEGACY_STATE_FALLBACK = os.environ.get("BENCH_STATE_LEGACY_FALLBACK") == "1"
ORCHESTRATOR_SERVICE = "agentic-serve-bench-orchestrator.service"
ORCHESTRATOR_TIMER = "agentic-serve-bench-orchestrator.timer"
ACTIVE_RUN_STATUSES = {"dispatching", "running"}

REMOTE_SNAPSHOT_SCRIPT = r"""
set -uo pipefail
echo "__WHOAMI__"
whoami 2>/dev/null || true
echo "__GPU__"
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>&1 || true
echo "__PROC__"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
echo "__PS__"
pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF {print $1}' | sort -nu | tr '\n' ',')
parents=""
grandparents=""
if [ -n "$pids" ]; then
  ps -o pid=,user=,ppid=,pgid=,sid=,stat=,etimes=,cmd= -p "${pids%,}" 2>/dev/null || true
  echo "__PARENTS__"
  parents=$(ps -o ppid= -p "${pids%,}" 2>/dev/null | awk '$1 > 1 {print $1}' | sort -nu | tr '\n' ',')
  if [ -n "$parents" ]; then
    ps -o pid=,user=,ppid=,pgid=,sid=,stat=,etimes=,cmd= -p "${parents%,}" 2>/dev/null || true
    grandparents=$(ps -o ppid= -p "${parents%,}" 2>/dev/null | awk '$1 > 1 {print $1}' | sort -nu | tr '\n' ',')
  fi
  echo "__GRANDPARENTS__"
  if [ -n "$grandparents" ]; then
    ps -o pid=,user=,ppid=,pgid=,sid=,stat=,etimes=,cmd= -p "${grandparents%,}" 2>/dev/null || true
  fi
fi
echo "__ENV__"
env_pids=$(printf "%s\n%s\n%s\n" "${pids%,}" "${parents%,}" "${grandparents%,}" | tr ',' '\n' | awk 'NF' | sort -nu | tr '\n' ' ')
for pid in $env_pids; do
  if [ -r "/proc/$pid/environ" ]; then
    tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | awk -v pid="$pid" -F= '$1 ~ /^BENCH_/ {print pid " " $0}'
  fi
done
echo "__PORTS__"
port_pids=""
for p in 8089 8090 8091 8092 8093 8094 8095 8096; do
  line=$(ss -ltnp 2>/dev/null | awk -v p=":$p" '$4 ~ p"$" {print; exit}')
  if [ -n "$line" ]; then
    printf "%s %s\n" "$p" "$line"
    pid=$(printf "%s\n" "$line" | sed -n "s/.*pid=\([0-9][0-9]*\).*/\1/p" | head -1)
    if [ -n "$pid" ]; then port_pids="${port_pids}${pid},"; fi
  fi
done
echo "__PORT_PS__"
if [ -n "$port_pids" ]; then
  ps -o pid=,user=,ppid=,pgid=,sid=,stat=,etimes=,cmd= -p "${port_pids%,}" 2>/dev/null || true
fi
echo "__PORT_ENV__"
for pid in $(printf "%s" "${port_pids%,}" | tr ',' ' '); do
  if [ -r "/proc/$pid/environ" ]; then
    tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | awk -v pid="$pid" -F= '$1 ~ /^BENCH_/ {print pid " " $0}'
  fi
done
"""


@dataclass(frozen=True)
class Job:
    host: str
    model_path: str
    tp: int
    short: str
    mode: str
    backend: str
    max_len: str
    gpu_mem: str
    concs: str
    profiles: str
    extra_env: str
    scope: str
    line_no: int

    @property
    def job_id(self) -> str:
        suffix = "" if self.backend == "vllm" else f"_{self.backend}"
        return f"{self.host}_{self.short}_tp{self.tp}_{self.mode}{suffix}"


@dataclass
class JobState:
    job: Job
    status: str
    gpus: str
    port: str
    attempt: str
    age_seconds: int | None
    max_len_override: str
    run_id: str


@dataclass
class GpuInfo:
    index: str
    uuid: str
    name: str
    memory_used_mib: int | None
    memory_total_mib: int | None
    util_pct: int | None


@dataclass
class PsInfo:
    user: str
    ppid: str
    pgid: str
    sid: str
    stat: str
    age_seconds: int | None
    cmd: str


@dataclass
class GpuProcess:
    gpu_index: str
    gpu_uuid: str
    pid: str
    process_name: str
    used_memory_mib: int | None
    user: str
    ppid: str
    pgid: str
    sid: str
    stat: str
    age_seconds: int | None
    cmd: str
    parent_user: str = ""
    parent_ppid: str = ""
    parent_pgid: str = ""
    parent_sid: str = ""
    parent_stat: str = ""
    parent_age_seconds: int | None = None
    parent_cmd: str = ""
    grandparent_user: str = ""
    grandparent_ppid: str = ""
    grandparent_pgid: str = ""
    grandparent_sid: str = ""
    grandparent_stat: str = ""
    grandparent_age_seconds: int | None = None
    grandparent_cmd: str = ""
    bench_run_id: str = ""
    bench_job_id: str = ""
    bench_scope: str = ""
    bench_port: str = ""
    bench_gpus: str = ""
    orphan_reason: str = ""
    kind: str = "unknown"


@dataclass
class PortListener:
    port: str
    detail: str
    pid: str
    user: str
    ppid: str
    pgid: str
    sid: str
    stat: str
    age_seconds: int | None
    cmd: str
    bench_run_id: str = ""
    bench_job_id: str = ""
    bench_scope: str = ""
    bench_port: str = ""
    bench_gpus: str = ""


@dataclass
class HostSnapshot:
    host: str
    ok: bool
    remote_user: str = ""
    gpus: list[GpuInfo] | None = None
    processes: list[GpuProcess] | None = None
    ports: list[str] | None = None
    port_listeners: list[PortListener] | None = None
    error: str = ""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_int(value: str) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_csv(line: str) -> list[str]:
    return next(csv.reader([line], skipinitialspace=True))


def parse_ps_line(line: str) -> tuple[str, PsInfo] | None:
    match = re.match(r"\s*(\d+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\d+)\s+(.*)", line)
    if match:
        pid, user, ppid, pgid, sid, stat, etimes, cmd = match.groups()
        return pid, PsInfo(
            user=user,
            ppid=ppid,
            pgid=pgid,
            sid=sid,
            stat=stat,
            age_seconds=parse_int(etimes),
            cmd=cmd,
        )
    legacy = re.match(r"\s*(\d+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(.*)", line)
    if legacy:
        pid, user, ppid, etimes, cmd = legacy.groups()
        return pid, PsInfo(
            user=user,
            ppid=ppid,
            pgid="",
            sid="",
            stat="",
            age_seconds=parse_int(etimes),
            cmd=cmd,
        )
    return None


def split_gpu_list(raw: str) -> list[str]:
    return [part for part in re.split(r"[,\s]+", raw.strip()) if part]


def compact_cmd(cmd: str, max_len: int = 96) -> str:
    clean = " ".join(cmd.split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3] + "..."


def human_age(seconds: int | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    return f"{hours}h{minutes % 60:02d}m"


def human_mem(used: int | None, total: int | None = None) -> str:
    if used is None:
        return "-"
    if total is None:
        return f"{used}MiB"
    return f"{used}/{total}MiB"


def read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return default


def extra_env_value(extra_env: str, key: str) -> str:
    try:
        parts = shlex.split(extra_env)
    except ValueError:
        parts = extra_env.split()
    prefix = f"{key}="
    for part in parts:
        if part.startswith(prefix):
            return part[len(prefix):]
    return ""


def canonical_state_scope(scope: str) -> str:
    if scope in {"synthetic", "latest", "synthetic-distributional", "synthetic_distributional"}:
        return "synthetic_distributional"
    if scope in {"archive", "trace_replay"}:
        return "trace_replay"
    return scope


def job_state_scope(job: Job) -> str:
    for key in ("RESULT_SCOPE", "DASHBOARD_SCOPE", "SCOPE"):
        value = extra_env_value(job.extra_env, key)
        if value:
            return canonical_state_scope(value)
    return canonical_state_scope(job.scope)


def control_dir_for_state(state_dir: Path) -> Path:
    if state_dir.name in {"synthetic_distributional", "trace_replay", "archived"}:
        return state_dir.parent / "control"
    return state_dir / "control"


def load_drained_hosts(state_dir: Path) -> tuple[set[str], Path]:
    path = control_dir_for_state(state_dir) / "drained-hosts.txt"
    drained: set[str] = set()
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            drained.add(line.split()[0])
    except OSError:
        pass
    return drained, path


def parse_blocked_gpu_line(line: str) -> tuple[str, str] | None:
    normalized = line.replace(":", " ")
    parts = normalized.split()
    if len(parts) < 2:
        return None
    host, gpu = parts[0], parts[1]
    if not host or parse_int(gpu) is None:
        return None
    return host, gpu


def sort_gpu_indices(values: Iterable[str]) -> list[str]:
    return sorted(values, key=lambda value: (parse_int(value) is None, parse_int(value) or 0, value))


def load_blocked_gpus(state_dir: Path) -> tuple[dict[str, set[str]], Path]:
    path = control_dir_for_state(state_dir) / "blocked-gpus.txt"
    blocked: dict[str, set[str]] = defaultdict(set)
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parsed = parse_blocked_gpu_line(line)
            if parsed is None:
                continue
            host, gpu = parsed
            blocked[host].add(gpu)
    except OSError:
        pass
    return blocked, path


def blocked_gpu_records(blocked: dict[str, set[str]]) -> list[dict[str, str]]:
    return [
        {"host": host, "gpu": gpu}
        for host in sorted(blocked)
        for gpu in sort_gpu_indices(blocked[host])
    ]


def state_path(state_dir: Path, job: Job, suffix: str) -> Path:
    # Job IDs contain model-version dots, so Path.with_suffix() would corrupt
    # names like Llama-3.1-8B_tp4_multi.
    scope = job_state_scope(job)
    if scope and scope != "all":
        if state_dir.name == scope:
            return state_dir / f"{job.job_id}.{suffix}"
        scoped = state_dir / scope / f"{job.job_id}.{suffix}"
        if scoped.exists() or not LEGACY_STATE_FALLBACK:
            return scoped
    return state_dir / f"{job.job_id}.{suffix}"


def scoped_state_dir(state_dir: Path, job: Job) -> Path:
    scope = job_state_scope(job)
    if scope and scope != "all" and state_dir.name != scope:
        return state_dir / scope
    return state_dir


def run_record_path(state_dir: Path, job: Job, run_id: str) -> Path:
    return scoped_state_dir(state_dir, job) / "runs" / f"{run_id}.json"


def read_run_record(state_dir: Path, job: Job, run_id: str) -> dict[str, Any]:
    if not run_id:
        return {}
    try:
        payload = json.loads(run_record_path(state_dir, job, run_id).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if str(payload.get("job_id") or "") not in {"", job.job_id}:
        return {}
    return payload


def timestamp_age_seconds(value: str, now: float) -> int | None:
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0, int(now - parsed.timestamp()))


def reconcile_state_with_run_record(state: JobState, state_dir: Path, now: float) -> JobState:
    record = read_run_record(state_dir, state.job, state.run_id)
    if str(record.get("status") or "") not in ACTIVE_RUN_STATUSES:
        return state
    started_age = timestamp_age_seconds(str(record.get("started_at") or ""), now)
    if (
        state.status != "running"
        and (started_age is None or started_age >= job_warmup_timeout_seconds(state.job))
    ):
        return state
    state.status = "running"
    state.port = str(record.get("port") or state.port)
    gpus = record.get("gpus")
    if isinstance(gpus, list) and gpus:
        state.gpus = ",".join(str(gpu) for gpu in gpus)
    elif record.get("gpus"):
        state.gpus = str(record.get("gpus"))
    if started_age is not None:
        state.age_seconds = started_age
    return state


def parse_job_rows(lines: Iterable[str], *, source: str, default_scope: str = "all") -> list[Job]:
    jobs: list[Job] = []
    scope = default_scope
    for line_no, raw in enumerate(lines, start=1):
        line = raw.strip()
        if line.startswith("# SCOPE:"):
            scope = line.split(":", 1)[1].strip() or "all"
            continue
        if not line or line.startswith("#"):
            continue
        parts = raw.rstrip("\n").split("|")
        while len(parts) < 11:
            parts.append("")
        host, model_path, tp, short, mode, backend, max_len, gpu_mem, concs, profiles, extra_env = parts[:11]
        host = host.strip()
        backend = (backend.strip() or "vllm")
        jobs.append(
            Job(
                host=host,
                model_path=model_path.strip(),
                tp=int(tp.strip()),
                short=short.strip(),
                mode=mode.strip(),
                backend=backend,
                max_len=max_len.strip(),
                gpu_mem=gpu_mem.strip(),
                concs=concs.strip(),
                profiles=profiles.strip(),
                extra_env=extra_env.strip(),
                scope=scope,
                line_no=line_no,
            )
        )
    return jobs


def job_from_manifest_record(record: dict[str, Any], *, scope: str, line_no: int) -> Job:
    return Job(
        host=str(record["host"]),
        model_path=str(record["model_path"]),
        tp=int(record["tp"]),
        short=str(record["short"]),
        mode=str(record["mode"]),
        backend=str(record.get("backend") or "vllm"),
        max_len=str(record["max_len"]),
        gpu_mem=str(record["gpu_mem"]),
        concs=" ".join(str(c) for c in record.get("concurrencies") or []),
        profiles=" ".join(str(p) for p in record.get("profiles") or []),
        extra_env=str(record.get("extra_env") or ""),
        scope=scope,
        line_no=line_no,
    )


def parse_jobs_file(jobs_file: Path) -> list[Job]:
    if jobs_file.suffix == ".json":
        payload = json.loads(jobs_file.read_text())
        scope = str(payload.get("scope") or "all")
        return [
            job_from_manifest_record(record, scope=scope, line_no=i)
            for i, record in enumerate(payload.get("jobs") or [], start=1)
            if isinstance(record, dict)
        ]
    return parse_job_rows(jobs_file.read_text().splitlines(), source=str(jobs_file))


def compile_jobs_from_config(jobs_config: Path, scope: str) -> list[Job]:
    manifest = compile_sweep.load_manifest(jobs_config)
    compile_sweep.validate(manifest)
    emitted, _skipped = compile_sweep.compile_jobs(manifest, scope)
    rows = [row for _cell, row in emitted]
    return parse_job_rows(rows, source=str(jobs_config), default_scope=scope)


def load_jobs(args: argparse.Namespace) -> list[Job]:
    if args.jobs_file:
        return parse_jobs_file(args.jobs_file)
    return compile_jobs_from_config(args.jobs_config, args.scope)


def load_job_states(jobs: Iterable[Job], state_dir: Path) -> list[JobState]:
    states: list[JobState] = []
    now = time.time()
    for job in jobs:
        status_file = state_path(state_dir, job, "status")
        status = read_text(status_file, "pending") or "pending"
        age_seconds: int | None = None
        try:
            age_seconds = max(0, int(now - status_file.stat().st_mtime))
        except OSError:
            pass
        state = JobState(
            job=job,
            status=status,
            gpus=read_text(state_path(state_dir, job, "gpus")),
            port=read_text(state_path(state_dir, job, "port")),
            attempt=read_text(state_path(state_dir, job, "attempt"), "0") or "0",
            age_seconds=age_seconds,
            max_len_override=read_text(state_path(state_dir, job, "max_len_override")),
            run_id=read_text(state_path(state_dir, job, "run_id")),
        )
        states.append(reconcile_state_with_run_record(state, state_dir, now))
    return states


def ssh_snapshot(host: str, timeout: int) -> HostSnapshot:
    command = f"bash -lc {shlex.quote(REMOTE_SNAPSHOT_SCRIPT)}"
    try:
        proc = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", host, command],
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return HostSnapshot(host=host, ok=False, error=str(exc))
    if proc.returncode != 0 and not proc.stdout:
        return HostSnapshot(host=host, ok=False, error=(proc.stderr or f"ssh exit {proc.returncode}").strip())
    return parse_host_snapshot(host, proc.stdout, proc.stderr)


def parse_host_snapshot(host: str, stdout: str, stderr: str) -> HostSnapshot:
    sections: dict[str, list[str]] = defaultdict(list)
    section = ""
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("__") and stripped.endswith("__"):
            section = stripped.strip("_").lower()
            continue
        if section:
            sections[section].append(line.rstrip())

    remote_user = next((line.strip() for line in sections.get("whoami", []) if line.strip()), "")

    gpus: list[GpuInfo] = []
    gpu_errors: list[str] = []
    uuid_to_index: dict[str, str] = {}
    for line in sections.get("gpu", []):
        stripped_line = line.strip()
        if not stripped_line:
            continue
        if "failed" in stripped_line.lower() or "error" in stripped_line.lower() or "nvml" in stripped_line.lower():
            gpu_errors.append(stripped_line)
            continue
        parts = parse_csv(line)
        if len(parts) < 6:
            gpu_errors.append(stripped_line)
            continue
        index, uuid, name, mem_used, mem_total, util = [part.strip() for part in parts[:6]]
        uuid_to_index[uuid] = index
        gpus.append(
            GpuInfo(
                index=index,
                uuid=uuid,
                name=name,
                memory_used_mib=parse_int(mem_used),
                memory_total_mib=parse_int(mem_total),
                util_pct=parse_int(util),
            )
        )

    ps_by_pid: dict[str, PsInfo] = {}
    for line in sections.get("ps", []):
        parsed = parse_ps_line(line)
        if parsed:
            pid, info = parsed
            ps_by_pid[pid] = info

    parent_by_pid: dict[str, PsInfo] = {}
    for line in sections.get("parents", []):
        parsed = parse_ps_line(line)
        if parsed:
            pid, info = parsed
            parent_by_pid[pid] = info

    grandparent_by_pid: dict[str, PsInfo] = {}
    for line in sections.get("grandparents", []):
        parsed = parse_ps_line(line)
        if parsed:
            pid, info = parsed
            grandparent_by_pid[pid] = info

    env_by_pid: dict[str, dict[str, str]] = defaultdict(dict)
    for line in sections.get("env", []):
        match = re.match(r"\s*(\d+)\s+([^=\s]+)=(.*)", line)
        if match:
            pid, key, value = match.groups()
            env_by_pid[pid][key] = value

    def bench_env_value(key: str, *pids: str) -> str:
        for pid in pids:
            if not pid:
                continue
            value = env_by_pid.get(pid, {}).get(key, "")
            if value:
                return value
        return ""

    processes: list[GpuProcess] = []
    for line in sections.get("proc", []):
        if not line.strip() or "No running processes" in line:
            continue
        parts = parse_csv(line)
        if len(parts) < 4:
            continue
        gpu_uuid, pid, process_name, used_memory = [part.strip() for part in parts[:4]]
        info = ps_by_pid.get(pid) or PsInfo("?", "?", "", "", "", None, process_name)
        parent = parent_by_pid.get(info.ppid)
        grandparent = grandparent_by_pid.get(parent.ppid) if parent else None
        processes.append(
            GpuProcess(
                gpu_index=uuid_to_index.get(gpu_uuid, "?"),
                gpu_uuid=gpu_uuid,
                pid=pid,
                process_name=process_name,
                used_memory_mib=parse_int(used_memory),
                user=info.user,
                ppid=info.ppid,
                pgid=info.pgid,
                sid=info.sid,
                stat=info.stat,
                age_seconds=info.age_seconds,
                cmd=info.cmd,
                parent_user=parent.user if parent else "",
                parent_ppid=parent.ppid if parent else "",
                parent_pgid=parent.pgid if parent else "",
                parent_sid=parent.sid if parent else "",
                parent_stat=parent.stat if parent else "",
                parent_age_seconds=parent.age_seconds if parent else None,
                parent_cmd=parent.cmd if parent else "",
                grandparent_user=grandparent.user if grandparent else "",
                grandparent_ppid=grandparent.ppid if grandparent else "",
                grandparent_pgid=grandparent.pgid if grandparent else "",
                grandparent_sid=grandparent.sid if grandparent else "",
                grandparent_stat=grandparent.stat if grandparent else "",
                grandparent_age_seconds=grandparent.age_seconds if grandparent else None,
                grandparent_cmd=grandparent.cmd if grandparent else "",
                bench_run_id=bench_env_value("BENCH_RUN_ID", pid, info.ppid, parent.ppid if parent else ""),
                bench_job_id=bench_env_value("BENCH_JOB_ID", pid, info.ppid, parent.ppid if parent else ""),
                bench_scope=bench_env_value("BENCH_SCOPE", pid, info.ppid, parent.ppid if parent else ""),
                bench_port=bench_env_value("BENCH_PORT", pid, info.ppid, parent.ppid if parent else ""),
                bench_gpus=bench_env_value("BENCH_GPUS", pid, info.ppid, parent.ppid if parent else ""),
            )
        )

    port_env_by_pid: dict[str, dict[str, str]] = defaultdict(dict)
    for line in sections.get("port_env", []):
        match = re.match(r"\s*(\d+)\s+([^=\s]+)=(.*)", line)
        if match:
            pid, key, value = match.groups()
            port_env_by_pid[pid][key] = value

    def port_env_value(key: str, pid: str) -> str:
        if not pid:
            return ""
        return port_env_by_pid.get(pid, {}).get(key, "")

    port_ps_by_pid: dict[str, PsInfo] = {}
    for line in sections.get("port_ps", []):
        parsed = parse_ps_line(line)
        if parsed:
            pid, info = parsed
            port_ps_by_pid[pid] = info

    ports = [line.strip() for line in sections.get("ports", []) if line.strip()]
    port_listeners: list[PortListener] = []
    for line in ports:
        match = re.match(r"^(\d+)\s+(.*)$", line.strip())
        if not match:
            continue
        port, detail = match.groups()
        pid_match = re.search(r"pid=([0-9]+)", detail)
        pid = pid_match.group(1) if pid_match else ""
        if not pid:
            continue
        info = port_ps_by_pid.get(pid) or PsInfo("?", "?", "", "", "", None, detail)
        port_listeners.append(
            PortListener(
                port=port,
                detail=detail,
                pid=pid,
                user=info.user,
                ppid=info.ppid,
                pgid=info.pgid,
                sid=info.sid,
                stat=info.stat,
                age_seconds=info.age_seconds,
                cmd=info.cmd,
                bench_run_id=port_env_value("BENCH_RUN_ID", pid),
                bench_job_id=port_env_value("BENCH_JOB_ID", pid),
                bench_scope=port_env_value("BENCH_SCOPE", pid),
                bench_port=port_env_value("BENCH_PORT", pid),
                bench_gpus=port_env_value("BENCH_GPUS", pid),
            )
        )
    error = stderr.strip()
    if not gpus and gpu_errors:
        error = "\n".join([part for part in [error, "; ".join(gpu_errors[:3])] if part])
    return HostSnapshot(
        host=host,
        ok=not (not gpus and gpu_errors),
        remote_user=remote_user,
        gpus=gpus,
        processes=processes,
        ports=ports,
        port_listeners=port_listeners,
        error=error,
    )


def running_jobs_by_host_gpu(states: list[JobState]) -> dict[tuple[str, str], list[JobState]]:
    out: dict[tuple[str, str], list[JobState]] = defaultdict(list)
    for state in states:
        if state.status != "running":
            continue
        for gpu in split_gpu_list(state.gpus):
            out[(state.job.host, gpu)].append(state)
    return out


def job_warmup_timeout_seconds(job: Job) -> int:
    return 900 if job.backend == "sglang" else 600


def process_matches_job(proc: GpuProcess, job: Job) -> bool:
    if proc.bench_job_id and proc.bench_job_id == job.job_id:
        return True
    text = " ".join(
        part
        for part in (
            proc.process_name,
            proc.cmd,
            proc.parent_cmd,
            proc.grandparent_cmd,
            job.short,
        )
        if part
    ).lower()
    model_path = job.model_path.lower()
    model_base = Path(job.model_path).name.lower()
    if model_path and model_path in text:
        return True
    return bool(model_base and model_base in text)


def assignment_is_live_or_warming(state: JobState, processes_by_gpu: dict[str, list[GpuProcess]]) -> bool:
    if state.age_seconds is None or state.age_seconds < job_warmup_timeout_seconds(state.job):
        return True
    assigned_gpus = split_gpu_list(state.gpus)
    return any(
        process_matches_job(proc, state.job)
        for gpu in assigned_gpus
        for proc in processes_by_gpu.get(gpu, [])
    )


def live_assignments_for_gpu(
    assignments: list[JobState],
    processes_by_gpu: dict[str, list[GpuProcess]],
) -> list[JobState]:
    return [
        state
        for state in assignments
        if assignment_is_live_or_warming(state, processes_by_gpu)
    ]


def unique_job_states(states: Iterable[JobState]) -> list[JobState]:
    out: dict[str, JobState] = {}
    for state in states:
        out[state.run_id or state.job.job_id] = state
    return list(out.values())


def implied_assignment_from_process(
    proc: GpuProcess,
    states_by_job_id: dict[str, JobState],
    state_dir: Path,
    host: str,
) -> JobState | None:
    if not proc.bench_job_id:
        return None
    state = states_by_job_id.get(proc.bench_job_id)
    if state is None or state.job.host != host:
        return None
    run_id = proc.bench_run_id or state.run_id
    if state.run_id and proc.bench_run_id and state.run_id != proc.bench_run_id:
        return None
    record = read_run_record(state_dir, state.job, run_id)
    if str(record.get("status") or "") not in ACTIVE_RUN_STATUSES:
        return None
    return JobState(
        job=state.job,
        status="running",
        gpus=proc.bench_gpus or state.gpus or proc.gpu_index,
        port=proc.bench_port or state.port,
        attempt=state.attempt,
        age_seconds=proc.age_seconds,
        max_len_override=state.max_len_override,
        run_id=run_id,
    )


def classify_process(proc: GpuProcess, snapshot: HostSnapshot, sweep_jobs: list[JobState]) -> str:
    cmd = " ".join(
        part
        for part in (
            proc.process_name,
            proc.cmd,
            proc.parent_cmd,
            proc.grandparent_cmd,
            proc.bench_run_id,
            proc.bench_job_id,
        )
        if part
    ).lower()
    markers = (
        "/tmp/inference-benchmark",
        "/data48/tmp/inference-benchmark",
        "sweep_all_profiles",
        "sweep_multiturn_profiles",
        "src.benchmark.runner",
        "/tmp/results/synthetic_distributional",
        "/tmp/results/trace_replay",
        "/tmp/results/archived",
        "/data48/tmp/results/synthetic_distributional",
        "/data48/tmp/results/trace_replay",
        "/data48/tmp/results/archived",
    )
    if any(marker in cmd for marker in markers):
        return "sweep"
    if sweep_jobs:
        return "sweep-slot"
    if proc.bench_run_id or proc.bench_job_id:
        return "sweep"
    if snapshot.remote_user and proc.user not in ("?", snapshot.remote_user):
        return "other-user"
    orphan_reason = same_user_orphan_reason(proc)
    if orphan_reason:
        proc.orphan_reason = orphan_reason
        return "same-user-orphan"
    return "same-user-nonsweep"


def same_user_orphan_reason(proc: GpuProcess) -> str:
    cmd = proc.cmd.lower()
    parent_cmd = proc.parent_cmd.lower()
    process_name = proc.process_name.lower()
    is_vllm_worker = "vllm::worker" in process_name or "vllm::worker" in cmd
    parent_is_engine = "vllm::enginecore" in parent_cmd
    if proc.ppid == "1":
        return "process parent is init"
    if is_vllm_worker and parent_is_engine and proc.parent_ppid == "1":
        return "vLLM engine parent is orphaned under init"
    return ""


def job_label(state: JobState) -> str:
    port = f":{state.port}" if state.port else ""
    age = human_age(state.age_seconds)
    return f"{state.job.job_id}{port} age={age}"


def collect_progress(args: argparse.Namespace) -> dict[str, Any]:
    jobs = load_jobs(args)
    states = load_job_states(jobs, args.state_dir)
    drained_hosts, drained_hosts_file = load_drained_hosts(args.state_dir)
    blocked_gpus, blocked_gpus_file = load_blocked_gpus(args.state_dir)
    states_by_host: dict[str, list[JobState]] = defaultdict(list)
    for state in states:
        states_by_host[state.job.host].append(state)

    running_by_gpu = running_jobs_by_host_gpu(states)
    snapshots = [ssh_snapshot(host, args.ssh_timeout) for host in args.hosts]
    return {
        "generated_at": now_iso(),
        "jobs": jobs,
        "states": states,
        "states_by_host": states_by_host,
        "running_by_gpu": running_by_gpu,
        "snapshots": snapshots,
        "orchestrator": orchestrator_status(),
        "control": {
            "drained_hosts": sorted(drained_hosts),
            "drained_hosts_file": str(drained_hosts_file),
            "blocked_gpus": blocked_gpu_records(blocked_gpus),
            "blocked_gpus_file": str(blocked_gpus_file),
        },
    }


def systemd_unit_status(unit: str) -> dict[str, Any]:
    properties = (
        "Id,LoadState,ActiveState,SubState,Result,UnitFileState,"
        "ExecMainCode,ExecMainStatus,NRestarts,ActiveEnterTimestamp,"
        "InactiveEnterTimestamp,StateChangeTimestamp,NextElapseUSecRealtime,"
        "LastTriggerUSec"
    )
    try:
        proc = subprocess.run(
            ["systemctl", "show", unit, f"--property={properties}", "--no-pager"],
            check=False,
            text=True,
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "id": unit,
            "ok": False,
            "load_state": "unknown",
            "active_state": "unknown",
            "sub_state": "unknown",
            "result": "unknown",
            "error": str(exc),
            "stderr": str(exc),
        }

    values: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value

    return {
        "id": values.get("Id") or unit,
        "ok": proc.returncode == 0,
        "load_state": values.get("LoadState", "unknown"),
        "active_state": values.get("ActiveState", "unknown"),
        "sub_state": values.get("SubState", "unknown"),
        "result": values.get("Result", "unknown"),
        "unit_file_state": values.get("UnitFileState", ""),
        "exec_main_code": values.get("ExecMainCode", ""),
        "exec_main_status": values.get("ExecMainStatus", ""),
        "n_restarts": values.get("NRestarts", ""),
        "active_enter_timestamp": values.get("ActiveEnterTimestamp", ""),
        "inactive_enter_timestamp": values.get("InactiveEnterTimestamp", ""),
        "state_change_timestamp": values.get("StateChangeTimestamp", ""),
        "next_elapse_realtime": values.get("NextElapseUSecRealtime", ""),
        "last_trigger": values.get("LastTriggerUSec", ""),
        "stderr": proc.stderr.strip(),
    }


def unit_faulted(unit: dict[str, Any]) -> bool:
    load_state = unit.get("load_state")
    active_state = unit.get("active_state")
    result = unit.get("result")
    exec_status = unit.get("exec_main_status")
    if load_state in {"not-found", "error", "bad-setting"}:
        return True
    if active_state == "failed":
        return True
    if result and result not in {"success", "unknown"}:
        return True
    if exec_status not in {"", "0"}:
        return True
    return False


def unit_status_unavailable(unit: dict[str, Any]) -> bool:
    return (
        not unit.get("ok")
        and unit.get("load_state") == "unknown"
        and unit.get("active_state") == "unknown"
        and bool(unit.get("error") or unit.get("stderr"))
    )


def orchestrator_status() -> dict[str, Any]:
    service = systemd_unit_status(ORCHESTRATOR_SERVICE)
    timer = systemd_unit_status(ORCHESTRATOR_TIMER)

    if unit_status_unavailable(service) or unit_status_unavailable(timer):
        health = "unknown"
        message = "bench orchestrator status is unavailable from this process"
    elif service.get("load_state") == "not-found" and timer.get("load_state") == "not-found":
        health = "not-installed"
        message = "bench orchestrator service/timer are not installed"
    elif unit_faulted(service) or unit_faulted(timer):
        health = "faulted"
        message = "bench orchestrator systemd unit is faulted"
    elif service.get("active_state") in {"active", "activating"}:
        health = "running"
        message = "bench orchestrator tick is running now"
    elif timer.get("active_state") == "active":
        health = "timer-active"
        message = "bench orchestrator timer is active"
    else:
        health = "inactive"
        message = "bench orchestrator is installed but not active"

    return {
        "health": health,
        "message": message,
        "service": service,
        "timer": timer,
    }


def build_report(args: argparse.Namespace, progress: dict[str, Any] | None = None) -> str:
    progress = progress or collect_progress(args)
    states: list[JobState] = progress["states"]
    states_by_host: dict[str, list[JobState]] = progress["states_by_host"]
    running_by_gpu: dict[tuple[str, str], list[JobState]] = progress["running_by_gpu"]
    snapshots: list[HostSnapshot] = progress["snapshots"]
    orchestrator: dict[str, Any] = progress.get("orchestrator") or {}
    lines: list[str] = []

    total_counts = Counter(state.status for state in states)
    lines.append("# Sweep Progress Snapshot")
    lines.append("")
    lines.append(f"- generated_at: {progress['generated_at']}")
    if orchestrator:
        lines.append(f"- orchestrator: {orchestrator.get('health', 'unknown')} ({orchestrator.get('message', 'unknown')})")
    if args.jobs_file:
        lines.append(f"- jobs_file: {args.jobs_file}")
    else:
        lines.append(f"- jobs_config: {args.jobs_config}")
        lines.append(f"- jobs_scope: {args.scope}")
    lines.append(f"- state_dir: {args.state_dir}")
    lines.append(f"- total_jobs: {len(states)} ({format_counts(total_counts)})")
    lines.append("")

    if orchestrator:
        service = orchestrator.get("service") or {}
        timer = orchestrator.get("timer") or {}
        lines.append("## Orchestrator")
        lines.append("")
        lines.append(f"- health: {orchestrator.get('health', 'unknown')}")
        lines.append(f"- message: {orchestrator.get('message', '-')}")
        lines.append(
            f"- service: load={service.get('load_state', '-')} "
            f"active={service.get('active_state', '-')}/{service.get('sub_state', '-')} "
            f"result={service.get('result', '-')}"
        )
        lines.append(
            f"- timer: load={timer.get('load_state', '-')} "
            f"active={timer.get('active_state', '-')}/{timer.get('sub_state', '-')} "
            f"next={timer.get('next_elapse_realtime', '-')}"
        )
        lines.append("")

    lines.append("## Host Progress")
    lines.append("")
    lines.append("| Host | Jobs | Running sweep jobs | Listening ports |")
    lines.append("| --- | --- | --- | --- |")
    for snapshot in snapshots:
        host_states = states_by_host.get(snapshot.host, [])
        counts = Counter(state.status for state in host_states)
        running = [state for state in host_states if state.status == "running"]
        running_text = "<br>".join(job_label(state) for state in running) or "-"
        ports_text = "<br>".join(snapshot.ports or []) if snapshot.ok and snapshot.ports else "-"
        if not snapshot.ok:
            ports_text = f"SSH ERROR: {snapshot.error}"
        lines.append(
            f"| {snapshot.host} | {len(host_states)} ({format_counts(counts)}) | "
            f"{running_text} | {ports_text} |"
        )
    lines.append("")

    lines.append("## GPU Occupancy")
    for snapshot in snapshots:
        lines.append("")
        lines.append(f"### {snapshot.host}")
        if not snapshot.ok:
            lines.append(f"- SSH ERROR: {snapshot.error}")
            continue
        lines.append(f"- ssh_user: {snapshot.remote_user or '?'}")
        lines.append("")
        lines.append("| GPU | Mem | Util | Sweep assignment | GPU processes |")
        lines.append("| --- | --- | --- | --- | --- |")
        processes_by_gpu: dict[str, list[GpuProcess]] = defaultdict(list)
        for proc in snapshot.processes or []:
            raw_jobs_for_gpu = running_by_gpu.get((snapshot.host, proc.gpu_index), [])
            jobs_for_gpu = [state for state in raw_jobs_for_gpu if process_matches_job(proc, state.job)]
            proc.kind = classify_process(proc, snapshot, jobs_for_gpu)
            processes_by_gpu[proc.gpu_index].append(proc)
        for gpu in snapshot.gpus or []:
            raw_jobs_for_gpu = running_by_gpu.get((snapshot.host, gpu.index), [])
            jobs_for_gpu = live_assignments_for_gpu(raw_jobs_for_gpu, processes_by_gpu)
            assignment = "<br>".join(job_label(state) for state in jobs_for_gpu) or "-"
            procs = processes_by_gpu.get(gpu.index, [])
            proc_text = "<br>".join(format_process(proc) for proc in procs) or "-"
            lines.append(
                f"| {gpu.index} | {human_mem(gpu.memory_used_mib, gpu.memory_total_mib)} | "
                f"{gpu.util_pct if gpu.util_pct is not None else '-'}% | {assignment} | {proc_text} |"
            )

        non_sweep = [
            proc
            for proc in (snapshot.processes or [])
            if proc.kind not in ("sweep", "sweep-slot")
        ]
        if non_sweep:
            lines.append("")
            lines.append("Non-sweep GPU processes:")
            for proc in sorted(non_sweep, key=lambda item: (item.gpu_index, item.user, item.pid)):
                lines.append(f"- gpu{proc.gpu_index}: {format_process(proc)}")
    lines.append("")

    lines.append("## Stop / Tail")
    lines.append("- Reporter loop lock: `/tmp/sweep-progress-reporter.lock`")
    lines.append("- Latest report: `/tmp/sweep-progress-latest.md`")
    lines.append("- History: `/tmp/sweep-progress-history.md`")
    lines.append("- Stop command: `pkill -f sweep-progress-reporter.lock`")
    lines.append("")
    return "\n".join(lines)


def port_to_json(line: str, listeners_by_port: dict[str, PortListener] | None = None) -> dict[str, Any]:
    match = re.match(r"^(\d+)\s+(.*)$", line.strip())
    if not match:
        return {"port": "", "detail": line.strip()}
    port, detail = match.groups()
    payload: dict[str, Any] = {"port": port, "detail": detail}
    listener = (listeners_by_port or {}).get(port)
    if listener:
        payload.update(
            {
                "pid": listener.pid,
                "user": listener.user,
                "ppid": listener.ppid,
                "pgid": listener.pgid,
                "sid": listener.sid,
                "stat": listener.stat,
                "age_seconds": listener.age_seconds,
                "age": human_age(listener.age_seconds),
                "command": compact_cmd(listener.cmd, 160),
                "bench_run_id": listener.bench_run_id,
                "bench_job_id": listener.bench_job_id,
                "bench_scope": listener.bench_scope,
                "bench_port": listener.bench_port,
                "bench_gpus": listener.bench_gpus,
            }
        )
    return payload


def job_state_to_json(state: JobState) -> dict[str, Any]:
    return {
        "id": state.job.job_id,
        "host": state.job.host,
        "model_path": state.job.model_path,
        "model_short": state.job.short,
        "tp": state.job.tp,
        "mode": state.job.mode,
        "backend": state.job.backend,
        "scope": state.job.scope,
        "status": state.status,
        "gpus": split_gpu_list(state.gpus),
        "port": state.port,
        "attempt": state.attempt,
        "age_seconds": state.age_seconds,
        "age": human_age(state.age_seconds),
        "max_len_override": state.max_len_override,
        "run_id": state.run_id,
    }


def process_to_json(proc: GpuProcess) -> dict[str, Any]:
    return {
        "gpu_index": proc.gpu_index,
        "gpu_uuid": proc.gpu_uuid,
        "pid": proc.pid,
        "process_name": proc.process_name,
        "used_memory_mib": proc.used_memory_mib,
        "user": proc.user,
        "ppid": proc.ppid,
        "pgid": proc.pgid,
        "sid": proc.sid,
        "stat": proc.stat,
        "age_seconds": proc.age_seconds,
        "age": human_age(proc.age_seconds),
        "command": compact_cmd(proc.cmd, 160),
        "parent_user": proc.parent_user,
        "parent_ppid": proc.parent_ppid,
        "parent_pgid": proc.parent_pgid,
        "parent_sid": proc.parent_sid,
        "parent_stat": proc.parent_stat,
        "parent_age_seconds": proc.parent_age_seconds,
        "parent_age": human_age(proc.parent_age_seconds),
        "parent_command": compact_cmd(proc.parent_cmd, 160) if proc.parent_cmd else "",
        "grandparent_user": proc.grandparent_user,
        "grandparent_ppid": proc.grandparent_ppid,
        "grandparent_pgid": proc.grandparent_pgid,
        "grandparent_sid": proc.grandparent_sid,
        "grandparent_stat": proc.grandparent_stat,
        "grandparent_age_seconds": proc.grandparent_age_seconds,
        "grandparent_age": human_age(proc.grandparent_age_seconds),
        "grandparent_command": compact_cmd(proc.grandparent_cmd, 160) if proc.grandparent_cmd else "",
        "bench_run_id": proc.bench_run_id,
        "bench_job_id": proc.bench_job_id,
        "bench_scope": proc.bench_scope,
        "bench_port": proc.bench_port,
        "bench_gpus": proc.bench_gpus,
        "orphan_reason": proc.orphan_reason,
        "kind": proc.kind,
    }


def gpu_status(gpu: GpuInfo, assignments: list[JobState], processes: list[GpuProcess]) -> str:
    kinds = {proc.kind for proc in processes}
    has_sweep = bool(assignments) or bool(kinds & {"sweep", "sweep-slot"})
    has_other = "other-user" in kinds
    has_orphan = "same-user-orphan" in kinds
    has_same = bool(kinds & {"same-user-nonsweep", "same-user-orphan"})
    if has_sweep and has_other:
        return "mixed-other-user"
    if has_sweep and has_same:
        return "mixed-same-user"
    if has_sweep:
        return "sweep"
    if has_other:
        return "other-user"
    if has_orphan:
        return "same-user-orphan"
    if has_same:
        return "same-user-nonsweep"
    if (gpu.memory_used_mib or 0) > IDLE_MEMORY_USED_THRESHOLD_MIB or (gpu.util_pct or 0) > 0:
        return "unknown-busy"
    return "free"


def build_gpu_state_json(args: argparse.Namespace, progress: dict[str, Any] | None = None) -> dict[str, Any]:
    progress = progress or collect_progress(args)
    states: list[JobState] = progress["states"]
    states_by_host: dict[str, list[JobState]] = progress["states_by_host"]
    running_by_gpu: dict[tuple[str, str], list[JobState]] = progress["running_by_gpu"]
    snapshots: list[HostSnapshot] = progress["snapshots"]
    states_by_job_id = {state.job.job_id: state for state in states}
    control = progress.get("control") or {}
    drained_hosts = set(control.get("drained_hosts") or [])
    blocked_by_host: dict[str, set[str]] = defaultdict(set)
    for entry in control.get("blocked_gpus") or []:
        host = str(entry.get("host") or "")
        gpu = str(entry.get("gpu") or "")
        if host and parse_int(gpu) is not None:
            blocked_by_host[host].add(gpu)
    total_counts = Counter(state.status for state in states)

    summary: Counter[str] = Counter()
    hosts_json: list[dict[str, Any]] = []

    for snapshot in snapshots:
        host_states = states_by_host.get(snapshot.host, [])
        host_counts = Counter(state.status for state in host_states)
        running = [state for state in host_states if state.status == "running"]
        host_json: dict[str, Any] = {
            "host": snapshot.host,
            "ok": snapshot.ok,
            "remote_user": snapshot.remote_user,
            "error": snapshot.error,
            "drained": snapshot.host in drained_hosts,
            "blocked_gpus": sort_gpu_indices(blocked_by_host.get(snapshot.host, set())),
            "job_counts": dict(sorted(host_counts.items())),
            "jobs_total": len(host_states),
            "running_jobs": [job_state_to_json(state) for state in running],
            "ports": [
                port_to_json(
                    port,
                    {listener.port: listener for listener in (snapshot.port_listeners or [])},
                )
                for port in (snapshot.ports or [])
            ],
            "gpus": [],
            "unmapped_processes": [],
        }

        summary["hosts_total"] += 1
        if not snapshot.ok:
            summary["hosts_error"] += 1
            hosts_json.append(host_json)
            continue

        summary["hosts_ok"] += 1
        processes_by_gpu: dict[str, list[GpuProcess]] = defaultdict(list)
        implied_by_gpu: dict[str, list[JobState]] = defaultdict(list)
        implied_by_run_id: dict[str, JobState] = {}
        for proc in snapshot.processes or []:
            raw_jobs_for_gpu = running_by_gpu.get((snapshot.host, proc.gpu_index), [])
            jobs_for_gpu = [state for state in raw_jobs_for_gpu if process_matches_job(proc, state.job)]
            implied_state = implied_assignment_from_process(proc, states_by_job_id, args.state_dir, snapshot.host)
            if implied_state is not None:
                jobs_for_gpu.append(implied_state)
                for gpu_index in split_gpu_list(implied_state.gpus) or [proc.gpu_index]:
                    implied_by_gpu[gpu_index].append(implied_state)
                implied_by_run_id[implied_state.run_id or implied_state.job.job_id] = implied_state
            proc.kind = classify_process(proc, snapshot, jobs_for_gpu)
            if proc.gpu_index == "?":
                host_json["unmapped_processes"].append(process_to_json(proc))
            else:
                processes_by_gpu[proc.gpu_index].append(proc)

        if implied_by_run_id:
            running_by_id = {state.run_id or state.job.job_id: state for state in running}
            running_by_id.update(implied_by_run_id)
            running = list(running_by_id.values())
            host_json["running_jobs"] = [job_state_to_json(state) for state in running]

        host_statuses: Counter[str] = Counter()
        for gpu in snapshot.gpus or []:
            raw_assignments = [
                *running_by_gpu.get((snapshot.host, gpu.index), []),
                *implied_by_gpu.get(gpu.index, []),
            ]
            assignments = live_assignments_for_gpu(unique_job_states(raw_assignments), processes_by_gpu)
            processes = sorted(
                processes_by_gpu.get(gpu.index, []),
                key=lambda item: (item.kind, item.user, item.pid),
            )
            blocked = gpu.index in blocked_by_host.get(snapshot.host, set())
            status = gpu_status(gpu, assignments, processes)
            host_statuses[status] += 1
            summary["gpus_total"] += 1
            summary[f"gpus_{status.replace('-', '_')}"] += 1
            if blocked:
                summary["gpus_blocked"] += 1
            host_json["gpus"].append(
                {
                    "index": gpu.index,
                    "uuid": gpu.uuid,
                    "name": gpu.name,
                    "memory_used_mib": gpu.memory_used_mib,
                    "memory_total_mib": gpu.memory_total_mib,
                    "util_pct": gpu.util_pct,
                    "status": status,
                    "blocked": blocked,
                    "assignments": [job_state_to_json(state) for state in assignments],
                    "processes": [process_to_json(proc) for proc in processes],
                }
            )

        host_json["gpu_status_counts"] = dict(sorted(host_statuses.items()))
        hosts_json.append(host_json)

    return {
        "generated_at": progress["generated_at"],
        "orchestrator": progress.get("orchestrator"),
        "jobs_file": str(args.jobs_file) if args.jobs_file else None,
        "jobs_config": str(args.jobs_config) if not args.jobs_file else None,
        "jobs_scope": args.scope,
        "state_dir": str(args.state_dir),
        "control": control,
        "total_jobs": len(states),
        "job_counts": dict(sorted(total_counts.items())),
        "summary": dict(sorted(summary.items())),
        "hosts": hosts_json,
    }


def format_counts(counts: Counter[str]) -> str:
    order = ("done", "running", "pending", "skipped", "failed", "known_oom")
    parts = [f"{key}={counts[key]}" for key in order if counts.get(key)]
    for key in sorted(counts):
        if key not in order:
            parts.append(f"{key}={counts[key]}")
    return ", ".join(parts) or "none"


def format_process(proc: GpuProcess) -> str:
    reason = f" orphan_reason={proc.orphan_reason}" if proc.orphan_reason else ""
    return (
        f"{proc.kind} pid={proc.pid} user={proc.user} mem={human_mem(proc.used_memory_mib)} "
        f"age={human_age(proc.age_seconds)}{reason} cmd=`{compact_cmd(proc.cmd)}`"
    )


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent), prefix=f".{path.name}.") as tmp:
        tmp.write(text)
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def append_history(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip())
        handle.write("\n\n---\n\n")


def run_once(args: argparse.Namespace) -> str:
    try:
        progress = collect_progress(args)
        report = build_report(args, progress)
        if args.json_out:
            gpu_state = build_gpu_state_json(args, progress)
            atomic_write(args.json_out, json.dumps(gpu_state, indent=2, sort_keys=True) + "\n")
    except Exception as exc:  # noqa: BLE001 - monitor must keep running.
        report = f"# Sweep Progress Snapshot\n\n- generated_at: {now_iso()}\n- health: reporter-error\n- error: `{exc}`\n"
        if args.json_out:
            fallback = {
                "generated_at": now_iso(),
                "health": "reporter-error",
                "error": str(exc),
                "hosts": [],
                "summary": {},
            }
            atomic_write(args.json_out, json.dumps(fallback, indent=2, sort_keys=True) + "\n")
    atomic_write(args.out, report)
    if args.history:
        append_history(args.history, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-file", type=Path, help="legacy row file or generated JSON manifest")
    parser.add_argument("--jobs-config", type=Path, default=DEFAULT_JOBS_CONFIG)
    parser.add_argument("--scope", default=os.environ.get("BENCH_JOBS_SCOPE", "all"))
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--hosts", nargs="+", default=list(DEFAULT_HOSTS))
    parser.add_argument("--ssh-timeout", type=int, default=20)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--out", type=Path, default=Path("/tmp/sweep-progress-latest.md"))
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--history", type=Path, default=Path("/tmp/sweep-progress-history.md"))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        report = run_once(args)
        first_counts = next((line for line in report.splitlines() if line.startswith("- total_jobs:")), "- total_jobs: unknown")
        print(f"{now_iso()} wrote {args.out} ({first_counts.removeprefix('- ')})", flush=True)
        if args.once:
            return 0
        time.sleep(max(30, args.interval_seconds))


if __name__ == "__main__":
    sys.exit(main())
