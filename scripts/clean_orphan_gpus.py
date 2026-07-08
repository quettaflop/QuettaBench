#!/usr/bin/env python3
"""Dry-run-first cleaner for orphaned same-user GPU processes.

This script deliberately does not read dashboard gpu-state.json. It collects a
fresh live snapshot through sweep_progress_report.py, applies the same
classification logic, gates cleanup candidates through an observation store, and
then either records a dry-run event or signals the exact observed PIDs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import sweep_progress_report as progress_report  # noqa: E402


DEFAULT_CONFIG_PATH = HERE / "gpu_cleanup.json"
DEFAULT_STATE_DIR = Path("/mnt/100g/agent-bench/state")
DEFAULT_AUDIT_LOG = DEFAULT_STATE_DIR / "gpu-cleanup-events.jsonl"
DEFAULT_OBSERVATION_STORE = DEFAULT_STATE_DIR / "gpu-orphan-observations.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "dry_run": True,
    "min_age_seconds": 900,
    "required_observations": 2,
    "allowed_statuses": ["same-user-orphan"],
    "allowed_process_kinds": ["same-user-orphan"],
    "allowed_orphan_reasons": [
        "process parent is init",
        "vLLM engine parent is orphaned under init",
    ],
    "allowed_parent_commands": ["VLLM::EngineCore"],
    "allow_direct_init_processes": True,
    "term_wait_seconds": 20,
    "allow_sigkill": False,
    "ssh_timeout_seconds": 12,
    "reclaim_same_user_nonsweep": {
        "enabled": True,
        "min_age_seconds": 3600,
        "required_observations": 2,
        "allowed_statuses": ["same-user-nonsweep"],
        "allowed_process_kinds": ["same-user-nonsweep"],
        "managed_ports": ["8089", "8090", "8091", "8092", "8093", "8094", "8095", "8096"],
        "protected_ports": [],
        "protected_pids": [],
        "allowed_server_markers": [
            "vllm::worker",
            "vllm::enginecore",
            "sglang",
            "sglang.launch_server",
            "vllm.entrypoints.openai.api_server",
        ],
    },
    "reclaim_stale_sweep_servers": {
        "enabled": True,
        "min_age_seconds": 3600,
        "required_observations": 2,
        "allowed_statuses": ["sweep"],
        "allowed_process_kinds": ["sweep"],
        "managed_ports": ["8089", "8090", "8091", "8092", "8093", "8094", "8095", "8096"],
        "protected_ports": [],
        "protected_pids": [],
        "allowed_server_markers": [
            "vllm::worker",
            "vllm::enginecore",
            "sglang",
            "sglang.launch_server",
            "vllm.entrypoints.openai.api_server",
        ],
    },
    "reclaim_drained_sweep_servers": {
        "enabled": True,
        "min_age_seconds": 21600,
        "required_observations": 2,
        "allowed_statuses": ["sweep"],
        "allowed_process_kinds": ["sweep"],
        "managed_ports": ["8089", "8090", "8091", "8092", "8093", "8094", "8095", "8096"],
        "protected_ports": [],
        "protected_pids": [],
        "max_gpu_util_pct": 0,
        "allowed_server_markers": [
            "vllm::worker",
            "vllm::enginecore",
            "sglang",
            "sglang.launch_server",
            "vllm.entrypoints.openai.api_server",
        ],
    },
    "hosts": list(progress_report.DEFAULT_HOSTS),
    "audit_log": str(DEFAULT_AUDIT_LOG),
    "observation_store": str(DEFAULT_OBSERVATION_STORE),
}


@dataclass
class CleanupCandidate:
    key: str
    host: str
    gpu_index: str
    gpu_status: str
    pid: str
    user: str
    ppid: str
    age_seconds: int | None
    command: str
    parent_user: str
    parent_ppid: str
    parent_command: str
    orphan_reason: str
    remote_user: str
    kill_pids: list[str]
    policy: str = "same-user-orphan"
    port: str = ""
    run_id: str = ""
    job_id: str = ""
    grandparent_ppid: str = ""
    grandparent_command: str = ""
    gpu_util_pct: int | None = None
    observation_count: int = 0
    first_seen: str = ""
    last_seen: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "host": self.host,
            "gpu_index": self.gpu_index,
            "gpu_status": self.gpu_status,
            "pid": self.pid,
            "user": self.user,
            "ppid": self.ppid,
            "age_seconds": self.age_seconds,
            "command": self.command,
            "parent_user": self.parent_user,
            "parent_ppid": self.parent_ppid,
            "parent_command": self.parent_command,
            "orphan_reason": self.orphan_reason,
            "remote_user": self.remote_user,
            "kill_pids": self.kill_pids,
            "policy": self.policy,
            "port": self.port,
            "run_id": self.run_id,
            "job_id": self.job_id,
            "grandparent_ppid": self.grandparent_ppid,
            "grandparent_command": self.grandparent_command,
            "gpu_util_pct": self.gpu_util_pct,
            "observation_count": self.observation_count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent), prefix=f".{path.name}.") as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def append_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    if not events:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True))
            handle.write("\n")


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return dict(default)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from exc


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [part for part in re.split(r"[,\s]+", value.strip()) if part]
    return [str(value)]


def load_reclaim_config(raw: dict[str, Any], key: str) -> dict[str, Any]:
    default = dict(DEFAULT_CONFIG[key])
    value = raw.get(key, {})
    if isinstance(value, dict):
        default.update(value)
    for key in (
        "allowed_statuses",
        "allowed_process_kinds",
        "managed_ports",
        "protected_ports",
        "protected_pids",
        "allowed_server_markers",
    ):
        default[key] = as_list(default.get(key))
    default["enabled"] = as_bool(default.get("enabled"))
    default["min_age_seconds"] = as_int(default.get("min_age_seconds"), 3600)
    default["required_observations"] = as_int(default.get("required_observations"), 2)
    if "max_gpu_util_pct" in default:
        default["max_gpu_util_pct"] = as_int(default.get("max_gpu_util_pct"), 0)
    return default


def load_config(path: Path) -> dict[str, Any]:
    raw = load_json(path, {})
    config = dict(DEFAULT_CONFIG)
    config.update(raw)
    config["reclaim_same_user_nonsweep"] = load_reclaim_config(raw, "reclaim_same_user_nonsweep")
    config["reclaim_stale_sweep_servers"] = load_reclaim_config(raw, "reclaim_stale_sweep_servers")
    config["reclaim_drained_sweep_servers"] = load_reclaim_config(raw, "reclaim_drained_sweep_servers")
    for key in (
        "allowed_statuses",
        "allowed_process_kinds",
        "allowed_orphan_reasons",
        "allowed_parent_commands",
        "hosts",
    ):
        config[key] = as_list(config.get(key))
    for key in (
        "enabled",
        "dry_run",
        "allow_direct_init_processes",
        "allow_sigkill",
    ):
        config[key] = as_bool(config.get(key))
    for key in (
        "min_age_seconds",
        "required_observations",
        "term_wait_seconds",
        "ssh_timeout_seconds",
    ):
        config[key] = as_int(config.get(key), int(DEFAULT_CONFIG[key]))
    return config


def lower_set(values: list[str]) -> set[str]:
    return {value.lower() for value in values}


def command_matches(command: str, allowed_fragments: list[str]) -> bool:
    if not allowed_fragments:
        return False
    lowered = command.lower()
    return any(fragment.lower() in lowered for fragment in allowed_fragments)


def process_reason_allowed(reason: str, allowed_reasons: list[str]) -> bool:
    if not allowed_reasons:
        return True
    return reason.lower() in lower_set(allowed_reasons)


def unique_pids(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        pid = str(value).strip()
        if not pid.isdigit() or pid == "1" or pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
    return out


def candidate_kill_pids(proc: dict[str, Any], config: dict[str, Any]) -> list[str]:
    pid = str(proc.get("pid") or "")
    ppid = str(proc.get("ppid") or "")
    parent_command = str(proc.get("parent_command") or "")

    if ppid and ppid != "1" and command_matches(parent_command, config["allowed_parent_commands"]):
        return unique_pids([ppid, pid])
    if ppid == "1" and config["allow_direct_init_processes"]:
        return unique_pids([pid])
    return []


def combined_process_text(proc: dict[str, Any]) -> str:
    return " ".join(
        str(proc.get(key) or "")
        for key in (
            "process_name",
            "command",
            "parent_command",
            "grandparent_command",
        )
    )


def port_pid_map(host_state: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for port in host_state.get("ports") or []:
        if not isinstance(port, dict):
            continue
        port_number = str(port.get("port") or "")
        detail = str(port.get("detail") or "")
        match = re.search(r"pid=([0-9]+)", detail)
        if port_number and match:
            out[match.group(1)] = port_number
    return out


def live_assignment_ports(host_state: dict[str, Any]) -> set[str]:
    ports: set[str] = set()
    for gpu in host_state.get("gpus") or []:
        for assignment in gpu.get("assignments") or []:
            if not isinstance(assignment, dict):
                continue
            port = str(assignment.get("port") or "")
            if port:
                ports.add(port)
    return ports


def port_info_pid(port_info: dict[str, Any]) -> str:
    pid = str(port_info.get("pid") or "")
    if pid:
        return pid
    detail = str(port_info.get("detail") or "")
    match = re.search(r"pid=([0-9]+)", detail)
    return match.group(1) if match else ""


def port_from_text(proc: dict[str, Any], managed_ports: set[str]) -> str:
    bench_port = str(proc.get("bench_port") or "")
    if bench_port in managed_ports:
        return bench_port
    text = combined_process_text(proc)
    patterns = (
        r"(?:--port|port=|PORT=)\s*=?\s*([0-9]{4,5})",
        r":([0-9]{4,5})(?:\D|$)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            port = match.group(1)
            if port in managed_ports:
                return port
    return ""


def process_port(proc: dict[str, Any], pid_to_port: dict[str, str], managed_ports: set[str]) -> str:
    text_port = port_from_text(proc, managed_ports)
    if text_port:
        return text_port
    for key in ("pid", "ppid", "parent_ppid"):
        pid = str(proc.get(key) or "")
        port = pid_to_port.get(pid, "")
        if port in managed_ports:
            return port
    return ""


def server_shape_allowed(proc: dict[str, Any], allowed_markers: list[str]) -> bool:
    text = combined_process_text(proc).lower()
    return any(marker.lower() in text for marker in allowed_markers)


def nonsweep_kill_pids(proc: dict[str, Any]) -> list[str]:
    # Prefer the top known server ancestor when available, then include the
    # immediate engine/worker PIDs so multi-process servers are not left half-alive.
    return unique_pids([
        str(proc.get("parent_ppid") or ""),
        str(proc.get("ppid") or ""),
        str(proc.get("pid") or ""),
    ])


def process_key(host: str, gpu_index: str, proc: dict[str, Any], *, policy: str, port: str = "") -> str:
    if policy == "same-user-orphan":
        parts = [
            host,
            gpu_index,
            str(proc.get("pid") or ""),
            str(proc.get("ppid") or ""),
            str(proc.get("orphan_reason") or ""),
        ]
        return "|".join(parts)
    parts = [
        policy,
        host,
        gpu_index,
        str(proc.get("pid") or ""),
        str(proc.get("ppid") or ""),
        port,
    ]
    return "|".join(parts)


def find_candidates(gpu_state: dict[str, Any], config: dict[str, Any]) -> list[CleanupCandidate]:
    allowed_statuses = set(config["allowed_statuses"])
    allowed_kinds = set(config["allowed_process_kinds"])
    reclaim_config = config["reclaim_same_user_nonsweep"]
    reclaim_statuses = set(reclaim_config["allowed_statuses"])
    reclaim_kinds = set(reclaim_config["allowed_process_kinds"])
    managed_ports = set(str(port) for port in reclaim_config["managed_ports"])
    stale_config = config["reclaim_stale_sweep_servers"]
    stale_statuses = set(stale_config["allowed_statuses"])
    stale_kinds = set(stale_config["allowed_process_kinds"])
    stale_managed_ports = set(str(port) for port in stale_config["managed_ports"])
    drained_config = config["reclaim_drained_sweep_servers"]
    drained_statuses = set(drained_config["allowed_statuses"])
    drained_kinds = set(drained_config["allowed_process_kinds"])
    drained_managed_ports = set(str(port) for port in drained_config["managed_ports"])
    candidates: list[CleanupCandidate] = []
    for host_state in gpu_state.get("hosts") or []:
        if not host_state.get("ok"):
            continue
        host = str(host_state.get("host") or "")
        host_drained = as_bool(host_state.get("drained"))
        remote_user = str(host_state.get("remote_user") or "")
        pid_to_port = port_pid_map(host_state)
        assigned_ports = live_assignment_ports(host_state)
        for gpu in host_state.get("gpus") or []:
            gpu_status = str(gpu.get("status") or "")
            assignments = gpu.get("assignments") or []
            gpu_index = str(gpu.get("index") or "")
            gpu_util_pct = gpu.get("util_pct") if isinstance(gpu.get("util_pct"), int) else None
            for proc in gpu.get("processes") or []:
                kind = str(proc.get("kind") or "")
                user = str(proc.get("user") or "")
                if remote_user and user not in {"?", remote_user}:
                    continue

                if gpu_status in allowed_statuses and kind in allowed_kinds:
                    reason = str(proc.get("orphan_reason") or "")
                    candidates.append(
                        CleanupCandidate(
                            key=process_key(host, gpu_index, proc, policy="same-user-orphan"),
                            host=host,
                            gpu_index=gpu_index,
                            gpu_status=gpu_status,
                            pid=str(proc.get("pid") or ""),
                            user=user,
                            ppid=str(proc.get("ppid") or ""),
                            age_seconds=proc.get("age_seconds") if isinstance(proc.get("age_seconds"), int) else None,
                            command=str(proc.get("command") or ""),
                            parent_user=str(proc.get("parent_user") or ""),
                            parent_ppid=str(proc.get("parent_ppid") or ""),
                            parent_command=str(proc.get("parent_command") or ""),
                            orphan_reason=reason,
                            remote_user=remote_user,
                            kill_pids=candidate_kill_pids(proc, config),
                            policy="same-user-orphan",
                            run_id=str(proc.get("bench_run_id") or ""),
                            job_id=str(proc.get("bench_job_id") or ""),
                            grandparent_ppid=str(proc.get("grandparent_ppid") or ""),
                            grandparent_command=str(proc.get("grandparent_command") or ""),
                        )
                    )

                if (
                    stale_config["enabled"]
                    and gpu_status in stale_statuses
                    and kind in stale_kinds
                    and not assignments
                ):
                    port = process_port(proc, pid_to_port, stale_managed_ports)
                    candidates.append(
                        CleanupCandidate(
                            key=process_key(host, gpu_index, proc, policy="stale-sweep-server", port=port),
                            host=host,
                            gpu_index=gpu_index,
                            gpu_status=gpu_status,
                            pid=str(proc.get("pid") or ""),
                            user=user,
                            ppid=str(proc.get("ppid") or ""),
                            age_seconds=proc.get("age_seconds") if isinstance(proc.get("age_seconds"), int) else None,
                            command=str(proc.get("command") or ""),
                            parent_user=str(proc.get("parent_user") or ""),
                            parent_ppid=str(proc.get("parent_ppid") or ""),
                            parent_command=str(proc.get("parent_command") or ""),
                            orphan_reason=str(proc.get("orphan_reason") or ""),
                            remote_user=remote_user,
                            kill_pids=nonsweep_kill_pids(proc),
                            policy="stale-sweep-server",
                            port=port,
                            run_id=str(proc.get("bench_run_id") or ""),
                            job_id=str(proc.get("bench_job_id") or ""),
                            grandparent_ppid=str(proc.get("grandparent_ppid") or ""),
                            grandparent_command=str(proc.get("grandparent_command") or ""),
                        )
                    )

                if (
                    drained_config["enabled"]
                    and host_drained
                    and assignments
                    and gpu_status in drained_statuses
                    and kind in drained_kinds
                ):
                    port = process_port(proc, pid_to_port, drained_managed_ports)
                    assignment = next((item for item in assignments if isinstance(item, dict)), {})
                    candidates.append(
                        CleanupCandidate(
                            key=process_key(host, gpu_index, proc, policy="drained-stale-sweep-server", port=port),
                            host=host,
                            gpu_index=gpu_index,
                            gpu_status=gpu_status,
                            pid=str(proc.get("pid") or ""),
                            user=user,
                            ppid=str(proc.get("ppid") or ""),
                            age_seconds=proc.get("age_seconds") if isinstance(proc.get("age_seconds"), int) else None,
                            command=str(proc.get("command") or ""),
                            parent_user=str(proc.get("parent_user") or ""),
                            parent_ppid=str(proc.get("parent_ppid") or ""),
                            parent_command=str(proc.get("parent_command") or ""),
                            orphan_reason=str(proc.get("orphan_reason") or ""),
                            remote_user=remote_user,
                            kill_pids=nonsweep_kill_pids(proc),
                            policy="drained-stale-sweep-server",
                            port=port,
                            run_id=str(proc.get("bench_run_id") or assignment.get("run_id") or ""),
                            job_id=str(proc.get("bench_job_id") or assignment.get("id") or ""),
                            grandparent_ppid=str(proc.get("grandparent_ppid") or ""),
                            grandparent_command=str(proc.get("grandparent_command") or ""),
                            gpu_util_pct=gpu_util_pct,
                        )
                    )

                if not reclaim_config["enabled"]:
                    continue
                if gpu_status not in reclaim_statuses or kind not in reclaim_kinds:
                    continue
                if assignments:
                    continue
                port = process_port(proc, pid_to_port, managed_ports)
                candidates.append(
                    CleanupCandidate(
                        key=process_key(host, gpu_index, proc, policy="same-user-nonsweep", port=port),
                        host=host,
                        gpu_index=gpu_index,
                        gpu_status=gpu_status,
                        pid=str(proc.get("pid") or ""),
                        user=user,
                        ppid=str(proc.get("ppid") or ""),
                        age_seconds=proc.get("age_seconds") if isinstance(proc.get("age_seconds"), int) else None,
                        command=str(proc.get("command") or ""),
                        parent_user=str(proc.get("parent_user") or ""),
                        parent_ppid=str(proc.get("parent_ppid") or ""),
                        parent_command=str(proc.get("parent_command") or ""),
                        orphan_reason=str(proc.get("orphan_reason") or ""),
                        remote_user=remote_user,
                        kill_pids=nonsweep_kill_pids(proc),
                        policy="same-user-nonsweep",
                        port=port,
                        run_id=str(proc.get("bench_run_id") or ""),
                        job_id=str(proc.get("bench_job_id") or ""),
                        grandparent_ppid=str(proc.get("grandparent_ppid") or ""),
                        grandparent_command=str(proc.get("grandparent_command") or ""),
                        )
                    )

        if stale_config["enabled"]:
            for port_info in host_state.get("ports") or []:
                if not isinstance(port_info, dict):
                    continue
                port = str(port_info.get("port") or "")
                if port not in stale_managed_ports or port in assigned_ports:
                    continue
                pid = port_info_pid(port_info)
                if not pid:
                    continue
                user = str(port_info.get("user") or "")
                if remote_user and user not in {"?", remote_user}:
                    continue
                proc = {
                    "pid": pid,
                    "ppid": str(port_info.get("ppid") or ""),
                    "age_seconds": port_info.get("age_seconds"),
                    "command": str(port_info.get("command") or port_info.get("detail") or ""),
                }
                candidates.append(
                    CleanupCandidate(
                        key=process_key(host, "", proc, policy="stale-sweep-listener", port=port),
                        host=host,
                        gpu_index="",
                        gpu_status="stale-listener",
                        pid=pid,
                        user=user,
                        ppid=str(port_info.get("ppid") or ""),
                        age_seconds=port_info.get("age_seconds")
                        if isinstance(port_info.get("age_seconds"), int)
                        else None,
                        command=str(port_info.get("command") or port_info.get("detail") or ""),
                        parent_user="",
                        parent_ppid="",
                        parent_command="",
                        orphan_reason="",
                        remote_user=remote_user,
                        kill_pids=unique_pids([pid]),
                        policy="stale-sweep-listener",
                        port=port,
                        run_id=str(port_info.get("bench_run_id") or ""),
                        job_id=str(port_info.get("bench_job_id") or ""),
                    )
                )
    return candidates


def load_observations(path: Path) -> dict[str, Any]:
    return load_json(path, {"candidates": {}})


def update_observations(
    candidates: list[CleanupCandidate],
    observations: dict[str, Any],
    timestamp: str,
) -> dict[str, Any]:
    previous = observations.get("candidates") or {}
    updated: dict[str, Any] = {}
    for candidate in candidates:
        old = previous.get(candidate.key) or {}
        count = as_int(old.get("count"), 0) + 1
        first_seen = str(old.get("first_seen") or timestamp)
        candidate.observation_count = count
        candidate.first_seen = first_seen
        candidate.last_seen = timestamp
        updated[candidate.key] = {
            "count": count,
            "first_seen": first_seen,
            "last_seen": timestamp,
            "host": candidate.host,
            "gpu_index": candidate.gpu_index,
            "pid": candidate.pid,
            "ppid": candidate.ppid,
            "orphan_reason": candidate.orphan_reason,
            "policy": candidate.policy,
            "port": candidate.port,
            "run_id": candidate.run_id,
        }
    return {"updated_at": timestamp, "candidates": updated}


def skip_reason(candidate: CleanupCandidate, config: dict[str, Any]) -> str:
    if candidate.policy in {
        "same-user-nonsweep",
        "stale-sweep-server",
        "stale-sweep-listener",
        "drained-stale-sweep-server",
    }:
        if candidate.policy == "stale-sweep-server":
            reclaim_config = config["reclaim_stale_sweep_servers"]
        elif candidate.policy == "stale-sweep-listener":
            reclaim_config = config["reclaim_stale_sweep_servers"]
        elif candidate.policy == "drained-stale-sweep-server":
            reclaim_config = config["reclaim_drained_sweep_servers"]
        else:
            reclaim_config = config["reclaim_same_user_nonsweep"]
        protected_ports = set(str(port) for port in reclaim_config["protected_ports"])
        protected_pids = set(str(pid) for pid in reclaim_config["protected_pids"])
        active_run_ids = config.get("_active_run_ids")
        if (
            candidate.policy != "drained-stale-sweep-server"
            and candidate.run_id
            and (active_run_ids is None or candidate.run_id in set(active_run_ids))
        ):
            return "process has active BENCH_RUN_ID lease"
        if not command_matches(
            " ".join([candidate.command, candidate.parent_command, candidate.grandparent_command]),
            reclaim_config["allowed_server_markers"],
        ):
            return "process does not match allowed benchmark server markers"
        if not candidate.port:
            return "no managed scheduler port found"
        if candidate.port in protected_ports:
            return "scheduler port is protected by config"
        if candidate.pid in protected_pids or candidate.ppid in protected_pids or candidate.parent_ppid in protected_pids:
            return "process pid is protected by config"
        if candidate.age_seconds is None:
            return "process age is unknown"
        if candidate.age_seconds < reclaim_config["min_age_seconds"]:
            return f"process age below min_age_seconds={reclaim_config['min_age_seconds']}"
        if candidate.policy == "drained-stale-sweep-server":
            max_gpu_util = as_int(reclaim_config.get("max_gpu_util_pct"), 0)
            if candidate.gpu_util_pct is None:
                return "gpu utilization is unknown"
            if candidate.gpu_util_pct > max_gpu_util:
                return f"gpu utilization above max_gpu_util_pct={max_gpu_util}"
        if candidate.observation_count < reclaim_config["required_observations"]:
            return f"waiting for {reclaim_config['required_observations']} observations"
        if candidate.remote_user and candidate.user not in {"?", candidate.remote_user}:
            return "process user differs from ssh user"
        if (
            candidate.parent_user
            and candidate.remote_user
            and candidate.parent_user not in {"?", candidate.remote_user}
        ):
            return "parent user differs from ssh user"
        if not candidate.kill_pids:
            return f"no configured kill target for {candidate.policy} shape"
        return ""

    if not candidate.orphan_reason:
        return "missing orphan reason"
    if not process_reason_allowed(candidate.orphan_reason, config["allowed_orphan_reasons"]):
        return "orphan reason is not allowed by config"
    if candidate.age_seconds is None:
        return "process age is unknown"
    if candidate.age_seconds < config["min_age_seconds"]:
        return f"process age below min_age_seconds={config['min_age_seconds']}"
    if candidate.observation_count < config["required_observations"]:
        return f"waiting for {config['required_observations']} observations"
    if candidate.remote_user and candidate.user not in {"?", candidate.remote_user}:
        return "process user differs from ssh user"
    if (
        candidate.parent_user
        and candidate.remote_user
        and candidate.parent_user not in {"?", candidate.remote_user}
    ):
        return "parent user differs from ssh user"
    if not candidate.kill_pids:
        return "no configured kill target for orphan shape"
    return ""


def active_run_ids_from_state(gpu_state: dict[str, Any]) -> set[str]:
    active: set[str] = set()
    for host_state in gpu_state.get("hosts") or []:
        if not isinstance(host_state, dict):
            continue
        for job in host_state.get("running_jobs") or []:
            if isinstance(job, dict) and job.get("run_id"):
                active.add(str(job["run_id"]))
        for gpu in host_state.get("gpus") or []:
            if not isinstance(gpu, dict):
                continue
            for assignment in gpu.get("assignments") or []:
                if isinstance(assignment, dict) and assignment.get("run_id"):
                    active.add(str(assignment["run_id"]))
    return active


def event_for_candidate(
    candidate: CleanupCandidate,
    *,
    action: str,
    timestamp: str,
    dry_run: bool,
    reason: str = "",
    signal_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event = {
        "timestamp": timestamp,
        "action": action,
        "dry_run": dry_run,
        "skip_reason": reason,
        "candidate": candidate.to_json(),
    }
    if signal_result is not None:
        event["signal_result"] = signal_result
    return event


def marker_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = defaultdict(list)
    section = ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("__") and line.endswith("__"):
            section = line.strip("_").lower()
            continue
        if section:
            sections[section].append(line)
    return dict(sections)


def signal_pids(host: str, pids: list[str], config: dict[str, Any]) -> dict[str, Any]:
    clean_pids = unique_pids(pids)
    if not clean_pids:
        return {"ok": False, "error": "no valid pids"}

    pid_words = " ".join(shlex.quote(pid) for pid in clean_pids)
    allow_sigkill = "1" if config["allow_sigkill"] else "0"
    term_wait = max(1, config["term_wait_seconds"])
    script = f"""
set -uo pipefail
pids=({pid_words})
echo "__BEFORE__"
ps -o pid=,user=,ppid=,stat=,etimes=,cmd= -p "$(IFS=,; echo "${{pids[*]}}")" 2>/dev/null || true
echo "__TERM__"
kill -TERM -- "${{pids[@]}}" 2>&1 || true
sleep {term_wait}
echo "__REMAINING_AFTER_TERM__"
remaining=()
for pid in "${{pids[@]}}"; do
  if kill -0 "$pid" 2>/dev/null; then
    echo "$pid"
    remaining+=("$pid")
  fi
done
if [ "{allow_sigkill}" = "1" ] && [ "${{#remaining[@]}}" -gt 0 ]; then
  echo "__KILL__"
  kill -KILL -- "${{remaining[@]}}" 2>&1 || true
  sleep 2
fi
echo "__REMAINING_AFTER_KILL__"
for pid in "${{pids[@]}}"; do
  if kill -0 "$pid" 2>/dev/null; then echo "$pid"; fi
done
"""
    timeout = max(10, config["ssh_timeout_seconds"] + term_wait + 10)
    try:
        proc = subprocess.run(
            [
                "ssh",
                "-o",
                "ConnectTimeout=5",
                "-o",
                "BatchMode=yes",
                host,
                f"bash -lc {shlex.quote(script)}",
            ],
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "ok": False,
            "pids": clean_pids,
            "remaining_pids": clean_pids,
            "error": str(exc),
        }
    sections = marker_sections(proc.stdout)
    remaining_key = "remaining_after_kill" if config["allow_sigkill"] else "remaining_after_term"
    remaining = [line.strip() for line in sections.get(remaining_key, []) if line.strip().isdigit()]
    return {
        "ok": proc.returncode == 0 and not remaining,
        "returncode": proc.returncode,
        "pids": clean_pids,
        "remaining_pids": remaining,
        "sections": sections,
        "stderr": proc.stderr.strip(),
    }


def collect_live_gpu_state(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    report_args = argparse.Namespace(
        jobs_file=args.jobs_file,
        jobs_config=args.jobs_config,
        scope=args.scope,
        state_dir=args.state_dir,
        hosts=args.hosts,
        ssh_timeout=config["ssh_timeout_seconds"],
    )
    progress = progress_report.collect_progress(report_args)
    return progress_report.build_gpu_state_json(report_args, progress)


def cleanup_from_state(
    gpu_state: dict[str, Any],
    config: dict[str, Any],
    *,
    observations_path: Path,
    audit_log: Path,
    dry_run: bool,
    timestamp: str | None = None,
) -> dict[str, Any]:
    timestamp = timestamp or now_iso()
    config = dict(config)
    config["_active_run_ids"] = active_run_ids_from_state(gpu_state)
    events: list[dict[str, Any]] = []
    candidates = find_candidates(gpu_state, config)
    observations = load_observations(observations_path)
    updated_observations = update_observations(candidates, observations, timestamp)
    atomic_write_json(observations_path, updated_observations)

    eligible: list[CleanupCandidate] = []
    for candidate in candidates:
        reason = skip_reason(candidate, config)
        if reason:
            events.append(
                event_for_candidate(
                    candidate,
                    action="skip",
                    timestamp=timestamp,
                    dry_run=dry_run,
                    reason=reason,
                )
            )
        else:
            eligible.append(candidate)

    if dry_run:
        for candidate in eligible:
            events.append(
                event_for_candidate(
                    candidate,
                    action="dry-run",
                    timestamp=timestamp,
                    dry_run=True,
                )
            )
    else:
        by_host: dict[str, list[CleanupCandidate]] = defaultdict(list)
        for candidate in eligible:
            by_host[candidate.host].append(candidate)
        for host, host_candidates in sorted(by_host.items()):
            pids: list[str] = []
            for candidate in host_candidates:
                pids.extend(candidate.kill_pids)
            signal_result = signal_pids(host, pids, config)
            for candidate in host_candidates:
                events.append(
                    event_for_candidate(
                        candidate,
                        action="signal",
                        timestamp=timestamp,
                        dry_run=False,
                        signal_result=signal_result,
                    )
                )

    append_jsonl(audit_log, events)
    counts = Counter(event["action"] for event in events)
    return {
        "generated_at": timestamp,
        "enabled": config["enabled"],
        "dry_run": dry_run,
        "candidates": len(candidates),
        "eligible": len(eligible),
        "events": dict(sorted(counts.items())),
        "audit_log": str(audit_log),
        "observation_store": str(observations_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(os.environ.get("GPU_CLEANUP_CONFIG", DEFAULT_CONFIG_PATH)))
    parser.add_argument("--jobs-file", type=Path)
    parser.add_argument("--jobs-config", type=Path, default=progress_report.DEFAULT_JOBS_CONFIG)
    parser.add_argument("--scope", default=os.environ.get("BENCH_JOBS_SCOPE", "synthetic_distributional"))
    parser.add_argument("--state-dir", type=Path, default=Path(os.environ.get("BENCH_STATE_ROOT", DEFAULT_STATE_DIR)))
    parser.add_argument("--hosts", nargs="+")
    parser.add_argument("--audit-log", type=Path)
    parser.add_argument("--observation-store", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--min-age-seconds", type=int)
    parser.add_argument("--required-observations", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="force audit-only mode")
    mode.add_argument("--execute", action="store_true", help="send signals to eligible orphan PIDs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.min_age_seconds is not None:
        config["min_age_seconds"] = args.min_age_seconds
    if args.required_observations is not None:
        config["required_observations"] = args.required_observations
    if args.dry_run:
        config["dry_run"] = True
    if args.execute:
        config["dry_run"] = False
    if args.hosts is None:
        args.hosts = config["hosts"] or list(progress_report.DEFAULT_HOSTS)

    audit_log = args.audit_log or Path(config["audit_log"])
    observation_store = args.observation_store or Path(config["observation_store"])
    timestamp = now_iso()

    if not config["enabled"]:
        summary = {
            "generated_at": timestamp,
            "enabled": False,
            "dry_run": config["dry_run"],
            "candidates": 0,
            "eligible": 0,
            "events": {},
            "audit_log": str(audit_log),
            "observation_store": str(observation_store),
        }
    else:
        gpu_state = collect_live_gpu_state(args, config)
        summary = cleanup_from_state(
            gpu_state,
            config,
            observations_path=observation_store,
            audit_log=audit_log,
            dry_run=config["dry_run"],
            timestamp=timestamp,
        )

    if args.json_out:
        atomic_write_json(args.json_out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
