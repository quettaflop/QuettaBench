#!/usr/bin/env python3
"""Reconcile sweep job state against actual coverage points.

The orchestrator tracks coarse jobs: host + model + tp + mode + backend. The
dashboard coverage is finer-grained: profile + concurrency rows inside each
coarse job. A job can therefore be marked `done` even when only a subset of its
expected profile/concurrency JSON files reached the dashboard data.

This script uses sweep.yaml as the desired matrix, data.json as the observed
coverage, and /tmp/bench_jobs/state as the dispatch state. It reports missing
coverage per job and can reset stale terminal jobs back to `pending`.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

import compile_sweep
import publish_sweep_state


HERE = Path(__file__).resolve().parent
BENCH_ROOT = HERE.parent

DEFAULT_PUBLIC_BASE = "https://pub-38e30ed030784867856634f1625c7130.r2.dev/json/current"
DEFAULT_DATA = f"{DEFAULT_PUBLIC_BASE}/data.json"
DEFAULT_SWEEP_YAML = HERE / "sweep.yaml"
DEFAULT_BENCH_JOBS = HERE / "bench_jobs.txt"
DEFAULT_STATE_DIR = Path("/tmp/bench_jobs/state")
DEFAULT_SWEEP_STATE = BENCH_ROOT / "dashboard" / "public" / "sweep-state.json"
DEFAULT_REPORT = Path("/tmp/sweep-coverage-reconcile.md")
DEFAULT_MISSING_JOBS = Path("/tmp/bench_jobs/missing_synthetic_distributional_bench_jobs.txt")

BLOCKING_STATUSES = {"done", "skipped", "failed", "known_oom"}
DEFAULT_RESET_STATUSES = {"done"}
COVERAGE_REQUEUE_COUNT_SUFFIX = "coverage_requeue_count"
COVERAGE_BLOCKER_SUFFIX = "coverage_blocker.json"
# --- Failure taxonomy (see docs/coverage-classification-rfc.md) -------------
# `failure_class` is *what actually happened*, captured at the launcher when
# possible and otherwise derived ONCE here (the only place reason strings are
# parsed). The dashboard never re-infers it.
FAILURE_CLASSES = {
    "none",               # success
    "model_missing",      # weights/config absent or unloadable
    "hw_infeasible",      # static: won't fit / unsupported arch
    "oom_kv_cache",       # engine-init or runtime OOM / no cache blocks
    "engine_crash",       # server/engine died on startup
    "requests_aborted",   # server up, 100% requests failed
    "low_success_rate",   # ran, success rate below threshold
    "timeout",            # warmup/serving exceeded budget
    "incomplete_partial", # some cells produced, some missing
    "not_attempted",      # never dispatched
    "driver_fault",       # XID / NVML / GPU fell off the bus
    "unknown",            # produced nothing, no positive evidence of a cause
}

# gpu_mem_util at/above which a captured OOM is treated as an irreducible limit
# (N/A) rather than a fixable "raise gpu_mem and retry" TODO.
MAX_GPU_MEM_UTIL = 0.95

# Coarse dispositions that exclude a cell from the fillable denominator.
NA_DISPOSITIONS = {"na"}

# Per-cell failure classes specific enough to override the job-level disposition
# (RFC §4.5): the client observed the server was up and this cell was rejected
# on its own merits. Weak client-side signals (requests_aborted / unknown) defer
# to the server-log-informed job class so they can't flip e.g. oom -> failed.
PER_CELL_OVERRIDE_CLASSES = {"low_success_rate"}

# THE invariant (RFC §4.4): a cell is `na` only with positive evidence of an
# irreducible limit. failure_class -> coarse disposition is the single source
# of truth for that policy and lives only in disposition_for_class().
NA_FAILURE_CLASSES = {"hw_infeasible", "low_success_rate"}  # + oom_kv_cache @ max util
FAILED_FAILURE_CLASSES = {
    "model_missing", "engine_crash", "requests_aborted", "timeout", "driver_fault",
}
# everything else (unknown, incomplete_partial, not_attempted, oom @ <max util)
# -> "todo": fillable work, never silently hidden as N/A.

PointKey = tuple[str, str, str, str, str, int]  # hw, model, backend, data_mode, profile, conc
ProfileKey = tuple[str, str, str, int, str, str, str]  # scope, host, model, tp, mode, backend, profile


@dataclass(frozen=True)
class JsonSource:
    label: str
    ref: str
    ok: bool
    error: str | None = None
    bytes_read: int = 0


@dataclass
class JobCoverage:
    job_id: str
    data_scope: str
    host: str
    hw_label: str
    model: str
    tp: int
    mode: str
    backend: str
    status: str
    reason: str | None
    attempt: int | None = None
    failure_metadata: dict[str, Any] | None = None
    expected: set[PointKey] = field(default_factory=set)
    present: set[PointKey] = field(default_factory=set)
    # Set by reset_stale_jobs when the job has hit the coverage requeue cap: its
    # still-missing cells have been retried the maximum number of times and still
    # can't be filled, so their 'todo' disposition is demoted to 'failed'.
    requeue_exhausted: bool = False

    @property
    def missing(self) -> set[PointKey]:
        return self.expected - self.present

    @property
    def is_stale_terminal(self) -> bool:
        return bool(self.missing) and self.status in BLOCKING_STATUSES


@dataclass
class ResetOutcome:
    reset: list[JobCoverage] = field(default_factory=list)
    exhausted: list[JobCoverage] = field(default_factory=list)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_url(ref: str) -> bool:
    return ref.startswith("http://") or ref.startswith("https://")


def load_json_ref(ref: str, label: str, timeout: float) -> tuple[Any | None, JsonSource]:
    try:
        if is_url(ref):
            req = urllib.request.Request(ref, headers={"User-Agent": "agentic-serve-sweep-reconcile/1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
        else:
            raw = Path(ref).read_bytes()
        return json.loads(raw), JsonSource(label=label, ref=ref, ok=True, bytes_read=len(raw))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return None, JsonSource(label=label, ref=ref, ok=False, error=str(exc))


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = yaml.safe_load(path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError(f"{path} did not parse to a YAML mapping")
    return manifest


def load_generated_state(sweep_yaml: Path, state_dir: Path) -> dict[str, Any]:
    publish_sweep_state.STATE_DIR = state_dir
    manifest = load_manifest(sweep_yaml)
    return publish_sweep_state.build_state(manifest)


def mode_to_data_mode(mode: str) -> str:
    if mode == "single":
        return "single-turn"
    if mode == "multi":
        return "multi-turn"
    return mode


def scope_matches(scope_value: str, scope: str) -> bool:
    return scope == "all" or compile_sweep.dashboard_scope_for(scope_value) == compile_sweep.dashboard_scope_for(scope)


def profile_key(item: dict[str, Any]) -> ProfileKey:
    return (
        compile_sweep.dashboard_scope_for(str(item.get("data_scope") or "archived")),
        str(item.get("host", "")),
        str(item.get("model", "")),
        int(item.get("tp", 0) or 0),
        str(item.get("mode", "")),
        str(item.get("backend", "vllm")),
        str(item.get("profile", "")),
    )


def point_from_row(row: dict[str, Any]) -> PointKey | None:
    cfg = row.get("config") or {}
    hardware = row.get("hardware")
    model = row.get("modelShort") or row.get("model_short")
    backend = cfg.get("backend") or "vllm"
    mode = cfg.get("mode")
    profile = cfg.get("profile")
    conc = cfg.get("concurrency")
    if not hardware or not model or not mode or not profile or conc is None:
        return None
    try:
        conc_i = int(conc)
    except (TypeError, ValueError):
        return None
    return (str(hardware), str(model), str(backend), str(mode), str(profile), conc_i)


def data_points(data: list[dict[str, Any]], scope: str) -> set[PointKey]:
    points: set[PointKey] = set()
    for row in data:
        row_scope = compile_sweep.dashboard_scope_for(str(row.get("dataScope") or "trace_replay"))
        if scope != "all" and row_scope != compile_sweep.dashboard_scope_for(scope):
            continue
        point = point_from_row(row)
        if point is not None:
            points.add(point)
    return points


def data_scope_counts(data: list[dict[str, Any]]) -> Counter[str]:
    return Counter(compile_sweep.dashboard_scope_for(str(row.get("dataScope") or "unknown")) for row in data)


def expected_by_job(
    state: dict[str, Any],
    scope: str,
    runnable_job_ids: set[str] | None,
) -> dict[str, JobCoverage]:
    blocked_profiles = {profile_key(item) for item in state.get("profile_infeasible", [])}
    jobs: dict[str, JobCoverage] = {}
    for cell in state.get("cells", []):
        data_scope = compile_sweep.dashboard_scope_for(str(cell.get("data_scope") or "archived"))
        if not scope_matches(data_scope, scope):
            continue
        output_scope = compile_sweep.dashboard_scope_for(scope) if scope != "all" else data_scope

        host = str(cell.get("host", ""))
        model = str(cell.get("model", ""))
        tp = int(cell.get("tp", 0) or 0)
        mode = str(cell.get("mode", ""))
        backend = str(cell.get("backend", "vllm"))
        hw_label = str(cell.get("hw_label", ""))
        status = str(cell.get("status") or "pending")
        reason = cell.get("reason")
        jid = publish_sweep_state.job_id(host, model, tp, mode, backend, ep=compile_sweep.ep_enabled(cell))
        if runnable_job_ids is not None and jid not in runnable_job_ids:
            continue

        cov = jobs.get(jid)
        if cov is None:
            cov = JobCoverage(
                job_id=jid,
                data_scope=output_scope,
                host=host,
                hw_label=hw_label,
                model=model,
                tp=tp,
                mode=mode,
                backend=backend,
                status=status,
                reason=str(reason) if reason else None,
                attempt=int(cell.get("attempt", 0) or 0),
                failure_metadata=cell.get("failure_metadata") if isinstance(cell.get("failure_metadata"), dict) else None,
            )
            jobs[jid] = cov

        data_mode = mode_to_data_mode(mode)
        profiles = [str(p) for p in cell.get("profiles") or []]
        concurrencies = [int(c) for c in cell.get("concurrencies") or []]
        for profile in profiles:
            if (data_scope, host, model, tp, mode, backend, profile) in blocked_profiles:
                continue
            for conc in concurrencies:
                cov.expected.add((hw_label, model, backend, data_mode, profile, conc))

    return jobs


def apply_present_points(jobs: dict[str, JobCoverage], present_points: set[PointKey]) -> None:
    for cov in jobs.values():
        cov.present = cov.expected & present_points


def parse_reset_statuses(raw: str) -> set[str]:
    statuses = {part.strip() for part in raw.split(",") if part.strip()}
    invalid = statuses - BLOCKING_STATUSES
    if invalid:
        allowed = ",".join(sorted(BLOCKING_STATUSES))
        raise argparse.ArgumentTypeError(f"invalid statuses {sorted(invalid)}; allowed: {allowed}")
    if not statuses:
        raise argparse.ArgumentTypeError("at least one status is required")
    return statuses


def parse_bench_jobs(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    if not path.exists():
        return rows
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) < 6:
            continue
        host, _model_path, tp, short, mode, backend = parts[:6]
        backend = backend or "vllm"
        try:
            tp_i = int(tp)
        except ValueError:
            continue
        ep = compile_sweep.ep_enabled_extra_env(parts[10]) if len(parts) >= 11 else False
        jid = publish_sweep_state.job_id(host.strip(), short.strip(), tp_i, mode.strip(), backend.strip(), ep=ep)
        rows[jid] = raw
    return rows


def compiled_bench_jobs(manifest: dict[str, Any]) -> tuple[dict[str, str], dict[str, dict[str, str]], str, int]:
    compile_sweep.validate(manifest)
    emitted, skipped = compile_sweep.compile_jobs(manifest)
    rows: dict[str, str] = {}
    rows_by_scope: dict[str, dict[str, str]] = defaultdict(dict)
    for cell, row in emitted:
        backend = str(cell.get("backend", "vllm"))
        jid = publish_sweep_state.job_id(
            str(cell["host"]),
            str(cell["model"]),
            int(cell["tp"]),
            str(cell["mode"]),
            backend,
            ep=compile_sweep.ep_enabled(cell),
        )
        rows[jid] = row
        rows_by_scope[compile_sweep.dashboard_scope_for(publish_sweep_state.cell_data_scope(cell))][jid] = row
    synthetic_emitted, _synthetic_skipped = compile_sweep.compile_jobs(manifest, "synthetic_distributional")
    for cell, row in synthetic_emitted:
        backend = str(cell.get("backend", "vllm"))
        jid = publish_sweep_state.job_id(
            str(cell["host"]),
            str(cell["model"]),
            int(cell["tp"]),
            str(cell["mode"]),
            backend,
            ep=compile_sweep.ep_enabled(cell),
        )
        rows_by_scope["synthetic_distributional"][jid] = row
    moe_ep_emitted, _moe_ep_skipped = compile_sweep.compile_jobs(manifest, "moe_ep")
    for cell, row in moe_ep_emitted:
        backend = str(cell.get("backend", "vllm"))
        jid = publish_sweep_state.job_id(
            str(cell["host"]),
            str(cell["model"]),
            int(cell["tp"]),
            str(cell["mode"]),
            backend,
            ep=compile_sweep.ep_enabled(cell),
        )
        rows_by_scope["moe_ep"][jid] = row
    return rows, rows_by_scope, compile_sweep.render_file(emitted), len(skipped)


def write_local_sweep_state(sweep_yaml: Path, state_dir: Path, out: Path) -> None:
    state = load_generated_state(sweep_yaml, state_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(state, indent=2) + "\n")


def target_state_dir_for_cov(state_dir: Path, cov: JobCoverage) -> Path:
    return state_dir if state_dir.name == cov.data_scope else state_dir / cov.data_scope


def read_int_file(path: Path, default: int = 0) -> int:
    try:
        return int(path.read_text().strip() or default)
    except (OSError, ValueError):
        return default


def write_coverage_blocker(
    target_state_dir: Path,
    cov: JobCoverage,
    *,
    scope: str,
    status: str,
    timestamp: str,
    requeue_count: int,
    max_requeues: int,
    reason: str,
) -> None:
    payload = {
        "generated_at": timestamp,
        "job_id": cov.job_id,
        "scope": scope,
        "status": status,
        "host": cov.host,
        "hardware": cov.hw_label,
        "model": cov.model,
        "tp": cov.tp,
        "mode": cov.mode,
        "backend": cov.backend,
        "present": len(cov.present),
        "expected": len(cov.expected),
        "missing": group_missing_for_job(cov.missing),
        "expected_points": points_payload(cov.expected),
        "present_points": points_payload(cov.present),
        "missing_points": missing_points_payload(cov),
        "missing_count": len(cov.missing),
        "attempt": cov.attempt,
        "failure": failure_payload(cov),
        "requeue_count": requeue_count,
        "max_requeues": max_requeues,
        "reason": reason,
    }
    (target_state_dir / f"{cov.job_id}.{COVERAGE_BLOCKER_SUFFIX}").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def reset_stale_jobs(
    jobs: list[JobCoverage],
    state_dir: Path,
    reset_statuses: set[str],
    scope: str,
    write_reason: bool,
    max_requeues: int,
) -> ResetOutcome:
    targets = [cov for cov in jobs if cov.status in reset_statuses and cov.missing]
    outcome = ResetOutcome()
    timestamp = now_iso()
    for cov in targets:
        target_state_dir = target_state_dir_for_cov(state_dir, cov)
        target_state_dir.mkdir(parents=True, exist_ok=True)
        count_path = target_state_dir / f"{cov.job_id}.{COVERAGE_REQUEUE_COUNT_SUFFIX}"
        current_count = read_int_file(count_path)
        if max_requeues >= 0 and current_count >= max_requeues:
            root_cause = failure_summary(cov)
            reason = (
                f"coverage incomplete for scope={scope}: "
                f"missing {len(cov.missing)}/{len(cov.expected)} points; "
                f"coverage requeue limit reached {current_count}/{max_requeues}"
            )
            if root_cause:
                reason = f"{reason}; last failure: {root_cause}"
            if write_reason:
                (target_state_dir / f"{cov.job_id}.reason").write_text(reason + "\n")
            write_coverage_blocker(
                target_state_dir,
                cov,
                scope=scope,
                status="requeue_exhausted",
                timestamp=timestamp,
                requeue_count=current_count,
                max_requeues=max_requeues,
                reason=reason,
            )
            cov.requeue_exhausted = True
            outcome.exhausted.append(cov)
            continue

        next_count = current_count + 1
        (target_state_dir / f"{cov.job_id}.status").write_text("pending\n")
        count_path.write_text(f"{next_count}\n")
        if write_reason:
            reason = (
                f"coverage incomplete for scope={scope}: "
                f"missing {len(cov.missing)}/{len(cov.expected)} points; "
                f"coverage requeue {next_count}/{max_requeues if max_requeues >= 0 else 'unlimited'} "
                f"by reconcile_sweep_coverage.py at {timestamp}"
            )
            (target_state_dir / f"{cov.job_id}.reason").write_text(reason + "\n")
        write_coverage_blocker(
            target_state_dir,
            cov,
            scope=scope,
            status="requeued",
            timestamp=timestamp,
            requeue_count=next_count,
            max_requeues=max_requeues,
            reason=reason if write_reason else "coverage incomplete; reset to pending",
        )
        outcome.reset.append(cov)
    return outcome


def group_missing_for_job(points: set[PointKey]) -> str:
    grouped: dict[str, list[int]] = defaultdict(list)
    for _hw, _model, _backend, _mode, profile, conc in points:
        grouped[profile].append(conc)
    parts = []
    for profile, concs in sorted(grouped.items()):
        shown = ",".join(str(c) for c in sorted(concs))
        parts.append(f"{profile}: C={shown}")
    return "; ".join(parts)


def point_payload(points: set[PointKey]) -> list[dict[str, Any]]:
    rows = []
    for hw, model, backend, mode, profile, conc in sorted(points):
        rows.append({
            "hardware": hw,
            "model": model,
            "backend": backend,
            "mode": mode,
            "profile": profile,
            "concurrency": conc,
        })
    return rows


def points_payload(points: set[PointKey]) -> list[dict[str, Any]]:
    return point_payload(points)


def optional_present_points(data: list[dict[str, Any]], scope: str, expected_points: set[PointKey]) -> set[PointKey]:
    """Return accepted synthetic points that are outside the required grid.

    Some synthetic runs intentionally exceed the current required sweep grid
    because a previously-waived high-concurrency or workaround cell completed.
    Keep those visible as optional coverage without letting them expand the
    required denominator or missing-work set.
    """
    if compile_sweep.dashboard_scope_for(scope) != "synthetic_distributional":
        return set()
    return {
        point
        for point in data_points(data, scope) - expected_points
        if point[4].endswith("-synth")
    }


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def derive_failure_class(metadata: dict[str, Any], reason: str | None, status: str) -> str:
    """Structured failure class for a terminal job.

    Prefers the launcher's explicit `failure_class` field. Falls back to a
    single centralized derivation from the legacy reason text + signals. This
    is the ONLY place reason strings are parsed; new runs carry `failure_class`
    directly and skip it entirely.
    """
    explicit = metadata.get("failure_class")
    if explicit in FAILURE_CLASSES:
        return explicit
    text = (reason or "").lower()
    if status == "known_oom":
        return "oom_kv_cache"
    if any(t in text for t in ("xid", "nvml", "driver", "gpu has fallen off",
                               "cuda error", "uncorrectable", "nvidia-smi",
                               "cuda initialization")):
        return "driver_fault"
    if any(t in text for t in ("can't load the configuration", "repo id must be",
                               "no such file or directory", "is not a valid model",
                               "no usable temporary directory")):
        return "model_missing"
    if any(t in text for t in ("no available memory for the cache blocks",
                               "out of memory", "cuda out of memory",
                               "kv-cache", "kv cache", "cache blocks")):
        return "oom_kv_cache"
    if "engine core" in text and ("fail" in text or "initialization failed" in text):
        return "engine_crash"
    if "success rate" in text and "below minimum" in text:
        return "low_success_rate"
    if any(t in text for t in ("requests failed", "no requests completed",
                               "server may not be functional", "abort:")):
        return "requests_aborted"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    present = metadata.get("expected_outputs_present")
    total = metadata.get("expected_outputs_total")
    if isinstance(present, int) and isinstance(total, int) and 0 < present < total:
        return "incomplete_partial"
    # "zero results / no retryable OOM" etc. -> we have NO positive evidence of a
    # cause, so it is NOT proven infeasible. The invariant: never N/A without
    # evidence; this stays fillable work to re-run / investigate.
    return "unknown"


def failure_class_label(failure_class: str) -> str:
    return {
        "none": "ok",
        "model_missing": "model not staged",
        "hw_infeasible": "infeasible (won't fit)",
        "oom_kv_cache": "OOM / KV-cache limit",
        "engine_crash": "engine crash",
        "requests_aborted": "requests aborted",
        "low_success_rate": "success rate below threshold",
        "timeout": "timeout",
        "incomplete_partial": "partial outputs",
        "not_attempted": "not attempted",
        "driver_fault": "driver fault",
        "unknown": "no captured cause",
    }.get(failure_class, failure_class.replace("_", " "))


def disposition_for_class(failure_class: str, evidence: dict[str, Any], status: str) -> str:
    """SINGLE SOURCE OF TRUTH: failure_class -> coarse disposition (na/todo/failed).

    Enforces the RFC invariant: `na` only with positive evidence of an
    irreducible limit (a captured OOM at max gpu_mem, a measured low success
    rate, or static infeasibility). Everything else is `todo` (fillable) or
    `failed` (has a cause to inspect) -- never silently N/A.
    """
    if status == "known_oom":
        return "na"
    if failure_class in NA_FAILURE_CLASSES:
        return "na"
    if failure_class == "oom_kv_cache":
        util = _to_float((evidence or {}).get("gpu_mem_util"))
        # A captured OOM is irreducible only at/above max util; below it, it is
        # a fixable "raise gpu_mem and retry" TODO (the 3090 vllm gpt-oss case).
        return "todo" if (util is not None and util < MAX_GPU_MEM_UTIL) else "na"
    if failure_class in FAILED_FAILURE_CLASSES:
        return "failed"
    return "todo"


def disposition_label_for(failure_class: str, disposition: str | None) -> str:
    if disposition is None:
        return ""
    specific = {
        ("na", "hw_infeasible"): "N/A — infeasible (won't fit)",
        ("na", "oom_kv_cache"): "N/A — OOM at max gpu_mem",
        ("na", "low_success_rate"): "N/A — low success rate",
        ("failed", "model_missing"): "failed — model not staged",
        ("failed", "engine_crash"): "failed — engine crash",
        ("failed", "requests_aborted"): "failed — server up, requests aborted",
        ("failed", "timeout"): "failed — timeout",
        ("failed", "driver_fault"): "failed — driver fault",
        ("todo", "oom_kv_cache"): "TODO — raise gpu_mem and retry",
        ("todo", "incomplete_partial"): "TODO — partial, re-queue",
    }
    return specific.get(
        (disposition, failure_class),
        {"na": "N/A", "failed": "failed — inspect", "todo": "TODO"}[disposition],
    )


def _evidence_for(metadata: dict[str, Any]) -> dict[str, Any]:
    """Structured evidence: prefer the launcher's `evidence` block, backfill
    from the flat legacy fields so the mapping has what it needs."""
    evidence = dict(metadata.get("evidence") or {})
    evidence.setdefault("outputs_present", metadata.get("expected_outputs_present"))
    evidence.setdefault("outputs_expected", metadata.get("expected_outputs_total"))
    if "gpu_mem_util" not in evidence and metadata.get("gpu_mem_util") is not None:
        evidence["gpu_mem_util"] = metadata.get("gpu_mem_util")
    return evidence


def failure_payload(cov: JobCoverage) -> dict[str, Any] | None:
    metadata = cov.failure_metadata or {}
    reason = str(metadata.get("reason") or cov.reason or "")
    if not metadata and not reason:
        return None
    failure_class = derive_failure_class(metadata, reason, cov.status)
    evidence = _evidence_for(metadata)
    attempt = metadata.get("attempt", cov.attempt)
    return {
        "failure_class": failure_class,
        # `category` kept as an alias of failure_class for older readers.
        "category": failure_class,
        "label": failure_class_label(failure_class),
        "evidence": evidence,
        "kind": metadata.get("kind"),
        "status": metadata.get("status", cov.status),
        "reason": reason or None,
        "attempt": attempt,
        "max_attempts": metadata.get("max_attempts"),
        "expected_outputs_present": metadata.get("expected_outputs_present"),
        "expected_outputs_total": metadata.get("expected_outputs_total"),
        "missing_outputs": metadata.get("missing_outputs") if isinstance(metadata.get("missing_outputs"), list) else [],
        "remote_log": metadata.get("remote_log"),
        "mirror_status": metadata.get("mirror_status"),
        "updated_at": metadata.get("updated_at"),
    }


def failure_summary(cov: JobCoverage) -> str | None:
    failure = failure_payload(cov)
    if not failure:
        return None
    attempt = failure.get("attempt")
    max_attempts = failure.get("max_attempts")
    attempts = ""
    if attempt is not None and max_attempts is not None:
        attempts = f" after {attempt}/{max_attempts} attempts"
    elif attempt is not None:
        attempts = f" after {attempt} attempts"
    reason = str(failure.get("reason") or "").strip()
    if len(reason) > 240:
        reason = reason[:237] + "..."
    return f"{failure.get('label')}{attempts}{(': ' + reason) if reason else ''}"


def coverage_disposition(cov: JobCoverage) -> str | None:
    """Classify terminal missing coverage.

    - "na":     reached a terminal, explainable limit -- a captured OOM or a
                low-success rejection after the retry path. Excluded from the
                fillable denominator.
    - "todo":   attempted but produced nothing AND no OOM was captured, so the
                cell is not proven infeasible. Kept as fillable TODO work so it
                gets re-queued / investigated instead of hidden as N/A.
    - "failed": a real failure (benchmark / driver / unknown) needing
                inspection. Stays red.
    """
    if not cov.missing or cov.status not in BLOCKING_STATUSES:
        return None
    fp = failure_payload(cov) or {}
    disposition = disposition_for_class(
        str(fp.get("failure_class") or "unknown"), fp.get("evidence") or {}, cov.status
    )
    # Exhausted the coverage requeue cap: retried the maximum number of times and
    # still can't fill these cells, so they are no longer freely-fillable 'todo'
    # work -- surface as 'failed' (needs inspection) instead of advertising them
    # as queueable on the coverage page.
    if disposition == "todo" and cov.requeue_exhausted:
        return "failed"
    return disposition


def coverage_disposition_label(cov: JobCoverage, disposition: str | None) -> str:
    """Human label for the disposition, specialised by failure_class."""
    fp = failure_payload(cov) or {}
    return disposition_label_for(str(fp.get("failure_class") or "unknown"), disposition)


def coverage_explanation(cov: JobCoverage, disposition: str | None) -> str | None:
    label = coverage_disposition_label(cov, disposition)
    summary = failure_summary(cov) or cov.reason
    if not summary:
        return label or None
    if disposition in ("na", "todo", "failed"):
        return f"{label}: {summary}"
    return summary


# --- Per-cell granularity (RFC §4.5) ---------------------------------------
# A single job covers many (profile, concurrency) cells. The launcher captures
# a per-cell failure class in failure_metadata["cell_outcomes"] keyed by
# "<profile>|<concurrency>" so a job that serves at low concurrency but fails
# at high concurrency splits correctly (e.g. conc 1-80 done, 120+ na_quality)
# instead of smearing one disposition across every missing cell.

def cell_failure_class(cov: JobCoverage, profile: str, conc: int) -> str | None:
    outcomes = (cov.failure_metadata or {}).get("cell_outcomes")
    if not isinstance(outcomes, dict):
        return None
    fc = outcomes.get(f"{profile}|{conc}")
    return fc if fc in FAILURE_CLASSES else None


def cell_effective_class(cov: JobCoverage, point: PointKey) -> str | None:
    """The failure class that drives this cell's disposition: the per-cell
    class only when it is specific enough to override (PER_CELL_OVERRIDE_CLASSES),
    otherwise the job-level class."""
    fc = cell_failure_class(cov, point[4], point[5])
    if fc in PER_CELL_OVERRIDE_CLASSES:
        return fc
    return (failure_payload(cov) or {}).get("failure_class")


def cell_disposition(cov: JobCoverage, point: PointKey) -> str | None:
    """Disposition for one missing cell. A cell-specific class (e.g. a
    low-success rejection at high concurrency) overrides the job level; weak
    client-side signals defer to the job-level (server-log-informed) class."""
    if not cov.missing or cov.status not in BLOCKING_STATUSES:
        return None
    fc = cell_failure_class(cov, point[4], point[5])
    if fc in PER_CELL_OVERRIDE_CLASSES:
        disposition = disposition_for_class(fc, _evidence_for(cov.failure_metadata or {}), cov.status)
        if disposition == "todo" and cov.requeue_exhausted:
            return "failed"
        return disposition
    return coverage_disposition(cov)


def cell_label(cov: JobCoverage, point: PointKey, disposition: str | None) -> str:
    fc = cell_failure_class(cov, point[4], point[5])
    if fc in PER_CELL_OVERRIDE_CLASSES:
        return disposition_label_for(fc, disposition)
    return coverage_disposition_label(cov, disposition)


def missing_points_payload(cov: JobCoverage) -> list[dict[str, Any]]:
    """Missing points annotated with their effective failure_class/disposition."""
    rows = point_payload(cov.missing)
    for row in rows:
        point: PointKey = (
            row["hardware"], row["model"], row["backend"],
            row["mode"], row["profile"], row["concurrency"],
        )
        disposition = cell_disposition(cov, point)
        row["failure_class"] = cell_effective_class(cov, point)
        row["disposition"] = disposition
        row["label"] = cell_label(cov, point, disposition)
    return rows


def format_job(cov: JobCoverage) -> str:
    return f"{cov.host}/{cov.hw_label}/{cov.model}/tp{cov.tp}/{cov.backend}/{cov.mode}"


def build_report(
    *,
    scope: str,
    data_source: JsonSource,
    data: list[dict[str, Any]],
    jobs: dict[str, JobCoverage],
    current_rows: dict[str, str],
    compiled_rows: dict[str, str],
    compiled_scope_rows: dict[str, str],
    compiled_skipped: int,
    reset_statuses: set[str],
    reset_outcome: ResetOutcome,
    max_requeues: int,
    limit: int,
    wrote_bench_jobs: bool,
    wrote_missing_jobs: Path | None,
    wrote_sweep_state: Path | None,
    wrote_blockers_json: Path | None,
) -> str:
    all_jobs = sorted(jobs.values(), key=lambda c: (c.host, c.hw_label, c.model, c.tp, c.backend, c.mode))
    missing_jobs = [cov for cov in all_jobs if cov.missing]
    stale_jobs = [cov for cov in missing_jobs if cov.status in BLOCKING_STATUSES]
    reset_candidates = [cov for cov in stale_jobs if cov.status in reset_statuses]
    expected_points = {point for cov in all_jobs for point in cov.expected}
    present_points = {point for cov in all_jobs for point in cov.present}
    optional_points = optional_present_points(data, scope, expected_points)
    missing_points = expected_points - present_points
    coverage_na_points = {
        point
        for cov in stale_jobs
        for point in cov.missing
        if cell_disposition(cov, point) == "na"
    }
    coverage_failed_points = {
        point
        for cov in stale_jobs
        for point in cov.missing
        if cell_disposition(cov, point) == "failed"
    }
    coverage_todo_points = {
        point
        for cov in stale_jobs
        for point in cov.missing
        if cell_disposition(cov, point) == "todo"
    }
    required_points = expected_points - coverage_na_points
    status_counts = Counter(cov.status for cov in all_jobs)
    missing_by_status = Counter(cov.status for cov in missing_jobs)

    drift_compiled_rows = compiled_scope_rows if scope != "all" else compiled_rows
    only_current = sorted(set(current_rows) - set(drift_compiled_rows))
    only_compiled = sorted(set(drift_compiled_rows) - set(current_rows))

    lines = [
        "# Sweep Coverage Reconcile",
        "",
        f"- generated_at: {now_iso()}",
        f"- scope: {scope}",
        f"- data source: {data_source.ref}",
        f"- data source status: {'ok' if data_source.ok else 'error'}"
        + (f" ({data_source.bytes_read} bytes)" if data_source.ok else f" ({data_source.error})"),
        f"- data rows: {len(data)} total, scopes={dict(data_scope_counts(data))}",
        f"- expected coverage points: {len(expected_points)}",
        f"- present expected points: {len(present_points)} / {len(expected_points)}",
        f"- optional present synthetic points outside required grid: {len(optional_points)}",
        f"- observed synthetic points: {len(present_points | optional_points)} / {len(required_points)} fillable ({len(expected_points)} grid points)",
        f"- missing expected points: {len(missing_points)}",
        f"- N/A attempted points (captured OOM + low-success): {len(coverage_na_points)}",
        f"- TODO demoted points (attempted, no captured OOM): {len(coverage_todo_points)}",
        f"- failed points needing inspection: {len(coverage_failed_points)}",
        f"- expected jobs: {len(all_jobs)}",
        f"- compiled runnable jobs for scope: {len(compiled_scope_rows)}",
        f"- jobs with missing coverage: {len(missing_jobs)}",
        f"- stale terminal/blocking jobs with missing coverage: {len(stale_jobs)}",
        f"- reset candidates for statuses {sorted(reset_statuses)}: {len(reset_candidates)}",
        f"- reset performed: {len(reset_outcome.reset)} jobs",
        f"- reset exhausted by coverage requeue limit {max_requeues if max_requeues >= 0 else 'unlimited'}: {len(reset_outcome.exhausted)} jobs",
        f"- job status counts: {dict(status_counts)}",
        f"- missing jobs by status: {dict(missing_by_status)}",
        "",
        "## Bench Jobs Drift",
        f"- current bench_jobs rows: {len(current_rows)}",
        f"- compiled runnable rows from sweep.yaml: {len(drift_compiled_rows)}",
        f"- compiled skipped rows: {compiled_skipped}",
        f"- rows in current bench_jobs only: {len(only_current)}",
        f"- rows in compiled sweep only: {len(only_compiled)}",
        f"- rewrote bench_jobs.txt: {wrote_bench_jobs}",
    ]

    if only_current[:limit]:
        lines.append("- current-only job ids:")
        lines.extend(f"  - {jid}" for jid in only_current[:limit])
    if len(only_current) > limit:
        lines.append(f"  - ... {len(only_current) - limit} more")
    if only_compiled[:limit]:
        lines.append("- compiled-only job ids:")
        lines.extend(f"  - {jid}" for jid in only_compiled[:limit])
    if len(only_compiled) > limit:
        lines.append(f"  - ... {len(only_compiled) - limit} more")

    lines.extend([
        "",
        "## Stale Terminal Or Blocking Jobs",
        "| job_id | status | job | present/expected | missing |",
        "|---|---:|---|---:|---|",
    ])
    for cov in stale_jobs[:limit]:
        lines.append(
            f"| {cov.job_id} | {cov.status} | {format_job(cov)} | "
            f"{len(cov.present)}/{len(cov.expected)} | {group_missing_for_job(cov.missing)} |"
        )
    if len(stale_jobs) > limit:
        lines.append(f"| ... | ... | {len(stale_jobs) - limit} more | ... | ... |")

    if reset_outcome.exhausted:
        lines.extend([
            "",
            "## Requeue Exhausted",
            "| job_id | status | job | present/expected | missing |",
            "|---|---:|---|---:|---|",
        ])
        for cov in reset_outcome.exhausted[:limit]:
            lines.append(
                f"| {cov.job_id} | {cov.status} | {format_job(cov)} | "
                f"{len(cov.present)}/{len(cov.expected)} | {group_missing_for_job(cov.missing)} |"
            )
        if len(reset_outcome.exhausted) > limit:
            lines.append(f"| ... | ... | {len(reset_outcome.exhausted) - limit} more | ... | ... |")

    lines.extend([
        "",
        "## Non-terminal Missing Jobs",
        "| job_id | status | job | present/expected | missing |",
        "|---|---:|---|---:|---|",
    ])
    non_terminal_missing = [cov for cov in missing_jobs if cov.status not in BLOCKING_STATUSES]
    for cov in non_terminal_missing[:limit]:
        lines.append(
            f"| {cov.job_id} | {cov.status} | {format_job(cov)} | "
            f"{len(cov.present)}/{len(cov.expected)} | {group_missing_for_job(cov.missing)} |"
        )
    if len(non_terminal_missing) > limit:
        lines.append(f"| ... | ... | {len(non_terminal_missing) - limit} more | ... | ... |")

    lines.extend([
        "",
        "## Outputs",
        f"- missing jobs file: {wrote_missing_jobs if wrote_missing_jobs else 'not written'}",
        f"- local sweep-state.json: {wrote_sweep_state if wrote_sweep_state else 'not written'}",
        f"- blockers json: {wrote_blockers_json if wrote_blockers_json else 'not written'}",
        "",
        "## Stop Condition",
        f"- Complete for scope={scope} means every expected point from sweep.yaml exists in data.json.",
        "- If stale terminal/blocking jobs are nonzero, the orchestrator can believe there is no work left while coverage is still missing.",
    ])
    return "\n".join(lines) + "\n"


def coverage_payload(
    *,
    scope: str,
    data_source: JsonSource,
    data: list[dict[str, Any]],
    jobs: dict[str, JobCoverage],
    reset_statuses: set[str],
    reset_outcome: ResetOutcome,
    max_requeues: int,
) -> dict[str, Any]:
    all_jobs = sorted(jobs.values(), key=lambda c: (c.host, c.hw_label, c.model, c.tp, c.backend, c.mode))
    missing_jobs = [cov for cov in all_jobs if cov.missing]
    stale_jobs = [cov for cov in missing_jobs if cov.status in BLOCKING_STATUSES]
    expected_points = {point for cov in all_jobs for point in cov.expected}
    present_points = {point for cov in all_jobs for point in cov.present}
    optional_points = optional_present_points(data, scope, expected_points)
    coverage_na_points = {
        point
        for cov in stale_jobs
        for point in cov.missing
        if cell_disposition(cov, point) == "na"
    }
    coverage_failed_points = {
        point
        for cov in stale_jobs
        for point in cov.missing
        if cell_disposition(cov, point) == "failed"
    }
    coverage_todo_points = {
        point
        for cov in stale_jobs
        for point in cov.missing
        if cell_disposition(cov, point) == "todo"
    }
    coverage_required_points = expected_points - coverage_na_points
    status_counts = Counter(cov.status for cov in all_jobs)
    missing_by_status = Counter(cov.status for cov in missing_jobs)

    def cov_payload(cov: JobCoverage, *, include_missing_points: bool) -> dict[str, Any]:
        failure = failure_payload(cov)
        disposition = coverage_disposition(cov)
        explanation = coverage_explanation(cov, disposition)
        payload = {
            "job_id": cov.job_id,
            "status": cov.status,
            "scope": cov.data_scope,
            "host": cov.host,
            "hardware": cov.hw_label,
            "model": cov.model,
            "tp": cov.tp,
            "mode": cov.mode,
            "backend": cov.backend,
            "present": len(cov.present),
            "expected": len(cov.expected),
            "missing_count": len(cov.missing),
            "missing": group_missing_for_job(cov.missing),
            "present_points": points_payload(cov.present),
            "attempt": cov.attempt,
            "failure": failure,
            "reason": cov.reason,
            "coverage_disposition": disposition,
            "coverage_failure_class": (failure or {}).get("failure_class"),
            "coverage_label": coverage_disposition_label(cov, disposition),
            "coverage_evidence": (failure or {}).get("evidence"),
            "coverage_explanation": explanation,
        }
        if include_missing_points:
            payload["missing_points"] = missing_points_payload(cov)
        return payload

    failure_category_counts = Counter(
        failure.get("category", "unknown")
        for cov in stale_jobs
        if (failure := failure_payload(cov))
    )
    disposition_counts = Counter(
        disposition
        for cov in stale_jobs
        if (disposition := coverage_disposition(cov))
    )
    disposition_point_counts = Counter()
    for cov in stale_jobs:
        disposition = coverage_disposition(cov)
        if disposition:
            disposition_point_counts[disposition] += len(cov.missing)

    return {
        "generated_at": now_iso(),
        "scope": scope,
        "data_source": {
            "ref": data_source.ref,
            "ok": data_source.ok,
            "error": data_source.error,
            "bytes_read": data_source.bytes_read,
        },
        "data_rows": len(data),
        "data_scopes": dict(data_scope_counts(data)),
        "expected_points": len(expected_points),
        "coverage_required_points": len(coverage_required_points),
        "coverage_missing_required_points": len((expected_points - present_points) - coverage_na_points),
        "coverage_na_points": len(coverage_na_points),
        "coverage_todo_points": len(coverage_todo_points),
        "coverage_failed_points": len(coverage_failed_points),
        "present_points": len(present_points),
        "observed_present_points": len(present_points | optional_points),
        "optional_present_points_count": len(optional_points),
        "optional_present_points": points_payload(optional_points),
        "missing_points": len(expected_points - present_points),
        "jobs_total": len(all_jobs),
        "jobs_with_missing_coverage": len(missing_jobs),
        "stale_terminal_jobs": len(stale_jobs),
        "job_status_counts": dict(status_counts),
        "missing_jobs_by_status": dict(missing_by_status),
        "reset_statuses": sorted(reset_statuses),
        "max_requeues": max_requeues,
        "reset_performed": [cov.job_id for cov in reset_outcome.reset],
        "reset_exhausted": [cov.job_id for cov in reset_outcome.exhausted],
        "failure_category_counts": dict(failure_category_counts),
        "failure_disposition_counts": dict(disposition_counts),
        "failure_disposition_point_counts": dict(disposition_point_counts),
        "jobs": [cov_payload(cov, include_missing_points=False) for cov in all_jobs],
        "blockers": [cov_payload(cov, include_missing_points=True) for cov in stale_jobs],
    }


def write_blockers_json(
    path: Path,
    *,
    scope: str,
    data_source: JsonSource,
    data: list[dict[str, Any]],
    jobs: dict[str, JobCoverage],
    reset_statuses: set[str],
    reset_outcome: ResetOutcome,
    max_requeues: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = coverage_payload(
        scope=scope,
        data_source=data_source,
        data=data,
        jobs=jobs,
        reset_statuses=reset_statuses,
        reset_outcome=reset_outcome,
        max_requeues=max_requeues,
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def narrow_row_to_missing(row: str, cov: JobCoverage) -> str:
    """Narrow a compiled bench-jobs row to only the profiles/concurrencies still
    missing for this job, so a coverage re-queue re-runs the missing cells rather
    than the whole grid. The row packs concs and profiles as separate space-lists,
    so the tightest expressible subset is {missing profiles} x {missing concs} --
    a superset of the exact missing cells but far smaller than the full grid.
    Falls back to the original row on an unexpected format or empty intersection
    (e.g. a never-run job whose whole grid is missing stays the full row)."""
    parts = row.split("|")
    if len(parts) < 11:
        return row
    missing_profiles = {str(p[4]) for p in cov.missing}
    missing_concs = {str(p[5]) for p in cov.missing}
    concs = [c for c in parts[8].split() if c in missing_concs]
    profiles = [p for p in parts[9].split() if p in missing_profiles]
    if not concs or not profiles:
        return row
    parts[8] = " ".join(concs)
    parts[9] = " ".join(profiles)
    return "|".join(parts)


def write_missing_bench_jobs(path: Path, missing_jobs: list[JobCoverage], compiled_rows: dict[str, str], scope: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Benchmark job subset with missing coverage.",
        "# GENERATED by scripts/reconcile_sweep_coverage.py.",
        f"# SCOPE: {scope}",
        "# Format matches scripts/bench_jobs.txt.",
        "",
    ]
    for cov in sorted(missing_jobs, key=lambda c: (c.host, c.hw_label, c.model, c.tp, c.backend, c.mode)):
        row = compiled_rows.get(cov.job_id)
        if row:
            lines.append(narrow_row_to_missing(row, cov))
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=(
            "trace_replay",
            "synthetic_distributional",
            "synthetic-distributional",
            "archived",
            "synthetic",
            "latest",
            "current",
            "fixed",
            "mse",
            "archive",
            "moe_ep",
            "all",
        ),
        default="synthetic_distributional",
    )
    parser.add_argument("--data", default=DEFAULT_DATA, help="data.json file path or URL; defaults to published R2 current data")
    parser.add_argument("--sweep-yaml", type=Path, default=DEFAULT_SWEEP_YAML)
    parser.add_argument("--bench-jobs", type=Path, default=DEFAULT_BENCH_JOBS)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--sweep-state-out", type=Path, default=DEFAULT_SWEEP_STATE)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="write markdown report to this path")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--write-bench-jobs", action="store_true", help="rewrite bench_jobs.txt from sweep.yaml")
    parser.add_argument(
        "--write-missing-jobs",
        nargs="?",
        const=str(DEFAULT_MISSING_JOBS),
        default=None,
        help="write a bench_jobs-format subset containing jobs with missing coverage",
    )
    parser.add_argument("--write-sweep-state", action="store_true", help="refresh local dashboard/public/sweep-state.json")
    parser.add_argument("--reset-stale", action="store_true", help="reset stale terminal jobs with missing coverage to pending")
    parser.add_argument(
        "--reset-statuses",
        type=parse_reset_statuses,
        default=DEFAULT_RESET_STATUSES,
        help="comma-separated terminal statuses to reset; default: done",
    )
    parser.add_argument(
        "--max-coverage-requeues",
        type=int,
        default=1,
        help="maximum automatic coverage requeues per job; use -1 for unlimited",
    )
    parser.add_argument(
        "--write-blockers-json",
        type=Path,
        default=None,
        help="write machine-readable coverage blocker/requeue summary JSON",
    )
    parser.add_argument("--no-reset-reason", action="store_true", help="do not write/reset .reason files")
    parser.add_argument("--fail-on-missing", action="store_true")
    parser.add_argument("--fail-on-stale", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.sweep_yaml)
    compiled_rows, compiled_rows_by_scope, compiled_text, compiled_skipped = compiled_bench_jobs(manifest)
    if args.scope == "all":
        compiled_scope_rows = compiled_rows
    else:
        compiled_scope_rows = compiled_rows_by_scope.get(compile_sweep.dashboard_scope_for(args.scope), {})
    current_rows = parse_bench_jobs(args.bench_jobs)

    data_raw, data_source = load_json_ref(args.data, "coverage data", args.timeout)
    if not data_source.ok:
        print(f"failed to read {args.data}: {data_source.error}", file=sys.stderr)
        return 2
    if not isinstance(data_raw, list):
        print(f"{args.data} did not contain a JSON array", file=sys.stderr)
        return 2
    data: list[dict[str, Any]] = [row for row in data_raw if isinstance(row, dict)]

    state = load_generated_state(args.sweep_yaml, args.state_dir)
    jobs = expected_by_job(state, args.scope, set(compiled_scope_rows))
    apply_present_points(jobs, data_points(data, args.scope))

    all_jobs = list(jobs.values())
    missing_jobs = [cov for cov in all_jobs if cov.missing]
    stale_jobs = [cov for cov in missing_jobs if cov.status in BLOCKING_STATUSES]

    reset_outcome = ResetOutcome()
    if args.reset_stale:
        reset_outcome = reset_stale_jobs(
            stale_jobs,
            args.state_dir,
            args.reset_statuses,
            args.scope,
            write_reason=not args.no_reset_reason,
            max_requeues=args.max_coverage_requeues,
        )

    wrote_bench_jobs = False
    if args.write_bench_jobs:
        args.bench_jobs.write_text(compiled_text)
        wrote_bench_jobs = True

    wrote_missing_jobs: Path | None = None
    if args.write_missing_jobs:
        wrote_missing_jobs = Path(args.write_missing_jobs)
        write_missing_bench_jobs(wrote_missing_jobs, missing_jobs, compiled_scope_rows, args.scope)

    wrote_sweep_state: Path | None = None
    if args.write_sweep_state:
        write_local_sweep_state(args.sweep_yaml, args.state_dir, args.sweep_state_out)
        wrote_sweep_state = args.sweep_state_out

    wrote_blockers_json: Path | None = None
    if args.write_blockers_json:
        write_blockers_json(
            args.write_blockers_json,
            scope=args.scope,
            data_source=data_source,
            data=data,
            jobs=jobs,
            reset_statuses=args.reset_statuses,
            reset_outcome=reset_outcome,
            max_requeues=args.max_coverage_requeues,
        )
        wrote_blockers_json = args.write_blockers_json

    report = build_report(
        scope=args.scope,
        data_source=data_source,
        data=data,
        jobs=jobs,
        current_rows=current_rows,
        compiled_rows=compiled_rows,
        compiled_scope_rows=compiled_scope_rows,
        compiled_skipped=compiled_skipped,
        reset_statuses=args.reset_statuses,
        reset_outcome=reset_outcome,
        max_requeues=args.max_coverage_requeues,
        limit=args.limit,
        wrote_bench_jobs=wrote_bench_jobs,
        wrote_missing_jobs=wrote_missing_jobs,
        wrote_sweep_state=wrote_sweep_state,
        wrote_blockers_json=wrote_blockers_json,
    )

    if not args.no_report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report)
        print(f"wrote {args.report}", file=sys.stderr)
    sys.stdout.write(report)

    if args.fail_on_stale and stale_jobs:
        return 4
    if args.fail_on_missing and any(cov.missing for cov in all_jobs):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
