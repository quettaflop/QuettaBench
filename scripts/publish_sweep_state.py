#!/usr/bin/env python3
"""Build sweep-state.json from sweep.yaml + orchestrator state files, upload to R2.

For each cell in sweep.yaml (including `known_oom` entries), produce a status
record by reading /mnt/100g/agent-bench/state/<scope>/<job_id>.status. Legacy
/tmp state fallback is enabled during migration and can be disabled with
BENCH_STATE_LEGACY_FALLBACK=0. Cells without a state file get status "pending".
known_oom entries override runtime status.

Output is written to dashboard/public/sweep-state.json and optionally uploaded
to R2 at s3://agent-bench/json/current/sweep-state.json so the live dashboard
can fetch it from the generated JSON prefix.

Run locally (no upload):
    python scripts/publish_sweep_state.py --no-upload

Run from orchestrator tick (default — uploads):
    python scripts/publish_sweep_state.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from compile_sweep import cell_data_scope as manifest_cell_data_scope
from compile_sweep import DEFAULT_INFEASIBILITY_KIND, MOE_EP_HOSTS, cell_for_output_scope, dashboard_scope_for, ep_enabled, is_moe_ep_scope, is_moe_model, matches_known_oom, parallelism_label, profile_infeasible_kinds, profile_infeasible_reasons, profiles_for_output_scope, resolve, result_scope_for

HERE = Path(__file__).resolve().parent
SWEEP_YAML = HERE / "sweep.yaml"
STATE_DIR = Path(os.environ.get("BENCH_STATE_ROOT", "/mnt/100g/agent-bench/state"))
LEGACY_STATE_DIR = Path(os.environ.get("BENCH_LEGACY_STATE_ROOT", "/tmp/bench_jobs/state"))
# Dashboard-JSON artifact output lands in the neutral artifact dir (env-overridable);
# the dashboard tree now lives in the separate QuettaBoard repo.
OUTPUT_FILE = Path(os.environ.get("BENCH_ARTIFACT_DIR", "/mnt/100g/agent-bench/artifacts")) / "sweep-state.json"

R2_ENDPOINT_DEFAULT = "https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com"
R2_BUCKET_DEFAULT = "agent-bench"
R2_KEY = "json/current/sweep-state.json"
LEGACY_STATE_FALLBACK = os.environ.get("BENCH_STATE_LEGACY_FALLBACK", "1") != "0"
DERIVED_STATE_SCOPES = ("synthetic_distributional", "moe_ep")
ACTIVE_RUN_STATUSES = {"dispatching", "running"}


def hw_label(host_cfg: dict, tp: int) -> str:
    base = str(host_cfg["hardware_label"])
    return base if tp == 1 else f"{base}x{tp}"


def job_id(host: str, model: str, tp: int, mode: str, backend: str = "vllm", ep: bool = False) -> str:
    jid = f"{host}_{model}_tp{tp}_{mode}"
    if backend != "vllm":
        jid += f"_{backend}"
    # NOTE: no `_ep` suffix. EP-on runs are already isolated from their EP-off
    # siblings by the per-scope state/result dirs (state/moe_ep/ vs
    # state/synthetic_distributional/), and the orchestrator writes those files
    # WITHOUT an _ep suffix. Appending one here made the reader look for
    # `<jid>_ep.status`, miss every EP job's state, treat it as pending, and drop
    # all failure dispositions (coverage showed 0 na/todo/failed). `ep` is kept
    # in the signature for call-site compatibility.
    return jid


def cell_data_scope(cell: dict) -> str:
    return manifest_cell_data_scope(cell)


def state_scope_aliases(data_scope: str) -> list[str]:
    aliases = [data_scope]
    if data_scope == "synthetic_distributional":
        aliases.append("synthetic")
    elif data_scope == "trace_replay":
        aliases.append("archive")
    return aliases


def state_file(jid: str, data_scope: str, suffix: str) -> Path:
    primary = None
    for scope in state_scope_aliases(data_scope):
        candidate = STATE_DIR / f"{jid}.{suffix}" if STATE_DIR.name == scope else STATE_DIR / scope / f"{jid}.{suffix}"
        if primary is None:
            primary = candidate
        if candidate.exists():
            return candidate

    assert primary is not None
    if not LEGACY_STATE_FALLBACK:
        return primary

    for scope in state_scope_aliases(data_scope):
        legacy_scoped = LEGACY_STATE_DIR / scope / f"{jid}.{suffix}"
        if legacy_scoped.exists():
            return legacy_scoped
    return LEGACY_STATE_DIR / f"{jid}.{suffix}"


def scoped_state_dir(data_scope: str) -> Path:
    if STATE_DIR.name == data_scope:
        return STATE_DIR
    return STATE_DIR / data_scope


def run_record_file(jid: str, data_scope: str, run_id: str) -> Path:
    return scoped_state_dir(data_scope) / "runs" / f"{run_id}.json"


def read_json_file(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def read_run_record(jid: str, data_scope: str, run_id: str) -> dict:
    if not run_id:
        return {}
    payload = read_json_file(run_record_file(jid, data_scope, run_id))
    if str(payload.get("job_id") or "") not in {"", jid}:
        return {}
    return payload


def read_state(jid: str, data_scope: str) -> dict:
    out: dict = {
        "status": "pending",
        "attempt": 0,
        "max_len_override": None,
        "reason": None,
        "updated_at": None,
        "run_id": None,
        "failure_metadata": None,
    }
    p = state_file(jid, data_scope, "status")
    if p.exists():
        out["status"] = p.read_text().strip() or "pending"
        out["updated_at"] = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()
    att = state_file(jid, data_scope, "attempt")
    if att.exists():
        try:
            out["attempt"] = int(att.read_text().strip())
        except ValueError:
            pass
    ov = state_file(jid, data_scope, "max_len_override")
    if ov.exists():
        try:
            out["max_len_override"] = int(ov.read_text().strip())
        except ValueError:
            pass
    rs = state_file(jid, data_scope, "reason")
    if rs.exists():
        txt = rs.read_text().strip()
        if txt:
            out["reason"] = txt
    rid = state_file(jid, data_scope, "run_id")
    if rid.exists():
        run_id = rid.read_text().strip()
        if run_id:
            out["run_id"] = run_id
            run_record = read_run_record(jid, data_scope, run_id)
            run_status = str(run_record.get("status") or "")
            if run_status in ACTIVE_RUN_STATUSES:
                out["status"] = "running"
                updated_at = run_record.get("updated_at") or run_record.get("started_at")
                if isinstance(updated_at, str) and updated_at:
                    out["updated_at"] = updated_at
            failure = run_record.get("failure")
            if isinstance(failure, dict) and failure and out["failure_metadata"] is None:
                out["failure_metadata"] = failure
    failure_path = state_file(jid, data_scope, "failure.json")
    failure_metadata = read_json_file(failure_path)
    if failure_metadata:
        out["failure_metadata"] = failure_metadata
    return out


def has_signature(jid: str, data_scope: str) -> bool:
    return state_file(jid, data_scope, "signature").exists()


def build_state(manifest: dict) -> dict:
    # Normalize host keys to strings — YAML parses unquoted `3090:` as int
    # but cells reference hosts by string name.
    hosts = {str(k): v for k, v in manifest["hosts"].items()}
    cells = []
    profile_infeasible = []

    # One record per sweep cell, augmented with runtime state. Derived scopes
    # are materialized here so coverage status uses the same scope/profile names
    # as the rows emitted by compile_sweep.py.
    state_cells: list[tuple[dict, dict]] = [(cell, cell) for cell in manifest["cells"]]
    for scope in DERIVED_STATE_SCOPES:
        moe_ep = is_moe_ep_scope(scope)
        for source_cell in manifest["cells"]:
            if manifest_cell_data_scope(source_cell) != "fixed":
                continue
            if moe_ep and not (
                is_moe_model(str(source_cell["model"]), manifest)
                and int(source_cell["tp"]) > 1
                and str(source_cell["host"]) in MOE_EP_HOSTS
            ):
                continue
            derived_cell = cell_for_output_scope(source_cell, scope, manifest)
            if not resolve(derived_cell, manifest)["profiles"]:
                continue
            state_cells.append((derived_cell, source_cell))

    for cell, source_cell in state_cells:
        host_name = str(cell["host"])
        host_cfg = hosts[host_name]
        tp = int(cell["tp"])
        mode = str(cell["mode"])
        model = str(cell["model"])
        backend = str(cell.get("backend", "vllm"))
        ep = ep_enabled(cell)
        source_scope = cell_data_scope(cell)
        data_scope = dashboard_scope_for(source_scope)
        state_scope = result_scope_for(source_scope)
        jid = job_id(host_name, model, tp, mode, backend, ep=ep)
        rt = read_state(jid, state_scope)
        resolved = resolve(cell, manifest)
        source_profile_reasons = profile_infeasible_reasons(
            source_cell,
            manifest,
            ignore_max_len_rules=data_scope in ("synthetic_distributional", "moe_ep"),
        )
        source_profile_kinds = profile_infeasible_kinds(
            source_cell,
            manifest,
            ignore_max_len_rules=data_scope in ("synthetic_distributional", "moe_ep"),
        )
        profile_reasons = {}
        profile_kinds = {}
        for profile, reason in source_profile_reasons.items():
            for output_profile in profiles_for_output_scope([profile], source_scope):
                profile_reasons[output_profile] = reason
                profile_kinds[output_profile] = source_profile_kinds.get(profile, DEFAULT_INFEASIBILITY_KIND)
        runnable_profiles = [
            str(profile) for profile in resolved["profiles"]
            if str(profile) not in profile_reasons
        ]
        if (
            rt["status"] in {"skipped", "failed"}
            and profile_reasons
            and runnable_profiles
            and not has_signature(jid, state_scope)
        ):
            rt["status"] = "pending"
            rt["reason"] = "legacy skipped state predates profile filtering; rerun reduced-profile job"

        for profile, reason in profile_reasons.items():
            profile_infeasible.append({
                "data_scope": data_scope,
                "source_scope": source_scope,
                "host": host_name,
                "hw_label": hw_label(host_cfg, tp),
                "model": model,
                "tp": tp,
                "mode": mode,
                "backend": backend,
                "profile": profile,
                "max_len": int(resolved["max_len"]),
                "reason": reason,
                "kind": profile_kinds.get(profile, DEFAULT_INFEASIBILITY_KIND),
            })

        cells.append({
            "data_scope": data_scope,
            "source_scope": source_scope,
            "host": host_name,
            "hw_label": hw_label(host_cfg, tp),
            "model": model,
            "tp": tp,
            "mode": mode,
            "backend": backend,
            "ep": ep,
            "parallelism": parallelism_label(ep, tp),
            "status": rt["status"],
            "attempt": rt["attempt"],
            "max_len": int(resolved["max_len"]),
            "gpu_mem": float(resolved["gpu_mem"]),
            "profiles": [str(p) for p in resolved["profiles"]],
            "concurrencies": [int(c) for c in resolved["concurrencies"]],
            "max_len_override": rt["max_len_override"],
            "reason": rt["reason"],
            "updated_at": rt["updated_at"],
            "run_id": rt["run_id"],
            "failure_metadata": rt["failure_metadata"],
        })

    # Known-OOM entries override any runtime state for matching cells, and are
    # appended as new (host, model, tp) entries if no sweep cell exists yet.
    for entry in manifest.get("known_oom", []):
        host_name = str(entry["host"])
        host_cfg = hosts[host_name]
        tp = int(entry["tp"])
        label = hw_label(host_cfg, tp)
        model = str(entry["model"])
        mode = str(entry.get("mode", "single"))
        backend = str(entry.get("backend", "vllm"))
        ep = ep_enabled(entry)
        kind = str(entry.get("kind", DEFAULT_INFEASIBILITY_KIND))
        matched = False
        for c in cells:
            if matches_known_oom(c, entry):
                c["status"] = "known_oom"
                c["reason"] = entry["reason"]
                c["kind"] = kind
                matched = True
        if not matched:
            cells.append({
                "data_scope": "archived",
                "source_scope": "known_oom",
                "host": host_name,
                "hw_label": label,
                "model": model,
                "tp": tp,
                "mode": mode,
                "backend": backend,
                "ep": ep,
                "parallelism": parallelism_label(ep, tp),
                "status": "known_oom",
                "attempt": 0,
                "max_len": None,
                "gpu_mem": None,
                "profiles": [],
                "concurrencies": [],
                "max_len_override": None,
                "reason": entry["reason"],
                "kind": kind,
                "updated_at": None,
            })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feasibility_ratio": float(manifest["feasibility_ratio"]),
        "hosts": {
            h: {
                "hardware_label": str(cfg["hardware_label"]),
                "vram_gb_per_gpu": int(cfg["vram_gb_per_gpu"]),
                "total_gpus": int(cfg["total_gpus"]),
            }
            for h, cfg in hosts.items()
        },
        "models": {
            m: {"weights_gb": int(cfg["weights_gb"])}
            for m, cfg in manifest["models"].items()
        },
        "cells": cells,
        "profile_infeasible": profile_infeasible,
    }


def upload_r2(path: Path, endpoint: str, bucket: str, profile: str) -> None:
    cmd = [
        "aws", "--profile", profile, "s3", "cp",
        str(path), f"s3://{bucket}/{R2_KEY}",
        "--endpoint-url", endpoint,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def main() -> int:
    global STATE_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", type=Path, default=SWEEP_YAML)
    ap.add_argument("--out", type=Path, default=OUTPUT_FILE)
    ap.add_argument("--no-upload", action="store_true", help="skip R2 upload")
    ap.add_argument("--state-dir", type=Path, default=Path(os.environ.get("BENCH_STATE_ROOT", str(STATE_DIR))))
    ap.add_argument("--endpoint", default=os.environ.get("R2_ENDPOINT", R2_ENDPOINT_DEFAULT))
    ap.add_argument("--bucket", default=os.environ.get("R2_BUCKET", R2_BUCKET_DEFAULT))
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE", "r2"))
    args = ap.parse_args()

    STATE_DIR = args.state_dir

    manifest = yaml.safe_load(args.yaml.read_text())
    state = build_state(manifest)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(state, indent=2) + "\n")
    print(f"wrote {args.out} ({len(state['cells'])} cells)", file=sys.stderr)

    if not args.no_upload:
        try:
            upload_r2(args.out, args.endpoint, args.bucket, args.profile)
            print(f"uploaded to s3://{args.bucket}/{R2_KEY}", file=sys.stderr)
        except subprocess.CalledProcessError as e:
            print(f"R2 upload failed: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
