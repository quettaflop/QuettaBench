#!/usr/bin/env python3
"""Compile sweep.yaml → benchmark launch manifests.

Reads the authoritative sweep matrix in sweep.yaml, applies the feasibility
rule and known_oom skiplist, and emits runnable cells. JSON is the structured
manifest format. The legacy pipe-delimited text format is kept for shell tools
that still consume row streams.

Run:  python scripts/compile_sweep.py
      python scripts/compile_sweep.py --dry-run   # print to stdout, don't write
      python scripts/compile_sweep.py --format json --scope synthetic_distributional
      python scripts/compile_sweep.py --verbose   # show skip reasons
"""
from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
SWEEP_YAML = HERE / "sweep.yaml"
BENCH_JOBS_TXT = HERE / "bench_jobs.txt"
BENCH_JOBS_JSON = HERE / "bench_jobs.json"

PRESET_KEYS = ("max_len", "gpu_mem", "concurrencies", "profiles")
CELL_REQUIRED = ("host", "model", "tp", "mode", "preset")
# Machine-readable infeasibility taxonomy shared by known_oom + profile_infeasible.
#   hw_permanent = the GPU physically can't (arch / compute capability); never fixable.
#   sw_fixable   = env/software gap (rebuild/upgrade the stack, bump a limit); fixable.
# Auto VRAM-feasibility skips are hardware limits, so the default is hw_permanent.
INFEASIBILITY_KINDS = ("hw_permanent", "sw_fixable")
DEFAULT_INFEASIBILITY_KIND = "hw_permanent"
SYNTHETIC_PROFILE_MAP = {
    "chat-singleturn": "chat-singleturn-synth",
    "chat-multiturn": "chat-multiturn-synth",
    "swebench-multiturn": "swebench-multiturn-synth",
    "terminalbench-multiturn": "terminalbench-multiturn-synth",
    "osworld-multiturn": "osworld-multiturn-synth",
}
SYNTHETIC_EXTRA_ENV = {
    "DISTRIBUTIONAL_SYNTHETIC_STYLE": "code",
    "DISTRIBUTIONAL_TARGET_CHARS_PER_TOKEN": "3.8",
    "DISTRIBUTIONAL_PREFIX_AWARE": "1",
    "DISTRIBUTIONAL_SHARED_PREFIX_TOKENS": "1024",
}
SYNTHETIC_TRACE_REPLAY_CONCURRENCIES = {
    "single": [1, 10, 20, 40, 80, 120, 160, 200, 256, 320, 500],
    "multi": [1, 5, 10, 20, 40, 80, 120, 160, 200, 256, 320],
}
DERIVED_SCOPE_SOURCE = {
    "latest": "fixed",  # legacy alias; the dashboard now exposes synthetic_distributional.
    "synthetic": "fixed",
    "synthetic-distributional": "fixed",
    "synthetic_distributional": "fixed",
    # EP-on MoE variant: derives from the same `fixed` cells and the same
    # synthetic launch shape as synthetic_distributional, but restricted to MoE
    # models at tp>1 on the EP hosts and with expert-parallelism enabled. Kept a
    # separate scope so its coverage never collides with the EP-off runs.
    "moe_ep": "fixed",
}
# Scopes that use the synthetic (distributional) launch shape: base profiles are
# remapped to their -synth variants and expanded onto the dense concurrency grid.
SYNTHETIC_SHAPE_SCOPES = {"synthetic_distributional", "moe_ep"}
# EP-on coverage is scoped to these hosts (the user's MoE/EP focus).
MOE_EP_HOSTS = {"h100", "h100-2", "a100", "3090"}


def load_manifest(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def validate(m: dict) -> None:
    for key in ("hosts", "models", "presets", "feasibility_ratio", "cells"):
        if key not in m:
            raise ValueError(f"sweep.yaml missing top-level key: {key}")
    for name, preset in m["presets"].items():
        missing = [k for k in PRESET_KEYS if k not in preset]
        if missing:
            raise ValueError(f"preset {name!r} missing keys: {missing}")
    for i, cell in enumerate(m["cells"]):
        missing = [k for k in CELL_REQUIRED if k not in cell]
        if missing:
            raise ValueError(f"cell #{i} missing keys: {missing}; cell={cell}")
        if cell["host"] not in m["hosts"]:
            raise ValueError(f"cell #{i}: unknown host {cell['host']!r}")
        if cell["model"] not in m["models"]:
            raise ValueError(f"cell #{i}: unknown model {cell['model']!r}")
        if cell["preset"] not in m["presets"]:
            raise ValueError(f"cell #{i}: unknown preset {cell['preset']!r}")
        if cell["mode"] not in ("single", "multi"):
            raise ValueError(f"cell #{i}: mode must be single|multi, got {cell['mode']!r}")
    for i, rule in enumerate(m.get("profile_infeasible", [])):
        if "reason" not in rule:
            raise ValueError(f"profile_infeasible #{i} missing reason")
        if "profiles" not in rule and "profile" not in rule:
            raise ValueError(f"profile_infeasible #{i} must specify profile or profiles")
        if "kind" in rule and rule["kind"] not in INFEASIBILITY_KINDS:
            raise ValueError(
                f"profile_infeasible #{i}: kind must be one of {INFEASIBILITY_KINDS}, got {rule['kind']!r}"
            )
    for i, entry in enumerate(m.get("known_oom", [])):
        if "kind" in entry and entry["kind"] not in INFEASIBILITY_KINDS:
            raise ValueError(
                f"known_oom #{i}: kind must be one of {INFEASIBILITY_KINDS}, got {entry['kind']!r}"
            )


# sglang's --mem-fraction-static reserves weights + KV pool and wants a higher
# fraction than vllm's --gpu-memory-utilization (sglang's own default is ~0.9).
# Bump every sglang run to at least this floor; higher explicit values are kept.
SGLANG_GPU_MEM_FLOOR = 0.92

# EP-on multi-turn sessions accumulate KV across turns, so the high-concurrency
# tail OOMs at the default budget. The EP-on derivation gets the full memory
# ceiling (reconcile's MAX_GPU_MEM_UTIL); the EP-off run of the same physical
# cell keeps its own gpu_mem. Applied only in the moe_ep scope (see
# cell_for_output_scope) so it never re-shapes an EP-off job.
EP_MULTI_GPU_MEM_FLOOR = 0.95


def resolve(cell: dict, manifest: dict) -> dict:
    """Merge preset defaults with cell overrides; return concrete launch params."""
    preset = manifest["presets"][cell["preset"]]
    out = {k: preset[k] for k in PRESET_KEYS}
    for k in PRESET_KEYS + ("extra_env",):
        if k in cell:
            out[k] = cell[k]
    if str(cell.get("backend", "")) == "sglang":
        out["gpu_mem"] = max(float(out["gpu_mem"]), SGLANG_GPU_MEM_FLOOR)
    return out


def matches_known_oom(cell: dict, entry: dict) -> bool:
    fields = {
        "host": str(cell["host"]),
        "model": str(cell["model"]),
        "tp": str(cell["tp"]),
        "mode": str(cell["mode"]),
        "backend": str(cell.get("backend", "vllm")),
    }
    for key, actual in fields.items():
        if key in entry and str(entry[key]) != actual:
            return False
    return True


def is_known_oom(cell: dict, manifest: dict) -> str | None:
    for entry in manifest.get("known_oom", []):
        if matches_known_oom(cell, entry):
            return entry["reason"]
    return None


def known_oom_kind(cell: dict, manifest: dict) -> str | None:
    """Infeasibility kind of the matching known_oom entry (default hw_permanent).

    Sibling of is_known_oom(): returns None when the cell is not known_oom, else
    the entry's `kind` (hw_permanent when the entry omits it).
    """
    for entry in manifest.get("known_oom", []):
        if matches_known_oom(cell, entry):
            return str(entry.get("kind", DEFAULT_INFEASIBILITY_KIND))
    return None


def feasibility_reason(cell: dict, manifest: dict) -> str | None:
    host = manifest["hosts"][cell["host"]]
    model = manifest["models"][cell["model"]]
    ratio = manifest["feasibility_ratio"]
    budget_gb = host["vram_gb_per_gpu"] * cell["tp"] * ratio
    if model["weights_gb"] > budget_gb:
        min_gb = math.ceil(model["weights_gb"] / ratio)
        have_gb = host["vram_gb_per_gpu"] * cell["tp"]
        return f"needs >={min_gb} GB VRAM (weights {model['weights_gb']} GB); this config has {have_gb} GB"
    return None


def feasibility_kind(cell: dict | None = None, manifest: dict | None = None) -> str:
    """VRAM-feasibility skips are always a hardware limit (won't physically fit)."""
    return DEFAULT_INFEASIBILITY_KIND


def _as_set(value) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(v) for v in value}
    return {str(value)}


def _matches_rule(cell: dict, resolved: dict, rule: dict, profile: str) -> bool:
    profiles = _as_set(rule.get("profiles")) | _as_set(rule.get("profile"))
    if profiles and profile not in profiles:
        return False

    backend = str(cell.get("backend", "vllm"))
    fields = {
        "host": str(cell["host"]),
        "model": str(cell["model"]),
        "tp": str(cell["tp"]),
        "mode": str(cell["mode"]),
        "backend": backend,
        "preset": str(cell["preset"]),
    }
    for key, actual in fields.items():
        if key in rule and str(rule[key]) != actual:
            return False

    max_len = int(resolved["max_len"])
    if "max_len_lt" in rule and not max_len < int(rule["max_len_lt"]):
        return False
    if "max_len_lte" in rule and not max_len <= int(rule["max_len_lte"]):
        return False
    if "max_len_gt" in rule and not max_len > int(rule["max_len_gt"]):
        return False
    if "max_len_gte" in rule and not max_len >= int(rule["max_len_gte"]):
        return False

    return True


def profile_infeasible_reasons(cell: dict, manifest: dict, *, ignore_max_len_rules: bool = False) -> dict[str, str]:
    resolved = resolve(cell, manifest)
    reasons: dict[str, str] = {}
    for profile in resolved["profiles"]:
        for rule in manifest.get("profile_infeasible", []):
            if ignore_max_len_rules and any(str(key).startswith("max_len_") for key in rule):
                continue
            if _matches_rule(cell, resolved, rule, str(profile)):
                reasons[str(profile)] = str(rule["reason"])
                break
    return reasons


def profile_infeasible_kinds(cell: dict, manifest: dict, *, ignore_max_len_rules: bool = False) -> dict[str, str]:
    """Parallel to profile_infeasible_reasons(): {profile: kind} for the matching
    rule (default hw_permanent when a rule omits `kind`). Same matching logic so
    the kind lines up 1:1 with the reason for each blocked profile."""
    resolved = resolve(cell, manifest)
    kinds: dict[str, str] = {}
    for profile in resolved["profiles"]:
        for rule in manifest.get("profile_infeasible", []):
            if ignore_max_len_rules and any(str(key).startswith("max_len_") for key in rule):
                continue
            if _matches_rule(cell, resolved, rule, str(profile)):
                kinds[str(profile)] = str(rule.get("kind", DEFAULT_INFEASIBILITY_KIND))
                break
    return kinds


def _extra_env_value(extra_env: str, key: str) -> str | None:
    try:
        parts = shlex.split(extra_env)
    except ValueError:
        parts = extra_env.split()
    prefix = f"{key}="
    for part in parts:
        if part.startswith(prefix):
            return part[len(prefix):]
    return None


def _ensure_extra_env(extra_env: str, key: str, value: str) -> str:
    if _extra_env_value(extra_env, key) is not None:
        return extra_env
    return f"{extra_env} {key}={value}".strip()


def _set_extra_env(extra_env: str, key: str, value: str) -> str:
    try:
        parts = shlex.split(extra_env)
    except ValueError:
        parts = extra_env.split()
    prefix = f"{key}="
    kept = [part for part in parts if not part.startswith(prefix)]
    kept.append(f"{key}={value}")
    return " ".join(kept)


def ep_enabled(cell: dict) -> bool:
    """Expert-parallelism flag for a cell (default off).

    EP-on jobs get a distinct `_ep` suffix on their job id and result dir (see
    publish_sweep_state.job_id + bench_orchestrator.sh) so an EP-on run never
    collides with its EP-off sibling. The flag is mirrored into the launch env
    as ENABLE_EP=1 so the remote launcher can add the expert-parallel server
    args, and so the orchestrator/reconcile can recover it from a compiled row.
    """
    return str(cell.get("ep", "")).strip().lower() in {"1", "true", "on", "yes"}


def ep_enabled_extra_env(extra_env: str) -> bool:
    """EP flag recovered from a compiled row's EXTRA_ENV (ENABLE_EP=1)."""
    return str(_extra_env_value(extra_env, "ENABLE_EP") or "").strip().lower() in {"1", "true", "on", "yes"}


def is_moe_model(model: str, manifest: dict) -> bool:
    """Whether a model is a mixture-of-experts model (models.<name>.moe: true)."""
    return bool(manifest.get("models", {}).get(model, {}).get("moe"))


def uses_synthetic_shape(scope: str) -> bool:
    """Scopes that remap profiles to -synth variants on the dense grid."""
    return dashboard_scope_for(scope) in SYNTHETIC_SHAPE_SCOPES


def is_moe_ep_scope(scope: str) -> bool:
    return dashboard_scope_for(scope) == "moe_ep"


def dashboard_scope_for(scope: str) -> str:
    if scope in {"latest", "synthetic", "synthetic-distributional", "synthetic_distributional"}:
        return "synthetic_distributional"
    if scope in {"archive", "trace_replay"}:
        return "trace_replay"
    if scope in {"current", "canonical", "fixed", "fixed-grid", "mse", "archived"}:
        return "archived"
    return scope


def result_scope_for(scope: str) -> str:
    if scope in {"latest", "synthetic", "synthetic-distributional", "synthetic_distributional"}:
        return "synthetic_distributional"
    if scope in {"archive", "trace_replay"}:
        return "trace_replay"
    if scope in {"canonical"}:
        return "current"
    if scope in {"fixed-grid"}:
        return "fixed"
    return scope


def job_record(cell: dict, manifest: dict) -> dict[str, Any]:
    host = manifest["hosts"][cell["host"]]
    model = manifest["models"][cell["model"]]
    resolved = resolve(cell, manifest)
    model_path = f"{host['model_root']}/{model['dir']}"
    extra_env = resolved.get("extra_env", "")
    source_scope = cell_data_scope(cell)
    extra_env = _set_extra_env(str(extra_env), "DASHBOARD_SCOPE", dashboard_scope_for(source_scope))
    extra_env = _set_extra_env(extra_env, "RESULT_SCOPE", result_scope_for(source_scope))
    ep = ep_enabled(cell)
    if ep:
        extra_env = _set_extra_env(extra_env, "ENABLE_EP", "1")
    backend = str(cell.get("backend", "vllm"))
    python_key = "python_sglang" if backend == "sglang" else "python"
    python_bin = str(host.get(python_key) or "")
    return {
        "host": str(cell["host"]),
        "model_path": model_path,
        "tp": int(cell["tp"]),
        "short": str(cell["model"]),
        "mode": str(cell["mode"]),
        "backend": backend,
        "ep": ep,
        "max_len": int(resolved["max_len"]),
        "gpu_mem": resolved["gpu_mem"],
        "concurrencies": [int(c) for c in resolved["concurrencies"]],
        "profiles": [str(profile) for profile in resolved["profiles"]],
        "extra_env": str(extra_env),
        "data_scope": dashboard_scope_for(source_scope),
        "result_scope": result_scope_for(source_scope),
        "hardware_label": str(host.get("hardware_label", cell["host"])),
        "python_bin": python_bin,
        "total_gpus": int(host.get("total_gpus", 0) or 0),
    }


def format_job_record(record: dict[str, Any]) -> str:
    concs = " ".join(str(c) for c in record["concurrencies"])
    profiles = " ".join(str(profile) for profile in record["profiles"])
    fields = [
        str(record["host"]),
        str(record["model_path"]),
        str(record["tp"]),
        str(record["short"]),
        str(record["mode"]),
        str(record["backend"]),
        str(record["max_len"]),
        str(record["gpu_mem"]),
        concs,
        profiles,
        str(record["extra_env"]),
    ]
    return "|".join(fields)


def render_row(cell: dict, manifest: dict) -> str:
    return format_job_record(job_record(cell, manifest))


def cell_data_scope(cell: dict) -> str:
    scope = cell.get("data_scope") or cell.get("dashboard_scope") or cell.get("scope")
    if scope:
        return str(scope)
    extra = str(cell.get("extra_env", ""))
    for key in ("DASHBOARD_SCOPE", "RESULT_SCOPE", "SCOPE"):
        scope = _extra_env_value(extra, key)
        if scope:
            return scope
    return "fixed" if str(cell.get("preset", "")).startswith("fixed_") else "current"


def coverage_grid_scope(scope: str) -> str:
    return DERIVED_SCOPE_SOURCE.get(scope, scope)


def cell_matches_requested_scope(cell_scope: str, requested_scope: str) -> bool:
    if requested_scope == "all":
        return True
    if requested_scope == "archived":
        return dashboard_scope_for(cell_scope) == "archived"
    return cell_scope == coverage_grid_scope(requested_scope)


def profiles_for_output_scope(profiles, requested_scope: str) -> list[str]:
    if not uses_synthetic_shape(requested_scope):
        return [str(profile) for profile in profiles]
    return [
        SYNTHETIC_PROFILE_MAP[str(profile)]
        for profile in profiles
        if str(profile) in SYNTHETIC_PROFILE_MAP
    ]


def cell_for_output_scope(cell: dict, requested_scope: str, manifest: dict | None = None) -> dict:
    out = dict(cell)
    synthetic_concurrencies = out.pop("synthetic_concurrencies", None)
    if requested_scope not in DERIVED_SCOPE_SOURCE:
        return out
    out["data_scope"] = dashboard_scope_for(requested_scope)
    if uses_synthetic_shape(requested_scope):
        profiles = out.get("profiles")
        if profiles is None:
            if manifest is None:
                raise ValueError("manifest is required to derive synthetic profiles")
            profiles = resolve(cell, manifest)["profiles"]
        out["profiles"] = profiles_for_output_scope(profiles, requested_scope)
        extra_env = str(out.get("extra_env", ""))
        for key, value in SYNTHETIC_EXTRA_ENV.items():
            extra_env = _ensure_extra_env(extra_env, key, value)
        out["extra_env"] = extra_env
        if is_moe_ep_scope(requested_scope):
            # Expert-parallelism on. ep_enabled(out) -> True, so job_record
            # injects ENABLE_EP=1 for the launcher and the cell is labelled EP-on.
            out["ep"] = "on"
            # KV-heavy multi-turn: raise only the EP-on derivation to the full
            # memory ceiling so the high-concurrency tail stops OOMing. Scoped
            # here so the EP-off run of the same cell keeps its gpu_mem (its job
            # signature is unchanged, so it is not re-queued).
            if str(out["mode"]) == "multi":
                if manifest is None:
                    raise ValueError("manifest is required to raise EP-on multi gpu_mem")
                out["gpu_mem"] = max(float(resolve(cell, manifest)["gpu_mem"]), EP_MULTI_GPU_MEM_FLOOR)
        mode = str(out["mode"])
        if synthetic_concurrencies is not None:
            out["concurrencies"] = [int(c) for c in synthetic_concurrencies]
        elif mode in SYNTHETIC_TRACE_REPLAY_CONCURRENCIES:
            out["concurrencies"] = list(SYNTHETIC_TRACE_REPLAY_CONCURRENCIES[mode])
    return out


def compile_jobs(manifest: dict, scope: str = "all"):
    emitted: list[tuple[dict, str]] = []
    skipped: list[tuple[dict, str, str]] = []  # (cell, status, reason)

    moe_ep = is_moe_ep_scope(scope)
    for cell in manifest["cells"]:
        if not cell_matches_requested_scope(cell_data_scope(cell), scope):
            continue
        if moe_ep and not (
            is_moe_model(str(cell["model"]), manifest)
            and int(cell["tp"]) > 1
            and str(cell["host"]) in MOE_EP_HOSTS
        ):
            continue
        reason = is_known_oom(cell, manifest)
        if reason:
            skipped.append((cell, "known_oom", reason))
            continue
        reason = feasibility_reason(cell, manifest)
        if reason:
            skipped.append((cell, "infeasible", reason))
            continue
        # Synthetic profiles are generated to fit the requested launch shape,
        # so they do not inherit real-trace max_len profile filters. Keep
        # structural filters such as unsupported GPU kernels.
        profile_reasons = profile_infeasible_reasons(
            cell,
            manifest,
            ignore_max_len_rules=uses_synthetic_shape(scope),
        )
        if profile_reasons:
            resolved = resolve(cell, manifest)
            runnable_profiles = [
                p for p in resolved["profiles"]
                if str(p) not in profile_reasons
            ]
            blocked = ", ".join(
                f"{profile}: {reason}"
                for profile, reason in sorted(profile_reasons.items())
            )
            if not runnable_profiles:
                skipped.append((cell, "profile_infeasible", blocked))
                continue
            emitted_cell = cell_for_output_scope(cell, scope, manifest)
            emitted_cell["profiles"] = runnable_profiles
            emitted_cell["profiles"] = profiles_for_output_scope(emitted_cell["profiles"], scope)
            if not emitted_cell["profiles"]:
                skipped.append((cell, "profile_infeasible", blocked))
                continue
            skipped.append((cell, "profile_infeasible", blocked))
            emitted.append((emitted_cell, render_row(emitted_cell, manifest)))
            continue
        emitted_cell = cell_for_output_scope(cell, scope, manifest)
        if not resolve(emitted_cell, manifest)["profiles"]:
            skipped.append((cell, "empty_scope", f"no profiles map into scope={scope}"))
            continue
        emitted.append((emitted_cell, render_row(emitted_cell, manifest)))
    return emitted, skipped


def render_file(emitted: list[tuple[dict, str]], scope: str = "all") -> str:
    lines = [
        "# Benchmark job matrix consumed by bench_orchestrator.sh.",
        "# GENERATED from scripts/sweep.yaml by scripts/compile_sweep.py — DO NOT EDIT DIRECTLY.",
        "# Format: HOST|MODEL_PATH|TP|SHORT|MODE|BACKEND|MAX_LEN|GPU_MEM|CONCS|PROFILES|EXTRA_ENV",
        f"# SCOPE: {scope}",
        "# MODE: single | multi",
        "# BACKEND: vllm | sglang",
        "# EXTRA_ENV: optional `KEY=VAL KEY=VAL`.",
        "",
    ]
    current_host: str | None = None
    for cell, row in emitted:
        if cell["host"] != current_host:
            if current_host is not None:
                lines.append("")
            current_host = cell["host"]
            lines.append(f"# === {current_host} ===")
        lines.append(row)
    return "\n".join(lines) + "\n"


def skipped_record(cell: dict, status: str, reason: str) -> dict[str, Any]:
    return {
        "host": str(cell.get("host", "")),
        "model": str(cell.get("model", "")),
        "tp": int(cell.get("tp", 0) or 0),
        "mode": str(cell.get("mode", "")),
        "backend": str(cell.get("backend", "vllm")),
        "preset": str(cell.get("preset", "")),
        "status": status,
        "reason": reason,
    }


def render_manifest(
    emitted: list[tuple[dict, str]],
    skipped: list[tuple[dict, str, str]],
    manifest: dict,
    scope: str = "all",
    *,
    source: Path = SWEEP_YAML,
) -> str:
    jobs = [job_record(cell, manifest) for cell, _row in emitted]
    payload = {
        "schema": "agentic-serve.bench-jobs.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(source),
        "scope": scope,
        "jobs": jobs,
        "skipped": [skipped_record(cell, status, reason) for cell, status, reason in skipped],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def scope_choices() -> tuple[str, ...]:
    return (
        "all",
        "trace_replay",
        "synthetic_distributional",
        "synthetic-distributional",
        "archived",
        "synthetic",
        "latest",
        "current",
        "fixed",
        "mse",
        "moe_ep",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", type=Path, default=SWEEP_YAML)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="print to stdout, don't write")
    ap.add_argument(
        "--format",
        choices=("text", "json"),
        help="output format; defaults to json for *.json outputs, otherwise text",
    )
    ap.add_argument("--list-hosts", action="store_true", help="print runnable hosts for the selected scope")
    ap.add_argument("--list-host-gpu-counts", action="store_true", help="print host=total_gpus from sweep.yaml")
    ap.add_argument(
        "--scope",
        choices=scope_choices(),
        default="all",
        help="emit only one dashboard scope",
    )
    ap.add_argument("--verbose", "-v", action="store_true", help="show skip reasons")
    args = ap.parse_args()

    manifest = load_manifest(args.yaml)
    validate(manifest)
    emitted, skipped = compile_jobs(manifest, args.scope)

    if args.list_host_gpu_counts:
        for host, config in sorted(manifest["hosts"].items(), key=lambda item: str(item[0])):
            print(f"{host}={int(config.get('total_gpus', 0) or 0)}")
        return 0

    if args.list_hosts:
        hosts = sorted({str(cell["host"]) for cell, _row in emitted})
        for host in hosts:
            print(host)
        return 0

    output_format = args.format or ("json" if args.out and args.out.suffix == ".json" else "text")
    out_path = args.out or (BENCH_JOBS_JSON if output_format == "json" else BENCH_JOBS_TXT)
    if output_format == "json":
        output = render_manifest(emitted, skipped, manifest, args.scope, source=args.yaml)
    else:
        output = render_file(emitted, args.scope)

    if args.dry_run:
        sys.stdout.write(output)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output)
        print(f"wrote {out_path} ({len(emitted)} rows)", file=sys.stderr)

    print(f"\nsummary: {len(emitted)} emitted, {len(skipped)} skipped", file=sys.stderr)
    if args.verbose or skipped:
        by_status: dict[str, list] = {}
        for cell, status, reason in skipped:
            by_status.setdefault(status, []).append((cell, reason))
        for status, items in sorted(by_status.items()):
            print(f"  {status} ({len(items)}):", file=sys.stderr)
            for cell, reason in items:
                print(
                    f"    {cell['host']} / {cell['model']} / tp{cell['tp']} / {cell['mode']}"
                    f"  -- {reason}",
                    file=sys.stderr,
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
