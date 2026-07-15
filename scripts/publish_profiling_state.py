#!/usr/bin/env python3
"""Build profiling-state.json from profiling_manifest.yaml and write to dashboard/public/.

Reads llm_predict_legacy/training/per_kernel/profiling_manifest.yaml (produced by Phase 1).
If the manifest does not exist, prints a warning and leaves the existing stub JSON in place.

The manifest is nested (gpus -> model -> per_kernel|per_op -> status_entry); the dashboard
consumes a flat cells list, so build_state() walks the nested dict and emits one cell per
(gpu, model) pair with four status columns.

Run (no upload):
    python scripts/publish_profiling_state.py --no-upload

Run (upload to R2):
    python scripts/publish_profiling_state.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
# TODO(phase-1): the per-kernel profiling manifest producer (llm_predict_legacy) is
# not yet ported into QuettaBench. Until it is, point PROFILING_MANIFEST at an
# externally-produced profiling_manifest.yaml; the legacy path below is only a
# best-effort fallback and will typically not exist in this repo layout.
MANIFEST = Path(
    os.environ.get(
        "PROFILING_MANIFEST",
        str(HERE.parent.parent / "llm_predict_legacy" / "training" / "per_kernel" / "profiling_manifest.yaml"),
    )
)
# Dashboard-JSON artifact output lands in the neutral artifact dir (env-overridable).
OUTPUT_FILE = Path(os.environ.get("BENCH_ARTIFACT_DIR", "/mnt/100g/agent-bench/artifacts")) / "profiling-state.json"

R2_ENDPOINT_DEFAULT = "https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com"
R2_BUCKET_DEFAULT = "agent-bench"
R2_KEY = "json/current/profiling-state.json"

# Populated by main() before build_state(). Kept at module-level so the builder
# can pick it up without threading an extra argument through call sites.
_RESULTS: dict = {}


def _normalise(raw: object, keep: tuple[str, ...]) -> dict:
    """Reduce a manifest status entry to the dashboard's minimal schema:
    {status, reason?, rows?, version?}. Accepts either a plain string status
    or a dict with extra fields."""
    if isinstance(raw, str):
        return {"status": raw}
    if isinstance(raw, dict):
        out: dict = {"status": raw.get("status", "missing")}
        for k in keep:
            v = raw.get(k)
            if v is not None:
                out[k] = v
        return out
    return {"status": "missing"}


_WALLCLOCK_RE = re.compile(
    r"^\|\s*([^|]+?)\s*\|\s*(supported|moe|hybrid_attn)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|"
    r"\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)%\s*\|"
    r"\s*([\d.]+|—)\s*\|\s*(-?[\d.]+%|—)\s*\|\s*([\d.]+|—)\s*\|\s*$"
)


def _parse_wallclock(md_path: Path) -> dict | None:
    """Parse {gpu}_wallclock_validation_seq167.md -> {target_seq, supported_mape, rows}."""
    if not md_path.is_file():
        return None
    supported_mape: float | None = None
    rows: list[dict] = []
    target_seq = 167
    for line in md_path.read_text().splitlines():
        m_seq = re.search(r"avg_seq within .*?of\s*(\d+)", line)
        if m_seq:
            target_seq = int(m_seq.group(1))
        m_mape = re.search(r"\*\*supported MAPE\*\*.*?\*\*([\d.]+)%\*\*", line)
        if m_mape and supported_mape is None:
            supported_mape = float(m_mape.group(1))
            continue
        m = _WALLCLOCK_RE.match(line)
        if not m:
            continue
        model = m.group(1).strip().replace("_(held-out)_", "").strip()
        arch = m.group(2).strip()
        backend = m.group(3).strip()
        profile = m.group(4).strip()
        avg_seq = int(m.group(5))
        pred_ms = float(m.group(7))
        meas_ms = float(m.group(8))
        err_pct = float(m.group(9))
        ncu_str = m.group(10).strip()
        ov_str = m.group(11).strip().rstrip("%")
        tpot_str = m.group(12).strip()
        row: dict = {
            "model": model, "arch": arch, "backend": backend, "profile": profile,
            "avg_seq": avg_seq, "predicted_ms": pred_ms, "measured_ms": meas_ms,
            "abs_err_pct": err_pct,
        }
        if ncu_str != "—":
            row["ncu_sum_ms"] = float(ncu_str)
        if ov_str != "—":
            row["overhead_pct"] = float(ov_str)
        if tpot_str != "—":
            row["median_tpot_ms"] = float(tpot_str)
        rows.append(row)
    if not rows:
        return None
    out: dict = {"target_seq": target_seq, "rows": rows}
    if supported_mape is not None:
        out["supported_mape"] = supported_mape
    return out


_SERVING_PROFILES = (
    "chat-singleturn", "coding-singleturn",
    "chat-multiturn", "swebench-multiturn", "terminalbench-multiturn", "osworld-multiturn",
    # Historical report names kept for archived profiling-state ingestion.
    "chat-multiturn-short", "chat-multiturn-medium", "chat-multiturn-long",
    "coding-agent", "prefill-heavy", "decode-heavy",
    "terminalbench-multiturn-short", "terminalbench-multiturn-medium",
    "swebench-multiturn-short", "swebench-multiturn-medium",
    "osworld-multiturn-short", "osworld-multiturn-medium",
)


def _parse_serving_e2e(md_path: Path) -> dict | None:
    """Parse {gpu}_serving_e2e_{profile}.md -> {mape, rows}."""
    if not md_path.is_file():
        return None
    rows: list[dict] = []
    mape: dict[str, float] = {}

    def _opt(s: str) -> float | None:
        s = s.strip().rstrip("%")
        return float(s) if s != "—" else None

    for line in md_path.read_text().splitlines():
        m_summary = re.match(
            r"^\|\s*(TTFT|TPOT|E2EL)\s*\|\s*([\d.]+)%\s*\|\s*(\d+)\s*\|", line)
        if m_summary:
            mape[m_summary.group(1).lower()] = float(m_summary.group(2))
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 17:
            continue
        cols = parts[1:-1]
        if len(cols) != 15:
            continue
        arch = cols[1].strip()
        if arch not in ("supported", "moe", "hybrid_attn"):
            continue
        model = cols[0].replace("_(held-out)_", "").strip()
        try:
            row: dict = {
                "model": model, "arch": arch, "backend": cols[2].strip(),
                "isl": int(cols[3]), "osl": int(cols[4]), "bs": int(cols[5]),
                "pred_ttft_ms": float(cols[6]),
                "meas_ttft_ms": float(cols[7]),
                "ttft_err_pct": float(cols[8].rstrip("%")),
            }
        except (ValueError, IndexError):
            continue
        row["pred_tpot_ms"] = _opt(cols[9])
        row["meas_tpot_ms"] = _opt(cols[10])
        row["tpot_err_pct"] = _opt(cols[11])
        try:
            row["pred_e2el_ms"] = float(cols[12])
        except ValueError:
            continue
        row["meas_e2el_ms"] = _opt(cols[13])
        row["e2el_err_pct"] = _opt(cols[14])
        rows.append(row)

    if not rows and not mape:
        return None
    return {"mape": mape, "rows": rows}




def _parse_conc_table(lines, start_idx):
    """Parse a per-concurrency MAPE table starting at start_idx."""
    rows = []
    for line in lines[start_idx:]:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 8:
            if rows:
                break
            continue
        cols = parts[1:-1]
        if len(cols) != 6:
            continue
        try:
            conc = int(cols[0])
            bs_eff = float(cols[1])
            ttft = float(cols[2].rstrip("%"))
            tpot = float(cols[3].rstrip("%"))
            e2el = float(cols[4].rstrip("%"))
            n = int(cols[5])
        except (ValueError, IndexError):
            continue
        rows.append({
            "conc": conc, "bs_eff": bs_eff,
            "ttft_mape": ttft, "tpot_mape": tpot, "e2el_mape": e2el, "n": n,
        })
    return rows


def _parse_serving_e2e_conc(md_path):
    """Parse {gpu}_serving_e2e_conc_{profile}.md -> {overall, per_conc, per_conc_moe}."""
    if not md_path.is_file():
        return None
    overall = {}
    overall_moe = {}
    per_conc = []
    per_conc_moe = []
    lines = md_path.read_text().splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"Overall supported MAPE: TPOT ([\d.]+)%, E2EL ([\d.]+)%", line)
        if m:
            overall["tpot"] = float(m.group(1))
            overall["e2el"] = float(m.group(2))
            continue
        m2 = re.match(r"Overall MoE MAPE: TPOT ([\d.]+)%, E2EL ([\d.]+)%", line)
        if m2:
            overall_moe["tpot"] = float(m2.group(1))
            overall_moe["e2el"] = float(m2.group(2))
            continue
        if "Per-concurrency MAPE (supported" in line:
            per_conc = _parse_conc_table(lines, i + 3)
        if "Per-concurrency MAPE (MoE" in line:
            per_conc_moe = _parse_conc_table(lines, i + 3)
    if not per_conc and not overall:
        return None
    out = {"overall": overall, "per_conc": per_conc}
    if per_conc_moe:
        out["overall_moe"] = overall_moe
        out["per_conc_moe"] = per_conc_moe
    return out

def _load_results(repo_root: Path) -> dict:
    """Best-effort: read training_report.json files and wallclock markdown reports."""
    results: dict = {"per_kernel": {}, "per_op": {}, "wallclock": {},
                     "serving_e2e": {}, "serving_e2e_perop": {}, "serving_e2e_conc": {},
                     "gemm_extrapolation": {}}
    pk_path = repo_root / "llm_predict_legacy/training/per_kernel/reports/training_report.json"
    po_path = repo_root / "llm_predict_legacy/training/per_op/reports/training_report.json"
    pk_reports = repo_root / "llm_predict_legacy" / "training" / "per_kernel" / "reports"
    po_reports = repo_root / "llm_predict_legacy" / "training" / "per_op" / "reports"
    for gpu in ("A100", "RTX3090", "RTX2080Ti", "H100"):
        wc_path = pk_reports / f"{gpu}_wallclock_validation_seq167.md"
        wc = _parse_wallclock(wc_path)
        if wc is not None:
            results["wallclock"][gpu] = wc
        # per-kernel serving_e2e
        profiles: dict = {}
        for profile in _SERVING_PROFILES:
            parsed = _parse_serving_e2e(pk_reports / f"{gpu}_serving_e2e_{profile}.md")
            if parsed is not None:
                profiles[profile] = parsed
        if profiles:
            results["serving_e2e"][gpu] = profiles
        # per-op serving_e2e (ablation)
        po_profiles: dict = {}
        for profile in _SERVING_PROFILES:
            parsed = _parse_serving_e2e(po_reports / f"{gpu}_serving_e2e_perop_{profile}.md")
            if parsed is not None:
                po_profiles[profile] = parsed
        if po_profiles:
            results["serving_e2e_perop"][gpu] = po_profiles
        # per-kernel serving_e2e concurrency sweeps
        conc_profiles = {}
        for profile in _SERVING_PROFILES:
            parsed = _parse_serving_e2e_conc(pk_reports / f"{gpu}_serving_e2e_conc_{profile}.md")
            if parsed is not None:
                conc_profiles[profile] = parsed
        if conc_profiles:
            results["serving_e2e_conc"][gpu] = conc_profiles
    if pk_path.is_file():
        try:
            pk = json.loads(pk_path.read_text())
            for gpu, entry in pk.items():
                results["per_kernel"][gpu] = {
                    "heldout_mape_per_family": entry.get("heldout_mape", {}),
                    "aggregate_err_per_model": entry.get("aggregate_err_pct", {}),
                }
        except Exception as e:
            print(f'[warn] per_kernel report parse failed: {e}', file=sys.stderr)
    if po_path.is_file():
        try:
            po = json.loads(po_path.read_text())
            for entry in po:
                results["per_op"][entry["gpu"]] = {
                    "heldout_mape": entry.get("heldout_mape"),
                    "pool_models": entry.get("pool_models", []),
                    "heldout_models": entry.get("heldout_models", []),
                }
        except Exception as e:
            print(f'[warn] per_op report parse failed: {e}', file=sys.stderr)
    # GEMM extrapolation (static JSON from one-time test runs)
    gemm_extrap_path = pk_reports / "gemm_extrapolation.json"
    if gemm_extrap_path.is_file():
        import json as _json
        results["gemm_extrapolation"] = _json.loads(gemm_extrap_path.read_text())

    return results


def build_state(manifest: dict) -> dict:
    """Walk manifest['gpus'][gpu][model] and emit a flat cells list."""
    gpus_data = manifest.get("gpus", {})
    if not isinstance(gpus_data, dict):
        raise ValueError(
            "manifest['gpus'] must be a nested dict (gpu -> model -> ...); "
            f"got {type(gpus_data).__name__}"
        )

    gpus = sorted(gpus_data.keys())
    all_models: set[str] = set()
    for models_data in gpus_data.values():
        if isinstance(models_data, dict):
            all_models.update(models_data.keys())
    models = sorted(all_models)

    cells: list[dict] = []
    for gpu in gpus:
        models_data = gpus_data.get(gpu, {})
        if not isinstance(models_data, dict):
            continue
        for model in sorted(models_data.keys()):
            entry = models_data[model] if isinstance(models_data[model], dict) else {}
            per_kernel = entry.get("per_kernel", {}) if isinstance(entry, dict) else {}
            per_op = entry.get("per_op", {}) if isinstance(entry, dict) else {}
            cells.append({
                "gpu": gpu,
                "model": model,
                "per_kernel_prefill":  _normalise(per_kernel.get("prefill_seq128_bs1"), ("reason", "rows")),
                "per_kernel_roofline": _normalise(per_kernel.get("roofline_sweep"), ("reason", "rows")),
                "per_op_cuda_events":  _normalise(per_op.get("cuda_events"), ("reason",)),
                "per_op_trained_pkl":  _normalise(per_op.get("trained_pkl"), ("reason", "version")),
            })

    return {
        "generated_at": manifest.get("generated_at") or datetime.now(timezone.utc).isoformat(),
        "gpus": gpus,
        "models": models,
        "cells": cells,
        "results": _RESULTS,
    }


def upload_r2(path: Path, endpoint: str, bucket: str, profile: str) -> None:
    cmd = [
        "aws", "--profile", profile, "s3", "cp",
        str(path), f"s3://{bucket}/{R2_KEY}",
        "--endpoint-url", endpoint,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--out", type=Path, default=OUTPUT_FILE)
    ap.add_argument("--no-upload", action="store_true", help="skip R2 upload")
    ap.add_argument("--endpoint", default=os.environ.get("R2_ENDPOINT", R2_ENDPOINT_DEFAULT))
    ap.add_argument("--bucket", default=os.environ.get("R2_BUCKET", R2_BUCKET_DEFAULT))
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE", "r2"))
    args = ap.parse_args()

    if not args.manifest.exists():
        print(
            f"WARNING: manifest not found at {args.manifest} -- "
            "leaving existing profiling-state.json stub in place.",
            file=sys.stderr,
        )
        return 0

    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML not installed. Run: pip install pyyaml", file=sys.stderr)
        return 1

    manifest = yaml.safe_load(args.manifest.read_text())
    global _RESULTS
    repo_root = HERE.parent.parent  # inference-benchmark/scripts -> inference-benchmark -> repo
    _RESULTS = _load_results(repo_root)
    state = build_state(manifest)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    def _sanitize(obj):
        if isinstance(obj, float) and (obj != obj):  # NaN check
            return None
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    args.out.write_text(json.dumps(_sanitize(state), indent=2) + "\n")
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
