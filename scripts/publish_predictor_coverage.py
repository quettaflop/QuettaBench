#!/usr/bin/env python3
"""Build predictor-coverage.json from kernels_labeled.csv + per_op_labeled.csv.

Walks the two predictor training CSVs and emits per-(gpu, model) row counts
for the dashboard's Coverage page. Mirrors the publish_sweep_state.py +
publish_profiling_state.py contract: write to dashboard/public/, optionally
upload to R2.

Run (no upload):
    python scripts/publish_predictor_coverage.py --no-upload

Run (upload to R2):
    python scripts/publish_predictor_coverage.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
KERNEL_CSV = REPO_ROOT / "llm_predict_legacy" / "training" / "per_kernel" / "data" / "kernels_labeled.csv"
PEROP_CSV = REPO_ROOT / "llm_predict_legacy" / "training" / "per_op" / "data" / "per_op_labeled.csv"
OUTPUT_FILE = HERE.parent / "dashboard" / "public" / "predictor-coverage.json"

R2_ENDPOINT_DEFAULT = "https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com"
R2_BUCKET_DEFAULT = "agent-bench"
R2_KEY = "json/current/predictor-coverage.json"

# Source-name suffix matching: anything ending in "_prefill" is per-model
# prefill ncu data; "roofline_sweep" / "misc_sweep" / "flash_sweep" are
# model-agnostic shared rows.
PREFILL_SUFFIX = "_prefill"
ROOFLINE_SOURCE = "roofline_sweep"
MISC_SOURCE = "misc_sweep"
FLASH_SOURCE_PREFIX = "flash"

# Expected denominators are best-effort guesses based on the sweep grids
# documented in predictor_notes.md. They drive the % bar but don't gate
# anything; if you re-tune the sweep, update these.
EXPECTED_PREFILL_PER_MODEL = 1500    # ~1.5k kernels per model prefill_seq128 sweep
EXPECTED_FLASH_PER_MODEL = 250       # planned flash sweep density
EXPECTED_ROOFLINE_PER_GPU = 3500     # post-fix nn.Linear sweep
EXPECTED_MISC_PER_GPU = 350          # rmsnorm/silu/elementwise grid
EXPECTED_PEROP_GRID = 512            # bs=1, seq=1..512 dense grid
EXPECTED_OPS = ["attn", "ffn", "norm_post", "norm_pre"]

GPUS_TARGET = ["A100", "RTX3090", "RTX2080Ti"]


def _classify_source(src: str) -> str:
    if src.endswith(PREFILL_SUFFIX):
        return "prefill"
    if src == ROOFLINE_SOURCE:
        return "roofline"
    if src == MISC_SOURCE:
        return "misc"
    if src.startswith(FLASH_SOURCE_PREFIX):
        return "flash"
    return "other"


def _coverage_status(have: int, expected: int) -> str:
    if have == 0:
        return "missing"
    if have >= expected:
        return "present"
    return "partial"


def build_kernel_coverage(csv_path: Path) -> tuple[list[dict], list[dict], set[str], set[str]]:
    """Walk kernels_labeled.csv -> (shared_rows, model_cells, gpus, models)."""
    if not csv_path.is_file():
        print(f"[warn] kernel csv not found: {csv_path}", file=sys.stderr)
        return [], [], set(), set()

    by_track: Counter = Counter()  # (gpu, model, track) -> rows
    held_out_flag: dict[tuple[str, str], bool] = {}
    with csv_path.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            gpu = row["gpu"]
            model = row["model"]
            src = row["source"]
            track = _classify_source(src)
            by_track[(gpu, model, track)] += 1
            try:
                ho = bool(int(row.get("held_out", "0")))
            except ValueError:
                ho = False
            held_out_flag.setdefault((gpu, model), ho)

    gpus = {k[0] for k in by_track}
    # Model "names" in this CSV include placeholder rows where model column
    # holds the source label (`roofline`, `misc_sweep`). Filter those out
    # of the per-model grid; they're handled in `shared`.
    models = {k[1] for k in by_track if k[1] not in {"roofline", "misc_sweep"}}

    shared: list[dict] = []
    for gpu in sorted(gpus):
        roof = sum(v for (g, _m, t), v in by_track.items() if g == gpu and t == "roofline")
        misc = sum(v for (g, _m, t), v in by_track.items() if g == gpu and t == "misc")
        # Sweep version inferred from row count: post-fix sweeps have
        # ~3.5k+ rows; older a@b sweeps were ~1k.
        ver = "post-fix" if roof >= 2500 else ("pre-fix" if roof > 0 else "unknown")
        shared.append({
            "gpu": gpu,
            "roofline_rows": roof,
            "misc_rows": misc,
            "sweep_version": ver,
            "expected_roofline": EXPECTED_ROOFLINE_PER_GPU,
            "expected_misc": EXPECTED_MISC_PER_GPU,
        })

    cells: list[dict] = []
    for gpu in sorted(gpus):
        for model in sorted(models):
            prefill = by_track.get((gpu, model, "prefill"), 0)
            flash = by_track.get((gpu, model, "flash"), 0)
            total = prefill + flash
            if total == 0:
                continue
            ho = held_out_flag.get((gpu, model), False)
            cells.append({
                "gpu": gpu,
                "model": model,
                "prefill_rows": prefill,
                "flash_rows": flash,
                "total_rows": total,
                "held_out": ho,
                "status": _coverage_status(prefill, EXPECTED_PREFILL_PER_MODEL),
                "expected_prefill": EXPECTED_PREFILL_PER_MODEL,
                "expected_flash": EXPECTED_FLASH_PER_MODEL,
            })

    return shared, cells, gpus, models


def build_perop_coverage(csv_path: Path) -> tuple[list[dict], set[str], set[str]]:
    """Walk per_op_labeled.csv -> (cells, gpus, models)."""
    if not csv_path.is_file():
        print(f"[warn] per-op csv not found: {csv_path}", file=sys.stderr)
        return [], set(), set()

    rows_per_op: dict[tuple[str, str], Counter] = defaultdict(Counter)
    grid_cells: dict[tuple[str, str], set[tuple[int, int]]] = defaultdict(set)
    held_out_flag: dict[tuple[str, str], bool] = {}
    with csv_path.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            gpu = row["gpu"]
            model = row["model"]
            op = row["op"]
            try:
                bs = int(row["bs"])
                seq = int(row["seq"])
            except (KeyError, ValueError):
                continue
            rows_per_op[(gpu, model)][op] += 1
            grid_cells[(gpu, model)].add((bs, seq))
            try:
                ho = bool(int(row.get("held_out", "0")))
            except ValueError:
                ho = False
            held_out_flag.setdefault((gpu, model), ho)

    gpus = {k[0] for k in rows_per_op}
    models = {k[1] for k in rows_per_op}

    cells: list[dict] = []
    for (gpu, model), counter in rows_per_op.items():
        cells_count = len(grid_cells[(gpu, model)])
        total = sum(counter.values())
        ops_present = sorted(counter.keys())
        ops_missing = sorted(set(EXPECTED_OPS) - set(ops_present))
        if cells_count >= 256:
            density = "dense"
        elif cells_count >= EXPECTED_PEROP_GRID // 2:
            density = "partial"
        else:
            density = "thin"
        if not ops_missing and density == "dense":
            status = "present"
        elif total == 0:
            status = "missing"
        else:
            status = "partial"
        cells.append({
            "gpu": gpu,
            "model": model,
            "rows_per_op": dict(counter),
            "ops_present": ops_present,
            "ops_missing": ops_missing,
            "total_rows": total,
            "grid_cells": cells_count,
            "density": density,
            "held_out": held_out_flag.get((gpu, model), False),
            "status": status,
            "expected_grid": EXPECTED_PEROP_GRID,
        })

    cells.sort(key=lambda c: (c["gpu"], c["model"]))
    return cells, gpus, models


def build_state(kernel_csv: Path, perop_csv: Path) -> dict:
    shared, kcells, kgpus, kmodels = build_kernel_coverage(kernel_csv)
    pcells, pgpus, pmodels = build_perop_coverage(perop_csv)

    all_gpus = sorted(set(GPUS_TARGET) | kgpus | pgpus)
    all_models = sorted(kmodels | pmodels)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gpus": all_gpus,
        "models": all_models,
        "expected_ops": EXPECTED_OPS,
        "per_kernel": {"shared": shared, "cells": kcells},
        "per_op": pcells,
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
    ap.add_argument("--kernel-csv", type=Path, default=KERNEL_CSV)
    ap.add_argument("--perop-csv", type=Path, default=PEROP_CSV)
    ap.add_argument("--out", type=Path, default=OUTPUT_FILE)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--endpoint", default=os.environ.get("R2_ENDPOINT", R2_ENDPOINT_DEFAULT))
    ap.add_argument("--bucket", default=os.environ.get("R2_BUCKET", R2_BUCKET_DEFAULT))
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE", "r2"))
    args = ap.parse_args()

    state = build_state(args.kernel_csv, args.perop_csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(state, indent=2) + "\n")
    print(
        f"wrote {args.out} "
        f"({len(state['per_kernel']['cells'])} kernel cells, "
        f"{len(state['per_op'])} per-op cells)",
        file=sys.stderr,
    )

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
