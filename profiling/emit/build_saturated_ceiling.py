#!/usr/bin/env python3
# profiling/emit/build_saturated_ceiling.py
"""Measured saturated-ITL ceiling anchors (median tpot_ms at KV pressure >= 2.5).

Needs QuettaSim/agentic-serve on PYTHONPATH (`configs.loader` and
`profiling.process.build_simulator_rows`).

    python3 -m profiling.emit.build_saturated_ceiling
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PRESSURE_THRESHOLD = 2.5
CLUSTER_SPLIT_OUTPUT = 50.0
CACHE_BLOCK_SIZE = 16

# Filled in main() after consumer imports succeed.
BENCH_BASE = None
CONCURRENCIES = None
PROFILES = None
build_turns = None


@dataclass(frozen=True)
class CeilingConfig:
    gpu: str
    model: str
    tp: int
    bench_dir: str
    available_kv_blocks: int
    out_json: Path


def _load_consumer():
    try:
        from profiling.process.build_simulator_rows import (  # noqa: E402
            BENCH_BASE as bench_base,
            CONCURRENCIES as concs,
            PROFILES as profiles,
            build_turns as turns_fn,
        )
        from configs.loader import all_deployments  # noqa: E402
    except ImportError as e:
        raise SystemExit(
            "build_saturated_ceiling needs QuettaSim/agentic-serve on PYTHONPATH "
            f"(could not import {e.name!r}). See profiling/README.md."
        ) from e
    return bench_base, concs, profiles, turns_fn, all_deployments


def _turns_with_pressure(cfg: CeilingConfig) -> list[tuple[float, float, float]]:
    """(output_tokens, tpot_meas, pressure) for every measured turn of the run."""
    root = BENCH_BASE / cfg.bench_dir
    out: list[tuple[float, float, float]] = []
    for prof in PROFILES:
        for c in CONCURRENCIES:
            f = root / f"{prof}_conc{c}.json"
            if not f.exists():
                continue
            turns, _shared_prefix = build_turns(f)
            for tn in turns:
                output = max(1.0, float(tn["output_tokens"]))
                sched = max(1.0, float(tn["scheduled_requests"]))
                ctx = (tn["cached_context_tokens"] + tn["new_prefill_tokens"]
                       + 0.5 * output)
                psb = max(1, math.ceil(ctx / CACHE_BLOCK_SIZE))
                pressure = sched * psb / cfg.available_kv_blocks
                meas = float(tn["tpot_meas"])
                if meas > 0:
                    out.append((output, meas, pressure))
    return out


def _anchors_for(sat: list[tuple[float, float]],
                 split: float = CLUSTER_SPLIT_OUTPUT) -> list[dict]:
    """Per-output-cluster median anchors for one (threshold, split) cut choice."""
    anchors = []
    for cluster in ([(o, m) for o, m in sat if o < split],
                    [(o, m) for o, m in sat if o >= split]):
        if not cluster:
            continue
        anchors.append({
            "output_tokens": round(st.median([o for o, _ in cluster])),
            "plateau_ms": round(st.median([m for _, m in cluster]), 1),
            "n": len(cluster),
        })
    anchors.sort(key=lambda a: a["output_tokens"])
    return anchors


def _sensitivity_note(turns: list[tuple[float, float, float]],
                      anchors: list[dict]) -> str:
    """The audit-v2 G9 sensitivity record, with the ACTUAL recomputed numbers.

    PRESSURE_THRESHOLD=2.5 is a one-time curve read-off ("saturates by ~2.5"), not
    a derivation — so the artifact carries what moving the cut to 2.0/3.0 does to
    its own anchors. CLUSTER_SPLIT_OUTPUT is checked the same way (40/60). The
    production cuts themselves are UNCHANGED."""
    parts: list[str] = []
    for thr in (2.0, 3.0):
        alt = _anchors_for([(o, m) for o, m, p in turns if p >= thr])
        moves = []
        for a in anchors:
            # match clusters by side of the split (anchor output < / >= split)
            side = [x for x in alt
                    if (x["output_tokens"] < CLUSTER_SPLIT_OUTPUT)
                    == (a["output_tokens"] < CLUSTER_SPLIT_OUTPUT)]
            if not side:
                moves.append(f"out={a['output_tokens']}: cluster empty")
                continue
            b = side[0]
            pct = (b["plateau_ms"] - a["plateau_ms"]) / a["plateau_ms"] * 100.0
            moves.append(f"out={a['output_tokens']}: {a['plateau_ms']}"
                         f"->{b['plateau_ms']} ms ({pct:+.2f}%)")
        parts.append(f"@{thr}: " + ", ".join(moves))
    sat25 = [(o, m) for o, m, p in turns if p >= PRESSURE_THRESHOLD]
    split_moves = []
    for split in (40.0, 60.0):
        alt = _anchors_for(sat25, split=split)
        same = ([a["plateau_ms"] for a in alt]
                == [a["plateau_ms"] for a in anchors])
        split_moves.append(f"split={split:g}: "
                           + ("anchors identical"
                              if same else
                              "plateau_ms " + "/".join(str(a["plateau_ms"]) for a in alt)
                              + f" vs {'/'.join(str(a['plateau_ms']) for a in anchors)}"))
    return (" Sensitivity (audit-v2 G9): PRESSURE_THRESHOLD=2.5 is a one-time "
            "curve read-off ('saturates by ~2.5'), not derived; recomputing this "
            "artifact's anchors at thresholds 2.0/3.0 gives — "
            + "; ".join(parts)
            + ". CLUSTER_SPLIT_OUTPUT=50 check — " + "; ".join(split_moves)
            + ". Production cuts unchanged (2026-06-10, lane L5).")


def build(cfg: CeilingConfig) -> dict | None:
    turns = _turns_with_pressure(cfg)
    sat = [(o, m) for o, m, p in turns if p >= PRESSURE_THRESHOLD]
    if not sat:
        # The cell never reaches the saturated regime at its pool (large-pool / multi-GPU MoE
        # cells) — the ramp never targets the ceiling there, so a measured ceiling is moot.
        # Warn + skip rather than crash a batch build; the manifest should keep status=inherited.
        print(f"SKIP {cfg.gpu} {cfg.model}: no saturated turns (pressure>={PRESSURE_THRESHOLD}) "
              f"at pool {cfg.available_kv_blocks} — keep inherited ceiling")
        return None
    anchors = _anchors_for(sat)
    all_ms = sorted(m for _, m in sat)
    return {
        "gpu": cfg.gpu,
        "model": cfg.model,
        "tensor_parallel": cfg.tp,
        "criterion": (f"median measured tpot_ms over turns at KV pressure >= "
                      f"{PRESSURE_THRESHOLD} (the saturated 'C=300+' asymptote)"),
        "pressure_threshold": PRESSURE_THRESHOLD,
        "cluster_split_output": CLUSTER_SPLIT_OUTPUT,
        "source": str(BENCH_BASE / cfg.bench_dir),
        "n_saturated_turns": len(sat),
        "max_measured_plateau_ms": round(all_ms[-1], 1),
        "p90_measured_plateau_ms": round(all_ms[min(len(all_ms) - 1, int(len(all_ms) * 0.9))], 1),
        "anchors": anchors,
        "lookup": ("linear interpolation in output between anchors; clamp to the "
                   "nearest anchor outside the range (monotone non-increasing: "
                   "short output saturates higher)"),
        "_notes": ("Measured-anchor replacement for the retired least-squares ceiling "
                   "118.7 + 3263/output (no fit; measured plateau medians + interpolation, "
                   "same pattern as the decode kernel grid). Regenerate: "
                   "python3 -m profiling.emit.build_saturated_ceiling."
                   + _sensitivity_note(turns, anchors)),
    }


def main() -> None:
    global BENCH_BASE, CONCURRENCIES, PROFILES, build_turns
    BENCH_BASE, CONCURRENCIES, PROFILES, build_turns, all_deployments = _load_consumer()
    configs = [
        CeilingConfig(d.gpu_key, d.model, d.tp, d.bench_dir, d.available_kv_blocks,
                      Path((d.data.get("saturated_ceiling") or {})["path"]))
        for d in all_deployments()
        if (d.data.get("saturated_ceiling") or {}).get("status") in ("measured", "derived")
        and (d.data.get("saturated_ceiling") or {}).get("path")
    ]
    for cfg in configs:
        if not (BENCH_BASE / cfg.bench_dir).exists():
            print(f"SKIP {cfg.gpu}: bench root missing ({BENCH_BASE / cfg.bench_dir})")
            continue
        payload = build(cfg)
        if payload is None:
            continue
        cfg.out_json.parent.mkdir(parents=True, exist_ok=True)
        cfg.out_json.write_text(json.dumps(payload, indent=2) + "\n")
        anchors = ", ".join(f"out={a['output_tokens']}->{a['plateau_ms']}ms(n={a['n']})"
                            for a in payload["anchors"])
        print(f"{cfg.gpu}: {payload['n_saturated_turns']} saturated turns -> anchors [{anchors}] "
              f"-> {cfg.out_json}")


if __name__ == "__main__":
    main()
