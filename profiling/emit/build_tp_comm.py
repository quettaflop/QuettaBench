#!/usr/bin/env python3
# profiling/emit/build_tp_comm.py
"""Measure the tensor-parallel prefill comm term from the LIKE-FOR-LIKE tp1/tp2 stage-split pair.

G3 de-fit (`ttft_pricing_defit_plan.md` Item 3; audit-v2 G3): `PREFILL_TP_COMM_MS_PER_TOKEN`
was a backed-out remainder (tp2 ttft.new 18.5 − GEMM/2 12.65 = 5.85 ms/1k) from an
instrumentation-INCONSISTENT pair (tp2 multiprocess api_server vs tp1 in-process LLM), which
physics said over-absorbed ~2.5 ms/1k of host IPC under a comm label (NCCL all-reduce band
~1–3 ms/1k).

This builder computes it like-for-like: BOTH legs are the SAME `serving_stage_split.py`
multiprocess api_server run (tp1: `serving_stage_split_H100.csv`, 2026-06-05; tp2:
`serving_stage_split_H100_tp2.csv`, 2026-06-10, GPUs 6+7 per `h100_setup.md`). Per leg, OLS
``prefill_span_ms ~ FLOOR + a·new + b·cached`` over the c1 cells; the comm term =
``a(tp2) − a(tp1)/2`` — the per-token cost the tp2 GPU-side prefill window carries ABOVE its
halved GEMM share. The host frontend is excluded by construction (it is a separate stage in
both legs and is charged separately in the sim, where it does not shard with tp).

Deterministic (closed-form OLS). Usage:
    python3 -m profiling.emit.build_tp_comm
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TP1_CSV = Path("profile_data/results/serving_stage_split_H100.csv")
TP2_CSV = Path("profile_data/results/serving_stage_split_H100_tp2.csv")
OUT_JSON = Path("profile_data/kernels/prefill_tp_comm_H100.json")
NCCL_PHYSICS_BAND_MS_PER_1K = (1.0, 3.0)  # external-lit all-reduce band, hidden=4096 bf16


def _span_new_rate(path: Path) -> dict:
    """OLS prefill_span_ms ~ floor + a·new + b·cached over the stage-split rows."""
    rows = list(csv.DictReader(path.open()))
    X = [(1.0, float(r["new"]), float(r["cached"])) for r in rows]
    y = [float(r["prefill_span_ms"]) for r in rows]
    # normal equations (3x3), no numpy dependency
    n = len(X)
    xtx = [[sum(X[k][i] * X[k][j] for k in range(n)) for j in range(3)] for i in range(3)]
    xty = [sum(X[k][i] * y[k] for k in range(n)) for i in range(3)]
    # gaussian elimination
    m = [xtx[i] + [xty[i]] for i in range(3)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        for r in range(3):
            if r != col and m[col][col]:
                f = m[r][col] / m[col][col]
                m[r] = [a - f * b for a, b in zip(m[r], m[col])]
    beta = [m[i][3] / m[i][i] for i in range(3)]
    return {"floor_ms": beta[0], "new_ms_per_tok": beta[1], "cached_ms_per_tok": beta[2],
            "n_rows": n}


def build_tpn(n: int, tpn_csv: Path, out_json: Path, gpu: str = "H100",
              tp1_csv: Path = TP1_CSV, host: str = "h100") -> None:
    """L11 extension (2026-06-11): the SAME G3 like-for-like method at tp degree ``n`` —
    comm_total = prefill_span.new(tpN) − span.new(tp1)/N (the TOTAL per-new-token cost the tpN
    GPU prefill window carries above its 1/N GEMM share). Consumed per-config via the deployment
    JSON key ``prefill_tp_comm_ms_per_token`` -> RooflineParams (None for every other config ->
    the tp2-measured per-extra-rank fallback, byte-identical). The tp2 artifact is untouched.

    L13 extension (2026-06-11): ``--gpu RTX3090`` runs the SAME estimator on the RTX3090's
    OWN stage-split legs (cross-GPU transfer of the H100 tp1 leg is INVALID — L11 proved the
    per-extra-rank comm extrapolation fails even within one GPU family; the 3090 host is pure
    PCIe, no NVLink). H100 default paths byte-untouched."""
    for p in (tp1_csv, tpn_csv):
        if not p.exists():
            raise SystemExit(f"missing {p} — pull both stage-split legs first")
    tp1 = _span_new_rate(tp1_csv)
    tpn = _span_new_rate(tpn_csv)
    comm = tpn["new_ms_per_tok"] - tp1["new_ms_per_tok"] / float(n)
    payload = {
        "schema": "prefill_tp_comm.v1",
        "gpu": f"{gpu}x{n}", "model": "Llama-3.1-8B", "tp": n,
        "method": ("like-for-like multiprocess api_server stage-split pair (same script both "
                   f"legs); comm_total = prefill_span.new(tp{n}) − prefill_span.new(tp1)/{n}"),
        "tp1_fit": {k: round(v, 6) if isinstance(v, float) else v for k, v in tp1.items()},
        f"tp{n}_fit": {k: round(v, 6) if isinstance(v, float) else v for k, v in tpn.items()},
        "constants": {"prefill_tp_comm_ms_per_token_total": comm},
        "fallback_it_replaces_ms_per_token": None,  # filled below for context
        "_notes": (f"TOTAL tp{n} comm per new token (NOT per-extra-rank). Pinned in the "
                   f"deployment JSON key prefill_tp_comm_ms_per_token of the {gpu}x{n} config; "
                   "every config without the pin keeps PREFILL_TP_COMM_MS_PER_TOKEN*(tp-1). "
                   f"Regenerate: python3 -m profiling.emit.build_tp_comm --tp {n}"
                   + (f" --gpu {gpu}" if gpu != "H100" else "") + ". Sources: "
                   f"serving_stage_split.py --tensor-parallel-size {{1,{n}}} on {host}."),
    }
    import simulator.ttft_queue_sim as _sim
    payload["fallback_it_replaces_ms_per_token"] = _sim.PREFILL_TP_COMM_MS_PER_TOKEN * (n - 1)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"tp1 span.new {tp1['new_ms_per_tok']*1e3:.3f} ms/1k | tp{n} "
          f"{tpn['new_ms_per_tok']*1e3:.3f} -> comm_total {comm*1e3:.4f} ms/1k "
          f"(fallback it replaces: {payload['fallback_it_replaces_ms_per_token']*1e3:.4f} ms/1k)")
    print(f"wrote {out_json}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tp", type=int, default=2,
                    help="tp degree of the second leg (2 = the original G3 artifact; "
                         "4 = the L11 H100x4 total-comm artifact)")
    ap.add_argument("--gpu", default="H100", choices=["H100", "RTX3090"],
                    help="GPU family of the stage-split legs (L13: RTX3090 uses its OWN "
                         "tp1 leg serving_stage_split_RTX3090.csv; H100 paths untouched)")
    a = ap.parse_args()
    if a.gpu == "RTX3090":
        build_tpn(a.tp,
                  Path(f"profile_data/results/serving_stage_split_RTX3090_tp{a.tp}.csv"),
                  Path(f"profile_data/kernels/prefill_tp_comm_RTX3090x{a.tp}.json"),
                  gpu="RTX3090",
                  tp1_csv=Path("profile_data/results/serving_stage_split_RTX3090.csv"),
                  host="3090")
        return
    if a.tp != 2:
        build_tpn(a.tp, Path(f"profile_data/results/serving_stage_split_H100_tp{a.tp}.csv"),
                  Path(f"profile_data/kernels/prefill_tp_comm_H100x{a.tp}.json"))
        return
    for p in (TP1_CSV, TP2_CSV):
        if not p.exists():
            raise SystemExit(f"missing {p} — pull both stage-split legs first")
    tp1 = _span_new_rate(TP1_CSV)
    tp2 = _span_new_rate(TP2_CSV)
    comm = tp2["new_ms_per_tok"] - tp1["new_ms_per_tok"] / 2.0
    in_band = NCCL_PHYSICS_BAND_MS_PER_1K[0] <= comm * 1e3 <= NCCL_PHYSICS_BAND_MS_PER_1K[1] + 1.0
    payload = {
        "schema": "prefill_tp_comm.v1",
        "gpu": "H100", "model": "Llama-3.1-8B",
        "method": ("like-for-like multiprocess api_server stage-split pair (same script both "
                   "legs); comm = prefill_span.new(tp2) − prefill_span.new(tp1)/2"),
        "tp1_fit": {k: round(v, 6) if isinstance(v, float) else v for k, v in tp1.items()},
        "tp2_fit": {k: round(v, 6) if isinstance(v, float) else v for k, v in tp2.items()},
        "constants": {"PREFILL_TP_COMM_MS_PER_TOKEN": comm},
        "nccl_physics_band_ms_per_1k": list(NCCL_PHYSICS_BAND_MS_PER_1K),
        "within_physics_band_plus_1": in_band,
        "retired_backed_out_remainder": 0.00585,
        "_notes": ("Replaces the instrumentation-inconsistent backed-out remainder 5.85 ms/1k "
                   "(audit-v2 G3): the like-for-like pair removes the multiprocess-vs-in-process "
                   "mismatch, and the residual lands at the top of the NCCL all-reduce physics "
                   "band. Regenerate: python3 -m profiling.emit.build_tp_comm. Sources: "
                   "serving_stage_split.py --tensor-parallel-size {1,2} on h100 (h100_setup.md)."),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"tp1 span.new {tp1['new_ms_per_tok']*1e3:.3f} ms/1k | tp2 {tp2['new_ms_per_tok']*1e3:.3f} "
          f"-> comm {comm*1e3:.4f} ms/1k (retired remainder 5.85; physics band "
          f"{NCCL_PHYSICS_BAND_MS_PER_1K})")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
