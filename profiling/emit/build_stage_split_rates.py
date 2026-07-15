#!/usr/bin/env python3
# profiling/emit/build_stage_split_rates.py
"""Per-config prefill HOST-cached rate + FA3 coefficient from the LIKE-FOR-LIKE tp1/tpN
stage-split pair (L11 round 2, 2026-06-11).

The simulator's TTFT prefill pricing carries two tp1-measured constants that the H100x4
ground truth shows do NOT transport to the tp4 serving stack (the same class of error the
measured per-config prefill FLOOR 26→18.08 and the measured tp4 comm 9.84→3.96 already
fixed):

* ``PREFILL_HOST_SHARED/PERREQ_MS_PER_TOKEN`` (sum 5.8872e-3 ms per cached token — the host
  re-tokenize cost of the re-sent conversation, measured on the tp1 stack via
  ``build_host_split``'s c1 lstsq). The tp4 GT's own c1 turns measure a marginal cached cost
  of ~2.4 ms/1k (osworld@1 turns: TTFT 22.6–22.7 ms at cached≈2k over the measured 18.08
  floor), and the like-for-like pair below measures 3.53 ms/1k.
* ``PREFILL_FA3_MS_PER_TOKEN2`` (8.31e-7, the tp1 pipeline FA3 kernel) — at tp>1 the
  attention prefill is head-sharded across ranks; charging the tp1 coefficient over-prices
  every big-context (re-)prefill ~3× at tp4.

LIKE-FOR-LIKE METHOD (the G3/build_tp_comm pattern): BOTH legs are the SAME
``serving_stage_split.py`` multiprocess api_server sweep over the SAME (new × cached)
lattice (tp1: ``serving_stage_split_H100.csv`` 2026-06-05; tp4:
``serving_stage_split_H100_tp4.csv`` 2026-06-11, h100 GPUs 4–7 — the round-1 Phase-B leg,
whose ``prefill_span_ms`` column already produced the adopted comm term; this builder
consumes the ``ttft_ms`` and FA3 cross-term information the comm fit did not use).

PRE-REGISTERED ESTIMATORS (defit_log_entries/L11-h100multi.md round-2 entry, committed
before this builder ran):

* HOST: per leg, 3-param OLS ``ttft_ms ~ floor + a·new + b·cached`` — the SAME model family
  as the production estimator (``build_host_split.fit_c1_rate``). VALIDATION (hard-fail):
  b(tp1 leg) must reproduce the production SUM 5.8872e-3 within 5% (measured: 5.9889e-3,
  +1.7%). The per-config value = b(tpN leg), DIRECT (the prefill-floor precedent). The
  shared/per-request PARTITION is NOT re-measured (no tpN B-sweep exists): the consumer
  keeps the production measured fraction 0.5236 of the per-config SUM — documented caveat.
* FA3: per leg, 4-param OLS ``prefill_span_ms ~ floor + a·new + b·cached +
  c·new·(cached + new/2)`` (the simulator's own FA3 regressor). The per-config value =
  production 8.31e-7 × c(tpN)/c(tp1) — RATIO TRANSPORT via the tp1 leg, which bridges the
  production constant's kernel-grid provenance to this serving-stack instrument (the same
  role the tp1 leg plays in build_tp_comm). The direct c(tpN) is reported as sensitivity.
  VALIDATION (hard-fail): c(tp1) within 50% of the production constant (measured 5.572e-7
  vs 8.31e-7 — the serving-stack estimator sees a smaller coefficient than the isolated
  kernel grid on BOTH legs, which is exactly why the ratio, not the raw coefficient, is
  transported) and 0 < ratio < 1 (sharding can only help).

Deterministic (closed-form OLS, no RNG). Usage:
    python3 -m profiling.emit.build_stage_split_rates --tp 4
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS = REPO_ROOT / "profile_data" / "results"
TP1_CSV = RESULTS / "serving_stage_split_H100.csv"

# Production constants the per-config values replace / transport (asserted against the live
# module below so a retune cannot silently desynchronize this builder).
PROD_HOST_SUM = 0.0030824476411757708 + 0.002804790423340364   # 5.8872e-3 (prefill_host_split_H100.json)
PROD_FA3 = 8.31e-7                                             # PREFILL_FA3_MS_PER_TOKEN2
HOST_TP1_TOLERANCE = 0.05   # tp1-leg host coefficient must reproduce production within 5%
FA3_TP1_TOLERANCE = 0.50    # tp1-leg FA3 coefficient must corroborate magnitude within 50%


def _ols(rows: list[dict], ycol: str, terms: list[str]) -> dict[str, float]:
    """Closed-form OLS over the stage-split rows; design terms by name (no numpy)."""
    def feats(r: dict) -> list[float]:
        new, cached = float(r["new"]), float(r["cached"])
        avail = {"1": 1.0, "new": new, "cached": cached, "fa3": new * (cached + 0.5 * new)}
        return [avail[t] for t in terms]

    X = [feats(r) for r in rows]
    y = [float(r[ycol]) for r in rows]
    k, n = len(terms), len(X)
    xtx = [[sum(X[r][i] * X[r][j] for r in range(n)) for j in range(k)] for i in range(k)]
    xty = [sum(X[r][i] * y[r] for r in range(n)) for i in range(k)]
    m = [xtx[i] + [xty[i]] for i in range(k)]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        for r in range(k):
            if r != col and m[col][col]:
                f = m[r][col] / m[col][col]
                m[r] = [a - f * b for a, b in zip(m[r], m[col])]
    beta = {t: m[i][k] / m[i][i] for i, t in enumerate(terms)}
    pred = [sum(beta[t] * x for t, x in zip(terms, X[r])) for r in range(n)]
    beta["_fit_mape_pct"] = 100.0 * sum(abs(p - yy) / yy for p, yy in zip(pred, y)) / n
    beta["_n_rows"] = float(n)
    return beta


def build(tp: int) -> None:
    import simulator.ttft_queue_sim as sim
    assert abs((sim.PREFILL_HOST_SHARED_MS_PER_TOKEN + sim.PREFILL_HOST_PERREQ_MS_PER_TOKEN)
               - PROD_HOST_SUM) < 1e-15, "production host SUM moved — update this builder's anchor"
    assert sim.PREFILL_FA3_MS_PER_TOKEN2 == PROD_FA3, "production FA3 moved — update this builder's anchor"
    shared_frac = sim.PREFILL_HOST_SHARED_MS_PER_TOKEN / PROD_HOST_SUM

    tpn_csv = RESULTS / f"serving_stage_split_H100_tp{tp}.csv"
    for p in (TP1_CSV, tpn_csv):
        if not p.exists():
            raise SystemExit(f"missing {p} — pull both stage-split legs first")
    rows1 = list(csv.DictReader(TP1_CSV.open()))
    rowsn = list(csv.DictReader(tpn_csv.open()))

    host1 = _ols(rows1, "ttft_ms", ["1", "new", "cached"])
    hostn = _ols(rowsn, "ttft_ms", ["1", "new", "cached"])
    drift = abs(host1["cached"] - PROD_HOST_SUM) / PROD_HOST_SUM
    if drift > HOST_TP1_TOLERANCE:
        raise SystemExit(f"tp1-leg host cached rate {host1['cached']:.6e} drifts "
                         f"{drift*100:.1f}% from the production SUM {PROD_HOST_SUM:.6e} — "
                         "estimator/instrument no longer like-for-like; refusing to emit")

    fa3_1 = _ols(rows1, "prefill_span_ms", ["1", "new", "cached", "fa3"])
    fa3_n = _ols(rowsn, "prefill_span_ms", ["1", "new", "cached", "fa3"])
    if abs(fa3_1["fa3"] - PROD_FA3) / PROD_FA3 > FA3_TP1_TOLERANCE:
        raise SystemExit(f"tp1-leg FA3 coefficient {fa3_1['fa3']:.4e} does not corroborate the "
                         f"production {PROD_FA3:.4e} within {FA3_TP1_TOLERANCE*100:.0f}% — refusing")
    ratio = fa3_n["fa3"] / fa3_1["fa3"]
    if not (0.0 < ratio < 1.0):
        raise SystemExit(f"FA3 tp{tp}/tp1 ratio {ratio:.4f} outside (0,1) — not a sharding signal")
    fa3_tpn = PROD_FA3 * ratio

    out_json = REPO_ROOT / "profile_data" / "kernels" / f"prefill_stage_rates_H100x{tp}.json"
    payload = {
        "schema": "prefill_stage_rates.v1",
        "gpu": f"H100x{tp}", "model": "Llama-3.1-8B", "tp": tp,
        "method": ("like-for-like multiprocess api_server stage-split pair (same script + "
                   "lattice both legs; the round-1 L11 Phase-B tp4 leg). HOST: 3-param OLS "
                   "ttft ~ floor + a*new + b*cached per leg (build_host_split.fit_c1_rate "
                   "family); per-config value = b(tpN) direct, tp1 leg validates the "
                   "estimator against the production SUM. FA3: 4-param OLS prefill_span ~ "
                   "floor + a*new + b*cached + c*new*(cached+new/2) per leg; per-config "
                   "value = production 8.31e-7 x c(tpN)/c(tp1) (ratio transport via the "
                   "tp1 bridge leg, the G3 pattern)."),
        "host_fit_tp1": {k: round(v, 9) for k, v in host1.items()},
        f"host_fit_tp{tp}": {k: round(v, 9) for k, v in hostn.items()},
        "host_tp1_vs_production_sum": {
            "production_sum_ms_per_tok": PROD_HOST_SUM,
            "tp1_leg_ms_per_tok": host1["cached"],
            "rel_drift": drift, "tolerance": HOST_TP1_TOLERANCE},
        "fa3_fit_tp1": {k: (round(v, 9) if k != "fa3" else v) for k, v in fa3_1.items()},
        f"fa3_fit_tp{tp}": {k: (round(v, 9) if k != "fa3" else v) for k, v in fa3_n.items()},
        "fa3_ratio_tpn_over_tp1": ratio,
        "constants": {
            "prefill_host_cached_ms_per_token": hostn["cached"],
            "prefill_fa3_ms_per_token2": fa3_tpn,
        },
        "partition_caveat": (
            f"shared/per-request partition stays the production measured fraction "
            f"{shared_frac:.4f} (tp1 B-sweep band point, prefill_host_split_H100.json): no "
            f"tp{tp} B-sweep exists; only the measured SUM is per-config. The consumer "
            "(_price_step) applies the fraction to the pinned SUM."),
        "fa3_direct_sensitivity": {
            f"direct_c_tp{tp}": fa3_n["fa3"],
            "note": ("the direct tpN serving-stack coefficient; outcome-equivalent to the "
                     "transported value in the round-2 preview (e2el 16.21 vs 16.32) — the "
                     "transport choice is provenance-driven, not outcome-driven.")},
        "fallbacks_it_replaces": {
            "host_sum_ms_per_tok": PROD_HOST_SUM,
            "PREFILL_FA3_MS_PER_TOKEN2": PROD_FA3},
        "_notes": (f"Pinned in the H100x{tp} deployment JSON top-level keys "
                   "prefill_host_cached_ms_per_token / prefill_fa3_ms_per_token2 -> "
                   "RooflineParams (loader). Every config without the pins keeps the module "
                   "constants byte-identically. Regenerate: python3 -m "
                   f"profiling.emit.build_stage_split_rates --tp {tp}. Sources: "
                   f"serving_stage_split.py --tensor-parallel-size {{1,{tp}}} on h100 "
                   "(h100_setup.md; tp4 leg measured 2026-06-11, GPUs 4-7)."),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"HOST cached: tp1 {host1['cached']*1e3:.4f} ms/1k (prod {PROD_HOST_SUM*1e3:.4f}, "
          f"drift {drift*100:.1f}%) | tp{tp} {hostn['cached']*1e3:.4f} ms/1k")
    print(f"FA3: tp1 {fa3_1['fa3']:.4e} | tp{tp} {fa3_n['fa3']:.4e} | ratio {ratio:.5f} "
          f"-> transported {fa3_tpn:.6e} (prod {PROD_FA3:.2e})")
    print(f"wrote {out_json}")


def build_consumer(gpu: str, tp: int) -> None:
    """L13 extension (--gpu RTX3090, 2026-06-11): per-config host-cached + FA3 pins from the
    consumer GPU's OWN stage-split legs (serving_stage_split_{gpu}.csv / _tp{tp}.csv).

    AMENDED ESTIMATORS (documented in defit_log_entries/L13-3090multi.md BEFORE any gate;
    decided on consumer-structure grounds, no gate output consulted):

    * HOST = the 4-param ttft OLS cached coefficient ``b`` of ``ttft ~ floor + a*new +
      b*cached + c*new*(cached+new/2)`` on the tp{tp} leg, DIRECT. The original 3-param
      family is UNUSABLE on the RTX3090: its measured FA3 coefficient is ~9x the H100's
      (7.46e-6 vs 8.31e-7), so the quadratic term's projection onto ``cached`` contaminates
      the 3-param coefficient by ~60% (3090 tp1: 3p 1.460e-2 vs 4p 9.014e-3) where on the
      H100 legs the two families agreed within ~6% (5.99e-3 vs 5.63e-3 around the production
      5.887e-3). The consumer (``_price_step``) charges HOST and FA3 as SEPARATE terms, so
      pinning the 3p coefficient alongside an FA3 pin would double-count the quadratic mass.
      CROSS-INSTRUMENT VALIDATION (hard-fail): the SAME 4-param estimator on the H100 tp1
      leg must reproduce the production host SUM within the builder's established 5%
      tolerance (measured: 5.6302e-3 vs 5.8872e-3, -4.4%).
    * FA3 = the 4-param span OLS coefficient ``c`` on the tp{tp} leg, DIRECT (pre-registered
      in L13: the H100 ratio-transport pattern has no {gpu}-measured production anchor to
      transport; L11 showed direct vs transported outcome-equivalent). Validations
      (hard-fail): c(tp1) > 0, and 0 < c(tp{tp})/c(tp1) <= 1 (head-sharding can only help).

    The 3-param sensitivity values are reported in the artifact. H100 default mode
    (``build``) byte-untouched."""
    import simulator.ttft_queue_sim as sim
    prod_sum_live = sim.PREFILL_HOST_SHARED_MS_PER_TOKEN + sim.PREFILL_HOST_PERREQ_MS_PER_TOKEN
    assert abs(prod_sum_live - PROD_HOST_SUM) < 1e-15, "production host SUM moved — update anchor"

    tp1_csv = RESULTS / f"serving_stage_split_{gpu}.csv"
    tpn_csv = RESULTS / f"serving_stage_split_{gpu}_tp{tp}.csv"
    for p in (tp1_csv, tpn_csv, TP1_CSV):
        if not p.exists():
            raise SystemExit(f"missing {p} — pull the stage-split legs first "
                             "(the H100 tp1 leg is the estimator-validation bridge)")
    rows1 = list(csv.DictReader(tp1_csv.open()))
    rowsn = list(csv.DictReader(tpn_csv.open()))
    rows_h100 = list(csv.DictReader(TP1_CSV.open()))

    # Cross-instrument estimator validation on the H100 bridge leg (hard-fail).
    host_h100_4p = _ols(rows_h100, "ttft_ms", ["1", "new", "cached", "fa3"])
    drift = abs(host_h100_4p["cached"] - PROD_HOST_SUM) / PROD_HOST_SUM
    if drift > HOST_TP1_TOLERANCE:
        raise SystemExit(f"4-param estimator on the H100 bridge leg gives "
                         f"{host_h100_4p['cached']:.6e}, drifting {drift*100:.1f}% from the "
                         f"production SUM {PROD_HOST_SUM:.6e} — estimator family no longer "
                         "validates cross-instrument; refusing to emit")

    host1 = _ols(rows1, "ttft_ms", ["1", "new", "cached", "fa3"])
    hostn = _ols(rowsn, "ttft_ms", ["1", "new", "cached", "fa3"])
    fa3_1 = _ols(rows1, "prefill_span_ms", ["1", "new", "cached", "fa3"])
    fa3_n = _ols(rowsn, "prefill_span_ms", ["1", "new", "cached", "fa3"])
    host1_3p = _ols(rows1, "ttft_ms", ["1", "new", "cached"])
    hostn_3p = _ols(rowsn, "ttft_ms", ["1", "new", "cached"])
    if fa3_1["fa3"] <= 0:
        raise SystemExit(f"tp1-leg FA3 coefficient {fa3_1['fa3']:.4e} not positive — refusing")
    ratio = fa3_n["fa3"] / fa3_1["fa3"]
    if not (0.0 < ratio <= 1.0):
        raise SystemExit(f"FA3 tp{tp}/tp1 ratio {ratio:.4f} outside (0,1] — not a sharding "
                         "signal; refusing")

    out_json = REPO_ROOT / "profile_data" / "kernels" / f"prefill_stage_rates_{gpu}x{tp}.json"
    payload = {
        "schema": "prefill_stage_rates.v1",
        "gpu": f"{gpu}x{tp}", "model": "Llama-3.1-8B", "tp": tp,
        "method": ("like-for-like multiprocess api_server stage-split pair (same script + "
                   f"lattice both legs, {gpu} host). HOST: 4-param OLS ttft ~ floor + a*new "
                   "+ b*cached + c*new*(cached+new/2) per leg; per-config value = b(tpN) "
                   "DIRECT (AMENDED from the H100 3-param family: the 3090's ~9x FA3 "
                   "coefficient contaminates the 3-param cached coefficient ~60% via the "
                   "quadratic projection, and the consumer charges HOST and FA3 separately; "
                   "the 4-param estimator reproduces the production H100 SUM within 5% on "
                   "the H100 bridge leg). FA3: same 4-param fit on prefill_span_ms; "
                   "per-config value = c(tpN) DIRECT (no production anchor to "
                   "ratio-transport on this GPU; L13 pre-registration)."),
        "host_fit_tp1_4p": {k: round(v, 9) for k, v in host1.items()},
        f"host_fit_tp{tp}_4p": {k: round(v, 9) for k, v in hostn.items()},
        "host_estimator_validation_h100_bridge": {
            "production_sum_ms_per_tok": PROD_HOST_SUM,
            "h100_tp1_leg_4p_ms_per_tok": host_h100_4p["cached"],
            "rel_drift": drift, "tolerance": HOST_TP1_TOLERANCE},
        "fa3_fit_tp1": {k: (round(v, 9) if k != "fa3" else v) for k, v in fa3_1.items()},
        f"fa3_fit_tp{tp}": {k: (round(v, 9) if k != "fa3" else v) for k, v in fa3_n.items()},
        "fa3_ratio_tpn_over_tp1": ratio,
        "constants": {
            "prefill_host_cached_ms_per_token": hostn["cached"],
            "prefill_fa3_ms_per_token2": fa3_n["fa3"],
        },
        "partition_caveat": (
            "shared/per-request partition stays the production measured fraction "
            f"{sim.PREFILL_HOST_SHARED_MS_PER_TOKEN / PROD_HOST_SUM:.4f} (tp1 B-sweep band "
            f"point, prefill_host_split_H100.json): no {gpu} B-sweep exists; only the "
            "measured SUM is per-config. The consumer (_price_step) applies the fraction "
            "to the pinned SUM."),
        "sensitivity_3param_family": {
            "host_cached_tp1_3p": host1_3p["cached"],
            f"host_cached_tp{tp}_3p": hostn_3p["cached"],
            "note": ("the original H100 3-param coefficients, contaminated here by the "
                     "FA3 quadratic projection (see method) — reported, NOT pinned.")},
        "fallbacks_it_replaces": {
            "host_sum_ms_per_tok": PROD_HOST_SUM,
            "PREFILL_FA3_MS_PER_TOKEN2": PROD_FA3},
        "_notes": (f"Pinned in the {gpu}x{tp} deployment JSON top-level keys "
                   "prefill_host_cached_ms_per_token / prefill_fa3_ms_per_token2 -> "
                   "RooflineParams (loader). Every config without the pins keeps the module "
                   "constants byte-identically. Regenerate: python3 -m "
                   f"profiling.emit.build_stage_split_rates --gpu {gpu} --tp {tp}. "
                   f"Sources: serving_stage_split.py --tensor-parallel-size {{1,{tp}}} on "
                   "the 3090 host (L13, 2026-06-11; GPUs 0 / 0-1 / 0-3, PIX-only PCIe)."),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"HOST cached 4p: tp1 {host1['cached']*1e3:.4f} ms/1k | tp{tp} "
          f"{hostn['cached']*1e3:.4f} ms/1k (3p sensitivity: {host1_3p['cached']*1e3:.4f} / "
          f"{hostn_3p['cached']*1e3:.4f}; h100-bridge drift {drift*100:.1f}%)")
    print(f"FA3: tp1 {fa3_1['fa3']:.4e} | tp{tp} {fa3_n['fa3']:.4e} (ratio {ratio:.5f}) "
          f"-> pinned DIRECT {fa3_n['fa3']:.6e} (module constant {PROD_FA3:.2e})")
    print(f"wrote {out_json}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tp", type=int, default=4, help="tp degree of the second leg")
    ap.add_argument("--gpu", default="H100", choices=["H100", "RTX3090"],
                    help="GPU family of the stage-split legs (L13: RTX3090 uses its OWN legs "
                         "+ the amended 4-param estimators; H100 default byte-untouched)")
    a = ap.parse_args()
    if a.tp < 2:
        raise SystemExit("--tp must be >= 2 (tp1 is the bridge leg)")
    if a.gpu != "H100":
        build_consumer(a.gpu, a.tp)
        return
    build(a.tp)


if __name__ == "__main__":
    main()
