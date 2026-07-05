# profiling/tests/test_build_stage_split_rates.py
"""Tests for profiling/emit/build_stage_split_rates.py (L11 round 2).

The builder derives the per-config prefill HOST-cached SUM and FA3 coefficient from the
like-for-like tp1/tpN serving_stage_split pair. The OLS core must be exact on synthetic
data; when the (gitignored) source CSVs are present, the committed artifact must
regenerate value-identically (the prefill_host_split both-ways pinning precedent).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from profiling.emit.build_stage_split_rates import (
    PROD_FA3, PROD_HOST_SUM, RESULTS, TP1_CSV, _ols,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ART = REPO_ROOT / "profile_data/kernels/prefill_stage_rates_H100x4.json"
TP4_CSV = RESULTS / "serving_stage_split_H100_tp4.csv"


def test_ols_recovers_exact_synthetic_coefficients():
    # y = 7 + 0.02*new + 0.005*cached + 3e-7*new*(cached+new/2), exactly linear in the
    # design -> the closed-form OLS must recover the coefficients to float precision.
    rows = []
    for new in (8.0, 128.0, 512.0, 1024.0, 2048.0):
        for cached in (0.0, 2000.0, 8000.0, 16000.0):
            y = 7.0 + 0.02 * new + 0.005 * cached + 3e-7 * new * (cached + 0.5 * new)
            rows.append({"new": new, "cached": cached, "y": y})
    beta = _ols(rows, "y", ["1", "new", "cached", "fa3"])
    assert abs(beta["1"] - 7.0) < 1e-9
    assert abs(beta["new"] - 0.02) < 1e-12
    assert abs(beta["cached"] - 0.005) < 1e-12
    assert abs(beta["fa3"] - 3e-7) < 1e-15
    assert beta["_fit_mape_pct"] < 1e-9
    # 3-param family (the production host estimator) on data without the cross term
    rows3 = [{"new": r["new"], "cached": r["cached"],
              "y": 5.0 + 0.01 * r["new"] + 0.004 * r["cached"]} for r in rows]
    beta3 = _ols(rows3, "y", ["1", "new", "cached"])
    assert abs(beta3["cached"] - 0.004) < 1e-12


@pytest.mark.skipif(not (TP1_CSV.exists() and TP4_CSV.exists()),
                    reason="gitignored stage-split CSVs not on this checkout")
def test_artifact_regenerates_from_the_csvs():
    rows1 = list(csv.DictReader(TP1_CSV.open()))
    rows4 = list(csv.DictReader(TP4_CSV.open()))
    host1 = _ols(rows1, "ttft_ms", ["1", "new", "cached"])
    host4 = _ols(rows4, "ttft_ms", ["1", "new", "cached"])
    fa3_1 = _ols(rows1, "prefill_span_ms", ["1", "new", "cached", "fa3"])
    fa3_4 = _ols(rows4, "prefill_span_ms", ["1", "new", "cached", "fa3"])
    art = json.loads(ART.read_text())
    assert art["constants"]["prefill_host_cached_ms_per_token"] == host4["cached"]
    assert art["constants"]["prefill_fa3_ms_per_token2"] == PROD_FA3 * (fa3_4["fa3"] / fa3_1["fa3"])
    # the pre-registered validations hold on the real data
    assert abs(host1["cached"] - PROD_HOST_SUM) / PROD_HOST_SUM < 0.05   # tp1 leg ~ production
    assert 0.0 < fa3_4["fa3"] / fa3_1["fa3"] < 1.0                       # sharding signal
