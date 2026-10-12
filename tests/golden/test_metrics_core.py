"""Golden: aggregate summaries and percentile values."""

import unittest

from quettabench.metrics.core import RequestResult, aggregate, pcts, percentile
from tests.golden._harness import check_json


def fixture_results():
    normal = [
        RequestResult(
            success=True,
            ttft=0.050 + 0.010 * i,
            itl=[0.020, 0.021, 0.019, 0.022],
            e2el=0.140 + 0.012 * i,
            input_tokens=100 + 10 * i,
            output_tokens=5,
            usage_reported=True,
            semaphore_acquired_at_s=0.1 * i,
            completed_at_s=0.1 * i + 0.140 + 0.012 * i,
            session_id=i % 2,
        )
        for i in range(4)
    ]
    single_token = RequestResult(
        success=True,
        ttft=0.080,
        e2el=0.080,
        input_tokens=64,
        output_tokens=1,
        usage_reported=True,
        semaphore_acquired_at_s=0.50,
        completed_at_s=0.58,
    )
    wall_clock_tpot = RequestResult(
        success=True,
        ttft=0.070,
        e2el=0.370,
        input_tokens=90,
        output_tokens=11,
        usage_reported=False,
        semaphore_acquired_at_s=0.60,
        completed_at_s=0.97,
    )
    partial_ttft = RequestResult(
        success=False,
        ttft=0.300,
        itl=[0.05],
        e2el=None,
        input_tokens=120,
        output_tokens=0,
        error="stream reset",
        error_kind="http_error",
        semaphore_acquired_at_s=1.0,
        completed_at_s=1.4,
    )
    timeout = RequestResult(
        success=False, error="client timeout", error_kind="timeout", semaphore_acquired_at_s=2.0, completed_at_s=3.0
    )
    held_out = RequestResult(
        success=True,
        ttft=9.0,
        e2el=9.5,
        input_tokens=1,
        output_tokens=2,
        excluded_from_summary=True,
        semaphore_acquired_at_s=0.0,
        completed_at_s=9.5,
    )
    return normal + [single_token, wall_clock_tpot, partial_ttft, timeout, held_out]


class AggregateGolden(unittest.TestCase):
    def test_summary_dict(self):
        results = fixture_results()
        summary = aggregate(
            results,
            duration_s=4.0,
            model="m",
            profile="p",
            concurrency=4,
            load_mode="closed",
            target_rate=0.0,
            warmup_concurrency=2,
        )
        tpots = [r.tpot for r in results]
        self.assertIsNone(tpots[4], "single-token request must have no tpot")
        check_json(
            self,
            "aggregate_summary",
            {
                "per_request_tpot_s": tpots,
                "summary": summary.to_dict(),
            },
        )


QUANTILES = (50, 90, 95, 99)
ARRAYS = {
    "empty": [],
    "single": [42.0],
    "two": [10.0, 20.0],
    "ties": [1.0, 2.0, 2.0, 2.0, 3.0],
    "seven": [3.0, 9.0, 1.0, 7.0, 5.0, 11.0, 13.0],
    "arange_1000": [float(i) for i in range(1000)],
}


def _call(fn, *args):
    try:
        return fn(*args)
    except Exception as e:  # the empty-array failure mode is part of the pin
        return {"error": type(e).__name__, "message": str(e)}


class PercentileGolden(unittest.TestCase):
    def test_both_implementations_on_fixed_arrays(self):
        record = {}
        for name, data in ARRAYS.items():
            record[name] = {
                "crossval_pct": {f"p{q}": _call(percentile, data, q, "nearest") for q in QUANTILES},
                "crossval_pcts": _call(pcts, data, "nearest"),
                "metrics_percentile": {f"p{q}": _call(percentile, data, q, "linear") for q in QUANTILES},
            }
        check_json(self, "percentiles", record)


if __name__ == "__main__":
    unittest.main()
