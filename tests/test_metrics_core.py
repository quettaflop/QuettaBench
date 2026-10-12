"""Unit tests for quettabench.metrics.core: percentile methods, the serving
summary, SLA verdict, burst window and run gates.

Percentile expectations are the table in tests/golden/REPORT.md (STOP
finding, item 4), one column per method.
"""

import types
import unittest

from quettabench.metrics.core import (
    RequestResult,
    burst_recovery,
    first_error,
    pcts,
    percentile,
    sla_verdict,
    success_rate_gate,
    summarize_serve,
    transient_gate,
    usage_gate,
)

SEVEN = [3.0, 9.0, 1.0, 7.0, 5.0, 11.0, 13.0]  # unsorted on purpose
TIES = [1.0, 2.0, 2.0, 2.0, 3.0]
TWO = [10.0, 20.0]
SINGLE = [42.0]
ARANGE = [float(i) for i in range(1000)]

# (array, {p: (nearest, linear)})
TABLE = [
    (SINGLE, {50: (42.0, 42.0), 90: (42.0, 42.0), 95: (42.0, 42.0), 99: (42.0, 42.0)}),
    (TWO, {50: (20.0, 15.0), 90: (20.0, 19.0), 95: (20.0, 19.5), 99: (20.0, 19.900000000000002)}),
    (TIES, {50: (2.0, 2.0), 90: (3.0, 2.6), 95: (3.0, 2.8), 99: (3.0, 2.96)}),
    (SEVEN, {50: (7.0, 7.0), 90: (13.0, 11.8), 95: (13.0, 12.399999999999999),
             99: (13.0, 12.879999999999999)}),
    (ARANGE, {50: (500.0, 499.5), 90: (900.0, 899.1), 95: (950.0, 949.05), 99: (990.0, 989.01)}),
]


class PercentileTest(unittest.TestCase):
    def test_report_table_both_methods(self):
        for values, by_p in TABLE:
            for p, (nearest, linear) in by_p.items():
                with self.subTest(n=len(values), p=p):
                    self.assertEqual(percentile(values, p, "nearest"), nearest)
                    self.assertEqual(percentile(values, p, "linear"), linear)

    def test_pcts_maps_the_four_quantiles(self):
        for values, by_p in TABLE:
            for method, col in (("nearest", 0), ("linear", 1)):
                self.assertEqual(pcts(values, method),
                                 {f"p{p}": v[col] for p, v in by_p.items()})

    def test_empty(self):
        self.assertEqual(percentile([], 99, "linear"), 0.0)
        with self.assertRaisesRegex(ValueError, "no values"):
            percentile([], 50, "nearest")
        with self.assertRaises(ValueError):
            pcts([], "nearest")

    def test_input_order_does_not_matter(self):
        self.assertEqual(percentile(SEVEN, 90, "nearest"), percentile(sorted(SEVEN), 90, "nearest"))
        self.assertEqual(percentile(SEVEN, 90, "linear"), percentile(sorted(SEVEN), 90, "linear"))

    def test_unknown_method_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown percentile method"):
            percentile([1.0], 50, "midpoint")


class ServeSummaryTest(unittest.TestCase):
    RECORDS = [
        {"ttft_ms": 10.0, "tpot_ms": 5.0, "out_tokens": 3},
        {"ttft_ms": 30.0, "tpot_ms": None, "out_tokens": 1},
        {"ttft_ms": 20.0, "tpot_ms": 7.0, "out_tokens": 3},
    ]

    def test_nearest(self):
        s = summarize_serve(self.RECORDS, 2.0, method="nearest")
        self.assertEqual(s["n"], 3)
        # n=3: int(0.5*3)=1 -> 20; int(0.9*3)=2 -> 30
        self.assertEqual(s["ttft_ms"], {"p50": 20.0, "p90": 30.0, "p95": 30.0, "p99": 30.0})
        # tpot over [5, 7]: int(0.5*2)=1 -> 7
        self.assertEqual(s["tpot_ms"], {"p50": 7.0, "p90": 7.0, "p95": 7.0, "p99": 7.0})
        self.assertEqual(s["req_s"], 1.5)
        self.assertEqual(s["tok_s"], 3.5)

    def test_linear_is_the_callers_choice(self):
        s = summarize_serve(self.RECORDS, 2.0, method="linear")
        self.assertEqual(s["ttft_ms"]["p50"], 20.0)
        self.assertEqual(s["tpot_ms"]["p50"], 6.0)

    def test_no_tpot_and_zero_wall(self):
        s = summarize_serve([{"ttft_ms": 1.0, "tpot_ms": None, "out_tokens": 1}], 0.0, "nearest")
        self.assertIsNone(s["tpot_ms"])
        self.assertEqual((s["req_s"], s["tok_s"]), (0.0, 0.0))

    def test_no_records_raises(self):
        with self.assertRaisesRegex(ValueError, "no completed requests"):
            summarize_serve([], 1.0, "nearest")


class SlaVerdictTest(unittest.TestCase):
    S = {"ttft_ms": {"p99": 900.0}, "tpot_ms": {"p99": 40.0}}

    def test_bounds(self):
        self.assertEqual(sla_verdict(self.S, 500.0, 50.0), ["ttft_p99"])
        self.assertEqual(sla_verdict(self.S, 500.0, 30.0), ["ttft_p99", "tpot_p99"])
        self.assertEqual(sla_verdict(self.S, 900.0, 40.0), [])  # at the bound passes
        self.assertEqual(sla_verdict(self.S, None, None), [])
        self.assertEqual(sla_verdict(self.S, 0, 0), [])

    def test_missing_tpot_is_not_checked(self):
        self.assertEqual(sla_verdict({"ttft_ms": {"p99": 1.0}, "tpot_ms": None}, 10.0, 1.0), [])


class BurstRecoveryTest(unittest.TestCase):
    def test_recovers_two_windows_after_peak(self):
        recs = ([{"arrival_s": t, "ttft_ms": 100.0} for t in range(0, 5)]
                + [{"arrival_s": t, "ttft_ms": 9000.0} for t in range(5, 10)]
                + [{"arrival_s": t, "ttft_ms": 100.0} for t in range(15, 20)])
        # buckets 0, 1 (peak), 3 (back under): recovery = (3 - 1) * 5 s
        self.assertEqual(burst_recovery(recs, 1000.0, 5.0), (10.0, 9000.0))

    def test_never_recovers(self):
        recs = [{"arrival_s": t, "ttft_ms": 9000.0} for t in range(0, 10)]
        self.assertEqual(burst_recovery(recs, 1000.0, 5.0), (None, 9000.0))

    def test_bucket_p99_is_nearest_rank(self):
        # one bucket of [1..10]: nearest p99 = sorted[int(0.99 * 10)] = 10, linear would be 9.91
        recs = [{"arrival_s": 0.0, "ttft_ms": float(v)} for v in range(1, 11)]
        self.assertEqual(burst_recovery(recs, 1.0, 5.0), (None, 10.0))

    def test_no_records_or_no_sla(self):
        self.assertEqual(burst_recovery([], 1000.0), (None, None))
        self.assertEqual(burst_recovery([{"arrival_s": 0.0, "ttft_ms": 1.0}], None), (None, None))


def _summary(n, ok, usage, kinds=None):
    return types.SimpleNamespace(num_requests=n, successful_requests=ok,
                                 usage_reported_requests=usage, error_kinds=kinds or {})


class GateTest(unittest.TestCase):
    def test_success_rate(self):
        self.assertEqual(success_rate_gate(_summary(0, 0, 0), 0.9),
                         ("abort", "ABORT: No requests completed. Minimum success rate: 90%"))
        verdict, msg = success_rate_gate(_summary(10, 8, 8, {"timeout": 1, "http_error": 1}), 0.9)
        self.assertEqual(verdict, "abort")
        self.assertEqual(msg, "ABORT: Success rate 80.0% below minimum 90% (8/10); "
                              "failures: http_error=1, timeout=1")
        self.assertEqual(success_rate_gate(_summary(10, 9, 9), 0.9), ("pass", None))

    def test_usage(self):
        self.assertEqual(usage_gate(_summary(10, 10, 0), False)[0], "abort")
        verdict, msg = usage_gate(_summary(10, 10, 0), True)
        self.assertEqual(verdict, "warn")
        self.assertIn("--allow-missing-usage was passed", msg)
        self.assertEqual(usage_gate(_summary(10, 10, 3), False), ("pass", None))
        self.assertEqual(usage_gate(_summary(10, 0, 0), False), ("pass", None))  # nothing succeeded

    def test_transient(self):
        verdict, msg = transient_gate(10, 4)
        self.assertEqual(verdict, "warn")
        lines = msg.split("\n")
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0], "WARNING: only 2.5 requests per concurrency slot "
                                   "(10 requests at concurrency 4).")
        self.assertEqual(lines[3], "         to at least 32 for a settled measurement.")
        for args in ((16, 4), (3, 1), (0, 8)):
            self.assertEqual(transient_gate(*args), ("pass", None))

    def test_first_error(self):
        rs = [None, RequestResult(success=True), RequestResult(success=False),
              RequestResult(success=False, error=ValueError("x"))]
        self.assertEqual(first_error(rs), "x")
        self.assertEqual(first_error(iter([RequestResult(success=True)])), "no error text recorded")


class CacheEstimateTests(unittest.TestCase):
    """Reusable prefix includes the previous reply only when it is the engine's output."""

    def _result(self, input_tokens=1000):
        from quettabench.metrics.core import RequestResult
        return RequestResult(success=True, input_tokens=input_tokens, output_tokens=200)

    def test_synthetic_reply_is_not_reusable(self):
        from quettabench.metrics.core import annotate_multi_turn_cache_estimate
        r = annotate_multi_turn_cache_estimate(
            self._result(1000), session_id=0, turn_index=1,
            previous_context_tokens=770, previous_output_tokens=200,
            reply_in_cache=False)
        self.assertEqual(r.cached_context_tokens, 770)
        self.assertEqual(r.new_prefill_tokens, 230)
        self.assertEqual(r.cache_estimate_source, "previous_prompt_tokens")

    def test_the_engines_own_reply_is_reusable(self):
        from quettabench.metrics.core import annotate_multi_turn_cache_estimate
        r = annotate_multi_turn_cache_estimate(
            self._result(1000), session_id=0, turn_index=1,
            previous_context_tokens=770, previous_output_tokens=200,
            reply_in_cache=True)
        self.assertEqual(r.cached_context_tokens, 970)
        self.assertEqual(r.new_prefill_tokens, 30)
        self.assertEqual(r.cache_estimate_source, "previous_prompt_and_reply_tokens")

    def test_reusable_prefix_cannot_exceed_the_prompt(self):
        from quettabench.metrics.core import annotate_multi_turn_cache_estimate
        r = annotate_multi_turn_cache_estimate(
            self._result(800), session_id=0, turn_index=1,
            previous_context_tokens=770, previous_output_tokens=200,
            reply_in_cache=True)
        self.assertEqual(r.cached_context_tokens, 800)
        self.assertEqual(r.new_prefill_tokens, 0)

if __name__ == "__main__":
    unittest.main()
