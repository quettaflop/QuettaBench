"""metrics/summary.py and step_fit.py on hand-computed cases."""

import contextlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path

from quettabench.config.schema import Config, EngineCfg, HardwareCfg, ModeCfg, WorkloadCfg
from quettabench.metrics.results import RunWriter
from quettabench.metrics.step_fit import fit, fit_run, ms_at_eval
from quettabench.metrics.summary import (
    ITL_COALESCED_MS,
    RequestResult,
    _busy_time_and_load,
    aggregate,
    aggregate_per_turn,
    annotate_multi_turn_cache_estimate,
    annotate_request_observability,
    burst_recovery,
    first_error,
    parse_prometheus,
    pcts,
    percentile,
    print_summary,
    sla_verdict,
    spec_decode_counts,
    steady_ttfts,
    success_rate_gate,
    summarize_serve,
    transient_gate,
    usage_gate,
)
from quettabench.workloads.base import Request
from quettabench.workloads.live import get_profile
from quettabench.workloads.sessions import resolve_multi_turn_num_sessions

SEVEN = [3.0, 9.0, 1.0, 7.0, 5.0, 11.0, 13.0]
TIES = [1.0, 2.0, 2.0, 2.0, 3.0]
TWO = [10.0, 20.0]
SINGLE = [42.0]
ARANGE = [float(i) for i in range(1000)]

TABLE = [
    (SINGLE, {50: (42.0, 42.0), 90: (42.0, 42.0), 95: (42.0, 42.0), 99: (42.0, 42.0)}),
    (TWO, {50: (20.0, 15.0), 90: (20.0, 19.0), 95: (20.0, 19.5), 99: (20.0, 19.900000000000002)}),
    (TIES, {50: (2.0, 2.0), 90: (3.0, 2.6), 95: (3.0, 2.8), 99: (3.0, 2.96)}),
    (SEVEN, {50: (7.0, 7.0), 90: (13.0, 11.8), 95: (13.0, 12.399999999999999), 99: (13.0, 12.879999999999999)}),
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
                self.assertEqual(pcts(values, method), {f"p{p}": v[col] for p, v in by_p.items()})

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
        self.assertEqual(s["ttft_ms"], {"p50": 20.0, "p90": 30.0, "p95": 30.0, "p99": 30.0})
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


class SteadyTtftTest(unittest.TestCase):
    def test_summarize_serve_drops_the_first_wave(self):
        records = [
            {"ttft_ms": 100.0 if i < 2 else 10.0, "tpot_ms": None, "out_tokens": 1, "index": i} for i in range(6)
        ]
        s = summarize_serve(records, 1.0, "nearest", closed_concurrency=2)
        self.assertEqual(s["ttft_ms"]["p99"], 100.0)
        self.assertEqual(s["steady_ttft_ms"], {"p50": 10.0, "p90": 10.0, "p95": 10.0, "p99": 10.0})

    def test_too_few_left_is_none(self):
        records = [{"ttft_ms": 1.0, "tpot_ms": None, "out_tokens": 1, "index": i} for i in range(5)]
        self.assertIsNone(summarize_serve(records, 1.0, "nearest", closed_concurrency=3)["steady_ttft_ms"])

    def test_open_load_has_no_steady_key(self):
        records = [{"ttft_ms": 1.0, "tpot_ms": None, "out_tokens": 1}]
        self.assertNotIn("steady_ttft_ms", summarize_serve(records, 1.0, "nearest"))

    def test_helper_counts_by_dispatch_index(self):
        self.assertEqual(steady_ttfts([(0, 9.0), (3, 1.0), (2, 2.0), (4, 3.0)], 2), [1.0, 2.0, 3.0])
        self.assertIsNone(steady_ttfts([(0, 9.0), (2, 2.0)], 2))


class SpecCountsTest(unittest.TestCase):
    def test_summed_over_engines(self):
        text = (
            'vllm:spec_decode_num_drafts_total{engine="0"} 10\n'
            'vllm:spec_decode_num_drafts_total{engine="1"} 5\n'
            'vllm:spec_decode_num_drafts_created{engine="0"} 1.7e9\n'
            'vllm:spec_decode_num_draft_tokens_total{engine="0"} 45\n'
            'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 20\n'
        )
        self.assertEqual(
            spec_decode_counts(parse_prometheus(text)), {"accepted_tokens": 20.0, "draft_tokens": 45.0, "drafts": 15.0}
        )

    def test_none_without_spec_decoding(self):
        self.assertIsNone(spec_decode_counts(parse_prometheus("process_open_fds 42\n")))


class SlaVerdictTest(unittest.TestCase):
    S = {"ttft_ms": {"p99": 900.0}, "tpot_ms": {"p99": 40.0}}

    def test_bounds(self):
        self.assertEqual(sla_verdict(self.S, 500.0, 50.0), ["ttft_p99"])
        self.assertEqual(sla_verdict(self.S, 500.0, 30.0), ["ttft_p99", "tpot_p99"])
        self.assertEqual(sla_verdict(self.S, 900.0, 40.0), [])
        self.assertEqual(sla_verdict(self.S, None, None), [])
        self.assertEqual(sla_verdict(self.S, 0, 0), [])

    def test_missing_tpot_is_not_checked(self):
        self.assertEqual(sla_verdict({"ttft_ms": {"p99": 1.0}, "tpot_ms": None}, 10.0, 1.0), [])


class BurstRecoveryTest(unittest.TestCase):
    def test_recovers_two_windows_after_peak(self):
        recs = (
            [{"arrival_s": t, "ttft_ms": 100.0} for t in range(0, 5)]
            + [{"arrival_s": t, "ttft_ms": 9000.0} for t in range(5, 10)]
            + [{"arrival_s": t, "ttft_ms": 100.0} for t in range(15, 20)]
        )
        self.assertEqual(burst_recovery(recs, 1000.0, 5.0), (10.0, 9000.0))

    def test_never_recovers(self):
        recs = [{"arrival_s": t, "ttft_ms": 9000.0} for t in range(0, 10)]
        self.assertEqual(burst_recovery(recs, 1000.0, 5.0), (None, 9000.0))

    def test_bucket_p99_is_nearest_rank(self):
        recs = [{"arrival_s": 0.0, "ttft_ms": float(v)} for v in range(1, 11)]
        self.assertEqual(burst_recovery(recs, 1.0, 5.0), (None, 10.0))

    def test_no_records_or_no_sla(self):
        self.assertEqual(burst_recovery([], 1000.0), (None, None))
        self.assertEqual(burst_recovery([{"arrival_s": 0.0, "ttft_ms": 1.0}], None), (None, None))


def _summary(n, ok, usage, kinds=None):
    return types.SimpleNamespace(
        num_requests=n, successful_requests=ok, usage_reported_requests=usage, error_kinds=kinds or {}
    )


class GateTest(unittest.TestCase):
    def test_success_rate(self):
        self.assertEqual(
            success_rate_gate(_summary(0, 0, 0), 0.9),
            ("abort", "ABORT: No requests completed. Minimum success rate: 90%"),
        )
        verdict, msg = success_rate_gate(_summary(10, 8, 8, {"timeout": 1, "http_error": 1}), 0.9)
        self.assertEqual(verdict, "abort")
        self.assertEqual(msg, "ABORT: Success rate 80.0% below minimum 90% (8/10); failures: http_error=1, timeout=1")
        self.assertEqual(success_rate_gate(_summary(10, 9, 9), 0.9), ("pass", None))

    def test_usage(self):
        self.assertEqual(usage_gate(_summary(10, 10, 0), False)[0], "abort")
        verdict, msg = usage_gate(_summary(10, 10, 0), True)
        self.assertEqual(verdict, "warn")
        self.assertIn("--allow-missing-usage was passed", msg)
        self.assertEqual(usage_gate(_summary(10, 10, 3), False), ("pass", None))
        self.assertEqual(usage_gate(_summary(10, 0, 0), False), ("pass", None))

    def test_transient(self):
        verdict, msg = transient_gate(10, 4)
        self.assertEqual(verdict, "warn")
        lines = msg.split("\n")
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0], "WARNING: only 2.5 requests per concurrency slot (10 requests at concurrency 4).")
        self.assertEqual(lines[3], "         to at least 32 for a settled measurement.")
        for args in ((16, 4), (3, 1), (0, 8)):
            self.assertEqual(transient_gate(*args), ("pass", None))

    def test_first_error(self):
        rs = [
            None,
            RequestResult(success=True),
            RequestResult(success=False),
            RequestResult(success=False, error=ValueError("x")),
        ]
        self.assertEqual(first_error(rs), "x")
        self.assertEqual(first_error(iter([RequestResult(success=True)])), "no error text recorded")


class CacheEstimateTests(unittest.TestCase):
    def _result(self, input_tokens=1000):
        from quettabench.metrics.summary import RequestResult

        return RequestResult(success=True, input_tokens=input_tokens, output_tokens=200)

    def test_synthetic_reply_is_not_reusable(self):
        from quettabench.metrics.summary import annotate_multi_turn_cache_estimate

        r = annotate_multi_turn_cache_estimate(
            self._result(1000),
            session_id=0,
            turn_index=1,
            previous_context_tokens=770,
            previous_output_tokens=200,
            reply_in_cache=False,
        )
        self.assertEqual(r.cached_context_tokens, 770)
        self.assertEqual(r.new_prefill_tokens, 230)
        self.assertEqual(r.cache_estimate_source, "previous_prompt_tokens")

    def test_the_engines_own_reply_is_reusable(self):
        from quettabench.metrics.summary import annotate_multi_turn_cache_estimate

        r = annotate_multi_turn_cache_estimate(
            self._result(1000),
            session_id=0,
            turn_index=1,
            previous_context_tokens=770,
            previous_output_tokens=200,
            reply_in_cache=True,
        )
        self.assertEqual(r.cached_context_tokens, 970)
        self.assertEqual(r.new_prefill_tokens, 30)
        self.assertEqual(r.cache_estimate_source, "previous_prompt_and_reply_tokens")

    def test_reusable_prefix_cannot_exceed_the_prompt(self):
        from quettabench.metrics.summary import annotate_multi_turn_cache_estimate

        r = annotate_multi_turn_cache_estimate(
            self._result(800),
            session_id=0,
            turn_index=1,
            previous_context_tokens=770,
            previous_output_tokens=200,
            reply_in_cache=True,
        )
        self.assertEqual(r.cached_context_tokens, 800)
        self.assertEqual(r.new_prefill_tokens, 0)


def _req(
    start, end, *, success=True, error_kind=None, ttft=0.01, input_tokens=10, output_tokens=5, usage_reported=True
):
    r = RequestResult(
        success=success,
        ttft=ttft,
        itl=[0.01] * max(0, output_tokens - 1),
        e2el=end - start,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usage_reported=usage_reported,
        error_kind=error_kind,
        error="boom" if error_kind else None,
    )
    r.semaphore_acquired_at_s = start
    r.completed_at_s = end
    return r


class TestBusyTime(unittest.TestCase):
    def test_contiguous_work_busy_equals_wall(self):
        busy, mean, peak = _busy_time_and_load([_req(0, 1), _req(1, 2), _req(2, 3)])
        self.assertAlmostEqual(busy, 3.0)
        self.assertAlmostEqual(mean, 1.0)
        self.assertEqual(peak, 1)

    def test_idle_gap_excluded_from_busy_time(self):
        busy, mean, peak = _busy_time_and_load([_req(0, 1), _req(5, 6)])
        self.assertAlmostEqual(busy, 2.0)
        self.assertAlmostEqual(mean, 1.0)
        self.assertEqual(peak, 1)

    def test_overlapping_requests_counted_once(self):
        busy, mean, peak = _busy_time_and_load([_req(0, 3), _req(0, 3), _req(0, 3)])
        self.assertAlmostEqual(busy, 3.0)
        self.assertAlmostEqual(mean, 3.0)
        self.assertEqual(peak, 3)

    def test_drain_is_busy_but_underutilized(self):
        """Mean in-flight (1.6 against a peak of 4) shows the drain."""
        busy, mean, peak = _busy_time_and_load([_req(0, 1), _req(0, 2), _req(0, 3), _req(0, 10)])
        self.assertAlmostEqual(busy, 10.0)
        self.assertEqual(peak, 4)
        self.assertAlmostEqual(mean, 1.6)
        self.assertLess(mean, peak)

    def test_no_usable_timestamps_is_not_a_crash(self):
        self.assertEqual(_busy_time_and_load([]), (0.0, 0.0, 0))
        bare = RequestResult(success=True)
        self.assertEqual(_busy_time_and_load([bare]), (0.0, 0.0, 0))

    def test_zero_length_interval_ignored(self):
        self.assertEqual(_busy_time_and_load([_req(1.0, 1.0)]), (0.0, 0.0, 0))


class TestBusyThroughput(unittest.TestCase):
    def test_busy_throughput_exceeds_wall_clock_when_idle(self):
        results = [_req(0, 1), _req(9, 10)]
        s = aggregate(results, duration_s=10.0, concurrency=1)
        self.assertAlmostEqual(s.busy_time_s, 2.0)
        self.assertAlmostEqual(s.request_throughput, 0.2)
        self.assertAlmostEqual(s.busy_request_throughput, 1.0)
        self.assertAlmostEqual(s.output_token_throughput, 1.0)
        self.assertAlmostEqual(s.busy_output_token_throughput, 5.0)

    def test_failed_requests_still_occupy_the_server(self):
        results = [_req(0, 1), _req(0, 30, success=False, error_kind="timeout")]
        s = aggregate(results, duration_s=30.0, concurrency=2)
        self.assertAlmostEqual(s.busy_time_s, 30.0)


class TestSteadyTtft(unittest.TestCase):
    @staticmethod
    def _wave(n, concurrency):
        results = []
        for i in range(n):
            r = _req(i, i + 1)
            r.ttft = 1.0 if i < concurrency else 0.1
            r.request_index = i
            results.append(r)
        return results

    def test_closed_loop_reports_steady_state(self):
        s = aggregate(self._wave(8, 2), duration_s=8.0, concurrency=2, load_mode="closed-loop")
        self.assertAlmostEqual(s.p99_ttft_ms, 1000.0)
        self.assertAlmostEqual(s.steady_median_ttft_ms, 100.0)
        self.assertAlmostEqual(s.steady_p99_ttft_ms, 100.0)

    def test_none_when_fewer_than_concurrency_remain(self):
        s = aggregate(self._wave(5, 3), duration_s=5.0, concurrency=3, load_mode="closed-loop")
        self.assertIsNone(s.steady_median_ttft_ms)
        self.assertIsNone(s.steady_p99_ttft_ms)

    def test_open_loop_has_none(self):
        s = aggregate(self._wave(8, 2), duration_s=8.0, concurrency=2, load_mode="open-loop")
        self.assertIsNone(s.steady_median_ttft_ms)


class TestCoalescedItl(unittest.TestCase):
    @staticmethod
    def _run(gaps_ms):
        r = _req(0, 1, output_tokens=len(gaps_ms) + 1)
        r.itl = [g / 1000 for g in gaps_ms]
        return aggregate([r], duration_s=1.0, concurrency=1)

    def test_fraction_counts_gaps_under_the_threshold(self):
        s = self._run([10.0] * 6 + [0.0, 0.01, ITL_COALESCED_MS / 2, ITL_COALESCED_MS * 2])
        self.assertAlmostEqual(s.itl_coalesced_frac, 0.3)

    def test_no_itl_is_zero(self):
        self.assertEqual(aggregate([_req(0, 1, output_tokens=1)], duration_s=1.0).itl_coalesced_frac, 0.0)

    def _printed(self, gaps_ms):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_summary(self._run(gaps_ms))
        return buf.getvalue()

    def test_warns_above_one_percent(self):
        self.assertIn("inter-token gaps are under", self._printed([10.0] * 90 + [0.0] * 10))

    def test_quiet_at_the_old_client_rate(self):
        self.assertNotIn("inter-token gaps are under", self._printed([10.0] * 2000 + [0.0]))


class TestFailureTaxonomy(unittest.TestCase):
    def test_timeouts_are_counted_and_classified(self):
        results = [
            _req(0, 1),
            _req(0, 30, success=False, error_kind="timeout"),
            _req(0, 1, success=False, error_kind="http_error"),
        ]
        s = aggregate(results, duration_s=30.0)
        self.assertEqual(s.successful_requests, 1)
        self.assertEqual(s.failed_requests, 2)
        self.assertEqual(s.timeout_requests, 1)
        self.assertEqual(s.error_kinds, {"timeout": 1, "http_error": 1})

    def test_partial_ttft_on_failures_is_retained_but_not_aggregated(self):
        """Their e2el is a timeout artifact, so they stay out of the percentiles."""
        results = [
            _req(0, 1, ttft=0.010),
            _req(0, 30, success=False, error_kind="timeout", ttft=0.999),
        ]
        s = aggregate(results, duration_s=30.0)
        self.assertEqual(s.failed_with_partial_ttft, 1)
        self.assertAlmostEqual(s.mean_ttft_ms, 10.0)

    def test_unclassified_failure_falls_back_to_unknown(self):
        s = aggregate([_req(0, 1, success=False, error_kind=None)], duration_s=1.0)
        self.assertEqual(s.error_kinds, {"unknown": 1})


class TestDiscardFirst(unittest.TestCase):
    def _run(self):
        results = [_req(i, i + 1) for i in range(10)]
        for r in results[:3]:
            r.excluded_from_summary = True
        return results

    def test_held_out_requests_leave_the_summary(self):
        s = aggregate(self._run(), duration_s=10.0, concurrency=1)
        self.assertEqual(s.excluded_requests, 3)
        self.assertEqual(s.num_requests, 7)
        self.assertEqual(s.successful_requests, 7)

    def test_window_excludes_the_held_out_span(self):
        s = aggregate(self._run(), duration_s=10.0, concurrency=1)
        self.assertAlmostEqual(s.measured_window_s, 7.0)
        self.assertAlmostEqual(s.request_throughput, 1.0)
        self.assertNotAlmostEqual(s.successful_requests / 10.0, s.request_throughput)

    def test_held_out_intervals_leave_busy_time(self):
        s = aggregate(self._run(), duration_s=10.0, concurrency=1)
        self.assertAlmostEqual(s.busy_time_s, 7.0)

    def test_no_exclusions_keeps_reported_duration(self):
        s = aggregate([_req(0, 1), _req(1, 2)], duration_s=5.0, concurrency=1)
        self.assertEqual(s.excluded_requests, 0)
        self.assertAlmostEqual(s.measured_window_s, 5.0)


class TestLoadModeDiagnostics(unittest.TestCase):
    def test_open_loop_records_mode_and_offered_rate(self):
        s = aggregate(
            [_req(i, i + 1) for i in range(10)], duration_s=10.0, concurrency=40, load_mode="open-loop", target_rate=2.0
        )
        self.assertEqual(s.load_mode, "open-loop")
        self.assertAlmostEqual(s.target_rate, 2.0)

    def test_open_loop_keeping_up_is_arrival_limited_not_draining(self):
        results = [_req(i * 0.5, i * 0.5 + 1) for i in range(20)]
        s = aggregate(results, duration_s=10.0, concurrency=40, load_mode="open-loop", target_rate=2.0)
        self.assertAlmostEqual(s.request_throughput, 2.0)
        self.assertLess(s.mean_inflight_requests, 40 * 0.7)

    def test_closed_loop_still_flags_draining(self):
        s = aggregate([_req(0, 10)], duration_s=10.0, concurrency=40, load_mode="closed-loop")
        self.assertEqual(s.load_mode, "closed-loop")
        self.assertLess(s.mean_inflight_requests, 40 * 0.7)


class TestOpenLoopUnits(unittest.TestCase):
    def _multi_turn(self, n_sessions=4, turns_each=5):
        out = []
        for sid in range(n_sessions):
            for t in range(turns_each):
                r = _req(sid * 1.0 + t * 0.1, sid * 1.0 + t * 0.1 + 0.5)
                r.session_id, r.turn_index = sid, t
                out.append(r)
        return out

    def test_session_throughput_counts_sessions_not_turns(self):
        s = aggregate(self._multi_turn(), duration_s=10.0, load_mode="open-loop", target_rate=0.4)
        self.assertEqual(s.sessions_completed, 4)
        self.assertAlmostEqual(s.session_throughput, 0.4)
        self.assertAlmostEqual(s.request_throughput, 2.0)

    def test_single_turn_has_no_sessions_so_falls_back_to_requests(self):
        s = aggregate([_req(i, i + 1) for i in range(10)], duration_s=10.0, load_mode="open-loop", target_rate=1.0)
        self.assertEqual(s.sessions_completed, 0)
        self.assertAlmostEqual(s.request_throughput, 1.0)


class TestSaturationVerdict(unittest.TestCase):
    @staticmethod
    def _verdict(offered, served, first, second):
        growing = first > 0.5 and second > 1.5 * first
        return (served < 0.75 * offered) or growing

    def test_healthy_rungs_are_not_flagged(self):
        self.assertFalse(self._verdict(0.5, 0.447, 9.8, 4.4))
        self.assertFalse(self._verdict(1.0, 0.82, 22.5, 26.7))

    def test_saturated_rung_caught_by_rate_despite_falling_inflight(self):
        self.assertTrue(self._verdict(2.0, 0.16, 125.6, 91.7))

    def test_growing_backlog_caught_even_when_rate_looks_fine(self):
        self.assertTrue(self._verdict(1.0, 0.95, 10.0, 40.0))

    def test_drain_alone_does_not_trip_the_rate_check(self):
        self.assertFalse(self._verdict(0.5, 0.447, 9.8, 4.4))


class TestWarmupWidthCoverage(unittest.TestCase):
    def test_peak_inflight_beyond_warmup_width_is_recorded(self):
        results = [_req(0, 5) for _ in range(39)]
        s = aggregate(
            results, duration_s=5.0, concurrency=8, load_mode="open-loop", target_rate=2.0, warmup_concurrency=8
        )
        self.assertEqual(s.max_inflight_requests, 39)
        self.assertGreater(s.max_inflight_requests, s.warmup_concurrency)

    def test_adequate_warmup_width_leaves_no_gap(self):
        results = [_req(0, 5) for _ in range(39)]
        s = aggregate(
            results, duration_s=5.0, concurrency=8, load_mode="open-loop", target_rate=2.0, warmup_concurrency=64
        )
        self.assertLessEqual(s.max_inflight_requests, s.warmup_concurrency)


class TestStationarity(unittest.TestCase):
    """Little's Law cannot fail: mean_inflight is sum(durations) / busy_time."""

    def test_steady_run_shows_no_drift(self):
        results = [_req(i * 1.0, i * 1.0 + 2.0) for i in range(20)]
        s = aggregate(results, duration_s=21.0)
        self.assertAlmostEqual(s.mean_inflight_first_half, s.mean_inflight_second_half, delta=0.3)

    def test_growing_queue_is_visible_in_halves(self):
        results = [_req(i * 1.0, i * 1.0 + 1.0 + i * 0.8) for i in range(20)]
        s = aggregate(results, duration_s=40.0)
        self.assertGreater(s.mean_inflight_second_half, s.mean_inflight_first_half)

    def test_drain_dominated_run_falls(self):
        results = [_req(0.0, 1.0 + i * 1.0) for i in range(20)]
        s = aggregate(results, duration_s=21.0)
        self.assertLess(s.mean_inflight_second_half, s.mean_inflight_first_half * 0.6)

    def test_littles_law_is_an_identity_not_a_check(self):
        results = [_req(i * 0.5, i * 0.5 + 2.0) for i in range(30)]
        s = aggregate(results, duration_s=20.0)
        mean_duration = s.busy_time_s * s.mean_inflight_requests / s.successful_requests
        self.assertAlmostEqual(
            s.busy_request_throughput * mean_duration,
            s.mean_inflight_requests,
            places=6,
            msg="Little's Law holds by construction here, so it validates nothing",
        )

    def test_single_request_does_not_crash(self):
        s = aggregate([_req(0.0, 1.0)], duration_s=1.0)
        self.assertEqual(s.mean_inflight_first_half, 0.0)
        self.assertEqual(s.mean_inflight_second_half, 0.0)


class TestUsageAccounting(unittest.TestCase):
    def test_missing_usage_is_visible_in_the_summary(self):
        results = [_req(0, 1, usage_reported=False, input_tokens=0, output_tokens=0)]
        s = aggregate(results, duration_s=1.0)
        self.assertEqual(s.successful_requests, 1)
        self.assertEqual(s.usage_reported_requests, 0)
        self.assertEqual(s.total_output_tokens, 0)
        self.assertEqual(s.output_token_throughput, 0.0)

    def test_usage_reported_requests_counted(self):
        s = aggregate([_req(0, 1), _req(1, 2)], duration_s=2.0)
        self.assertEqual(s.usage_reported_requests, 2)


class RunnerCacheMetadataTests(unittest.TestCase):
    def test_annotates_successful_multi_turn_request(self):
        result = RequestResult(success=True, input_tokens=250, output_tokens=40)

        annotate_multi_turn_cache_estimate(
            result,
            session_id=12,
            turn_index=2,
            previous_context_tokens=175,
        )

        self.assertEqual(result.session_id, 12)
        self.assertEqual(result.turn_index, 2)
        self.assertEqual(result.previous_context_tokens, 175)
        self.assertEqual(result.total_context_tokens, 250)
        self.assertEqual(result.cached_context_tokens, 175)
        self.assertEqual(result.new_prefill_tokens, 75)
        self.assertAlmostEqual(result.cache_hit_rate, 0.7)
        self.assertEqual(result.cache_estimate_source, "previous_prompt_tokens")

    def test_block_aligned_cache_estimate_accounts_for_uncached_tail(self):
        result = RequestResult(success=True, input_tokens=250, output_tokens=40)

        annotate_multi_turn_cache_estimate(
            result,
            session_id=12,
            turn_index=2,
            previous_context_tokens=175,
            cache_block_size=16,
        )

        self.assertEqual(result.cache_block_size, 16)
        self.assertEqual(result.block_aligned_cached_context_tokens, 160)
        self.assertEqual(result.uncached_prefix_tail_tokens, 15)
        self.assertEqual(result.block_aligned_new_prefill_tokens, 90)
        self.assertAlmostEqual(result.block_aligned_cache_hit_rate, 160 / 250)
        self.assertEqual(result.total_context_blocks, 16)
        self.assertEqual(result.cached_context_blocks, 10)
        self.assertEqual(result.new_prefill_blocks, 6)

    def test_cache_estimate_clamps_previous_context_to_current_prompt(self):
        result = RequestResult(success=True, input_tokens=128, output_tokens=40)

        annotate_multi_turn_cache_estimate(
            result,
            session_id=1,
            turn_index=3,
            previous_context_tokens=256,
            cache_block_size=16,
        )

        self.assertEqual(result.cached_context_tokens, 128)
        self.assertEqual(result.new_prefill_tokens, 0)
        self.assertEqual(result.cache_hit_rate, 1.0)
        self.assertEqual(result.block_aligned_cached_context_tokens, 128)
        self.assertEqual(result.block_aligned_new_prefill_tokens, 0)
        self.assertEqual(result.new_prefill_blocks, 0)

    def test_cache_estimate_skips_block_fields_when_block_size_missing_or_invalid(self):
        for block_size in (None, 0, -8):
            with self.subTest(block_size=block_size):
                result = RequestResult(success=True, input_tokens=128, output_tokens=40)
                annotate_multi_turn_cache_estimate(
                    result,
                    session_id=1,
                    turn_index=1,
                    previous_context_tokens=64,
                    cache_block_size=block_size,
                )

                self.assertEqual(result.cached_context_tokens, 64)
                self.assertEqual(result.new_prefill_tokens, 64)
                self.assertIsNone(result.block_aligned_cached_context_tokens)
                self.assertIsNone(result.block_aligned_new_prefill_tokens)

    def test_failed_request_gets_identity_fields_but_no_cache_math(self):
        result = RequestResult(success=False, input_tokens=250, error="timeout")

        annotate_multi_turn_cache_estimate(
            result,
            session_id=9,
            turn_index=4,
            previous_context_tokens=200,
            cache_block_size=16,
        )

        self.assertEqual(result.session_id, 9)
        self.assertEqual(result.turn_index, 4)
        self.assertEqual(result.previous_context_tokens, 200)
        self.assertEqual(result.cache_estimate_source, "unavailable")
        self.assertIsNone(result.total_context_tokens)
        self.assertIsNone(result.block_aligned_cached_context_tokens)

    def test_request_observability_records_shape_timing_and_metadata(self):
        request = Request(
            [],
            0,
            messages=[
                {"role": "system", "content": "s"},
                {"role": "user", "content": "hello"},
            ],
            output_length=32,
            metadata={"planned_new_prefill_tokens": 17},
        )
        result = RequestResult(success=True)

        annotate_request_observability(
            result,
            request_index=5,
            request=request,
            scheduled_at_s=1.0,
            dispatch_started_at_s=1.1,
            semaphore_acquired_at_s=1.4,
            completed_at_s=2.0,
        )

        self.assertEqual(result.request_index, 5)
        self.assertEqual(result.max_tokens_requested, 32)
        self.assertEqual(result.message_count, 2)
        self.assertEqual(result.prompt_chars, 6)
        self.assertAlmostEqual(result.client_schedule_delay_s, 0.1)
        self.assertAlmostEqual(result.client_queue_wait_s, 0.3)
        self.assertAlmostEqual(result.client_request_wall_s, 0.6)
        self.assertEqual(result.request_metadata["planned_new_prefill_tokens"], 17)

    def test_request_observability_allows_unscheduled_multiturn_requests(self):
        request = Request([], 0, output_length=8, messages=[{"role": "user", "content": "hello"}], metadata={})
        result = RequestResult(success=True)

        annotate_request_observability(
            result,
            request_index=0,
            request=request,
            scheduled_at_s=None,
            dispatch_started_at_s=2.0,
            semaphore_acquired_at_s=2.25,
            completed_at_s=2.75,
        )

        self.assertIsNone(result.scheduled_at_s)
        self.assertIsNone(result.client_schedule_delay_s)
        self.assertAlmostEqual(result.client_queue_wait_s, 0.25)
        self.assertAlmostEqual(result.client_request_wall_s, 0.5)

    def test_per_turn_summary_includes_cache_estimate_medians(self):
        first = annotate_multi_turn_cache_estimate(
            RequestResult(success=True, input_tokens=100, output_tokens=20),
            session_id=0,
            turn_index=0,
            previous_context_tokens=0,
            cache_block_size=16,
        )
        second = annotate_multi_turn_cache_estimate(
            RequestResult(success=True, input_tokens=260, output_tokens=30),
            session_id=1,
            turn_index=0,
            previous_context_tokens=200,
            cache_block_size=16,
        )

        summaries = aggregate_per_turn({0: [first, second]})

        self.assertEqual(len(summaries), 1)
        summary = summaries[0]
        self.assertEqual(summary.median_input_tokens, 180)
        self.assertEqual(summary.median_output_tokens, 25)
        self.assertEqual(summary.median_new_prefill_tokens, 80)
        self.assertEqual(summary.median_cached_context_tokens, 100)
        self.assertAlmostEqual(summary.median_cache_hit_rate, (0.0 + 200 / 260) / 2)
        self.assertEqual(summary.median_block_aligned_cached_context_tokens, 96)
        self.assertEqual(summary.median_block_aligned_new_prefill_tokens, 84)
        self.assertAlmostEqual(
            summary.median_block_aligned_cache_hit_rate,
            (0.0 + 192 / 260) / 2,
        )

    def test_per_turn_summary_preserves_nonzero_cache_estimates_from_run_results(self):
        result = annotate_multi_turn_cache_estimate(
            RequestResult(success=True, input_tokens=1531, output_tokens=9),
            session_id=0,
            turn_index=1,
            previous_context_tokens=1416,
            cache_block_size=16,
        )

        summary = aggregate_per_turn({1: [result]})[0]

        self.assertEqual(summary.avg_cached_context_tokens, 1416)
        self.assertEqual(summary.median_cached_context_tokens, 1416)
        self.assertEqual(summary.avg_new_prefill_tokens, 115)
        self.assertEqual(summary.median_new_prefill_tokens, 115)
        self.assertAlmostEqual(summary.avg_cache_hit_rate, 1416 / 1531)
        self.assertAlmostEqual(summary.median_cache_hit_rate, 1416 / 1531)
        self.assertEqual(summary.avg_block_aligned_cached_context_tokens, 1408)
        self.assertEqual(summary.median_block_aligned_cached_context_tokens, 1408)

    def test_per_turn_summary_includes_client_queue_wait_median(self):
        first = RequestResult(success=True, input_tokens=100, output_tokens=20)
        first.client_queue_wait_s = 0.010
        second = RequestResult(success=True, input_tokens=100, output_tokens=20)
        second.client_queue_wait_s = 0.030

        summary = aggregate_per_turn({0: [first, second]})[0]

        self.assertEqual(summary.median_client_queue_wait_ms, 20)

    def test_run_dir_serializes_prediction_observability_fields(self):
        result = RequestResult(
            success=True,
            ttft=0.01234,
            itl=[0.004, 0.006],
            e2el=0.030,
            input_tokens=250,
            output_tokens=3,
        )
        result = annotate_request_observability(
            result,
            request_index=7,
            request=Request(
                [],
                0,
                messages=[{"role": "user", "content": "hello world"}],
                output_length=64,
                metadata={"planned_new_prefill_tokens": 75},
            ),
            scheduled_at_s=1.0,
            dispatch_started_at_s=1.2,
            semaphore_acquired_at_s=1.5,
            completed_at_s=1.8,
        )
        annotate_multi_turn_cache_estimate(
            result,
            session_id=3,
            turn_index=2,
            previous_context_tokens=175,
            cache_block_size=16,
        )
        cfg = Config(
            engine=EngineCfg(),
            workload=WorkloadCfg(source="chat-multiturn-short"),
            mode=ModeCfg(prefix_cache_block_size=16),
            hardware=HardwareCfg(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            writer = RunWriter(tmp, cfg)
            writer.append(result)
            writer.finish({"successful_requests": 1})
            run = json.loads((Path(tmp) / "run.json").read_text())
            (row,) = [json.loads(line) for line in (Path(tmp) / "requests.jsonl").open()]

        self.assertEqual(run["config"]["mode"]["prefix_cache_block_size"], 16)
        self.assertEqual(row["request_index"], 7)
        self.assertEqual(row["max_tokens_requested"], 64)
        self.assertEqual(row["message_count"], 1)
        self.assertEqual(row["session_id"], 3)
        self.assertEqual(row["turn_index"], 2)
        self.assertEqual(row["block_aligned_cached_context_tokens"], 160)
        self.assertEqual(row["block_aligned_new_prefill_tokens"], 90)
        self.assertEqual(row["request_metadata"]["planned_new_prefill_tokens"], 75)
        self.assertAlmostEqual(row["client_queue_wait_ms"], 300.0)

    def test_synthetic_multiturn_sessions_floor_at_concurrency(self):
        profile = get_profile("swebench-multiturn-short")

        sessions, source = resolve_multi_turn_num_sessions(profile, concurrency=320)

        self.assertEqual(sessions, 320)
        self.assertEqual(source, "concurrency_floor")


STEPS = [60, 230, 400]
GIB = 1 << 30


def batch(ctx, bs, step, wall, phase="fit", attempt=0, tp=1):
    cell = {"ctx": ctx, "bs": bs, "tp": tp, "step_point": step, "phase": phase, "attempt": attempt}
    return [RequestResult(success=True, e2el=wall, batch_wall_s=wall, cell=dict(cell)) for _ in range(bs)]


def attempt_rows(ctx, bs, walls, attempt):
    return [r for step, wall in zip(STEPS, walls) for r in batch(ctx, bs, step, wall, attempt=attempt)]


class FitTest(unittest.TestCase):
    def test_exact_line(self):
        slope, inter, r2 = fit([64, 128, 256], [1.14, 1.78, 3.06])
        self.assertAlmostEqual(slope, 0.01)
        self.assertAlmostEqual(inter, 0.5)
        self.assertAlmostEqual(r2, 1.0)

    def test_noisy_three_points(self):
        self.assertEqual(fit([1, 2, 3], [1, 3, 2]), (0.5, 1.0, 0.25))

    def test_flat_ys_gives_r2_zero(self):
        self.assertEqual(fit([1, 2, 3], [5.0, 5.0, 5.0]), (0.0, 5.0, 0.0))


class MsAtEvalTest(unittest.TestCase):
    def test_eval_at_centroid_is_the_slope(self):
        self.assertEqual(ms_at_eval(10.0, 0.4, 128, 128, 4096), 10.0)

    def test_no_attention_share_no_shift(self):
        self.assertEqual(ms_at_eval(10.0, 0.0, 512, 128, 4096), 10.0)

    def test_shift_scales_with_attn_and_distance(self):
        self.assertEqual(ms_at_eval(10.0, 0.5, 1000, 100, 800), 15.0)
        self.assertEqual(ms_at_eval(10.0, 0.5, 0, 100, 900), 9.5)


class FitRunTest(unittest.TestCase):
    def test_entry_matches_the_direct_fit(self):
        walls = [0.28, 1.01, 1.73]
        rows = batch(4096, 2, 400, 9.0, phase="warm", attempt=None) + attempt_rows(4096, 2, walls, 0)
        (entry,) = fit_run(rows, eval_pos=50, kv_bytes_per_token=131072, weights_gib=15.0)
        slope, inter, r2 = fit(STEPS, walls)
        kv_gib = 4096 * 2 * 131072 / GIB
        attn = kv_gib / (kv_gib + 15.0)
        self.assertEqual(entry["ms_per_step"], slope * 1e3)
        self.assertEqual(entry["ms_at_eval"], ms_at_eval(slope * 1e3, attn, 50, 230.0, 4096))
        self.assertEqual(
            (entry["r2"], entry["prefill_s"], entry["kv_gib"], entry["attn_share"]), (r2, inter, kv_gib, attn)
        )
        self.assertEqual((entry["raw"], entry["step_points"], entry["mean_steps"]), (walls, STEPS, 230.0))
        self.assertEqual(entry["attempts"], 1)

    def test_best_r2_attempt_wins_earliest_on_tie(self):
        bent, straight = [0.35, 1.51, 2.06], [0.27, 0.96, 1.65]
        rows = (
            attempt_rows(1024, 1, bent, 0)
            + attempt_rows(1024, 1, straight, 1)
            + attempt_rows(1024, 1, [0.27, 0.96, 1.6500001], 2)
        )
        (entry,) = fit_run(rows, 50, 131072, 15.0)
        self.assertEqual((entry["raw"], entry["attempts"]), (straight, 3))
        tie = attempt_rows(1024, 1, bent, 0) + attempt_rows(1024, 1, list(bent), 1)
        self.assertEqual(fit_run(tie, 50, 131072, 15.0)[0]["r2"], fit(STEPS, bent)[2])

    def test_cells_keep_first_seen_order(self):
        rows = attempt_rows(100, 8, [1, 2, 3.5], 0) + attempt_rows(100, 1, [1, 2, 3], 0)
        self.assertEqual([(e["ctx"], e["bs"]) for e in fit_run(rows, 50, 1, 1.0)], [(100, 8), (100, 1)])

    def test_measured_rows_pass_through(self):
        row = {
            "cell": {"ctx": 1024, "bs": 4, "tp": 2, "step_point": None},
            "measured_ms_per_step": 3.5,
            "group": "decode_loop",
            "kv": "fp16",
        }
        self.assertEqual(
            fit_run([row], 50, 1, 1.0), [{"ctx": 1024, "bs": 4, "tp": 2, "ms_per_step": 3.5, "measured": True}]
        )

    def test_rows_without_a_cell_or_fit_phase_are_ignored(self):
        rows = [RequestResult(success=True, e2el=1.0)] + batch(100, 1, 64, 0.5, phase="text", attempt=None)
        self.assertEqual(fit_run(rows, 50, 1, 1.0), [])


if __name__ == "__main__":
    unittest.main()
