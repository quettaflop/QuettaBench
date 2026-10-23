"""metrics/summary.py and step_fit.py on hand-computed cases."""

import contextlib
import io
import json
import tempfile
import types
import unittest
from array import array
from pathlib import Path

from quettabench.config.schema import Config, EngineCfg, HardwareCfg, ModeCfg, WorkloadCfg
from quettabench.metrics.results import RunWriter
from quettabench.metrics.step_fit import fit, fit_run, ms_at_eval
from quettabench.metrics.summary import (
    CACHE_DRIFT_LIMIT,
    CACHE_DRIFT_NOISE_SDS,
    ITL_COALESCED_MS,
    NO_COUNT,
    RequestResult,
    _busy_time_and_load,
    aggregate,
    aggregate_per_turn,
    annotate_multi_turn_cache_estimate,
    annotate_request_observability,
    bin_widths,
    burst_recovery,
    chunk_shares,
    first_error,
    parse_prometheus,
    pcts,
    percentile,
    print_summary,
    sla_verdict,
    spec_decode_counts,
    steady_section,
    steady_ttfts,
    steady_window,
    success_rate_gate,
    summarize_serve,
    tokens_by_arrival,
    transient_gate,
    usage_gate,
    width_weighted,
    workload_mix,
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
    (TWO, {50: (10.0, 15.0), 90: (20.0, 19.0), 95: (20.0, 19.5), 99: (20.0, 19.900000000000002)}),
    (TIES, {50: (2.0, 2.0), 90: (3.0, 2.6), 95: (3.0, 2.8), 99: (3.0, 2.96)}),
    (SEVEN, {50: (7.0, 7.0), 90: (13.0, 11.8), 95: (13.0, 12.399999999999999), 99: (13.0, 12.879999999999999)}),
    (ARANGE, {50: (499.0, 499.5), 90: (899.0, 899.1), 95: (949.0, 949.05), 99: (989.0, 989.01)}),
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
        # Two TPOTs: the nearest-rank median is the lower one.
        self.assertEqual(s["tpot_ms"], {"p50": 5.0, "p90": 7.0, "p95": 7.0, "p99": 7.0})
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

    def test_a_failed_request_fails_the_verdict_before_any_p99_is_read(self):
        # The survivors here would PASS every bound; one failed request is a FAIL regardless.
        self.assertEqual(sla_verdict(self.S, 900.0, 40.0, failed=1), ["failed_requests"])
        self.assertEqual(sla_verdict({"ttft_ms": None, "tpot_ms": None}, 1.0, 1.0, failed=2), ["failed_requests"])


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
        # Every successful response must report usage; three of ten is an abort, not a pass.
        verdict, msg = usage_gate(_summary(10, 10, 3), False)
        self.assertEqual(verdict, "abort")
        self.assertIn("7 of 10 successful responses carried no usage block", msg)
        self.assertEqual(usage_gate(_summary(10, 10, 10), False), ("pass", None))
        self.assertEqual(usage_gate(_summary(10, 8, 8, {"timeout": 2}), False), ("pass", None))
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
    r.chunk_chars = [1] * max(1, output_tokens)  # a character a chunk, so a token a chunk
    return r


class TestBusyTime(unittest.TestCase):
    def test_contiguous_work_busy_equals_wall(self):
        busy, mean, peak = _busy_time_and_load([_req(0, 1), _req(1, 2), _req(2, 3)])
        self.assertAlmostEqual(busy, 3.0)
        self.assertAlmostEqual(mean, 1.0)
        self.assertEqual(peak, 1)

    def test_idle_gap_is_out_of_busy_time_but_in_the_mean_in_flight(self):
        # 2 s of requests over the 6 s from the first send to the last completion.
        busy, mean, peak = _busy_time_and_load([_req(0, 1), _req(5, 6)])
        self.assertAlmostEqual(busy, 2.0)
        self.assertAlmostEqual(mean, 2 / 6)
        self.assertEqual(peak, 1)

    def test_the_whole_run_is_timed_from_the_rows(self):
        # Sent at 0.5 s and 1 s by the driver's clock, done at 2 s and 4 s, in a 5 s measure(): 3.5 s.
        rows = [_req(0.4, 2.0), _req(0.9, 4.0)]
        rows[0].sent_at_s, rows[1].sent_at_s = 0.5, 1.0
        s = aggregate(rows, duration_s=5.0)
        self.assertAlmostEqual(s.duration_s, 3.5)
        self.assertAlmostEqual(s.mean_inflight_requests, (1.5 + 3.0) / 3.5)

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

    def test_the_whole_run_window_is_first_send_to_last_completion(self):
        # Sent at 0 s and 1 s, done at 1 s and 2 s, in a 5 s run: the rates divide by 2 s, and say so.
        s = aggregate([_req(0, 1), _req(1, 2)], duration_s=5.0, concurrency=1)
        self.assertEqual(s.excluded_requests, 0)
        self.assertEqual((s.measured_window_s, s.measured_window), (2.0, "first send to last completion"))
        self.assertAlmostEqual(s.request_throughput, 1.0)
        # With no request timed, the run's wall time stands in, named as such.
        bare = RequestResult(success=True, ttft=0.1, e2el=0.2, output_tokens=2, usage_reported=True)
        s = aggregate([bare], duration_s=5.0)
        self.assertEqual((s.measured_window_s, s.measured_window), (5.0, "run wall time"))


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
        self.assertAlmostEqual(s.request_throughput, 20 / 10.5)  # first send at 0 s, last completion at 10.5 s
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

    def test_a_session_run_has_no_index_based_steady_ttft(self):
        single = [_req(i * 0.1, i * 0.1 + 0.5) for i in range(12)]
        for i, r in enumerate(single):
            r.request_index = i
        self.assertIsNotNone(
            aggregate(single, duration_s=2.0, load_mode="closed-loop", concurrency=2).steady_median_ttft_ms
        )
        s = aggregate(self._multi_turn(), duration_s=10.0, load_mode="closed-loop", concurrency=2)
        self.assertIsNone(s.steady_median_ttft_ms)
        self.assertIsNone(s.steady_p99_ttft_ms)

    def test_session_throughput_counts_sessions_not_turns(self):
        s = aggregate(self._multi_turn(), duration_s=10.0, load_mode="open-loop", target_rate=0.4)
        self.assertEqual((s.sessions_completed, s.sessions_failed), (4, 0))
        # First send at 0 s; the last turn, session 3's fifth, ends at 3 + 0.4 + 0.5 = 3.9 s.
        self.assertAlmostEqual(s.session_throughput, 4 / 3.9)
        self.assertAlmostEqual(s.request_throughput, 20 / 3.9)

    def test_a_session_with_a_failed_turn_is_not_completed(self):
        rows = self._multi_turn()
        rows[7].success = False  # session 1, turn 2
        s = aggregate(rows, duration_s=10.0, load_mode="open-loop", target_rate=0.4)
        self.assertEqual((s.sessions_completed, s.sessions_failed), (3, 1))
        self.assertAlmostEqual(s.session_throughput, 3 / 3.9)

    def test_a_session_short_of_its_planned_turns_is_not_completed(self):
        rows = [r for r in self._multi_turn() if not (r.session_id == 2 and r.turn_index == 4)]
        planned = {sid: 5 for sid in range(4)}
        s = aggregate(rows, duration_s=10.0, load_mode="open-loop", target_rate=0.4, planned_turns=planned)
        self.assertEqual((s.sessions_completed, s.sessions_failed), (3, 1))

    def test_the_server_count_is_the_cache_hit_and_the_estimate_only_its_ceiling(self):
        # Turn 2 of two sessions repeats a 100-token prompt in full: the estimate says 100% could be reused;
        # the engine reports 50 and 0 tokens actually reused.
        rows = []
        for sid, reported in ((0, 50), (1, 0)):
            first, second = _req(0.0, 1.0, input_tokens=100), _req(1.0, 2.0, input_tokens=100)
            second.server_cached_tokens = reported
            for turn, r in enumerate((first, second)):
                annotate_multi_turn_cache_estimate(r, sid, turn, previous_context_tokens=100 if turn else 0)
            rows += [first, second]
        s = aggregate(rows, duration_s=2.0)
        self.assertEqual((s.server_cached_tokens, s.server_cache_hit_rate), (50, 0.25))
        turns = aggregate_per_turn({0: rows[0::2], 1: rows[1::2]})
        self.assertEqual(turns[1].median_cache_hit_rate, 1.0)
        self.assertEqual(turns[1].median_server_cache_hit_rate, 0.25)
        self.assertIsNone(turns[0].median_server_cache_hit_rate)

    def test_a_server_that_reports_no_cached_tokens_leaves_the_hit_rate_none(self):
        s = aggregate([_req(0.0, 1.0, input_tokens=100)], duration_s=1.0)
        self.assertIsNone(s.server_cached_tokens)
        self.assertIsNone(s.server_cache_hit_rate)

    def test_steady_window_is_the_interval_with_every_agent_busy(self):
        # Window is 3 s to 5 s; closing at the (N - C + 1)-th finish would count 95 s of drain as steady.
        times = {
            1: (0.0, 0.0, 1.0),
            2: (0.0, 0.0, 3.0),
            3: (0.0, 1.0, 2.0),
            4: (0.0, 2.0, 4.0),
            5: (0.0, 3.0, 100.0),
            6: (0.0, 4.0, 5.0),
        }
        w = steady_window(times, concurrency=2)
        self.assertTrue(w["valid"], w)
        self.assertEqual((w["start_s"], w["end_s"], w["window_s"], w["agents"]), (3.0, 5.0, 2.0, 2))

    def test_the_window_opens_at_the_last_agents_own_first_play(self):
        # Agent 0 plays four 2 ms sessions before agent 1 has even started, so the first four admissions are all
        # agent 0's; the first C admissions would open the window at 0.008 s, long before agent 3's first play
        # ends at 10.03 s. Every agent then plays back to back to the 320 s deadline.
        times, agent_of = {}, {}
        sid = 0
        for agent in range(4):
            t = agent * 0.01
            lengths = [0.002] * 4 + [5.0] * 64 if agent == 0 else [10.0] + [5.0] * 62
            for length in lengths:
                times[sid], agent_of[sid] = (t, t, t + length), agent
                t, sid = t + length, sid + 1
        with_agents = steady_window(times, concurrency=4, end_s=320.0, agent_of=agent_of)
        self.assertTrue(with_agents["valid"], with_agents)
        self.assertAlmostEqual(with_agents["start_s"], 10.03)
        self.assertAlmostEqual(steady_window(times, concurrency=4, end_s=320.0)["start_s"], 0.008)
        never = {k: v for k, v in times.items() if agent_of[k] != 3}
        w = steady_window(never, concurrency=4, end_s=320.0, agent_of=agent_of)
        self.assertEqual((w["valid"], w["reason"]), (False, "1 of 4 agents never finished a first session"))

    def test_the_completion_that_opened_the_window_is_not_counted(self):
        # One agent, three 1 s sessions: the window is 1 s to 3 s and holds two completions, not the cold one at 1 s.
        rows, times = self._agents_back_to_back([1] * 3, agents=1)
        s = steady_section(rows, times, concurrency=1, bin_s=1.0)
        self.assertEqual((s["start_s"], s["end_s"]), (1.0, 3.0))
        self.assertEqual((s["requests_completed"], s["sessions_completed"]), (2, 2))
        self.assertAlmostEqual(s["sessions_per_hour"], 3600.0)
        self.assertAlmostEqual(s["request_throughput"], 1.0)

    def test_the_synchronized_first_sessions_are_outside_the_window(self):
        # Opens at the last first-session finish (7 s), closes at the first finish after the last admission (14 s).
        times = {
            0: (0.0, 0.0, 4.0),
            1: (0.0, 0.0, 6.0),
            2: (0.0, 0.0, 7.0),
            3: (0.0, 4.0, 9.0),
            4: (0.0, 6.0, 10.0),
            5: (0.0, 7.0, 12.0),
            6: (0.0, 9.0, 14.0),
            7: (0.0, 10.0, 16.0),
            8: (0.0, 12.0, 18.0),
        }
        w = steady_window(times, concurrency=3)
        self.assertTrue(w["valid"], w)
        self.assertEqual((w["start_s"], w["end_s"], w["window_s"]), (7.0, 14.0, 7.0))

    @staticmethod
    def _two_recycling_agents():
        # Two agents play back to back from 0 s, one in 200 s plays and one in 300 s plays, until a 1000 s
        # deadline. The first plays finish at 200 s and 300 s; the last start at 800 s and 900 s.
        times = {f"a{k}": (200.0 * k, 200.0 * k, 200.0 * k + 200) for k in range(5)}
        times.update({f"b{k}": (300.0 * k, 300.0 * k, 300.0 * k + 300) for k in range(4)})
        return times

    def test_a_recycled_window_opens_after_the_first_plays_and_runs_to_the_deadline(self):
        times = self._two_recycling_agents()
        w = steady_section([], times, concurrency=2, end_s=1000.0)
        self.assertTrue(w["valid"], w)
        self.assertEqual((w["start_s"], w["end_s"], w["window_s"]), (300.0, 1000.0, 700.0))
        # Finished inside (300, 1000]: a1 to a4 at 400, 600, 800 and 1000 s, b1 and b2 at 600 and 900 s; b0's
        # finish at 300 s opened the window and was a cold first play.
        self.assertEqual(w["plays_completed"], 6)

    def test_a_recycled_window_is_judged_by_its_length_not_the_plays_started(self):
        # Four plays for two agents (N = 2C) would void a per-session run; recycled, the window is 700 s.
        times = {"a0": (0.0, 0.0, 200.0), "a1": (200.0, 200.0, 1100.0), "b0": (0.0, 0.0, 300.0)}
        times["b1"] = (300.0, 300.0, 1200.0)
        self.assertTrue(steady_window(times, concurrency=2, end_s=1000.0)["valid"])
        self.assertFalse(steady_window(times, concurrency=2)["valid"])

    def test_a_recycled_window_names_the_condition_that_voids_it(self):
        times = self._two_recycling_agents()
        short = steady_section([], times, concurrency=2, end_s=400.0)
        self.assertFalse(short["valid"])
        self.assertEqual(short["reason"], "the window lasts 100.0 s; a recycled window needs at least 300 s")
        # a1 at 400 s; b0's finish at 300 s opened the window and was a cold first play
        self.assertEqual((short["window_s"], short["plays_completed"]), (100.0, 1))
        late = steady_window({"a0": (0.0, 0.0, 120.0), "b0": (0.0, 0.0, 150.0)}, concurrency=2, end_s=100.0)
        self.assertEqual(late["reason"], "the first plays end at 150.0 s, after the 100 s deadline")
        idle = dict(times, a2=(450.0, 450.0, 600.0))  # a's agent idles from a1's end at 400 s to 450 s
        del idle["a3"], idle["a4"]
        idle.update(a3=(600.0, 600.0, 800.0), a4=(800.0, 800.0, 1000.0))
        self.assertIn("an agent was idle 50.000 s from 400.000 s", steady_window(idle, 2, end_s=1000.0)["reason"])

    def test_two_sessions_per_agent_or_fewer_leave_no_steady_window_and_no_substitute(self):
        # Three sessions for two agents: the agents are full from 0 s to 2 s, but that is their synchronized first
        # sessions, not a steady state. TB at 32 agents on 34 sessions is this case.
        times = {0: (0.0, 0.0, 1.0), 1: (0.0, 0.0, 3.0), 2: (0.0, 1.0, 2.0), 3: (0.0, 2.0, 4.0)}
        w = steady_window(times, concurrency=2)
        self.assertFalse(w["valid"])
        self.assertIn("4 sessions for 2 agents; a steady window needs more than 4", w["reason"])
        self.assertNotIn("output_token_throughput", steady_section([], times, 2))

    def test_an_agent_left_idle_by_recorded_arrivals_voids_the_window(self):
        # The first sessions end at 1 s and 3 s, so the window opens at 3 s; S4 arrives only at 5 s although S2
        # freed its agent at 4 s: one agent sat idle for 1 s inside the window.
        times = {
            0: (0.0, 0.0, 1.0),
            1: (0.0, 0.0, 3.0),
            2: (1.0, 1.0, 4.0),
            3: (3.0, 3.0, 6.0),
            4: (5.0, 5.0, 7.0),
            5: (6.0, 6.0, 8.0),
        }
        w = steady_window(times, concurrency=2)
        self.assertFalse(w["valid"])
        self.assertIn("an agent was idle 1.000 s from 4.000 s", w["reason"])

    def test_an_agent_idle_from_before_the_window_voids_it_for_the_part_inside(self):
        # The first sessions end at 1 s and 2 s, so the window opens at 2 s, but S2 arrives only at 3 s and S3 at
        # 3.5 s: the agent idle since 1 s stays idle for the window's first 1.5 s.
        times = {
            0: (0.0, 0.0, 1.0),
            1: (0.0, 0.0, 2.0),
            2: (3.0, 3.0, 6.0),
            3: (3.5, 3.5, 7.0),
            4: (6.0, 6.0, 9.0),
            5: (7.0, 7.0, 10.0),
        }
        w = steady_window(times, concurrency=2)
        self.assertFalse(w["valid"])
        self.assertIn("an agent was idle 1.500 s from 2.000 s", w["reason"])

    def test_workload_mix_reports_served_prompt_tokens_over_the_recorded_count(self):
        # A served model with another tokenizer counts the same text differently; the ratio shows it per request.
        drifted = RequestResult(success=True, input_tokens=950, output_tokens=5, recorded_input_tokens=1000)
        unrecorded = RequestResult(success=True, input_tokens=100, output_tokens=5)
        mix = workload_mix([drifted, unrecorded], [drifted, unrecorded], set())
        self.assertEqual(mix["input_tokens_served_over_recorded"]["p50"], 0.95)
        self.assertIsNone(workload_mix([unrecorded], [unrecorded], set())["input_tokens_served_over_recorded"])

    def _two_agents_ten_sessions(self):
        # Window is 3 s to 15 s; S9's last three turns fall in the drain.
        turns = {sid: 6 if sid == 9 else 3 for sid in range(10)}
        times = {sid: (0.0, 3.0 * (sid // 2), 3.0 * (sid // 2) + turns[sid]) for sid in range(10)}
        rows = []
        for sid, (_, admitted, _) in times.items():
            for t in range(turns[sid]):
                r = _req(admitted + t, admitted + t + 1.0)
                r.session_id, r.turn_index = sid, t
                r.semaphore_acquired_at_s, r.completed_at_s = admitted + t, admitted + t + 1.0
                rows.append(r)
        return rows, times

    def test_steady_section_counts_tokens_by_arrival_and_shows_the_window_was_steady(self):
        rows, times = self._two_agents_ten_sessions()
        next(r for r in rows if (r.session_id, r.turn_index) == (8, 2)).success = False
        s = steady_section(rows, times, concurrency=2, bin_s=4.0)
        self.assertEqual((s["start_s"], s["end_s"]), (3.0, 15.0))
        # 24 turns start between 3 s and 14 s and one fails; S0's and S1's last turns end at 3 s, the opening, and
        # a completion at the opening was a cold first play, so it is not counted.
        self.assertEqual((s["requests_completed"], s["requests_inside"]), (23, 23))
        self.assertAlmostEqual(s["output_token_throughput"], 5 * 23 / 12.0)
        self.assertEqual(s["sessions_completed"], 6)  # S2 to S7; S0 and S1 opened the window, S8 failed, S9 runs on
        self.assertAlmostEqual(s["sessions_per_hour"], 3600 * 6 / 12.0)
        self.assertEqual(s["ttft_ms"][50], 10.0)
        # Every request: (1000 - 10) ms over 4 tokens after the first, so 247.5 ms per token.
        self.assertAlmostEqual(s["tpot_ms"][90], 247.5)
        self.assertAlmostEqual(s["interactivity_p90_tok_s"], 1000 / 247.5)
        self.assertEqual(s["chunk_gap_ms"][50], 10.0)
        tl = s["timeline"]
        self.assertEqual((tl["bin_s"], tl["bins"]), (4.0, 3))
        self.assertAlmostEqual(tl["output_tok_s"]["max"], 10.0)
        # The failed turn's 5 tokens are missing from the last bin, but its 1 s in flight counts: it held the server.
        self.assertAlmostEqual(tl["output_tok_s"]["min"], 35 / 4.0)
        self.assertAlmostEqual(tl["requests_in_flight"]["max"], 2.0)
        self.assertAlmostEqual(tl["requests_in_flight"]["min"], 2.0)

    def _agents_back_to_back(self, turns_of, agents=4):
        # Each agent runs its sessions back to back, one 1 s turn after another; turns_of[s] is session s's turns.
        rows, times, free = [], {}, [0.0] * agents
        for sid, n in enumerate(turns_of):
            admitted = free[sid % agents]
            for t in range(n):
                r = _req(admitted + t, admitted + t + 1.0)
                r.session_id, r.turn_index = sid, t
                r.sent_at_s, r.completed_at_s = admitted + t, admitted + t + 1.0
                rows.append(r)
            times[sid] = (admitted, admitted, admitted + n)
            free[sid % agents] = admitted + n
        return rows, times

    def test_a_window_whose_cache_hit_share_moved_between_its_halves_is_not_steady(self):
        # A cache still filling, or starting to evict, reads a hit share the server will not keep. 80 sessions of
        # 3 turns over 4 agents: the window is 3 s to 60 s, so each half holds about 38 plays.
        rows, times = self._agents_back_to_back([3] * 80)
        for r in rows:
            r.server_cached_tokens = 9 if r.sent_at_s < 31.5 else 5
        s = steady_section(rows, times, concurrency=4)
        self.assertFalse(s["valid"])
        self.assertIn("moved from 0.900 in the window's first half to 0.500", s["reason"])
        self.assertIn("over 0.050", s["reason"])
        self.assertFalse(s["cache"]["steady"])
        self.assertAlmostEqual(s["cache"]["noise_sd"], 0.0)
        self.assertTrue(all(n >= 30 for n in s["cache"]["plays_per_half"]))

    def test_a_deeper_second_half_is_a_mix_shift_not_cache_drift(self):
        # The second half holds 6-turn sessions, the first 3-turn ones; turn 0 is never cached and later turns are,
        # so the raw share rises, but at the window's own mix of turn depths the halves agree.
        rows, times = self._agents_back_to_back([3] * 40 + [6] * 40)
        for r in rows:
            r.server_cached_tokens = 0 if r.turn_index == 0 else 9
        s = steady_section(rows, times, concurrency=4)
        self.assertTrue(s["valid"], s.get("reason"))
        c = s["cache"]
        self.assertTrue(c["steady"])
        self.assertAlmostEqual(c["drift"], 0.0)
        raw = []
        for first in (True, False):
            half = [r for r in rows if 3.0 <= r.sent_at_s and (r.sent_at_s < 46.5) == first]
            raw.append(sum(r.server_cached_tokens for r in half) / sum(r.input_tokens for r in half))
        self.assertLess(raw[0] + 0.02, raw[1])  # the raw shares did move with the mix

    def test_a_per_play_hit_or_miss_widens_the_limit_past_its_floor(self):
        # A data-parallel server hits a whole play or none of it: every third play in the first half and every second
        # in the second misses, a drift over 0.05 that resampling plays shows is within noise.
        rows, times = self._agents_back_to_back([3] * 80)
        for r in rows:
            miss_every = 3 if r.sent_at_s < 31.5 else 2
            r.server_cached_tokens = 0 if r.session_id % miss_every == 0 else 9
        c = steady_section(rows, times, concurrency=4)["cache"]
        self.assertGreater(abs(c["drift"]), CACHE_DRIFT_LIMIT)
        self.assertGreater(c["limit"], abs(c["drift"]))
        self.assertAlmostEqual(c["limit"], CACHE_DRIFT_NOISE_SDS * c["noise_sd"])
        self.assertTrue(c["steady"])

    def test_too_few_plays_report_the_cache_shares_without_judging(self):
        rows, times = self._two_agents_ten_sessions()
        for r in rows:
            r.server_cached_tokens = 9 if r.semaphore_acquired_at_s < 9.0 else 5
        s = steady_section(rows, times, concurrency=2, bin_s=4.0)
        self.assertTrue(s["valid"])
        self.assertAlmostEqual(s["cache"]["hit_first_half"], 0.9)
        self.assertAlmostEqual(s["cache"]["hit_second_half"], 0.5)
        self.assertIsNone(s["cache"]["steady"])
        for r in rows:
            r.server_cached_tokens = None
        self.assertIsNone(steady_section(rows, times, concurrency=2, bin_s=4.0)["cache"])

    def test_a_silent_tail_books_its_tokens_at_the_finish_chunk(self):
        # 1,000 tokens in 51 chunks: 50 visible chunks of 2 characters from 0.1 s to 5.0 s, then 15 s of silence
        # and a finish chunk carrying a 900-character tool call at 20.0 s. 1,000 characters, so a token each.
        r = RequestResult(success=True, ttft=0.1, itl=[0.1] * 49 + [15.0], output_tokens=1000, input_tokens=7)
        r.sent_at_s, r.chunk_chars = 0.0, [2] * 50 + [900]
        self.assertEqual(tokens_by_arrival([r], 0.0, 10.0, [10.0]), (100, 7, [100]))
        self.assertEqual(tokens_by_arrival([r], 0.0, 30.0, [10.0, 10.0, 10.0]), (1000, 7, [100, 0, 900]))
        # Spread evenly over chunks, the first 10 s would have read 50 * 1000 / 51, about 980 tokens.

    def test_chunks_held_in_arrays_give_the_figures_lists_gave(self):
        def silent_tail(make):
            r = RequestResult(
                success=True, ttft=0.1, itl=make("d", [0.1] * 49 + [15.0]), output_tokens=1000, input_tokens=7
            )
            r.sent_at_s, r.completed_at_s, r.chunk_chars = 0.0, 20.0, make("q", [2] * 50 + [900])
            return r

        figures = []
        for make in (lambda code, values: values, array):
            r = silent_tail(make)
            self.assertEqual(tokens_by_arrival([r], 0.0, 10.0, [10.0]), (100, 7, [100]))
            self.assertEqual(tokens_by_arrival([r], 0.0, 30.0, [10.0, 10.0, 10.0]), (1000, 7, [100, 0, 900]))
            s = aggregate([r], duration_s=20.0)
            self.assertAlmostEqual(s.mean_itl_ms, 398.0)  # (49 * 100 ms + 15000 ms) / 50
            figures.append(s.to_dict())
        self.assertEqual(figures[0], figures[1])

    def test_chunk_shares_follow_the_server_counts_else_the_characters_and_are_never_negative(self):
        def shares(output_tokens, chars=None, counts=None):
            r = RequestResult(success=True, output_tokens=output_tokens, chunk_chars=chars, chunk_tokens=counts)
            return chunk_shares(r, len(chars or counts))

        # A role chunk, three content chunks of 2, 1 and 3 characters, a bare finish chunk, 6 tokens: the chunks
        # with characters are not one a token, so the characters split them.
        self.assertEqual(shares(6, [0, 2, 1, 3, 0]), [0, 2, 1, 3, 0])
        # The same chunks with 3 tokens: one per chunk with characters, whatever their length (vLLM's usual
        # stream; the character split would have read 1, 0.5 and 1.5).
        self.assertEqual(shares(3, [0, 2, 1, 3, 0]), [0, 1, 1, 1, 0])
        # Fewer tokens than chunks: 2 tokens over three 1-character chunks, nothing on the empty ones.
        self.assertEqual(shares(2, [0, 1, 1, 1, 0]), [0, 2 / 3, 2 / 3, 2 / 3, 0])
        # A tool call the parser emits whole on the finish chunk: 20 of 25 characters, so 8 of 10 tokens.
        self.assertEqual(shares(10, [0, 5, 0, 20]), [0, 2, 0, 8])
        # The server's cumulative counts under continuous usage: the third chunk carried 3 tokens (speculative).
        self.assertEqual(shares(6, [0, 1, 1, 1, 0], [0, 1, 4, 6, 6]), [0, 1, 3, 2, 0])
        # One chunk without a count: back to the characters.
        self.assertEqual(shares(4, [0, 1, 3], [0, NO_COUNT, 4]), [0, 1, 3])
        # No characters at all, or none recorded: the finish chunk carries everything.
        self.assertEqual(shares(4, [0, 0, 0]), [0, 0, 4])
        self.assertEqual(chunk_shares(RequestResult(success=True, output_tokens=4), 3), [0, 0, 4])

    def test_the_window_workload_mix_is_what_its_requests_asked_of_the_server(self):
        def row(sid, turn, sent, done, isl, cached):
            r = _req(sent, done, input_tokens=isl, output_tokens=10)
            r.session_id, r.turn_index, r.sent_at_s, r.server_cached_tokens = sid, turn, sent, cached
            return r

        # Session 1 waits 0.5 s then 0.2 s between turns; session 2 sends its second turn at once.
        a = [row(1, 0, 0.0, 1.0, 100, 0), row(1, 1, 1.5, 3.0, 200, 150), row(1, 2, 3.2, 4.0, 300, 250)]
        b = [row(2, 0, 0.0, 2.0, 50, None), row(2, 1, 2.0, 5.0, 150, 100)]
        mix = workload_mix(a + b, a + b, [1, 2], {1: "short", 2: "long"})
        self.assertEqual(mix["requests"], 5)
        self.assertEqual(mix["input_tokens"], {"p10": 50, "p50": 150, "p90": 300, "mean": 160.0})
        self.assertEqual(mix["output_tokens"], {"p10": 10, "p50": 10, "p90": 10, "mean": 10.0})
        # Uncached 100, 50, 50 and 50; the row without a server count is left out.
        self.assertEqual(mix["uncached_prefill_tokens"], {"p10": 50, "p50": 50, "p90": 100, "mean": 62.5})
        waits = mix["tool_wait_s"]
        for stat, want in (("p10", 0.0), ("p50", 0.2), ("p90", 0.5), ("mean", 0.7 / 3)):
            self.assertAlmostEqual(waits[stat], want)
        self.assertEqual(mix["session_turns"], {"p10": 2, "p50": 2, "p90": 3, "mean": 2.5})
        self.assertEqual(mix["plays_by_stratum"], {"short": 1, "long": 1})

    def test_a_last_bin_under_half_a_bin_folds_into_the_one_before(self):
        self.assertEqual(bin_widths(10.001, 10.0), [10.001])  # a 0.001 s bin of its own would read rate 0 and cv 1.0
        self.assertEqual(bin_widths(24.0, 10.0), [10.0, 14.0])
        self.assertEqual(bin_widths(25.0, 10.0), [10.0, 10.0, 5.0])  # exactly half a bin stands
        self.assertEqual(bin_widths(26.0, 10.0), [10.0, 10.0, 6.0])
        self.assertEqual(bin_widths(4.0, 10.0), [4.0])
        # The 12 s window from 3 s to 15 s in 5 s bins: [5, 7], each at 10 tok/s, not a third bin of 2 s.
        rows, times = self._two_agents_ten_sessions()
        tl = steady_section(rows, times, concurrency=2, bin_s=5.0)["timeline"]
        self.assertEqual(tl["bins"], 2)
        self.assertEqual(
            (tl["output_tok_s"]["min"], tl["output_tok_s"]["max"], tl["output_tok_s"]["cv"]), (10.0, 10.0, 0.0)
        )

    def test_bin_rates_are_weighted_by_bin_width(self):
        # 10 tok/s over 10 s and 20 tok/s over 15 s: mean 400 / 25 = 16; variance (10 * 36 + 15 * 16) / 25 = 24.
        s = width_weighted([10.0, 20.0], [10.0, 15.0])
        self.assertEqual((s["min"], s["mean"], s["max"]), (10.0, 16.0, 20.0))
        self.assertAlmostEqual(s["cv"], 24**0.5 / 16)

    def test_the_cold_start_ttft_is_reported_beside_the_window_not_dropped(self):
        rows, times = self._two_agents_ten_sessions()
        for (sid, turn), ttft in {(0, 0): 0.5, (1, 0): 0.3}.items():
            next(r for r in rows if (r.session_id, r.turn_index) == (sid, turn)).ttft = ttft
        s = steady_section(rows, times, concurrency=2, bin_s=4.0)
        # The first sessions' turn 0 on the cold cache: 300 ms and 500 ms.
        self.assertEqual(s["first_turn_ttft_ms"], {"n": 2, 50: 300.0, 90: 500.0, 99: 500.0})
        # Sent before the window opened at 3 s: S0's and S1's three turns, 10 ms apart from the two cold ones.
        self.assertEqual(s["ramp"]["ttft_ms"], {"n": 6, 50: 10.0, 90: 500.0, 99: 500.0})
        self.assertEqual(s["ttft_ms"][99], 10.0)
        # Without a window there is no ramp, but the first turns still count.
        short = {sid: t for sid, t in times.items() if sid < 4}
        none = steady_section(rows, short, concurrency=2)
        self.assertFalse(none["valid"])
        self.assertEqual(none["first_turn_ttft_ms"]["n"], 2)
        self.assertNotIn("ramp", none)

    def test_the_cold_start_sample_is_each_agents_own_first_session(self):
        # Agent 0 plays three quick sessions before the others start: the first four admissions are agent 0's
        # and would report its 1 ms first turn four times. With the agent map, each agent's own first turn counts.
        rows, times, agent_of = [], {}, {}
        starts = {0: [0.0, 0.01, 0.02, 0.03], 1: [0.04], 2: [0.05], 3: [0.06]}
        first_ttft = {0: 0.001, 1: 0.1, 2: 0.2, 3: 0.3}
        sid = 0
        for agent, admitted in starts.items():
            for k, t in enumerate(admitted):
                r = _req(t, t + 0.005, ttft=first_ttft[agent] if k == 0 else 0.001)
                r.session_id, r.turn_index, r.sent_at_s, r.completed_at_s = sid, 0, t, t + 0.005
                rows.append(r)
                times[sid], agent_of[sid] = (t, t, t + 0.005), agent
                sid += 1
        s = steady_section(rows, times, concurrency=4, agent_of=agent_of)
        self.assertEqual(s["first_turn_ttft_ms"], {"n": 4, 50: 100.0, 90: 300.0, 99: 300.0})
        self.assertEqual(steady_section(rows, times, concurrency=4)["first_turn_ttft_ms"][99], 1.0)

    def test_the_window_takes_a_request_by_its_send_not_its_slot_grant(self):
        # S2's first turn got its slot at 2.95 s, before the window opened at 3 s, but was sent at 3 s.
        rows, times = self._two_agents_ten_sessions()
        r = next(r for r in rows if (r.session_id, r.turn_index) == (2, 0))
        r.semaphore_acquired_at_s, r.sent_at_s = 2.95, 3.0
        s = steady_section(rows, times, concurrency=2, bin_s=4.0)
        self.assertEqual(s["requests_inside"], 24)  # every turn sent from 3 s to 14 s
        self.assertEqual(s["ramp"]["ttft_ms"]["n"], 6)  # S0's and S1's three turns, sent before 3 s

    def test_a_request_straddling_the_window_edge_counts_only_its_tokens_inside(self):
        rows, times = self._two_agents_ten_sessions()
        # Stretch S9's third turn (14 s to 15 s) so its five chunks arrive at 14.01, 14.51, 15.01, 15.51 and 16.01 s.
        r = next(r for r in rows if (r.session_id, r.turn_index) == (9, 2))
        r.itl, r.e2el, r.completed_at_s = [0.5] * 4, 2.02, 16.02
        s = steady_section(rows, times, concurrency=2, bin_s=4.0)
        # 24 turns start inside the 12 s window: 23 deliver all 5 tokens inside it, S9's third only 2.
        self.assertAlmostEqual(s["output_token_throughput"], (5 * 23 + 2) / 12.0)

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

    def test_multiturn_session_count_ignores_concurrency(self):
        profile = get_profile("swebench-qwen-multiturn-medium")

        sessions, source = resolve_multi_turn_num_sessions(profile, 320)

        self.assertEqual((sessions, source), (320, "cli"))


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
