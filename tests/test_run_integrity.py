"""Tests for the run-integrity guarantees added in benchmark schema v5.

Four independent ways a run could previously look clean while being wrong:

  1. Timed-out requests were dropped entirely, and they are systematically the
     SLOWEST ones -- so the latency tail was censored and nothing said so.
  2. A server that never sent a usage block produced 0 tok/s and 0 total tokens
     while still reporting a full set of latency percentiles.
  3. Throughput was always divided by total wall clock, so a schedule that left
     the server idle reported the offered rate as if it were capacity.
  4. Underutilization (the multi-turn per-turn barrier drain) was invisible,
     because during a drain the server is busy -- just underused -- so no
     denominator can reveal it.
"""

import unittest

from src.benchmark.metrics import RequestResult, _busy_time_and_load, aggregate


def _req(start, end, *, success=True, error_kind=None, ttft=0.01,
         input_tokens=10, output_tokens=5, usage_reported=True):
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
        # 0-1 busy, 1-5 idle, 5-6 busy. Wall clock is 6s; only 2s were worked.
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
        """The multi-turn barrier case: busy time cannot detect a drain.

        Four requests start together and finish at 1s/2s/3s/10s. The server is
        never idle, so busy == wall clock and the denominator is unchanged.
        mean_inflight (1.6 against a peak of 4) is the only signal that most of
        the run was spent with the machine nearly empty.
        """
        busy, mean, peak = _busy_time_and_load(
            [_req(0, 1), _req(0, 2), _req(0, 3), _req(0, 10)]
        )
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
        # Two 1s requests inside a 10s window: 8s of the wall clock is idle.
        results = [_req(0, 1), _req(9, 10)]
        s = aggregate(results, duration_s=10.0, concurrency=1)
        self.assertAlmostEqual(s.busy_time_s, 2.0)
        self.assertAlmostEqual(s.request_throughput, 0.2)
        self.assertAlmostEqual(s.busy_request_throughput, 1.0)
        # Token throughputs follow the same denominator swap.
        self.assertAlmostEqual(s.output_token_throughput, 1.0)
        self.assertAlmostEqual(s.busy_output_token_throughput, 5.0)

    def test_failed_requests_still_occupy_the_server(self):
        """A timed-out request held a slot; excluding it would inflate busy throughput."""
        results = [_req(0, 1), _req(0, 30, success=False, error_kind="timeout")]
        s = aggregate(results, duration_s=30.0, concurrency=2)
        self.assertAlmostEqual(s.busy_time_s, 30.0)


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
        """Failed requests keep their partial TTFT for diagnosis only.

        Their e2el is an artifact of the timeout, so letting them into the
        percentiles would be worse than dropping them. They must be counted.
        """
        results = [
            _req(0, 1, ttft=0.010),
            _req(0, 30, success=False, error_kind="timeout", ttft=0.999),
        ]
        s = aggregate(results, duration_s=30.0)
        self.assertEqual(s.failed_with_partial_ttft, 1)
        # Only the successful request's 10ms TTFT reaches the summary.
        self.assertAlmostEqual(s.mean_ttft_ms, 10.0)

    def test_unclassified_failure_falls_back_to_unknown(self):
        s = aggregate([_req(0, 1, success=False, error_kind=None)], duration_s=1.0)
        self.assertEqual(s.error_kinds, {"unknown": 1})


class TestDiscardFirst(unittest.TestCase):
    def _run(self):
        """10 sequential 1s requests; the first 3 are held out."""
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
        """Throughput must not be divided by time the numerator no longer counts."""
        s = aggregate(self._run(), duration_s=10.0, concurrency=1)
        # Kept requests span t=3..t=10.
        self.assertAlmostEqual(s.measured_window_s, 7.0)
        self.assertAlmostEqual(s.request_throughput, 1.0)
        # Against the full 10s duration this would have read 0.7 req/s.
        self.assertNotAlmostEqual(s.successful_requests / 10.0, s.request_throughput)

    def test_held_out_intervals_leave_busy_time(self):
        s = aggregate(self._run(), duration_s=10.0, concurrency=1)
        self.assertAlmostEqual(s.busy_time_s, 7.0)

    def test_no_exclusions_keeps_reported_duration(self):
        s = aggregate([_req(0, 1), _req(1, 2)], duration_s=5.0, concurrency=1)
        self.assertEqual(s.excluded_requests, 0)
        self.assertAlmostEqual(s.measured_window_s, 5.0)


class TestLoadModeDiagnostics(unittest.TestCase):
    """`concurrency` means different things per mode, so diagnosis must too.

    Under closed loop it is the in-flight cap, so mean-inflight << concurrency
    means draining. Under open loop it caps nothing (it only sizes warmup), so
    that comparison is meaningless -- the real question is whether the server
    kept up with the offered arrival rate.
    """

    def test_open_loop_records_mode_and_offered_rate(self):
        s = aggregate([_req(i, i + 1) for i in range(10)], duration_s=10.0,
                      concurrency=40, load_mode="open-loop", target_rate=2.0)
        self.assertEqual(s.load_mode, "open-loop")
        self.assertAlmostEqual(s.target_rate, 2.0)

    def test_open_loop_keeping_up_is_arrival_limited_not_draining(self):
        """Regression: mean-inflight 5.8 vs concurrency 40 previously read as
        'draining' even though the server was matching the offered rate."""
        results = [_req(i * 0.5, i * 0.5 + 1) for i in range(20)]
        s = aggregate(results, duration_s=10.0, concurrency=40,
                      load_mode="open-loop", target_rate=2.0)
        self.assertAlmostEqual(s.request_throughput, 2.0)
        # Well under concurrency=40, which under open loop implies nothing.
        self.assertLess(s.mean_inflight_requests, 40 * 0.7)

    def test_closed_loop_still_flags_draining(self):
        s = aggregate([_req(0, 10)], duration_s=10.0, concurrency=40,
                      load_mode="closed-loop")
        self.assertEqual(s.load_mode, "closed-loop")
        self.assertLess(s.mean_inflight_requests, 40 * 0.7)


class TestWarmupWidthCoverage(unittest.TestCase):
    def test_peak_inflight_beyond_warmup_width_is_recorded(self):
        """The DeepSeek-V4 failure: warmup at width 8, run peaked at 39, so
        unseen batch shapes JIT-compiled mid-measurement."""
        results = [_req(0, 5) for _ in range(39)]
        s = aggregate(results, duration_s=5.0, concurrency=8,
                      load_mode="open-loop", target_rate=2.0, warmup_concurrency=8)
        self.assertEqual(s.max_inflight_requests, 39)
        self.assertGreater(s.max_inflight_requests, s.warmup_concurrency)

    def test_adequate_warmup_width_leaves_no_gap(self):
        results = [_req(0, 5) for _ in range(39)]
        s = aggregate(results, duration_s=5.0, concurrency=8,
                      load_mode="open-loop", target_rate=2.0, warmup_concurrency=64)
        self.assertLessEqual(s.max_inflight_requests, s.warmup_concurrency)


class TestStationarity(unittest.TestCase):
    """Splitting the window is the only non-circular stationarity test we have.

    Little's Law cannot serve: mean_inflight is defined as
    sum(durations)/busy_time, so busy_throughput x mean_duration == mean_inflight
    identically. Halving the window uses the time ordering that definition
    discards, so it can actually fail.
    """

    def test_steady_run_shows_no_drift(self):
        results = [_req(i * 1.0, i * 1.0 + 2.0) for i in range(20)]
        s = aggregate(results, duration_s=21.0)
        self.assertAlmostEqual(s.mean_inflight_first_half,
                               s.mean_inflight_second_half, delta=0.3)

    def test_growing_queue_is_visible_in_halves(self):
        results = [_req(i * 1.0, i * 1.0 + 1.0 + i * 0.8) for i in range(20)]
        s = aggregate(results, duration_s=40.0)
        self.assertGreater(s.mean_inflight_second_half, s.mean_inflight_first_half)

    def test_drain_dominated_run_falls(self):
        """Closed-loop burst then drain: second half must be much lower."""
        results = [_req(0.0, 1.0 + i * 1.0) for i in range(20)]
        s = aggregate(results, duration_s=21.0)
        self.assertLess(s.mean_inflight_second_half, s.mean_inflight_first_half * 0.6)

    def test_littles_law_is_an_identity_not_a_check(self):
        """Pin the reason we do NOT ship a Little's Law validation."""
        results = [_req(i * 0.5, i * 0.5 + 2.0) for i in range(30)]
        s = aggregate(results, duration_s=20.0)
        mean_duration = s.busy_time_s * s.mean_inflight_requests / s.successful_requests
        self.assertAlmostEqual(
            s.busy_request_throughput * mean_duration,
            s.mean_inflight_requests, places=6,
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
        # This is exactly the shape that used to pass silently.
        self.assertEqual(s.total_output_tokens, 0)
        self.assertEqual(s.output_token_throughput, 0.0)

    def test_usage_reported_requests_counted(self):
        s = aggregate([_req(0, 1), _req(1, 2)], duration_s=2.0)
        self.assertEqual(s.usage_reported_requests, 2)


if __name__ == "__main__":
    unittest.main()
