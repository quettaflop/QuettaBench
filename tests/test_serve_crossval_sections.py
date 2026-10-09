"""SPECSUM, SLA, burst and the goodput sweep on serve's one path, with the
summary lines in the exact formats legacy/trace_serve.py printed."""

import contextlib
import io
import unittest
from unittest import mock

from quettabench.config.schema import ModeCfg, WorkloadCfg
from quettabench.metrics.core import RequestResult
from quettabench.metrics.results import RunWriter, legacy_lines
from quettabench.modes import serve
from quettabench.workloads.base import Request
from quettabench.workloads.live import RealTrace
from tests.test_serve_procedure import FakeDriver, config, single_trace


class TokenDriver(FakeDriver):
    """Returns generated token ids: the reference with token 3 changed."""

    async def send(self, request, **kw):
        result = await super().send(request, **kw)
        ref = request.output_token_ids or []
        result.generated_token_ids = [t if i != 2 else -1 for i, t in enumerate(ref)]
        return result


def run(trace, driver, mode, workload=serve.DEFAULT_WORKLOAD):
    writer = RunWriter(None, config(mode))
    with contextlib.redirect_stdout(io.StringIO()) as out:
        try:
            summary = serve.run(trace, driver, mode, writer, workload)
        except SystemExit as e:
            return e.code, out.getvalue()
    return summary, out.getvalue()


PCT = {"p50": 10.04, "p90": 20.0, "p95": 30.0, "p99": 40.06}


def tail(out, tag):
    return [line for line in out.splitlines() if line.startswith(tag)]


class LineFormats(unittest.TestCase):
    """Inputs and expected text taken from trace_serve.py's print statements."""

    def test_every_line(self):
        summary = {
            "workload": {"expected": "aa", "replayed": "aa", "match": True, "n": 4, "of": 4},
            "serve": {"n": 4, "ttft_ms": PCT, "tpot_ms": PCT, "req_s": 2.5, "tok_s": 99.95},
            "sla": {"verdict": "FAIL", "bound": ["ttft_p99"], "ttft_p99": 40.06, "tpot_p99": 40.06,
                    "rate_req_s": 2.5, "sla_ttft_ms": 30.0, "sla_tpot_ms": None},
            "burst": {"peak_ttft_p99": 40.06, "recovery_s": None, "target_ms": 30.0},
            "spec": {"spec": "measured", "accept_len": 2.0, "accept_rate": 0.5, "n": 4},
        }
        self.assertEqual(legacy_lines(summary), [
            "WORKLOADSUM expected=aa replayed=aa match=1 n=4/4",
            ("SERVESUM n=4 ttft_ms_p50=10.0 ttft_ms_p90=20.0 ttft_ms_p95=30.0 ttft_ms_p99=40.1 "
             "tpot_ms_p50=10.040 tpot_ms_p90=20.000 tpot_ms_p95=30.000 tpot_ms_p99=40.060 "
             "req_s=2.500 tok_s=100.0"),
            ("SLASUM verdict=FAIL bound=ttft_p99 ttft_p99=40.1/30.0 tpot_p99=40.060/- "
             "rate_req_s=2.500"),
            "BURSTSUM peak_ttft_p99=40.1 recovery_s=na target_ms=30.0",
            "SPECSUM spec=measured accept_len=2.00 accept_rate=0.500 n=4",
        ])

    def test_off_and_void_and_goodput(self):
        self.assertEqual(legacy_lines({"serve": {"n": 1, "ttft_ms": PCT, "tpot_ms": None,
                                                 "req_s": 1.0, "tok_s": 1.0},
                                       "spec": {"spec": "off"}}), [
            ("SERVESUM n=1 ttft_ms_p50=10.0 ttft_ms_p90=20.0 ttft_ms_p95=30.0 ttft_ms_p99=40.1 "
             "tpot_ms_p50=- req_s=1.000 tok_s=1.0"),
            "SPECSUM spec=off note=needs-real-text-trace"])
        bad = {"expected": "aa", "replayed": "bb", "match": False, "n": 3, "of": 4}
        self.assertEqual(legacy_lines({"workload": bad})[1],
                         "WORKLOADVOID replayed workload differs from the trace; comparison is void")
        g = {"max_req_s": None, "speed": None, "fail_bound": [], "iterations": 1}
        self.assertEqual(legacy_lines({"workload": bad, "goodput": g}), [
            "WORKLOADSUM expected=aa replayed=bb match=0 n=3/4",
            "WORKLOADVOID replayed workload differs from the trace; sweep is void"])
        self.assertEqual(legacy_lines({"goodput": {**g, "fail_bound": ["ttft_p99"]}}), [
            ("GOODPUTSUM max_req_s=none bound=ttft_p99 iterations=1 "
             "note=no-passing-speed-within-budget")])
        self.assertEqual(legacy_lines({"goodput": {"max_req_s": 3.21, "speed": 2.0,
                                                   "fail_bound": [], "iterations": 6}}), [
            "GOODPUTSUM max_req_s=3.210 speed=2 bound=none-within-range iterations=6"])


class Sections(unittest.TestCase):
    def test_specsum_scores_the_reference_reply(self):
        trace = RealTrace([Request([1], 1, order=i, output_length=4, output_token_ids=[5, 6, 7, 8])
                           for i in range(3)], "t", "jsonl")
        summary, out = run(trace, TokenDriver(), ModeCfg(warmup=0, concurrency=3))
        self.assertEqual({r.accepted_tokens for r in summary.rows}, {2})
        self.assertEqual(tail(out, "SPECSUM"),
                         ["SPECSUM spec=measured accept_len=2.00 accept_rate=0.500 n=3"])

    def test_no_tokens_from_the_driver_means_spec_off(self):
        summary, out = run(single_trace(3), FakeDriver(), ModeCfg(warmup=0, concurrency=3))
        self.assertEqual(summary.sections["spec"], {"spec": "off"})
        self.assertEqual(tail(out, "SPECSUM"), ["SPECSUM spec=off note=needs-real-text-trace"])

    def test_sla_and_burst_follow_the_mode(self):
        summary, out = run(single_trace(4), FakeDriver(),
                           ModeCfg(warmup=0, concurrency=4, sla_ttft_ms=0.5, sla_tpot_ms=100))
        self.assertEqual(summary.sections["sla"]["bound"], ["ttft_p99"])
        self.assertEqual(summary.sections["serve"]["ttft_ms"]["p99"], 1.0)
        self.assertTrue(tail(out, "SLASUM verdict=FAIL bound=ttft_p99 ttft_p99=1.0/0.5"))
        self.assertTrue(tail(out, "BURSTSUM peak_ttft_p99=1.0 recovery_s=na target_ms=0.5"))
        summary, out = run(single_trace(4), FakeDriver(), ModeCfg(warmup=0, concurrency=4))
        self.assertNotIn("sla", summary.sections)
        self.assertFalse(tail(out, "SLASUM") or tail(out, "BURSTSUM"))


class GoodputSweep(unittest.TestCase):
    """measure() is replaced by one whose TTFT grows with speed, so the SLA
    flips at a known speed: 100 ms per unit of speed against a 450 ms bound."""

    def fake_measure(self, speeds):
        async def measure(trace, driver, mode, emit, workload, request_id):
            speeds.append(workload.speed)
            pairs = []
            for i, q in enumerate(trace.requests()):
                r = RequestResult(success=True, ttft=0.1 * workload.speed, e2el=1.0,
                                  output_tokens=3, input_tokens=1, usage_reported=True,
                                  scheduled_at_s=i * 0.1, semaphore_acquired_at_s=i * 0.1)
                emit(q, r)
                pairs.append((q, r))
            return [r for _, r in pairs], None, pairs, 1.0 / workload.speed
        return measure

    def test_bisects_to_the_boundary_within_six_probes(self):
        speeds = []
        mode = ModeCfg(load="open", warmup=0, sla_ttft_ms=450.0, goodput_sweep=True)
        with mock.patch.object(serve, "measure", self.fake_measure(speeds)):
            summary, out = run(single_trace(4), FakeDriver(), mode,
                               WorkloadCfg(arrival="poisson:10"))
        g = summary.sections["goodput"]
        self.assertLessEqual(len(speeds), 6)
        self.assertEqual(len(tail(out, "GOODPROBE")), len(speeds))
        self.assertTrue(4.0 <= g["speed"] < 4.5, speeds)
        self.assertEqual(g["fail_bound"], ["ttft_p99"])
        self.assertEqual(tail(out, "GOODPUTSUM"), [
            (f"GOODPUTSUM max_req_s={g['max_req_s']:.3f} speed={g['speed']:g} bound=ttft_p99 "
             f"iterations={len(speeds)}")])
        self.assertFalse(tail(out, "SERVESUM"))

    def test_no_passing_speed_exits_1(self):
        mode = ModeCfg(load="open", warmup=0, sla_ttft_ms=1.0, goodput_sweep=True)
        with mock.patch.object(serve, "measure", self.fake_measure([])):
            code, out = run(single_trace(4), FakeDriver(), mode, WorkloadCfg(arrival="poisson:10"))
        self.assertEqual(code, 1)
        self.assertIn("note=no-passing-speed-within-budget", out)

    def test_needs_open_load_and_a_bound(self):
        for mode in (ModeCfg(goodput_sweep=True, sla_ttft_ms=1.0),
                     ModeCfg(goodput_sweep=True, load="open")):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                serve.run(single_trace(2), FakeDriver(), mode, RunWriter(None, config(mode)))


if __name__ == "__main__":
    unittest.main()
