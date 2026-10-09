"""serve.run: one procedure over a Trace and a Driver, with no server."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path

from quettabench.config.schema import (
    Config,
    EngineCfg,
    HardwareCfg,
    ModeCfg,
    WorkloadCfg,
)
from quettabench.metrics.core import RequestResult
from quettabench.metrics.results import RunWriter
from quettabench.modes import serve
from quettabench.workloads.base import Request
from quettabench.workloads.live import RealTrace
from quettabench.workloads.sessions import SessionTrace, TurnPacing


class FakeDriver:
    """Answers every request at once; fail(request) picks the failures."""

    model, version = "m", None

    def __init__(self, fail=lambda r: False):
        self.fail, self.sent, self.resets = fail, [], 0

    @asynccontextmanager
    async def session(self, concurrency, open_loop=False):
        yield None

    async def send(self, request, request_id=None, capture_text=False, **sampling):
        self.sent.append((copy.deepcopy(request.messages), request_id, sampling))
        if self.fail(request):
            return RequestResult(success=False, e2el=0.01, error="boom", error_kind="http_error")
        n = request.max_tokens or 4
        return RequestResult(success=True, ttft=0.001, itl=[0.001] * (n - 1), e2el=0.001 * n,
                             input_tokens=request.prompt_len or 8, output_tokens=n,
                             usage_reported=True,
                             generated_text=f"reply{request.turn}" if capture_text else None)

    async def reset(self):
        self.resets += 1
        return "reset"


def config(mode):
    return Config(engine=EngineCfg(), workload=WorkloadCfg(), mode=mode, hardware=HardwareCfg())


def single_trace(n=6):
    return RealTrace([Request([i + 1] * 4, 4, order=i, output_length=3) for i in range(n)],
                     "t.jsonl", "jsonl")


def session_trace(sessions=2, turns=3):
    """Turn t's messages hold every earlier turn's assistant placeholder."""
    reqs, replies = [], {}
    for s in range(sessions):
        placeholders = [{"role": "assistant", "content": ""} for _ in range(turns)]
        replies[s] = placeholders
        history = []
        for t in range(turns):
            history = history + [{"role": "user", "content": f"s{s}t{t}"}]
            reqs.append(Request([], 0, order=len(reqs), output_length=2, session_id=s, turn=t,
                                messages=list(history)))
            history = history + [placeholders[t]]
    trace = SessionTrace(reqs, "s.jsonl", "jsonl")
    trace.replies, trace.pacing = replies, TurnPacing("per-session")
    return trace


def run(trace, driver, mode, workload=serve.DEFAULT_WORKLOAD, results_dir=None):
    writer = RunWriter(results_dir, config(mode))
    with contextlib.redirect_stdout(io.StringIO()) as out:
        try:
            summary = serve.run(trace, driver, mode, writer, workload)
        except SystemExit as e:
            return e.code, out.getvalue()
    return summary, out.getvalue()


class SingleTurn(unittest.TestCase):
    def test_closed_loop_rows_and_run_json(self):
        with tempfile.TemporaryDirectory() as td:
            summary, out = run(single_trace(), FakeDriver(),
                               ModeCfg(concurrency=2, warmup=2, reset_prefix_cache=True,
                                       discard_first=1), results_dir=td)
            rows = [json.loads(x) for x in (Path(td) / "requests.jsonl").read_text().splitlines()]
            run_json = json.loads((Path(td) / "run.json").read_text())
        self.assertEqual(len(rows), 6)
        self.assertEqual({r["ttft_basis"] for r in rows}, {"slot"})
        self.assertEqual(sorted(r["request_index"] for r in rows), list(range(6)))
        self.assertEqual(summary.core.excluded_requests, 1)
        self.assertEqual(run_json["run_status"], {"status": "ok", "reason": None})
        self.assertTrue(run_json["summary"]["workload"]["match"])
        self.assertIn("Warmup done: 2/2 succeeded.", out)
        self.assertIn("Prefix cache: reset", out)

    def test_open_loop_measures_from_arrival_and_stamps_the_schedule(self):
        trace = single_trace(4)
        before = trace.workload_hash()
        summary, _ = run(trace, FakeDriver(), ModeCfg(load="open", warmup=0),
                         WorkloadCfg(arrival="poisson:500", seed=3))
        self.assertEqual({r.ttft_basis for r in summary.rows}, {"arrival"})
        self.assertTrue(all(r.arrival_ts is not None for r in trace.requests()))
        self.assertNotEqual(trace.workload_hash(), before)
        self.assertEqual(summary.core.target_rate, 500.0)

    def test_failures_mark_the_run_partial(self):
        summary, _ = run(single_trace(8), FakeDriver(fail=lambda r: r.order == 0),
                         ModeCfg(concurrency=2, warmup=0, min_success_rate=0.5))
        self.assertEqual((summary.status, summary.reason), ("partial", "1 of 8 requests failed"))

    def test_mostly_failed_run_aborts(self):
        with tempfile.TemporaryDirectory() as td:
            code, out = run(single_trace(), FakeDriver(fail=lambda r: True),
                            ModeCfg(warmup=0), results_dir=td)
            status = json.loads((Path(td) / "run.json").read_text())["run_status"]
        self.assertEqual(code, 1)
        self.assertIn("ABORT: 6/6 requests failed", out)
        self.assertEqual(status["status"], "aborted")

    def test_request_ids_name_profile_and_index(self):
        driver = FakeDriver()
        run(single_trace(2), driver, ModeCfg(warmup=0, trace_request_ids=True, concurrency=4))
        ids = sorted(rid for _, rid, _ in driver.sent)
        self.assertEqual(ids[0], "agenticbench__p=t.jsonl__c=4__t=0__s=0__i=0")


class Sessions(unittest.TestCase):
    def test_replies_feed_later_turns(self):
        driver = FakeDriver()
        summary, _ = run(session_trace(), driver, ModeCfg(concurrency=2, warmup=0))
        last_turn = next(m for m, _, _ in driver.sent if m[-1]["content"] == "s0t2")
        self.assertEqual([m["content"] for m in last_turn if m["role"] == "assistant"],
                         ["reply0", "reply1"])
        self.assertEqual([t.turn_index for t in summary.turns], [0, 1, 2])
        self.assertEqual({(r.session_id, r.turn_index) for r in summary.rows},
                         {(s, t) for s in range(2) for t in range(3)})

    def test_a_dropped_turn_fails_the_identity_gate(self):
        with tempfile.TemporaryDirectory() as td:
            code, out = run(session_trace(), FakeDriver(fail=lambda r: (r.session_id, r.turn) == (1, 1)),
                            ModeCfg(warmup=0, min_success_rate=0.5), results_dir=td)
            run_json = json.loads((Path(td) / "run.json").read_text())
            rows = (Path(td) / "requests.jsonl").read_text().splitlines()
        self.assertEqual(code, 3)
        self.assertIn("WORKLOADVOID", out)
        self.assertEqual(run_json["run_status"], {"status": "failed", "reason": "workload identity"})
        self.assertEqual(run_json["summary"]["workload"]["n"], 5)
        self.assertEqual(len(rows), 5)

    def test_interleaved_barrier_skips_dead_sessions(self):
        trace = session_trace()
        trace.pacing = TurnPacing("interleaved")
        code, out = run(trace, FakeDriver(fail=lambda r: (r.session_id, r.turn) == (0, 0)),
                        ModeCfg(warmup=0, min_success_rate=0.0))
        self.assertEqual(code, 3)
        self.assertIn("Turn 2/3: dispatching 1 requests...", out)


if __name__ == "__main__":
    unittest.main()
