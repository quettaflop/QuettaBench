"""serve.run over a Trace and a Driver with no server."""

import asyncio
import collections
import contextlib
import copy
import gc
import http.client
import io
import json
import os
import tempfile
import time
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest import mock

from quettabench import runner
from quettabench.config import loader
from quettabench.config.schema import Config, ConfigError, EngineCfg, HardwareCfg, ModeCfg, WorkloadCfg
from quettabench.engines import launcher
from quettabench.engines.drivers import HttpDriver
from quettabench.metrics.results import RunWriter, summary_tag_lines
from quettabench.metrics.summary import RequestResult
from quettabench.modes import serve
from quettabench.workloads.base import Request
from quettabench.workloads.live import RealTrace
from quettabench.workloads.plan import freeze, sent_digest
from quettabench.workloads.sessions import SessionTrace, TurnPacing
from tests.contracts import helpers
from tests.golden._harness import DENSE_CONFIG, RUNS, grid_argv, run_grid_cli


class FakeDriver:
    model, version = "m", None

    def __init__(self, fail=lambda r: False, short=None):
        self.fail, self.sent, self.resets = fail, [], 0
        self.short = short or {}  # (session, turn) -> output tokens to return instead of the planned length

    @asynccontextmanager
    async def session(self, concurrency, open_loop=False):
        yield None

    async def send(self, request, request_id=None, capture_text=False, **sampling):
        self.sent.append((copy.deepcopy(request.messages), request_id, sampling))
        if self.fail(request):
            return RequestResult(success=False, e2el=0.01, error="boom", error_kind="http_error")
        n = self.short.get((request.session_id, request.turn), request.max_tokens or 4)
        return RequestResult(
            sent_perf_s=time.perf_counter(),
            success=True,
            ttft=0.001,
            itl=[0.001] * (n - 1),
            e2el=0.001 * n,
            input_tokens=request.prompt_len or 8,
            output_tokens=n,
            usage_reported=True,
            generated_text=f"reply{request.turn}" if capture_text else None,
        )

    async def reset(self):
        self.resets += 1
        return "reset"


def config(mode):
    return Config(engine=EngineCfg(), workload=WorkloadCfg(), mode=mode, hardware=HardwareCfg())


def single_trace(n=6):
    return RealTrace([Request([i + 1] * 4, 4, order=i, output_length=3) for i in range(n)], "t.jsonl", "jsonl")


def session_trace(sessions=2, turns=3):
    reqs, replies = [], {}
    for s in range(sessions):
        placeholders = [{"role": "assistant", "content": ""} for _ in range(turns)]
        replies[s] = placeholders
        history = []
        for t in range(turns):
            history = history + [{"role": "user", "content": f"s{s}t{t}"}]
            reqs.append(Request([], 0, order=len(reqs), output_length=2, session_id=s, turn=t, messages=list(history)))
            history = history + [placeholders[t]]
    trace = SessionTrace(reqs, "s.jsonl", "jsonl")
    trace.replies, trace.pacing = replies, TurnPacing("per-session")
    return trace


def recorded_trace(sessions=5, turns=2):
    """Sessions whose assistant turns are recorded, as a trajectory file's are: nothing is filled in live."""
    reqs = []
    for s in range(sessions):
        history = []
        for t in range(turns):
            history = history + [{"role": "user", "content": f"s{s}t{t}"}]
            reqs.append(Request([], 0, order=len(reqs), output_length=2, session_id=s, turn=t, messages=list(history)))
            history = history + [{"role": "assistant", "content": f"recorded {s} {t}"}]
    trace = SessionTrace(reqs, "r.jsonl", "jsonl")
    trace.pacing = TurnPacing("per-session")
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
            summary, out = run(
                single_trace(),
                FakeDriver(),
                ModeCfg(concurrency=2, warmup=2, reset_prefix_cache=True, discard_first=1),
                results_dir=td,
            )
            rows = [json.loads(x) for x in (Path(td) / "requests.jsonl").read_text().splitlines()]
            run_json = json.loads((Path(td) / "run.json").read_text())
        self.assertEqual(len(rows), 6)
        # The driver's send instant, on the run clock: after the slot was granted, before the reply was done.
        self.assertTrue(all(r["semaphore_acquired_at_ms"] <= r["sent_at_ms"] <= r["completed_at_ms"] for r in rows))
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
        summary, _ = run(
            trace, FakeDriver(), ModeCfg(load="open", warmup=0), WorkloadCfg(arrival="poisson:500", seed=3)
        )
        self.assertEqual({r.ttft_basis for r in summary.rows}, {"arrival"})
        self.assertTrue(all(r.arrival_ts is not None for r in trace.requests()))
        self.assertNotEqual(trace.workload_hash(), before)
        self.assertEqual(summary.core.target_rate, 500.0)

    def test_a_failed_request_marks_the_run_partial(self):
        with tempfile.TemporaryDirectory() as td:
            summary, out = run(
                single_trace(8),
                FakeDriver(fail=lambda r: r.order == 0),
                ModeCfg(concurrency=2, warmup=0, min_success_rate=0.5),
                results_dir=td,
            )
            run_json = json.loads((Path(td) / "run.json").read_text())
        self.assertEqual((summary.status, summary.reason), ("partial", "1 of 8 requests failed"))
        self.assertEqual(run_json["run_status"], {"status": "partial", "reason": "1 of 8 requests failed"})
        workload = run_json["summary"]["workload"]
        self.assertEqual((workload["match"], workload["n"], workload["of"]), (True, 8, 8))
        self.assertNotIn("WORKLOAD_MISMATCH", out)

    def test_a_reply_with_no_chunk_is_a_failed_request_not_one_timed_by_its_e2el(self):
        class Empty(FakeDriver):
            async def send(self, request, **kw):
                result = await super().send(request, **kw)
                if request.order == 3:
                    result.ttft, result.itl = None, []
                return result

        with tempfile.TemporaryDirectory() as td:
            summary, _ = run(single_trace(6), Empty(), ModeCfg(concurrency=2, warmup=0), results_dir=td)
            rows = [json.loads(x) for x in (Path(td) / "requests.jsonl").read_text().splitlines()]
        bad = [r for r in rows if not r["success"]]
        self.assertEqual(
            [(r["request_index"], r["error_kind"], r["error"]) for r in bad],
            [(3, "empty_reply", "the reply carried no chunk with choices")],
        )
        self.assertEqual(summary.sections["serve"]["n"], 5)
        self.assertEqual((summary.status, summary.core.failed_requests), ("partial", 1))

    def test_a_run_with_a_failed_request_withholds_its_latency_percentiles(self):
        with tempfile.TemporaryDirectory() as td:
            summary, out = run(
                single_trace(8),
                FakeDriver(fail=lambda r: r.order == 0),
                ModeCfg(concurrency=2, warmup=0, min_success_rate=0.5),
                results_dir=td,
            )
            doc = json.loads((Path(td) / "run.json").read_text())
        s = doc["summary"]
        self.assertEqual(doc["run_status"]["status"], "partial")
        self.assertEqual([s[k] for k in ("mean_ttft_ms", "p99_ttft_ms", "p99_tpot_ms", "p99_e2el_ms")], [None] * 4)
        self.assertEqual((s["serve"]["ttft_ms"], s["serve"]["tpot_ms"]), (None, None))
        self.assertIn("1 requests failed", s["withheld"]["latency_percentiles"])
        self.assertGreater(s["output_token_throughput"], 0)  # throughput and counts stay
        self.assertIn("Latency:                  withheld", out)
        self.assertIn("SERVE_SUMMARY n=7 ttft_ms=withheld tpot_ms=withheld req_s=", out)
        # With no failure the percentiles are there.
        clean, _ = run(single_trace(8), FakeDriver(), ModeCfg(concurrency=2, warmup=0))
        self.assertIsNotNone(clean.core.p99_ttft_ms)

    def test_failures_under_min_success_rate_abort(self):
        with tempfile.TemporaryDirectory() as td:
            code, _ = run(
                single_trace(8),
                FakeDriver(fail=lambda r: r.order < 2),
                ModeCfg(concurrency=2, warmup=0, min_success_rate=0.9),
                results_dir=td,
            )
            status = json.loads((Path(td) / "run.json").read_text())["run_status"]
        self.assertEqual((code, status["status"]), (1, "aborted"))

    def test_a_request_changed_after_the_plan_was_frozen_fails_the_identity_gate(self):
        # Sending request 0 changes request 2's prompt in place; the old gate hashed the same objects on both
        # sides after the run and passed.
        trace = single_trace(4)

        class Tamper(FakeDriver):
            async def send(self, request, **kw):
                if request.order == 0:
                    trace.requests()[2].prompt_token_ids[0] = 999
                return await super().send(request, **kw)

        code, out = run(trace, Tamper(), ModeCfg(concurrency=1, warmup=0))
        self.assertEqual(code, 3)
        self.assertIn("first_mismatch=2 (differs from the plan)", out)
        self.assertIn("WORKLOAD_MISMATCH replayed workload differs from the trace; comparison is void", out)

    def test_mostly_failed_run_aborts(self):
        with tempfile.TemporaryDirectory() as td:
            code, out = run(single_trace(), FakeDriver(fail=lambda r: True), ModeCfg(warmup=0), results_dir=td)
            status = json.loads((Path(td) / "run.json").read_text())["run_status"]
        self.assertEqual(code, 1)
        self.assertIn("ABORT: 6/6 requests failed", out)
        self.assertEqual(status["status"], "aborted")

    def test_request_ids_name_profile_and_index(self):
        driver = FakeDriver()
        run(single_trace(2), driver, ModeCfg(warmup=0, trace_request_ids=True, concurrency=4))
        ids = sorted(rid for _, rid, _ in driver.sent)
        self.assertEqual(ids[0], "agenticbench__p=t.jsonl__c=4__t=0__s=0__i=0")


class TimedDriver(FakeDriver):
    """Every request takes 5 ms; counts the sessions and requests open at once."""

    def __init__(self, last_turn=2):
        super().__init__()
        self.last_turn = last_turn
        self.open_sessions: set = set()
        self.inflight = 0
        self.peak = {"sessions": 0, "requests": 0}

    async def send(self, request, **kw):
        sent = time.perf_counter()
        self.open_sessions.add(request.session_id)
        self.inflight += 1
        self.peak["sessions"] = max(self.peak["sessions"], len(self.open_sessions))
        self.peak["requests"] = max(self.peak["requests"], self.inflight)
        await asyncio.sleep(0.005)
        self.inflight -= 1
        if request.turn == self.last_turn:
            self.open_sessions.discard(request.session_id)
        result = await super().send(request, **kw)
        result.sent_perf_s = sent
        return result


class Sessions(unittest.TestCase):
    def test_an_agent_holds_its_slot_from_first_turn_to_last(self):
        driver = TimedDriver()
        summary, out = run(session_trace(12, 3), driver, ModeCfg(concurrency=3, warmup=0))
        # 12 sessions through 3 agents: never more than 3 sessions open, and a turn never queues for a slot.
        self.assertEqual(driver.peak, {"sessions": 3, "requests": 3})
        self.assertLess(max(r.client_queue_wait_s for r in summary.rows), 0.001)
        self.assertEqual(len(summary.rows), 36)
        self.assertIn("3 agents at once", out)

    def test_a_session_run_has_a_steady_ttft_only_in_its_steady_window(self):
        def steady_ttft_keys(node, path=""):
            for k, v in node.items() if isinstance(node, dict) else ():
                p = f"{path}.{k}".lstrip(".")
                if "steady" in k and "ttft" in k and v is not None:
                    yield p
                yield from steady_ttft_keys(v, p)

        with tempfile.TemporaryDirectory() as td:
            summary, out = run(session_trace(12, 3), TimedDriver(), ModeCfg(concurrency=3, warmup=0), results_dir=td)
            run_json = json.loads((Path(td) / "run.json").read_text())
        self.assertEqual(list(steady_ttft_keys(run_json)), [])
        self.assertNotIn("steady_ttft_ms", out)
        self.assertTrue(run_json["summary"]["steady"]["valid"])
        self.assertIsNotNone(run_json["summary"]["steady"]["ttft_ms"])
        # The cold start is in run.json and on the console: three agents' first turns, then the ramp.
        self.assertEqual(run_json["summary"]["steady"]["first_turn_ttft_ms"]["n"], 3)
        self.assertGreaterEqual(run_json["summary"]["steady"]["ramp"]["ttft_ms"]["n"], 9)
        self.assertIn("Cold start TTFT: first turns p50 1.0 ms, p99 1.0 ms (n=3); before the window", out)
        # A single-turn closed loop keeps the requests past the first C as its steady TTFT.
        single, out = run(single_trace(6), FakeDriver(), ModeCfg(concurrency=2, warmup=0))
        self.assertIsNotNone(single.sections["serve"]["steady_ttft_ms"])
        self.assertIn("steady_ttft_ms_p50=", out)

    def test_agents_keep_their_slots_through_recorded_waits_and_the_run_has_a_steady_window(self):
        trace = session_trace(60, 5)
        for r in trace.requests():
            r.metadata = {"tool_wait_after_ms": 3.0}
        trace.pacing = TurnPacing("per-session", use_recorded_waits=True)
        driver = TimedDriver(last_turn=4)
        # A full-generation gc pass in a process holding the whole suite can stall the loop past the 50 ms handover
        # bound and void the window; this test is about the dispatch, so gc waits until it is done.
        gc.disable()
        try:
            summary, _ = run(trace, driver, ModeCfg(concurrency=8, warmup=0))
        finally:
            gc.enable()
        # A session's waits count as open: still 8 sessions at most, 8 requests at most, no queue for a slot.
        self.assertEqual(driver.peak, {"sessions": 8, "requests": 8})
        # Waiting for a slot would cost a whole 5 ms request; below that is event-loop jitter under 8 coroutines.
        self.assertLess(max(r.client_queue_wait_s for r in summary.rows), 0.005)
        self.assertEqual(len(summary.rows), 300)
        steady = summary.sections["steady"]
        self.assertTrue(steady["valid"], steady)
        self.assertEqual((steady["sessions"], steady["agents"]), (60, 8))
        # 8 agents, each in flight 5 ms of every 8 ms (3 ms tool wait): about 5 requests in flight on average.
        self.assertAlmostEqual(steady["timeline"]["requests_in_flight"]["mean"], 5.0, delta=1.0)
        self.assertLess(steady["timeline"]["requests_in_flight"]["max"], 8.0)
        last_finish = max(r.completed_at_s for r in summary.rows)
        self.assertTrue(0.0 <= steady["start_s"] < steady["end_s"] <= last_finish)

    def test_client_health_times_a_stalled_loop_and_each_gc_pass(self):
        async def go():
            health = serve.ClientHealth()
            health.start()
            await asyncio.sleep(0.2)
            time.sleep(0.06)  # the loop cannot run while this blocks
            gc.collect()
            await asyncio.sleep(0.05)
            await health.stop()
            return health.section()

        c = asyncio.run(go())
        # The stall is about a fifth of the wall time: past p99, short of p50.
        self.assertGreaterEqual(c["loop_lag_ms"]["max"], 50.0)
        self.assertGreaterEqual(c["loop_lag_ms"]["99"], 50.0)
        self.assertLess(c["loop_lag_ms"]["50"], 50.0)
        self.assertGreaterEqual((c["gc_passes"], c["gc_gen2_passes"]), (1, 1))
        self.assertIn("event loop late by", serve.client_line(c))

    def test_loop_lag_percentiles_weigh_each_sample_by_the_wall_time_it_spans(self):
        # 199 on-time samples span 1.99 s and one 1 s stall spans 1.01 s: the stall is a third of the time, so
        # p99 is the stall although it is one sample in 200.
        samples = [(0.0, 0.0)] * 199 + [(0.0, 1.0)]
        self.assertEqual(serve.loop_lag_ms(samples), {"50": 0.0, "99": 1000.0, "max": 1000.0})
        self.assertIsNone(serve.loop_lag_ms([]))

    def test_steady_ttft_is_withheld_when_the_loop_lags_inside_the_window(self):
        def steady():
            return {"valid": True, "start_s": 1.0, "end_s": 2.0, "ttft_ms": {50: 10.0, 90: 10.0, 99: 10.0}}

        on_time = [(1.0 + i / 100, 0.0) for i in range(100)]
        # A 0.5 s stall before the window does not count against it.
        calm = steady()
        serve.withhold_on_client_lag(calm, [(0.5, 0.5)] + on_time)
        self.assertEqual((calm["client_loop_lag_ms"]["99"], calm["client_loop_lag_samples"]), (0.0, 100))
        self.assertIsNotNone(calm["ttft_ms"])
        self.assertNotIn("withheld", calm)
        # 20 ms late for 3% of the window's time: over the 5 ms bar at p99.
        late = steady()
        serve.withhold_on_client_lag(late, on_time + [(1.5, 0.02)])
        self.assertIsNone(late["ttft_ms"])
        self.assertEqual(
            late["withheld"], {"ttft_ms": "client event loop late by 20.0 ms at p99 in the window, over 5 ms"}
        )

    def test_a_session_run_whose_client_blocks_withholds_its_steady_ttft(self):
        class BlockingDriver(TimedDriver):
            async def send(self, request, **kw):
                time.sleep(0.01)  # holds the event loop, as a slow client would
                return await super().send(request, **kw)

        summary, out = run(session_trace(12, 3), BlockingDriver(), ModeCfg(concurrency=3, warmup=0))
        steady = summary.sections["steady"]
        self.assertTrue(steady["valid"], steady)
        self.assertGreater(steady["client_loop_lag_ms"]["99"], 5.0)
        self.assertIsNone(steady["ttft_ms"])
        self.assertIn("ttft_ms withheld: client event loop late by", out)

    def test_a_recycled_replay_keeps_every_agent_busy_to_the_deadline_with_a_salt_per_play(self):
        driver = TimedDriver(last_turn=1)
        # An 80 ms run: the 300 s floor on a recycled window would void it.
        with mock.patch.object(serve, "MIN_RECYCLED_WINDOW_S", 0.0):
            summary, out = run(recorded_trace(5, 2), driver, ModeCfg(concurrency=4, warmup=0, duration_s=0.08))
        plays = summary.sections["plays"]
        self.assertGreater(plays["plays"], 5)  # 5 sessions, more plays: the sample was recycled
        self.assertEqual((plays["sessions"], plays["duration_s"]), (5, 0.08))
        self.assertEqual(driver.peak, {"sessions": 4, "requests": 4})
        by_play = collections.Counter(r.session_id for r in summary.rows)
        self.assertEqual(set(by_play.values()), {2})  # every play ran to its end, the ones past the deadline too
        self.assertTrue(any(str(sid).endswith("#1") for sid in by_play))
        self.assertTrue(all(r.success for r in summary.rows))
        salts = {m[0]["content"][:18] for m, _, _ in driver.sent if len(m) == 1}
        self.assertEqual(len(salts), plays["plays"])
        self.assertTrue(all(s.startswith("[rid:") for s in salts))
        self.assertTrue(summary.sections["workload"]["match"])
        self.assertNotIn("WORKLOAD_MISMATCH", out)
        steady = summary.sections["steady"]
        self.assertTrue(steady["valid"], steady)
        self.assertEqual(steady["end_s"], 0.08)
        self.assertIn("Recycled replay: 5 sessions in a cycle, 4 agents for 0.08 s", out)
        self.assertIsNone(plays["plays_by_stratum"])

    def test_a_recycled_plan_hash_is_the_sample_seed_and_play_rule_not_the_plays_started(self):
        def recycled(duration_s, seed=0):
            trace = recorded_trace(5, 2)
            mode = ModeCfg(concurrency=4, warmup=0, duration_s=duration_s)
            summary, _ = run(trace, TimedDriver(last_turn=1), mode, WorkloadCfg(seed=seed))
            return summary.sections["workload"], summary.sections["plays"]["plays"], freeze(trace).plan_hash

        short, long_, other_seed = recycled(0.03), recycled(0.06), recycled(0.03, seed=1)
        self.assertNotEqual(short[1], long_[1])  # the deadline let a different number of plays start
        self.assertNotEqual(short[0]["expected"], long_[0]["expected"])
        self.assertEqual(short[0]["plan_hash"], long_[0]["plan_hash"])
        self.assertNotEqual(short[0]["plan_hash"], other_seed[0]["plan_hash"])
        # Salted plays are not the plain sample: a recycled run does not compare with a run of the sample as is.
        self.assertNotEqual(short[0]["plan_hash"], short[2])
        self.assertTrue(short[0]["match"] and long_[0]["match"])

    def test_a_cache_salt_replay_sends_the_recorded_prompt_with_one_salt_per_play(self):
        class Salts(TimedDriver):
            async def send(self, request, **kw):
                self.seen.append((request.session_id, request.turn, request.cache_salt, json.dumps(request.messages)))
                return await super().send(request, **kw)

        trace, driver = recorded_trace(5, 2), Salts(last_turn=1)
        driver.seen = []
        source = {(r.session_id, r.turn): json.dumps(r.messages) for r in trace.requests()}
        mode = ModeCfg(concurrency=4, warmup=0, duration_s=0.05)
        summary, _ = run(trace, driver, mode, WorkloadCfg(salt="cache_salt"))
        plays = summary.sections["plays"]
        self.assertEqual(plays["salt"], "cache_salt")
        self.assertIn("cache_salt field", plays["play_rule"])
        # Every play its own salt, the same on each of its turns.
        salts = {(sid, salt) for sid, _, salt, _ in driver.seen}
        self.assertEqual(len(salts), plays["plays"])
        self.assertEqual(len({salt for _, salt in salts}), plays["plays"])
        # The text sent is the recorded text, byte for byte.
        for sid, turn, _, messages in driver.seen:
            self.assertEqual(messages, source[int(sid.split("#")[0]), turn])
        self.assertTrue(summary.sections["workload"]["match"])
        text_plan = freeze(recorded_trace(5, 2), recycled=True, seed=42).plan_hash
        self.assertNotEqual(summary.sections["workload"]["plan_hash"], text_plan)

    def test_a_recycled_replay_counts_plays_per_stratum(self):
        trace = recorded_trace(5, 2)
        trace.stratum_of = {s: "long" if s % 2 else "short" for s in range(5)}
        summary, _ = run(trace, TimedDriver(last_turn=1), ModeCfg(concurrency=2, warmup=0, duration_s=0.05))
        by_stratum = summary.sections["plays"]["plays_by_stratum"]
        self.assertEqual(sum(by_stratum.values()), summary.sections["plays"]["plays"])
        self.assertEqual(set(by_stratum), {"long", "short"})
        # 50 ms is far short of a recycled window's 300 s floor: no steady numbers, and the reason says why.
        self.assertIn("a recycled window needs at least 300 s", summary.sections["steady"]["reason"])

    def test_a_recycled_replay_refuses_a_trace_that_fills_replies_live(self):
        trace = session_trace()
        trace.live_replies = True
        with self.assertRaisesRegex(ConfigError, "needs recorded replies"):
            run(trace, FakeDriver(), ModeCfg(concurrency=2, warmup=0, duration_s=0.05))

    def test_weights_revision_is_the_cached_snapshot_or_none(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(serve.weights_revision(td))
        with mock.patch.object(serve, "snapshot_download", return_value="/hf/hub/models--a--b/snapshots/dba1be0"):
            self.assertEqual(serve.weights_revision("a/b"), "dba1be0")
        with mock.patch.object(serve, "snapshot_download", side_effect=serve.LocalEntryNotFoundError("no")):
            self.assertIsNone(serve.weights_revision("a/b"))

    def test_run_json_carries_the_trace_and_its_sample(self):
        with tempfile.TemporaryDirectory() as td:
            run(session_trace(3, 2), FakeDriver(), ModeCfg(concurrency=2, warmup=0), results_dir=td)
            section = json.loads((Path(td) / "run.json").read_text())["summary"]["trace"]
        self.assertEqual((section["path"], section["sample"]), ("s.jsonl", []))

    def test_a_session_run_records_client_health(self):
        summary, out = run(session_trace(4, 2), FakeDriver(), ModeCfg(concurrency=2, warmup=0))
        self.assertGreater(summary.sections["client"]["loop_lag_samples"], -1)
        self.assertIn("Client: event loop late by", out)

    def test_a_short_reply_is_a_failed_turn_under_exact_output_length(self):
        # Session 1's second turn comes back one token long: the turn fails and the session stops there, so its
        # last turn is never sent.
        with tempfile.TemporaryDirectory() as td:
            summary, out = run(
                session_trace(),
                FakeDriver(short={(1, 1): 1}),
                ModeCfg(concurrency=2, warmup=0, exact_output_length=True, min_success_rate=0.0),
                results_dir=td,
            )
            rows = [json.loads(x) for x in (Path(td) / "requests.jsonl").read_text().splitlines()]
        bad = [r for r in rows if not r["success"]]
        self.assertEqual(
            [(r["session_id"], r["turn_index"], r["error"]) for r in bad], [(1, 1, "1 output tokens, 2 planned")]
        )
        self.assertEqual(summary.status, "partial")
        self.assertEqual(summary.sections["workload"]["turns_not_sent"], [{"session": 1, "turn": 2}])
        self.assertNotIn("WORKLOAD_MISMATCH", out)

    def test_replies_feed_later_turns(self):
        driver = FakeDriver()
        summary, _ = run(session_trace(), driver, ModeCfg(concurrency=2, warmup=0))
        last_turn = next(m for m, _, _ in driver.sent if m[-1]["content"] == "s0t2")
        self.assertEqual([m["content"] for m in last_turn if m["role"] == "assistant"], ["reply0", "reply1"])
        self.assertEqual([t.turn_index for t in summary.turns], [0, 1, 2])
        self.assertEqual(
            {(r.session_id, r.turn_index) for r in summary.rows}, {(s, t) for s in range(2) for t in range(3)}
        )

    @staticmethod
    def tamper_on_first_send(trace, change):
        """A driver that, while session 0's first turn is out, changes session 0's second turn in place."""

        class Tamper(FakeDriver):
            async def send(self, request, **kw):
                if (request.session_id, request.turn) == (0, 0):
                    change(next(r for r in trace.requests() if (r.session_id, r.turn) == (0, 1)))
                return await super().send(request, **kw)

        return Tamper()

    def test_a_user_message_changed_after_the_freeze_fails_the_identity_gate(self):
        trace = session_trace()
        driver = self.tamper_on_first_send(trace, lambda r: r.messages[-1].update(content="tampered"))
        code, out = run(trace, driver, ModeCfg(concurrency=1, warmup=0))
        self.assertEqual(code, 3)
        self.assertIn("first_mismatch=0:1 (differs from the plan)", out)

    def test_a_recorded_assistant_turn_changed_under_live_replies_fails_the_identity_gate(self):
        # Only the placeholders the run splices are left out of the digest; a recorded assistant turn is hashed.
        trace = recorded_trace(2, 3)
        trace.live_replies = True
        driver = self.tamper_on_first_send(trace, lambda r: r.messages[1].update(content="rewritten"))
        code, out = run(trace, driver, ModeCfg(concurrency=1, warmup=0))
        self.assertEqual(code, 3)
        self.assertIn("first_mismatch=0:1 (differs from the plan)", out)

    def test_an_engine_reply_spliced_into_a_placeholder_passes_the_identity_gate(self):
        driver = FakeDriver()
        summary, _ = run(session_trace(), driver, ModeCfg(concurrency=1, warmup=0))
        second = next(m for m, _, _ in driver.sent if m[-1]["content"] == "s0t1")
        self.assertEqual(second[1], {"role": "assistant", "content": "reply0"})  # the plan froze it empty
        w = summary.sections["workload"]
        self.assertEqual((w["match"], w["n"], w["of"]), (True, 6, 6))
        self.assertNotIn("first_mismatch", w)

    def test_a_failed_turn_ends_its_session_and_the_run_is_partial(self):
        # Two sessions of three turns; session 1's second turn fails, so its third is never sent.
        trace = session_trace()
        with tempfile.TemporaryDirectory() as td:
            summary, out = run(
                trace,
                FakeDriver(fail=lambda r: (r.session_id, r.turn) == (1, 1)),
                ModeCfg(warmup=0, min_success_rate=0.5),
                results_dir=td,
            )
            run_json = json.loads((Path(td) / "run.json").read_text())
            rows = (Path(td) / "requests.jsonl").read_text().splitlines()
        self.assertEqual(
            run_json["run_status"],
            {"status": "partial", "reason": "1 of 5 requests failed; 1 later turns of their sessions not sent"},
        )
        w = run_json["summary"]["workload"]
        self.assertEqual((w["match"], w["n"], w["of"]), (True, 5, 6))
        self.assertEqual(w["turns_not_sent"], [{"session": 1, "turn": 2}])
        # What runs compare on is the whole plan, failure or not: the same as a fresh trace's.
        self.assertEqual(w["plan_hash"], freeze(session_trace()).plan_hash)
        self.assertEqual(len(rows), 5)
        self.assertIn("match=1 n=5/6 not_sent=1", out)
        self.assertNotIn("WORKLOAD_MISMATCH", out)
        # A failed turn withholds every latency figure, the per-turn table's too.
        self.assertIsNone(run_json["summary"]["per_turn"][0]["median_ttft_ms"])
        self.assertIn("2 requests failed or went unsent", run_json["summary"]["withheld"]["latency_percentiles"])
        self.assertIn("  withheld    withheld    withheld    withheld", out)

    def test_turns_a_failed_turn_left_unsent_count_against_the_success_rate(self):
        # 4 of 5 sent turns succeed (80%), but 4 of the 6 planned (67%): under a 75% floor the run aborts.
        with tempfile.TemporaryDirectory() as td:
            code, out = run(
                session_trace(),
                FakeDriver(fail=lambda r: (r.session_id, r.turn) == (1, 1)),
                ModeCfg(warmup=0, min_success_rate=0.75),
                results_dir=td,
            )
        self.assertEqual(code, 1)
        self.assertIn("ABORT: Success rate 66.7% below minimum 75% (4/6); failures: http_error=1, not_sent=1", out)

    def test_a_failed_turn_ends_its_play_in_a_recycled_replay(self):
        driver = TimedDriver(last_turn=1)
        driver.fail = lambda r: (r.session_id, r.turn) == ("2#0", 0)
        summary, out = run(
            recorded_trace(5, 2), driver, ModeCfg(concurrency=4, warmup=0, duration_s=0.08, min_success_rate=0.5)
        )
        w = summary.sections["workload"]
        self.assertTrue(w["match"])
        self.assertEqual(w["turns_not_sent"], [{"session": "2#0", "turn": 1}])
        self.assertEqual((w["n"], w["of"]), (len(summary.rows), len(summary.rows) + 1))
        self.assertEqual(summary.status, "partial")
        self.assertNotIn("WORKLOAD_MISMATCH", out)

    def test_interleaved_barrier_skips_dead_sessions(self):
        trace = session_trace()
        trace.pacing = TurnPacing("interleaved")
        summary, out = run(
            trace, FakeDriver(fail=lambda r: (r.session_id, r.turn) == (0, 0)), ModeCfg(warmup=0, min_success_rate=0.0)
        )
        self.assertIn("Turn 2/3: dispatching 1 requests...", out)
        not_sent = summary.sections["workload"]["turns_not_sent"]
        self.assertEqual(not_sent, [{"session": 0, "turn": 1}, {"session": 0, "turn": 2}])
        self.assertEqual(summary.status, "partial")


class TokenDriver(FakeDriver):
    async def send(self, request, **kw):
        result = await super().send(request, **kw)
        ref = request.output_token_ids or []
        result.generated_token_ids = [t if i != 2 else -1 for i, t in enumerate(ref)]
        return result


def run_sections(trace, driver, mode, workload=serve.DEFAULT_WORKLOAD):
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
    def test_every_line(self):
        summary = {
            "workload": {"plan_hash": "pp", "expected": "aa", "executed_hash": "aa", "match": True, "n": 4, "of": 4},
            "serve": {"n": 4, "ttft_ms": PCT, "tpot_ms": PCT, "req_s": 2.5, "tok_s": 99.95},
            "sla": {
                "verdict": "FAIL",
                "bound": ["ttft_p99"],
                "ttft_p99": 40.06,
                "tpot_p99": 40.06,
                "rate_req_s": 2.5,
                "sla_ttft_ms": 30.0,
                "sla_tpot_ms": None,
            },
            "burst": {"peak_ttft_p99": 40.06, "recovery_s": None, "target_ms": 30.0},
            "spec_decode": {"spec": "measured", "accept_len": 2.0, "accept_rate": 0.5, "n": 4},
        }
        self.assertEqual(
            summary_tag_lines(summary),
            [
                "WORKLOAD_CHECK plan=pp expected=aa executed=aa match=1 n=4/4",
                (
                    "SERVE_SUMMARY n=4 ttft_ms_p50=10.0 ttft_ms_p90=20.0 ttft_ms_p95=30.0 ttft_ms_p99=40.1 "
                    "tpot_ms_p50=10.040 tpot_ms_p90=20.000 tpot_ms_p95=30.000 tpot_ms_p99=40.060 "
                    "req_s=2.500 tok_s=100.0"
                ),
                ("SLA_SUMMARY verdict=FAIL bound=ttft_p99 ttft_p99=40.1/30.0 tpot_p99=40.060/- rate_req_s=2.500"),
                "BURST_SUMMARY peak_ttft_p99=40.1 recovery_s=na target_ms=30.0",
                "SPEC_DECODE_SUMMARY spec=measured accept_len=2.00 accept_rate=0.500 n=4",
            ],
        )

    def test_sla_summary_without_tpot_samples_prints_a_dash(self):
        sla = {
            "verdict": "PASS",
            "bound": [],
            "ttft_p99": 5.0,
            "tpot_p99": None,
            "rate_req_s": 1.0,
            "sla_ttft_ms": 30.0,
            "sla_tpot_ms": 10.0,
        }
        self.assertEqual(
            summary_tag_lines({"sla": sla}),
            ["SLA_SUMMARY verdict=PASS bound=none ttft_p99=5.0/30.0 tpot_p99=-/10.0 rate_req_s=1.000"],
        )
        records = [{"ttft_ms": 5.0, "index": 0, "out_tokens": 1, "arrival_s": 0.0, "tpot_ms": None}]
        with mock.patch.object(serve, "serve_records", return_value=records):
            sections = serve.summary_sections([], 1.0, ModeCfg(sla_ttft_ms=30.0, sla_tpot_ms=10.0))
        self.assertIsNone(sections["sla"]["tpot_p99"])

    def test_off_and_void_and_goodput(self):
        self.assertEqual(
            summary_tag_lines(
                {
                    "serve": {"n": 1, "ttft_ms": PCT, "tpot_ms": None, "req_s": 1.0, "tok_s": 1.0},
                    "spec_decode": {"spec": "off"},
                }
            ),
            [
                (
                    "SERVE_SUMMARY n=1 ttft_ms_p50=10.0 ttft_ms_p90=20.0 ttft_ms_p95=30.0 ttft_ms_p99=40.1 "
                    "tpot_ms_p50=- req_s=1.000 tok_s=1.0"
                ),
                "SPEC_DECODE_SUMMARY spec=off note=needs-real-text-trace",
            ],
        )
        bad = {
            "plan_hash": "pp",
            "expected": "aa",
            "executed_hash": "bb",
            "match": False,
            "n": 3,
            "of": 4,
            "first_mismatch": {"session": 1, "turn": 2, "reason": "differs from the plan"},
        }
        self.assertEqual(
            summary_tag_lines({"workload": bad})[1],
            "WORKLOAD_MISMATCH replayed workload differs from the trace; comparison is void",
        )
        g = {"max_req_s": None, "speed": None, "fail_bound": [], "iterations": 1}
        self.assertEqual(
            summary_tag_lines({"workload": bad, "goodput": g}),
            [
                "WORKLOAD_CHECK plan=pp expected=aa executed=bb match=0 n=3/4 first_mismatch=1:2 (differs from the plan)",
                "WORKLOAD_MISMATCH replayed workload differs from the trace; sweep is void",
            ],
        )
        self.assertEqual(
            summary_tag_lines({"goodput": {**g, "fail_bound": ["ttft_p99"]}}),
            [("GOODPUT_SUMMARY max_req_s=none bound=ttft_p99 iterations=1 note=no-passing-speed-within-budget")],
        )
        self.assertEqual(
            summary_tag_lines({"goodput": {"max_req_s": 3.21, "speed": 2.0, "fail_bound": [], "iterations": 6}}),
            ["GOODPUT_SUMMARY max_req_s=3.210 speed=2 bound=none-within-range iterations=6"],
        )


class Sections(unittest.TestCase):
    def test_spec_decode_summary_scores_the_reference_reply(self):
        trace = RealTrace(
            [Request([1], 1, order=i, output_length=4, output_token_ids=[5, 6, 7, 8]) for i in range(3)], "t", "jsonl"
        )
        summary, out = run_sections(trace, TokenDriver(), ModeCfg(warmup=0, concurrency=3))
        self.assertEqual({r.accepted_tokens for r in summary.rows}, {2})
        self.assertEqual(
            tail(out, "SPEC_DECODE_SUMMARY"),
            ["SPEC_DECODE_SUMMARY spec=measured accept_len=2.00 accept_rate=0.500 n=3"],
        )

    def test_no_tokens_from_the_driver_means_spec_off(self):
        summary, out = run_sections(single_trace(3), FakeDriver(), ModeCfg(warmup=0, concurrency=3))
        self.assertEqual(summary.sections["spec_decode"], {"spec": "off"})
        self.assertEqual(tail(out, "SPEC_DECODE_SUMMARY"), ["SPEC_DECODE_SUMMARY spec=off note=needs-real-text-trace"])

    def test_http_spec_comes_from_server_counter_deltas(self):
        scrapes = iter([(100, 300, 150), (160, 480, 240)])

        def metrics():
            drafts, draft_tokens, accepted = next(scrapes)
            return (
                f'vllm:spec_decode_num_drafts_total{{engine="0"}} {drafts}\n'
                f'vllm:spec_decode_num_draft_tokens_total{{engine="0"}} {draft_tokens}\n'
                f'vllm:spec_decode_num_accepted_tokens_total{{engine="0"}} {accepted}\n'
            )

        with helpers.serve(metrics) as (endpoint, _):
            driver = HttpDriver("completions", url=endpoint + "/v1/completions", model="m")
            summary, out = run_sections(single_trace(3), driver, ModeCfg(warmup=0, concurrency=3))
        self.assertEqual(
            summary.sections["spec_decode"],
            {
                "spec": "measured",
                "source": "server_metrics",
                "accepted_tokens": 90,
                "draft_tokens": 180,
                "drafts": 60,
                "accept_rate": 0.5,
                "accept_len": 2.5,
                "n": 60,
            },
        )
        self.assertEqual(
            tail(out, "SPEC_DECODE_SUMMARY"),
            [
                "SPEC_DECODE_SUMMARY spec=measured source=server_metrics accept_len=2.50 accept_rate=0.500 "
                "drafts=60 draft_tokens=180 accepted_tokens=90"
            ],
        )

    def test_http_without_spec_counters_is_spec_off(self):
        with helpers.serve() as (endpoint, _):
            driver = HttpDriver("completions", url=endpoint + "/v1/completions", model="m")
            summary, _ = run_sections(single_trace(3), driver, ModeCfg(warmup=0, concurrency=3))
        self.assertEqual(summary.sections["spec_decode"], {"spec": "off"})

    def test_a_failed_metrics_scrape_reads_as_no_counters(self):
        driver = HttpDriver("completions", url="http://127.0.0.1:1/v1/completions", model="m")
        body = mock.MagicMock(**{"__enter__.return_value.read.return_value": b"\xff\xfe"})
        for name, effect in (
            ("refused", ConnectionRefusedError()),
            ("bad status", http.client.BadStatusLine("x")),
            ("not text", None),
        ):
            with (
                self.subTest(name),
                mock.patch("quettabench.engines.launcher.urlopen", side_effect=effect, return_value=body),
            ):
                self.assertIsNone(driver.spec_decode_counters())

    def test_sla_and_burst_follow_the_mode(self):
        summary, out = run_sections(
            single_trace(4), FakeDriver(), ModeCfg(warmup=0, concurrency=4, sla_ttft_ms=0.5, sla_tpot_ms=100)
        )
        self.assertEqual(summary.sections["sla"]["bound"], ["ttft_p99"])
        self.assertEqual(summary.sections["serve"]["ttft_ms"]["p99"], 1.0)
        self.assertTrue(tail(out, "SLA_SUMMARY verdict=FAIL bound=ttft_p99 ttft_p99=1.0/0.5"))
        self.assertTrue(tail(out, "BURST_SUMMARY peak_ttft_p99=1.0 recovery_s=na target_ms=0.5"))
        doc = summary.to_dict()
        self.assertEqual([doc[k]["percentile_method"] for k in ("serve", "sla", "burst")], ["nearest"] * 3)
        self.assertEqual(doc["percentile_method"], "linear")
        summary, out = run_sections(single_trace(4), FakeDriver(), ModeCfg(warmup=0, concurrency=4))
        self.assertNotIn("sla", summary.sections)
        self.assertFalse(tail(out, "SLA_SUMMARY") or tail(out, "BURST_SUMMARY"))


class GoodputSweep(unittest.TestCase):
    """100 ms per unit of speed against a 450 ms bound."""

    def fake_measure(self, speeds):
        async def measure(trace, driver, mode, emit, workload, request_id):
            speeds.append(workload.speed)
            pairs = []
            for i, q in enumerate(trace.requests()):
                r = RequestResult(
                    success=True,
                    ttft=0.1 * workload.speed,
                    e2el=1.0,
                    output_tokens=3,
                    input_tokens=1,
                    usage_reported=True,
                    scheduled_at_s=i * 0.1,
                    semaphore_acquired_at_s=i * 0.1,
                )
                emit(q, r)
                pairs.append((q, r))
            executed = {i: sent_digest(q) for i, q in enumerate(trace.requests())}
            return serve.Measurement([r for _, r in pairs], None, pairs, 1.0 / workload.speed, None, executed=executed)

        return measure

    def test_bisects_to_the_boundary_within_six_probes(self):
        speeds = []
        mode = ModeCfg(load="open", warmup=0, sla_ttft_ms=450.0, goodput_sweep=True)
        with mock.patch.object(serve, "measure", self.fake_measure(speeds)):
            summary, out = run_sections(single_trace(4), FakeDriver(), mode, WorkloadCfg(arrival="poisson:10"))
        g = summary.sections["goodput"]
        self.assertLessEqual(len(speeds), 6)
        self.assertEqual(len(tail(out, "GOODPUT_PROBE")), len(speeds))
        self.assertTrue(4.0 <= g["speed"] < 4.5, speeds)
        self.assertEqual(g["fail_bound"], ["ttft_p99"])
        self.assertEqual(
            tail(out, "GOODPUT_SUMMARY"),
            [
                (
                    f"GOODPUT_SUMMARY max_req_s={g['max_req_s']:.3f} speed={g['speed']:g} bound=ttft_p99 "
                    f"iterations={len(speeds)}"
                )
            ],
        )
        self.assertFalse(tail(out, "SERVE_SUMMARY"))

    def test_an_sla_run_with_a_failed_request_fails_and_reads_no_p99(self):
        mode = ModeCfg(concurrency=2, warmup=0, min_success_rate=0.5, sla_ttft_ms=1e6)
        summary, out = run(single_trace(8), FakeDriver(fail=lambda r: r.order == 0), mode)
        sla = summary.sections["sla"]
        self.assertEqual((sla["verdict"], sla["bound"], sla["failed_requests"]), ("FAIL", ["failed_requests"], 1))
        self.assertIsNone(sla["ttft_p99"])
        self.assertIn("SLA_SUMMARY verdict=FAIL bound=failed_requests ttft_p99=-/1000000.0", out)

    def test_the_sweep_returns_the_best_probes_requests_not_the_last_ones(self):
        speeds = []
        mode = ModeCfg(load="open", warmup=0, sla_ttft_ms=450.0, goodput_sweep=True)
        with mock.patch.object(serve, "measure", self.fake_measure(speeds)):
            summary, _ = run_sections(single_trace(4), FakeDriver(), mode, WorkloadCfg(arrival="poisson:10"))
        best = summary.sections["goodput"]["speed"]
        self.assertNotEqual(speeds[-1], best)  # the bisection ended on another probe
        self.assertAlmostEqual(summary.core.duration_s, 1.0 / best)  # the fake run lasts 1 / speed

    def test_a_probe_with_a_failed_request_fails_whatever_its_survivors_p99(self):
        failing = self.fake_measure([])

        async def one_failure(trace, driver, mode, emit, workload, request_id):
            out = await failing(trace, driver, mode, emit, workload, request_id)
            out.pairs[0][1].success = False
            return out

        mode = ModeCfg(load="open", warmup=0, sla_ttft_ms=1e6, goodput_sweep=True, min_success_rate=0.5)
        with mock.patch.object(serve, "measure", one_failure):
            code, out = run_sections(single_trace(4), FakeDriver(), mode, WorkloadCfg(arrival="poisson:10"))
        self.assertEqual(code, 1)
        self.assertIn("GOODPUT_PROBE speed=1 verdict=FAIL bound=failed_requests", out)

    def test_no_passing_speed_exits_1(self):
        mode = ModeCfg(load="open", warmup=0, sla_ttft_ms=1.0, goodput_sweep=True)
        with mock.patch.object(serve, "measure", self.fake_measure([])):
            code, out = run_sections(single_trace(4), FakeDriver(), mode, WorkloadCfg(arrival="poisson:10"))
        self.assertEqual(code, 1)
        self.assertIn("note=no-passing-speed-within-budget", out)


def grid_run(*sets):
    with tempfile.TemporaryDirectory() as td:
        run = run_grid_cli(grid_argv("llama", *sets, "--results-dir", td), {})
        doc = json.loads((Path(td) / "llama" / "run.json").read_text())
        rows = [json.loads(line) for line in (Path(td) / "llama" / "requests.jsonl").read_text().splitlines()]
    return run, doc, rows


class GridRunFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli, cls.doc, cls.rows = grid_run()

    def test_steps_are_the_result_lines(self):
        results = [line for line in self.cli["stdout"] if line.startswith("RESULT")]
        self.assertEqual(len(self.doc["steps"]), len(results))
        for s, line in zip(self.doc["steps"], results):
            self.assertIn(
                f"ctx={s['ctx']} bs={s['bs']} ms_per_step={s['ms_per_step']:.3f} ms_at_eval={s['ms_at_eval']:.3f}", line
            )
            self.assertIn(f"r2={s['r2']:.5f}", line)
        self.assertEqual(self.doc["run_status"], {"status": "ok", "reason": None})
        self.assertTrue(self.doc["summary"]["grid"]["done"])

    def test_rows_cover_every_batch(self):
        batches = {}
        for r in self.rows:
            c = r["cell"]
            key = (c["ctx"], c["bs"], c["step_point"], c["phase"], c["attempt"])
            batches.setdefault(key, []).append(r)
        self.assertEqual(len(batches), self.cli["generate_calls"] - 1)
        for (ctx, bs, step, phase, _), rows in batches.items():
            self.assertEqual(len(rows), bs)
            self.assertEqual(len({r["batch_wall_s"] for r in rows}), 1)
            self.assertEqual({r["output_tokens"] for r in rows}, {step})
            self.assertEqual(all(r.get("excluded_from_summary", False) for r in rows), phase != "fit")

    def test_dump_tokens_writes_text_rows(self):
        _, _, rows = grid_run("--set", "mode.grid=smoke", "--set", "mode.return_outputs=text")
        text = [r for r in rows if r["cell"]["phase"] == "text"]
        self.assertTrue(text)
        self.assertEqual({r["cell"]["step_point"] for r in text}, {64})
        self.assertEqual(text[0]["token_ids"], list(range(64)))


class ModelConfig(unittest.TestCase):
    def test_a_repo_id_reads_its_cached_snapshot(self):
        with tempfile.TemporaryDirectory() as cache:
            repo = Path(cache, "models--org--toy")
            snap = repo / "snapshots" / "abc123"
            snap.mkdir(parents=True)
            (repo / "refs").mkdir()
            (repo / "refs" / "main").write_text("abc123")
            (snap / "config.json").write_text(json.dumps(DENSE_CONFIG))
            with mock.patch("huggingface_hub.constants.HF_HUB_CACHE", cache):
                self.assertEqual(serve.resolve_kv_bytes_per_token(EngineCfg(model="org/toy")), 131072)
                with self.assertRaisesRegex(ConfigError, "huggingface-cli download org/missing"):
                    serve.resolve_kv_bytes_per_token(EngineCfg(model="org/missing"))

    def test_a_local_dir_reads_its_config(self):
        with tempfile.TemporaryDirectory() as model:
            Path(model, "config.json").write_text(json.dumps(DENSE_CONFIG))
            self.assertEqual(serve.resolve_kv_bytes_per_token(EngineCfg(model=model)), 131072)


class GridRunOverHttp(unittest.TestCase):
    def test_smoke_grid_end_to_end(self):
        with tempfile.TemporaryDirectory() as td, helpers.serve() as (endpoint, posts):
            Path(td, "config.json").write_text(json.dumps(DENSE_CONFIG))
            (cfg,) = loader.load(
                RUNS / "llama.yaml",
                [
                    f"engine.model={td}",
                    "mode.grid=smoke",
                    "mode.fit_attempts=1",
                    "hardware.visible_devices=0",
                    "hardware.backend=cpu",
                    "hardware.link_profile=nvlink",
                ],
                env={},
            )
            handle = launcher.NoServer(Path(td) / "out", endpoint=endpoint)
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(launcher, "start", return_value=handle) as start,
                mock.patch.object(launcher, "accelerator", return_value=("FAKE-80GiB", 80 << 30)),
            ):
                runner.run_one(cfg, Path(td) / "out")
            doc = json.loads((Path(td) / "out" / "run.json").read_text())
        start.assert_called_once()
        (step,) = doc["steps"]
        self.assertEqual((step["ctx"], step["bs"], step["step_points"]), (100, 1, [60, 230, 400]))
        self.assertEqual(doc["run_status"]["status"], "ok")
        resets = [path for path, _ in posts if path == "/reset_prefix_cache"]
        self.assertEqual(len(resets), 1)
        sent = [b["max_tokens"] for path, b in posts if path == "/v1/completions"]
        self.assertEqual(sent, [20, 400, 60, 230, 400])


if __name__ == "__main__":
    unittest.main()
