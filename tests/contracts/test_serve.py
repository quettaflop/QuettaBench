"""serve.run over a Trace and a Driver with no server."""

import contextlib
import copy
import http.client
import io
import json
import os
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest import mock

from quettabench import runner
from quettabench.config import loader
from quettabench.config.schema import Config, ConfigError, EngineCfg, HardwareCfg, ModeCfg, WorkloadCfg
from quettabench.engines import launcher
from quettabench.engines.drivers import HttpDriver
from quettabench.metrics.core import RequestResult
from quettabench.metrics.results import RunWriter, summary_tag_lines
from quettabench.modes import serve
from quettabench.workloads.base import Request
from quettabench.workloads.live import RealTrace
from quettabench.workloads.sessions import SessionTrace, TurnPacing
from tests.contracts import helpers
from tests.golden._harness import DENSE_CONFIG, RUNS, grid_argv, run_grid_cli


class FakeDriver:
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
        return RequestResult(
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
        self.assertNotIn("WORKLOADVOID", out)

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

    def test_a_request_changed_after_hashing_fails_the_identity_gate(self):
        trace = single_trace(4)
        trace.workload_hash = lambda: "0" * 16
        code, out = run(trace, FakeDriver(), ModeCfg(warmup=0))
        self.assertEqual(code, 3)
        self.assertIn("WORKLOADVOID replayed workload differs from the trace; comparison is void", out)

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


class Sessions(unittest.TestCase):
    def test_replies_feed_later_turns(self):
        driver = FakeDriver()
        summary, _ = run(session_trace(), driver, ModeCfg(concurrency=2, warmup=0))
        last_turn = next(m for m, _, _ in driver.sent if m[-1]["content"] == "s0t2")
        self.assertEqual([m["content"] for m in last_turn if m["role"] == "assistant"], ["reply0", "reply1"])
        self.assertEqual([t.turn_index for t in summary.turns], [0, 1, 2])
        self.assertEqual(
            {(r.session_id, r.turn_index) for r in summary.rows}, {(s, t) for s in range(2) for t in range(3)}
        )

    def test_a_dropped_turn_fails_the_identity_gate(self):
        with tempfile.TemporaryDirectory() as td:
            code, out = run(
                session_trace(),
                FakeDriver(fail=lambda r: (r.session_id, r.turn) == (1, 1)),
                ModeCfg(warmup=0, min_success_rate=0.5),
                results_dir=td,
            )
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
        code, out = run(
            trace, FakeDriver(fail=lambda r: (r.session_id, r.turn) == (0, 0)), ModeCfg(warmup=0, min_success_rate=0.0)
        )
        self.assertEqual(code, 3)
        self.assertIn("Turn 2/3: dispatching 1 requests...", out)


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
            "workload": {"expected": "aa", "replayed": "aa", "match": True, "n": 4, "of": 4},
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
            "spec": {"spec": "measured", "accept_len": 2.0, "accept_rate": 0.5, "n": 4},
        }
        self.assertEqual(
            summary_tag_lines(summary),
            [
                "WORKLOADSUM expected=aa replayed=aa match=1 n=4/4",
                (
                    "SERVESUM n=4 ttft_ms_p50=10.0 ttft_ms_p90=20.0 ttft_ms_p95=30.0 ttft_ms_p99=40.1 "
                    "tpot_ms_p50=10.040 tpot_ms_p90=20.000 tpot_ms_p95=30.000 tpot_ms_p99=40.060 "
                    "req_s=2.500 tok_s=100.0"
                ),
                ("SLASUM verdict=FAIL bound=ttft_p99 ttft_p99=40.1/30.0 tpot_p99=40.060/- rate_req_s=2.500"),
                "BURSTSUM peak_ttft_p99=40.1 recovery_s=na target_ms=30.0",
                "SPECSUM spec=measured accept_len=2.00 accept_rate=0.500 n=4",
            ],
        )

    def test_slasum_without_tpot_samples_prints_a_dash(self):
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
            ["SLASUM verdict=PASS bound=none ttft_p99=5.0/30.0 tpot_p99=-/10.0 rate_req_s=1.000"],
        )
        records = [{"ttft_ms": 5.0, "index": 0, "out_tokens": 1, "arrival_s": 0.0, "tpot_ms": None}]
        with mock.patch.object(serve, "serve_records", return_value=records):
            sections = serve.crossval_sections([], 1.0, ModeCfg(sla_ttft_ms=30.0, sla_tpot_ms=10.0))
        self.assertIsNone(sections["sla"]["tpot_p99"])

    def test_off_and_void_and_goodput(self):
        self.assertEqual(
            summary_tag_lines(
                {
                    "serve": {"n": 1, "ttft_ms": PCT, "tpot_ms": None, "req_s": 1.0, "tok_s": 1.0},
                    "spec": {"spec": "off"},
                }
            ),
            [
                (
                    "SERVESUM n=1 ttft_ms_p50=10.0 ttft_ms_p90=20.0 ttft_ms_p95=30.0 ttft_ms_p99=40.1 "
                    "tpot_ms_p50=- req_s=1.000 tok_s=1.0"
                ),
                "SPECSUM spec=off note=needs-real-text-trace",
            ],
        )
        bad = {"expected": "aa", "replayed": "bb", "match": False, "n": 3, "of": 4}
        self.assertEqual(
            summary_tag_lines({"workload": bad})[1],
            "WORKLOADVOID replayed workload differs from the trace; comparison is void",
        )
        g = {"max_req_s": None, "speed": None, "fail_bound": [], "iterations": 1}
        self.assertEqual(
            summary_tag_lines({"workload": bad, "goodput": g}),
            [
                "WORKLOADSUM expected=aa replayed=bb match=0 n=3/4",
                "WORKLOADVOID replayed workload differs from the trace; sweep is void",
            ],
        )
        self.assertEqual(
            summary_tag_lines({"goodput": {**g, "fail_bound": ["ttft_p99"]}}),
            [("GOODPUTSUM max_req_s=none bound=ttft_p99 iterations=1 note=no-passing-speed-within-budget")],
        )
        self.assertEqual(
            summary_tag_lines({"goodput": {"max_req_s": 3.21, "speed": 2.0, "fail_bound": [], "iterations": 6}}),
            ["GOODPUTSUM max_req_s=3.210 speed=2 bound=none-within-range iterations=6"],
        )


class Sections(unittest.TestCase):
    def test_specsum_scores_the_reference_reply(self):
        trace = RealTrace(
            [Request([1], 1, order=i, output_length=4, output_token_ids=[5, 6, 7, 8]) for i in range(3)], "t", "jsonl"
        )
        summary, out = run_sections(trace, TokenDriver(), ModeCfg(warmup=0, concurrency=3))
        self.assertEqual({r.accepted_tokens for r in summary.rows}, {2})
        self.assertEqual(tail(out, "SPECSUM"), ["SPECSUM spec=measured accept_len=2.00 accept_rate=0.500 n=3"])

    def test_no_tokens_from_the_driver_means_spec_off(self):
        summary, out = run_sections(single_trace(3), FakeDriver(), ModeCfg(warmup=0, concurrency=3))
        self.assertEqual(summary.sections["spec"], {"spec": "off"})
        self.assertEqual(tail(out, "SPECSUM"), ["SPECSUM spec=off note=needs-real-text-trace"])

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
            summary.sections["spec"],
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
            tail(out, "SPECSUM"),
            [
                "SPECSUM spec=measured source=server_metrics accept_len=2.50 accept_rate=0.500 "
                "drafts=60 draft_tokens=180 accepted_tokens=90"
            ],
        )

    def test_http_without_spec_counters_is_spec_off(self):
        with helpers.serve() as (endpoint, _):
            driver = HttpDriver("completions", url=endpoint + "/v1/completions", model="m")
            summary, _ = run_sections(single_trace(3), driver, ModeCfg(warmup=0, concurrency=3))
        self.assertEqual(summary.sections["spec"], {"spec": "off"})

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
                self.assertIsNone(driver.spec_counters())

    def test_sla_and_burst_follow_the_mode(self):
        summary, out = run_sections(
            single_trace(4), FakeDriver(), ModeCfg(warmup=0, concurrency=4, sla_ttft_ms=0.5, sla_tpot_ms=100)
        )
        self.assertEqual(summary.sections["sla"]["bound"], ["ttft_p99"])
        self.assertEqual(summary.sections["serve"]["ttft_ms"]["p99"], 1.0)
        self.assertTrue(tail(out, "SLASUM verdict=FAIL bound=ttft_p99 ttft_p99=1.0/0.5"))
        self.assertTrue(tail(out, "BURSTSUM peak_ttft_p99=1.0 recovery_s=na target_ms=0.5"))
        doc = summary.to_dict()
        self.assertEqual([doc[k]["percentile_method"] for k in ("serve", "sla", "burst")], ["nearest"] * 3)
        self.assertEqual(doc["percentile_method"], "linear")
        summary, out = run_sections(single_trace(4), FakeDriver(), ModeCfg(warmup=0, concurrency=4))
        self.assertNotIn("sla", summary.sections)
        self.assertFalse(tail(out, "SLASUM") or tail(out, "BURSTSUM"))


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
            return [r for _, r in pairs], None, pairs, 1.0 / workload.speed, None

        return measure

    def test_bisects_to_the_boundary_within_six_probes(self):
        speeds = []
        mode = ModeCfg(load="open", warmup=0, sla_ttft_ms=450.0, goodput_sweep=True)
        with mock.patch.object(serve, "measure", self.fake_measure(speeds)):
            summary, out = run_sections(single_trace(4), FakeDriver(), mode, WorkloadCfg(arrival="poisson:10"))
        g = summary.sections["goodput"]
        self.assertLessEqual(len(speeds), 6)
        self.assertEqual(len(tail(out, "GOODPROBE")), len(speeds))
        self.assertTrue(4.0 <= g["speed"] < 4.5, speeds)
        self.assertEqual(g["fail_bound"], ["ttft_p99"])
        self.assertEqual(
            tail(out, "GOODPUTSUM"),
            [
                (
                    f"GOODPUTSUM max_req_s={g['max_req_s']:.3f} speed={g['speed']:g} bound=ttft_p99 "
                    f"iterations={len(speeds)}"
                )
            ],
        )
        self.assertFalse(tail(out, "SERVESUM"))

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
        _, _, rows = grid_run("--set", "mode.grid=smoke", "--set", "mode.returns=text")
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
                self.assertEqual(serve.kv_bytes_per_token(EngineCfg(model="org/toy")), 131072)
                with self.assertRaisesRegex(ConfigError, "huggingface-cli download org/missing"):
                    serve.kv_bytes_per_token(EngineCfg(model="org/missing"))

    def test_a_local_dir_reads_its_config(self):
        with tempfile.TemporaryDirectory() as model:
            Path(model, "config.json").write_text(json.dumps(DENSE_CONFIG))
            self.assertEqual(serve.kv_bytes_per_token(EngineCfg(model=model)), 131072)


class GridRunOverHttp(unittest.TestCase):
    def test_smoke_grid_end_to_end(self):
        with tempfile.TemporaryDirectory() as td, helpers.serve() as (endpoint, posts):
            Path(td, "config.json").write_text(json.dumps(DENSE_CONFIG))
            (cfg,) = loader.load(
                RUNS / "llama.yaml",
                [
                    f"engine.model={td}",
                    "mode.grid=smoke",
                    "mode.vmin_attempts=1",
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
