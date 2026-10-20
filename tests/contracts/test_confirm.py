"""tools/confirm: the gates on hand-made run files, the matrix pinning, the window estimate, and every phase once
against the fake server on the fixture sessions, with qbench run as a black box."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from quettabench.modes.serve import first_start_stagger_s
from tests.contracts.test_workloads import write_fixtures
from tools.confirm import confirm

ROOT = Path(__file__).resolve().parents[2]
# Replies stream at 20 microseconds a token so a session of the fixture's longest turns takes well under a second.
FAST_TIMING = ("--ttft-a", "0.001", "--ttft-b", "0", "--tpot", "0.00002", "--workers", "1", "--no-hash")
# The smoke's stand-in server: serve's 1% latency bounds would fail a 1 ms TTFT or 0 ms TPOT on any machine load;
# 500 ms and 5 ms put them at 5 ms of lag and 1% of a reply, still a fast server.
STAND_IN_TIMING = ("--ttft-a", "0.5", "--ttft-b", "0", "--tpot", "0.005", "--workers", "1", "--no-hash")
RUN_YAML = """\
model: qwen38_flash_next
engine:
  model: Qwen/Qwen3.8-Flash-Next
  tp: 1
  prefix_caching: true
workload:
  kind: sessions
  use_recorded_waits: false
  stratified: true
mode:
  mode: serve
  load: closed
  duration_s: 60
  warmup: 0
  reset_prefix_cache: true
  exact_output_length: true
  timeout: 1800
hardware:
  launcher: attach
matrix:
  workload.source: [swebench-qwen-multiturn-medium]
  mode.concurrency+engine.extra.vllm.max_cudagraph_capture_size:
  - [2, 64]
  - [4, 256]
"""


def run_json(status="ok", failed=0, total=24, latency_valid=True, lag99=1.2, lagmax=None, match=True, cache=0.9):
    reasons = [] if latency_valid else ["client stalls could have moved ttft p99 by 3.0%, over 1%"]
    lagmax = lag99 * 2 if lagmax is None else lagmax
    identity = {"match": match, "n": total, "of": total}
    if not match:
        identity["first_mismatch"] = {"session": "3#0", "turn": 1, "reason": "differs from the plan"}
    return {
        "run_status": {"status": status, "reason": None if status == "ok" else "latency withheld"},
        "summary": {
            "num_requests": total,
            "failed_requests": failed,
            "validity": {"latency_valid": latency_valid, "throughput_valid": not failed, "reasons": reasons},
            "client": {"loop_lag_ms": {"50": 0.3, "99": lag99, "max": lagmax}},
            "workload": identity,
            "server_cache_hit_rate": cache,
            "median_ttft_ms": 200.0,
            "median_tpot_ms": 20.0,
        },
    }


def rows(n=24, usage=True, exact=True):
    out = 100 if exact else 90
    return [{"success": True, "usage_reported": usage, "output_tokens": out, "max_tokens_requested": 100}] * n


def failed_gates(gates):
    return [name for name, passed, _ in gates if not passed]


class MatrixPinning(unittest.TestCase):
    def test_a_sweep_yaml_is_pinned_to_the_first_choice_of_every_axis(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sweep.yaml"
            path.write_text(RUN_YAML)
            self.assertEqual(
                confirm.first_point(path),
                [
                    'workload.source=["swebench-qwen-multiturn-medium"]',
                    "mode.concurrency+engine.extra.vllm.max_cudagraph_capture_size=[[2, 64]]",
                ],
            )
            path.write_text("engine:\n  model: m\n")
            self.assertEqual(confirm.first_point(path), [])


class ClientGates(unittest.TestCase):
    def test_a_clean_run_passes_every_gate(self):
        self.assertEqual(failed_gates(confirm.client_gates(run_json())), [])

    def test_each_cause_fails_its_own_gate(self):
        self.assertEqual(failed_gates(confirm.client_gates(run_json(failed=1))), ["no failed request"])
        self.assertEqual(failed_gates(confirm.client_gates(run_json(total=0))), ["no failed request"])
        late = run_json(lag99=7.0, lagmax=20.0)
        self.assertEqual(failed_gates(confirm.client_gates(late)), ["event loop p99 lag under 5 ms"])
        self.assertEqual(confirm.client_gates(late)[1][2], "p99 7.0 ms")
        self.assertEqual(failed_gates(confirm.client_gates(run_json(lagmax=60.0))), ["no loop stall over 50 ms"])

    def test_latency_withheld_on_the_fake_server_is_a_note_not_a_failed_gate(self):
        run = run_json(status="partial", latency_valid=False)
        self.assertEqual(failed_gates(confirm.client_gates(run)), [])
        self.assertEqual(
            confirm.client_notes(run),
            [
                "run status partial; latency withheld on the fake server: "
                "client stalls could have moved ttft p99 by 3.0%, over 1%"
            ],
        )
        self.assertEqual(confirm.client_notes(run_json()), ["run status ok; latency valid on the fake server"])

    def test_a_run_without_the_client_section_fails_both_loop_gates_rather_than_passing_them(self):
        run = run_json()
        del run["summary"]["client"]
        self.assertEqual(
            failed_gates(confirm.client_gates(run)), ["event loop p99 lag under 5 ms", "no loop stall over 50 ms"]
        )


class SmokeGates(unittest.TestCase):
    def test_a_clean_smoke_passes_with_no_warning(self):
        gates, warnings = confirm.smoke_gates(run_json(), rows())
        self.assertEqual((failed_gates(gates), warnings), ([], []))

    def test_the_run_level_gates_are_the_real_runs(self):
        self.assertEqual(failed_gates(confirm.smoke_gates(run_json(status="partial"), rows())[0]), ["run finished ok"])
        self.assertEqual(failed_gates(confirm.smoke_gates(run_json(failed=1), rows())[0]), ["no failed request"])
        self.assertEqual(failed_gates(confirm.smoke_gates(run_json(latency_valid=False), rows())[0]), ["latency valid"])

    def test_each_cause_fails_its_own_gate(self):
        self.assertEqual(
            failed_gates(confirm.smoke_gates(run_json(), rows(usage=False))[0]), ["every reply carries usage"]
        )
        self.assertEqual(
            failed_gates(confirm.smoke_gates(run_json(), rows(exact=False))[0]), ["every reply is its planned length"]
        )
        gates, _ = confirm.smoke_gates(run_json(match=False), rows())
        self.assertEqual(failed_gates(gates), ["every request as planned"])
        self.assertIn('"reason": "differs from the plan"', gates[-1][2])
        self.assertEqual(
            failed_gates(confirm.smoke_gates(run_json(), [])[0]),
            ["every reply carries usage", "every reply is its planned length"],
        )

    def test_a_server_without_cached_token_counts_is_a_warning_not_a_failure(self):
        gates, warnings = confirm.smoke_gates(run_json(cache=None), rows())
        self.assertEqual(failed_gates(gates), [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("--enable-prompt-tokens-details", warnings[0])


def turn(max_tokens, tool_wait_ms=None):
    meta = {"tool_wait_after_ms": tool_wait_ms} if tool_wait_ms is not None else {}
    return SimpleNamespace(max_tokens=max_tokens, metadata=meta)


class WindowEstimate(unittest.TestCase):
    # Session A: 0.1 + 99 * 0.01 = 1.09 s, a 1.0 s wait, then 0.1 + 49 * 0.01 = 0.59 s; the last turn's wait is not
    # slept. Session B: 0.1 + 9 * 0.01 = 0.19 s. Three agents cycle A, B, A, each starting 10 ms after the one before.
    def test_the_window_opens_when_the_last_first_session_ends(self):
        sessions = [[turn(100, 1000.0), turn(50, 5000.0)], [turn(10)]]
        self.assertAlmostEqual(first_start_stagger_s(3), 0.01)
        self.assertAlmostEqual(confirm.session_s(sessions[0], 0.1, 0.01, waits=True), 2.68)
        self.assertAlmostEqual(confirm.session_s(sessions[0], 0.1, 0.01, waits=False), 1.68)
        self.assertAlmostEqual(confirm.window_opens_s(sessions, 3, 0.1, 0.01, waits=True), 2.70)
        self.assertAlmostEqual(confirm.window_opens_s(sessions, 3, 0.1, 0.01, waits=False), 1.70)
        self.assertAlmostEqual(confirm.window_opens_s(sessions, 2, 0.1, 0.01, waits=True), 2.68)


class EveryPhase(unittest.TestCase):
    """The whole tool once, qbench as a subprocess on the fixture sessions: the client self-test on a fast fake
    server, then the server, smoke and sizing phases on a second one standing in for the real server."""

    def test_every_phase_passes_against_the_fake_server(self):
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as td:
            write_fixtures(td)
            configs = Path(td) / "configs"
            (configs / "runs").mkdir(parents=True)
            for name in ("models", "hardware", "grids.yaml", "workloads.yaml"):
                (configs / name).symlink_to(ROOT / "configs" / name)
            (configs / "runs" / "confirm_test.yaml").write_text(RUN_YAML)
            port = confirm.free_port()
            out = io.StringIO()
            try:
                # the gates' thresholds are tested on hand-made runs; the machine's load must not fail the plumbing
                with (
                    mock.patch.object(confirm, "LOOP_LAG_LIMIT_MS", 1e9),
                    mock.patch.object(confirm, "STALL_LIMIT_MS", 1e9),
                    mock.patch.object(confirm, "FAKE_TIMING", FAST_TIMING),
                    confirm.FakeServer(port, "Qwen/Qwen3.8-Flash-Next", Path(td) / "server.jsonl", STAND_IN_TIMING),
                    contextlib.redirect_stdout(out),
                ):
                    argv = [
                        "configs/runs/confirm_test.yaml",
                        "--agents",
                        "2",
                        "--allow-unverified",
                        "--server",
                        str(port),
                    ]
                    argv += ["--results-dir", "results/confirm", "--selftest-s", "1", "--workdir", td]
                    code = confirm.main(argv)
            finally:
                os.chdir(cwd)
            report = json.loads((Path(td) / "results" / "confirm" / "confirm.json").read_text())
        printed = out.getvalue()
        self.assertEqual((code, printed.splitlines()[-1]), (0, "CONFIRM PASS"), printed)
        self.assertEqual(list(report), ["config", "data", "client", "server", "smoke", "sizing"])
        self.assertTrue(all(phase["passed"] for phase in report.values()), printed)
        self.assertIn("not in data/MANIFEST.json", report["data"]["warnings"][0])
        self.assertEqual(failed_gates(report["client"]["gates"]), [])
        self.assertEqual((failed_gates(report["smoke"]["gates"]), report["smoke"]["warnings"]), ([], []))
        self.assertGreater(report["sizing"]["opens_s"], 0.0)
        self.assertGreaterEqual(report["sizing"]["duration_s_advised"], 600)
