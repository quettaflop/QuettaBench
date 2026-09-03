"""Defaults and wait math for multi-turn scheduling."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.workloads.dataset import TrajectoryMultiTurnDataset

from src.benchmark.runner import get_args, inter_turn_wait_s
from src.workloads.dataset import session_arrival_ms, turn_wait_metadata


class TestInterTurnWait(unittest.TestCase):
    def test_default_is_zero(self):
        req = SimpleNamespace(metadata={"tool_wait_after_ms": 1500, "human_wait_after_ms": 800})
        self.assertEqual(inter_turn_wait_s(req), 0.0)
        self.assertEqual(inter_turn_wait_s(None), 0.0)

    def test_cli_waits_sum(self):
        self.assertAlmostEqual(
            inter_turn_wait_s(None, tool_wait_ms=200, human_wait_ms=50),
            0.25,
        )

    def test_recorded_waits_opt_in(self):
        req = SimpleNamespace(metadata={"tool_wait_after_ms": 1000, "human_wait_after_ms": 500})
        self.assertEqual(inter_turn_wait_s(req, use_recorded_waits=False), 0.0)
        self.assertAlmostEqual(inter_turn_wait_s(req, use_recorded_waits=True), 1.5)

    def test_recorded_plus_cli(self):
        req = SimpleNamespace(metadata={"tool_wait_after_ms": 1000})
        self.assertAlmostEqual(
            inter_turn_wait_s(req, tool_wait_ms=250, human_wait_ms=250, use_recorded_waits=True),
            1.5,
        )


class TestTrajectoryWaitFields(unittest.TestCase):
    def test_tool_ms_alias(self):
        self.assertEqual(
            turn_wait_metadata({"tool_ms": 12.5}),
            {"tool_wait_after_ms": 12.5, "human_wait_after_ms": 0.0},
        )

    def test_human_alias(self):
        md = turn_wait_metadata({"human_wait_ms": 80})
        self.assertEqual(md["human_wait_after_ms"], 80.0)

    def test_arrival_aliases(self):
        self.assertEqual(session_arrival_ms({}), 0.0)
        self.assertEqual(session_arrival_ms({"arrival_time": 1500}), 1500.0)
        self.assertEqual(session_arrival_ms({"arrival_time_ms": 200}), 200.0)


class TestTrajectoryLoadWaits(unittest.TestCase):
    def test_loads_tool_ms_and_arrival(self):
        payload = {
            "session_id": "s1",
            "source": "test",
            "arrival_time_ms": 2500,
            "turns": [
                {
                    "turn_idx": 0,
                    "messages": [{"role": "user", "content": "hello world " * 20}],
                    "osl_tokens": 8,
                    "tool_ms": 40,
                },
                {
                    "turn_idx": 1,
                    "messages": [
                        {"role": "user", "content": "hello world " * 20},
                        {"role": "assistant", "content": "ok"},
                        {"role": "user", "content": "next " * 20},
                    ],
                    "osl_tokens": 8,
                    "human_ms": 90,
                },
                {
                    "turn_idx": 2,
                    "messages": [{"role": "user", "content": "third " * 20}],
                    "osl_tokens": 8,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "traj.jsonl"
            path.write_text(json.dumps(payload) + "\n")
            ds = TrajectoryMultiTurnDataset(
                str(path), min_turns=3, max_turns=10, num_sessions=1
            )
            session = ds.sessions[0]
            self.assertEqual(session.arrival_time_ms, 2500.0)
            self.assertEqual(session.turns[0].metadata["tool_wait_after_ms"], 40.0)
            self.assertEqual(session.turns[1].metadata["human_wait_after_ms"], 90.0)


class TestCliDefaults(unittest.TestCase):
    def test_turn_pacing_defaults_to_per_session(self):
        with patch("sys.argv", ["runner"]):
            args = get_args()
        self.assertEqual(args.turn_pacing, "per-session")
        self.assertEqual(args.tool_wait_ms, 0.0)
        self.assertEqual(args.human_wait_ms, 0.0)
        self.assertFalse(args.use_recorded_waits)
        self.assertFalse(args.use_recorded_arrivals)
        self.assertEqual(args.load_mode, "closed-loop")


if __name__ == "__main__":
    unittest.main()
