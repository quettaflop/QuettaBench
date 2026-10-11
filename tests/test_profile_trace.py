"""Profile sources drawn into Traces in the order the old runner drew them."""

import contextlib
import dataclasses
import io
import json
import tempfile
import unittest
from pathlib import Path

from quettabench import workloads
from quettabench.config.schema import Config, EngineCfg, HardwareCfg, ModeCfg, WorkloadCfg
from quettabench.workloads.base import Request, workload_hash
from quettabench.workloads.live import MooncakeTrace, PromptJsonlTrace
from quettabench.workloads.sessions import TrajectorySessionTrace, TurnPacing


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def config(source, kind="live"):
    return Config(engine=EngineCfg(), workload=WorkloadCfg(kind=kind, source=source),
                  mode=ModeCfg(), hardware=HardwareCfg())


class SingleTurn(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.dir = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def test_warmup_draws_come_first_without_a_rewind(self):
        path = write_jsonl(self.dir / "c.jsonl", [
            {"system": "s", "user": f"u{i}", "osl_tokens": 2 + i} for i in range(4)])
        source = PromptJsonlTrace(path, seed=5)
        drawn = [source.get_next_request() for _ in range(9)]
        trace = PromptJsonlTrace(path, seed=5).draw(7, warmup=2)
        self.assertEqual([r.messages for r in trace.warmup_requests(2)],
                         [d.messages for d in drawn[:2]])
        self.assertEqual([r.messages for r in trace.requests()], [d.messages for d in drawn[2:]])
        self.assertEqual([r.max_tokens for r in trace.requests()],
                         [d.max_tokens for d in drawn[2:]])
        self.assertEqual([r.order for r in trace.requests()], list(range(7)))

    def test_a_rewindable_trace_restarts_after_warmup(self):
        path = write_jsonl(self.dir / "m.jsonl", [
            {"timestamp": i, "input_length": 30 + i, "output_length": 3, "hash_ids": [i]}
            for i in range(5)])
        trace = MooncakeTrace(path).draw(5, warmup=3)
        first = trace.requests()[0]
        self.assertEqual(first.prompt_token_ids, trace.warmup_requests(1)[0].prompt_token_ids)
        self.assertEqual(first.prompt_len, 30)
        self.assertEqual(first.metadata["request_index"], 0)

    def test_messages_and_metadata_never_reach_the_hash(self):
        plain = Request([1, 2], 2, order=0, output_length=4)
        chat = dataclasses.replace(plain, messages=[{"content": "x"}], metadata={"k": 1})
        self.assertEqual(chat.to_dict(), plain.to_dict())
        self.assertEqual(workload_hash([chat]), workload_hash([plain]))

    def test_build_names_the_profile(self):
        trace = workloads.build(config("test"))
        self.assertEqual(trace.source_meta(), {"profile": "test", "dataset": "test", "path": "",
                                               "reqs": 100})
        self.assertEqual(trace.requests()[0].max_tokens, 20)

    def test_multi_turn_profile_is_refused_single_turn(self):
        with self.assertRaisesRegex(ValueError, "is multi-turn; run it with workload.kind=sessions"):
            workloads.build(config("swebench-multiturn-short"))


class MultiTurn(unittest.TestCase):
    def build(self, **kw):
        sessions = []
        for s in range(3):
            turns = [{"turn_idx": t, "messages": [{"role": "user", "content": f"s{s} t{t}"}],
                      "osl_tokens": 2, "tool_ms": 5} for t in range(13 + s)]
            sessions.append({"session_id": f"S{s}", "arrival_time_ms": 100 * s, "turns": turns})
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        path = write_jsonl(Path(td.name) / "t.jsonl", sessions)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            trace = TrajectorySessionTrace(path, min_turns=13, num_sessions=3, **kw)
        return trace, out.getvalue()

    def test_sessions_carry_turns_arrivals_and_waits(self):
        trace, out = self.build(warmup=4, pacing=TurnPacing("interleaved"))
        self.assertIn("Loaded 3 sessions, max 15 turns per session", out)
        for turns in trace.sessions():
            sid = turns[0].session_id
            source = int(turns[0].metadata["source_session_id"][1:])
            self.assertEqual([r.messages[0]["content"] for r in turns],
                             [f"s{source} t{t}" for t in range(13 + source)])
            self.assertEqual([r.turn for r in turns], list(range(len(turns))))
            self.assertEqual(turns[0].arrival_ts, source / 10.0)
            self.assertIsNone(turns[1].arrival_ts)
            self.assertEqual(turns[2].metadata["tool_wait_after_ms"], 5.0)
            self.assertEqual(trace.replies[sid], [])
        self.assertEqual(trace.pacing.pacing, "interleaved")
        self.assertEqual(trace.max_turns, 15)
        firsts = [s[0].messages for s in trace.sessions()]
        self.assertEqual([r.messages for r in trace.warmup_requests(4)], firsts + [
            trace.sessions()[0][1].messages])
        self.assertEqual([r.order for r in trace.warmup_requests(4)], [0, 1, 2, 3])

    def test_turns_past_max_turn_index_are_not_in_the_trace(self):
        trace, _ = self.build(max_turn_index=4)
        self.assertEqual({len(t) for t in trace.sessions()}, {5})
        self.assertEqual(trace.max_turns, 15)

    def test_single_turn_profile_is_refused(self):
        with self.assertRaisesRegex(ValueError, "does not use a multi-turn dataset"):
            workloads.build(config("test", kind="sessions"))


if __name__ == "__main__":
    unittest.main()
