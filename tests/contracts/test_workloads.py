"""Traces, arrivals, grid prompts, the profile registry, profile sources, real-trace datasets and every runnable profile's pinned content hash."""

import contextlib
import dataclasses
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import patch

from quettabench import runner
from quettabench.config.schema import Config, ConfigError, EngineCfg, HardwareCfg, ModeCfg, WorkloadCfg
from quettabench.modes.serve import make_trace_request_id
from quettabench.workloads.arrival import (
    parse_arrival,
    poisson_arrivals,
    ramp_arrivals,
    rebase_arrivals,
)
from quettabench.workloads.base import Request, workload_hash
from quettabench.workloads.grid import GridTrace, clone_ids
from quettabench.workloads.live import (
    PROFILES,
    MooncakeTrace,
    PromptJsonlTrace,
    RealTrace,
    ShareGPTTrace,
    WorkloadProfile,
    active_profiles,
    get_profile,
    load_profiles,
)
from quettabench.workloads.sessions import (
    SessionTrace,
    TrajectorySessionTrace,
    TurnPacing,
    resolve_multi_turn_num_sessions,
)


def write_jsonl(td, rows):
    p = Path(td) / "t.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


class RealTraceTest(unittest.TestCase):
    def test_supports_spec_only_with_real_assistant_tokens(self):
        with tempfile.TemporaryDirectory() as td:
            plain = RealTrace.from_jsonl(write_jsonl(td, [{"prompt_token_ids": [1, 2]}]))
            self.assertFalse(plain.supports_spec)
            self.assertEqual(plain.source_meta()["text_mode"], "synthetic")
            real = RealTrace.from_jsonl(write_jsonl(td, [
                {"prompt_token_ids": [1, 2]},
                {"prompt_token_ids": [3], "output_token_ids": [4, 5]}]))
            self.assertTrue(real.supports_spec)
            self.assertEqual(real.source_meta()["text_mode"], "real")

    def test_empty_file_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                RealTrace.from_jsonl(write_jsonl(td, []))


class ArrivalTest(unittest.TestCase):
    def test_spec_strings_give_the_generator_schedules(self):
        for seed in (0, 7, 42):
            self.assertEqual(parse_arrival("poisson:2.0", 50, seed), poisson_arrivals(50, 2.0, seed))
            self.assertEqual(parse_arrival("ramp:0.3:3.0", 50, seed), ramp_arrivals(50, 0.3, 3.0, seed))
        self.assertEqual(parse_arrival("steady", 3), [0.0, 0.0, 0.0])

    def test_unknown_spec_is_refused(self):
        with self.assertRaises(ValueError):
            parse_arrival("gamma:2", 4)

    def test_rebase_shifts_by_the_earliest_and_a_missing_time_is_zero(self):
        self.assertEqual(rebase_arrivals([5.0, 6.0, 9.0], speed=2.0), [0.0, 0.5, 2.0])
        self.assertEqual(rebase_arrivals([5.0, None, 7.0], speed=1.0), [5.0, 0.0, 7.0])
        self.assertEqual(rebase_arrivals([None, 3.0], speed=1.0), [0.0, 3.0])


class GridTraceTest(unittest.TestCase):
    CELLS = [(64, 2), (128, 3), (1000, 1)]
    STEPS = [10, 20, 40]

    def test_cells_over_maxlen_carry_no_requests(self):
        grid = GridTrace(self.CELLS, self.STEPS, "distinct", maxlen=1040)
        self.assertEqual(grid.cells_over_maxlen, [(1000, 1)])  # 1000 + 40 + 2 > 1040
        reqs = grid.requests()
        self.assertEqual(len(reqs), (2 + 3) * len(self.STEPS))
        self.assertEqual(sorted({r.output_length for r in reqs}), self.STEPS)
        self.assertEqual({r.prompt_len for r in reqs}, {64, 128})

    def test_prompt_modes(self):
        distinct, _ = GridTrace(self.CELLS, self.STEPS, "distinct", 4096).cell_prompts(64, 2)
        self.assertEqual(distinct[1][:3], [11, 12, 13])
        clone, _ = GridTrace(self.CELLS, self.STEPS, "clone", 4096).cell_prompts(64, 2)
        self.assertEqual(clone, [clone_ids(64)] * 2)
        with tempfile.TemporaryDirectory() as td:
            trace = RealTrace.from_jsonl(write_jsonl(td, [{"prompt_token_ids": [7] * 80}]))
        ids, cycled = GridTrace(self.CELLS, self.STEPS, "trace", 4096, trace=trace).cell_prompts(64, 2)
        self.assertTrue(cycled)
        self.assertEqual(ids, [[7] * 64] * 2)

    def test_corpus_windows_need_a_file(self):
        with self.assertRaises(ValueError):
            GridTrace(self.CELLS, self.STEPS, "corpus", 4096).cell_prompts(64, 1)
        with tempfile.TemporaryDirectory() as td:
            corpus = Path(td) / "c.txt"
            corpus.write_text("abcdef")
            grid = GridTrace(self.CELLS, self.STEPS, "corpus", 4096, corpus_path=corpus,
                             tokenizer=lambda: (lambda text: {"input_ids": [ord(c) for c in text]}))
            ids, _ = grid.cell_prompts(8, 1)
        self.assertEqual(ids, [[97, 98, 99, 100, 101, 102, 97, 98]])

    def test_grid_hash_covers_every_request(self):
        grid = GridTrace(self.CELLS, self.STEPS, "distinct", 4096)
        self.assertEqual(grid.workload_hash(), GridTrace(self.CELLS, self.STEPS, "distinct", 4096).workload_hash())
        self.assertNotEqual(grid.workload_hash(),
                            GridTrace(self.CELLS, [10, 20, 41], "distinct", 4096).workload_hash())


class SessionTraceTest(unittest.TestCase):
    def test_sessions_group_by_id_and_order_by_turn(self):
        with tempfile.TemporaryDirectory() as td:
            trace = SessionTrace.from_jsonl(write_jsonl(td, [
                {"prompt_token_ids": [1], "session_id": "b", "turn": 1},
                {"prompt_token_ids": [2], "session_id": "a", "turn": 0},
                {"prompt_token_ids": [3], "session_id": "b", "turn": 0}]))
        self.assertIsInstance(trace, SessionTrace)
        self.assertEqual([[r.prompt_token_ids[0] for r in s] for s in trace.sessions()],
                         [[3, 1], [2]])
        self.assertEqual(trace.pacing, TurnPacing())
        trace.pacing = TurnPacing(pacing="interleaved", tool_wait_ms=50.0)
        self.assertEqual(trace.source_meta()["pacing"], "interleaved")
        self.assertEqual(trace.source_meta()["sessions"], 2)


def write_rows(path, rows):
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
        path = write_rows(self.dir / "c.jsonl", [
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
        path = write_rows(self.dir / "m.jsonl", [
            {"timestamp": i, "input_length": 30 + i, "output_length": 3, "hash_ids": [i]}
            for i in range(5)])
        trace = MooncakeTrace(path).draw(5, warmup=3)
        first = trace.requests()[0]
        self.assertEqual(first.prompt_token_ids, trace.warmup_requests(1)[0].prompt_token_ids)
        self.assertEqual(first.prompt_len, 30)
        self.assertEqual(first.metadata["request_index"], 0)

    def test_a_wrapped_lap_follows_the_last_at_the_trace_pace(self):
        path = write_rows(self.dir / "m.jsonl", [
            {"timestamp": t, "input_length": 30, "output_length": 3, "hash_ids": [i]}
            for i, t in enumerate((1000, 3000, 5000))])
        trace = MooncakeTrace(path).draw(7, warmup=0)
        self.assertEqual([r.arrival_ts for r in trace.requests()],
                         [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0])

    def test_a_wrap_over_a_zero_span_trace_is_refused_under_arrival_trace(self):
        path = write_rows(self.dir / "m.jsonl", [
            {"timestamp": 1000, "input_length": 30, "output_length": 3, "hash_ids": [i]}
            for i in range(3)])
        for arrival, n, refused in (("trace", 4, True), ("trace", 3, False), ("steady", 4, False)):
            cfg = config("mooncake-trace")
            cfg = dataclasses.replace(cfg, workload=dataclasses.replace(
                cfg.workload, arrival=arrival, num_requests=n))
            with self.subTest(arrival=arrival, n=n), \
                    mock.patch.object(runner, "single_turn_trace",
                                      return_value=MooncakeTrace(path)):
                if refused:
                    with self.assertRaisesRegex(ConfigError, "span 0 ms"):
                        runner.build_trace(cfg)
                else:
                    self.assertEqual(len(runner.build_trace(cfg).requests()), n)

    def test_messages_and_metadata_never_reach_the_hash(self):
        plain = Request([1, 2], 2, order=0, output_length=4)
        chat = dataclasses.replace(plain, messages=[{"content": "x"}], metadata={"k": 1})
        self.assertEqual(chat.to_dict(), plain.to_dict())
        self.assertEqual(workload_hash([chat]), workload_hash([plain]))

    def test_build_names_the_profile(self):
        trace = runner.build_trace(config("test"))
        self.assertEqual(trace.source_meta(), {"profile": "test", "dataset": "test", "path": "",
                                               "reqs": 100})
        self.assertEqual(trace.requests()[0].max_tokens, 20)

    def test_multi_turn_profile_is_refused_single_turn(self):
        with self.assertRaisesRegex(ValueError, "is multi-turn; run it with workload.kind=sessions"):
            runner.build_trace(config("swebench-multiturn-short"))


class MultiTurn(unittest.TestCase):
    def build(self, **kw):
        sessions = []
        for s in range(3):
            turns = [{"turn_idx": t, "messages": [{"role": "user", "content": f"s{s} t{t}"}],
                      "osl_tokens": 2, "tool_ms": 5} for t in range(13 + s)]
            sessions.append({"session_id": f"S{s}", "arrival_time_ms": 100 * s, "turns": turns})
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        path = write_rows(Path(td.name) / "t.jsonl", sessions)
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
            runner.build_trace(config("test", kind="sessions"))


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


class RealTraceWorkloadTests(unittest.TestCase):
    def test_trace_request_id_encodes_benchmark_dimensions(self):
        request_id = make_trace_request_id(
            profile_name="swebench-multiturn-synth",
            concurrency=320,
            session_id=7,
            turn_index=2,
            request_index=647,
        )

        self.assertEqual(
            request_id,
            "agenticbench__p=swebench-multiturn-synth__c=320__t=2__s=7__i=647",
        )

    def test_sharegpt_dataset_reserves_output_and_safety_margin(self):
        fake_dataset = [
            {
                "conversations": [
                    {"from": "human", "value": words(100)},
                    {"from": "gpt", "value": words(40)},
                ],
            },
            {
                "conversations": [
                    {"from": "human", "value": words(70)},
                    {"from": "gpt", "value": words(30)},
                ],
            },
        ]
        fake_datasets_module = types.SimpleNamespace(
            load_dataset=lambda *args, **kwargs: fake_dataset
        )

        with patch.dict(sys.modules, {"datasets": fake_datasets_module}):
            trace = ShareGPTTrace(
                num_prompts=10,
                system_prompt="",
                max_isl_tokens=4096,
                max_osl_tokens=4096,
                min_osl_tokens=1,
                max_total_tokens=180,
                context_safety_margin_tokens=10,
            )

            request = trace.get_next_request()

        self.assertEqual(request.max_tokens, int(30 * 1.35))
        self.assertLessEqual(
            int(70 * 1.35) + request.max_tokens,
            180 - 10,
        )

    def test_sharegpt_dataset_uses_tokenizer_prompt_count_when_available(self):
        fake_dataset = [
            {
                "conversations": [
                    {"from": "human", "value": words(100)},
                    {"from": "gpt", "value": words(40)},
                ],
            },
            {
                "conversations": [
                    {"from": "human", "value": words(70)},
                    {"from": "gpt", "value": words(30)},
                ],
            },
        ]
        fake_datasets_module = types.SimpleNamespace(
            load_dataset=lambda *args, **kwargs: fake_dataset
        )

        class FakeTokenizer:
            def apply_chat_template(self, messages, add_generation_prompt, tokenize):
                content = " ".join(message["content"] for message in messages)
                return [0] * (180 if "w99" in content else 100)

        fake_transformers_module = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: FakeTokenizer()
            )
        )

        with patch.dict(sys.modules, {
            "datasets": fake_datasets_module,
            "transformers": fake_transformers_module,
        }):
            trace = ShareGPTTrace(
                num_prompts=10,
                system_prompt="",
                max_isl_tokens=4096,
                max_osl_tokens=4096,
                min_osl_tokens=1,
                max_total_tokens=210,
                context_safety_margin_tokens=10,
                tokenizer_name="fake-tokenizer",
            )

            self.assertEqual(len(trace.samples), 1)
            request = trace.get_next_request()

        self.assertEqual(request.max_tokens, int(30 * 1.35))

    def test_trajectory_dataset_reserves_output_and_safety_margin(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            session = {
                "session_id": "trace-1",
                "source": "unit-trace",
                "turns": [
                    {
                        "turn_idx": 0,
                        "messages": [{"role": "user", "content": words(50)}],
                        "osl_tokens": 20,
                    },
                    {
                        "turn_idx": 1,
                        "messages": [
                            {"role": "user", "content": words(50)},
                            {"role": "assistant", "content": words(20)},
                            {"role": "user", "content": words(20)},
                        ],
                        "osl_tokens": 20,
                    },
                ],
            }
            path.write_text(json.dumps(session) + "\n", encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()):
                trace = TrajectorySessionTrace(
                    str(path),
                    min_turns=1,
                    max_turns=2,
                    num_sessions=1,
                    max_isl_tokens=120,
                    max_osl_tokens=50,
                    context_safety_margin_tokens=10,
                )

            self.assertEqual(len(trace.sessions()), 1)
            turns = trace.sessions()[0]
            self.assertEqual(len(turns), 1)
            meta = turns[0].metadata
            self.assertEqual(meta["source_session_id"], "trace-1")
            self.assertEqual(meta["source_turn_index"], 0)
            self.assertEqual(meta["trace_content_source"], "unit-trace")
            self.assertLessEqual(
                meta["planned_total_with_output_tokens"],
                meta["context_window_tokens"] - meta["context_safety_margin_tokens"],
            )

    def test_build_gives_trajectories_the_profile_window_and_margin(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            path.write_text(
                json.dumps({
                    "session_id": "trace-2",
                    "source": "unit-trace",
                    "turns": [
                        {
                            "turn_idx": 0,
                            "messages": [{"role": "user", "content": words(50)}],
                            "osl_tokens": 20,
                        }
                    ],
                }) + "\n",
                encoding="utf-8",
            )
            profile = WorkloadProfile(
                name="fixture-real-trace",
                isl_tokens=4096,
                osl_tokens=50,
                isl_stddev=0.0,
                description="fixture",
                dataset="swebench-multi-turn",
                file_path=str(path),
                mode="multi-turn",
                min_turns=1,
                max_turns=2,
                num_sessions=1,
                agent_type="coding",
                turn_style="multi-turn",
                data_source="swebench",
            )
            cfg = Config(engine=EngineCfg(), workload=WorkloadCfg(kind="sessions", source=profile.name),
                         mode=ModeCfg(concurrency=1), hardware=HardwareCfg())

            with patch.dict(PROFILES, {profile.name: profile}), \
                    contextlib.redirect_stdout(io.StringIO()):
                trace = runner.build_trace(cfg)

            meta = trace.requests()[0].metadata
            self.assertEqual(meta["context_window_tokens"], 4096)
            self.assertEqual(meta["context_safety_margin_tokens"], 256)


class ProfileRegistryTests(unittest.TestCase):
    def test_profiles_are_the_yaml_entries(self):
        self.assertEqual(sorted(PROFILES), ["chat-multiturn-short", "chat-singleturn",
                                            "coding-singleturn", "mooncake-trace", "random-1k",
                                            "swebench-multiturn-short", "test"])

    def test_unknown_key_is_an_error(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "w.yaml"
            path.write_text("x:\n  isl_tokens: 1\n  osl_tokens: 1\n  isl_stddev: 0.0\n"
                            "  description: d\n  dataset: test\n  isl: 3\n")
            with self.assertRaisesRegex(ValueError, r"w.yaml: x: unknown profile key\(s\) \['isl'\]"):
                load_profiles(path)

    def test_site_packages_install_names_the_editable_fix(self):
        with tempfile.TemporaryDirectory() as td:
            shutil.copytree(Path(runner.__file__).parent, Path(td) / "site-packages" / "quettabench",
                            ignore=shutil.ignore_patterns("__pycache__"))
            proc = subprocess.run([sys.executable, "-c", "import quettabench.workloads.live"],
                                  cwd=td, env={**os.environ, "PYTHONPATH": f"{td}/site-packages"},
                                  capture_output=True, text=True, check=False)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("configs/workloads.yaml not found", proc.stderr)
        self.assertIn("pip install -e .", proc.stderr)

    def test_listings_hide_inactive_profiles(self):
        self.assertEqual(active_profiles(), ["chat-singleturn", "coding-singleturn",
                                             "mooncake-trace"])
        with self.assertRaises(ValueError) as e:
            get_profile("nope")
        self.assertIn(str(active_profiles()), str(e.exception))
        self.assertNotIn("random-1k", str(e.exception))

    def test_legacy_multiturn_profiles_remain_explicitly_runnable(self):
        self.assertEqual(get_profile("swebench-multiturn-short").dataset, "swebench-multi-turn")

    def test_multi_turn_num_sessions_uses_concurrency_floor(self):
        profile = get_profile("swebench-multiturn-short")

        effective, source = resolve_multi_turn_num_sessions(
            profile,
            profile.num_sessions + 1,
        )

        self.assertEqual(effective, profile.num_sessions + 1)
        self.assertEqual(source, "concurrency_floor")

    def test_multi_turn_num_sessions_keeps_profile_default_when_large_enough(self):
        profile = get_profile("chat-multiturn-short")

        effective, source = resolve_multi_turn_num_sessions(
            profile,
            max(1, profile.num_sessions - 1),
        )

        self.assertEqual(effective, profile.num_sessions)
        self.assertEqual(source, "profile_default")


SEED = 7
NUM_REQUESTS = 12
WARMUP = 3

# name: (workload_hash, content digest)
PINS = {
    "chat-multiturn-short": ("c51f134d3946f53d", "5e2d31c9032a7cb8"),
    "chat-singleturn": ("1b84966cb9c4facc", "84b2b4875f8a177f"),
    "coding-singleturn": ("15384d696a4d7a1b", "0359e7a0be1623a2"),
    "mooncake-trace": ("a03be7b0ed268d81", "b5ed59cd7a0d7aff"),
    "random-1k": ("00740b698acbf0f7", "e90e87f402815bd2"),
    "swebench-multiturn-short": ("9f0aa9d6f98e5bd3", "2e44a22e09c81ea1"),
    "test": ("7ae706bdf8eeae1d", "a5c809ace7fb7a08"),
    "cut": ("0e8b8a8de4442ecb", "c8f75cbcdd1e78dc"),
}


def tagged_words(n, tag):
    return " ".join(f"{tag}{i}" for i in range(n))


def sharegpt_rows():
    rows = [{"conversations": [{"from": "human", "value": "lonely"}]},
            {"conversations": [{"from": "gpt", "value": tagged_words(80, "g")},
                               {"from": "human", "value": tagged_words(30, "h")}]}]
    for k in range(40):
        convs = []
        for t in range(1 + k % 12):
            convs.append({"from": "human", "value": tagged_words(20 + (k * 37 + t * 11) % 600, f"h{k}.{t}.")})
            convs.append({"from": "gpt", "value": tagged_words(30 + (k * 53 + t * 7) % 900, f"g{k}.{t}.")})
        rows.append({"conversations": convs})
    return rows


def trajectory_rows(tag):
    rows = []
    for k, n in enumerate((2, 5, 12, 15, 25, 35, 70, 90, 160, 9, 22)):
        turns, history = [], []
        for t in range(n):
            history = history + [{"role": "user", "content": tagged_words(5 + (k + t) % 40, f"{tag}{k}.{t}.")}]
            turn = {"messages": list(history), "osl_tokens": 50 + (k * 13 + t) % 3000}
            if t % 3 == 0:
                turn["turn_idx"] = t
            if t % 4 == 1:
                turn["tool_ms"] = 10 * t
            if t % 5 == 2:
                turn["human_wait_after_ms"] = 3 * t
            turns.append(turn)
        row = {"session_id": f"{tag}-{k}", "source": tag, "turns": turns}
        if k % 2:
            row["arrival_time_ms"] = 250 * k
        rows.append(row)
    return rows


def write_fixtures(root):
    files = {
        "data/coding_agent_prompts.jsonl": [
            {"system": f"sys{i}", "user": tagged_words(10 + i, f"u{i}."), "osl_tokens": 100 + i}
            for i in range(5)],
        "data/mooncake/conversation_trace.jsonl": [
            {"timestamp": 10 * i, "input_length": 700 + 300 * i, "output_length": 5 + i,
             "hash_ids": list(range(i % 3, i % 3 + 1 + i))} for i in range(4)],
        "data/swebench_trajectories.jsonl": trajectory_rows("swe"),
    }
    for rel, rows in files.items():
        path = Path(root) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))


class FakeTokenizer:
    """Lossy on purpose: encode drops ids divisible by 7, so the random-token profiles pad with their rng."""

    vocab_size = 1000

    def decode(self, ids):
        return " ".join(str(int(i)) for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [int(t) for t in text.split() if t.isdigit() and int(t) % 7]

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"
        return [0] * len(text.split()) if tokenize else text


FAKE_MODULES = {
    "datasets": types.SimpleNamespace(load_dataset=lambda *a, **k: sharegpt_rows()),
    "transformers": types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeTokenizer())),
}

MULTI_TURN = ("sharegpt-multi-turn", "swebench-multi-turn")


def profile_config(name, kind, max_turn_index=None):
    return Config(engine=EngineCfg(model="fake-model"), hardware=HardwareCfg(),
                  workload=WorkloadCfg(kind=kind, source=name, seed=SEED,
                                       num_requests=NUM_REQUESTS, max_turn_index=max_turn_index),
                  mode=ModeCfg(concurrency=4, warmup=WARMUP))


def row(r):
    return [r.prompt_token_ids, r.prompt_len, r.order, r.output_length, r.max_tokens,
            r.session_id, r.turn, r.arrival_ts, r.seed, r.source, r.messages, r.metadata]


def digest(trace):
    record = {
        "requests": [row(r) for r in trace.requests()],
        "warmup": [row(r) for r in trace.warmup_requests(WARMUP)],
        "source_meta": trace.source_meta(),
        "max_turns": getattr(trace, "max_turns", None),
        "replies": {str(k): v for k, v in getattr(trace, "replies", {}).items()},
        "sessions": [[r.order for r in s] for s in trace.sessions()]
        if hasattr(trace, "sessions") else None,
    }
    return hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()[:16]


@contextlib.contextmanager
def fixture_cwd():
    old = os.getcwd()
    with tempfile.TemporaryDirectory() as td, mock.patch.dict(sys.modules, FAKE_MODULES):
        write_fixtures(td)
        os.chdir(td)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                yield
        finally:
            os.chdir(old)


def pinned_build(name, max_turn_index=None):
    kind = "sessions" if get_profile(name).dataset in MULTI_TURN else "live"
    trace = runner.build_trace(profile_config(name, kind, max_turn_index))
    return trace.workload_hash(), digest(trace)


def runnable():
    return sorted(PROFILES)


class ProfileHashPins(unittest.TestCase):
    def test_every_runnable_profile_is_pinned(self):
        self.assertEqual(sorted(PINS), sorted(runnable() + ["cut"]))

    def test_profiles_build_the_pinned_traces(self):
        with fixture_cwd():
            got = {name: pinned_build(name) for name in runnable()}
            got["cut"] = pinned_build("swebench-multiturn-short", max_turn_index=3)
        for name, pin in PINS.items():
            with self.subTest(profile=name):
                self.assertEqual(got[name], pin)


if __name__ == "__main__":
    unittest.main()
