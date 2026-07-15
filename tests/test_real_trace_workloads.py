import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.benchmark.runner import make_trace_request_id
from src.workloads.dataset import ShareGPTDataset, TrajectoryMultiTurnDataset, make_dataset
from src.workloads.profiles import WorkloadProfile


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
            dataset = ShareGPTDataset(
                num_prompts=10,
                system_prompt="",
                max_isl_tokens=4096,
                max_osl_tokens=4096,
                min_osl_tokens=1,
                max_total_tokens=180,
                context_safety_margin_tokens=10,
            )

            request = dataset.get_next_request()

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
            dataset = ShareGPTDataset(
                num_prompts=10,
                system_prompt="",
                max_isl_tokens=4096,
                max_osl_tokens=4096,
                min_osl_tokens=1,
                max_total_tokens=210,
                context_safety_margin_tokens=10,
                tokenizer_name="fake-tokenizer",
            )
            dataset._load()

            self.assertEqual(len(dataset._samples), 1)
            request = dataset.get_next_request()

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

            dataset = TrajectoryMultiTurnDataset(
                filepath=str(path),
                min_turns=1,
                max_turns=2,
                num_sessions=1,
                max_isl_tokens=120,
                max_osl_tokens=50,
                context_safety_margin_tokens=10,
            )

            self.assertEqual(len(dataset.sessions), 1)
            turns = dataset.sessions[0].turns
            self.assertEqual(len(turns), 1)
            meta = turns[0].metadata
            self.assertEqual(meta["source_session_id"], "trace-1")
            self.assertEqual(meta["source_turn_index"], 0)
            self.assertEqual(meta["trace_content_source"], "unit-trace")
            self.assertLessEqual(
                meta["planned_total_with_output_tokens"],
                meta["context_window_tokens"] - meta["context_safety_margin_tokens"],
            )

    def test_make_dataset_applies_context_cap_to_real_trace_profiles(self):
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

            dataset = make_dataset(
                profile,
                max_context_tokens=120,
                context_safety_margin_tokens=10,
            )

            meta = dataset.sessions[0].turns[0].metadata
            self.assertEqual(meta["context_window_tokens"], 120)
            self.assertEqual(meta["context_safety_margin_tokens"], 10)


if __name__ == "__main__":
    unittest.main()
