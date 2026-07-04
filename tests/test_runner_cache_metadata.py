import unittest
import contextlib
import io
import json
import tempfile
from pathlib import Path

from src.benchmark.metrics import (
    BenchmarkSummary,
    RequestResult,
    aggregate_per_turn,
    annotate_multi_turn_cache_estimate,
    annotate_request_observability,
)
from src.benchmark.runner import (
    BENCHMARK_SCHEMA_VERSION,
    _resolve_cache_state,
    _resolve_tri_state,
    resolve_multi_turn_num_sessions,
    save_results,
)
from src.workloads.dataset import BenchmarkRequest
from src.workloads.profiles import get_profile


class RunnerCacheMetadataTests(unittest.TestCase):
    def test_annotates_successful_multi_turn_request(self):
        result = RequestResult(success=True, input_tokens=250, output_tokens=40)

        annotate_multi_turn_cache_estimate(
            result,
            session_id=12,
            turn_index=2,
            previous_context_tokens=175,
        )

        self.assertEqual(result.session_id, 12)
        self.assertEqual(result.turn_index, 2)
        self.assertEqual(result.previous_context_tokens, 175)
        self.assertEqual(result.total_context_tokens, 250)
        self.assertEqual(result.cached_context_tokens, 175)
        self.assertEqual(result.new_prefill_tokens, 75)
        self.assertAlmostEqual(result.cache_hit_rate, 0.7)
        self.assertEqual(result.cache_estimate_source, "previous_prompt_tokens")

    def test_block_aligned_cache_estimate_accounts_for_uncached_tail(self):
        result = RequestResult(success=True, input_tokens=250, output_tokens=40)

        annotate_multi_turn_cache_estimate(
            result,
            session_id=12,
            turn_index=2,
            previous_context_tokens=175,
            cache_block_size=16,
        )

        self.assertEqual(result.cache_block_size, 16)
        self.assertEqual(result.block_aligned_cached_context_tokens, 160)
        self.assertEqual(result.uncached_prefix_tail_tokens, 15)
        self.assertEqual(result.block_aligned_new_prefill_tokens, 90)
        self.assertAlmostEqual(result.block_aligned_cache_hit_rate, 160 / 250)
        self.assertEqual(result.total_context_blocks, 16)
        self.assertEqual(result.cached_context_blocks, 10)
        self.assertEqual(result.new_prefill_blocks, 6)

    def test_cache_estimate_clamps_previous_context_to_current_prompt(self):
        result = RequestResult(success=True, input_tokens=128, output_tokens=40)

        annotate_multi_turn_cache_estimate(
            result,
            session_id=1,
            turn_index=3,
            previous_context_tokens=256,
            cache_block_size=16,
        )

        self.assertEqual(result.cached_context_tokens, 128)
        self.assertEqual(result.new_prefill_tokens, 0)
        self.assertEqual(result.cache_hit_rate, 1.0)
        self.assertEqual(result.block_aligned_cached_context_tokens, 128)
        self.assertEqual(result.block_aligned_new_prefill_tokens, 0)
        self.assertEqual(result.new_prefill_blocks, 0)

    def test_cache_estimate_skips_block_fields_when_block_size_missing_or_invalid(self):
        for block_size in (None, 0, -8):
            with self.subTest(block_size=block_size):
                result = RequestResult(success=True, input_tokens=128, output_tokens=40)
                annotate_multi_turn_cache_estimate(
                    result,
                    session_id=1,
                    turn_index=1,
                    previous_context_tokens=64,
                    cache_block_size=block_size,
                )

                self.assertEqual(result.cached_context_tokens, 64)
                self.assertEqual(result.new_prefill_tokens, 64)
                self.assertIsNone(result.block_aligned_cached_context_tokens)
                self.assertIsNone(result.block_aligned_new_prefill_tokens)

    def test_failed_request_gets_identity_fields_but_no_cache_math(self):
        result = RequestResult(success=False, input_tokens=250, error="timeout")

        annotate_multi_turn_cache_estimate(
            result,
            session_id=9,
            turn_index=4,
            previous_context_tokens=200,
            cache_block_size=16,
        )

        self.assertEqual(result.session_id, 9)
        self.assertEqual(result.turn_index, 4)
        self.assertEqual(result.previous_context_tokens, 200)
        self.assertEqual(result.cache_estimate_source, "unavailable")
        self.assertIsNone(result.total_context_tokens)
        self.assertIsNone(result.block_aligned_cached_context_tokens)

    def test_request_observability_records_shape_timing_and_metadata(self):
        request = BenchmarkRequest(
            messages=[
                {"role": "system", "content": "s"},
                {"role": "user", "content": "hello"},
            ],
            max_tokens=32,
            metadata={"planned_new_prefill_tokens": 17},
        )
        result = RequestResult(success=True)

        annotate_request_observability(
            result,
            request_index=5,
            request=request,
            scheduled_at_s=1.0,
            dispatch_started_at_s=1.1,
            semaphore_acquired_at_s=1.4,
            completed_at_s=2.0,
        )

        self.assertEqual(result.request_index, 5)
        self.assertEqual(result.max_tokens_requested, 32)
        self.assertEqual(result.message_count, 2)
        self.assertEqual(result.prompt_chars, 6)
        self.assertAlmostEqual(result.client_schedule_delay_s, 0.1)
        self.assertAlmostEqual(result.client_queue_wait_s, 0.3)
        self.assertAlmostEqual(result.client_request_wall_s, 0.6)
        self.assertEqual(result.request_metadata["planned_new_prefill_tokens"], 17)

    def test_request_observability_allows_unscheduled_multiturn_requests(self):
        request = BenchmarkRequest(messages=[{"role": "user", "content": "hello"}], max_tokens=8)
        result = RequestResult(success=True)

        annotate_request_observability(
            result,
            request_index=0,
            request=request,
            scheduled_at_s=None,
            dispatch_started_at_s=2.0,
            semaphore_acquired_at_s=2.25,
            completed_at_s=2.75,
        )

        self.assertIsNone(result.scheduled_at_s)
        self.assertIsNone(result.client_schedule_delay_s)
        self.assertAlmostEqual(result.client_queue_wait_s, 0.25)
        self.assertAlmostEqual(result.client_request_wall_s, 0.5)

    def test_per_turn_summary_includes_cache_estimate_medians(self):
        first = annotate_multi_turn_cache_estimate(
            RequestResult(success=True, input_tokens=100, output_tokens=20),
            session_id=0,
            turn_index=0,
            previous_context_tokens=0,
            cache_block_size=16,
        )
        second = annotate_multi_turn_cache_estimate(
            RequestResult(success=True, input_tokens=260, output_tokens=30),
            session_id=1,
            turn_index=0,
            previous_context_tokens=200,
            cache_block_size=16,
        )

        summaries = aggregate_per_turn({0: [first, second]})

        self.assertEqual(len(summaries), 1)
        summary = summaries[0]
        self.assertEqual(summary.median_input_tokens, 180)
        self.assertEqual(summary.median_output_tokens, 25)
        self.assertEqual(summary.median_new_prefill_tokens, 80)
        self.assertEqual(summary.median_cached_context_tokens, 100)
        self.assertAlmostEqual(summary.median_cache_hit_rate, (0.0 + 200 / 260) / 2)
        self.assertEqual(summary.median_block_aligned_cached_context_tokens, 96)
        self.assertEqual(summary.median_block_aligned_new_prefill_tokens, 84)
        self.assertAlmostEqual(
            summary.median_block_aligned_cache_hit_rate,
            (0.0 + 192 / 260) / 2,
        )

    def test_per_turn_summary_preserves_nonzero_cache_estimates_from_run_results(self):
        result = annotate_multi_turn_cache_estimate(
            RequestResult(success=True, input_tokens=1531, output_tokens=9),
            session_id=0,
            turn_index=1,
            previous_context_tokens=1416,
            cache_block_size=16,
        )

        summary = aggregate_per_turn({1: [result]})[0]

        self.assertEqual(summary.avg_cached_context_tokens, 1416)
        self.assertEqual(summary.median_cached_context_tokens, 1416)
        self.assertEqual(summary.avg_new_prefill_tokens, 115)
        self.assertEqual(summary.median_new_prefill_tokens, 115)
        self.assertAlmostEqual(summary.avg_cache_hit_rate, 1416 / 1531)
        self.assertAlmostEqual(summary.median_cache_hit_rate, 1416 / 1531)
        self.assertEqual(summary.avg_block_aligned_cached_context_tokens, 1408)
        self.assertEqual(summary.median_block_aligned_cached_context_tokens, 1408)

    def test_per_turn_summary_includes_client_queue_wait_median(self):
        first = RequestResult(success=True, input_tokens=100, output_tokens=20)
        first.client_queue_wait_s = 0.010
        second = RequestResult(success=True, input_tokens=100, output_tokens=20)
        second.client_queue_wait_s = 0.030

        summary = aggregate_per_turn({0: [first, second]})[0]

        self.assertEqual(summary.median_client_queue_wait_ms, 20)

    def test_save_results_serializes_prediction_observability_fields(self):
        result = RequestResult(
            success=True,
            ttft=0.01234,
            itl=[0.004, 0.006],
            e2el=0.030,
            input_tokens=250,
            output_tokens=3,
        )
        result = annotate_request_observability(
            result,
            request_index=7,
            request=BenchmarkRequest(
                messages=[{"role": "user", "content": "hello world"}],
                max_tokens=64,
                metadata={"planned_new_prefill_tokens": 75},
            ),
            scheduled_at_s=1.0,
            dispatch_started_at_s=1.2,
            semaphore_acquired_at_s=1.5,
            completed_at_s=1.8,
        )
        annotate_multi_turn_cache_estimate(
            result,
            session_id=3,
            turn_index=2,
            previous_context_tokens=175,
            cache_block_size=16,
        )
        summary = BenchmarkSummary(
            model="meta-llama/Llama-3.1-8B-Instruct",
            profile="chat-multiturn",
            concurrency=80,
            num_requests=1,
            successful_requests=1,
        )
        config = {
            "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
            "profile": "chat-multiturn",
            "prediction_metadata": {"prefix_cache_block_size": 16},
        }

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "result.json"
            with contextlib.redirect_stdout(io.StringIO()):
                save_results(summary, [result], str(out), config)
            data = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(data["config"]["benchmark_schema_version"], BENCHMARK_SCHEMA_VERSION)
        self.assertEqual(data["config"]["prediction_metadata"]["prefix_cache_block_size"], 16)
        row = data["per_request"][0]
        self.assertEqual(row["request_index"], 7)
        self.assertEqual(row["max_tokens_requested"], 64)
        self.assertEqual(row["message_count"], 1)
        self.assertEqual(row["session_id"], 3)
        self.assertEqual(row["turn_index"], 2)
        self.assertEqual(row["block_aligned_cached_context_tokens"], 160)
        self.assertEqual(row["block_aligned_new_prefill_tokens"], 90)
        self.assertEqual(row["request_metadata"]["planned_new_prefill_tokens"], 75)
        self.assertAlmostEqual(row["client_queue_wait_ms"], 300.0)

    def test_cache_state_metadata_resolution(self):
        chat = get_profile("chat-multiturn")
        random = get_profile("random-1k")

        self.assertEqual(_resolve_cache_state("auto", chat), ("expected_on", "profile_default"))
        self.assertEqual(_resolve_cache_state("auto", random), ("expected_off", "profile_default"))
        self.assertEqual(_resolve_cache_state("off", chat), ("off", "cli"))

    def test_tri_state_metadata_resolution(self):
        self.assertEqual(_resolve_tri_state("auto"), ("unknown", "not_reported"))
        self.assertEqual(_resolve_tri_state("on"), ("on", "cli"))

    def test_synthetic_multiturn_sessions_floor_at_concurrency(self):
        profile = get_profile("swebench-multiturn-synth")

        sessions, source = resolve_multi_turn_num_sessions(profile, concurrency=320)

        self.assertEqual(sessions, 320)
        self.assertEqual(source, "concurrency_floor")


if __name__ == "__main__":
    unittest.main()
