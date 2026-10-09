"""InProcessDriver builds vLLM from engine_kwargs only when asked, and
times a grid batch as one generate() call."""

import types
import unittest
from unittest import mock

from quettabench.engines import drivers
from quettabench.engines.base import LaunchSpec
from quettabench.engines.drivers import InProcessDriver
from quettabench.workloads.base import Request
from tests.golden._harness import crossval_path, fresh_import

KWARGS = {"model": "m", "tensor_parallel_size": 2}
CELL = {"ctx": 8, "bs": 3, "tp": 1, "step_point": 5, "phase": "fit", "attempt": 0}


class Builders(unittest.TestCase):
    def test_llm_and_async_engine_get_the_same_kwargs(self):
        with crossval_path(stubs=True):
            vllm = fresh_import("vllm")
            vllm.reset()
            driver = InProcessDriver(LaunchSpec(adapter="vllm", engine_kwargs=KWARGS))
            self.assertEqual(vllm.CALLS, [])
            driver.llm()
            with self.assertRaises(vllm.StubHalt):
                driver.async_engine()
            self.assertEqual(vllm.CALLS, [("LLM", KWARGS), ("AsyncEngineArgs", KWARGS)])

    def test_llm_is_built_once(self):
        with crossval_path(stubs=True):
            vllm = fresh_import("vllm")
            vllm.reset()
            driver = InProcessDriver(LaunchSpec(adapter="vllm", engine_kwargs=KWARGS))
            self.assertIs(driver.llm(), driver.llm())
            self.assertEqual(vllm.CALLS, [("LLM", KWARGS)])


class GridBatch(unittest.TestCase):
    def run_batch(self, seconds):
        with crossval_path(stubs=True):
            vllm = fresh_import("vllm")
            vllm.reset()
            vllm.generate_seconds = lambda prompt_len, batch, max_tokens, call_no: seconds
            driver = InProcessDriver(LaunchSpec(adapter="vllm", engine_kwargs=KWARGS))
            requests = [Request([7] * 8, 8, output_length=5) for _ in range(3)]
            clock = types.SimpleNamespace(perf_counter=vllm.CLOCK.perf_counter)
            with mock.patch.object(drivers, "time", clock):
                driver.warmup()
                rows = list(driver.run(requests, cell=CELL))
            return rows, vllm.CLOCK.generate_calls

    def test_one_generate_shared_wall(self):
        rows, calls = self.run_batch(0.25)
        self.assertEqual(calls, 2)  # warmup + the batch
        self.assertEqual(len(rows), 3)
        for r in rows:
            self.assertEqual((r.e2el, r.batch_wall_s, r.ttft), (0.25, 0.25, None))
            self.assertEqual((r.output_tokens, r.input_tokens, r.ttft_basis), (5, 8, "batch"))
            self.assertEqual(r.cell, CELL)
            self.assertEqual(r.generated_token_ids, [0, 1, 2, 3, 4])
        self.assertIsNot(rows[0].cell, CELL)


if __name__ == "__main__":
    unittest.main()
