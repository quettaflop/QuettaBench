"""InProcessDriver builds vLLM from engine_kwargs only when asked."""

import unittest

from quettabench.engines.base import LaunchSpec
from quettabench.engines.drivers import InProcessDriver
from tests.golden._harness import crossval_path, fresh_import

KWARGS = {"model": "m", "tensor_parallel_size": 2}


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


if __name__ == "__main__":
    unittest.main()
