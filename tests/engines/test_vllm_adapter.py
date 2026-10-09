"""translate(cfg).engine_kwargs equals vmin_fit's golden LLM kwargs for every
run yaml, and server_argv is the api_server argv the launch scripts use.

Documented spelling difference: translate always passes
pipeline_parallel_size; vmin_fit left it out at pp == 1.
"""

import json
import unittest

from quettabench.config import loader
from quettabench.engines import base
from quettabench.engines import vllm as vllm_adapter
from tests.golden._harness import ENV_OVERRIDE, FIXTURES, REPO

GOLDEN = json.loads((FIXTURES / "engine_args.json").read_text())
RUNS = REPO / "configs" / "runs"
PINNED = ["engine.model=<MODEL>", "allow_unverified=true",
          "hardware.backend=cpu", "hardware.link_profile=nvlink"]


# The loader refuses qwen (hybrid without kv_bytes, a capacity-gate fact);
# kv_bytes is not a launch field, so any value unblocks the snapshot.
UNBLOCK = {"qwen": ["engine.kv_bytes=1"]}


def spec_for(name, env=None, sets=()):
    sets = [*PINNED, *UNBLOCK.get(name, ()), *sets]
    (cfg,) = loader.load(RUNS / f"{name}.yaml", sets, env=env or {})
    return vllm_adapter.ADAPTER.translate(cfg)


def with_pp(kwargs):
    return {"pipeline_parallel_size": 1, **kwargs}


class KwargsMatchVminFit(unittest.TestCase):
    def test_every_run_yaml(self):
        for name in sorted(GOLDEN):
            with self.subTest(name=name):
                self.assertEqual(spec_for(name).engine_kwargs,
                                 with_pp(GOLDEN[name]["vmin_fit"]))

    def test_env_override(self):
        for name in sorted(GOLDEN):
            with self.subTest(name=name):
                self.assertEqual(spec_for(name, env=ENV_OVERRIDE).engine_kwargs,
                                 GOLDEN[name]["vmin_fit_env_override"])

    def test_nograph(self):
        for name in sorted(GOLDEN):
            with self.subTest(name=name):
                spec = spec_for(name, sets=["engine.cudagraph=false"])
                self.assertEqual(spec.engine_kwargs, with_pp(GOLDEN[name]["vmin_fit_nograph"]))
                self.assertIn("cudagraph=False", spec.server_unsupported)


class ServerArgv(unittest.TestCase):
    def test_llama_is_the_launch_script_argv(self):
        spec = spec_for("llama")
        self.assertEqual(list(spec.server_argv), [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", "<MODEL>", "--port", "8000", "--tensor-parallel-size", "1",
            "--dtype", "float16", "--max-model-len", "9216",
            "--gpu-memory-utilization", "0.9",
            "--enable-chunked-prefill", "--trust-remote-code"])
        self.assertEqual(spec.env, {"VLLM_SERVER_DEV_MODE": "1",
                                    "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1"})
        self.assertEqual((spec.health_path, spec.port), ("/health", 8000))

    def test_fields_without_a_flag_block_the_server_only(self):
        self.assertEqual(spec_for("deepseek").server_unsupported, ("kv_dtype=fp8_ds_mla",))
        self.assertEqual(spec_for("glm53").server_unsupported, ("quant=fp8",))
        self.assertEqual(spec_for("deepseek").server_argv, ())

    def test_expert_parallel_and_prefix_caching_switches(self):
        argv = spec_for("glm53flash", sets=["engine.quant=null", "engine.prefix_caching=true"]
                        ).server_argv
        self.assertIn("--enable-expert-parallel", argv)
        self.assertIn("--enable-prefix-caching", argv)

    def test_nsys_prefix_is_prof_sh(self):
        spec = spec_for("llama", sets=["engine.instrumentation=nsys", "run_name=r1"])
        self.assertEqual(list(spec.server_argv[:7]), [
            "nsys", "profile", "-t", "cuda,nvtx", "-o", "results/r1/nsys",
            "--force-overwrite=true"])

    def test_torch_instrumentation_is_unsupported(self):
        with self.assertRaises(base.UnsupportedField):
            spec_for("llama", sets=["engine.instrumentation=torch"])

    def test_disagg_is_unsupported(self):
        with self.assertRaises(base.UnsupportedField):
            spec_for("llama", sets=["engine.serving_style=disagg:1p1d"])


class Extra(unittest.TestCase):
    def test_scalar_extra_appended_last_on_both_sides(self):
        spec = spec_for("llama", sets=["engine.extra.vllm={gdn_prefill_backend: triton}"])
        self.assertEqual(list(spec.server_argv[-2:]), ["--gdn-prefill-backend", "triton"])
        self.assertEqual(spec.engine_kwargs["gdn_prefill_backend"], "triton")

    def test_bad_shape_raises(self):
        for value in ("{speculative_config: {k: 1}}", "{tp: [1, 2]}", "notamapping",
                      "{tensor_parallel_size: 2}"):
            with self.subTest(value=value), self.assertRaises(base.UnsupportedField):
                spec_for("llama", sets=[f"engine.extra.vllm={value}"])


if __name__ == "__main__":
    unittest.main()
