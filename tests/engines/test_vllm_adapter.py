"""translate(cfg).engine_kwargs equals vmin_fit's golden LLM kwargs (captured
before vmin_fit.py retired in 9f9e0ee) for every run yaml, and server_argv is the `vllm serve <model>` form the H200 campaign
harness (v4sweep.py) launched.

translate always passes pipeline_parallel_size; vmin_fit used to leave it
out at pp == 1, a spelling difference only (golden regenerated with it).
"""

import json
import unittest
from pathlib import Path

from quettabench.config import loader
from quettabench.engines import base, launcher, vllm as vllm_adapter
from quettabench.engines.drivers import HttpDriver, InProcessDriver
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


class KwargsMatchVminFit(unittest.TestCase):
    def test_every_run_yaml(self):
        for name in sorted(GOLDEN):
            with self.subTest(name=name):
                self.assertEqual(spec_for(name).engine_kwargs,
                                 GOLDEN[name]["vmin_fit"])

    def test_env_override(self):
        for name in sorted(GOLDEN):
            with self.subTest(name=name):
                self.assertEqual(spec_for(name, env=ENV_OVERRIDE).engine_kwargs,
                                 GOLDEN[name]["vmin_fit_env_override"])

    def test_nograph(self):
        for name in sorted(GOLDEN):
            with self.subTest(name=name):
                spec = spec_for(name, sets=["engine.cudagraph=false"])
                self.assertEqual(spec.engine_kwargs, GOLDEN[name]["vmin_fit_nograph"])
                self.assertIn("cudagraph=False", spec.server_unsupported)


class ServerArgv(unittest.TestCase):
    def test_llama_is_the_vllm_serve_argv(self):
        spec = spec_for("llama")
        self.assertEqual(list(spec.server_argv), [
            "vllm", "serve", "<MODEL>", "--port", "8000", "--tensor-parallel-size", "1",
            "--dtype", "float16", "--max-model-len", "9216",
            "--gpu-memory-utilization", "0.9",
            "--enable-chunked-prefill", "--trust-remote-code"])
        self.assertEqual(spec.env, {"VLLM_SERVER_DEV_MODE": "1",
                                    "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1"})
        self.assertEqual((spec.health_path, spec.port), ("/health", 8000))

    def test_fields_without_a_flag_block_the_server_only(self):
        blocked = spec_for("glm53", sets=["engine.pp=2"])
        self.assertEqual(blocked.server_unsupported, ("pp=2",))
        self.assertEqual(blocked.server_argv, ())
        self.assertEqual(blocked.engine_kwargs["pipeline_parallel_size"], 2)

    def test_quant_is_read_from_the_checkpoint_on_the_server(self):
        spec = spec_for("glm53")
        self.assertEqual(spec.server_unsupported, ())
        self.assertNotIn("--quantization", spec.server_argv)
        self.assertEqual(spec.engine_kwargs["quantization"], "fp8")

    def test_kv_dtype_is_kv_cache_dtype(self):
        argv = list(spec_for("deepseek").server_argv)
        self.assertEqual(argv[argv.index("--kv-cache-dtype") + 1], "fp8_ds_mla")
        self.assertNotIn("--kv-cache-dtype", spec_for("llama").server_argv)

    def test_provisioned_venv_runs_its_own_vllm(self):
        spec = spec_for("llama", sets=["hardware.provision.py=/venv/bin/python3"])
        self.assertEqual(list(spec.server_argv[:2]), ["/venv/bin/vllm", "serve"])

    def test_h200_campaign_flags(self):
        """v4sweep.py glm_tep: kv fp8, EP, seq and batched-token caps via extra.vllm."""
        argv = list(spec_for("llama", sets=[
            "engine.kv_dtype=fp8", "engine.ep=true", "engine.gpu_util=0.95",
            "engine.extra.vllm={max_num_seqs: 32, max_num_batched_tokens: 4096}"]).server_argv)
        for pair in (["--kv-cache-dtype", "fp8"], ["--gpu-memory-utilization", "0.95"],
                     ["--max-num-seqs", "32"], ["--max-num-batched-tokens", "4096"]):
            i = argv.index(pair[0])
            self.assertEqual(argv[i:i + 2], pair)
        self.assertIn("--enable-expert-parallel", argv)

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

    def test_afd_is_unsupported(self):
        with self.assertRaisesRegex(base.UnsupportedField, "serving_style=afd:2a6f"):
            spec_for("llama", sets=["engine.serving_style=afd:2a6f"])


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


class GridDriver(unittest.TestCase):
    """engine.driver picks how a grid run reaches vLLM; in-process stays the default."""

    def cfg(self, *sets):
        (cfg,) = loader.load(RUNS / "llama.yaml", [*PINNED, *sets], env={})
        return cfg

    def test_default_grid_is_in_process(self):
        cfg = self.cfg()
        adapter = vllm_adapter.ADAPTER
        self.assertFalse(adapter.needs_server(cfg))
        self.assertIsInstance(adapter.driver(adapter.translate(cfg), launcher.NoServer(Path("r"))),
                              InProcessDriver)

    def test_http_grid_launches_the_server(self):
        cfg = self.cfg("engine.driver=http")
        adapter = vllm_adapter.ADAPTER
        spec = adapter.translate(cfg)
        self.assertTrue(adapter.needs_server(cfg))
        self.assertEqual(spec.server_argv, spec_for("llama").server_argv)
        handle = launcher.NoServer(Path("r"), endpoint="http://127.0.0.1:8000")
        driver = adapter.driver(spec, handle)
        self.assertIsInstance(driver, HttpDriver)
        self.assertEqual(driver.url, "http://127.0.0.1:8000/v1/completions")

    def test_http_grid_returns_no_text_or_logits(self):
        for returns in ("text", "logits"):
            with self.subTest(returns=returns), self.assertRaises(base.UnsupportedField):
                spec_for("llama", sets=["engine.driver=http", f"mode.returns={returns}"])


if __name__ == "__main__":
    unittest.main()
