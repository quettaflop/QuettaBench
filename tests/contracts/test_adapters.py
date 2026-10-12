"""Engine adapters: LaunchSpec data and the vllm and quettaserve translations."""

import contextlib
import dataclasses
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from quettabench import cli
from quettabench.config import loader
from quettabench.config.schema import ConfigError
from quettabench.engines import base, launcher, quettaserve, vllm as vllm_adapter
from quettabench.engines.drivers import HttpDriver, InProcessDriver
from quettabench.profiling import native
from quettabench.workloads.base import Request
from quettabench.workloads.grid import GridTrace
from tests.golden._harness import ENV_OVERRIDE, FIXTURES, REPO


class LaunchSpecTests(unittest.TestCase):
    def test_frozen(self):
        spec = base.LaunchSpec(adapter="x")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            spec.port = 1

    def test_child_argv_sets_and_drops_env(self):
        spec = base.LaunchSpec(adapter="x", env={"A": "1", "B": "two"}, env_drop=("C",))
        self.assertEqual(spec.child_argv(["run", "--x"]), ["env", "-u", "C", "A=1", "B=two", "run", "--x"])

    def test_unknown_engine(self):
        with self.assertRaisesRegex(ValueError, "unknown engine 'nope'"):
            base.get("nope")

    def test_unsupported_field_names_engine_and_field(self):
        err = base.UnsupportedField("vllm", "kv_dtype", "no flag")
        self.assertEqual((err.engine, err.field), ("vllm", "kv_dtype"))
        self.assertIn("kv_dtype", str(err))


GOLDEN = json.loads((FIXTURES / "engine_args.json").read_text())
RUNS = REPO / "configs" / "runs"
PINNED = ["engine.model=<MODEL>", "allow_unverified=true", "hardware.backend=cpu", "hardware.link_profile=nvlink"]


# The loader refuses qwen without kv_bytes; it is not a launch field, so any value unblocks the snapshot.
UNBLOCK = {"qwen": ["engine.kv_bytes=1"]}


def spec_for(name, env=None, sets=()):
    sets = [*PINNED, *UNBLOCK.get(name, ()), *sets]
    (cfg,) = loader.load(RUNS / f"{name}.yaml", sets, env=env or {})
    return vllm_adapter.ADAPTER.translate(cfg)


class KwargsMatchVminFit(unittest.TestCase):
    def test_every_run_yaml(self):
        for name in sorted(GOLDEN):
            with self.subTest(name=name):
                self.assertEqual(spec_for(name).engine_kwargs, GOLDEN[name]["vmin_fit"])

    def test_env_override(self):
        for name in sorted(GOLDEN):
            with self.subTest(name=name):
                self.assertEqual(
                    spec_for(name, env=ENV_OVERRIDE, sets=["mode.returns=text"]).engine_kwargs,
                    GOLDEN[name]["vmin_fit_env_override"],
                )

    def test_nograph(self):
        for name in sorted(GOLDEN):
            with self.subTest(name=name):
                spec = spec_for(name, sets=["engine.cudagraph=false", "mode.returns=text"])
                self.assertEqual(spec.engine_kwargs, GOLDEN[name]["vmin_fit_nograph"])
                self.assertIn("--enforce-eager", spec.server_argv)


class ServerArgv(unittest.TestCase):
    def test_llama_is_the_vllm_serve_argv(self):
        spec = spec_for("llama")
        self.assertEqual(
            list(spec.server_argv),
            [
                "vllm",
                "serve",
                "<MODEL>",
                "--port",
                "8000",
                "--tensor-parallel-size",
                "1",
                "--dtype",
                "float16",
                "--max-model-len",
                "9216",
                "--gpu-memory-utilization",
                "0.9",
                "--no-enable-prefix-caching",
                "--enable-chunked-prefill",
                "--trust-remote-code",
            ],
        )
        self.assertEqual(spec.env, {"VLLM_SERVER_DEV_MODE": "1", "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1"})
        self.assertEqual((spec.health_path, spec.port), ("/health", 8000))

    def test_pp_is_pipeline_parallel_size(self):
        spec = spec_for("glm53", sets=["engine.pp=2"])
        argv = list(spec.server_argv)
        self.assertEqual(argv[argv.index("--pipeline-parallel-size") + 1], "2")
        self.assertEqual(spec.engine_kwargs["pipeline_parallel_size"], 2)
        self.assertNotIn("--pipeline-parallel-size", spec_for("glm53").server_argv)

    def test_quant_is_read_from_the_checkpoint_on_the_server(self):
        spec = spec_for("glm53")
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
        argv = list(
            spec_for(
                "llama",
                sets=[
                    "engine.kv_dtype=fp8",
                    "engine.ep=true",
                    "engine.gpu_util=0.95",
                    "engine.extra.vllm={max_num_seqs: 32, max_num_batched_tokens: 4096}",
                ],
            ).server_argv
        )
        for pair in (
            ["--kv-cache-dtype", "fp8"],
            ["--gpu-memory-utilization", "0.95"],
            ["--max-num-seqs", "32"],
            ["--max-num-batched-tokens", "4096"],
        ):
            i = argv.index(pair[0])
            self.assertEqual(argv[i : i + 2], pair)
        self.assertIn("--enable-expert-parallel", argv)

    def test_expert_parallel_and_prefix_caching_switches(self):
        argv = spec_for("glm53flash", sets=["engine.quant=null", "engine.prefix_caching=true"]).server_argv
        self.assertIn("--enable-expert-parallel", argv)
        self.assertIn("--enable-prefix-caching", argv)

    def test_nsys_prefix_is_prof_sh(self):
        spec = spec_for("llama", sets=["engine.instrumentation=nsys", "run_name=r1"])
        self.assertEqual(
            list(spec.server_argv[:7]),
            [
                "nsys",
                "profile",
                "-t",
                "cuda,nvtx",
                "-o",
                str(Path("results/r1/nsys").resolve()),
                "--force-overwrite=true",
            ],
        )

    def test_nsys_writes_into_the_run_dir_kernel_summary_reads(self):
        (cfg,) = loader.load(RUNS / "llama.yaml", [*PINNED, "engine.instrumentation=nsys"], env={})
        with tempfile.TemporaryDirectory() as td:
            argv = vllm_adapter.ADAPTER.translate(cfg, Path(td, "r")).server_argv
            out = Path(argv[argv.index("-o") + 1])
            self.assertTrue(out.is_absolute())
            self.assertEqual(out.with_suffix(".nsys-rep").name, native.TRACE_FILES["nsys"][0])
            self.assertEqual(out.parent, Path(td, "r").resolve())

    def test_torch_instrumentation_is_unsupported(self):
        with self.assertRaises(base.UnsupportedField):
            spec_for("llama", sets=["engine.instrumentation=torch"])

    def test_layers_is_unsupported(self):
        with self.assertRaisesRegex(base.UnsupportedField, "layers=2"):
            spec_for("llama", sets=["engine.layers=2"])

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
        for value in ("{speculative_config: {k: 1}}", "{tp: [1, 2]}", "notamapping", "{tensor_parallel_size: 2}"):
            with self.subTest(value=value), self.assertRaises(base.UnsupportedField):
                spec_for("llama", sets=[f"engine.extra.vllm={value}"])


class GridDriver(unittest.TestCase):
    """mode.returns picks how a grid run reaches vLLM."""

    def cfg(self, *sets):
        (cfg,) = loader.load(RUNS / "llama.yaml", [*PINNED, *sets], env={})
        return cfg

    def test_a_returns_grid_is_in_process(self):
        adapter = vllm_adapter.ADAPTER
        for returns in ("text", "logits"):
            with self.subTest(returns=returns):
                cfg = self.cfg(f"mode.returns={returns}")
                self.assertFalse(adapter.needs_server(cfg))
                driver = adapter.driver(adapter.translate(cfg), launcher.NoServer(Path("r")))
                self.assertIsInstance(driver, InProcessDriver)

    def test_pp_and_eager_timing_grids_launch_the_server(self):
        adapter = vllm_adapter.ADAPTER
        for setting, flag in (
            ("engine.pp=2", "--pipeline-parallel-size"),
            ("engine.cudagraph=false", "--enforce-eager"),
        ):
            with self.subTest(setting=setting):
                cfg = self.cfg(setting)
                spec = adapter.translate(cfg)
                self.assertTrue(adapter.needs_server(cfg))
                self.assertIn(flag, spec.server_argv)

    def test_a_timing_grid_launches_the_server(self):
        cfg = self.cfg()
        adapter = vllm_adapter.ADAPTER
        spec = adapter.translate(cfg)
        self.assertTrue(adapter.needs_server(cfg))
        self.assertEqual(spec.server_argv, spec_for("llama").server_argv)
        handle = launcher.NoServer(Path("r"), endpoint="http://127.0.0.1:8000")
        driver = adapter.driver(spec, handle)
        self.assertIsInstance(driver, HttpDriver)
        self.assertEqual(driver.url, "http://127.0.0.1:8000/v1/completions")

    def test_a_grid_trace_goes_to_completions_without_building_its_prompts(self):
        spec = vllm_adapter.ADAPTER.translate(self.cfg())
        handle = launcher.NoServer(Path("r"), endpoint="http://127.0.0.1:8000")
        with mock.patch.object(GridTrace, "requests", side_effect=AssertionError("built every cell")):
            driver = vllm_adapter.ADAPTER.driver(spec, handle, GridTrace([(100, 1)], [60, 230], "distinct", 4096))
        self.assertEqual(driver.url, "http://127.0.0.1:8000/v1/completions")

    def test_the_driver_asks_for_the_served_model_name(self):
        spec = vllm_adapter.ADAPTER.translate(self.cfg("engine.extra.vllm={served_model_name: llama}"))
        self.assertEqual(spec.server_argv[-2:], ("--served-model-name", "llama"))
        driver = vllm_adapter.ADAPTER.driver(spec, launcher.NoServer(Path("r"), endpoint="http://127.0.0.1:8000"))
        self.assertEqual(driver.model, "llama")

    def test_no_max_model_len_flag_when_unset(self):
        spec = vllm_adapter.ADAPTER.translate(self.cfg("engine.max_model_len=null"))
        self.assertNotIn("--max-model-len", spec.server_argv)
        self.assertNotIn("max_model_len", spec.engine_kwargs)

    def test_the_driver_sends_the_servers_api_key(self):
        cfg = self.cfg()
        handle = launcher.NoServer(Path("r"), endpoint="http://127.0.0.1:8000")
        keyed = dataclasses.replace(cfg, engine=dataclasses.replace(cfg.engine, extra={"vllm": {"api_key": "sk-1"}}))
        drivers = {
            name: vllm_adapter.ADAPTER.driver(vllm_adapter.ADAPTER.translate(c), handle)
            for name, c in (("plain", cfg), ("keyed", keyed))
        }
        self.assertEqual({n: d.api_key for n, d in drivers.items()}, {"plain": "test", "keyed": "sk-1"})

    def test_chat_requests_go_to_chat_completions(self):
        spec = vllm_adapter.ADAPTER.translate(self.cfg())
        handle = launcher.NoServer(Path("r"), endpoint="http://127.0.0.1:8000")
        chat = mock.Mock(**{"requests.return_value": [Request([], 3, messages=[{"role": "user"}])]})
        ids = mock.Mock(**{"requests.return_value": [Request([1, 2], 2, messages=[{"role": "user"}])]})
        urls = {
            name: vllm_adapter.ADAPTER.driver(spec, handle, trace).url
            for name, trace in (("chat", chat), ("ids", ids), ("none", None))
        }
        self.assertEqual(
            urls,
            {
                "chat": "http://127.0.0.1:8000/v1/chat/completions",
                "ids": "http://127.0.0.1:8000/v1/completions",
                "none": "http://127.0.0.1:8000/v1/completions",
            },
        )


def quettaserve_spec(name, cell, link="nvlink", sets=()):
    sets = ["allow_unverified=true", "hardware.backend=cuda", f"hardware.link_profile={link}", *sets]
    (cfg,) = loader.load(RUNS / f"{name}.yaml", sets, env={})
    return quettaserve.ADAPTER.translate(cfg, cell=cell)


DS = ["engine.model=/ckpt/ds", "engine.extra.qs_cfg=/ckpt/ds/config.json"]
QW = ["engine.model=/ckpt/qw"]


class DeepseekMoe(unittest.TestCase):
    def test_nvlink_cell(self):
        spec = quettaserve_spec("deepseek", (1024, 16), sets=DS)
        self.assertEqual(
            spec.env,
            {
                "XVAL_LINK_PROFILE": "nvlink",
                "NCCL_IB_DISABLE": "1",
                "DS_CFG": "/ckpt/ds/config.json",
                "DS_CKPT": "/ckpt/ds",
                "DS_WORLD": "8",
                "DS_TIMING_STEPS": "100",
                "DS_BATCH": "16",
                "DS_BENCH_PROMPT": "1024",
                "DS_MAX_SEQ": "1152",
            },
        )
        self.assertEqual(spec.env_drop, ("DS_LAYERS",))
        self.assertEqual(
            list(spec.bench_argv),
            ["cargo", "test", "-p", "deepseek", "--test", "batch_bench", "--release", "--", "--nocapture"],
        )
        self.assertEqual(spec.cwd, ".")
        self.assertEqual(spec.ok_pattern, r"^\[batch_bench\] .*layers=all")
        self.assertEqual(spec.log_header, ">>> ctx=1024 bs=16 world=8 max_seq=1152 link=nvlink nccl=auto/auto")
        self.assertEqual(spec.server_argv, ())

    def test_nsys_writes_one_report_per_cell_in_the_run_dir(self):
        (cfg,) = loader.load(
            RUNS / "deepseek.yaml",
            [
                "allow_unverified=true",
                "hardware.backend=cuda",
                "hardware.link_profile=nvlink",
                "engine.instrumentation=nsys",
                *DS,
            ],
            env={},
        )
        spec = quettaserve.ADAPTER.translate(cfg, "/abs/run")
        outs = [c.bench_argv[c.bench_argv.index("-o") + 1] for _, _, c in spec.cells]
        self.assertEqual(outs, [f"/abs/run/nsys-{ctx}x{bs}" for ctx, bs in cfg.grid.cells])

    def test_pcie_exports_every_nonempty_collective_var(self):
        spec = quettaserve_spec("deepseek", (8192, 64), link="pcie", sets=DS)
        self.assertEqual(
            spec.env,
            {
                "XVAL_LINK_PROFILE": "pcie",
                "NCCL_ALGO": "allreduce:tree;allgather:ring",
                "NCCL_PROTO": "Simple",
                "NCCL_P2P_LEVEL": "SYS",
                "NCCL_IB_DISABLE": "1",
                "LLMSRV_NO_NVLS": "1",
                "LLMSRV_TWOSHOT_MIN_BYTES": "0",
                "DS_CFG": "/ckpt/ds/config.json",
                "DS_CKPT": "/ckpt/ds",
                "DS_WORLD": "8",
                "DS_TIMING_STEPS": "100",
                "DS_BATCH": "64",
                "DS_BENCH_PROMPT": "8192",
                "DS_MAX_SEQ": "8320",
            },
        )
        self.assertTrue(spec.log_header.endswith("nccl=allreduce:tree;allgather:ring/Simple"))

    def test_layers_sets_ds_layers_and_its_pass_pattern(self):
        spec = quettaserve_spec("deepseek", (1024, 16), sets=[*DS, "engine.layers=6"])
        self.assertEqual(spec.env["DS_LAYERS"], "6")
        self.assertEqual(spec.env_drop, ())
        self.assertEqual(spec.ok_pattern, r"^\[batch_bench\] .*layers=6")

    def test_layers_without_batch_bench_is_unsupported(self):
        with self.assertRaisesRegex(base.UnsupportedField, "only batch_bench reads QW_LAYERS"):
            quettaserve_spec("qwen3", (4096, 4), sets=[*QW, "engine.layers=6"])

    def test_moe_without_cfg_is_refused(self):
        with self.assertRaisesRegex(ConfigError, "DS_CFG"):
            quettaserve_spec("deepseek", (1024, 1), sets=["engine.model=/ckpt/ds"])

    def test_afd_is_unsupported(self):
        with self.assertRaisesRegex(base.UnsupportedField, "serving_style=afd:2a6f"):
            quettaserve_spec("deepseek", (1024, 1), sets=[*DS, "engine.serving_style=afd:2a6f"])

    def test_fields_the_bench_cannot_carry_are_refused(self):
        for field, shown in (
            ("engine.pp=2", "pp=2"),
            ("engine.cudagraph=false", "cudagraph=False"),
            ("engine.gpu_util=0.5", "gpu_util=0.5"),
        ):
            with self.subTest(field), self.assertRaisesRegex(base.UnsupportedField, shown):
                quettaserve_spec("deepseek", (1024, 1), sets=[*DS, field])
        # Model yaml facts (dtype, kv_dtype, ep) describe the checkpoint.
        self.assertTrue(quettaserve_spec("deepseek", (1024, 1), sets=DS).bench_argv)

    def test_cli_refuses_pp_on_quettaserve(self):
        with (
            self.assertRaises(SystemExit) as e,
            mock.patch.dict(os.environ, {}, clear=True),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            cli.main(
                [
                    "run",
                    str(RUNS / "deepseek.yaml"),
                    "--dry-run",
                    "--allow-unverified",
                    "--set",
                    "engine.name=quettaserve",
                    "--set",
                    "engine.pp=2",
                    "--set",
                    "engine.cudagraph=false",
                    "--set",
                    "engine.model=/x",
                    "--set",
                    "engine.extra.qs_cfg=/c",
                ]
            )
        self.assertIn("quettaserve: pp=2 has no launch flag", str(e.exception.code))

    def test_prebuilt_binary_and_qs_dir(self):
        spec = quettaserve_spec(
            "deepseek", None, sets=[*DS, "engine.extra.bench_bin=/b/batch_bench", "engine.extra.qs_dir=/qs"]
        )
        self.assertEqual(list(spec.bench_argv), ["/b/batch_bench", "--nocapture"])
        self.assertEqual(spec.cwd, "/qs")
        self.assertNotIn("DS_BATCH", spec.env)


class Qwen3Gdn(unittest.TestCase):
    def test_rust_kernels_without_a_gdn_dir(self):
        spec = quettaserve_spec("qwen3", (4096, 4), sets=QW)
        self.assertEqual(
            spec.env,
            {
                "XVAL_LINK_PROFILE": "nvlink",
                "NCCL_IB_DISABLE": "1",
                "QS_KERNELS": "rust",
                "QS_CUSTOM_AR": "1",
                "QW_MODE": "graph",
                "QW_CKPT": "/ckpt/qw",
                "QW_WORLD": "4",
                "QW_TIMING_STEPS": "100",
                "QW_BATCH": "4",
                "QW_BENCH_PROMPT": "4096",
                "QW_MAX_SEQ": "4224",
            },
        )
        self.assertEqual(spec.env_drop, ())
        self.assertEqual(
            list(spec.bench_argv),
            ["cargo", "test", "-p", "qwen", "--test", "decode_loop", "--release", "--", "--nocapture"],
        )
        self.assertEqual(spec.ok_pattern, "^LOOP ")
        self.assertEqual(
            spec.log_header,
            ">>> ctx=4096 bs=4 world=4 max_seq=4224 link=nvlink nccl=auto/auto kern=rust ar=1 mode=graph gdn_dir=none",
        )

    def test_exact_kernels_and_eager_with_a_gdn_dir(self):
        spec = quettaserve_spec("qwen3", (1024, 1), sets=[*QW, "engine.extra.gdn_dir=/gdn"])
        self.assertEqual(
            {k: spec.env[k] for k in ("QS_KERNELS", "QS_VLLM_GDN_DIR", "QW_MODE")},
            {"QS_KERNELS": "exact", "QS_VLLM_GDN_DIR": "/gdn", "QW_MODE": "eager"},
        )

    def test_single_rank_has_no_custom_all_reduce(self):
        spec = quettaserve_spec("qwen3", (1024, 1), sets=[*QW, "engine.tp=1"])
        self.assertNotIn("QS_CUSTOM_AR", spec.env)
        self.assertIn(" ar=0 ", spec.log_header)


class OkPattern(unittest.TestCase):
    def test_matches_the_workload_records(self):
        records = json.loads((FIXTURES / "config_workloads.json").read_text())
        for name in ("deepseek", "qwen3"):
            bench = records[name]["cleared"]["xval_config"]["workload"]["bench"]
            self.assertEqual(quettaserve.OK_BY_TEST[bench["test"]], bench["ok"])


if __name__ == "__main__":
    unittest.main()
