"""The quettaserve spec carries exactly what scripts/crossval/bench.sh exports.

Expected values are read off bench.sh by hand and written as literals:
:15 <P>_CKPT, :20-21 moe <P>_CFG and unset <P>_LAYERS, :31 XVAL_LINK_PROFILE,
:33-36 steps/headroom/world, :42-45 non-empty collective vars, :48-60 the gdn
rule, :66-69 max_seq and the `>>>` header, :75-79 per-cell exports and argv.
"""

import contextlib
import io
import json
import unittest

from quettabench import cli
from quettabench.config import loader
from quettabench.config.schema import ConfigError
from quettabench.engines import base, quettaserve
from tests.golden._harness import FIXTURES, REPO

RUNS = REPO / "configs" / "runs"


def spec_for(name, cell, link="nvlink", sets=()):
    sets = ["allow_unverified=true", "hardware.backend=cuda",
            f"hardware.link_profile={link}", *sets]
    (cfg,) = loader.load(RUNS / f"{name}.yaml", sets, env={})
    return quettaserve.ADAPTER.translate(cfg, cell=cell)


DS = ["engine.model=/ckpt/ds", "engine.extra.qs_cfg=/ckpt/ds/config.json"]
QW = ["engine.model=/ckpt/qw"]


class DeepseekMoe(unittest.TestCase):
    def test_nvlink_cell(self):
        spec = spec_for("deepseek", (1024, 16), sets=DS)
        self.assertEqual(spec.env, {
            "XVAL_LINK_PROFILE": "nvlink", "NCCL_IB_DISABLE": "1",
            "DS_CFG": "/ckpt/ds/config.json", "DS_CKPT": "/ckpt/ds", "DS_WORLD": "8",
            "DS_TIMING_STEPS": "100", "DS_BATCH": "16", "DS_BENCH_PROMPT": "1024",
            "DS_MAX_SEQ": "1152"})
        self.assertEqual(spec.env_drop, ("DS_LAYERS",))
        self.assertEqual(list(spec.bench_argv), [
            "cargo", "test", "-p", "deepseek", "--test", "batch_bench", "--release",
            "--", "--nocapture"])
        self.assertEqual(spec.cwd, ".")
        self.assertEqual(spec.ok_pattern, r"^\[batch_bench\] .*layers=all")
        self.assertEqual(spec.log_header,
                         ">>> ctx=1024 bs=16 world=8 max_seq=1152 link=nvlink nccl=auto/auto")
        self.assertEqual(spec.server_argv, ())

    def test_nsys_writes_one_report_per_cell_in_the_run_dir(self):
        (cfg,) = loader.load(RUNS / "deepseek.yaml",
                             ["allow_unverified=true", "hardware.backend=cuda",
                              "hardware.link_profile=nvlink", "engine.instrumentation=nsys",
                              *DS], env={})
        spec = quettaserve.ADAPTER.translate(cfg, "/abs/run")
        outs = [c.bench_argv[c.bench_argv.index("-o") + 1] for _, _, c in spec.cells]
        self.assertEqual(outs, [f"/abs/run/nsys-{ctx}x{bs}" for ctx, bs in cfg.grid.cells])

    def test_pcie_exports_every_nonempty_collective_var(self):
        spec = spec_for("deepseek", (8192, 64), link="pcie", sets=DS)
        self.assertEqual(spec.env, {
            "XVAL_LINK_PROFILE": "pcie", "NCCL_ALGO": "allreduce:tree;allgather:ring",
            "NCCL_PROTO": "Simple", "NCCL_P2P_LEVEL": "SYS", "NCCL_IB_DISABLE": "1",
            "LLMSRV_NO_NVLS": "1", "LLMSRV_TWOSHOT_MIN_BYTES": "0",
            "DS_CFG": "/ckpt/ds/config.json", "DS_CKPT": "/ckpt/ds", "DS_WORLD": "8",
            "DS_TIMING_STEPS": "100", "DS_BATCH": "64", "DS_BENCH_PROMPT": "8192",
            "DS_MAX_SEQ": "8320"})
        self.assertTrue(spec.log_header.endswith("nccl=allreduce:tree;allgather:ring/Simple"))

    def test_layers_sets_ds_layers_and_its_pass_pattern(self):
        spec = spec_for("deepseek", (1024, 16), sets=[*DS, "engine.layers=6"])
        self.assertEqual(spec.env["DS_LAYERS"], "6")
        self.assertEqual(spec.env_drop, ())
        self.assertEqual(spec.ok_pattern, r"^\[batch_bench\] .*layers=6")

    def test_layers_without_batch_bench_is_unsupported(self):
        with self.assertRaisesRegex(base.UnsupportedField, "only batch_bench reads QW_LAYERS"):
            spec_for("qwen3", (4096, 4), sets=[*QW, "engine.layers=6"])

    def test_moe_without_cfg_is_refused(self):
        with self.assertRaisesRegex(ConfigError, "DS_CFG"):
            spec_for("deepseek", (1024, 1), sets=["engine.model=/ckpt/ds"])

    def test_afd_is_unsupported(self):
        with self.assertRaisesRegex(base.UnsupportedField, "serving_style=afd:2a6f"):
            spec_for("deepseek", (1024, 1), sets=[*DS, "engine.serving_style=afd:2a6f"])

    def test_fields_the_bench_cannot_carry_are_refused(self):
        for field, shown in (("engine.pp=2", "pp=2"), ("engine.cudagraph=false", "cudagraph=False"),
                             ("engine.gpu_util=0.5", "gpu_util=0.5")):
            with self.subTest(field), self.assertRaisesRegex(base.UnsupportedField, shown):
                spec_for("deepseek", (1024, 1), sets=[*DS, field])
        # model yaml facts (deepseek's dtype, kv_dtype, ep) describe the checkpoint
        self.assertTrue(spec_for("deepseek", (1024, 1), sets=DS).bench_argv)

    def test_cli_refuses_pp_on_quettaserve(self):
        with self.assertRaises(SystemExit) as e, \
                contextlib.redirect_stdout(io.StringIO()):
            cli.main(["run", str(RUNS / "deepseek.yaml"), "--dry-run", "--allow-unverified",
                      "--set", "engine.name=quettaserve", "--set", "engine.pp=2",
                      "--set", "engine.cudagraph=false", "--set", "engine.model=/x",
                      "--set", "engine.extra.qs_cfg=/c"])
        self.assertIn("quettaserve: pp=2 has no launch flag", str(e.exception.code))

    def test_prebuilt_binary_and_qs_dir(self):
        spec = spec_for("deepseek", None, sets=[*DS, "engine.extra.bench_bin=/b/batch_bench",
                                                "engine.extra.qs_dir=/qs"])
        self.assertEqual(list(spec.bench_argv), ["/b/batch_bench", "--nocapture"])
        self.assertEqual(spec.cwd, "/qs")
        self.assertNotIn("DS_BATCH", spec.env)


class Qwen3Gdn(unittest.TestCase):
    def test_rust_kernels_without_a_gdn_dir(self):
        spec = spec_for("qwen3", (4096, 4), sets=QW)
        self.assertEqual(spec.env, {
            "XVAL_LINK_PROFILE": "nvlink", "NCCL_IB_DISABLE": "1",
            "QS_KERNELS": "rust", "QS_CUSTOM_AR": "1", "QW_MODE": "graph",
            "QW_CKPT": "/ckpt/qw", "QW_WORLD": "4", "QW_TIMING_STEPS": "100",
            "QW_BATCH": "4", "QW_BENCH_PROMPT": "4096", "QW_MAX_SEQ": "4224"})
        self.assertEqual(spec.env_drop, ())
        self.assertEqual(list(spec.bench_argv), [
            "cargo", "test", "-p", "qwen", "--test", "decode_loop", "--release",
            "--", "--nocapture"])
        self.assertEqual(spec.ok_pattern, "^LOOP ")
        self.assertEqual(spec.log_header,
                         ">>> ctx=4096 bs=4 world=4 max_seq=4224 link=nvlink nccl=auto/auto"
                         " kern=rust ar=1 mode=graph gdn_dir=none")

    def test_exact_kernels_and_eager_with_a_gdn_dir(self):
        spec = spec_for("qwen3", (1024, 1), sets=[*QW, "engine.extra.gdn_dir=/gdn"])
        self.assertEqual({k: spec.env[k] for k in ("QS_KERNELS", "QS_VLLM_GDN_DIR", "QW_MODE")},
                         {"QS_KERNELS": "exact", "QS_VLLM_GDN_DIR": "/gdn", "QW_MODE": "eager"})

    def test_single_rank_has_no_custom_all_reduce(self):
        spec = spec_for("qwen3", (1024, 1), sets=[*QW, "engine.tp=1"])
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
