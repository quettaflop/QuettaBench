"""Offline compare checks: readers, pairing, withhold rules, agreement rows."""

import ast
import calendar
import contextlib
import importlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from quettabench import compare
from quettabench.config import loader
from quettabench.config.schema import ConfigError, check_ep_legal
from quettabench.engines.drivers import BenchDriver
from quettabench.metrics.results import RunWriter
from quettabench.metrics.step_fit import fit_run
from quettabench.workloads.base import synth_ids
from quettabench.workloads.live import RealTrace
from tests.contracts.helpers import bench_config, bench_run_dir
from tests.golden._harness import FIXTURES, grid_argv, run_grid_cli

REPO = Path(__file__).resolve().parents[2]
QB = REPO / "quettabench"


def _scripts(suffix):
    return sorted(p for p in QB.rglob(f"*{suffix}") if "__pycache__" not in p.parts)


def _config(name, *sets, env=None):
    (cfg,) = loader.load(REPO / "configs" / "runs" / f"{name}.yaml", ["allow_unverified=true", *sets], env=env or {})
    return cfg


def synth_requests(count, ctx, seed=0):
    return [
        {
            "prompt_token_ids": synth_ids(ctx, seed * 1_000_003 + j),
            "prompt_len": ctx,
            "seed": seed,
            "order": j,
            "source": "SYNTHETIC",
        }
        for j in range(count)
    ]


def load_trace(path):
    return [r.to_dict() for r in RealTrace.from_jsonl(path).requests()]


class CompareScripts(unittest.TestCase):
    def test_python_scripts_parse(self):
        # Parse only: importing would pull in vllm, which CI lacks.
        scripts = _scripts(".py")
        self.assertTrue(scripts, "no python scripts under quettabench or analysis")
        for script in scripts:
            ast.parse(script.read_text(), filename=str(script))

    def test_no_shell_entry_points_left(self):
        self.assertEqual(_scripts(".sh"), [])

    def test_serve_wires_the_identity_gate(self):
        self.assertIn("workload_hash", (QB / "modes" / "serve.py").read_text())
        self.assertIn("WORKLOAD_CHECK", (QB / "metrics" / "results.py").read_text())

    def test_backend_detection_and_no_hard_nvidia_smi(self):
        # QBENCH_BACKEND wins; with no accel tooling on PATH detection says cpu.
        with mock.patch("shutil.which", return_value=None):
            self.assertEqual(_config("llama").hardware.backend, "cpu")
            self.assertEqual(_config("llama").hardware.link_profile, "nvlink")
        self.assertEqual(
            _config("llama", "engine.dtype=bfloat16", env={"QBENCH_BACKEND": "tpu"}).hardware.backend, "tpu"
        )

    def test_pp_wired_in_grid_and_device_count(self):
        # pipeline_parallel_size comes from QBENCH_PP or the workload's pp.
        src = (QB / "config" / "env_map.py").read_text() + (QB / "engines" / "vllm.py").read_text()
        self.assertIn("pipeline_parallel_size", src)
        self.assertIn("QBENCH_PP", src)
        self.assertIn('"pp": e.pp', (QB / "modes" / "serve.py").read_text())
        self.assertIn('f"META {k}={v}"', (QB / "metrics" / "results.py").read_text())
        ds = _config("deepseek").engine
        self.assertEqual(ds.tp * ds.pp, 8)
        ds = _config("deepseek", env={"QBENCH_PP": "2"}).engine
        self.assertEqual(ds.tp * ds.pp, 16)

    def test_link_profile_detection(self):
        # A caller override wins; non-cuda backends take the no-pins nvlink profile.
        self.assertEqual(_config("llama", env={"QBENCH_LINK_PROFILE": "pcie"}).hardware.link_profile, "pcie")
        self.assertEqual(
            _config("llama", "engine.dtype=bfloat16", env={"QBENCH_BACKEND": "tpu"}).hardware.link_profile, "nvlink"
        )

    def test_provision_defaults(self):
        self.assertEqual(_config("llama").hardware.provision["cache_root"], "/workspace/.qbench-cache")

    def test_goodput_search_and_verdict(self):
        from quettabench.metrics.summary import sla_verdict
        from quettabench.modes.serve import goodput_search

        gen = goodput_search(start=1.0)
        speeds, speed = [], next(gen)
        try:
            while True:
                speeds.append(speed)
                speed = gen.send(speed < 4.0)  # SLA passes strictly below 4.0.
        except StopIteration:
            pass
        self.assertLessEqual(len(speeds), 6)
        passing = [x for x in speeds if x < 4.0]
        failing = [x for x in speeds if x >= 4.0]
        self.assertTrue(passing and failing, speeds)
        self.assertGreater(max(passing), 3.0, speeds)
        s = {"ttft_ms": {"p99": 900.0}, "tpot_ms": {"p99": 40.0}}
        self.assertEqual(sla_verdict(s, 500.0, 50.0), ["ttft_p99"])
        self.assertEqual(sla_verdict(s, None, None), [])
        self.assertIn("GOODPUT_SUMMARY", (QB / "metrics" / "results.py").read_text())

    def test_injected_arrivals(self):
        from quettabench.workloads import arrival
        from quettabench.workloads.base import workload_hash

        importlib.reload(arrival)
        self.assertIsNone(arrival.parse_arrival("trace", 8))
        p = arrival.parse_arrival("poisson:2.0", 8)
        self.assertEqual(p, arrival.parse_arrival("poisson:2.0", 8))
        self.assertEqual(sorted(p), p)
        b = arrival.parse_arrival("burst:2x4@5.0", 8)
        self.assertEqual(b, [0.0] * 4 + [5.0] * 4)
        with self.assertRaises(ValueError):
            arrival.parse_arrival("burst:1x4@5.0", 8)
        r = arrival.parse_arrival("ramp:0.5:4.0", 8)
        self.assertEqual(len(r), 8)
        reqs = synth_requests(4, 64, seed=1)
        h0 = workload_hash(reqs)
        for req, t in zip(reqs, b[4:]):
            req["arrival_ts"] = round(t, 6)
        self.assertNotEqual(h0, workload_hash(reqs))

    def test_energy_integration(self):
        from quettabench.profiling.energy_cost_accuracy import integrate_power

        self.assertEqual(integrate_power([100.0, 100.0, 100.0], 1.0), 300.0)
        self.assertEqual(integrate_power([], 1.0), 0.0)
        self.assertIn("ENERGY_SUMMARY", (QB / "profiler.py").read_text())

    def test_cost_math(self):
        from quettabench.profiling.energy_cost_accuracy import cost_per_m_tokens

        # 2 GPUs at $3.60/hr for 30 min producing 1M tokens: $3.60, $3.60/M.
        usd, per_m = cost_per_m_tokens(3.6, 2, 1800.0, 1_000_000)
        self.assertAlmostEqual(usd, 3.6)
        self.assertAlmostEqual(per_m, 3.6)
        self.assertIn("COST_SUMMARY", (QB / "profiler.py").read_text())

    def test_gsm8k_answer_parse(self):
        from quettabench.profiling.energy_cost_accuracy import gsm8k_answer as f

        self.assertEqual(f("Natalia sold 48 clips. #### 72"), "72")
        self.assertEqual(f("so the total is $1,234."), "1234")
        self.assertEqual(f("the answer is 3.5 apples"), "3.5")
        self.assertIsNone(f("no numbers here"))
        # ACCURACY_SUMMARY prints from the profile's accuracy section, not SPEC_DECODE_SUMMARY's records.
        self.assertIn("ACCURACY_SUMMARY", (QB / "profiler.py").read_text())

    def test_serving_style_contract(self):
        sys.path.insert(0, str(REPO))
        try:
            from quettabench.config import schema

            self.assertEqual(schema.parse_serving_style("aggregated"), {"mode": "aggregated"})
            self.assertEqual(schema.parse_serving_style(None), {"mode": "aggregated"})
            self.assertEqual(schema.parse_serving_style("disagg:2p2d"), {"mode": "disagg", "prefill": 2, "decode": 2})
            self.assertEqual(schema.parse_serving_style("ep8dp2"), {"mode": "ep_dp", "ep": 8, "dp": 2})
            self.assertEqual(schema.parse_serving_style("afd:2a6f"), {"mode": "afd", "attn": 2, "ffn": 6})
            for bad in (
                "disagg:0p1d",
                "disagg:2p",
                "ep8",
                "wideEP",
                "disagg:xp1d",
                "ep0dp2",
                "ep8dp0",
                "afd:0a1f",
                "afd:2a",
                "afd:2f6a",
            ):
                with self.assertRaises(ValueError):
                    schema.parse_serving_style(bad)
            schema.check_ep_legal(8, 256)
            with self.assertRaises(ValueError):
                schema.check_ep_legal(7, 256)
            self.assertFalse(schema.serving_style_compatible("disagg:1p1d", "aggregated"))
            self.assertFalse(schema.serving_style_compatible("afd:1a1f", "aggregated"))
            self.assertFalse(schema.serving_style_compatible("afd:1a1f", "disagg:1p1d"))
            # Pools are sized in GPUs: afd needs attn + ffn, not tp * pp.
            engine = SimpleNamespace(serving_style="afd:2a6f", tp=4, pp=1)
            self.assertEqual(schema.device_count(engine), 8)
            engine = SimpleNamespace(serving_style="disagg:2p2d", tp=8, pp=1)
            self.assertEqual(schema.device_count(engine), 4)
            engine = SimpleNamespace(serving_style="aggregated", tp=4, pp=2)
            self.assertEqual(schema.device_count(engine), 8)
            self.assertTrue(schema.serving_style_compatible("aggregated", "aggregated"))
        finally:
            sys.path.pop(0)

    def test_glm53_workload_contract(self):
        # GLM-5.3 is MLA + DSA, so kv_bytes_per_token uses the latent formula.
        e = _config("glm53").engine
        layers, kv_lora_rank, rope, bf16 = 78, 512, 64, 2
        self.assertEqual(e.kv_bytes_per_token, layers * (kv_lora_rank + rope) * bf16)
        self.assertEqual((e.quant, e.tp * e.pp, e.ep), ("fp8", 8, True))
        self.assertFalse(e.verified, "bring-up gate: runs need ALLOW_UNVERIFIED=1")
        check_ep_legal(e.tp, 256)

    def test_glm53flash_workload_contract(self):
        # Hybrid: only the 11 sparse-MLA layers hold per-token KV, so kv_bytes_per_token is mandatory.
        e = _config("glm53flash").engine
        mla_layers, kv_lora_rank, rope, bf16 = 11, 512, 0, 2
        self.assertEqual(e.kv_bytes_per_token, mla_layers * (kv_lora_rank + rope) * bf16)
        self.assertEqual((e.quant, e.tp, e.ep), ("fp8", 4, True))
        self.assertFalse(e.verified, "bring-up gate: runs need ALLOW_UNVERIFIED=1")
        check_ep_legal(e.tp, 288)

    def test_grid_run_supports_moe_bringup(self):
        src = (QB / "cli.py").read_text() + (QB / "engines" / "vllm.py").read_text()
        self.assertIn("enable_expert_parallel", src)
        self.assertIn("allow-unverified", src)

    def _bench(self, workload, stub_body, sets=(), env=None):
        with tempfile.TemporaryDirectory() as td:
            stub = Path(td) / "stub"
            stub.write_text("#!/usr/bin/env bash\n" + stub_body + "\n")
            stub.chmod(0o755)
            argv = [
                sys.executable,
                "-m",
                "quettabench.cli",
                "run",
                str(REPO / "configs" / "runs" / f"{workload}.yaml"),
                "--allow-unverified",
                "--results-dir",
                td,
                "--set",
                "engine.name=quettaserve",
                "--set",
                "engine.model=/dev/null",
                "--set",
                f"engine.extra.bench_bin={stub}",
                "--set",
                f"engine.extra.quettaserve_dir={td}",
            ]
            for item in sets:
                argv += ["--set", item]
            out = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                cwd=REPO,
                env={**os.environ, "QBENCH_BACKEND": "cpu", **(env or {})},
            )
            log = Path(td) / workload / "bench.log"
            return out, log.read_text() if log.exists() else ""

    BENCH_LINE = (
        'echo "[batch_bench] world=${DS_WORLD} bs=${DS_BATCH}'
        " max_seq=${DS_MAX_SEQ} layers=${DS_LAYERS:-all}"
        " prompt=${DS_BENCH_PROMPT} steps=92: median 1.000 ms/step,"
        ' mean 1.0, best 1.0 -> 1.0 tok/s (median)"'
    )

    def test_bench_runs_the_deepseek_grid_at_the_workload_tp(self):
        out, log = self._bench(
            "deepseek", self.BENCH_LINE, ["engine.extra.quettaserve_config=/dev/null"], env={"DS_LAYERS": "6"}
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        cfg = _config("deepseek")
        tp = cfg.engine.tp
        cells = re.findall(r"^\[batch_bench\] world=(\d+) .*layers=(\w+)", log, re.M)
        self.assertEqual(len(cells), len(cfg.grid.cells))
        self.assertEqual(set(cells), {(str(tp), "all")})

    def test_bench_fails_when_a_cell_produces_no_line(self):
        out, _ = self._bench("deepseek", "exit 0", ["engine.extra.quettaserve_config=/dev/null"])
        self.assertEqual(out.returncode, 1)
        self.assertIn("FAIL", out.stderr)

    QWEN_LINE = (
        'echo "LOOP ctx=${QW_BENCH_PROMPT} bs=${QW_BATCH} kv=bf16'
        ' ms_per_step=1.000 gdn=${QS_KERNELS} mode=${QW_MODE} k=100"'
    )

    def test_bench_qwen_defaults_exact_to_eager(self):
        # exact has no graph wiring: with cubins present the driver picks the eager step.
        out, log = self._bench("qwen3", self.QWEN_LINE, ["engine.extra.gdn_dir=/dev/null"])
        self.assertEqual(out.returncode, 0, out.stderr)
        modes = re.findall(r"^LOOP .*mode=(\w+)", log, re.M)
        self.assertEqual(len(modes), len(_config("qwen3").grid.cells))
        self.assertEqual(set(modes), {"eager"})
        self.assertIn("kern=exact", log)

    def test_bench_requires_a_bench_section(self):
        out, _ = self._bench("llama", "exit 0")
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("no quettaserve: block", out.stderr)


class EngineBenchContract(unittest.TestCase):
    """Every QuettaServe bench cell needs a vLLM cell, or the table drops the row."""

    def test_llama_grid_covers_the_batch_bench_cells(self):
        grid = {tuple(c) for c in _config("llama").grid.cells}
        expected = {(ctx, bs) for ctx in (100, 1024, 4096) for bs in (1, 8, 16, 32, 64)}
        self.assertTrue(expected <= grid, f"missing cells: {sorted(expected - grid)}")

    def test_llama_single_grid_covers_the_tp_bench_cells(self):
        grid = {tuple(c) for c in _config("llama-tp1").grid.cells}
        expected = {(ctx, 1) for ctx in (100, 1024, 4096, 8192)}
        self.assertTrue(expected <= grid, f"missing cells: {sorted(expected - grid)}")

    def test_deepseek_grid_covers_the_batch_bench_cells(self):
        grid = {tuple(c) for c in _config("deepseek").grid.cells}
        expected = {(ctx, bs) for ctx in (1024, 8192, 16384) for bs in (1, 4, 16, 64)}
        self.assertEqual(grid, expected, f"grid drift: {sorted(grid ^ expected)}")


class TableParity(unittest.TestCase):
    """Ratios print only when both sides timed the same quantity."""

    def _table(self, bench_text, kv="float16", env=None, expect_rc=0, serving_style="aggregated", quant="bf16"):

        base = {
            "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu": "test",
            "vllm": "0",
            "mode": "FULL",
            "model": "m",
            "dtype": "float16",
            "grid": "g",
            "method": "slope",
            "kv_cache_dtype": kv,
            "nccl_algo": "allreduce:tree;allgather:ring",
            "nccl_proto": "Simple",
            "serving_style": serving_style,
            "quant": quant,
            "points": [{"ctx": 1024, "bs": 1, "tp": 1, "ms_per_step": 5.0, "tok_s": 200.0, "r2": 1.0, "kv": kv}],
        }
        with tempfile.TemporaryDirectory() as td:
            bench = Path(td) / "bench.log"
            cache = Path(td) / "base.json"
            bench.write_text(bench_text)
            cache.write_text(json.dumps(base))
            out = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "quettabench.cli",
                    "compare",
                    str(bench),
                    str(cache),
                    "--json",
                    str(Path(td) / "compare.json"),
                ],
                capture_output=True,
                text=True,
                env={**os.environ, **(env or {})},
                cwd=REPO,
            )
        self.assertEqual(out.returncode, expect_rc, out.stderr)
        return out.stdout

    def test_loop_bench_gets_a_ratio(self):
        out = self._table("LOOP ctx=1024 bs=1 kv=fp16 ms_per_step=10.000 k=100\n")
        self.assertIn("0.50x", out)
        self.assertNotIn("withheld", out)

    def test_quant_mismatch_flags_and_withholds(self):
        # nvfp4 engine vs bf16 baseline: flagged QUANT, withheld.
        out = self._table(
            "LOOP ctx=1024 bs=1 kv=fp16 ms_per_step=10.000 k=100\n", quant="bf16", env={"QBENCH_QUANT": "nvfp4"}
        )
        self.assertIn("QUANT MISMATCH", out)
        self.assertIn("QUANT bf16/nvfp4", out)
        self.assertNotIn("0.50x", out)
        self.assertIn("quantization", (QB / "engines" / "vllm.py").read_text())
        self.assertIn('"quant": e.quant or e.dtype', (QB / "modes" / "serve.py").read_text())

    def test_serving_style_mismatch_withholds(self):
        # Disagg baseline vs aggregated engine row: never a ratio.
        out = self._table(
            "LOOP ctx=1024 bs=1 kv=fp16 ms_per_step=10.000 k=100\n",
            serving_style="disagg:1p1d",
            env={"QBENCH_SERVING_STYLE": "aggregated"},
        )
        self.assertIn("SERVING MISMATCH", out)
        self.assertNotIn("0.50x", out)
        out2 = self._table(
            "LOOP ctx=1024 bs=1 kv=fp16 ms_per_step=10.000 k=100\n",
            serving_style="aggregated",
            env={"QBENCH_SERVING_STYLE": "aggregated"},
        )
        self.assertIn("0.50x", out2)

    def test_eager_loop_row_is_grouped_and_flagged(self):
        # Eager rows keep the ratio but get their own group and an EAGER flag.
        out = self._table("LOOP ctx=1024 bs=1 kv=fp16 ms_per_step=10.000 gdn=Vllm mode=eager\n")
        self.assertIn("decode_loop_eager", out)
        self.assertIn("EAGER", out)
        self.assertIn("0.50x", out)
        self.assertIn("engine gdn kernel(s): Vllm", out)

    def test_graph_loop_row_has_no_eager_flag(self):
        out = self._table("LOOP ctx=1024 bs=1 kv=fp16 ms_per_step=10.000 gdn=Tiled mode=graph\n")
        self.assertNotIn("EAGER", out)
        self.assertNotIn("decode_loop_eager", out)

    def test_bench_adapter_wires_the_gdn_family(self):
        text = (QB / "engines" / "quettaserve.py").read_text()
        self.assertIn('"exact" if gdn_dir else "rust"', text)
        self.assertIn('"eager" if kernels == "exact" else "graph"', text)
        self.assertIn("mode={mode}", text)

    def test_per_step_bench_gets_no_ratio(self):
        out = self._table("decode_batch_paged/1024x1\n                        time:   [9.9 ms 10.0 ms 10.1 ms]\n")
        self.assertIn("withheld", out)
        self.assertNotIn("0.50x", out)

    def test_kv_mismatch_is_flagged(self):
        out = self._table("LOOP ctx=1024 bs=1 kv=nvfp4 ms_per_step=10.000 k=100\n")
        self.assertIn("KV nvfp4/fp16", out)

    def test_deepseek_kv_difference_is_flagged(self):
        # bf16 engine vs fp8 baseline: KV flagged, ratio withheld.
        out = self._table("LOOP ctx=1024 bs=1 kv=bf16 ms_per_step=10.000 k=100\n", kv="fp8")
        self.assertIn("KV bf16/fp8", out)
        self.assertNotIn("0.50x", out)
        self.assertIn("--", out)

    BB = (
        "[batch_bench] world=1 bs=1 max_seq=1636 layers=all prompt=1024 "
        "steps=92: median {ms} ms/step, mean 10.100, best 9.900 "
        "-> 100.0 tok/s (median)\n"
    )

    def test_stock_deepseek_bench_line_gets_kv_flag_and_withheld_ratio(self):
        # batch_bench line: unstated KV is ?, ratio withheld.
        out = self._table(self.BB.format(ms="10.000"), kv="fp8")
        self.assertNotIn("0.50x", out)
        self.assertIn("KV ?/fp8", out)
        self.assertIn("batch_bench", out)
        self.assertIn("--", out)

    def test_loop_line_supersedes_the_bench_summary(self):
        out = self._table(self.BB.format(ms="20.000") + "LOOP ctx=1024 bs=1 kv=fp16 ms_per_step=10.000 k=100\n")
        self.assertIn("10.000", out)
        self.assertNotIn("20.000", out)

    def test_truncated_model_bench_lines_are_ignored(self):
        line = self.BB.format(ms="10.000").replace("layers=all", "layers=6")
        out = self._table(line, expect_rc=1)
        self.assertIn("truncated", out)
        self.assertNotIn("10.000", out)
        self.assertNotIn("synchronized step", out)

    def test_h200_grid_is_the_table(self):
        grid = {tuple(c) for c in _config("llama-big").grid.cells}
        expected = {(c, b) for c in (1024, 8192, 16384) for b in (1, 4, 16, 64)}
        self.assertEqual(grid, expected)

    def test_duplicate_cells_keep_the_last_measurement(self):
        out = self._table(
            "decode_batch_paged/1024x1\n"
            "                        time:   [19.9 ms 20.0 ms 20.1 ms]\n"
            "decode_batch_paged/1024x1\n"
            "                        time:   [9.9 ms 10.0 ms 10.1 ms]\n"
        )
        self.assertIn("10.000", out)
        self.assertNotIn("20.000", out)

    def test_tp_mismatch_is_called_out(self):
        out = self._table(
            "LOOP ctx=1024 bs=1 tp=2 kv=fp16 ms_per_step=10.000 k=100\n",
            expect_rc=1,
        )
        self.assertIn("tp mismatch", out)

    def test_comm_bound_cell_is_flagged(self):
        # comm_bound_bs comes from the workload record (deepseek sets 64).
        import sys
        import time

        base = {
            "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu": "test",
            "vllm": "0",
            "mode": "FULL",
            "model": "DeepSeek-V4-Flash-0731",
            "dtype": "float16",
            "grid": "g",
            "method": "slope",
            "kv_cache_dtype": "float16",
            "nccl_algo": "allreduce:tree;allgather:ring",
            "nccl_proto": "Simple",
            "points": [
                {"ctx": 1024, "bs": 64, "tp": 1, "ms_per_step": 5.0, "tok_s": 12800.0, "r2": 1.0, "kv": "float16"}
            ],
        }
        bench_text = "LOOP ctx=1024 bs=64 kv=fp16 ms_per_step=10.000 k=100\n"
        with tempfile.TemporaryDirectory() as td:
            bench = Path(td) / "bench.log"
            cache = Path(td) / "base.json"
            bench.write_text(bench_text)
            cache.write_text(json.dumps(base))
            out = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "quettabench.cli",
                    "compare",
                    str(bench),
                    str(cache),
                    "--json",
                    str(Path(td) / "compare.json"),
                ],
                capture_output=True,
                text=True,
                env=os.environ,
                cwd=REPO,
            )
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("COMM", out.stdout)

    def test_no_comm_flag_without_workload_comm_bound(self):
        # Absent comm_bound_bs means never comm-bound.
        import sys
        import time

        base = {
            "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu": "test",
            "vllm": "0",
            "mode": "FULL",
            "model": "Llama-3.1-8B-Instruct",
            "dtype": "float16",
            "grid": "g",
            "method": "slope",
            "kv_cache_dtype": "float16",
            "nccl_algo": "allreduce:tree;allgather:ring",
            "nccl_proto": "Simple",
            "points": [
                {"ctx": 1024, "bs": 64, "tp": 1, "ms_per_step": 5.0, "tok_s": 12800.0, "r2": 1.0, "kv": "float16"}
            ],
        }
        bench_text = "LOOP ctx=1024 bs=64 kv=fp16 ms_per_step=10.000 k=100\n"
        with tempfile.TemporaryDirectory() as td:
            bench = Path(td) / "bench.log"
            cache = Path(td) / "base.json"
            bench.write_text(bench_text)
            cache.write_text(json.dumps(base))
            out = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "quettabench.cli",
                    "compare",
                    str(bench),
                    str(cache),
                    "--json",
                    str(Path(td) / "compare.json"),
                ],
                capture_output=True,
                text=True,
                env=os.environ,
                cwd=REPO,
            )
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertNotIn("COMM", out.stdout)
        self.assertIn("0.50x", out.stdout)

    def test_grid_prompt_modes(self):
        # clone reproduces the engine's synthetic prompt so both sides route alike.
        self.assertIn("(i * 137 + 11) % 100_000", (QB / "workloads" / "grid.py").read_text())
        src = (QB / "workloads" / "grid.py").read_text() + (QB / "config" / "schema.py").read_text()
        self.assertIn("prompt_mode", src)
        for mode in ('"clone"', '"corpus"', '"distinct"'):
            self.assertIn(mode, src)
        self.assertIn('return_outputs: str = "none"', src)

    def test_cache_records_prompt_mode(self):
        self.assertIn("prompt_mode", (QB / "compare.py").read_text())

    def test_corpus_mode_requires_an_explicit_corpus(self):
        # No experimental text ships in the repo; corpus mode takes QBENCH_CORPUS.
        self.assertIn("QBENCH_CORPUS", (QB / "modes" / "serve.py").read_text())

    def test_hash_canonicalization(self):
        # The gate hashes the completed set, so completion order must not move the hash.
        import random

        from quettabench.workloads.base import workload_hash

        reqs = synth_requests(24, 64, seed=5)
        for i, r in enumerate(reqs):
            r["arrival_ts"] = round(0.25 * i, 6)
        h = workload_hash(reqs)
        shuffled = list(reqs)
        random.Random(7).shuffle(shuffled)
        self.assertNotEqual([r["order"] for r in shuffled], [r["order"] for r in reqs])
        self.assertEqual(h, workload_hash(shuffled))
        prompt_mut = [dict(r) for r in reqs]
        ids = list(prompt_mut[3]["prompt_token_ids"])
        ids[0] += 1
        prompt_mut[3]["prompt_token_ids"] = ids
        self.assertNotEqual(h, workload_hash(prompt_mut))
        arrival_mut = [dict(r) for r in reqs]
        arrival_mut[3]["arrival_ts"] = arrival_mut[3]["arrival_ts"] + 0.5
        self.assertNotEqual(h, workload_hash(arrival_mut))

    def test_workload_identity_hash(self):
        from quettabench.workloads.base import workload_hash

        reqs = synth_requests(12, 64, seed=2)
        for i, r in enumerate(reqs):
            r["output_length"] = 128 + i
        h = workload_hash(reqs)
        self.assertEqual(h, workload_hash(list(reversed(reqs))), "hash must be order-independent")
        self.assertNotEqual(h, workload_hash(reqs[:-1]), "dropping a request must change the hash")
        altered = [dict(r) for r in reqs]
        altered[0] = {**altered[0], "output_length": altered[0]["output_length"] + 1}
        self.assertNotEqual(h, workload_hash(altered), "changing requested output must change the hash")

    def test_chat_requests_hash_their_messages(self):
        from quettabench.workloads.base import Request, workload_hash

        def chat(text, reply="r"):
            return [
                Request(
                    [],
                    0,
                    output_length=8,
                    messages=[{"role": "user", "content": text}, {"role": "assistant", "content": reply}],
                )
            ]

        self.assertNotEqual(workload_hash(chat("a")), workload_hash(chat("b")))
        # Assistant content is the engine's reply in a live session, so it does not count.
        self.assertEqual(workload_hash(chat("a", "x")), workload_hash(chat("a", "y")))
        ids = [Request([1, 2, 3], 3, output_length=8)]
        with_messages = [Request([1, 2, 3], 3, output_length=8, messages=chat("a")[0].messages)]
        self.assertEqual(workload_hash(ids), workload_hash(with_messages))

    def test_serving_emits_verdicts(self):
        schema = (QB / "config" / "schema.py").read_text()
        for field in ("sla_ttft_ms", "sla_tpot_ms"):
            self.assertIn(field, schema)
        lines = (QB / "metrics" / "results.py").read_text()
        for tag in ("SLA_SUMMARY", "BURST_SUMMARY", "SPEC_DECODE_SUMMARY"):
            self.assertIn(tag, lines)
        self.assertIn("FIDELITY", (QB / "compare.py").read_text())

    def test_burst_recovery_and_acceptance(self):
        from quettabench.metrics.summary import acceptance, burst_recovery

        self.assertEqual(acceptance([1, 2, 3, 4], [1, 2, 9, 4]), (2, 0.5))
        self.assertEqual(acceptance([], [])[0], 0)
        recs = (
            [{"arrival_s": t, "ttft_ms": 100.0} for t in range(0, 5)]
            + [{"arrival_s": t, "ttft_ms": 9000.0} for t in range(5, 10)]
            + [{"arrival_s": t, "ttft_ms": 100.0} for t in range(15, 20)]
        )
        rec_s, peak = burst_recovery(recs, sla_ttft_ms=1000.0, window_s=5.0)
        self.assertEqual(peak, 9000.0)
        self.assertIsNotNone(rec_s)
        self.assertGreater(rec_s, 0)
        never = [{"arrival_s": t, "ttft_ms": 9000.0} for t in range(0, 10)]
        self.assertIsNone(burst_recovery(never, 1000.0, 5.0)[0])

    def test_r2_tokenize_and_output_ids_survive_load(self):
        from quettabench.workloads import tokenize as trace_tokenize

        importlib.reload(trace_tokenize)
        recs = [
            {
                "model": "DS",
                "session_id": "s0",
                "turns": [{"turn_index": 0, "prompt_text": "ab", "reasoning_text": "c", "content_text": "d"}],
            }
        ]
        reqs = trace_tokenize.r2_to_requests(recs, encode=lambda s: [ord(ch) for ch in s])
        self.assertEqual(reqs[0]["prompt_token_ids"], [97, 98])
        self.assertEqual(reqs[0]["output_token_ids"], [99, 100])
        self.assertEqual(reqs[0]["output_length"], 2)
        self.assertEqual(reqs[0]["source"], "DS")
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            for r in reqs:
                fh.write(json.dumps(r) + "\n")
            p = fh.name
        loaded = load_trace(p)
        os.unlink(p)
        self.assertEqual(loaded[0]["output_token_ids"], [99, 100])

    def test_trace_shapes_parse(self):

        from quettabench.workloads.grid import take

        tf = tempfile.mktemp()
        open(tf, "w").write(
            json.dumps({"prompt_token_ids": [1, 2, 3, 4, 5]})
            + "\n"
            + json.dumps({"prompt_len": 40})
            + "\n"
            + json.dumps({"input_length": 64, "output_length": 128})
            + "\n"
        )
        tr = load_trace(tf)
        self.assertEqual([r["prompt_len"] for r in tr], [5, 40, 64])
        self.assertEqual(tr[2]["output_length"], 128)
        prompts, cycled = take(tr, 6, 20)
        self.assertTrue(all(len(p) == 20 for p in prompts))
        self.assertTrue(cycled)
        os.remove(tf)

    def test_grid_knows_the_trace_mode(self):
        src = (QB / "workloads" / "grid.py").read_text() + (QB / "config" / "env_map.py").read_text()
        self.assertIn('"trace"', src)
        self.assertIn("QBENCH_TRACE", src)

    def test_baseline_age_ignores_local_timezone(self):
        out = self._table(
            "LOOP ctx=1024 bs=1 kv=fp16 ms_per_step=10.000 k=100\n",
            env={"TZ": "Asia/Tokyo"},
        )
        self.assertIn("(0.0d)", out)


class ConfigTree(unittest.TestCase):
    def test_grid_sizes_and_step_points(self):
        self.assertEqual(len(_config("llama").grid.cells), 15)
        self.assertEqual(len(_config("deepseek").grid.cells), 12)
        self.assertEqual(len(_config("qwen3").grid.cells), 16)
        self.assertEqual(_config("llama").grid.step_points, (60, 230, 400))

    def test_decode_batch_run_cells(self):
        grid = _config("decode_batch_llama").grid
        self.assertEqual(grid.cells, tuple((20000, bs) for bs in (1, 2, 4, 8, 11, 16, 24)))
        self.assertEqual(grid.cells_over_maxlen, ())

    def test_qwen3_resolves_dense_hybrid_kv(self):
        e = _config("qwen3").engine
        # Hybrid stack: KV lives only in the 16 full-attention layers.
        self.assertEqual(
            (e.tp, e.kv_bytes_per_token, e.dtype, e.ep, e.verified), (4, 16 * 2 * 4 * 256 * 2, "bfloat16", False, False)
        )

    def test_pcie_pins_the_collective(self):
        coll = _config("llama", "hardware.link_profile=pcie").hardware.collective
        self.assertEqual(
            set(coll),
            {
                "NCCL_ALGO",
                "NCCL_PROTO",
                "NCCL_P2P_LEVEL",
                "NCCL_IB_DISABLE",
                "NCCL_SOCKET_IFNAME",
                "LLMSRV_NO_NVLS",
                "LLMSRV_TWOSHOT_MIN_BYTES",
            },
        )
        self.assertEqual(
            (coll["NCCL_ALGO"], coll["NCCL_PROTO"], coll["NCCL_P2P_LEVEL"]),
            ("allreduce:tree;allgather:ring", "Simple", "SYS"),
        )

    def test_nvlink_exports_only_ib(self):
        # nvlink: pins come back empty so NCCL keeps NVLS/LL128; only IB stays off.
        coll = _config("llama", "hardware.link_profile=nvlink").hardware.collective
        self.assertEqual({k: v for k, v in coll.items() if v}, {"NCCL_IB_DISABLE": "1"})

    def test_unknown_link_profile_is_refused(self):
        with self.assertRaisesRegex(ConfigError, "unknown link profile 'tcp'"):
            _config("llama", "hardware.link_profile=tcp")

    def test_max_seq_headroom(self):
        h = _config("llama").hardware
        self.assertEqual((h.timing_steps, h.max_seq_headroom), (100, 28))


class TraceWorkloads(unittest.TestCase):
    def test_load_trace_carries_arrivals_and_sessions(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.jsonl"
            p.write_text(
                json.dumps({"input_length": 8, "output_length": 4, "timestamp": 2000, "session_id": "s1"})
                + "\n"
                + json.dumps({"prompt_token_ids": [5, 6, 7], "output_length": 2, "arrival_ts": 3.5})
                + "\n"
            )
            reqs = load_trace(str(p))
        self.assertEqual(reqs[0]["arrival_ts"], 2.0)
        self.assertEqual(reqs[0]["session_id"], "s1")
        self.assertEqual(reqs[0]["output_length"], 4)
        self.assertEqual(reqs[1]["prompt_token_ids"], [5, 6, 7])
        self.assertEqual(reqs[1]["arrival_ts"], 3.5)
        self.assertEqual(reqs[1]["output_length"], 2)

    def test_serve_helpers_are_gpu_free(self):
        # The serving procedure must import without vLLM (CI has no GPU stack).
        from quettabench.metrics.summary import summarize_serve
        from quettabench.modes import serve  # noqa: F401
        from quettabench.workloads.arrival import rebase_arrivals

        self.assertEqual(rebase_arrivals([10.0, 11.0, 14.0], speed=2.0), [0.0, 0.5, 2.0])
        self.assertEqual(rebase_arrivals([None, None], speed=1.0), [0.0, 0.0])
        s = summarize_serve(
            [
                {"ttft_ms": 10.0, "tpot_ms": 5.0, "out_tokens": 3},
                {"ttft_ms": 30.0, "tpot_ms": None, "out_tokens": 1},
                {"ttft_ms": 20.0, "tpot_ms": 7.0, "out_tokens": 3},
            ],
            wall_s=2.0,
            method="nearest",
        )
        self.assertEqual(s["n"], 3)
        self.assertEqual(s["ttft_ms"]["p50"], 20.0)
        self.assertEqual(s["ttft_ms"]["p99"], 30.0)
        self.assertEqual(s["tpot_ms"]["p50"], 5.0)
        self.assertEqual(s["tpot_ms"]["p90"], 7.0)
        self.assertAlmostEqual(s["tok_s"], 3.5)


def text_run(out):
    run = run_grid_cli(
        grid_argv(
            "llama",
            "--set",
            "mode.return_outputs=text",
            "--set",
            f"run_name={out.name}",
            "--results-dir",
            str(out.parent),
        ),
        {},
    )
    assert run["exit"] is None, run["exit"]
    return compare.read_rows(out)


def logit_row(i, argmax, gap, logprob):
    return {"request_index": i, "logits": {"argmax": argmax, "top2_gap": gap, "logprob": logprob}}


REF = [logit_row(0, [5, 6, 7, 8], [2.0, 0.05, 1.0, 3.0], [-0.1, -0.2, -0.3, -0.4])]


class Greedy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as td:
            cls.rows = text_run(Path(td) / "a")

    def test_same_run_agrees_everywhere(self):
        rows = compare.greedy(self.rows, self.rows)
        per_request = [r for r in rows if r["kind"] == "greedy"]
        self.assertEqual(len(per_request), sum(1 for r in self.rows if r.get("token_ids")))
        self.assertTrue(all(r["prefix"] == r["of"] == 64 for r in per_request))
        summary = rows[-1]
        self.assertEqual((summary["kind"], summary["agree4plus"]), ("greedy_summary", summary["prompts"]))

    def test_a_flipped_token_ends_the_prefix(self):
        theirs = json.loads(json.dumps(self.rows))
        first = next(r for r in theirs if r.get("token_ids"))
        first["token_ids"][10] += 1
        rows = {r["request"]: r for r in compare.greedy(self.rows, theirs) if r["kind"] == "greedy"}
        cell = first["cell"]
        hit = rows[f"{cell['ctx']}x{cell['bs']} #0"]
        self.assertEqual((hit["prefix"], hit["of"]), (10, 64))
        self.assertIn(
            "AGREE 10/64 tokens ( 15.6%)", "\n".join(compare.render_agreement(compare.greedy(self.rows, theirs)))
        )

    def test_runs_without_returns_give_no_rows(self):
        self.assertEqual(compare.compare_runs([{"cell": None}], [{"cell": None}]), [])


class Logit(unittest.TestCase):
    def test_identical_logits_pass(self):
        rows = compare.logit(REF, REF)
        self.assertEqual(rows[-1]["verdict"], "PASS")
        self.assertEqual(rows[-1]["delta_max"], 0.0)

    def test_near_tie_flip_passes_and_hard_flip_fails(self):
        near = [logit_row(0, [5, 9, 7, 8], [0, 0, 0, 0], [-0.1, -0.25, -0.3, -0.4])]
        rows = compare.logit(near, REF)
        self.assertEqual((rows[-1]["near_tie_flips"], rows[-1]["verdict"]), (1, "PASS"))
        hard = [logit_row(0, [5, 6, 9, 8], [0, 0, 0, 0], [-0.1, -0.2, -0.3, -0.9])]
        rows = compare.logit(hard, REF)
        self.assertEqual(rows[-1]["verdict"], "FAIL")
        self.assertAlmostEqual(rows[-1]["delta_max"], 0.5)
        text = compare.render_agreement(rows)
        self.assertEqual(text[0], "DISAGREE req 0 pos=2 reference_gap=1.0000 reference=7 ours=9")
        self.assertEqual(text[-1], "FIDELITY verdict=FAIL hard_disagreements=1")

    def test_one_bf16_step_is_a_near_tie(self):
        ref = [logit_row(0, [5, 6], [0.125, 0.126], [-0.1, -0.2])]
        rows = compare.logit([logit_row(0, [9, 6], [0, 0], [-0.1, -0.2])], ref)
        self.assertEqual((rows[-1]["near_tie_flips"], rows[-1]["verdict"]), (1, "PASS"))
        rows = compare.logit([logit_row(0, [5, 9], [0, 0], [-0.1, -0.2])], ref)
        self.assertEqual(rows[-1]["verdict"], "FAIL")


class Band(unittest.TestCase):
    ONE_FLIP = [logit_row(0, [5, 9, 7, 8], [0, 0, 0, 0], [-0.1, -0.2, -0.3, -0.4])]
    TWO_FLIPS = [logit_row(0, [1, 9, 7, 8], [0, 0, 0, 0], [-0.1, -0.2, -0.3, -0.4])]

    def test_inside_the_worst_floor_run_passes(self):
        row = compare.band(self.ONE_FLIP, REF, [REF, self.TWO_FLIPS])
        self.assertEqual((row["flips"], row["floor_flips"], row["verdict"]), (1, [0, 2], "PASS"))
        self.assertEqual(
            compare.render_agreement([row]), ["BAND verdict=PASS flips=1 floor=2 floor_runs=0,2 positions=4"]
        )

    def test_above_the_floor_fails(self):
        row = compare.band(self.TWO_FLIPS, REF, [self.ONE_FLIP])
        self.assertEqual(row["verdict"], "FAIL")

    def test_different_position_counts_are_withheld(self):
        short = [logit_row(0, [5, 6], [0, 0], [-0.1, -0.2])]
        self.assertEqual(compare.band(self.ONE_FLIP, REF, [short])["verdict"], "WITHHELD")


NOW = calendar.timegm(time.strptime("2026-09-21T14:13:20Z", "%Y-%m-%dT%H:%M:%SZ")) + 2.5 * 86400


def golden_pair(log="bench_loop.log", **kw):
    return compare.pair(
        compare.read_bench_log(FIXTURES / log), compare.read_baseline(FIXTURES / "baseline_llama.json"), now=NOW, **kw
    )


def by_cell(report):
    return {(r["ctx"], r["bs"]): r for r in report.rows}


class GoldenRows(unittest.TestCase):
    def test_loop_log_reasons(self):
        report = golden_pair()
        rows = by_cell(report)
        self.assertEqual(rows[(100, 1)]["withheld"], ["kv"])
        self.assertEqual(rows[(4096, 8)]["withheld"], ["kv"])
        # LOW R2, RAW and EAGER are flags only: the ratio stays.
        for cell, flag in (((1024, 16), "LOW R2"), ((4096, 64), "RAW"), ((100, 8), "EAGER")):
            self.assertIn(flag, rows[cell]["flags"])
            self.assertEqual(rows[cell]["withheld"], [])
            self.assertIsNotNone(rows[cell]["ratio"])
        self.assertEqual(report.ours_only, [(2048, 4, 2), (8192, 1, 1)])
        self.assertEqual(report.tps_without_theirs, [2])
        self.assertAlmostEqual(report.age_days, 2.5)

    def test_criterion_log_withholds_on_method(self):
        report = golden_pair("bench_criterion.log")
        self.assertFalse(report.comparable)
        self.assertTrue(all("method" in r["withheld"] for r in report.rows))
        forced = golden_pair("bench_criterion.log", allow_method_mismatch=True)
        self.assertTrue(all(r["ratio"] is not None for r in forced.rows))


class StaleBaseline(unittest.TestCase):
    def test_the_warning_names_a_justfile_recipe(self):
        report = golden_pair()
        report.age_days = compare.STALE_DAYS + 1
        self.assertIn("re-run `just run configs/runs/<wl>.yaml`", compare.render(report))
        justfile = (Path(__file__).resolve().parents[2] / "justfile").read_text()
        self.assertIn("\nrun cfg *args:", justfile)


class WorkloadGate(unittest.TestCase):
    def sides(self, ours_hash, theirs_hash):
        ours = compare.read_bench_log(FIXTURES / "bench_loop.log")
        theirs = compare.read_baseline(FIXTURES / "baseline_llama.json")
        ours.header["workload_hash"], theirs.header["workload_hash"] = ours_hash, theirs_hash
        return ours, theirs

    def test_different_hashes_withhold_every_row(self):
        report = compare.pair(*self.sides("aaaa", "bbbb"), now=NOW)
        self.assertTrue(report.workload_mismatch)
        self.assertTrue(all("workload" in r["withheld"] and r["ratio"] is None for r in report.rows))
        self.assertIn("WORKLOAD MISMATCH engine=aaaa vllm=bbbb", compare.render(report))

    def test_one_missing_hash_joins_on_cells_alone(self):
        for hashes in (("aaaa", None), (None, "bbbb"), ("aaaa", "aaaa")):
            report = compare.pair(*self.sides(*hashes), now=NOW)
            self.assertFalse(report.workload_mismatch)
            self.assertEqual(compare.render(report), compare.render(golden_pair()))


BASELINE = FIXTURES / "baseline_llama.json"


def run(argv, env=None):
    out = io.StringIO()
    with mock.patch.dict(os.environ, env or {}, clear=True), contextlib.redirect_stdout(out):
        compare.main([str(a) for a in argv])
    return out.getvalue()


class Subcommand(unittest.TestCase):
    def test_floor_with_a_file_input_is_a_parse_error(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as stop:
            run([FIXTURES / "bench_loop.log", BASELINE, "--floor", FIXTURES])
        self.assertEqual(stop.exception.code, 2)
        self.assertIn("must be run dirs", err.getvalue())

    def test_json_report(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "r.json"
            run([FIXTURES / "bench_loop.log", BASELINE, "--json", out])
            report = json.loads(out.read_text())
        self.assertEqual(report["theirs"]["header"]["engine"], "vllm")
        self.assertEqual(report["ours"]["header"]["method"], "loop")
        rows = {(r["ctx"], r["bs"]): r for r in report["rows"]}
        self.assertIsNone(rows[(4096, 8)]["ratio"])
        self.assertEqual(rows[(4096, 8)]["withheld"], ["kv"])
        self.assertEqual(report["ours_only"], [[2048, 4, 2], [8192, 1, 1]])

    def test_default_report_path(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(compare, "COMPARE_DIR", Path(td)):
            run([FIXTURES / "bench_loop.log", BASELINE])
            self.assertTrue((Path(td) / "bench_loop_vs_baseline_llama.json").is_file())

    def test_method_mismatch_forced_by_flag_or_env(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [FIXTURES / "bench_criterion.log", BASELINE, "--json", Path(td) / "r.json"]
            self.assertIn("ratios are withheld", run(argv))
            for forced in (run(argv + ["--allow-method-mismatch"]), run(argv, {"QBENCH_ALLOW_METHOD_MISMATCH": "1"})):
                self.assertNotIn("ratios are withheld", forced)
                self.assertIn("1.04x", forced)

    def test_engine_facts_from_env(self):
        with tempfile.TemporaryDirectory() as td:
            text = run(
                [FIXTURES / "bench_loop.log", BASELINE, "--json", Path(td) / "r.json"],
                {"NCCL_ALGO": "Ring", "QBENCH_QUANT": "nvfp4"},
            )
        self.assertIn("NCCL MISMATCH engine=Ring/None vllm=auto/auto", text)
        self.assertIn("QUANT MISMATCH engine=nvfp4 vllm=float16; ratios withheld", text)

    def test_fidelity_or_band_fail_exits_1_after_the_json(self):
        for kind in ("logit_summary", "band"):
            with (
                self.subTest(kind),
                tempfile.TemporaryDirectory() as td,
                mock.patch.object(compare, "read_side"),
                mock.patch.object(compare, "read_rows"),
                mock.patch.object(compare, "render", return_value=""),
                mock.patch.object(compare, "pair", return_value=mock.Mock(rows=[], agreement=[])),
                mock.patch.object(compare, "report_json", return_value={}),
                mock.patch.object(compare, "compare_runs", return_value=[{"kind": kind, "verdict": "FAIL"}]),
            ):
                with self.assertRaises(SystemExit) as e:
                    run([td, td, "--json", Path(td) / "r.json"])
                self.assertEqual(e.exception.code, 1)
                self.assertTrue((Path(td) / "r.json").is_file())

    def test_missing_input(self):
        with self.assertRaises(SystemExit) as e:
            run(["/nonexistent/run", BASELINE])
        self.assertEqual(e.exception.code, "no results at /nonexistent/run")


def side_by_cell(side):
    return {(r["ctx"], r["bs"], r["tp"]): r for r in side.rows}


class BenchLog(unittest.TestCase):
    def test_loop_log(self):
        side = compare.read_bench_log(FIXTURES / "bench_loop.log")
        self.assertEqual(side.header["method"], "loop")
        self.assertEqual(
            side.notes, ["ignored 1 truncated-model batch_bench line(s) (layers != all)", "engine gdn kernel(s): Tiled"]
        )
        cells = side_by_cell(side)
        self.assertEqual(len(cells), 8)
        self.assertEqual(cells[(100, 1, 1)]["group"], "batch_bench")
        self.assertEqual(cells[(100, 1, 1)]["kv"], "?")
        self.assertEqual(cells[(100, 8, 1)]["mode"], "eager")
        self.assertEqual(cells[(2048, 4, 2)]["ms_per_step"], 4.1)
        self.assertTrue(all(r["measured"] for r in side.rows))

    def test_criterion_log(self):
        side = compare.read_bench_log(FIXTURES / "bench_criterion.log")
        self.assertEqual(side.header["method"], "criterion")
        cells = side_by_cell(side)
        self.assertEqual(cells[(100, 1, 1)]["group"], "decode_graph")
        self.assertEqual(cells[(1024, 1, 2)]["group"], "decode_tp2")
        self.assertIsNone(cells[(1024, 1, 1)]["kv"])

    def test_engine_facts_fill_the_header(self):
        side = compare.read_bench_log(FIXTURES / "bench_loop.log", {"quant": "nvfp4", "nccl_algo": "Ring"})
        self.assertEqual((side.header["quant"], side.header["nccl_algo"]), ("nvfp4", "Ring"))


class Baseline(unittest.TestCase):
    def test_golden_baseline(self):
        side = compare.read_baseline(FIXTURES / "baseline_llama.json")
        self.assertEqual(len(side.rows), 15)
        h = side.header
        self.assertEqual(
            (h["engine"], h["version"], h["method"], h["timestamp"]),
            ("vllm", "0.0.0+stub", "slope", "2026-09-21T14:13:20Z"),
        )
        self.assertEqual((h["quant"], h["serving_style"], h["comm_bound_bs"]), ("float16", "aggregated", None))
        raw = side_by_cell(side)[(4096, 64, 1)]
        self.assertIsNone(raw["ms_at_eval"])
        self.assertFalse(raw["measured"])

    def test_age(self):
        side = compare.read_baseline(FIXTURES / "baseline_llama.json")
        self.assertTrue(math.isclose(compare.age_days(side, 1790000000 + 86400), 1.0))


class RunDir(unittest.TestCase):
    def test_bench_run_reads_as_its_log(self):
        for log in ("bench_loop.log", "bench_criterion.log"):
            with self.subTest(log=log), tempfile.TemporaryDirectory() as td:
                run = compare.read_run(bench_run_dir(Path(td) / "qs", FIXTURES / log))
                side = compare.read_bench_log(FIXTURES / log)
                self.assertEqual(run.rows, side.rows)
                self.assertEqual(run.notes, side.notes)
                self.assertEqual(run.header["method"], side.header["method"])

    def test_writer_rows_read_as_hand_written_rows(self):
        def hand_written(out, log):
            out.mkdir(parents=True)
            shutil.copy(log, out / "bench.log")
            rows = list(BenchDriver(out / "bench.log").rows())
            with (out / "requests.jsonl").open("w") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            RunWriter(out, bench_config(out.name)).finish(None, steps=fit_run(rows, 0, 1, 1.0))
            return out

        for log in ("bench_loop.log", "bench_criterion.log"):
            # Clock and commit frozen in the writer's module: a commit mid-test changes bench_sha.
            with (
                self.subTest(log=log),
                tempfile.TemporaryDirectory() as td,
                mock.patch.dict(
                    RunWriter.finish.__globals__,
                    {"_now": lambda: "2026-10-09T00:00:00+00:00", "bench_sha": lambda *a, **k: "0123abc"},
                ),
            ):
                ours = compare.read_run(bench_run_dir(Path(td) / "qs", FIXTURES / log))
                theirs = compare.read_run(hand_written(Path(td) / "hand" / "qs", FIXTURES / log))
                self.assertEqual(ours, theirs)
                self.assertTrue(ours.rows)


if __name__ == "__main__":
    unittest.main()
