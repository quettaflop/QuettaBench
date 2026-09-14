"""Offline checks for scripts/crossval.

CI has no GPU, so this pins the wiring: scripts parse, shell helpers keep
their exec bit, workloads.json stays consistent, the grids cover the cells
the engine bench times, and table.py only compares like with like.
"""

import ast
import json
import os
import re
import subprocess
import unittest
from pathlib import Path

CROSSVAL = Path(__file__).resolve().parents[1] / "scripts" / "crossval"
VERIFIED_CRATE = "llama"


def _scripts(suffix):
    return sorted(p for p in CROSSVAL.rglob(f"*{suffix}") if "__pycache__" not in p.parts)


def _load_workloads():
    return json.loads((CROSSVAL / "workloads.json").read_text())


class CrossvalScripts(unittest.TestCase):
    def test_python_scripts_parse(self):
        # Parse only: importing would pull in vllm, which CI does not have.
        scripts = _scripts(".py")
        self.assertTrue(scripts, "no python scripts under scripts/crossval")
        for script in scripts:
            ast.parse(script.read_text(), filename=str(script))

    def test_shell_scripts_have_valid_syntax(self):
        scripts = _scripts(".sh")
        self.assertTrue(scripts, "no shell scripts under scripts/crossval")
        for script in scripts:
            subprocess.run(["bash", "-n", str(script)], check=True)

    def test_shell_scripts_are_executable(self):
        for script in _scripts(".sh"):
            self.assertTrue(
                os.access(script, os.X_OK),
                f"{script.name} is exec'd directly and must keep its +x bit",
            )

    def test_agreement_inputs_present(self):
        for name in ("prompts.txt", "texts.txt"):
            lines = [l for l in (CROSSVAL / name).read_text().splitlines() if l.strip()]
            self.assertTrue(lines, f"{name} is empty")

    def test_vmin_fit_supports_moe_bringup(self):
        # deepseek needs expert parallel on the vLLM side, and an explicit
        # opt-in to run a workload before it is certified.
        src = (CROSSVAL / "vmin_fit.py").read_text()
        self.assertIn("enable_expert_parallel", src)
        self.assertIn("allow-unverified", src)

    def test_ds_bench_supports_a_prebuilt_binary(self):
        # The air-gapped GPU nodes have no cargo; the deepseek sweep must
        # accept a cross-built test binary.
        self.assertIn("DS_BENCH_BIN", (CROSSVAL / "ds_bench.sh").read_text())


class WorkloadsConfig(unittest.TestCase):
    def test_top_level_shape(self):
        cfg = _load_workloads()
        self.assertIsInstance(cfg.get("step_points"), list)
        self.assertTrue(cfg["step_points"])
        self.assertIsInstance(cfg.get("grids"), dict)
        self.assertIsInstance(cfg.get("workloads"), dict)

    def test_every_workload_names_an_existing_grid(self):
        cfg = _load_workloads()
        grids = cfg["grids"]
        for name, workload in cfg["workloads"].items():
            grid = workload.get("grid")
            self.assertIn(
                grid, grids,
                f"workload {name} names grid {grid!r}, absent from grids",
            )

    def test_verified_crate_grid_is_context_batch_pairs(self):
        cfg = _load_workloads()
        grid_name = cfg["workloads"][VERIFIED_CRATE]["grid"]
        grid = cfg["grids"][grid_name]
        self.assertTrue(grid, f"{VERIFIED_CRATE} grid {grid_name} is empty")
        for cell in grid:
            self.assertEqual(len(cell), 2, f"grid cell {cell} is not (context, batch)")
            self.assertTrue(all(isinstance(x, int) for x in cell))

    def test_tp_suffixed_workloads_declare_matching_tp(self):
        cfg = _load_workloads()
        for name, workload in cfg["workloads"].items():
            m = re.search(r"-tp(\d+)$", name)
            if m:
                self.assertEqual(
                    workload.get("tp"), int(m.group(1)),
                    f"workload {name} must declare tp={m.group(1)}",
                )

    def test_workloads_declare_verified(self):
        for name, workload in _load_workloads()["workloads"].items():
            self.assertIn("verified", workload, f"{name} missing the verified flag")

    def test_deepseek_is_moe_expert_parallel_at_tp8(self):
        # The MoE workload must enable expert parallel (vmin_fit passes it to
        # vLLM) and set tp so it pairs with the engine's DS_WORLD. Its
        # kv_cache_dtype must be a value vLLM accepts, not an engine-side label.
        ds = _load_workloads()["workloads"]["deepseek"]
        self.assertIs(ds.get("expert_parallel"), True)
        self.assertEqual(ds.get("tp"), 8)
        self.assertIn(ds.get("kv_dtype"), {"fp8", "fp8_e4m3", "fp8_e5m2", "auto", None})

    def test_grid_cells_fit_workload_maxlen(self):
        cfg = _load_workloads()
        top = max(cfg["step_points"])
        for name, workload in cfg["workloads"].items():
            for ctx, bs in cfg["grids"][workload["grid"]]:
                self.assertLessEqual(
                    ctx + top + 2, workload["maxlen"],
                    f"workload {name} cell ({ctx}, {bs}) exceeds maxlen "
                    f"{workload['maxlen']}; vmin_fit would SKIP it",
                )


class EngineBenchContract(unittest.TestCase):
    """Cells frozen from QuettaServe llama/benches/latency.rs: the batch
    groups sweep ctx {100, 1024, 4096} x bs {1, 8, 16, 32, 64}; the
    tensor-parallel groups sweep ctx {100, 1024, 4096, 8192} at bs 1. Every
    cell the bench times needs a vLLM cell, or the table drops the row."""

    def test_llama_grid_covers_the_batch_bench_cells(self):
        grid = {tuple(c) for c in _load_workloads()["grids"]["llama"]}
        expected = {(ctx, bs) for ctx in (100, 1024, 4096) for bs in (1, 8, 16, 32, 64)}
        self.assertTrue(expected <= grid, f"missing cells: {sorted(expected - grid)}")

    def test_llama_single_grid_covers_the_tp_bench_cells(self):
        grid = {tuple(c) for c in _load_workloads()["grids"]["llama_single"]}
        expected = {(ctx, 1) for ctx in (100, 1024, 4096, 8192)}
        self.assertTrue(expected <= grid, f"missing cells: {sorted(expected - grid)}")

    def test_deepseek_grid_covers_the_batch_bench_cells(self):
        # deepseek/tests/batch_bench.rs times one (ctx, bs) per invocation;
        # ds_bench.sh reads these cells from workloads.json and loops them, so
        # this grid is the single source the sweep cannot drift from.
        grid = {tuple(c) for c in _load_workloads()["grids"]["deepseek"]}
        expected = {(ctx, bs) for ctx in (1024, 8192, 16384) for bs in (1, 4, 16, 64)}
        self.assertEqual(grid, expected, f"grid drift: {sorted(grid ^ expected)}")


class TableParity(unittest.TestCase):
    """table.py prints ratios only when both sides timed the same quantity:
    the engine's pipelined decode_loop against the baseline's slope. A
    criterion per-step log gets its cells printed with ratios withheld, and
    rows whose KV formats differ carry a KV flag."""

    def _table(self, bench_text, kv="float16", env=None, expect_rc=0):
        import sys
        import tempfile
        import time

        base = {
            "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu": "test", "vllm": "0", "mode": "FULL", "model": "m",
            "dtype": "float16", "grid": "g", "method": "slope", "kv_dtype": kv,
            "points": [{"ctx": 1024, "bs": 1, "tp": 1, "ms_per_step": 5.0,
                        "tok_s": 200.0, "r2": 1.0, "kv": kv}],
        }
        with tempfile.TemporaryDirectory() as td:
            bench = Path(td) / "bench.log"
            cache = Path(td) / "base.json"
            bench.write_text(bench_text)
            cache.write_text(json.dumps(base))
            out = subprocess.run(
                [sys.executable, str(CROSSVAL / "table.py"), str(bench), str(cache)],
                capture_output=True, text=True, env={**os.environ, **(env or {})},
            )
        self.assertEqual(out.returncode, expect_rc, out.stderr)
        return out.stdout

    def test_loop_bench_gets_a_ratio(self):
        out = self._table("LOOP ctx=1024 bs=1 kv=fp16 ms_per_step=10.000 k=100\n")
        self.assertIn("0.50x", out)
        self.assertNotIn("withheld", out)

    def test_per_step_bench_gets_no_ratio(self):
        out = self._table(
            "decode_batch_paged/1024x1\n"
            "                        time:   [9.9 ms 10.0 ms 10.1 ms]\n"
        )
        self.assertIn("withheld", out)
        self.assertNotIn("0.50x", out)

    def test_kv_mismatch_is_flagged(self):
        out = self._table("LOOP ctx=1024 bs=1 kv=nvfp4 ms_per_step=10.000 k=100\n")
        self.assertIn("KV nvfp4/fp16", out)

    def test_deepseek_kv_difference_is_flagged(self):
        # The engine caches the MLA latent in bf16, the vLLM fork in fp8; the
        # methods still match (both marginal decode), so the row prints a ratio
        # and the KV difference is flagged rather than hidden.
        out = self._table(
            "LOOP ctx=1024 bs=1 kv=bf16 ms_per_step=10.000 k=100\n", kv="fp8"
        )
        self.assertIn("KV bf16/fp8", out)
        self.assertIn("0.50x", out)

    BB = (
        "[batch_bench] world=1 bs=1 max_seq=1636 layers=all prompt=1024 "
        "steps=92: median {ms} ms/step, mean 10.100, best 9.900 "
        "-> 100.0 tok/s (median)\n"
    )

    def test_stock_deepseek_bench_line_gets_a_ratio_and_kv_flag(self):
        # The stock engine bench line is parsed directly (no engine patch);
        # its median is the marginal step, so the ratio prints, and the
        # unstated KV format is flagged as ? against the baseline's.
        out = self._table(self.BB.format(ms="10.000"), kv="fp8")
        self.assertIn("0.50x", out)
        self.assertIn("KV ?/fp8", out)
        self.assertIn("batch_bench", out)

    def test_loop_line_supersedes_the_bench_summary(self):
        out = self._table(
            self.BB.format(ms="20.000")
            + "LOOP ctx=1024 bs=1 kv=fp16 ms_per_step=10.000 k=100\n"
        )
        self.assertIn("10.000", out)
        self.assertNotIn("20.000", out)

    def test_truncated_model_bench_lines_are_ignored(self):
        # A DS_LAYERS run times a different model; comparing it against the
        # full-model baseline would be nonsense, so it never becomes a row.
        line = self.BB.format(ms="10.000").replace("layers=all", "layers=6")
        out = self._table(line, expect_rc=1)
        self.assertIn("truncated", out)
        self.assertNotIn("10.000", out)

    def test_h200_grid_is_the_table(self):
        grid = {tuple(c) for c in _load_workloads()["grids"]["table_h200"]}
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

    def test_baseline_age_ignores_local_timezone(self):
        out = self._table(
            "LOOP ctx=1024 bs=1 kv=fp16 ms_per_step=10.000 k=100\n",
            env={"TZ": "Asia/Tokyo"},
        )
        self.assertIn("(0.0d)", out)


if __name__ == "__main__":
    unittest.main()
