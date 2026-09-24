"""Offline checks for scripts/crossval.

CI has no GPU, so this pins the wiring: scripts parse, shell helpers keep
their exec bit, workloads.json stays consistent, the grids cover the cells
the engine bench times, and table.py only compares like with like.
"""

import ast
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
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

    def test_shell_scripts_do_not_inline_python(self):
        for sh in _scripts(".sh"):
            self.assertNotIn("python3 -c", sh.read_text(), f"{sh.name} inlines python")

    def test_xval_wires_serving_and_identity_gate(self):
        # The orchestrator runs a trace serving replay with the workload-identity
        # gate when XVAL_SERVE_TRACE is set.
        text = (CROSSVAL / "xval.sh").read_text()
        self.assertIn("XVAL_SERVE_TRACE", text)
        self.assertIn("trace_serve.py", text)
        self.assertIn("synth_bench.py", text)
        self.assertIn("WORKLOADSUM", text)

    def test_backend_detection_and_no_hard_nvidia_smi(self):
        # backend() honors XVAL_BACKEND and reports cpu when no accel tooling is on
        # PATH; the device picker and link detection must not hard-require nvidia-smi.
        env = {k: v for k, v in os.environ.items() if k != "XVAL_BACKEND"}
        env["PATH"] = "/nonexistent"
        out = subprocess.run(
            [sys.executable, str(CROSSVAL / "xval_config.py"), "backend"],
            capture_output=True, text=True, env=env).stdout.strip()
        self.assertEqual(out, "cpu")
        out = subprocess.run(
            [sys.executable, str(CROSSVAL / "xval_config.py"), "backend"],
            capture_output=True, text=True, env={**os.environ, "XVAL_BACKEND": "tpu"}).stdout.strip()
        self.assertEqual(out, "tpu")
        # free_gpu.sh on a non-cuda backend returns devices without calling nvidia-smi
        fg = subprocess.run(["bash", str(CROSSVAL / "free_gpu.sh"), "2"],
                            capture_output=True, text=True,
                            env={**os.environ, "XVAL_BACKEND": "tpu"})
        self.assertEqual(fg.returncode, 0, fg.stderr)
        self.assertEqual(fg.stdout.strip(), "0,1")

    def test_pp_wired_in_grid_and_device_count(self):
        # vmin_fit passes pipeline_parallel_size (from XVAL_PP or the workload's pp),
        # and the device count the orchestrator provisions is tp * pp.
        src = (CROSSVAL / "vmin_fit.py").read_text()
        self.assertIn("pipeline_parallel_size", src)
        self.assertIn("XVAL_PP", src)
        self.assertIn("META pp=", src)  # cache.py parses META one key per line
        d1 = subprocess.run([sys.executable, str(CROSSVAL / "xval_config.py"), "devices", "deepseek"],
                            capture_output=True, text=True, check=True).stdout.strip()
        self.assertEqual(d1, "8")  # deepseek tp8, pp1
        d2 = subprocess.run([sys.executable, str(CROSSVAL / "xval_config.py"), "devices", "deepseek"],
                            capture_output=True, text=True,
                            env={**os.environ, "XVAL_PP": "2"}).stdout.strip()
        self.assertEqual(d2, "16")  # tp8 * pp2

    def test_link_profile_cli(self):
        # One detection for bench.sh, vllm.sh and vmin_fit: caller override wins,
        # non-cuda backends take the no-pins nvlink profile.
        out = subprocess.run(
            [sys.executable, str(CROSSVAL / "xval_config.py"), "link-profile"],
            capture_output=True, text=True,
            env={**os.environ, "XVAL_LINK_PROFILE": "pcie"}).stdout.strip()
        self.assertEqual(out, "pcie")
        env = {k: v for k, v in os.environ.items() if k != "XVAL_LINK_PROFILE"}
        env["XVAL_BACKEND"] = "tpu"
        out = subprocess.run(
            [sys.executable, str(CROSSVAL / "xval_config.py"), "link-profile"],
            capture_output=True, text=True, env=env).stdout.strip()
        self.assertEqual(out, "nvlink")

    def test_provision_cli(self):
        # bootstrap.sh reads provisioning defaults through the config CLI.
        out = subprocess.run(
            [sys.executable, str(CROSSVAL / "xval_config.py"), "provision", "cache_root"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        self.assertEqual(out, "/workspace/.xval-cache")

    def test_bootstrap_dry_run_plans_without_writing(self):
        # XVAL_DRYRUN prints the plan and must not create .xval_env.
        envfile = CROSSVAL / ".xval_env"
        self.assertFalse(envfile.exists(), "stale .xval_env in the tree")
        out = subprocess.run(
            ["bash", str(CROSSVAL / "bootstrap.sh"), "qwen3"],
            capture_output=True, text=True,
            env={**os.environ, "XVAL_DRYRUN": "1"},
        )
        self.assertRegex(out.stderr, r"would:|MISSING:")
        self.assertFalse(envfile.exists(), "dry run must not write .xval_env")

    def test_lss_export_covers_or_fails_loudly(self):
        # Full probe coverage reproduces the reference key set; one missing cell
        # must exit 2 with a MISSING line, never an interpolated row.
        with tempfile.TemporaryDirectory() as td:
            ref = Path(td) / "H200" / "meta-llama" / "Llama-3.1-8B" / "bf16"
            (ref / "tp1").mkdir(parents=True)
            (ref / "tp1" / "dense.csv").write_text(
                "layer,tokens,time_us\nact_fn,1,3.3\nact_fn,2,3.4\n")
            (ref / "tp1" / "per_sequence.csv").write_text(
                "layer,sequences,time_us\nlm_head,1,248.0\n")
            (ref / "meta.yaml").write_text("hardware: H200\nmodel: m\n")
            probes = Path(td) / "probes.jsonl"
            recs = [
                {"table": "dense", "tp": 1, "layer": "act_fn", "tokens": 1, "time_us": 3.31},
                {"table": "dense", "tp": 1, "layer": "act_fn", "tokens": 1, "time_us": 3.39},
                {"table": "dense", "tp": 1, "layer": "act_fn", "tokens": 2, "time_us": 3.44},
                {"table": "per_sequence", "tp": 1, "layer": "lm_head", "sequences": 1,
                 "time_us": 250.0},
            ]
            probes.write_text("".join(json.dumps(r) + "\n" for r in recs))
            out = Path(td) / "out"
            full = subprocess.run(
                [sys.executable, str(CROSSVAL / "lss_export.py"), "--probes", str(probes),
                 "--reference", str(ref), "--out", str(out)],
                capture_output=True, text=True)
            self.assertEqual(full.returncode, 0, full.stdout + full.stderr)
            tree = out / "profiler" / "perf" / "H200" / "meta-llama" / "Llama-3.1-8B" / "bf16"
            dense = (tree / "tp1" / "dense.csv").read_text().splitlines()
            self.assertEqual(dense[0], "layer,tokens,time_us")
            self.assertEqual(dense[1], "act_fn,1,3.35")  # median of the duplicates
            self.assertTrue((tree / "meta.yaml").exists())
            probes.write_text(json.dumps(recs[0]) + "\n")  # drop coverage
            gap = subprocess.run(
                [sys.executable, str(CROSSVAL / "lss_export.py"), "--probes", str(probes),
                 "--reference", str(ref), "--out", str(out)],
                capture_output=True, text=True)
            self.assertEqual(gap.returncode, 2, gap.stdout + gap.stderr)
            self.assertIn("MISSING table=dense tp=1 layer=act_fn tokens=2", gap.stdout)
            self.assertIn("MISSING", (out / "gaps.txt").read_text())

    def test_lss_first_token_fit(self):
        # A constant live-minus-sim offset must be recovered exactly and the
        # corrected TTFT error must collapse; the join is by request id, not row
        # order, and rides on matching input token counts.
        with tempfile.TemporaryDirectory() as td:
            live = Path(td) / "bench_dir"
            live.mkdir()
            sim_rows = ["instance id,request id,model,input,output,arrival,end_time,"
                        "latency,queuing_delay,TTFT,TPOT,ITL"]
            with open(live / "requests.jsonl", "w") as fh:
                for i in range(10):
                    sim_ttft_ms = 10.0 + i
                    toks = 256 * (i + 1)
                    fh.write(json.dumps({
                        "request_id": f"bench-{i}", "input_toks": toks, "output_toks": 8,
                        "queued_ts": 1000.0 + i,
                        "first_token_ts": 1000.0 + i + (sim_ttft_ms + 6.3) / 1e3,
                        "last_token_ts": 1000.0 + i + 1.0}) + "\n")
                    sim_rows.append(f"0,{i},m,{toks},8,0,0,0,0,{sim_ttft_ms * 1e6:.0f},0,\"[]\"")
            sim = Path(td) / "sim.csv"
            sim.write_text("\n".join(sim_rows) + "\n")
            meta = Path(td) / "meta.yaml"
            meta.write_text("hardware: H200\n")
            out = subprocess.run(
                [sys.executable, str(CROSSVAL / "lss_first_token.py"), "--live", str(live),
                 "--sim", str(sim), "--meta", str(meta)],
                capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
            self.assertIn("overhead_ms=6.300", out.stdout)
            self.assertIn("after_mape=0.0%", out.stdout)
            self.assertIn("first_token_overhead_us: 6300", meta.read_text())

    def test_agreement_inputs_present(self):
        for name in ("prompts.txt", "texts.txt"):
            lines = [l for l in (CROSSVAL / name).read_text().splitlines() if l.strip()]
            self.assertTrue(lines, f"{name} is empty")

    def test_vmin_fit_supports_moe_bringup(self):
        # expert parallel on vLLM, plus an opt-in to run before certified.
        src = (CROSSVAL / "vmin_fit.py").read_text()
        self.assertIn("enable_expert_parallel", src)
        self.assertIn("allow-unverified", src)

    def _bench(self, workload, prefix, stub_body, env=None):
        import subprocess as sp

        with tempfile.TemporaryDirectory() as td:
            stub = Path(td) / "stub"
            stub.write_text("#!/usr/bin/env bash\n" + stub_body + "\n")
            stub.chmod(0o755)
            log = Path(td) / "sweep.log"
            run_env = {**os.environ, f"{prefix}_CKPT": "/dev/null",
                       f"{prefix}_BENCH_BIN": str(stub), "QS_DIR": td}
            run_env.pop("QS_KERNELS", None)
            run_env.pop(f"{prefix}_MODE", None)
            run_env.update(env or {})
            out = sp.run(
                ["bash", str(CROSSVAL / "bench.sh"), workload, str(log)],
                capture_output=True, text=True, cwd=td, env=run_env,
            )
            return out, log.read_text() if log.exists() else ""

    BENCH_LINE = (
        'echo "[batch_bench] world=${DS_WORLD} bs=${DS_BATCH}'
        ' max_seq=${DS_MAX_SEQ} layers=${DS_LAYERS:-all}'
        ' prompt=${DS_BENCH_PROMPT} steps=92: median 1.000 ms/step,'
        ' mean 1.0, best 1.0 -> 1.0 tok/s (median)"'
    )

    def test_bench_runs_the_deepseek_grid_at_the_workload_tp(self):
        # world from the workload tp, every cell runs, DS_LAYERS blocked.
        out, log = self._bench("deepseek", "DS", self.BENCH_LINE,
                               env={"DS_CFG": "/dev/null", "DS_LAYERS": "6"})
        self.assertEqual(out.returncode, 0, out.stderr)
        cfg = _load_workloads()
        tp = cfg["workloads"]["deepseek"]["tp"]
        cells = re.findall(r"^\[batch_bench\] world=(\d+) .*layers=(\w+)", log, re.M)
        self.assertEqual(len(cells), len(cfg["grids"]["deepseek"]))
        self.assertEqual(set(cells), {(str(tp), "all")})

    def test_bench_fails_when_a_cell_produces_no_line(self):
        out, _ = self._bench("deepseek", "DS", "exit 0", env={"DS_CFG": "/dev/null"})
        self.assertEqual(out.returncode, 1)
        self.assertIn("FAIL", out.stderr)

    QWEN_LINE = (
        'echo "LOOP ctx=${QW_BENCH_PROMPT} bs=${QW_BATCH} kv=bf16'
        ' ms_per_step=1.000 gdn=${QS_KERNELS} mode=${QW_MODE} k=100"'
    )

    def test_bench_qwen_defaults_exact_to_eager(self):
        # exact has no graph wiring, so with cubins present the driver must
        # pick the eager step and stamp kern/mode into every cell.
        out, log = self._bench("qwen3", "QW", self.QWEN_LINE,
                               env={"QS_VLLM_GDN_DIR": "/dev/null"})
        self.assertEqual(out.returncode, 0, out.stderr)
        cfg = _load_workloads()
        modes = re.findall(r"^LOOP .*mode=(\w+)", log, re.M)
        self.assertEqual(len(modes), len(cfg["grids"]["qwen3"]))
        self.assertEqual(set(modes), {"eager"})
        self.assertIn("kern=exact", log)

    def test_bench_requires_a_bench_section(self):
        out, _ = self._bench("llama", "XX", "exit 0")
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("bench", out.stderr + out.stdout)


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
        # kv_dtype is the shipped MLA layout; kv_bytes = (512+64) x 43 layers.
        ds = _load_workloads()["workloads"]["deepseek"]
        self.assertIs(ds.get("expert_parallel"), True)
        self.assertEqual(ds.get("tp"), 8)
        self.assertEqual(ds.get("kv_dtype"), "fp8_ds_mla")
        self.assertEqual(ds.get("kv_bytes"), (512 + 64) * 43)

    def test_qwen3_is_dense_no_expert_parallel(self):
        # Dense model: no expert_parallel, tp 4, bfloat16.
        qw = _load_workloads()["workloads"]["qwen3"]
        self.assertNotIn("expert_parallel", qw)
        self.assertEqual(qw.get("tp"), 4)
        # Hybrid stack: KV lives only in the 16 full-attention layers:
        # 16 layers x 2 (K+V) x 4 kv heads x 256 head_dim x 2 bytes (bf16).
        self.assertEqual(qw.get("kv_bytes"), 16 * 2 * 4 * 256 * 2)
        self.assertEqual(qw.get("dtype"), "bfloat16")
        self.assertEqual(qw.get("weights_gib"), 54)
        self.assertEqual(qw.get("maxlen"), 40960)
        self.assertFalse(qw.get("verified"))

    def test_qwen3_grid_is_16_cells(self):
        # ctx {1024,4096,8192,16384} x bs {1,4,16,64} = 16 cells.
        grid = _load_workloads()["grids"]["qwen3"]
        self.assertEqual(len(grid), 16)
        expected = {(c, b) for c in (1024, 4096, 8192, 16384) for b in (1, 4, 16, 64)}
        self.assertEqual({tuple(c) for c in grid}, expected)

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
        # bench.sh loops these cells from workloads.json.
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
        import time

        base = {
            "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu": "test", "vllm": "0", "mode": "FULL", "model": "m",
            "dtype": "float16", "grid": "g", "method": "slope", "kv_dtype": kv,
            "nccl_algo": "allreduce:tree;allgather:ring", "nccl_proto": "Simple",
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

    def test_eager_loop_row_is_grouped_and_flagged(self):
        # eager rows keep the ratio but get their own group and an EAGER flag.
        out = self._table(
            "LOOP ctx=1024 bs=1 kv=fp16 ms_per_step=10.000 gdn=Vllm mode=eager\n"
        )
        self.assertIn("decode_loop_eager", out)
        self.assertIn("EAGER", out)
        self.assertIn("0.50x", out)
        self.assertIn("engine gdn kernel(s): Vllm", out)

    def test_graph_loop_row_has_no_eager_flag(self):
        out = self._table(
            "LOOP ctx=1024 bs=1 kv=fp16 ms_per_step=10.000 gdn=Tiled mode=graph\n"
        )
        self.assertNotIn("EAGER", out)
        self.assertNotIn("decode_loop_eager", out)

    def test_bench_script_wires_the_gdn_family(self):
        # The gdn family owns the kernel env and the eager default for exact.
        text = (CROSSVAL / "bench.sh").read_text()
        self.assertIn("QS_KERNELS=exact", text)
        self.assertIn("MODE=eager", text)
        self.assertIn("mode=$MODE", text)

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
        # bf16 engine vs fp8 baseline: KV flagged, ratio withheld (unofficial).
        out = self._table(
            "LOOP ctx=1024 bs=1 kv=bf16 ms_per_step=10.000 k=100\n", kv="fp8"
        )
        self.assertIn("KV bf16/fp8", out)
        self.assertNotIn("0.50x", out)
        self.assertIn("--", out)

    BB = (
        "[batch_bench] world=1 bs=1 max_seq=1636 layers=all prompt=1024 "
        "steps=92: median {ms} ms/step, mean 10.100, best 9.900 "
        "-> 100.0 tok/s (median)\n"
    )

    def test_stock_deepseek_bench_line_gets_kv_flag_and_withheld_ratio(self):
        # batch_bench line parsed directly: unstated KV is ?, ratio withheld.
        out = self._table(self.BB.format(ms="10.000"), kv="fp8")
        self.assertNotIn("0.50x", out)
        self.assertIn("KV ?/fp8", out)
        self.assertIn("batch_bench", out)
        self.assertIn("--", out)

    def test_loop_line_supersedes_the_bench_summary(self):
        out = self._table(
            self.BB.format(ms="20.000")
            + "LOOP ctx=1024 bs=1 kv=fp16 ms_per_step=10.000 k=100\n"
        )
        self.assertIn("10.000", out)
        self.assertNotIn("20.000", out)

    def test_truncated_model_bench_lines_are_ignored(self):
        # truncated run is dropped and reported plainly, not as a mismatch.
        line = self.BB.format(ms="10.000").replace("layers=all", "layers=6")
        out = self._table(line, expect_rc=1)
        self.assertIn("truncated", out)
        self.assertNotIn("10.000", out)
        self.assertNotIn("synchronized step", out)

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

    def test_comm_bound_cell_is_flagged(self):
        # comm_bound_bs comes from the workload record (deepseek sets 64);
        # the model field carries the display name, matched via workloads().
        import sys
        import time

        base = {
            "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu": "test", "vllm": "0", "mode": "FULL",
            "model": "DeepSeek-V4-Flash-0731",
            "dtype": "float16", "grid": "g", "method": "slope", "kv_dtype": "float16",
            "nccl_algo": "allreduce:tree;allgather:ring", "nccl_proto": "Simple",
            "points": [{"ctx": 1024, "bs": 64, "tp": 1, "ms_per_step": 5.0,
                        "tok_s": 12800.0, "r2": 1.0, "kv": "float16"}],
        }
        bench_text = "LOOP ctx=1024 bs=64 kv=fp16 ms_per_step=10.000 k=100\n"
        with tempfile.TemporaryDirectory() as td:
            bench = Path(td) / "bench.log"
            cache = Path(td) / "base.json"
            bench.write_text(bench_text)
            cache.write_text(json.dumps(base))
            out = subprocess.run(
                [sys.executable, str(CROSSVAL / "table.py"), str(bench), str(cache)],
                capture_output=True, text=True, env=os.environ,
            )
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("COMM", out.stdout)

    def test_no_comm_flag_without_workload_comm_bound(self):
        # Absent comm_bound_bs means never comm-bound: a tp1 llama-class row
        # at bs=64 must keep its ratio instead of inventing a threshold.
        import sys
        import time

        base = {
            "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu": "test", "vllm": "0", "mode": "FULL",
            "model": "Llama-3.1-8B-Instruct",
            "dtype": "float16", "grid": "g", "method": "slope", "kv_dtype": "float16",
            "nccl_algo": "allreduce:tree;allgather:ring", "nccl_proto": "Simple",
            "points": [{"ctx": 1024, "bs": 64, "tp": 1, "ms_per_step": 5.0,
                        "tok_s": 12800.0, "r2": 1.0, "kv": "float16"}],
        }
        bench_text = "LOOP ctx=1024 bs=64 kv=fp16 ms_per_step=10.000 k=100\n"
        with tempfile.TemporaryDirectory() as td:
            bench = Path(td) / "bench.log"
            cache = Path(td) / "base.json"
            bench.write_text(bench_text)
            cache.write_text(json.dumps(base))
            out = subprocess.run(
                [sys.executable, str(CROSSVAL / "table.py"), str(bench), str(cache)],
                capture_output=True, text=True, env=os.environ,
            )
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertNotIn("COMM", out.stdout)
        self.assertIn("0.50x", out.stdout)

    def test_vmin_fit_prompt_modes(self):
        # clone must reproduce the engine's synthetic prompt so both sides
        # route alike; all three regimes must be selectable.
        src = (CROSSVAL / "vmin_fit.py").read_text()
        self.assertIn("(i * 137 + 11) % 100_000", src)
        self.assertIn("prompt_mode", src)
        for mode in ('"clone"', '"corpus"', '"distinct"'):
            self.assertIn(mode, src)
        self.assertIn("dump_tokens", src.replace("-", "_"))

    def test_cache_records_prompt_mode(self):
        # A baseline without its routing regime is not comparable later.
        self.assertIn("prompt_mode", (CROSSVAL / "cache.py").read_text())
        self.assertIn("prompt_mode", (CROSSVAL / "table.py").read_text())

    def test_corpus_mode_requires_an_explicit_corpus(self):
        # No experimental text ships in the repo; corpus mode takes XVAL_CORPUS.
        self.assertFalse((CROSSVAL / "routing_corpus.txt").exists())
        self.assertIn("XVAL_CORPUS", (CROSSVAL / "vmin_fit.py").read_text())

    def test_workload_identity_hash(self):
        # The gate: same requests -> same hash regardless of order; a dropped or
        # altered request changes it. This is what proves systems saw equal work.
        sys.path.insert(0, str(CROSSVAL))
        try:
            import synth_bench
            importlib.reload(synth_bench)
            reqs = synth_bench.swebench_requests(12, seed=2)
            h = synth_bench.workload_hash(reqs)
            self.assertEqual(h, synth_bench.workload_hash(list(reversed(reqs))),
                             "hash must be order-independent")
            self.assertNotEqual(h, synth_bench.workload_hash(reqs[:-1]),
                                "dropping a request must change the hash")
            altered = [dict(r) for r in reqs]
            altered[0] = {**altered[0], "output_length": altered[0]["output_length"] + 1}
            self.assertNotEqual(h, synth_bench.workload_hash(altered),
                                "changing requested output must change the hash")
            out = subprocess.run(
                [sys.executable, str(CROSSVAL / "synth_bench.py"), "hash", "/dev/stdin"],
                input="".join(json.dumps(r) + "\n" for r in reqs),
                capture_output=True, text=True,
            )
            self.assertEqual(out.stdout.strip(), h, out.stderr)
        finally:
            sys.path.pop(0)

    def test_seed_deterministic_and_labeled(self):
        # Same seed + profile is byte-identical, labeled SYNTHETIC, carries the
        # seed; a different seed or arrival time changes the workload hash.
        def gen(seed):
            fd, p = tempfile.mkstemp(suffix=".jsonl"); os.close(fd)
            subprocess.run([sys.executable, str(CROSSVAL / "synth_bench.py"), "gen",
                            "--profile", "terminal_bench", "--n", "8", "--seed", str(seed),
                            "--out", p], check=True, capture_output=True)
            return p
        a, b, c = gen(1), gen(1), gen(2)
        self.assertEqual(Path(a).read_bytes(), Path(b).read_bytes(),
                         "same seed must produce a byte-identical trace")
        rows = [json.loads(l) for l in open(a)]
        self.assertTrue(all(r["source"] == "SYNTHETIC" for r in rows))
        self.assertTrue(all(r["seed"] == 1 for r in rows))
        sys.path.insert(0, str(CROSSVAL))
        try:
            import synth_bench
            importlib.reload(synth_bench)
            self.assertNotEqual(synth_bench.workload_hash(synth_bench.load_trace(a)),
                                synth_bench.workload_hash(synth_bench.load_trace(c)),
                                "a different seed must change the hash")
            reqs = synth_bench.load_trace(a)
            bumped = [dict(x) for x in reqs]
            bumped[0]["arrival_ts"] = bumped[0].get("arrival_ts", 0.0) + 1.0
            self.assertNotEqual(synth_bench.workload_hash(reqs),
                                synth_bench.workload_hash(bumped),
                                "arrival time is a hash input")
        finally:
            sys.path.pop(0)
        for p in (a, b, c):
            os.unlink(p)

    def test_profiles_have_distinct_signatures(self):
        # Each profile must be a distinct serving signature, not a renamed copy:
        # distinct burst and gap specs, and a distinct realized arrival spread.
        sys.path.insert(0, str(CROSSVAL))
        try:
            import synth_bench
            importlib.reload(synth_bench)
            specs = synth_bench.AGENTIC
            self.assertGreater(len({s["burst"] for s in specs.values()}), 1)
            self.assertGreater(len({s["gap"] for s in specs.values()}), 1)
            def span(rs):
                return max(r["arrival_ts"] for r in rs) - min(r["arrival_ts"] for r in rs)
            dr = synth_bench.agentic_requests("deep_research", 40, seed=1)
            tb = synth_bench.agentic_requests("terminal_bench", 40, seed=1)
            self.assertGreater(span(dr), span(tb),
                               "deep_research must arrive over a longer span than terminal_bench")
        finally:
            sys.path.pop(0)

    def test_serving_emits_verdicts(self):
        text = (CROSSVAL / "trace_serve.py").read_text()
        for tag in ("--sla-ttft-ms", "--sla-tpot-ms", "SLASUM", "BURSTSUM", "SPECSUM"):
            self.assertIn(tag, text)
        self.assertIn("FIDELITY", (CROSSVAL / "xval.sh").read_text())

    def test_burst_recovery_and_acceptance(self):
        sys.path.insert(0, str(CROSSVAL))
        try:
            import trace_serve
            importlib.reload(trace_serve)
            self.assertEqual(trace_serve._acceptance([1, 2, 3, 4], [1, 2, 9, 4]), (2, 0.5))
            self.assertEqual(trace_serve._acceptance([], [])[0], 0)
            recs = ([{"arrival_s": t, "ttft_ms": 100.0} for t in range(0, 5)]
                    + [{"arrival_s": t, "ttft_ms": 9000.0} for t in range(5, 10)]
                    + [{"arrival_s": t, "ttft_ms": 100.0} for t in range(15, 20)])
            rec_s, peak = trace_serve.burst_recovery(recs, sla_ttft_ms=1000.0, window_s=5.0)
            self.assertEqual(peak, 9000.0)
            self.assertIsNotNone(rec_s)
            self.assertGreater(rec_s, 0)
            never = [{"arrival_s": t, "ttft_ms": 9000.0} for t in range(0, 10)]
            self.assertIsNone(trace_serve.burst_recovery(never, 1000.0, 5.0)[0])
        finally:
            sys.path.pop(0)

    def test_r2_tokenize_and_output_ids_survive_load(self):
        # The real-text path: agent-bench turns -> requests carrying output_token_ids
        # (reasoning+content), and load_trace must preserve them or SPECSUM can never
        # fire. encode is stubbed so the test needs no tokenizer or GPU.
        sys.path.insert(0, str(CROSSVAL))
        try:
            import trace_tokenize, synth_bench
            importlib.reload(trace_tokenize)
            importlib.reload(synth_bench)
            recs = [{"model": "DS", "session_id": "s0", "turns": [
                {"turn_index": 0, "prompt_text": "ab", "reasoning_text": "c", "content_text": "d"}]}]
            reqs = trace_tokenize.r2_to_requests(recs, encode=lambda s: [ord(ch) for ch in s])
            self.assertEqual(reqs[0]["prompt_token_ids"], [97, 98])
            self.assertEqual(reqs[0]["output_token_ids"], [99, 100])  # reasoning + content
            self.assertEqual(reqs[0]["output_length"], 2)
            self.assertEqual(reqs[0]["source"], "DS")
            with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
                for r in reqs:
                    fh.write(json.dumps(r) + "\n")
                p = fh.name
            loaded = synth_bench.load_trace(p)
            os.unlink(p)
            self.assertEqual(loaded[0]["output_token_ids"], [99, 100])
        finally:
            sys.path.pop(0)

    def test_agentic_profiles(self):
        # Every agentic profile is deterministic, multi-turn, session-grouped,
        # in-range, and round-trips through load_trace.
        sys.path.insert(0, str(CROSSVAL))
        try:
            import synth_bench
            importlib.reload(synth_bench)
            for name, spec in synth_bench.AGENTIC.items():
                a = synth_bench.agentic_requests(name, 40, seed=3)
                b = synth_bench.agentic_requests(name, 40, seed=3)
                self.assertEqual([r["prompt_token_ids"] for r in a],
                                 [r["prompt_token_ids"] for r in b], f"{name} not deterministic")
                self.assertTrue(any(r["turn"] > 0 for r in a), f"{name} has no multi-turn session")
                self.assertGreater(len({r["session_id"] for r in a}), 0)
                for r in a:
                    self.assertGreaterEqual(r["output_length"], spec["output"][0])
                    self.assertLessEqual(r["output_length"], spec["output"][1])
                    if spec["image"]:
                        self.assertEqual(r.get("image_tokens"), spec["image"])
                with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
                    for r in a:
                        fh.write(json.dumps(r) + "\n")
                    p = fh.name
                loaded = synth_bench.load_trace(p)
                os.unlink(p)
                self.assertEqual(loaded[0]["turn"], a[0]["turn"])
                self.assertEqual(loaded[0].get("session_id"), a[0]["session_id"])
        finally:
            sys.path.pop(0)

    def test_synth_bench_distinct_and_trace(self):
        # The synthetic runner must produce DISTINCT per-request streams (so MoE
        # routing spreads, unlike clone) and parse the three trace shapes.
        sys.path.insert(0, str(CROSSVAL))
        try:
            import synth_bench
            importlib.reload(synth_bench)
            reqs = synth_bench.synth_requests(4, 32, seed=3)
            ids = [tuple(r["prompt_token_ids"]) for r in reqs]
            self.assertEqual(len(set(ids)), 4, "synth requests must be distinct")
            self.assertTrue(all(r["prompt_len"] == 32 for r in reqs))
            import json as _j, tempfile, os as _os
            tf = tempfile.mktemp()
            open(tf, "w").write(
                _j.dumps({"prompt_token_ids": [1, 2, 3, 4, 5]}) + "\n"
                + _j.dumps({"prompt_len": 40}) + "\n"
                + _j.dumps({"input_length": 64, "output_length": 128}) + "\n"
            )
            tr = synth_bench.load_trace(tf)
            self.assertEqual([r["prompt_len"] for r in tr], [5, 40, 64])
            self.assertEqual(tr[2]["output_length"], 128)
            prompts, cycled = synth_bench.take(tr, 6, 20)
            self.assertTrue(all(len(p) == 20 for p in prompts))
            self.assertTrue(cycled)
            _os.remove(tf)
        finally:
            sys.path.pop(0)

    def test_vmin_fit_knows_synth_and_trace_modes(self):
        src = (CROSSVAL / "vmin_fit.py").read_text()
        for m in ('"synth"', '"trace"'):
            self.assertIn(m, src)
        self.assertIn("XVAL_TRACE", src)

    def test_baseline_age_ignores_local_timezone(self):
        out = self._table(
            "LOOP ctx=1024 bs=1 kv=fp16 ms_per_step=10.000 k=100\n",
            env={"TZ": "Asia/Tokyo"},
        )
        self.assertIn("(0.0d)", out)


class XvalConfigWorkloads(unittest.TestCase):
    """workloads(), grids(), and step_points() contracts."""

    def _load_xval_config(self):
        sys.path.insert(0, str(CROSSVAL))
        try:
            import xval_config
            importlib.reload(xval_config)
            return xval_config
        finally:
            sys.path.pop(0)

    def test_workloads_contains_deepseek_and_llama(self):
        xc = self._load_xval_config()
        wls = xc.workloads()
        self.assertIn("deepseek", wls)
        self.assertIn("llama", wls)

    def test_workloads_contains_json_only_crates(self):
        # JSON-only crates must survive the merge; xval.yaml must not drop them.
        xc = self._load_xval_config()
        wls = xc.workloads()
        for crate in ("smoke" if "smoke" in _load_workloads()["workloads"] else "qwen",
                      "qwen", "llama-h200"):
            if crate in _load_workloads()["workloads"]:
                self.assertIn(crate, wls, f"JSON-only crate {crate!r} missing from merged workloads()")

    def test_grids_contains_deepseek_and_llama(self):
        xc = self._load_xval_config()
        gs = xc.grids()
        self.assertIn("deepseek", gs)
        self.assertIn("llama", gs)

    def test_grids_contains_json_only_grids(self):
        # JSON-only grids (smoke, prof) must survive the merge.
        xc = self._load_xval_config()
        gs = xc.grids()
        json_grids = _load_workloads()["grids"]
        for grid in ("smoke", "prof"):
            if grid in json_grids:
                self.assertIn(grid, gs, f"JSON-only grid {grid!r} missing from merged grids()")

    def test_deepseek_merged_record_has_vmin_fit_keys(self):
        # vmin_fit.py reads dtype, weights_gib, maxlen, kv_bytes from the workload.
        xc = self._load_xval_config()
        ds = xc.workloads()["deepseek"]
        for key in ("dtype", "weights_gib", "maxlen", "kv_bytes"):
            self.assertIn(key, ds, f"deepseek merged record missing {key!r} (needed by vmin_fit)")

    def test_workloads_matches_workloads_json(self):
        # xval.yaml workloads superset-matches workloads.json for deepseek and llama.
        xc = self._load_xval_config()
        wls = xc.workloads()
        json_wls = _load_workloads()["workloads"]
        for key in ("deepseek", "llama"):
            self.assertIn(key, wls, f"workloads() missing {key}")
            self.assertIn(key, json_wls, f"workloads.json missing {key}")

    def test_grids_matches_workloads_json(self):
        xc = self._load_xval_config()
        gs = xc.grids()
        json_gs = _load_workloads()["grids"]
        for key in ("deepseek", "llama"):
            self.assertIn(key, gs)
            self.assertIn(key, json_gs)
            # Cell lists must be identical.
            self.assertEqual(
                [list(c) for c in gs[key]],
                [list(c) for c in json_gs[key]],
                f"grids()[{key!r}] diverges from workloads.json",
            )

    def test_llama_grid_has_15_cells(self):
        xc = self._load_xval_config()
        gs = xc.grids()
        self.assertEqual(len(gs["llama"]), 15)

    def test_llama_tp_defaults_to_1(self):
        # llama has no tp key in xval.yaml; vmin_fit uses wl.get("tp", 1).
        xc = self._load_xval_config()
        wls = xc.workloads()
        self.assertEqual(wls["llama"].get("tp", 1), 1)

    def test_llama_dtype_is_float16(self):
        xc = self._load_xval_config()
        wls = xc.workloads()
        self.assertEqual(wls["llama"]["dtype"], "float16")

    def test_deepseek_tp_is_8(self):
        xc = self._load_xval_config()
        wls = xc.workloads()
        self.assertEqual(wls["deepseek"]["tp"], 8)

    def test_deepseek_grid_has_12_cells(self):
        xc = self._load_xval_config()
        gs = xc.grids()
        self.assertEqual(len(gs["deepseek"]), 12)

    def test_qwen3_workload_resolves_dense(self):
        # qwen3 merges from workloads.json + xval.yaml; no expert_parallel.
        xc = self._load_xval_config()
        wls = xc.workloads()
        self.assertIn("qwen3", wls)
        qw = wls["qwen3"]
        self.assertEqual(qw.get("tp"), 4)
        # Hybrid stack: KV lives only in the 16 full-attention layers:
        # 16 layers x 2 (K+V) x 4 kv heads x 256 head_dim x 2 bytes (bf16).
        self.assertEqual(qw.get("kv_bytes"), 16 * 2 * 4 * 256 * 2)
        self.assertEqual(qw.get("dtype"), "bfloat16")
        self.assertNotIn("expert_parallel", qw)
        self.assertFalse(qw.get("verified"))

    def test_qwen3_grid_resolves_16_cells(self):
        xc = self._load_xval_config()
        gs = xc.grids()
        self.assertIn("qwen3", gs)
        self.assertEqual(len(gs["qwen3"]), 16)

    def test_vmin_fit_cfg_shape(self):
        # Assembled CFG must expose step_points, workloads, grids.
        xc = self._load_xval_config()
        cfg = {
            "workloads": xc.workloads(),
            "grids": xc.grids(),
            "step_points": xc.step_points(),
        }
        self.assertIn("step_points", cfg)
        self.assertIsInstance(cfg["step_points"], list)
        self.assertTrue(cfg["step_points"])
        self.assertIn("workloads", cfg)
        self.assertIn("grids", cfg)

    def test_step_points_matches_workloads_json(self):
        xc = self._load_xval_config()
        self.assertEqual(xc.step_points(), _load_workloads()["step_points"])

    def test_workloads_fallback_without_yaml(self):
        # workloads() falls back to workloads.json when xval.yaml is absent.
        import importlib
        import shutil
        td = Path(tempfile.mkdtemp())
        try:
            shutil.copy(CROSSVAL / "xval_config.py", td / "xval_config.py")
            shutil.copy(CROSSVAL / "workloads.json", td / "workloads.json")
            sys.path.insert(0, str(td))
            import xval_config as _fresh
            importlib.reload(_fresh)
            wls = _fresh.workloads()
            gs = _fresh.grids()
            sp = _fresh.step_points()
        finally:
            sys.path.pop(0)
            shutil.rmtree(td)
        self.assertIn("deepseek", wls)
        self.assertIn("llama", wls)
        self.assertIn("deepseek", gs)
        self.assertIn("llama", gs)
        self.assertEqual(sp, [60, 230, 400])


class XvalConfig(unittest.TestCase):
    """xval.yaml and xval_config.py contracts."""

    def _import_xval_config(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "xval_config", CROSSVAL / "xval_config.py"
        )
        mod = importlib.util.load_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_xval_yaml_parses(self):
        import yaml
        text = (CROSSVAL / "xval.yaml").read_text()
        data = yaml.safe_load(text)
        self.assertIsInstance(data, dict)
        self.assertTrue(data, "xval.yaml parsed to empty")

    def test_loader_resolves_collective_block(self):
        import yaml
        # Load via the module directly (no vllm needed; xval_config is stdlib+yaml).
        os.environ.pop("XVAL_LINK_PROFILE", None)
        sys.path.insert(0, str(CROSSVAL))
        try:
            import xval_config
            importlib.reload(xval_config)
            coll = xval_config.collective()
        finally:
            sys.path.pop(0)
        expected_keys = {
            "NCCL_ALGO", "NCCL_PROTO", "NCCL_P2P_LEVEL",
            "NCCL_IB_DISABLE", "NCCL_SOCKET_IFNAME",
            "LLMSRV_NO_NVLS", "LLMSRV_TWOSHOT_MIN_BYTES",
        }
        self.assertEqual(set(coll.keys()), expected_keys)
        # Values from xval.yaml must match the documented defaults.
        self.assertEqual(coll["NCCL_ALGO"], "allreduce:tree;allgather:ring")
        self.assertEqual(coll["NCCL_PROTO"], "Simple")
        self.assertEqual(coll["NCCL_P2P_LEVEL"], "SYS")

    def test_collective_nvlink_profile_exports_only_ib(self):
        # nvlink profile: pins come back empty (= not exported) so NCCL keeps
        # NVLS/LL128 and the engine keeps peer-allreduce; only IB stays off.
        sys.path.insert(0, str(CROSSVAL))
        try:
            import xval_config
            importlib.reload(xval_config)
            coll = xval_config.collective(profile="nvlink")
        finally:
            sys.path.pop(0)
        self.assertEqual(coll["NCCL_IB_DISABLE"], "1")
        for key in (
            "NCCL_ALGO", "NCCL_PROTO", "NCCL_P2P_LEVEL",
            "NCCL_SOCKET_IFNAME", "LLMSRV_NO_NVLS", "LLMSRV_TWOSHOT_MIN_BYTES",
        ):
            self.assertEqual(coll[key], "", key)

    def test_collective_profile_resolves_from_env(self):
        sys.path.insert(0, str(CROSSVAL))
        os.environ["XVAL_LINK_PROFILE"] = "nvlink"
        try:
            import xval_config
            importlib.reload(xval_config)
            coll = xval_config.collective()
        finally:
            del os.environ["XVAL_LINK_PROFILE"]
            sys.path.pop(0)
        self.assertEqual(coll["NCCL_ALGO"], "")
        self.assertEqual(coll["NCCL_IB_DISABLE"], "1")

    def test_collective_rejects_unknown_profile(self):
        sys.path.insert(0, str(CROSSVAL))
        try:
            import xval_config
            importlib.reload(xval_config)
            with self.assertRaises(ValueError):
                xval_config.collective(profile="tcp")
        finally:
            sys.path.pop(0)

    def test_link_detection_has_no_sigpipe_grep(self):
        # Piping nvidia-smi into grep -q SIGPIPEs it under pipefail and every
        # box misdetects as pcie; detection must capture first, then match.
        for sh in _scripts(".sh"):
            text = sh.read_text()
            if "nvidia-smi topo" not in text:
                continue
            self.assertNotRegex(
                text, r"nvidia-smi topo[^\n|]*\|\s*grep",
                f"{sh.name}: pipes nvidia-smi topo into grep (SIGPIPE misdetect)",
            )
            self.assertIn(
                '_topo="$(nvidia-smi topo', text,
                f"{sh.name}: must capture topo output before matching",
            )

    def test_loader_resolves_max_seq_headroom(self):
        sys.path.insert(0, str(CROSSVAL))
        try:
            import xval_config
            importlib.reload(xval_config)
            p = xval_config.run_params()
        finally:
            sys.path.pop(0)
        self.assertEqual(p["max_seq_headroom"], 28)
        self.assertEqual(p["timing_steps"], 100)
        # max_seq formula: ctx + timing_steps + headroom = ctx + 128 at defaults.
        ctx = 1024
        self.assertEqual(ctx + p["timing_steps"] + p["max_seq_headroom"], ctx + 128)

    def test_loader_falls_back_to_defaults_without_yaml(self):
        """xval_config must return valid defaults even with no xval.yaml present."""
        import importlib
        import shutil
        # Point _HERE at a temp dir with no xval.yaml.
        td = Path(tempfile.mkdtemp())
        os.environ.pop("XVAL_LINK_PROFILE", None)
        try:
            shutil.copy(CROSSVAL / "xval_config.py", td / "xval_config.py")
            sys.path.insert(0, str(td))
            import xval_config as _fresh
            importlib.reload(_fresh)
            coll = _fresh.collective()
            p = _fresh.run_params()
        finally:
            sys.path.pop(0)
            shutil.rmtree(td)
        self.assertEqual(coll["NCCL_ALGO"], "allreduce:tree;allgather:ring")
        self.assertEqual(p["timing_steps"], 100)
        self.assertEqual(p["max_seq_headroom"], 28)


class TraceWorkloads(unittest.TestCase):
    """swebench profile + arrival/session passthrough + serving-summary math."""

    def _sb(self):
        sys.path.insert(0, str(CROSSVAL))
        try:
            import synth_bench
            importlib.reload(synth_bench)
            return synth_bench
        finally:
            sys.path.pop(0)

    def test_swebench_profile_is_deterministic_and_grouped(self):
        sb = self._sb()
        a = sb.swebench_requests(24, seed=3, groups=4, prefix_frac=0.5)
        b = sb.swebench_requests(24, seed=3, groups=4, prefix_frac=0.5)
        self.assertEqual(a, b)
        plen = int(4096 * 0.5)
        # Same session shares the repo-context prefix; different sessions differ.
        self.assertEqual(a[0]["prompt_token_ids"][:plen], a[4]["prompt_token_ids"][:plen])
        self.assertNotEqual(a[0]["prompt_token_ids"][:plen], a[1]["prompt_token_ids"][:plen])
        for r in a:
            self.assertGreaterEqual(r["prompt_len"], 4096)
            self.assertLessEqual(r["prompt_len"], 24576)
            self.assertGreaterEqual(r["output_length"], 128)
            self.assertLessEqual(r["output_length"], 1024)
        arrivals = [r["arrival_ts"] for r in a]
        self.assertEqual(arrivals, sorted(arrivals))
        self.assertEqual({r["session_id"] for r in a}, {f"g{i}" for i in range(4)})

    def test_load_trace_carries_arrivals_and_sessions(self):
        sb = self._sb()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.jsonl"
            p.write_text(
                json.dumps({"input_length": 8, "output_length": 4,
                            "timestamp": 2000, "session_id": "s1"}) + "\n"
                + json.dumps({"prompt_token_ids": [5, 6, 7], "output_length": 2,
                              "arrival_ts": 3.5}) + "\n"
            )
            reqs = sb.load_trace(str(p))
        self.assertEqual(reqs[0]["arrival_ts"], 2.0)  # Mooncake ms -> seconds
        self.assertEqual(reqs[0]["session_id"], "s1")
        self.assertEqual(reqs[0]["output_length"], 4)
        self.assertEqual(reqs[1]["prompt_token_ids"], [5, 6, 7])
        self.assertEqual(reqs[1]["arrival_ts"], 3.5)
        self.assertEqual(reqs[1]["output_length"], 2)

    def test_trace_serve_helpers_are_gpu_free(self):
        # The serving replay must import without vLLM (CI has no GPU stack).
        sys.path.insert(0, str(CROSSVAL))
        try:
            import trace_serve
            importlib.reload(trace_serve)
        finally:
            sys.path.pop(0)
        offs = trace_serve.plan_arrivals(
            [{"arrival_ts": 10.0}, {"arrival_ts": 11.0}, {"arrival_ts": 14.0}], speed=2.0)
        self.assertEqual(offs, [0.0, 0.5, 2.0])
        self.assertEqual(trace_serve.plan_arrivals([{}, {}], speed=1.0), [0.0, 0.0])
        s = trace_serve.summarize(
            [{"ttft_ms": 10.0, "tpot_ms": 5.0, "out_tokens": 3},
             {"ttft_ms": 30.0, "tpot_ms": None, "out_tokens": 1},
             {"ttft_ms": 20.0, "tpot_ms": 7.0, "out_tokens": 3}], wall_s=2.0)
        self.assertEqual(s["n"], 3)
        self.assertEqual(s["ttft_ms"]["p50"], 20.0)
        self.assertEqual(s["ttft_ms"]["p99"], 30.0)
        self.assertEqual(s["tpot_ms"]["p50"], 7.0)  # single-token req skipped
        self.assertEqual(s["tpot_ms"]["p90"], 7.0)
        self.assertAlmostEqual(s["tok_s"], 3.5)


if __name__ == "__main__":
    unittest.main()
