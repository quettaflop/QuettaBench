"""Offline checks for the cross-validation path.

CI has no GPU, so this pins the wiring: scripts parse, no shell entry
points are left, the workload records (pinned in the config golden, which
holds the loader to them) stay consistent, the grids cover the cells the
engine bench times, and crossval only compares like with like.
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
from unittest import mock

from quettabench.config import loader
from quettabench.config.schema import ConfigError
from quettabench.config.validate import check_ep_legal

REPO = Path(__file__).resolve().parents[1]
CROSSVAL = REPO / "scripts" / "crossval"
QB = REPO / "quettabench"
VERIFIED_CRATE = "llama"


def _scripts(suffix):
    roots = (CROSSVAL, QB, REPO / "analysis")
    return sorted(p for root in roots for p in root.rglob(f"*{suffix}")
                  if "__pycache__" not in p.parts)


def _load_workloads():
    """The merged records, grids and step points the retired workloads.json
    held, as the config golden pins them for the loader."""
    fixtures = REPO / "tests" / "golden" / "fixtures"
    records = json.loads((fixtures / "config_workloads.json").read_text())
    glob = json.loads((fixtures / "config_global.json").read_text())["cleared"]
    return {"workloads": {n: r["cleared"]["xval_config"]["workload"] for n, r in records.items()},
            "grids": glob["grids"], "step_points": glob["step_points"]}


def _config(name, *sets, env=None):
    """The loader's Config for configs/runs/<name>.yaml under exactly `env`."""
    (cfg,) = loader.load(REPO / "configs" / "runs" / f"{name}.yaml",
                         ["allow_unverified=true", *sets], env=env or {})
    return cfg


class CrossvalScripts(unittest.TestCase):
    def test_python_scripts_parse(self):
        # Parse only: importing would pull in vllm, which CI does not have.
        scripts = _scripts(".py")
        self.assertTrue(scripts, "no python scripts under scripts/crossval")
        for script in scripts:
            ast.parse(script.read_text(), filename=str(script))

    def test_no_shell_entry_points_left(self):
        # qbench, provision and the justfile replaced every crossval shell script.
        self.assertEqual(_scripts(".sh"), [])

    def test_serve_wires_the_identity_gate(self):
        # Every serve run hashes what it served against the trace's own hash.
        self.assertIn("workload_hash", (QB / "modes" / "serve.py").read_text())
        self.assertIn("WORKLOADSUM", (QB / "metrics" / "results.py").read_text())

    def test_backend_detection_and_no_hard_nvidia_smi(self):
        # XVAL_BACKEND wins; with no accel tooling on PATH detection says cpu, and
        # link detection on a non-cuda backend never calls nvidia-smi.
        with mock.patch("shutil.which", return_value=None):
            self.assertEqual(_config("llama").hardware.backend, "cpu")
            self.assertEqual(_config("llama").hardware.link_profile, "nvlink")
        self.assertEqual(_config("llama", env={"XVAL_BACKEND": "tpu"}).hardware.backend, "tpu")

    def test_pp_wired_in_grid_and_device_count(self):
        # A grid run passes pipeline_parallel_size (from XVAL_PP or the workload's pp),
        # and the device count the orchestrator provisions is tp * pp.
        src = (QB / "config" / "env_map.py").read_text() + (QB / "engines" / "vllm.py").read_text()
        self.assertIn("pipeline_parallel_size", src)
        self.assertIn("XVAL_PP", src)
        # the grid run prints META one key per line
        self.assertIn('"pp": e.pp', (QB / "modes" / "_grid.py").read_text())
        self.assertIn('f"META {k}={v}"', (QB / "metrics" / "results.py").read_text())
        ds = _config("deepseek").engine
        self.assertEqual(ds.tp * ds.pp, 8)  # deepseek tp8, pp1
        ds = _config("deepseek", env={"XVAL_PP": "2"}).engine
        self.assertEqual(ds.tp * ds.pp, 16)  # tp8 * pp2

    def test_link_profile_detection(self):
        # One detection for both engines: caller override wins, non-cuda backends
        # take the no-pins nvlink profile.
        self.assertEqual(_config("llama", env={"XVAL_LINK_PROFILE": "pcie"}).hardware.link_profile,
                         "pcie")
        self.assertEqual(_config("llama", env={"XVAL_BACKEND": "tpu"}).hardware.link_profile,
                         "nvlink")

    def test_provision_defaults(self):
        # provision.py reads its defaults from the hardware yaml.
        self.assertEqual(_config("llama").hardware.provision["cache_root"],
                         "/workspace/.xval-cache")

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
                [sys.executable, str(REPO / "analysis" / "lss_export.py"), "--probes", str(probes),
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
                [sys.executable, str(REPO / "analysis" / "lss_export.py"), "--probes", str(probes),
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
                [sys.executable, str(REPO / "analysis" / "lss_first_token.py"), "--live", str(live),
                 "--sim", str(sim), "--meta", str(meta)],
                capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
            self.assertIn("overhead_ms=6.300", out.stdout)
            self.assertIn("after_mape=0.0%", out.stdout)
            self.assertIn("first_token_overhead_us: 6300", meta.read_text())

    def test_goodput_search_and_verdict(self):
        # The sweep must bracket the PASS/FAIL boundary within its probe cap;
        # an uncapped search would double forever on a flat SLA response.
        from quettabench.metrics.core import sla_verdict
        from quettabench.modes.serve import goodput_search
        gen = goodput_search(start=1.0)
        speeds, speed = [], next(gen)
        try:
            while True:
                speeds.append(speed)
                speed = gen.send(speed < 4.0)  # SLA passes strictly below 4.0
        except StopIteration:
            pass
        self.assertLessEqual(len(speeds), 6)
        passing = [x for x in speeds if x < 4.0]
        failing = [x for x in speeds if x >= 4.0]
        self.assertTrue(passing and failing, speeds)
        self.assertGreater(max(passing), 3.0, speeds)  # tightened to the boundary
        s = {"ttft_ms": {"p99": 900.0}, "tpot_ms": {"p99": 40.0}}
        self.assertEqual(sla_verdict(s, 500.0, 50.0), ["ttft_p99"])
        self.assertEqual(sla_verdict(s, None, None), [])
        self.assertIn("GOODPUTSUM", (QB / "metrics" / "results.py").read_text())

    def test_injected_arrivals(self):
        # Injected schedules must be deterministic, must match their spec, and
        # must change the workload hash (arrival_ts is a hash input), so a
        # trace-paced and an injected run can never be conflated.
        sys.path.insert(0, str(REPO))
        try:
            from quettabench.workloads import arrival, synth_bench
            importlib.reload(arrival)
            importlib.reload(synth_bench)
            self.assertIsNone(arrival.parse_arrival("trace", 8))
            p = arrival.parse_arrival("poisson:2.0", 8)
            self.assertEqual(p, arrival.parse_arrival("poisson:2.0", 8))
            self.assertEqual(sorted(p), p)
            b = arrival.parse_arrival("burst:2x4@5.0", 8)
            self.assertEqual(b, [0.0] * 4 + [5.0] * 4)
            with self.assertRaises(ValueError):
                arrival.parse_arrival("burst:1x4@5.0", 8)  # spec too small
            r = arrival.parse_arrival("ramp:0.5:4.0", 8)
            self.assertEqual(len(r), 8)
            reqs = synth_bench.synth_requests(4, 64, seed=1)
            h0 = synth_bench.workload_hash(reqs)
            for req, t in zip(reqs, b[4:]):  # the non-zero burst; t=0 equals the default
                req["arrival_ts"] = round(t, 6)
            self.assertNotEqual(h0, synth_bench.workload_hash(reqs))
        finally:
            sys.path.pop(0)

    def test_energy_integration(self):
        from quettabench.profiling.wrapped import integrate_power
        self.assertEqual(integrate_power([100.0, 100.0, 100.0], 1.0), 300.0)
        self.assertEqual(integrate_power([], 1.0), 0.0)
        self.assertIn("ENERGYSUM", (QB / "profiler.py").read_text())

    def test_cost_math(self):
        from quettabench.profiling.wrapped import cost_per_m_tokens
        # 2 GPUs at $3.60/hr for 30 min producing 1M tokens: $3.60, $3.60/M.
        usd, per_m = cost_per_m_tokens(3.6, 2, 1800.0, 1_000_000)
        self.assertAlmostEqual(usd, 3.6)
        self.assertAlmostEqual(per_m, 3.6)
        self.assertIn("COSTSUM", (QB / "profiler.py").read_text())

    def test_gsm8k_answer_parse(self):
        from quettabench.profiling.wrapped import gsm8k_answer as f
        self.assertEqual(f("Natalia sold 48 clips. #### 72"), "72")
        self.assertEqual(f("so the total is $1,234."), "1234")
        self.assertEqual(f("the answer is 3.5 apples"), "3.5")
        self.assertIsNone(f("no numbers here"))
        # ACCSUM prints from the profile's own accuracy section, so SPECSUM's
        # records can no longer clobber it the way trace_serve's shared `acc` could.
        self.assertIn("ACCSUM", (QB / "profiler.py").read_text())

    def test_serving_style_contract(self):
        # Item 1a: serving topology parses, disagg GPU pools are disjoint, ep must
        # divide the expert count, and disagg rows never pair with aggregated.
        sys.path.insert(0, str(REPO))
        try:
            from quettabench.config import validate
            importlib.reload(validate)
            self.assertEqual(validate.parse_serving_style("aggregated"), {"mode": "aggregated"})
            self.assertEqual(validate.parse_serving_style(None), {"mode": "aggregated"})
            self.assertEqual(validate.parse_serving_style("disagg:2p2d"),
                             {"mode": "disagg", "prefill": 2, "decode": 2})
            self.assertEqual(validate.parse_serving_style("ep8dp2"),
                             {"mode": "ep_dp", "ep": 8, "dp": 2})
            for bad in ("disagg:0p1d", "disagg:2p", "ep8", "wideEP", "disagg:xp1d",
                        "ep0dp2", "ep8dp0"):
                with self.assertRaises(ValueError):
                    validate.parse_serving_style(bad)
            pre, dec = validate.disagg_gpu_sets(
                validate.parse_serving_style("disagg:2p2d"), [0, 1, 2, 3])
            self.assertEqual((pre, dec), ([0, 1], [2, 3]))
            self.assertFalse(set(pre) & set(dec))
            with self.assertRaises(ValueError):
                validate.disagg_gpu_sets(
                    validate.parse_serving_style("disagg:2p2d"), [0, 1, 2])
            validate.check_ep_legal(8, 256)  # DeepSeek-V4-Flash: 256 experts
            with self.assertRaises(ValueError):
                validate.check_ep_legal(7, 256)
            self.assertFalse(validate.serving_style_compatible("disagg:1p1d", "aggregated"))
            self.assertTrue(validate.serving_style_compatible("aggregated", "aggregated"))
            # disagg KV config: prefill producer and decode consumer share the port
            # and pool size, take distinct ranks; a wrong role raises.
            pre = validate.disagg_kv_config("kv_producer", 0, "P2pConnector")
            dec = validate.disagg_kv_config("kv_consumer", 1, "P2pConnector")
            self.assertEqual(pre["kv_role"], "kv_producer")
            self.assertEqual(dec["kv_rank"], 1)
            self.assertEqual(pre["kv_port"], dec["kv_port"])
            self.assertEqual(pre["kv_parallel_size"], dec["kv_parallel_size"])
            with self.assertRaises(ValueError):
                validate.disagg_kv_config("decode", 1, "P2pConnector")
        finally:
            sys.path.pop(0)

    def test_glm53_workload_contract(self):
        # GLM-5.3 is MLA + DSA, so kv_bytes must be the latent formula; the dense
        # per-head formula in the grid run would overestimate KV several-fold and skew
        # the capacity gate. The checkpoint is pre-quantized fp8, recorded as quant
        # so the table flags any pairing against a bf16 baseline. No bench section:
        # QuettaServe has no GLM crate, this workload is vLLM-side only for now.
        wl = _load_workloads()["workloads"]["glm53"]
        layers, kv_lora_rank, rope, bf16 = 78, 512, 64, 2  # zai-org/GLM-5.3 config.json
        self.assertEqual(wl["kv_bytes"], layers * (kv_lora_rank + rope) * bf16)
        self.assertEqual(wl["quant"], "fp8")
        self.assertEqual(wl["tp"], 8)
        self.assertTrue(wl["expert_parallel"])
        self.assertFalse(wl["verified"], "bring-up gate: runs need ALLOW_UNVERIFIED=1")
        self.assertNotIn("bench", wl)
        self.assertNotIn("comm_bound_bs", wl)  # measured per model, never inherited
        check_ep_legal(wl["tp"], 256)  # 256 routed experts shard on tp8
        cfg = _config("glm53")
        self.assertEqual(cfg.mode.grid, wl["grid"])
        self.assertEqual(cfg.engine.tp * cfg.engine.pp, 8)

    def test_glm53flash_workload_contract(self):
        # GLM-5.3-Flash is a hybrid: 34 KDA linear-attention layers keep fixed-size
        # recurrent state, only the 11 sparse-MLA layers hold per-token KV (rope-free
        # latent 512). Its config even has head_dim 0, so the grid run's dense formula
        # would be garbage; kv_bytes is mandatory. A different model from glm53,
        # engine branch kev/glm53flash, and never comparable across the pair.
        wl = _load_workloads()["workloads"]["glm53flash"]
        mla_layers, kv_lora_rank, rope, bf16 = 11, 512, 0, 2  # zai-org/GLM-5.3-Flash text_config
        self.assertEqual(wl["kv_bytes"], mla_layers * (kv_lora_rank + rope) * bf16)
        self.assertEqual(wl["quant"], "fp8")
        self.assertEqual(wl["tp"], 4)
        self.assertTrue(wl["expert_parallel"])
        self.assertFalse(wl["verified"], "bring-up gate: runs need ALLOW_UNVERIFIED=1")
        self.assertNotIn("bench", wl)
        self.assertNotIn("comm_bound_bs", wl)
        check_ep_legal(wl["tp"], 288)  # 288 routed experts shard on tp4
        self.assertEqual(_config("glm53flash").mode.grid, wl["grid"])

    def test_agreement_inputs_present(self):
        for name in ("prompts.txt", "texts.txt"):
            lines = [line for line in (REPO / "data" / name).read_text().splitlines() if line.strip()]
            self.assertTrue(lines, f"{name} is empty")

    def test_grid_run_supports_moe_bringup(self):
        # expert parallel on vLLM, plus an opt-in to run before certified.
        src = (QB / "cli.py").read_text() + (QB / "engines" / "vllm.py").read_text()
        self.assertIn("enable_expert_parallel", src)
        self.assertIn("allow-unverified", src)

    def _bench(self, workload, stub_body, sets=(), env=None):
        """qbench run of the workload on QuettaServe with a stub bench binary."""
        with tempfile.TemporaryDirectory() as td:
            stub = Path(td) / "stub"
            stub.write_text("#!/usr/bin/env bash\n" + stub_body + "\n")
            stub.chmod(0o755)
            argv = [sys.executable, "-m", "quettabench.cli", "run",
                    str(REPO / "configs" / "runs" / f"{workload}.yaml"), "--allow-unverified",
                    "--results-dir", td, "--set", "engine.name=quettaserve",
                    "--set", "engine.model=/dev/null", "--set", f"engine.extra.bench_bin={stub}",
                    "--set", f"engine.extra.qs_dir={td}"]
            for item in sets:
                argv += ["--set", item]
            out = subprocess.run(argv, capture_output=True, text=True, cwd=REPO,
                                 env={**os.environ, "XVAL_BACKEND": "cpu", **(env or {})})
            log = Path(td) / workload / "bench.log"
            return out, log.read_text() if log.exists() else ""

    BENCH_LINE = (
        'echo "[batch_bench] world=${DS_WORLD} bs=${DS_BATCH}'
        ' max_seq=${DS_MAX_SEQ} layers=${DS_LAYERS:-all}'
        ' prompt=${DS_BENCH_PROMPT} steps=92: median 1.000 ms/step,'
        ' mean 1.0, best 1.0 -> 1.0 tok/s (median)"'
    )

    def test_bench_runs_the_deepseek_grid_at_the_workload_tp(self):
        # world from the workload tp, every cell runs, DS_LAYERS blocked.
        out, log = self._bench("deepseek", self.BENCH_LINE, ["engine.extra.qs_cfg=/dev/null"],
                               env={"DS_LAYERS": "6"})
        self.assertEqual(out.returncode, 0, out.stderr)
        cfg = _load_workloads()
        tp = cfg["workloads"]["deepseek"]["tp"]
        cells = re.findall(r"^\[batch_bench\] world=(\d+) .*layers=(\w+)", log, re.M)
        self.assertEqual(len(cells), len(cfg["grids"]["deepseek"]))
        self.assertEqual(set(cells), {(str(tp), "all")})

    def test_bench_fails_when_a_cell_produces_no_line(self):
        out, _ = self._bench("deepseek", "exit 0", ["engine.extra.qs_cfg=/dev/null"])
        self.assertEqual(out.returncode, 1)
        self.assertIn("FAIL", out.stderr)

    QWEN_LINE = (
        'echo "LOOP ctx=${QW_BENCH_PROMPT} bs=${QW_BATCH} kv=bf16'
        ' ms_per_step=1.000 gdn=${QS_KERNELS} mode=${QW_MODE} k=100"'
    )

    def test_bench_qwen_defaults_exact_to_eager(self):
        # exact has no graph wiring, so with cubins present the driver must
        # pick the eager step and stamp kern/mode into every cell.
        out, log = self._bench("qwen3", self.QWEN_LINE, ["engine.extra.gdn_dir=/dev/null"])
        self.assertEqual(out.returncode, 0, out.stderr)
        cfg = _load_workloads()
        modes = re.findall(r"^LOOP .*mode=(\w+)", log, re.M)
        self.assertEqual(len(modes), len(cfg["grids"]["qwen3"]))
        self.assertEqual(set(modes), {"eager"})
        self.assertIn("kern=exact", log)

    def test_bench_requires_a_bench_section(self):
        out, _ = self._bench("llama", "exit 0")
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("no quettaserve: block", out.stderr)


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
                    f"{workload['maxlen']}; the grid run would SKIP it",
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
        # The QuettaServe bench loops the run's grid cells.
        grid = {tuple(c) for c in _load_workloads()["grids"]["deepseek"]}
        expected = {(ctx, bs) for ctx in (1024, 8192, 16384) for bs in (1, 4, 16, 64)}
        self.assertEqual(grid, expected, f"grid drift: {sorted(grid ^ expected)}")


class TableParity(unittest.TestCase):
    """crossval prints ratios only when both sides timed the same quantity:
    the engine's pipelined decode_loop against the baseline's slope. A
    criterion per-step log gets its cells printed with ratios withheld, and
    rows whose KV formats differ carry a KV flag."""

    def _table(self, bench_text, kv="float16", env=None, expect_rc=0,
               serving_style="aggregated", quant="bf16"):
        import sys
        import time

        base = {
            "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu": "test", "vllm": "0", "mode": "FULL", "model": "m",
            "dtype": "float16", "grid": "g", "method": "slope", "kv_dtype": kv,
            "nccl_algo": "allreduce:tree;allgather:ring", "nccl_proto": "Simple",
            "serving_style": serving_style, "quant": quant,
            "points": [{"ctx": 1024, "bs": 1, "tp": 1, "ms_per_step": 5.0,
                        "tok_s": 200.0, "r2": 1.0, "kv": kv}],
        }
        with tempfile.TemporaryDirectory() as td:
            bench = Path(td) / "bench.log"
            cache = Path(td) / "base.json"
            bench.write_text(bench_text)
            cache.write_text(json.dumps(base))
            out = subprocess.run(
                [sys.executable, "-m", "quettabench.crossval", str(bench), str(cache),
                 "--json", str(Path(td) / "xval.json")],
                capture_output=True, text=True, env={**os.environ, **(env or {})}, cwd=REPO,
            )
        self.assertEqual(out.returncode, expect_rc, out.stderr)
        return out.stdout

    def test_loop_bench_gets_a_ratio(self):
        out = self._table("LOOP ctx=1024 bs=1 kv=fp16 ms_per_step=10.000 k=100\n")
        self.assertIn("0.50x", out)
        self.assertNotIn("withheld", out)

    def test_quant_mismatch_flags_and_withholds(self):
        # An nvfp4 engine row against a bf16 baseline is flagged QUANT and withheld;
        # The grid run passes quantization through and records META quant.
        out = self._table(
            "LOOP ctx=1024 bs=1 kv=fp16 ms_per_step=10.000 k=100\n",
            quant="bf16", env={"XVAL_QUANT": "nvfp4"})
        self.assertIn("QUANT MISMATCH", out)
        self.assertIn("QUANT bf16/nvfp4", out)
        self.assertNotIn("0.50x", out)
        self.assertIn("quantization", (QB / "engines" / "vllm.py").read_text())  # kwargs built there
        self.assertIn('"quant": e.quant or e.dtype', (QB / "modes" / "_grid.py").read_text())

    def test_serving_style_mismatch_withholds(self):
        # A disagg baseline against an aggregated engine row is never a ratio.
        out = self._table(
            "LOOP ctx=1024 bs=1 kv=fp16 ms_per_step=10.000 k=100\n",
            serving_style="disagg:1p1d", env={"XVAL_SERVING_STYLE": "aggregated"})
        self.assertIn("SERVING MISMATCH", out)
        self.assertNotIn("0.50x", out)
        # matching topology still earns its ratio
        out2 = self._table(
            "LOOP ctx=1024 bs=1 kv=fp16 ms_per_step=10.000 k=100\n",
            serving_style="aggregated", env={"XVAL_SERVING_STYLE": "aggregated"})
        self.assertIn("0.50x", out2)

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

    def test_bench_adapter_wires_the_gdn_family(self):
        # The gdn family owns the kernel env and the eager default for exact.
        text = (QB / "engines" / "quettaserve.py").read_text()
        self.assertIn('"exact" if gdn_dir else "rust"', text)
        self.assertIn('"eager" if kernels == "exact" else "graph"', text)
        self.assertIn("mode={mode}", text)

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
                [sys.executable, "-m", "quettabench.crossval", str(bench), str(cache),
                 "--json", str(Path(td) / "xval.json")],
                capture_output=True, text=True, env=os.environ, cwd=REPO,
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
                [sys.executable, "-m", "quettabench.crossval", str(bench), str(cache),
                 "--json", str(Path(td) / "xval.json")],
                capture_output=True, text=True, env=os.environ, cwd=REPO,
            )
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertNotIn("COMM", out.stdout)
        self.assertIn("0.50x", out.stdout)

    def test_grid_prompt_modes(self):
        # clone must reproduce the engine's synthetic prompt so both sides
        # route alike; all three regimes must be selectable.
        self.assertIn("(i * 137 + 11) % 100_000", (QB / "workloads" / "grid.py").read_text())
        src = (QB / "workloads" / "grid.py").read_text() + (QB / "config" / "schema.py").read_text()
        self.assertIn("prompt_mode", src)
        for mode in ('"clone"', '"corpus"', '"distinct"'):
            self.assertIn(mode, src)
        self.assertIn('returns: str = "none"', src)  # text: the routing-parity token dump

    def test_cache_records_prompt_mode(self):
        # A baseline without its routing regime is not comparable later.
        self.assertIn("prompt_mode", (QB / "crossval" / "reader.py").read_text())
        self.assertIn("prompt_mode", (QB / "crossval" / "compare.py").read_text())

    def test_corpus_mode_requires_an_explicit_corpus(self):
        # No experimental text ships in the repo; corpus mode takes XVAL_CORPUS.
        self.assertFalse((CROSSVAL / "routing_corpus.txt").exists())
        self.assertIn("XVAL_CORPUS", (QB / "modes" / "_grid.py").read_text())

    def test_hash_canonicalization(self):
        # The gate hashes the completed set, not the completion order: a true
        # random shuffle of the done-list must not move the hash, while touching
        # one prompt token or one arrival must. Without order-invariance the
        # gate would void every run whose scheduler finished out of order.
        import random
        sys.path.insert(0, str(REPO))
        try:
            from analysis.synthetic import agentic
            from quettabench.workloads import synth_bench
            importlib.reload(synth_bench)
            reqs = agentic.agentic_requests("deep_research", 24, seed=5)
            h = synth_bench.workload_hash(reqs)
            shuffled = list(reqs)
            random.Random(7).shuffle(shuffled)
            self.assertNotEqual([r["order"] for r in shuffled], [r["order"] for r in reqs])
            self.assertEqual(h, synth_bench.workload_hash(shuffled))
            prompt_mut = [dict(r) for r in reqs]
            ids = list(prompt_mut[3]["prompt_token_ids"])
            ids[0] += 1
            prompt_mut[3]["prompt_token_ids"] = ids
            self.assertNotEqual(h, synth_bench.workload_hash(prompt_mut))
            arrival_mut = [dict(r) for r in reqs]
            arrival_mut[3]["arrival_ts"] = arrival_mut[3]["arrival_ts"] + 0.5
            self.assertNotEqual(h, synth_bench.workload_hash(arrival_mut))
        finally:
            sys.path.pop(0)

    def test_workload_identity_hash(self):
        # The gate: same requests -> same hash regardless of order; a dropped or
        # altered request changes it. This is what proves systems saw equal work.
        sys.path.insert(0, str(REPO))
        try:
            from quettabench.workloads import synth_bench
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
                [sys.executable, str(QB / "workloads" / "synth_bench.py"), "hash", "/dev/stdin"],
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
            fd, p = tempfile.mkstemp(suffix=".jsonl")
            os.close(fd)
            subprocess.run([sys.executable, str(REPO / "analysis" / "synthetic" / "agentic.py"), "gen",
                            "--profile", "terminal_bench", "--n", "8", "--seed", str(seed),
                            "--out", p], check=True, capture_output=True)
            return p
        a, b, c = gen(1), gen(1), gen(2)
        self.assertEqual(Path(a).read_bytes(), Path(b).read_bytes(),
                         "same seed must produce a byte-identical trace")
        rows = [json.loads(line) for line in open(a)]
        self.assertTrue(all(r["source"] == "SYNTHETIC" for r in rows))
        self.assertTrue(all(r["seed"] == 1 for r in rows))
        sys.path.insert(0, str(REPO))
        try:
            from quettabench.workloads import synth_bench
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
        sys.path.insert(0, str(REPO))
        try:
            from analysis.synthetic import agentic
            specs = agentic.AGENTIC
            self.assertGreater(len({s["burst"] for s in specs.values()}), 1)
            self.assertGreater(len({s["gap"] for s in specs.values()}), 1)
            def span(rs):
                return max(r["arrival_ts"] for r in rs) - min(r["arrival_ts"] for r in rs)
            dr = agentic.agentic_requests("deep_research", 40, seed=1)
            tb = agentic.agentic_requests("terminal_bench", 40, seed=1)
            self.assertGreater(span(dr), span(tb),
                               "deep_research must arrive over a longer span than terminal_bench")
        finally:
            sys.path.pop(0)

    def test_serving_emits_verdicts(self):
        schema = (QB / "config" / "schema.py").read_text()
        for field in ("sla_ttft_ms", "sla_tpot_ms"):
            self.assertIn(field, schema)
        lines = (QB / "metrics" / "results.py").read_text()
        for tag in ("SLASUM", "BURSTSUM", "SPECSUM"):
            self.assertIn(tag, lines)
        self.assertIn("FIDELITY", (QB / "crossval" / "agreement.py").read_text())

    def test_burst_recovery_and_acceptance(self):
        from quettabench.metrics.core import acceptance, burst_recovery
        self.assertEqual(acceptance([1, 2, 3, 4], [1, 2, 9, 4]), (2, 0.5))
        self.assertEqual(acceptance([], [])[0], 0)
        recs = ([{"arrival_s": t, "ttft_ms": 100.0} for t in range(0, 5)]
                + [{"arrival_s": t, "ttft_ms": 9000.0} for t in range(5, 10)]
                + [{"arrival_s": t, "ttft_ms": 100.0} for t in range(15, 20)])
        rec_s, peak = burst_recovery(recs, sla_ttft_ms=1000.0, window_s=5.0)
        self.assertEqual(peak, 9000.0)
        self.assertIsNotNone(rec_s)
        self.assertGreater(rec_s, 0)
        never = [{"arrival_s": t, "ttft_ms": 9000.0} for t in range(0, 10)]
        self.assertIsNone(burst_recovery(never, 1000.0, 5.0)[0])

    def test_r2_tokenize_and_output_ids_survive_load(self):
        # The real-text path: agent-bench turns -> requests carrying output_token_ids
        # (reasoning+content), and load_trace must preserve them or SPECSUM can never
        # fire. encode is stubbed so the test needs no tokenizer or GPU.
        sys.path.insert(0, str(REPO))
        try:
            from quettabench.workloads import synth_bench, tokenize as trace_tokenize
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
        sys.path.insert(0, str(REPO))
        try:
            from analysis.synthetic import agentic
            from quettabench.workloads import synth_bench
            importlib.reload(synth_bench)
            for name, spec in agentic.AGENTIC.items():
                a = agentic.agentic_requests(name, 40, seed=3)
                b = agentic.agentic_requests(name, 40, seed=3)
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
        sys.path.insert(0, str(REPO))
        try:
            from quettabench.workloads import synth_bench
            importlib.reload(synth_bench)
            reqs = synth_bench.synth_requests(4, 32, seed=3)
            ids = [tuple(r["prompt_token_ids"]) for r in reqs]
            self.assertEqual(len(set(ids)), 4, "synth requests must be distinct")
            self.assertTrue(all(r["prompt_len"] == 32 for r in reqs))
            import json as _j
            import os as _os
            import tempfile
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

    def test_grid_knows_synth_and_trace_modes(self):
        src = (QB / "workloads" / "grid.py").read_text() + (QB / "config" / "env_map.py").read_text()
        for m in ('"synth"', '"trace"'):
            self.assertIn(m, src)
        self.assertIn("XVAL_TRACE", src)

    def test_baseline_age_ignores_local_timezone(self):
        out = self._table(
            "LOOP ctx=1024 bs=1 kv=fp16 ms_per_step=10.000 k=100\n",
            env={"TZ": "Asia/Tokyo"},
        )
        self.assertIn("(0.0d)", out)


class ConfigTree(unittest.TestCase):
    """configs/ through the loader: grids, step points and the collective env."""

    def test_grid_sizes_and_step_points(self):
        self.assertEqual(len(_config("llama").grid.cells), 15)
        self.assertEqual(len(_config("deepseek").grid.cells), 12)
        self.assertEqual(len(_config("qwen3").grid.cells), 16)
        self.assertEqual(_config("llama").grid.step_points, (60, 230, 400))

    def test_decode_batch_run_has_the_retired_probes_cells(self):
        # decode_batch_probe.py's defaults: --ctx 20000 --batches 1 2 4 8 11 16 24.
        grid = _config("decode_batch_llama").grid
        self.assertEqual(grid.cells, tuple((20000, bs) for bs in (1, 2, 4, 8, 11, 16, 24)))
        self.assertEqual(grid.cells_over_maxlen, ())

    def test_qwen3_resolves_dense_hybrid_kv(self):
        e = _config("qwen3").engine
        # Hybrid stack: KV lives only in the 16 full-attention layers:
        # 16 layers x 2 (K+V) x 4 kv heads x 256 head_dim x 2 bytes (bf16).
        self.assertEqual((e.tp, e.kv_bytes, e.dtype, e.ep, e.verified),
                         (4, 16 * 2 * 4 * 256 * 2, "bfloat16", False, False))

    def test_pcie_pins_the_collective(self):
        coll = _config("llama", "hardware.link_profile=pcie").hardware.collective
        self.assertEqual(set(coll), {"NCCL_ALGO", "NCCL_PROTO", "NCCL_P2P_LEVEL",
                                     "NCCL_IB_DISABLE", "NCCL_SOCKET_IFNAME",
                                     "LLMSRV_NO_NVLS", "LLMSRV_TWOSHOT_MIN_BYTES"})
        self.assertEqual((coll["NCCL_ALGO"], coll["NCCL_PROTO"], coll["NCCL_P2P_LEVEL"]),
                         ("allreduce:tree;allgather:ring", "Simple", "SYS"))

    def test_nvlink_exports_only_ib(self):
        # nvlink: pins come back empty (= not exported) so NCCL keeps NVLS/LL128
        # and the engine keeps peer-allreduce; only IB stays off.
        coll = _config("llama", "hardware.link_profile=nvlink").hardware.collective
        self.assertEqual({k: v for k, v in coll.items() if v}, {"NCCL_IB_DISABLE": "1"})

    def test_unknown_link_profile_is_refused(self):
        with self.assertRaisesRegex(ConfigError, "unknown link profile 'tcp'"):
            _config("llama", "hardware.link_profile=tcp")

    def test_max_seq_headroom(self):
        h = _config("llama").hardware
        # max_seq formula: ctx + timing_steps + headroom = ctx + 128 at defaults.
        self.assertEqual((h.timing_steps, h.max_seq_headroom), (100, 28))


class TraceWorkloads(unittest.TestCase):
    """swebench profile + arrival/session passthrough + serving-summary math."""

    def _sb(self):
        sys.path.insert(0, str(REPO))
        try:
            from quettabench.workloads import synth_bench
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

    def test_serve_helpers_are_gpu_free(self):
        # The serving procedure must import without vLLM (CI has no GPU stack).
        from quettabench.metrics.core import summarize_serve
        from quettabench.modes import serve  # noqa: F401
        from quettabench.workloads.arrival import rebase_arrivals
        self.assertEqual(rebase_arrivals([10.0, 11.0, 14.0], speed=2.0), [0.0, 0.5, 2.0])
        self.assertEqual(rebase_arrivals([None, None], speed=1.0), [0.0, 0.0])
        s = summarize_serve(
            [{"ttft_ms": 10.0, "tpot_ms": 5.0, "out_tokens": 3},
             {"ttft_ms": 30.0, "tpot_ms": None, "out_tokens": 1},
             {"ttft_ms": 20.0, "tpot_ms": 7.0, "out_tokens": 3}], wall_s=2.0, method="nearest")
        self.assertEqual(s["n"], 3)
        self.assertEqual(s["ttft_ms"]["p50"], 20.0)
        self.assertEqual(s["ttft_ms"]["p99"], 30.0)
        self.assertEqual(s["tpot_ms"]["p50"], 7.0)  # single-token req skipped
        self.assertEqual(s["tpot_ms"]["p90"], 7.0)
        self.assertAlmostEqual(s["tok_s"], 3.5)


if __name__ == "__main__":
    unittest.main()
