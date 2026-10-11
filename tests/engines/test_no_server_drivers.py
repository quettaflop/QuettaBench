"""Which driver a run gets, and the QuettaServe bench run end to end.

A grid run on vLLM launches nothing and times in-process; a serve run talks
to the launched server's /v1/completions. A QuettaServe grid run carries
every cell's spec and serve.run sends them through BenchDriver, writing one
row per cell with its measured ms per step; `qbench run` does the same with
the SKIP rules applied, and --dry-run prints each cell's shell line.
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from quettabench import cli
from quettabench.config import loader
from quettabench.engines import launcher, quettaserve, vllm as vllm_adapter
from quettabench.engines.base import LaunchSpec
from quettabench.engines.drivers import BenchDriver, HttpDriver, InProcessDriver
from quettabench.metrics.results import RunWriter
from quettabench.modes import serve
from tests.golden._harness import REPO

RUNS = REPO / "configs" / "runs"
FAKE = Path(__file__).with_name("fake_bench.py")
PINNED = ["allow_unverified=true", "hardware.backend=cpu", "hardware.link_profile=nvlink"]


def config(name, *sets):
    (cfg,) = loader.load(RUNS / f"{name}.yaml", [*PINNED, *sets], env={})
    return cfg


class VllmDriver(unittest.TestCase):
    def test_grid_run_is_in_process(self):
        cfg = config("llama", "engine.model=/w")
        self.assertFalse(vllm_adapter.ADAPTER.needs_server(cfg))
        spec = vllm_adapter.ADAPTER.translate(cfg)
        driver = vllm_adapter.ADAPTER.driver(spec, launcher.NoServer(Path("/r")))
        self.assertIsInstance(driver, InProcessDriver)

    def test_serve_run_talks_to_the_launched_server(self):
        cfg = config("llama", "engine.model=/w", "mode.grid=null", "mode.timeout=7")
        self.assertTrue(vllm_adapter.ADAPTER.needs_server(cfg))
        spec = vllm_adapter.ADAPTER.translate(cfg)
        handle = launcher.NoServer(Path("/r"), endpoint="http://127.0.0.1:8000")
        driver = vllm_adapter.ADAPTER.driver(spec, handle)
        self.assertIsInstance(driver, HttpDriver)
        self.assertEqual((driver.url, driver.model, driver.timeout),
                         ("http://127.0.0.1:8000/v1/completions", "/w", 7))


class QuettaServeBenchRun(unittest.TestCase):
    def test_grid_spec_lists_every_cell(self):
        cfg = config("qwen3", "engine.name=quettaserve", "engine.model=/ckpt/qw")
        self.assertFalse(quettaserve.ADAPTER.needs_server(cfg))
        spec = quettaserve.ADAPTER.translate(cfg)
        self.assertEqual([(c, b) for c, b, _ in spec.cells], list(cfg.grid.cells))
        for ctx, bs, cell in spec.cells:
            self.assertEqual(cell, quettaserve.ADAPTER.translate(cfg, cell=(ctx, bs)))

    def test_serve_run_writes_measured_rows(self):
        cfg = config("llama", "engine.model=/w", "mode.grid=smoke")
        cells = tuple((ctx, bs, LaunchSpec(
            adapter="quettaserve", bench_argv=(sys.executable, str(FAKE)), cwd=".",
            env={"QW_BENCH_PROMPT": str(ctx), "QW_BATCH": str(bs), "QW_WORLD": "1"},
            ok_pattern=r"^LOOP ", log_header=f">>> ctx={ctx} bs={bs} world=1"))
            for ctx, bs in ((1024, 1), (1024, 16)))
        with tempfile.TemporaryDirectory() as td:
            driver = BenchDriver(Path(td) / "bench.log", cells)
            serve.run(None, driver, cfg.mode, RunWriter(td, cfg))
            doc = json.loads((Path(td) / "run.json").read_text())
            rows = [json.loads(line) for line in
                    (Path(td) / "requests.jsonl").read_text().splitlines()]
        self.assertEqual(doc["steps"], [
            {"ctx": 1024, "bs": 1, "tp": 1, "ms_per_step": 3.5, "measured": True},
            {"ctx": 1024, "bs": 16, "tp": 1, "ms_per_step": 3.5, "measured": True}])
        self.assertEqual([(r["cell"]["bs"], r["measured_ms_per_step"]) for r in rows],
                         [(1, 3.5), (16, 3.5)])
        self.assertNotIn("meta", doc["summary"]["grid"])


class QuettaServeThroughCli(unittest.TestCase):
    """qwen3's 16 cells on the fake bench: max_model_len 10000 drops the four
    ctx=16384 cells, a 141 GiB GPU at cap_frac 0.3 drops (8192, 64)."""
    SETS = [*PINNED, "engine.name=quettaserve", "engine.model=/ckpt/qw", "engine.tp=1",
            f"engine.extra.bench_bin={FAKE}", "engine.max_model_len=10000", "mode.cap_frac=0.3"]

    def qbench(self, *argv):
        args = ["run", str(RUNS / "qwen3.yaml"), *(a for s in self.SETS for a in ("--set", s)),
                *argv]
        out = io.StringIO()
        with contextlib.redirect_stdout(out), mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(BenchDriver, "device", return_value=("H200", 141 * 2**30)):
            self.assertEqual(cli.main(args), 0)
        return out.getvalue()

    def test_dry_run_prints_each_cells_shell_line(self):
        lines = [line for line in self.qbench("--dry-run").splitlines()
                 if line.startswith("cell ")]
        self.assertEqual(len(lines), 16)
        self.assertEqual(lines[0], (
            "cell ctx=1024 bs=1 = env XVAL_LINK_PROFILE=nvlink NCCL_IB_DISABLE=1 QS_KERNELS=rust QW_MODE=graph "
            "QW_CKPT=/ckpt/qw QW_WORLD=1 QW_TIMING_STEPS=100 QW_BATCH=1 QW_BENCH_PROMPT=1024 "
            f"QW_MAX_SEQ=1152 {FAKE} --nocapture"))

    def test_run_skips_then_measures_the_rest(self):
        with tempfile.TemporaryDirectory() as td:
            out = self.qbench("--results-dir", td)
            run = Path(td) / "qwen3"
            doc = json.loads((run / "run.json").read_text())
            rows = [json.loads(line) for line in (run / "requests.jsonl").read_text().splitlines()]
        skips = [line for line in out.splitlines() if line.startswith("SKIP")]
        # cap: (141 * 0.9 - 54 weights - 5 reserve) GiB / 65536 B per token;
        # need: (8192 + 100 timing steps + 28 headroom) * 64 > 0.3 * cap.
        self.assertEqual(skips, ["SKIP ctx=8192 bs=64 need_kv=532480 cap=1112473",
                                 *(f"SKIP ctx=16384 bs={bs} reason=maxlen" for bs in (1, 4, 16, 64))])
        self.assertEqual(len(rows), 11)
        self.assertEqual({r["measured_ms_per_step"] for r in rows}, {3.5})
        self.assertNotIn((8192, 64), {(r["cell"]["ctx"], r["cell"]["bs"]) for r in rows})
        self.assertEqual(len(doc["steps"]), 11)
        self.assertTrue(all(s["measured"] for s in doc["steps"]))
        self.assertEqual(len(doc["summary"]["grid"]["events"]), 5)


if __name__ == "__main__":
    unittest.main()
