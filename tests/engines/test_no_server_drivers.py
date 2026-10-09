"""Which driver a run gets, and the QuettaServe bench run end to end.

A grid run on vLLM launches nothing and times in-process; a serve run talks
to the launched server's /v1/completions. A QuettaServe grid run carries
every cell's spec and serve.run sends them through BenchDriver, writing one
row per cell with its measured ms per step.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

from quettabench.config import loader
from quettabench.engines import launcher, quettaserve
from quettabench.engines import vllm as vllm_adapter
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
            self.assertEqual(cell, quettaserve.ADAPTER.translate(cfg, (ctx, bs)))

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


if __name__ == "__main__":
    unittest.main()
