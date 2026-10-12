"""GPU smoke: the kernel probes on tiny grids each write a non-empty table, and with MODEL set a qbench run on the one-cell smoke grid writes run.json with a step and gpu metadata; the module skips without CUDA.

    MODEL=/path/to/weights python -m pytest tests/contracts/test_probes_gpu.py -v
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

REPO = Path(__file__).resolve().parents[2]
LABEL = "SMOKE"


def run_probe(out_dir: Path, probe: str, *args: str, torchrun_world: int = 0) -> None:
    """Run one probe to completion against out_dir; raises on nonzero exit."""
    launcher = [sys.executable]
    if torchrun_world:
        launcher += ["-m", "torch.distributed.run", f"--nproc_per_node={torchrun_world}"]
    cmd = launcher + [
        "-m", f"quettabench.probes.{probe}", "--gpu-label", LABEL, "--out", str(out_dir),
    ] + list(args)
    subprocess.run(cmd, check=True, timeout=600, cwd=REPO)


def written_tables(out_dir: Path) -> list[Path]:
    """Non-empty CSV/JSON tables the probe left under out_dir."""
    return [
        p
        for suffix in ("*.csv", "*.json")
        for p in out_dir.rglob(suffix)
        if p.stat().st_size > 0
    ]


def test_gemm_probe_tiny_grid(tmp_path):
    run_probe(tmp_path, "gemm", "--pairs", "4096:4096", "--m-axis", "1,64,1024")
    assert written_tables(tmp_path), "gemm probe wrote no table"


def test_elementwise_probe_capped(tmp_path):
    run_probe(tmp_path, "elementwise", "--max-mem-gb", "2")
    assert written_tables(tmp_path), "elementwise probe wrote no table"


def test_cross_attn_probe_tiny_grid(tmp_path):
    run_probe(
        tmp_path, "cross_attn",
        "--n-heads", "8", "--n-kv-heads", "2", "--head-dim", "128",
        "--q-axis", "128,512", "--ctx-axis", "1024,4096",
    )
    assert written_tables(tmp_path), "cross-attn probe wrote no table"


def test_flash_attn_probe_tiny_grid(tmp_path):
    run_probe(
        tmp_path, "flash_attn",
        "--tag", "tp1", "--n-heads", "8", "--n-kv-heads", "2", "--head-dim", "128",
        "--layers", "4", "--kv-axis", "1024,4096", "--batch-axis", "1,8",
    )
    assert written_tables(tmp_path), "flash-attn probe wrote no table"


def test_vllm_allreduce_probe_world2(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    run_probe(tmp_path, "vllm_allreduce", torchrun_world=2)
    assert written_tables(tmp_path), "vllm all-reduce probe wrote no table"


@unittest.skipUnless(os.environ.get("MODEL"), "needs MODEL (a Llama weights dir)")
class CrossvalBaselineSmoke(unittest.TestCase):
    def test_vllm_baseline_smoke_grid(self):
        env = {**os.environ, "GPU_UTIL": os.environ.get("GPU_UTIL", "0.6")}
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(
                [os.environ.get("VLLM_PY", sys.executable), "-m", "quettabench.cli", "run",
                 "configs/runs/llama.yaml", "--set", "mode.grid=smoke",
                 "--set", "run_name=llama-smoke",
                 "--set", f"engine.model={os.environ['MODEL']}", "--results-dir", td],
                cwd=REPO, check=True, timeout=1200, env=env,
            )
            doc = json.loads((Path(td) / "llama-smoke" / "run.json").read_text())
        self.assertEqual(doc["run_status"]["status"], "ok")
        self.assertTrue(doc["steps"], "the smoke cell produced no step")
        self.assertTrue(doc["summary"]["grid"]["meta"]["gpu"], "run missing gpu metadata")
