# profiling/tests/test_probe_smoke_gpu.py

"""Execute the kernel probes with tiny grids on a real GPU and check each one
writes a non-empty table. Skips wholesale without CUDA, so hosted CI stays
green; on a GPU host (or a self-hosted runner) the probes really run:

    python -m pytest profiling/tests/test_probe_smoke_gpu.py -v

The python must have torch and vllm. The all-reduce case needs 2 free GPUs.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

PROBES_DIR = Path(__file__).resolve().parents[1] / "kernel_composed" / "probes"
LABEL = "SMOKE"


def run_probe(out_dir: Path, script: str, *args: str, torchrun_world: int = 0) -> None:
    """Run one probe to completion against out_dir; raises on nonzero exit."""
    launcher = [sys.executable]
    if torchrun_world:
        launcher += ["-m", "torch.distributed.run", f"--nproc_per_node={torchrun_world}"]
    cmd = launcher + [
        str(PROBES_DIR / script), "--gpu-label", LABEL, "--out-dir", str(out_dir),
    ] + list(args)
    subprocess.run(cmd, check=True, timeout=600)


def written_tables(out_dir: Path) -> list[Path]:
    """Non-empty CSV/JSON tables the probe left under out_dir."""
    return [
        p
        for suffix in ("*.csv", "*.json")
        for p in out_dir.rglob(suffix)
        if p.stat().st_size > 0
    ]


def test_gemm_probe_tiny_grid(tmp_path):
    run_probe(tmp_path, "gemm_probe.py", "--pairs", "4096:4096", "--m-axis", "1,64,1024")
    assert written_tables(tmp_path), "gemm probe wrote no table"


def test_elementwise_probe_capped(tmp_path):
    run_probe(tmp_path, "elementwise_probe.py", "--max-mem-gb", "2")
    assert written_tables(tmp_path), "elementwise probe wrote no table"


def test_cross_attn_probe_tiny_grid(tmp_path):
    run_probe(
        tmp_path, "cross_attn_probe.py",
        "--n-heads", "8", "--n-kv-heads", "2", "--head-dim", "128",
        "--q-axis", "128,512", "--ctx-axis", "1024,4096",
    )
    assert written_tables(tmp_path), "cross-attn probe wrote no table"


def test_flash_attn_probe_tiny_grid(tmp_path):
    run_probe(
        tmp_path, "flash_attn_probe.py",
        "--tag", "tp1", "--n-heads", "8", "--n-kv-heads", "2", "--head-dim", "128",
        "--layers", "4", "--kv-axis", "1024,4096", "--batch-axis", "1,8",
    )
    assert written_tables(tmp_path), "flash-attn probe wrote no table"


def test_vllm_allreduce_probe_world2(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    run_probe(tmp_path, "vllm_allreduce_probe.py", torchrun_world=2)
    assert written_tables(tmp_path), "vllm all-reduce probe wrote no table"
