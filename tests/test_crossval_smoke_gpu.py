"""GPU smoke for the vLLM baseline: qbench run on the one-cell smoke grid
must write run.json with a step section and the gpu metadata. Skips without
CUDA and MODEL, so hosted CI collects and skips it; a GPU runner executes
it.

    MODEL=/path/to/weights python -m pytest tests/test_crossval_smoke_gpu.py
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

REPO = Path(__file__).resolve().parents[1]


def _cuda_and_model():
    if torch is None or not os.environ.get("MODEL"):
        return False
    return torch.cuda.is_available()


@unittest.skipUnless(_cuda_and_model(), "needs CUDA and MODEL (a Llama weights dir)")
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


if __name__ == "__main__":
    unittest.main()
