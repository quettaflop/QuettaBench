"""GPU smoke for the vLLM baseline: vllm.sh on the one-cell smoke grid must
write a baseline json. Skips without CUDA and MODEL, so hosted CI collects
and skips it; a GPU runner executes it.

    MODEL=/path/to/weights python -m pytest tests/test_crossval_smoke_gpu.py
"""

import json
import os
import subprocess
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

CROSSVAL = Path(__file__).resolve().parents[1] / "scripts" / "crossval"


def _cuda_and_model():
    if torch is None or not os.environ.get("MODEL"):
        return False
    return torch.cuda.is_available()


@unittest.skipUnless(_cuda_and_model(), "needs CUDA and MODEL (a Llama weights dir)")
class CrossvalBaselineSmoke(unittest.TestCase):
    def test_vllm_baseline_smoke_grid(self):
        env = {**os.environ, "GPU_UTIL": os.environ.get("GPU_UTIL", "0.6")}
        subprocess.run(
            [
                str(CROSSVAL / "vllm.sh"), "llama", "smoke",
                os.environ["MODEL"], os.environ.get("VLLM_PY", "python3"),
            ],
            check=True, timeout=1200, env=env,
        )
        baseline = CROSSVAL / "baselines" / "vllm-llama-smoke.json"
        self.assertTrue(baseline.is_file(), "vllm.sh wrote no baseline")
        data = json.loads(baseline.read_text())
        self.assertIn("gpu", data, "baseline missing gpu metadata")


if __name__ == "__main__":
    unittest.main()
