"""GPU smoke test for the moved crossval vLLM baseline.

Runs the vLLM half of the crossval (the part that lives in this repo and can
run without QuettaServe's engine): vllm.sh on the one-cell "smoke" grid, which
drives vmin_fit.py to load the model and fit a decode slope, then writes
baselines/vllm-llama.json. Confirms the moved scripts execute under real vLLM,
not just parse.

Skips unless CUDA is present AND MODEL points at a Llama weights dir, so hosted
CI (no GPU) skips it. GPU_UTIL defaults to 0.6 so the load fits a shared GPU.
Run it from the gpu-smoke workflow on a self-hosted GPU runner:

    MODEL=/path/to/Llama-3.1-8B-Instruct python -m pytest tests/test_crossval_smoke_gpu.py -v
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
        baseline = CROSSVAL / "baselines" / "vllm-llama.json"
        self.assertTrue(baseline.is_file(), "vllm.sh wrote no baseline")
        data = json.loads(baseline.read_text())
        self.assertIn("gpu", data, "baseline missing gpu metadata")


if __name__ == "__main__":
    unittest.main()
