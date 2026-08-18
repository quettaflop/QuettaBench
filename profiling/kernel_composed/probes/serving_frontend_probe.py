#!/usr/bin/env python3
"""Live-server probe -> the device-YAML `frontend:` and `serving:` blocks, for a GPU
that has NO ground-truth serving runs yet (for a GPU that HAS GT, prefer
QuettaSim's ``profiling/derive_serving_yaml.py``, which fits the same blocks from GT).

Same decomposition as derive_serving_yaml.py -- the frontend/host cost is
(measured latency - kernel-composed compute prediction) -- but the measurements come
from controlled requests against a live vLLM OpenAI server instead of GT JSONs:

  1. single-stream, sweeping prompt length -> floor_ms + new_ms_per_token (TTFT residual)
  2. single-stream decode                  -> decode_overhead_ms_per_layer (TPOT residual)
  3. concurrency sweep                     -> mult_curve / lanes_curve reference points

Start vLLM separately (the GT harness does the same), then point this at it:

  vllm serve <model> --tensor-parallel-size 1 --port 8000 &
  python serving_frontend_probe.py --gpu-label A100 --model Qwen3.5-9B \
      --model-yaml $QUETTASIM/engine/device_spec/models/qwen3.5-9b.yaml \
      --base-url http://localhost:8000/v1

Rank/step timing is read from the stream (time-to-first-token). Prints the blocks;
does NOT edit the YAML. UNVALIDATED without a live server -- run on the target GPU.
"""
from __future__ import annotations
import argparse, json, statistics as st, time, urllib.request
from pathlib import Path
import sys

from _paths import quettasim_root

ROOT = quettasim_root()
sys.path.insert(0, str(ROOT))
from engine.loaders.config_loader import load_kernel_gpu, load_kernel_model  # noqa: E402
from engine.backends.kernel_composed_cost import KernelComposedCost  # noqa: E402

_SWEEP_PROMPTS = [128, 256, 512, 1024, 2048]   # single-stream prompt lengths (tokens)
_CONC_SWEEP = [1, 5, 10, 20, 40, 80]


def _stream_once(base_url: str, model: str, prompt_tokens: int, out_tokens: int = 8) -> tuple[float, float]:
    """(ttft_ms, tpot_ms) for one streamed completion of `prompt_tokens` 'hi' tokens."""
    body = json.dumps({
        "model": model, "prompt": "hi " * prompt_tokens, "max_tokens": out_tokens,
        "stream": True, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(f"{base_url}/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first_at = None
    n = 0
    with urllib.request.urlopen(req) as r:
        for line in r:
            if not line.startswith(b"data:") or b"[DONE]" in line:
                continue
            now = time.perf_counter()
            if first_at is None:
                first_at = now
            n += 1
    last = time.perf_counter()
    ttft_ms = (first_at - t0) * 1000.0
    tpot_ms = ((last - first_at) / max(1, n - 1)) * 1000.0 if n > 1 else 0.0
    return ttft_ms, tpot_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-label", required=True)
    ap.add_argument("--model", required=True, help="served model name (OpenAI 'model' field)")
    ap.add_argument("--model-yaml", required=True, help="engine model YAML for the compute prediction")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--out-dir", default=None, help="unused (kept for run_probe.sh parity)")
    a = ap.parse_args()

    gpu = load_kernel_gpu(ROOT / "engine" / "device_spec" / f"{a.gpu_label.lower()}.yaml")
    model = load_kernel_model(Path(a.model_yaml))
    cost = KernelComposedCost(model, gpu, tp=a.tp)   # compute-only prediction (frontend off in the raw YAML)

    # 1+2. single-stream: TTFT residual vs prompt length -> floor + new rate; TPOT residual -> decode overhead.
    import numpy as np
    X, y_ttft, tpot_resid = [], [], []
    for p in _SWEEP_PROMPTS:
        ttft, tpot = _stream_once(a.base_url, a.model, p)
        compute_prefill = cost.prefill_step_ms([p])          # engine prefill compute for p tokens
        y_ttft.append(ttft - compute_prefill)
        X.append([1.0, float(p)])
        compute_decode = cost.decode_step_ms([p])
        tpot_resid.append(tpot - compute_decode)
    coef, *_ = np.linalg.lstsq(np.array(X), np.array(y_ttft), rcond=None)
    floor, rate_n = max(0.0, float(coef[0])), max(0.0, float(coef[1]))
    dov = max(0.0, st.median(tpot_resid)) / max(1, int(model.n_layers))

    # 3. concurrency sweep: median TTFT reference points (operator fits the curves from these
    #    + the single-stream floor; mult/lanes are degenerate from one observable, so emit the
    #    measured shape rather than an over-confident fit).
    conc_ttft = {}
    for c in _CONC_SWEEP:
        samples = [_stream_once(a.base_url, a.model, 512)[0] for _ in range(c)]
        conc_ttft[c] = round(st.median(samples), 1)

    print(f"[{a.gpu_label} / {a.model} / tp{a.tp}] live-server frontend probe\n")
    print(f"# paste into device_spec/{a.gpu_label.lower()}.yaml (serving_frontend_probe.py):")
    print("frontend:")
    print(f"  floor_ms: {round(floor, 2)}")
    print(f"  new_ms_per_token: {round(rate_n, 6)}")
    print("  cached_ms_per_token: 0.0   # measure with a cache-hit sweep if the deploy uses APC")
    print("  # mult_curve / lanes_curve: fit from the concurrency reference points below")
    print(f"  # median TTFT (ms) by concurrency at 512-token prompt: {conc_ttft}")
    print("serving:")
    print(f"  decode_overhead_ms_per_layer: {round(dov, 5)}")
    print(f"  ttft_overhead_ms: {round(floor, 2)}")
    print("  max_concurrent_prefills: 2")


if __name__ == "__main__":
    main()
