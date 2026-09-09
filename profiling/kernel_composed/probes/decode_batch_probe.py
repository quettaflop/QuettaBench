#!/usr/bin/env python3
"""Controlled decode-step-vs-batch curve, to pin the batched-decode pricing at concurrency.

Cross-validation found the PD sim's c16 TBT p50 over-predicts by ~21%, localised (via a
component decomposition + the decode server's throughput log) to the BATCHED decode step
scaling slightly too steeply -- the batch-1 step and the batch distribution both match a
clean run, but the sim's step grows faster with batch than reality, and the MoE term is
the dominant grower (sim MoE 3.3 -> 10.7 ms from batch 1 -> 11). The server-log estimate
is noisy (few samples, mixed contexts), so this probe measures the curve cleanly: it
holds exactly B requests in steady-state decode and times the inter-token latency, for a
sweep of B, at a fixed long context.

Method. Against a SINGLE colocated vLLM server (the decode kernels are identical to a PD
decode pool's -- pure decode, no prefill interleave once warm), fire B concurrent
streaming completions that share one long cached prefix (so prefill is a near-instant APC
hit and every request is in the decode phase together) and each generate `--gen` tokens.
The median inter-token gap across the steady middle of the generation is the real decode
step time at batch B. Sweep B and compare to KernelComposed.fused_step_ms(0, B, ctx).

  vllm serve <model> --tensor-parallel-size 4 --port 8000 --enable-prefix-caching ...
  python profiling/probes/decode_batch_probe.py --base-url http://127.0.0.1:8000/v1 \
      --model qwen3-235b --model-yaml qwen3-235b-a22b-fp8 --tp 4 \
      --ctx 20000 --batches 1 2 4 8 11 16 24 --gen 64

Prints the measured vs sim decode-step curve and the per-batch ratio; writes
cuda_event/decode_batch/<gpu>_tp<N>.csv. Disjoint from the pd_sweep trace eval (synthetic
shared-prefix content), so its curve is legitimate calibration for the batched-decode
composition. Does NOT edit any grid -- it reports the correction to apply.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / ".." / ".."))


def _one(base_url: str, model: str, prompt: str, gen: int, itl_out: list) -> None:
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": gen,
                       "temperature": 0.0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(f"{base_url}/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    prev = None
    gaps = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if line.startswith(b"data:") and b'"text"' in line:
                now = time.perf_counter()
                if prev is not None:
                    gaps.append((now - prev) * 1e3)
                prev = now
    itl_out.append(gaps)


def _batch_itl(base_url: str, model: str, prompt: str, B: int, gen: int, diverse: bool = True) -> list:
    """Median-of-middle inter-token gaps with B requests decoding together.

    ``diverse`` appends a distinct suffix per request so each routes to DIFFERENT experts
    (a real serving batch of distinct sessions activates far more of the 128 experts than
    B identical requests, which share one routing and make MoE decode look artificially
    flat). The shared long prefix keeps prefill a fast APC hit so all B reach decode
    together; the distinct tail drives divergent routing during the timed generation."""
    import random
    per = []
    prompts = [prompt + (f" Distinct continuation seed {random.randint(0, 10**9)} number {i}: "
                         if diverse else "") for i in range(B)]
    th = [threading.Thread(target=_one, args=(base_url, model, prompts[i], gen, per)) for i in range(B)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    # steady middle: drop first/last 20% of each request's gaps, pool
    pool = []
    for gaps in per:
        if len(gaps) >= 5:
            lo, hi = int(len(gaps) * 0.2), int(len(gaps) * 0.8)
            pool += gaps[lo:hi]
    return pool


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen3-235b")
    ap.add_argument("--model-yaml", default="qwen3-235b-a22b-fp8")
    ap.add_argument("--device-yaml", default=str(HERE / ".." / ".." / "device_spec" / "h200.yaml"))
    ap.add_argument("--gpu-label", default="H200")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=20000, help="shared cached context length (tokens)")
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 11, 16, 24])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from engine.factory import build_kernel_composed_cost
    cost = build_kernel_composed_cost(a.device_yaml, a.model_yaml, tp=a.tp, gpu_mem_util=0.92)

    # one shared long prompt: cached after the first request, so all B are decode-only
    prompt = "The quick brown fox. " * (a.ctx // 5)
    print("warming the shared prefix ...", flush=True)
    _batch_itl(a.base_url, a.model, prompt, 1, 8)
    time.sleep(1.0)

    rows = []
    print(f"\n{'batch':>6}{'n':>7}{'real_ms':>9}{'sim_ms':>9}{'ratio':>7}")
    for B in a.batches:
        pool = _batch_itl(a.base_url, a.model, prompt, B, a.gen)
        if not pool:
            print(f"{B:>6}  (no samples)"); continue
        real = st.median(pool)
        sim = cost.fused_step_ms(0, B, tuple([float(a.ctx)] * B))
        rows.append({"gpu": a.gpu_label, "tp": a.tp, "ctx": a.ctx, "batch": B,
                     "real_ms": round(real, 3), "sim_ms": round(sim, 3),
                     "ratio": round(sim / real, 3), "n": len(pool)})
        print(f"{B:>6}{len(pool):>7}{real:>9.2f}{sim:>9.2f}{sim / real:>7.2f}", flush=True)

    if rows:
        over = [r for r in rows if r["batch"] >= 8]
        if over:
            print(f"\nmedian sim/real at batch>=8: {st.median(r['ratio'] for r in over):.2f} "
                  f"(1.00 = perfect; >1 = sim over-prices batched decode)")
    out = Path(a.out) if a.out else (Path(a.device_yaml).parent.parent / "data" / "kernel_data"
                                     / "cuda_event" / "decode_batch" / f"{a.gpu_label}_tp{a.tp}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
