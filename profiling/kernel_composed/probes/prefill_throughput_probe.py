"""Sustained FRESH-prefill throughput of one live vLLM engine (tokens/s at saturation).

Why: the saturated P engines of the 4-node PD cells process ~8.0k tok/s of 8192-token
chunks (0% APC hit, GPU 100%) while the composed kernel model prices a fresh 8192-token
step at 1620 ms = 5.06k tok/s. The c=1 calibration (single-stream miss sweep) never
exercised a queue of large fresh prompts. This probe does: it keeps ``--concurrency``
random-token prompts of ``--prompt-len`` tokens in flight for ``--seconds`` with
max_tokens=1 and reports completed prompt tokens per second, prompts per second and
per-request latency. Fresh = every prompt is a new random id sequence (no prefix hits;
``/reset_prefix_cache`` first). Single node, no proxy, ~2 minutes.

  python profiling/kernel_composed/probes/prefill_throughput_probe.py \
      --base-url http://127.0.0.1:8300 --gpu-label RTXPRO6000 \
      --prompt-len 8192 --concurrency 4 --seconds 90 --out build/probes/prefill_tput_tp8.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics as st
import time
from pathlib import Path

import aiohttp


async def main(a):
    rng = random.Random(a.seed)
    done = []            # (t_end, prompt_tokens, latency_s)
    stop = time.monotonic() + a.seconds
    async with aiohttp.ClientSession() as h:
        try:
            async with h.post(a.base_url + "/reset_prefix_cache") as r:
                print("reset_prefix_cache:", r.status)
        except Exception as e:  # noqa: BLE001
            print("reset_prefix_cache failed:", e)

        async def worker(i):
            while time.monotonic() < stop:
                prompt = [rng.randrange(1000, 100000) for _ in range(a.prompt_len)]
                body = {"model": a.model, "prompt": prompt, "max_tokens": 1, "temperature": 0.0}
                t0 = time.monotonic()
                try:
                    async with h.post(a.base_url + "/v1/completions", json=body,
                                      timeout=aiohttp.ClientTimeout(total=600)) as r:
                        j = await r.json()
                        if r.status != 200:
                            print("error", r.status, str(j)[:200]); continue
                        pt = (j.get("usage") or {}).get("prompt_tokens", a.prompt_len)
                        cached = ((j.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                        done.append((time.monotonic(), pt, time.monotonic() - t0, cached))
                except Exception as e:  # noqa: BLE001
                    print("exc", repr(e)[:120])
        t_start = time.monotonic()
        await asyncio.gather(*(worker(i) for i in range(a.concurrency)))
    if not done:
        raise SystemExit("no completions")
    # steady state: drop the first 20% of wall time
    t_lo = t_start + 0.2 * (done[-1][0] - t_start)
    steady = [d for d in done if d[0] >= t_lo]
    span = done[-1][0] - t_lo
    toks = sum(d[1] for d in steady)
    out = {"gpu": a.gpu_label, "base_url": a.base_url, "prompt_len": a.prompt_len,
           "concurrency": a.concurrency,
           "n_total": len(done), "n_steady": len(steady), "steady_span_s": round(span, 1),
           "tokens_per_s": round(toks / span, 1), "prompts_per_s": round(len(steady) / span, 3),
           "latency_s_p50": round(st.median(d[2] for d in steady), 3),
           "latency_s_mean": round(st.mean(d[2] for d in steady), 3),
           "cached_tokens_mean": round(st.mean(d[3] for d in steady), 1),
           "ms_per_8192_equiv": round(8192 / (toks / span) * 1e3, 0)}
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8300")
    ap.add_argument("--gpu-label", default="H200")
    ap.add_argument("--model", default="Qwen3-235B-A22B")
    ap.add_argument("--prompt-len", type=int, default=8192)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    asyncio.run(main(ap.parse_args()))
