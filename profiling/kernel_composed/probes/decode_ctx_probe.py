#!/usr/bin/env python3
"""The ORDINARY serving decode step: B streams on DISTINCT, long contexts, measured live.

Why. In the colocated agentic cells 96-98% of engine steps are pure decode with every
running request in the batch, so a plan's TPOT is set by that step -- and on tp4pp2 the
served step (61 ms, client ITL p50 @0.1) is 27% slower than tp8's (48 ms), while the
shared-prefix decode probe (B streams on ONE 12k prefix) shows 7% and the composition 2%.
The difference between the probe and serving is the contexts: real batches hold 16-32
different sessions of 10-30k tokens each. This probe reproduces that.

Method. For each (ctx mode, B): build B prompts -- ``shared`` = one random-id prefix
plus a distinct suffix (the old probe's setting, the control), ``distinct`` = B
independent random-id prompts of the given length(s) -- pre-warm each one with
max_tokens=1 so its KV is resident in the prefix cache (the pool must hold B x ctx),
then fire all B streams at once (every prompt is an APC hit, so all B reach decode
together) and take the median inter-token gap over the steady middle of each stream.
Compared with KernelComposed.fused_step_ms(0, B, ctx_list) on the SAME context list.

  python decode_ctx_probe.py --base-url http://127.0.0.1:8300/v1 --model Qwen3-235B-A22B \\
      --model-yaml qwen3-235b-a22b --device-yaml device_spec/rtxpro6000.yaml --tp 8 \\
      --configs shared:12000:16 distinct:12000:16 distinct:20000:16 distinct:30000:16 \\
               distinct:20000:24 distinct:12000:32 mixed:8000-32000:16 \\
      --out build/probes/decode_ctx_tp8.csv

``mixed:LO-HI:B`` draws B lengths uniformly in [LO, HI]. ``churn:LEN:N`` prefills N throwaway
LEN-token prompts to fragment the pool before the configs that follow (no row). Writes one
CSV row per measured config.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics as st
import sys
import threading
import time
import urllib.request
from pathlib import Path

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
HERE = Path(__file__).resolve().parent


def _quettasim_root() -> Path:
    env = os.environ.get("QUETTASIM")
    if env:
        return Path(env)
    for parent in HERE.parents:
        if (parent / "engine" / "factory.py").is_file():
            return parent
    raise SystemExit("set QUETTASIM to the QuettaSim checkout (engine/factory.py not found)")


sys.path.insert(0, str(_quettasim_root()))


def _post(base_url: str, body: dict, timeout: float = 900.0):
    req = urllib.request.Request(f"{base_url}/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return _OPENER.open(req, timeout=timeout)


def warm(base_url: str, model: str, ids: list[int]) -> int:
    """Prefill one prompt into the prefix cache; returns the cached_tokens vLLM reports
    (0 on the first warm, ~len(ids) if it was already resident)."""
    with _post(base_url, {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0.0}) as r:
        j = json.loads(r.read())
    return int(((j.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0))


def stream(base_url: str, model: str, ids: list[int], gen: int, out: list, idx: int) -> None:
    times, cached = [], None
    try:
        with _post(base_url, {"model": model, "prompt": ids, "max_tokens": gen, "temperature": 0.0,
                              "stream": True, "ignore_eos": True,
                              "stream_options": {"include_usage": True}}) as r:
            for line in r:
                if not line.startswith(b"data:"):
                    continue
                if b'"text"' in line:
                    times.append(time.perf_counter())
                elif b"prompt_tokens_details" in line:
                    try:
                        u = json.loads(line[5:])["usage"]
                        cached = int((u.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
                    except Exception:  # noqa: BLE001
                        pass
    except Exception as e:  # noqa: BLE001
        out[idx] = {"err": repr(e)[:200], "times": times, "cached": cached}
        return
    out[idx] = {"err": None, "times": times, "cached": cached}


def prompts_for(mode: str, spec: str, B: int, rng: random.Random) -> list[list[int]]:
    if mode == "shared":
        n = int(spec)
        base = [rng.randrange(1000, 100000) for _ in range(n)]
        return [base + [rng.randrange(1000, 100000) for _ in range(8)] for _ in range(B)]
    if mode == "distinct":
        n = int(spec)
        return [[rng.randrange(1000, 100000) for _ in range(n)] for _ in range(B)]
    if mode == "mixed":
        lo, hi = (int(x) for x in spec.split("-"))
        return [[rng.randrange(1000, 100000) for _ in range(rng.randint(lo, hi))] for _ in range(B)]
    raise SystemExit(f"bad ctx mode {mode!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8300/v1")
    ap.add_argument("--model", default="Qwen3-235B-A22B")
    ap.add_argument("--model-yaml", default="qwen3-235b-a22b")
    ap.add_argument("--device-yaml", required=True)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--ep", action="store_true")
    ap.add_argument("--configs", nargs="+",
                    default=["shared:12000:16", "distinct:12000:16", "distinct:20000:16",
                             "distinct:30000:16", "distinct:20000:24", "distinct:12000:32",
                             "mixed:8000-32000:16"])
    ap.add_argument("--gen", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from engine.factory import build_kernel_composed_cost
    cost = build_kernel_composed_cost(a.device_yaml, a.model_yaml, tp=a.tp, pp=a.pp,
                                      ep=(a.tp if a.ep else 1), expert_shards=(a.tp if a.ep else None),
                                      gpu_mem_util=0.90)
    rng = random.Random(a.seed)
    rows = []
    for cfg in a.configs:
        mode, spec, b = cfg.split(":"); B = int(b)
        if mode == "churn":
            # churn:LEN:N -- prefill N distinct random prompts of LEN tokens (max_tokens=1) and
            # keep none of them: several pool-fulls of allocations and prefix-cache evictions,
            # so the blocks the NEXT configs' streams get are scattered across the pool the way
            # a long-running engine's are, instead of freshly carved. No row is written.
            n, L = B, int(spec); t0 = time.perf_counter()
            for _ in range(n):
                warm(a.base_url, a.model, [rng.randrange(1000, 100000) for _ in range(L)])
            print(f"  churn: {n} x {L} tokens ({n*L/1e6:.2f}M) in {time.perf_counter()-t0:.0f}s", flush=True)
            continue
        prompts = prompts_for(mode, spec, B, rng)
        ctxs = [len(p) for p in prompts]
        t0 = time.perf_counter()
        for p in prompts:                       # resident KV for every prompt
            warm(a.base_url, a.model, p)
        hits = [warm(a.base_url, a.model, p) for p in prompts[:2]]   # confirm residency
        print(f"  {cfg}: warmed {sum(ctxs)} tokens in {time.perf_counter()-t0:.0f}s "
              f"(re-warm cached_tokens={hits})", flush=True)
        out: list = [None] * B
        th = [threading.Thread(target=stream, args=(a.base_url, a.model, prompts[i], a.gen, out, i))
              for i in range(B)]
        for t in th:
            t.start()
        for t in th:
            t.join()
        pool, per_stream, errs, cached = [], [], 0, []
        for o in out:
            if o is None or o["err"]:
                errs += 1; continue
            tms = o["times"]
            gaps = [(tms[i] - tms[i - 1]) * 1e3 for i in range(1, len(tms))]
            if len(gaps) >= 10:
                lo, hi = int(len(gaps) * 0.2), int(len(gaps) * 0.8)
                mid = gaps[lo:hi]
                pool += mid; per_stream.append(st.median(mid))
            if o["cached"] is not None:
                cached.append(o["cached"])
        if not pool:
            print(f"    no samples ({errs} stream errors)", flush=True); continue
        real = st.median(pool)
        # the sim on the same context list, at the middle of the generation
        ctx_mid = [c + a.gen * 0.5 for c in ctxs]
        sim = cost.fused_step_ms(0, B, tuple(float(c) for c in ctx_mid))
        rows.append({"tp": a.tp, "pp": a.pp, "ep": int(a.ep), "mode": mode, "spec": spec, "batch": B,
                     "ctx_mean": round(sum(ctxs) / B), "ctx_min": min(ctxs), "ctx_max": max(ctxs),
                     "real_ms": round(real, 2), "real_stream_med_min": round(min(per_stream), 2),
                     "real_stream_med_max": round(max(per_stream), 2), "sim_ms": round(sim, 2),
                     "apc_hit_frac": round(st.median(cached) / (sum(ctxs) / B), 3) if cached else "",
                     "n": len(pool), "errors": errs})
        print(f"    B={B:>3} ctx~{sum(ctxs)/B:6.0f}: real step {real:6.1f} ms  sim {sim:6.1f}  "
              f"(per-stream median {min(per_stream):.1f}..{max(per_stream):.1f}, errors {errs})", flush=True)
    out_p = Path(a.out); out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {out_p}")


if __name__ == "__main__":
    main()
