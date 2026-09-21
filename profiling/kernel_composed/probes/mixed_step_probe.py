#!/usr/bin/env python3
"""The MIXED step: a fresh prefill chunk co-scheduled with a decode batch, measured live.

Why. Colocated serving of the agentic traces lives almost entirely in this step: 93% of
turns prefill < 2048 NEW tokens (the rest of the context is an APC hit) while the
engine is decoding for everyone else, so a turn's TTFT and every resident's TBT are set
by how long a (chunk P, batch B) step takes. The two probes the device curves come from
measure the corners -- prefill_throughput_probe (P, 0) and decode_batch_probe (0, B) --
and the composition prices the mixed step as one shared pass with a chunk curve
(prefill_chunk_scale_mixed) that has never been measured directly on this device.

Method. Against ONE colocated vLLM engine: hold exactly B streams decoding together on a
shared cached prefix of --ctx tokens (as decode_batch_probe does), then inject fresh
random-id prompts of P tokens (max_tokens=1) one at a time. The step that carries the
chunk shows up in every decode stream as one long inter-token gap, so

    real_mixed_ms(P, B) = median over injections of ( median over streams of the
                          longest gap that ends inside the injection's window )

and the streams' gaps outside every window are the decode-only step (0, B) as a check
against the decode probe. B = 0 is the pure-prefill corner (the request's own latency).
P + B <= max_num_batched_tokens (8192) keeps each chunk inside one step. Compared with
KernelComposed.fused_step_ms(P, B, [ctx]*B) and its decode corner.

  python mixed_step_probe.py --base-url http://127.0.0.1:8300/v1 --model Qwen3-235B-A22B \\
      --model-yaml qwen3-235b-a22b --device-yaml device_spec/rtxpro6000.yaml --tp 8 \\
      --ctx 12000 --batches 0 8 16 32 48 --chunks 128 256 512 1024 2048 4096 \\
      --out build/probes/mixed_tp8.csv

Writes one CSV row per (batch, chunk). Does NOT edit any grid; tools/fit_model_curves.py
--mixed turns the rows into prefill_chunk_scale_mixed knots.
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


class Stream:
    """One decode stream; records the wall time of every token it receives."""

    def __init__(self, base_url: str, model: str, prompt: str, gen: int, stop: threading.Event):
        self.times: list[float] = []
        self.err: str | None = None
        self._args = (base_url, model, prompt, gen, stop)
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        base_url, model, prompt, gen, stop = self._args
        body = json.dumps({"model": model, "prompt": prompt, "max_tokens": gen,
                           "temperature": 0.0, "stream": True, "ignore_eos": True}).encode()
        req = urllib.request.Request(f"{base_url}/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with _OPENER.open(req, timeout=900) as r:
                for line in r:
                    if stop.is_set():
                        break                     # closing the socket aborts the request
                    if line.startswith(b"data:") and b'"text"' in line:
                        self.times.append(time.perf_counter())
        except Exception as e:  # noqa: BLE001
            self.err = repr(e)[:200]

    def gaps(self):
        t = self.times
        return [(t[i], (t[i] - t[i - 1]) * 1e3) for i in range(1, len(t))]   # (end, ms)


def inject(base_url: str, model: str, ids: list[int]) -> tuple[float, float, float]:
    """One fresh prefill request; returns (t_send, t_done, latency_ms)."""
    body = json.dumps({"model": model, "prompt": ids, "max_tokens": 1,
                       "temperature": 0.0}).encode()
    req = urllib.request.Request(f"{base_url}/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with _OPENER.open(req, timeout=600) as r:
        j = json.loads(r.read())
    t1 = time.perf_counter()
    cached = ((j.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    inject.last_cached = cached          # the caller knows how much SHOULD be resident
    return t0, t1, (t1 - t0) * 1e3


def reset_prefix_cache(base_url: str) -> None:
    root = base_url[:-3] if base_url.endswith("/v1") else base_url
    try:
        _OPENER.open(urllib.request.Request(root + "/reset_prefix_cache", data=b"",
                                            method="POST"), timeout=30).read()
    except Exception as e:  # noqa: BLE001
        print(f"reset_prefix_cache failed: {e!r}"[:160], flush=True)


def measure_batch(a, B: int, chunks: list[int], rng: random.Random, cost) -> list[dict]:
    rows = []
    stop = threading.Event()
    streams: list[Stream] = []
    # --resident R: every injected chunk is the continuation of ONE R-token prefix that is
    # already in the prefix cache (warmed here, once), so the step computes P new tokens
    # that attend to R resident ones -- the shape of a real agentic turn, where the chunk
    # is a few hundred tokens on top of a 15-20k cached context. 0 = fresh chunks.
    resident_ids: list[int] = []
    if a.resident > 0:
        resident_ids = [rng.randrange(1000, 100000) for _ in range(a.resident)]
        t0w, t1w, _ = inject(a.base_url, a.model, resident_ids)
        print(f"    warmed a {a.resident}-token resident prefix in {(t1w - t0w)*1e3:.0f} ms", flush=True)
    if B > 0:
        prefix = "The quick brown fox. " * (a.ctx // 5)
        # warm the shared prefix so all B streams hit the cache and decode together
        w = Stream(a.base_url, a.model, prefix, 4, threading.Event()); w.thread.start(); w.thread.join()
        streams = [Stream(a.base_url, a.model,
                          prefix + f" Distinct continuation seed {rng.randint(0, 10**9)} number {i}: ",
                          a.gen, stop) for i in range(B)]
        for s in streams:
            s.thread.start()
        time.sleep(a.settle_s)                      # let the batch reach steady decode
    windows: list[tuple[float, float]] = []
    try:
        for P in chunks:
            if B + P > a.max_batched_tokens:
                print(f"    B={B} P={P}: skipped, {B}+{P} > max_num_batched_tokens", flush=True)
                continue
            lat, wins, hits = [], [], []
            for _ in range(a.injections):
                ids = resident_ids + [rng.randrange(1000, 100000) for _ in range(P)]
                t0, t1, ms = inject(a.base_url, a.model, ids)
                lat.append(ms); wins.append((t0, t1)); windows.append((t0, t1))
                hits.append(getattr(inject, "last_cached", 0))
                time.sleep(a.gap_s)
            if a.resident > 0 and hits and st.median(hits) < 0.9 * a.resident:
                print(f"      warning: resident prefix hit only {st.median(hits):.0f} of {a.resident} tokens", flush=True)
            elif a.resident == 0 and hits and max(hits) > 0:
                print(f"      warning: fresh chunk reported {max(hits)} cached tokens", flush=True)
            if B > 0:
                per_inj = []
                for t0, t1 in wins:
                    spikes = []
                    for s in streams:
                        inwin = [ms for end, ms in s.gaps() if t0 <= end <= t1 + 0.03]
                        if inwin:
                            spikes.append(max(inwin))
                    if spikes:
                        per_inj.append(st.median(spikes))
                real_mixed = st.median(per_inj) if per_inj else float("nan")
            else:
                real_mixed = st.median(lat)         # pure prefill: the step IS the latency
            sim_mixed = cost.fused_step_ms(P, B, tuple([float(a.ctx)] * B)) if B else \
                cost.fused_step_ms(P, 0, 0)
            sim_dec = cost.fused_step_ms(0, B, tuple([float(a.ctx)] * B)) if B else 0.0
            rows.append({"tp": a.tp, "pp": a.pp, "ep": int(a.ep), "ctx": a.ctx, "batch": B,
                         "resident": a.resident,
                         "chunk": P, "real_mixed_ms": round(real_mixed, 2),
                         "real_prefill_lat_ms": round(st.median(lat), 2),
                         "sim_mixed_ms": round(sim_mixed, 2), "sim_decode_ms": round(sim_dec, 2),
                         "n_inj": len(lat)})
            print(f"    B={B:>3} P={P:>5}: real mixed {real_mixed:8.1f} ms  sim {sim_mixed:8.1f}"
                  f"  (prefill req latency {st.median(lat):7.1f} ms)", flush=True)
    finally:
        stop.set()
    for s in streams:
        s.thread.join(timeout=30)
    if B > 0:
        # decode-only step: every gap outside every injection window, steady middle
        quiet = []
        for s in streams:
            g = s.gaps()
            lo, hi = int(len(g) * 0.1), int(len(g) * 0.9)
            for end, ms in g[lo:hi]:
                if not any(t0 - 0.05 <= end <= t1 + 0.05 for t0, t1 in windows):
                    quiet.append(ms)
        dec = st.median(quiet) if quiet else float("nan")
        for r in rows:
            r["real_decode_ms"] = round(dec, 2)
        errs = [s.err for s in streams if s.err]
        print(f"    B={B:>3} decode-only step: real {dec:.1f} ms  sim {rows[0]['sim_decode_ms'] if rows else 0:.1f}"
              f"{'  stream errors: ' + str(len(errs)) if errs else ''}", flush=True)
    else:
        for r in rows:
            r["real_decode_ms"] = 0.0
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8300/v1")
    ap.add_argument("--model", default="Qwen3-235B-A22B")
    ap.add_argument("--model-yaml", default="qwen3-235b-a22b")
    ap.add_argument("--device-yaml", required=True)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--ep", action="store_true", help="the engine runs --enable-expert-parallel")
    ap.add_argument("--ctx", type=int, default=12000)
    ap.add_argument("--gen", type=int, default=4000, help="stream length; streams are aborted when done")
    ap.add_argument("--batches", type=int, nargs="+", default=[0, 8, 16, 32, 48])
    ap.add_argument("--chunks", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096])
    ap.add_argument("--injections", type=int, default=6)
    ap.add_argument("--gap-s", type=float, default=0.5)
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument("--max-batched-tokens", type=int, default=8192)
    ap.add_argument("--resident", type=int, default=0,
                    help="inject chunks as continuations of one cached R-token prefix (0 = fresh)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from engine.factory import build_kernel_composed_cost
    cost = build_kernel_composed_cost(a.device_yaml, a.model_yaml, tp=a.tp, pp=a.pp,
                                      expert_shards=(a.tp if a.ep else None), gpu_mem_util=0.90)
    rng = random.Random(a.seed)
    reset_prefix_cache(a.base_url)
    rows = []
    for B in a.batches:
        print(f"  batch {B}", flush=True)
        rows += measure_batch(a, B, a.chunks, rng, cost)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
