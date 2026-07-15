#!/usr/bin/env python3
# profiling/probes/serving_stage_split.py
"""Serving-level per-stage cost breakdown of c1 prefill TTFT vs (new, cached) tokens.

This is the LIVE-SERVER successor to ``prefill_stage_split.py`` (offline, host/device split
UNRELIABLE) and an extension of ``live_ttft_probe.py`` (full client wall TTFT, trustworthy).

GOAL
----
Partition the wall TTFT of a c1 ``/v1/chat/completions`` request into per-stage spans as a
function of (new, cached) prompt tokens:

    HTTP-recv/parse | chat-template | tokenize | enqueue/ZMQ-IPC | scheduler-admit |
    model-forward (DEVICE) | framework-dispatch | sample | detokenize | response-stream

and resolve the two open residuals from prefill_law_defit_trace.md:
  * the NEW above-roofline dispatch residual (~6 ms/1k)  -> framework-dispatch stage
  * the CACHED host residual (~3.7 ms/1k)                 -> HTTP-recv/parse + chat-template + IPC

It does this with THREE measurement lanes, layered so the un-patched lanes already give a
reliable 3-way split and the patched lanes (run only if the verified hooks land) refine it:

  LANE 0 (always, no patch)  : client perf_counter around aiohttp SSE  -> wall TTFT  (== live_ttft_probe)
  LANE A (no patch)          : Prometheus /metrics _sum delta-scrape per c1 request
                               -> queue_span + prefill_span + frontend_residual (3-way partition)
  LANE B (run command emitted): nsys --cuda-graph-trace=node wrapping the live server, with
                               NVTX request-window markers from a worker monkeypatch
                               -> DEVICE kernel time vs framework-dispatch (the offline split could not do)

The offline torch.profiler self-time split failed three ways (see prefill_stage_split_results.md):
  (1) eager: Sigma(self_time) over-counts wall -> host = wall - device goes negative.
  (2) CUDA graphs: per-kernel CUPTI activities collapse into one graph-exec row -> device ~ 0.
  (3) v1 multiprocessing: forward runs in the EngineCore SUBPROCESS -> main-proc profiler sees 0 kernels.
Lane A sidesteps device timing entirely (uses the engine's own SCHEDULED->first_token wall via
Prometheus). Lane B fixes the device split by attaching nsys to the CUDA-owning subprocess and
re-expanding graph nodes (--cuda-graph-trace=node), bracketed by in-engine NVTX ranges.

------------------------------------------------------------------------------------------------
RUN NOTES (learned the hard way -- carried over from prefill_stage_split.py / live_ttft_probe.py)
------------------------------------------------------------------------------------------------
  * Run from a CLEAN cwd. A stray ``flash_attn.py`` in the cwd shadows the flash_attn pip package
    that vLLM needs ('flash_attn' is not a package'). This script also strips its own dir from
    sys.path defensively (the cuda_events scripts do the same).
  * Server stats MUST be ON (default) for Lane A. Do NOT pass --disable-log-stats. The /metrics
    endpoint only populates the histograms when stats are enabled.   [VERIFY ON H100]
  * The ``new`` tail is freshly random PER TRIAL so it is a REAL cache miss (else the warmup primes
    it and every measured trial is a hit -- the original prefill_stage_split bug).
  * c1 only: concurrency 1 is what makes the Prometheus _sum delta-scrape isolate one request --
    between two scrapes exactly one request's worth of time is added to each cumulative _sum.
  * env (per profiling/docs/h100_setup.md, but GPU 7 per the task, NOT 6):
        CUDA_VISIBLE_DEVICES=7
        TMPDIR=/data48/kevinlau/tmp
        XDG_CACHE_HOME=/data48/kevinlau/tmp/.cache
        PYTHON=~/miniconda3/envs/vllm/bin/python
        MODEL=/data48/kevinlau/models/Llama-3.1-8B-Instruct

Example (Lane 0 + Lane A, no patch, self-launching the server on GPU 7):
    CUDA_VISIBLE_DEVICES=7 TMPDIR=/data48/kevinlau/tmp XDG_CACHE_HOME=/data48/kevinlau/tmp/.cache \
      ~/miniconda3/envs/vllm/bin/python \
      profiling/gpu_profiling/vllm/serving_stage_split.py \
      --model /data48/kevinlau/models/Llama-3.1-8B-Instruct --port 8771 \
      --out profile_data/results/serving_stage_split_H100.csv

Lane B (device split) is a SEPARATE nsys-wrapped run; this script PRINTS the exact command
(see ``--emit-nsys-cmd``) rather than launching nsys itself, so the profiled and un-profiled
walls stay comparable. See profiling/docs/serving_stage_split_plan.md for the full recipe.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import statistics as st
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# --- defensive: strip our own dir from sys.path so a stray flash_attn.py can't shadow the pkg ---
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
for _p in [e for e in sys.path if e == _SCRIPT_DIR]:
    sys.path.remove(_p)

import aiohttp  # noqa: E402  (after the sys.path scrub)


# ------------------------------------------------------------------------------------------------
# Prompt construction -- reused verbatim from live_ttft_probe.py so the wall TTFT reproduces the
# banked prefill_live_ttft_H100.csv rates (new 29.4 / cached 5.89 ms/1k). Words are ~1 Llama token.
# ------------------------------------------------------------------------------------------------
_VOCAB = ("the of and to in a is that for it as was with on be at by this had not are but from or "
          "have an they which one you were her all she there would their we him been has when who "
          "will more no if out so up said what its about into than them can only other new some "
          "could time these two may then do first any my now such like our over man me even most "
          "made after also did many before must through back years where much your way well down").split()
_RNG = random.Random(0)
_WORDS = [_RNG.choice(_VOCAB) for _ in range(40000)]   # fixed cached-prefix corpus (HIT after prime)
_TAIL = random.Random(777)                              # fresh new-tail per trial -> real MISS on new


def cached_prefix(cached: int) -> str:
    return " ".join(_WORDS[: int(cached * 0.96)])


def fresh_tail(new: int) -> str:
    return " ".join(_TAIL.choice(_VOCAB) for _ in range(int(new * 0.96)))


# ------------------------------------------------------------------------------------------------
# Prometheus /metrics histogram _sum delta-scrape  (LANE A, no patch).
#
# vLLM v1 exposes per-request lifecycle timings ONLY as Prometheus histograms (and OTel spans),
# NOT on the OpenAI HTTP response body. In particular RequestOutput.metrics (the old v0
# RequestMetrics with arrival_time/first_scheduled_time/model_forward_time/time_in_queue) is
# ALWAYS None on the v1 HTTP generate path in 0.19.0 -- the v0 object was removed; the v1 timestamps
# live in engine-core RequestStateStats, aggregated into IterationStats, surfaced ONLY via /metrics.
#   [VERIFY ON H100] RequestOutput.metrics is None on the v1 server path; do NOT build on it.
#
# Each histogram exposes a cumulative ``*_sum`` counter. At c1, between a scrape BEFORE send and a
# scrape AFTER the SSE first token, exactly one request's value is added to each _sum, so
# (sum_after - sum_before) isolates THAT request's value. This is the whole trick.
#
#   [VERIFY ON H100] the exact metric names below on vLLM 0.19.0. Names are stable across 0.8->0.19
#   but request_prefill_time / request_queue_time are newer histograms -- confirm they EXIST and
#   populate (server launched WITHOUT --disable-log-stats). If a name is absent the scrape just
#   yields 0.0 for that span and the column is left blank (frontend_residual still computable).
# ------------------------------------------------------------------------------------------------
# metric_key -> (prometheus metric base name).  We scrape "<base>_sum".
_PROM_METRICS = {
    # queued_ts -> scheduled_ts  (scheduler admit / queue wait). ~0 at c1.
    "queue_span_s":   "vllm:request_queue_time_seconds",
    # scheduled_ts -> first_token_ts  (model-forward WALL incl. python/dispatch; stages 7+8).
    "prefill_span_s": "vllm:request_prefill_time_seconds",
    # full e2e for a cross-check.
    "e2e_s":          "vllm:e2e_request_latency_seconds",
    # to pin new+cached against the OpenAI usage block.
    "prompt_tokens":  "vllm:request_prompt_tokens",
    # vLLM's own TTFT histogram (first-token latency) -- cross-check vs client wall.
    "ttft_s":         "vllm:time_to_first_token_seconds",
}


def _parse_prom_sums(text: str) -> dict:
    """Parse a Prometheus exposition text body -> {metric_key: cumulative_sum_float}.

    We only need the ``<base>_sum`` line of each histogram (a single float, labels ignored at c1
    because there is one model / one finished-request stream). If a model label is present the
    _sum line still parses; with one model there is exactly one such line.
        [VERIFY ON H100] that there is a single model label series for these histograms (one model
        loaded) so the bare ``<base>_sum`` match is unambiguous. If multiple label series appear,
        tighten the match to the model_name label.
    """
    wanted = {f"{base}_sum": key for key, base in _PROM_METRICS.items()}
    out = {key: None for key in _PROM_METRICS}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        # line looks like:  vllm:request_prefill_time_seconds_sum{model_name="..."} 1.234
        sp = line.rsplit(" ", 1)
        if len(sp) != 2:
            continue
        name_and_labels, val = sp
        metric_name = name_and_labels.split("{", 1)[0]
        key = wanted.get(metric_name)
        if key is not None:
            try:
                out[key] = float(val)
            except ValueError:
                pass
    return out


async def scrape_metrics(session: aiohttp.ClientSession, base_url: str) -> dict:
    """GET <base_url>/metrics and return the parsed _sum dict (None per missing metric)."""
    try:
        async with session.get(base_url + "/metrics") as resp:
            if resp.status != 200:
                return {k: None for k in _PROM_METRICS}
            return _parse_prom_sums(await resp.text())
    except aiohttp.ClientError:
        return {k: None for k in _PROM_METRICS}


def _sub(after, before):
    if after is None or before is None:
        return None
    return after - before


# ------------------------------------------------------------------------------------------------
# Client SSE timing  (LANE 0, no patch) -- mirrors live_ttft_probe.ttft_once but ALSO splits the
# client side into connect/send vs first-byte so stages 1 + 11 (HTTP recv + response-stream) are
# not blindly lumped. ``stream_options.include_usage`` pins prompt_tokens (= new+cached actual).
# ------------------------------------------------------------------------------------------------
async def ttft_once(session, url, model, content, request_id=None):
    payload = {"model": model, "messages": [{"role": "user", "content": content}],
               "max_tokens": 1, "temperature": 0.0, "stream": True,
               "stream_options": {"include_usage": True}}
    headers = {"Authorization": "Bearer test", "Content-Type": "application/json"}
    if request_id is not None:
        # runner.py sends a stable id via X-Request-Id (--trace-request-ids) but never reads it
        # back. We propagate it so a future Lane-B per-request join key exists.
        #   [VERIFY ON H100] header name vLLM 0.19.0 honors for client-supplied request ids
        #   (candidates: "X-Request-Id"). If unsupported it is simply ignored.
        headers["X-Request-Id"] = request_id
    t_send = time.perf_counter()
    async with session.post(url, json=payload, headers=headers) as resp:
        t_status = time.perf_counter()          # headers received (stage 1 upper bound on loopback)
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
        ttft_ms = None
        usage = None
        async for raw in resp.content:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            ds = line[len("data:"):].strip()
            if ds == "[DONE]":
                break
            ch = json.loads(ds)
            cc = ch.get("choices", [])
            if ttft_ms is None and cc and cc[0].get("delta", {}).get("content") is not None:
                ttft_ms = (time.perf_counter() - t_send) * 1000.0
            if ch.get("usage"):
                usage = ch["usage"]   # arrives in the final chunk when include_usage=True
        return {
            "ttft_ms": ttft_ms,
            "client_connect_ms": (t_status - t_send) * 1000.0,   # stage 1 (recv/connect), loopback
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
        }


# ------------------------------------------------------------------------------------------------
# One (new, cached) measurement at c1, wrapping each trial with a Prometheus before/after scrape.
# ------------------------------------------------------------------------------------------------
async def measure(session, base_url, model, cached, new, trials):
    chat_url = base_url + "/v1/chat/completions"
    prefix = cached_prefix(cached)
    if cached > 0:
        await ttft_once(session, chat_url, model, prefix)   # prime -> cached blocks resident (HIT)

    samples = []   # list of per-trial dicts
    for i in range(trials + 1):   # +1 warmup (discarded)
        tail = fresh_tail(new)
        content = (prefix + " " + tail) if cached > 0 else tail
        before = await scrape_metrics(session, base_url)
        res = await ttft_once(session, chat_url, model, content,
                              request_id=f"sss-{cached}-{new}-{i}")
        after = await scrape_metrics(session, base_url)
        # Per-request spans via _sum delta (LANE A). At c1 one request advances each _sum once.
        queue_span = _sub(after["queue_span_s"], before["queue_span_s"])
        prefill_span = _sub(after["prefill_span_s"], before["prefill_span_s"])
        e2e_span = _sub(after["e2e_s"], before["e2e_s"])
        ttft_prom = _sub(after["ttft_s"], before["ttft_s"])
        samples.append({
            "ttft_ms": res["ttft_ms"],
            "client_connect_ms": res["client_connect_ms"],
            "prompt_tokens": res["prompt_tokens"],
            "queue_span_ms": None if queue_span is None else queue_span * 1000.0,
            "prefill_span_ms": None if prefill_span is None else prefill_span * 1000.0,
            "e2e_span_ms": None if e2e_span is None else e2e_span * 1000.0,
            "ttft_prom_ms": None if ttft_prom is None else ttft_prom * 1000.0,
        })

    body = samples[1:]   # drop warmup

    def med(key):
        vals = [s[key] for s in body if s[key] is not None]
        return st.median(vals) if vals else None

    ttft_ms = med("ttft_ms")
    queue_ms = med("queue_span_ms")
    prefill_ms = med("prefill_span_ms")
    # frontend_residual = wall - (queue + prefill) - (client connect already inside wall on loopback)
    # This is stages 1-5 + 9-11 lumped: the HOST FRONTEND + IPC residual the offline LLM lacked.
    frontend_residual = None
    if ttft_ms is not None and queue_ms is not None and prefill_ms is not None:
        frontend_residual = ttft_ms - queue_ms - prefill_ms

    # actual prompt tokens from the usage block (pins the regressor against requested new+cached).
    pt = [s["prompt_tokens"] for s in body if s["prompt_tokens"] is not None]
    prompt_tokens_actual = st.median(pt) if pt else None

    return {
        "new": new, "cached": cached, "n": trials,
        "prompt_tokens_actual": prompt_tokens_actual,
        "ttft_ms": ttft_ms,
        "client_connect_ms": med("client_connect_ms"),
        "queue_span_ms": queue_ms,
        "prefill_span_ms": prefill_ms,
        "frontend_residual_ms": frontend_residual,
        "e2e_span_ms": med("e2e_span_ms"),
        "ttft_prom_ms": med("ttft_prom_ms"),
    }


# ------------------------------------------------------------------------------------------------
# Server lifecycle -- reuse the cached_prefill_batch_ttft.py launch + health-poll verbatim.
# Same bench config the live probes already validated against the fitted serving law:
#   prefix-cache + chunked-prefill, gpu-mem 0.9 (per results doc; arg-overridable), GPU 7.
# CRITICAL: stats ON (no --disable-log-stats) so /metrics populates for LANE A.   [VERIFY ON H100]
# ------------------------------------------------------------------------------------------------
def wait_health(port, timeout=420):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def launch_server(model, port, gpu_mem, max_model_len, api_key, hash_algo, log_path, tp=1):
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", model, "--served-model-name", "llama",
           "--host", "127.0.0.1", "--port", str(port),
           "--dtype", "bfloat16", "--gpu-memory-utilization", str(gpu_mem),
           "--max-model-len", str(max_model_len),
           "--tensor-parallel-size", str(tp),
           "--enable-prefix-caching", "--enable-chunked-prefill",
           "--api-key", api_key,
           # keep request logging quiet but STATS ON (do NOT add --disable-log-stats).
           "--no-enable-log-requests"]
    # prefix_caching_hash_algo: toggling sha256 vs builtin attributes the block-hash cost to the
    # frontend/scheduler-admit residual (the external-lit SHA-256 lead). Guarded -- older/newer
    # CLIs may name it differently.   [VERIFY ON H100] exact flag name on 0.19.0.
    if hash_algo:
        cmd += ["--prefix-caching-hash-algo", hash_algo]
    print("launching:", " ".join(cmd), flush=True)
    logf = open(log_path, "w")
    return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)


# ------------------------------------------------------------------------------------------------
# LANE B emit -- print (do NOT run) the exact nsys wrapper command for the DEVICE split run, plus
# the worker-side NVTX/CUDA-event monkeypatch the bench operator injects via --worker-extension-cls
# (or a sitecustomize shim). We only EMIT it so the profiled wall stays separate from Lane 0/A.
# Everything here is VERIFY-ON-H100 (paths vary across vLLM versions).
# ------------------------------------------------------------------------------------------------
_WORKER_EXT_HINT = r'''
# --- LANE B worker-side hook (emit-only; the operator drops this into a module on the H100) ---
# Inject via:  vllm serve ... --worker-extension-cls my_ext.NvtxForwardTimer
#   [VERIFY ON H100] the v1 worker entrypoint path + that --worker-extension-cls is the supported
#   in-subprocess injection on 0.19.0. Candidates:
#       vllm.v1.worker.gpu_worker.Worker.execute_model
#       vllm.v1.worker.gpu_model_runner.GPUModelRunner.execute_model
#   and that torch.cuda.nvtx ranges emitted here land in the nsys NVTX_EVENTS table joinable to
#   CUPTI_ACTIVITY_KIND_KERNEL rows by timestamp.
#
# import torch, time
# from vllm.v1.worker.gpu_model_runner import GPUModelRunner   # VERIFY path
# _orig = GPUModelRunner.execute_model
# _RING = []  # (start_evt, end_evt, host_wall_ms, new, cached)
# def timed_execute_model(self, scheduler_output, *a, **k):
#     # read per-step (new,cached) from scheduler_output.num_scheduled_tokens (VERIFY shape);
#     # at c1 exactly one prefilling request per step -> 1:1 map.
#     s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
#     torch.cuda.nvtx.range_push("vllm_prefill_step")   # NVTX window for the nsys kernel join
#     t0 = time.perf_counter(); s.record()
#     out = _orig(self, scheduler_output, *a, **k)
#     e.record(); host_wall = (time.perf_counter() - t0) * 1e3
#     torch.cuda.nvtx.range_pop()
#     _RING.append((s, e, host_wall))   # elapsed_time read OFF the hot path (sync on flush)
#     return out
# GPUModelRunner.execute_model = timed_execute_model
'''


def emit_nsys_cmd(model, port, gpu_mem, max_model_len, api_key, out_sqlite):
    py = sys.executable
    nsys_cmd = (
        f"nsys profile -t cuda,nvtx,cublas --cuda-graph-trace=node \\\n"
        f"  --capture-range=cudaProfilerApi --capture-range-end=stop \\\n"
        f"  -o {out_sqlite} -f true -- \\\n"
        f"  {py} -m vllm.entrypoints.openai.api_server \\\n"
        f"    --model {model} --served-model-name llama --host 127.0.0.1 --port {port} \\\n"
        f"    --dtype bfloat16 --gpu-memory-utilization {gpu_mem} "
        f"--max-model-len {max_model_len} \\\n"
        f"    --enable-prefix-caching --enable-chunked-prefill --api-key {api_key} \\\n"
        f"    --no-enable-log-requests --worker-extension-cls my_ext.NvtxForwardTimer"
    )
    print("\n" + "=" * 96)
    print("LANE B (DEVICE split) -- run this SEPARATELY (nsys-wrapped server), then drive it with")
    print("this same script (--no-launch, pointing at the nsys server). Parse the .sqlite with the")
    print("existing extractor pattern (profiling/process/_legacy/extract_nsys_prefill_breakdown.py),")
    print("restricting Sigma(end-start) to the per-step prefill NVTX range:")
    print("=" * 96)
    print(nsys_cmd)
    print(_WORKER_EXT_HINT)
    print("After nsys export to sqlite, device_kernel_ms(window) = SUM(k.end-k.start) over")
    print("CUPTI_ACTIVITY_KIND_KERNEL rows whose [start,end] fall inside the 'vllm_prefill_step'")
    print("NVTX range (NVTX_EVENTS table). framework_dispatch_ms = prefill_span_ms - device_kernel_ms.")
    print("=" * 96 + "\n")


# ------------------------------------------------------------------------------------------------
# CSV + driver
# ------------------------------------------------------------------------------------------------
_CSV_FIELDS = [
    "new", "cached", "n", "prompt_tokens_actual",
    "ttft_ms",              # LANE 0 client wall (ground truth, == live_ttft_probe)
    "client_connect_ms",    # stage 1 (HTTP recv/connect), loopback ~const
    "queue_span_ms",        # stage 6  (LANE A Prom request_queue_time)
    "prefill_span_ms",      # stages 7+8 wall (LANE A Prom request_prefill_time)
    "frontend_residual_ms", # stages 1-5 + 9-11 lumped = wall - queue - prefill (host frontend + IPC)
    "e2e_span_ms",          # LANE A Prom e2e (cross-check)
    "ttft_prom_ms",         # LANE A Prom ttft (cross-check vs client wall)
]


async def run_sweep(base_url, model, news, cacheds, trials, out_path):
    rows = []
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        # Sanity: confirm /metrics is reachable and at least one target histogram is present.
        probe = await scrape_metrics(s, base_url)
        present = [k for k, v in probe.items() if v is not None]
        print(f"/metrics reachable; histograms present at start: {present}", flush=True)
        if "prefill_span_s" not in present:
            print("[warn] vllm:request_prefill_time_seconds_sum NOT found -- LANE A prefill_span "
                  "will be blank. VERIFY the metric name / that stats are enabled on 0.19.0.",
                  flush=True)
        for cached in cacheds:
            for new in news:
                r = await measure(s, base_url, model, cached, new, trials)
                rows.append(r)
                pt = r["prompt_tokens_actual"]
                print(f"  new={new:>5} cached={cached:>6}  ttft={_f(r['ttft_ms']):>8}  "
                      f"queue={_f(r['queue_span_ms']):>7}  prefill={_f(r['prefill_span_ms']):>8}  "
                      f"frontend={_f(r['frontend_residual_ms']):>8}  "
                      f"pt={pt if pt is not None else '?'}", flush=True)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: _round(r.get(k)) for k in _CSV_FIELDS})
    print(f"\nwrote {out}", flush=True)
    return rows


def _f(x):
    return "n/a" if x is None else f"{x:.2f}"


def _round(x):
    if isinstance(x, float):
        return round(x, 3)
    return "" if x is None else x


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="/data48/kevinlau/models/Llama-3.1-8B-Instruct")
    ap.add_argument("--served-model-name", default="llama",
                    help="model name sent in the OpenAI payload (matches --served-model-name)")
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--api-key", default="test")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--tensor-parallel-size", type=int, default=1,
                    help="tp for the launched server (G3 like-for-like pair: run tp1 AND tp2 "
                         "with THIS same script/stack; needs tp GPUs in CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--hash", default="sha256", choices=["sha256", "builtin", ""],
                    help="prefix_caching_hash_algo for the launched server (block-hash attribution)")
    ap.add_argument("--news", default="8,128,512,1024,2048", help="fresh-token counts to sweep")
    ap.add_argument("--cacheds", default="0,2000,8000,16000", help="cached-prefix token counts")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out", default="profile_data/results/serving_stage_split_H100.csv")
    ap.add_argument("--no-launch", action="store_true",
                    help="server already running (e.g. the nsys-wrapped LANE B server)")
    ap.add_argument("--server-log", default="vllm_server_stage_split.log")
    ap.add_argument("--emit-nsys-cmd", action="store_true",
                    help="print the LANE B nsys wrapper command + worker hook and exit")
    a = ap.parse_args()

    base_url = f"http://127.0.0.1:{a.port}"

    if a.emit_nsys_cmd:
        emit_nsys_cmd(a.model, a.port, a.gpu_mem, a.max_model_len, a.api_key,
                      out_sqlite="serving_stage_split_nsys")
        return

    news = [int(x) for x in a.news.split(",")]
    cacheds = [int(x) for x in a.cacheds.split(",")]

    proc = None
    if not a.no_launch:
        proc = launch_server(a.model, a.port, a.gpu_mem, a.max_model_len, a.api_key,
                             a.hash or None, a.server_log, tp=a.tensor_parallel_size)
    try:
        if not wait_health(a.port):
            print(f"SERVER DID NOT BECOME HEALTHY -- see {a.server_log}", flush=True)
            sys.exit(1)
        print("server healthy; starting c1 stage-split sweep", flush=True)
        asyncio.run(run_sweep(base_url, a.served_model_name, news, cacheds, a.trials, a.out))
        # Always remind the operator how to get the DEVICE split (LANE B is a separate run).
        emit_nsys_cmd(a.model, a.port, a.gpu_mem, a.max_model_len, a.api_key,
                      out_sqlite="serving_stage_split_nsys")
        print("Next: analyze with profiling/gpu_profiling/vllm/analyze_serving_stage_split.py", flush=True)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except Exception:
                proc.kill()


if __name__ == "__main__":
    main()
