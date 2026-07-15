#!/usr/bin/env python3
# profiling/probes/serving_herd_scaling.py
"""Herd-scaling probe: how c1 prefill TTFT and its per-stage spans grow when N
identical requests arrive as a SYNCHRONIZED barrier (asyncio.gather).

Companion to serving_stage_split.py (c1-only). Every request in a burst shares the SAME
primed cached prefix (APC hit -> GPU only prefills each request's small fresh `new` tail),
so GPU prefill work stays ~flat across the burst and the TTFT growth with concurrency
isolates FRONTEND + scheduler SERIALIZATION -- the mechanism the v2 queue sim under-
predicts in the sub-saturation band.

TWO JOBS:
  1. VERIFY server-side. The client wall TTFT can be inflated by THIS single-process asyncio
     client serializing N concurrent SSE reads. So we ALSO scrape vLLM's own
     `time_to_first_token_seconds` histogram (delta/conc = mean SERVER TTFT, immune to the
     client event loop). server_frontend = server_ttft - queue - prefill. If THAT grows with
     conc, the serialization is genuinely in the server frontend, not the client.
  2. CHARACTERIZE. Sweep (new, cached) x conc so the per-request frontend service F and its
     token-dependence + sub-linear GPU-overlap can be fit for a serving-frontend term.

Per (new, cached, conc): prime the prefix, fire `conc` requests concurrently (each a fresh
`new` tail), scrape /metrics _sum before/after the burst -> mean per-request spans
(delta/conc), record client wall TTFT median/max.

Run (GPU 7, self-launches the server):
  CUDA_VISIBLE_DEVICES=7 TMPDIR=/data48/kevinlau/tmp XDG_CACHE_HOME=/data48/kevinlau/tmp/.cache \
    ~/miniconda3/envs/vllm/bin/python \
    profiling/gpu_profiling/vllm/serving_herd_scaling.py \
    --news 128,2048 --cacheds 0,8000 --concs 1,5,10,20 \
    --out profile_data/results/serving_herd_scaling_H100.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import statistics as st
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
for _p in [e for e in sys.path if e == _SCRIPT_DIR]:
    sys.path.remove(_p)

import aiohttp  # noqa: E402

# --- prompt construction (verbatim from serving_stage_split.py: ~1 Llama token/word) ---
_VOCAB = ("the of and to in a is that for it as was with on be at by this had not are but from or "
          "have an they which one you were her all she there would their we him been has when who "
          "will more no if out so up said what its about into than them can only other new some "
          "could time these two may then do first any my now such like our over man me even most "
          "made after also did many before must through back years where much your way well down").split()
_RNG = random.Random(0)
_WORDS = [_RNG.choice(_VOCAB) for _ in range(40000)]
_TAIL = random.Random(777)


def cached_prefix(cached: int) -> str:
    return " ".join(_WORDS[: int(cached * 0.96)])


def fresh_tail(new: int) -> str:
    return " ".join(_TAIL.choice(_VOCAB) for _ in range(int(new * 0.96)))


_PROM_METRICS = {
    "queue_span_s":   "vllm:request_queue_time_seconds",
    "prefill_span_s": "vllm:request_prefill_time_seconds",
    "e2e_s":          "vllm:e2e_request_latency_seconds",
    "ttft_s":         "vllm:time_to_first_token_seconds",  # SERVER-side TTFT (client-loop immune)
}


def _parse_prom_sums(text: str) -> dict:
    wanted = {f"{base}_sum": key for key, base in _PROM_METRICS.items()}
    out = {key: None for key in _PROM_METRICS}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        sp = line.rsplit(" ", 1)
        if len(sp) != 2:
            continue
        name_and_labels, val = sp
        key = wanted.get(name_and_labels.split("{", 1)[0])
        if key is not None:
            try:
                out[key] = float(val)
            except ValueError:
                pass
    return out


async def scrape(session, base_url):
    try:
        async with session.get(base_url + "/metrics") as resp:
            if resp.status != 200:
                return {k: None for k in _PROM_METRICS}
            return _parse_prom_sums(await resp.text())
    except aiohttp.ClientError:
        return {k: None for k in _PROM_METRICS}


def _sub(a, b):
    return None if (a is None or b is None) else a - b


async def ttft_once(session, url, model, content, _retried=False):
    payload = {"model": model, "messages": [{"role": "user", "content": content}],
               "max_tokens": 1, "temperature": 0.0, "stream": True,
               "stream_options": {"include_usage": True}}
    headers = {"Authorization": "Bearer test", "Content-Type": "application/json"}
    t_send = time.perf_counter()
    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
            ttft_ms = None
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
            return ttft_ms
    except (aiohttp.ServerDisconnectedError, aiohttp.ClientOSError):
        # Stale keep-alive (uvicorn closes idle conns between trials); retry once on a
        # fresh connection with fresh timing -- the dead-socket attempt is discarded.
        if _retried:
            raise
        return await ttft_once(session, url, model, content, _retried=True)


async def burst(session, base_url, model, cached, new, conc, trials):
    """Fire `conc` concurrent requests (barrier), `trials` times. Returns mean per-request
    spans (delta/conc) for the SERVER metrics + the client wall TTFT distribution."""
    chat_url = base_url + "/v1/chat/completions"
    prefix = cached_prefix(cached)
    if cached > 0:
        await ttft_once(session, chat_url, model, prefix)  # prime -> APC hit for the burst

    c_med, c_max, q_ms, p_ms, sttft_ms = [], [], [], [], []
    c_all = []  # pooled per-request client TTFTs (staircase shape across the herd)
    for t in range(trials + 1):  # +1 warmup
        contents = [(prefix + " " + fresh_tail(new)) if cached > 0 else fresh_tail(new)
                    for _ in range(conc)]
        before = await scrape(session, base_url)
        res = await asyncio.gather(*[ttft_once(session, chat_url, model, c) for c in contents])
        after = await scrape(session, base_url)
        if t == 0:
            continue  # discard warmup
        tt = sorted(x for x in res if x is not None)
        if not tt:
            continue
        c_all.extend(tt)
        c_med.append(tt[len(tt) // 2])
        c_max.append(tt[-1])
        dq = _sub(after["queue_span_s"], before["queue_span_s"])
        dp = _sub(after["prefill_span_s"], before["prefill_span_s"])
        dt = _sub(after["ttft_s"], before["ttft_s"])
        if dq is not None:
            q_ms.append(dq * 1000.0 / conc)
        if dp is not None:
            p_ms.append(dp * 1000.0 / conc)
        if dt is not None:
            sttft_ms.append(dt * 1000.0 / conc)

    def med(xs):
        return st.median(xs) if xs else None

    def pct(xs, q_frac):
        if not xs:
            return None
        s = sorted(xs)
        return s[min(len(s) - 1, int(round(q_frac * (len(s) - 1))))]

    cm, q, p, sttft = med(c_med), med(q_ms), med(p_ms), med(sttft_ms)
    # SERVER frontend = server ttft - engine(queue+prefill): immune to client event-loop.
    server_frontend = None if None in (sttft, q, p) else sttft - q - p
    # CLIENT frontend = client wall - engine: includes any client-side serialization.
    client_frontend = None if None in (cm, q, p) else cm - q - p
    return {"new": new, "cached": cached, "conc": conc, "trials": trials,
            "ttft_client_med_ms": cm, "ttft_client_max_ms": med(c_max),
            "ttft_client_p10_ms": pct(c_all, 0.10), "ttft_client_p25_ms": pct(c_all, 0.25),
            "ttft_client_p75_ms": pct(c_all, 0.75), "ttft_client_p90_ms": pct(c_all, 0.90),
            "ttft_server_ms": sttft, "mean_queue_ms": q, "mean_prefill_ms": p,
            "server_frontend_ms": server_frontend, "client_frontend_ms": client_frontend}


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


def launch_server(model, port, gpu_mem, max_model_len, api_key, log_path):
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", model, "--served-model-name", "llama",
           "--host", "127.0.0.1", "--port", str(port),
           "--dtype", "bfloat16", "--gpu-memory-utilization", str(gpu_mem),
           "--max-model-len", str(max_model_len), "--tensor-parallel-size", "1",
           "--enable-prefix-caching", "--enable-chunked-prefill",
           "--api-key", api_key, "--prefix-caching-hash-algo", "sha256",
           "--no-enable-log-requests"]
    print("launching:", " ".join(cmd), flush=True)
    return subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)


_CSV_FIELDS = ["decoys", "new", "cached", "conc", "trials", "ttft_client_med_ms",
               "ttft_client_max_ms", "ttft_client_p10_ms", "ttft_client_p25_ms",
               "ttft_client_p75_ms", "ttft_client_p90_ms", "ttft_server_ms",
               "mean_queue_ms", "mean_prefill_ms", "server_frontend_ms",
               "client_frontend_ms"]


# --- decoy streaming load (frontend-under-load variant) --------------------------
#
# The idle-loop L(c) curve may not transfer to production: at mid concurrency the
# API server's event loop is simultaneously pumping SSE deltas for every decoding
# request while it tokenizes the arriving herd. `--decoys D` runs the burst sweep
# while D requests (unique small prompts, ignore_eos, auto-restarted) stream through
# the same server from a SEPARATE process (so the burst client's loop stays clean).
# Caveat: restarted decoys add small observations to the /metrics deltas (~128-tok
# prefills, small TTFTs); the burst signal is 10-100x larger.


async def _decoy_stream(session, url, model, content, max_tokens):
    payload = {"model": model, "messages": [{"role": "user", "content": content}],
               "max_tokens": max_tokens, "temperature": 1.0, "stream": True,
               "ignore_eos": True}
    headers = {"Authorization": "Bearer test", "Content-Type": "application/json"}
    async with session.post(url, json=payload, headers=headers) as resp:
        if resp.status != 200:
            raise RuntimeError(f"decoy HTTP {resp.status}")
        got_first = False
        async for raw in resp.content:
            if not got_first and raw.strip().startswith(b"data:"):
                got_first = True
                yield True  # first token seen
        yield False


async def _daemon_main(base_url, model, n, prompt_tokens, max_tokens):
    chat_url = base_url + "/v1/chat/completions"
    first_seen = [False] * n
    announced = False

    async def runner(i):
        nonlocal announced
        rng = random.Random(9000 + i)
        conn_words = " ".join(rng.choice(_VOCAB) for _ in range(int(prompt_tokens * 0.96)))
        while True:
            try:
                async for evt in _decoy_stream(session, chat_url, model,
                                               f"decoy {i}: " + conn_words, max_tokens):
                    if evt and not first_seen[i]:
                        first_seen[i] = True
                        if all(first_seen) and not announced:
                            announced = True
                            print("READY", flush=True)
            except Exception:
                await asyncio.sleep(0.5)

    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None),
                                     connector=conn) as session:
        await asyncio.gather(*[runner(i) for i in range(n)])


def _write_csv(rows, out_path):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else ("" if v is None else v))
                        for k, v in r.items()})


async def run(base_url, model, news, cacheds, concs, trials, out_path, rows=None, decoys=0):
    rows = [] if rows is None else rows
    conn = aiohttp.TCPConnector(limit=0)  # no client-side cap -> true concurrency
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600), connector=conn) as s:
        present = [k for k, v in (await scrape(s, base_url)).items() if v is not None]
        print(f"/metrics histograms present: {present}  (decoys={decoys})", flush=True)
        for cached in cacheds:
            for new in news:
                for conc in concs:
                    r = await burst(s, base_url, model, cached, new, conc, trials)
                    r["decoys"] = decoys
                    rows.append(r)
                    _write_csv(rows, out_path)  # incremental: a crash keeps completed cells
                    def f(x):
                        return "n/a" if x is None else f"{x:.1f}"
                    print(f"  D={decoys:>3} new={new:>4} cached={cached:>5} conc={conc:>3} | "
                          f"cli_med={f(r['ttft_client_med_ms']):>7} srv_ttft={f(r['ttft_server_ms']):>7} "
                          f"queue={f(r['mean_queue_ms']):>6} prefill={f(r['mean_prefill_ms']):>6} "
                          f"| SRV_frontend={f(r['server_frontend_ms']):>7} "
                          f"cli_frontend={f(r['client_frontend_ms']):>7}", flush=True)
    print(f"wrote {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="/data48/kevinlau/models/Llama-3.1-8B-Instruct")
    ap.add_argument("--served-model-name", default="llama")
    ap.add_argument("--port", type=int, default=8792)
    ap.add_argument("--api-key", default="test")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--news", default="128,2048", help="fresh-token counts (isolate new dependence)")
    ap.add_argument("--cacheds", default="0,8000", help="shared primed-prefix token counts (cached dependence)")
    ap.add_argument("--concs", default="1,5,10,20", help="burst concurrencies (overlap curve)")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--decoys", default="0", help="streaming decoy counts; sweep per value (0 = idle loop)")
    ap.add_argument("--decoy-prompt-tokens", type=int, default=128)
    ap.add_argument("--decoy-max-tokens", type=int, default=2048)
    ap.add_argument("--decoy-daemon", type=int, default=0, help="[internal] run as decoy daemon")
    ap.add_argument("--out", default="profile_data/results/serving_herd_scaling_H100.csv")
    ap.add_argument("--no-launch", action="store_true")
    ap.add_argument("--server-log", default="vllm_server_herd_scaling.log")
    a = ap.parse_args()

    base_url = f"http://127.0.0.1:{a.port}"
    if a.decoy_daemon > 0:
        asyncio.run(_daemon_main(base_url, a.served_model_name, a.decoy_daemon,
                                 a.decoy_prompt_tokens, a.decoy_max_tokens))
        return

    news = [int(x) for x in a.news.split(",")]
    cacheds = [int(x) for x in a.cacheds.split(",")]
    concs = [int(x) for x in a.concs.split(",")]
    decoy_counts = [int(x) for x in a.decoys.split(",")]
    proc = None
    if not a.no_launch:
        proc = launch_server(a.model, a.port, a.gpu_mem, a.max_model_len, a.api_key, a.server_log)
    try:
        if not wait_health(a.port):
            print(f"SERVER DID NOT BECOME HEALTHY -- see {a.server_log}", flush=True)
            sys.exit(1)
        print("server healthy; starting herd-scaling sweep", flush=True)
        rows = []
        for d in decoy_counts:
            dproc = None
            if d > 0:
                dproc = subprocess.Popen(
                    [sys.executable, __file__, "--decoy-daemon", str(d),
                     "--port", str(a.port), "--served-model-name", a.served_model_name,
                     "--decoy-prompt-tokens", str(a.decoy_prompt_tokens),
                     "--decoy-max-tokens", str(a.decoy_max_tokens), "--no-launch"],
                    stdout=subprocess.PIPE, text=True, bufsize=1)
                print(f"waiting for {d} decoys to reach steady streaming ...", flush=True)
                deadline = time.time() + 300
                ready = False
                while time.time() < deadline:
                    line = dproc.stdout.readline()
                    if not line:
                        break  # daemon died
                    if "READY" in line:
                        ready = True
                        break
                if not ready:
                    print(f"DECOY DAEMON FAILED (d={d}); skipping this block", flush=True)
                    dproc.terminate()
                    continue
                time.sleep(3)  # let decoy TTFT/prefill observations land before scrapes
            try:
                asyncio.run(run(base_url, a.served_model_name, news, cacheds, concs,
                                a.trials, a.out, rows=rows, decoys=d))
            finally:
                if dproc is not None:
                    dproc.terminate()
                    try:
                        dproc.wait(timeout=15)
                    except Exception:
                        dproc.kill()
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except Exception:
                proc.kill()


if __name__ == "__main__":
    main()
