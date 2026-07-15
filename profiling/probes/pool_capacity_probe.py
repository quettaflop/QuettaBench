#!/usr/bin/env python3
# profiling/probes/pool_capacity_probe.py
"""Effective KV-pool capacity under APC: prime K distinct T-token prompts (K*T well
past the nominal pool), then probe newest->oldest. The hit/miss boundary (server-side
prefill-span delta: ~T-token re-prefill on a miss, ~one block on a hit) is the USABLE
pool in tokens -- watermark/rounding/transients included. Also prints the server's own
startup KV-block lines for the static number.

Run (self-launches the server):
  CUDA_VISIBLE_DEVICES=3 python pool_capacity_probe.py --port 8794 \
      --tokens 4096 --count 120 --out pool_capacity_H100.txt
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
for _p in [e for e in sys.path if e == _SCRIPT_DIR]:
    sys.path.remove(_p)

import aiohttp  # noqa: E402

_VOCAB = ("the of and to in a is that for it as was with on be at by this had not are but from or "
          "have an they which one you were her all she there would their we him been has when who "
          "will more no if out so up said what its about into than them can only other new some").split()


def prompt_for(i: int, tokens: int) -> str:
    rng = random.Random(31337 + i)
    return f"pool probe {i}: " + " ".join(rng.choice(_VOCAB) for _ in range(int(tokens * 0.96)))


async def prefill_span(session, base_url):
    try:
        async with session.get(base_url + "/metrics") as resp:
            for line in (await resp.text()).splitlines():
                if line.startswith("vllm:request_prefill_time_seconds_sum"):
                    return float(line.rsplit(" ", 1)[1])
    except aiohttp.ClientError:
        pass
    return None


async def ask(session, url, model, content):
    payload = {"model": model, "messages": [{"role": "user", "content": content}],
               "max_tokens": 1, "temperature": 0.0}
    headers = {"Authorization": "Bearer test", "Content-Type": "application/json"}
    async with session.post(url, json=payload, headers=headers) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
        await resp.json()


async def run(base_url, model, tokens, count, out_path):
    chat_url = base_url + "/v1/chat/completions"
    conn = aiohttp.TCPConnector(limit=0)
    lines = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600), connector=conn) as s:
        t0 = time.time()
        for i in range(count):
            await ask(s, chat_url, model, prompt_for(i, tokens))
        lines.append(f"primed {count} x {tokens} tok in {time.time()-t0:.0f}s "
                     f"({count*tokens} tok total)")
        # probe newest -> oldest; hit = tiny prefill delta, miss = ~full re-prefill
        boundary = None
        for i in range(count - 1, -1, -1):
            before = await prefill_span(s, base_url)
            await ask(s, chat_url, model, prompt_for(i, tokens))
            after = await prefill_span(s, base_url)
            span_ms = (after - before) * 1000.0 if None not in (before, after) else -1.0
            miss = span_ms > tokens * 0.02  # ~0.02 ms/tok prefill; hit is ~1 block
            lines.append(f"probe {i}: prefill={span_ms:.1f}ms {'MISS' if miss else 'hit'}")
            if miss and boundary is None:
                boundary = i
                break
        if boundary is None:
            lines.append(f"NO MISS FOUND: pool >= {count*tokens} tokens; raise --count")
        else:
            resident = (count - 1 - boundary) * tokens
            lines.append(f"first miss at prompt {boundary} (newest->oldest) -> "
                         f"effective pool ~= {resident} tokens "
                         f"({resident/16:.0f} blocks; pinned 27250 = 436000 tok)")
    out = Path(out_path)
    out.write_text("\n".join(lines) + "\n")
    for ln in lines[-3:]:
        print(ln, flush=True)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data48/kevinlau/models/Llama-3.1-8B-Instruct")
    ap.add_argument("--served-model-name", default="llama")
    ap.add_argument("--port", type=int, default=8794)
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--count", type=int, default=120)
    ap.add_argument("--out", default="pool_capacity_H100.txt")
    ap.add_argument("--server-log", default="vllm_pool_probe.log")
    a = ap.parse_args()

    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", a.model, "--served-model-name", a.served_model_name,
           "--host", "127.0.0.1", "--port", str(a.port),
           "--dtype", "bfloat16", "--gpu-memory-utilization", "0.9",
           "--max-model-len", "32768", "--tensor-parallel-size", "1",
           "--enable-prefix-caching", "--enable-chunked-prefill",
           "--api-key", "test", "--prefix-caching-hash-algo", "sha256",
           "--no-enable-log-requests"]
    proc = subprocess.Popen(cmd, stdout=open(a.server_log, "w"), stderr=subprocess.STDOUT)
    try:
        if not wait_health(a.port):
            print(f"SERVER NOT HEALTHY -- see {a.server_log}")
            sys.exit(1)
        for line in Path(a.server_log).read_text().splitlines():
            if re.search(r"GPU KV cache size|# GPU blocks|kv cache", line, re.I):
                print("SERVER:", line.split("]")[-1].strip(), flush=True)
        asyncio.run(run(f"http://127.0.0.1:{a.port}", a.served_model_name,
                        a.tokens, a.count, a.out))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
