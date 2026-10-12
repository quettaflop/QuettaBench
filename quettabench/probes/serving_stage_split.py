#!/usr/bin/env python3
"""Concurrency-1 prefill TTFT split by (new, cached) tokens from client, server and nsys."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import statistics as st
import sys
import time
from pathlib import Path

import aiohttp

from quettabench.probes.base import launch_vllm, run_recorded

# Words are ~1 Llama token each.
_VOCAB = (
    "the of and to in a is that for it as was with on be at by this had not are but from or "
    "have an they which one you were her all she there would their we him been has when who "
    "will more no if out so up said what its about into than them can only other new some "
    "could time these two may then do first any my now such like our over man me even most "
    "made after also did many before must through back years where much your way well down"
).split()
_RNG = random.Random(0)
_WORDS = [_RNG.choice(_VOCAB) for _ in range(40000)]
_TAIL = random.Random(777)


def cached_prefix(cached: int) -> str:
    return " ".join(_WORDS[: int(cached * 0.96)])


def fresh_tail(new: int) -> str:
    return " ".join(_TAIL.choice(_VOCAB) for _ in range(int(new * 0.96)))


# vLLM v1 exposes per-request timings only as histograms; at concurrency 1 the _sum delta is the value.
_PROM_METRICS = {
    "queue_span_s": "vllm:request_queue_time_seconds",
    # Scheduled to first token: forward wall including Python dispatch.
    "prefill_span_s": "vllm:request_prefill_time_seconds",
    "e2e_s": "vllm:e2e_request_latency_seconds",
    "prompt_tokens": "vllm:request_prompt_tokens",
    "ttft_s": "vllm:time_to_first_token_seconds",
}


def _parse_prom_sums(text: str) -> dict:
    """Labels are ignored: one model means one line per histogram."""
    wanted = {f"{base}_sum": key for key, base in _PROM_METRICS.items()}
    out = {key: None for key in _PROM_METRICS}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
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
    """A missing metric is None."""
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


async def ttft_once(session, url, model, content, request_id=None):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Content-Type": "application/json"}
    if request_id is not None:
        headers["X-Request-Id"] = request_id
    t_send = time.perf_counter()
    async with session.post(url, json=payload, headers=headers) as resp:
        t_status = time.perf_counter()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
        ttft_ms = None
        usage = None
        async for raw in resp.content:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            ds = line[len("data:") :].strip()
            if ds == "[DONE]":
                break
            ch = json.loads(ds)
            cc = ch.get("choices", [])
            if ttft_ms is None and cc and cc[0].get("delta", {}).get("content") is not None:
                ttft_ms = (time.perf_counter() - t_send) * 1000.0
            if ch.get("usage"):
                usage = ch["usage"]
        return {
            "ttft_ms": ttft_ms,
            "client_connect_ms": (t_status - t_send) * 1000.0,
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
        }


async def measure(session, base_url, model, cached, new, trials):
    chat_url = base_url + "/v1/chat/completions"
    prefix = cached_prefix(cached)
    if cached > 0:
        await ttft_once(session, chat_url, model, prefix)

    samples = []
    for i in range(trials + 1):
        tail = fresh_tail(new)
        content = (prefix + " " + tail) if cached > 0 else tail
        before = await scrape_metrics(session, base_url)
        res = await ttft_once(session, chat_url, model, content, request_id=f"sss-{cached}-{new}-{i}")
        after = await scrape_metrics(session, base_url)
        queue_span = _sub(after["queue_span_s"], before["queue_span_s"])
        prefill_span = _sub(after["prefill_span_s"], before["prefill_span_s"])
        e2e_span = _sub(after["e2e_s"], before["e2e_s"])
        ttft_prom = _sub(after["ttft_s"], before["ttft_s"])
        samples.append(
            {
                "ttft_ms": res["ttft_ms"],
                "client_connect_ms": res["client_connect_ms"],
                "prompt_tokens": res["prompt_tokens"],
                "queue_span_ms": None if queue_span is None else queue_span * 1000.0,
                "prefill_span_ms": None if prefill_span is None else prefill_span * 1000.0,
                "e2e_span_ms": None if e2e_span is None else e2e_span * 1000.0,
                "ttft_prom_ms": None if ttft_prom is None else ttft_prom * 1000.0,
            }
        )

    body = samples[1:]

    def med(key):
        vals = [s[key] for s in body if s[key] is not None]
        return st.median(vals) if vals else None

    ttft_ms = med("ttft_ms")
    queue_ms = med("queue_span_ms")
    prefill_ms = med("prefill_span_ms")
    # Host frontend plus IPC: everything in the client wall outside queue and prefill.
    frontend_residual = None
    if ttft_ms is not None and queue_ms is not None and prefill_ms is not None:
        frontend_residual = ttft_ms - queue_ms - prefill_ms

    pt = [s["prompt_tokens"] for s in body if s["prompt_tokens"] is not None]
    prompt_tokens_actual = st.median(pt) if pt else None

    return {
        "new": new,
        "cached": cached,
        "n": trials,
        "prompt_tokens_actual": prompt_tokens_actual,
        "ttft_ms": ttft_ms,
        "client_connect_ms": med("client_connect_ms"),
        "queue_span_ms": queue_ms,
        "prefill_span_ms": prefill_ms,
        "frontend_residual_ms": frontend_residual,
        "e2e_span_ms": med("e2e_span_ms"),
        "ttft_prom_ms": med("ttft_prom_ms"),
    }


def launch_server(model, port, gpu_mem, max_model_len, api_key, hash_algo, run_dir, tp=1):
    # sha256 vs builtin hashing isolates the block-hash cost inside the frontend residual.
    flags = {
        "served_model_name": "llama",
        "host": "127.0.0.1",
        "api_key": api_key,
        # Stats must stay on (no disable_log_stats) or /metrics is empty.
        "no_enable_log_requests": True,
    }
    if hash_algo:
        flags["prefix_caching_hash_algo"] = hash_algo
    return launch_vllm(
        run_dir,
        model,
        port=port,
        tp=tp,
        gpu_util=gpu_mem,
        max_model_len=max_model_len,
        prefix_caching=True,
        vllm_flags=flags,
        timeout_s=420,
    )


# Printed, never run, so the profiled wall stays apart from the client and /metrics lanes.
_WORKER_EXT_HINT = (
    "Lane B needs a worker extension (vllm serve --worker-extension-cls) that wraps GPUModelRunner.execute_model "
    "in an NVTX range named vllm_prefill_step between two CUDA events, reading elapsed_time off the hot path."
)


def emit_nsys_cmd(model, port, gpu_mem, max_model_len, out_sqlite):
    """The key comes from VLLM_API_KEY, so it is never printed."""
    py = sys.executable
    nsys_cmd = (
        f"nsys profile -t cuda,nvtx,cublas --cuda-graph-trace=node \\\n"
        f"  --capture-range=cudaProfilerApi --capture-range-end=stop \\\n"
        f"  -o {out_sqlite} -f true -- \\\n"
        f"  {py} -m vllm.entrypoints.openai.api_server \\\n"
        f"    --model {model} --served-model-name llama --host 127.0.0.1 --port {port} \\\n"
        f"    --dtype bfloat16 --gpu-memory-utilization {gpu_mem} "
        f"--max-model-len {max_model_len} \\\n"
        f"    --enable-prefix-caching --enable-chunked-prefill \\\n"
        f"    --no-enable-log-requests --worker-extension-cls my_ext.NvtxForwardTimer"
    )
    print("\n" + "=" * 96)
    print("LANE B (DEVICE split) -- run this SEPARATELY (nsys-wrapped server), then drive it with")
    print("this same script (--no-launch, pointing at the nsys server). Export VLLM_API_KEY")
    print("first, set to the --api-key you pass this script.")
    print("restricting Sigma(end-start) to the per-step prefill NVTX range:")
    print("=" * 96)
    print(nsys_cmd)
    print(_WORKER_EXT_HINT)
    print("After nsys export to sqlite, device_kernel_ms(window) = SUM(k.end-k.start) over")
    print("CUPTI_ACTIVITY_KIND_KERNEL rows whose [start,end] fall inside the 'vllm_prefill_step'")
    print("NVTX range (NVTX_EVENTS table). framework_dispatch_ms = prefill_span_ms - device_kernel_ms.")
    print("=" * 96 + "\n")


_CSV_FIELDS = [
    "new",
    "cached",
    "n",
    "prompt_tokens_actual",
    "ttft_ms",
    "client_connect_ms",
    "queue_span_ms",
    "prefill_span_ms",
    "frontend_residual_ms",
    "e2e_span_ms",
    "ttft_prom_ms",
]


async def run_sweep(base_url, model, news, cacheds, trials, out_path, api_key):
    rows = []
    timeout = aiohttp.ClientTimeout(total=300)
    auth = {"Authorization": f"Bearer {api_key}"}
    async with aiohttp.ClientSession(timeout=timeout, headers=auth) as s:
        probe = await scrape_metrics(s, base_url)
        present = [k for k, v in probe.items() if v is not None]
        print(f"/metrics reachable; histograms present at start: {present}", flush=True)
        if "prefill_span_s" not in present:
            print(
                "[warn] vllm:request_prefill_time_seconds_sum NOT found -- LANE A prefill_span "
                "will be blank. VERIFY the metric name / that stats are enabled on 0.19.0.",
                flush=True,
            )
        for cached in cacheds:
            for new in news:
                r = await measure(s, base_url, model, cached, new, trials)
                rows.append(r)
                pt = r["prompt_tokens_actual"]
                print(
                    f"  new={new:>5} cached={cached:>6}  ttft={_f(r['ttft_ms']):>8}  "
                    f"queue={_f(r['queue_span_ms']):>7}  prefill={_f(r['prefill_span_ms']):>8}  "
                    f"frontend={_f(r['frontend_residual_ms']):>8}  "
                    f"pt={pt if pt is not None else '?'}",
                    flush=True,
                )

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


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default=None, help="model dir (Llama-3.1-8B-Instruct); needed unless --no-launch")
    ap.add_argument(
        "--served-model-name",
        default="llama",
        help="model name sent in the OpenAI payload (matches --served-model-name)",
    )
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--api-key", default="test")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="tp for the launched server (run tp1 AND tp2 with this same script "
        "for the tp_comm pair; needs tp GPUs in CUDA_VISIBLE_DEVICES)",
    )
    ap.add_argument(
        "--hash",
        default="sha256",
        choices=["sha256", "builtin", ""],
        help="prefix_caching_hash_algo for the launched server (block-hash attribution)",
    )
    ap.add_argument("--news", default="8,128,512,1024,2048", help="fresh-token counts to sweep")
    ap.add_argument("--cacheds", default="0,2000,8000,16000", help="cached-prefix token counts")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument(
        "--out",
        required=True,
        help="the stage-split CSV; its dir also gets run.json "
        "(e.g. results/probes/<gpu>/serving_stage_split/<date>/serving_stage_split_<gpu>.csv)",
    )
    ap.add_argument(
        "--no-launch", action="store_true", help="server already running (e.g. the nsys-wrapped LANE B server)"
    )
    ap.add_argument(
        "--emit-nsys-cmd", action="store_true", help="print the LANE B nsys wrapper command + worker hook and exit"
    )
    a = ap.parse_args(argv)
    if a.model is None and (a.emit_nsys_cmd or not a.no_launch):
        ap.error("--model is required to launch the server or print its nsys command")

    base_url = f"http://127.0.0.1:{a.port}"

    if a.emit_nsys_cmd:
        emit_nsys_cmd(a.model, a.port, a.gpu_mem, a.max_model_len, out_sqlite="serving_stage_split_nsys")
        return False

    news = [int(x) for x in a.news.split(",")]
    cacheds = [int(x) for x in a.cacheds.split(",")]

    server = None
    if not a.no_launch:
        server = launch_server(
            a.model,
            a.port,
            a.gpu_mem,
            a.max_model_len,
            a.api_key,
            a.hash or None,
            Path(a.out).parent / "server",
            tp=a.tensor_parallel_size,
        )
    try:
        print("server healthy; starting c1 stage-split sweep", flush=True)
        asyncio.run(run_sweep(base_url, a.served_model_name, news, cacheds, a.trials, a.out, api_key=a.api_key))
        emit_nsys_cmd(a.model, a.port, a.gpu_mem, a.max_model_len, out_sqlite="serving_stage_split_nsys")
        print("Next: parse the nsys sqlite (CUPTI kernels inside the vllm_prefill_step NVTX range).", flush=True)
    finally:
        if server is not None:
            server.stop()


if __name__ == "__main__":
    run_recorded(main, "serving_stage_split", file_flag="--out")
