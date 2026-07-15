#!/usr/bin/env python3
# profiling/probes/serving_decode_grid.py
"""SERVING-context decode grid: live-server (B, T) sweep of steady-state mid-stream ITL.

WHY (the L8 tp4 paradox, see profiling/docs/defit_log_entries/L8-h100x4.md and L11-h100multi.md):
the ISOLATED kernel lattice (cuda_events/decode_steps.py) measures the per-step wall of a bare
engine stepping a synthetic batch — at tp4 that wall sits ABOVE what real serving achieves at
the same (B, ctx) (chat c320: 6.56 ms served vs 11.5 ms isolated B=320 step), because the
isolated per-step fixed costs (4-way NCCL launch + host walls; S12 walls-vs-trace +12–35%)
overlap/pipeline under continuous serving. The simulator prices ``decode_step_ms(b_eff, ctx)``
as the per-step SERVING cost (kernel_tpot's TPOT floor AND ttft_queue_sim's ``_price_step``
drain rate), so the right measurement is the live per-request MID-STREAM ITL at steady decode.

METHOD (pre-registered in L11-h100multi.md):
  * self-launch a vLLM OpenAI api_server (``launch_server``/``wait_health`` reused from
    serving_stage_split.py; ``-m vllm.entrypoints.openai.api_server`` -> the L8 flash_attn.py
    sys.path-shadowing failure mode does not apply, but still run from a clean scripts copy);
    engine flags = the L8 lattice / GT-bench class config:
        --no-enable-prefix-caching --max-num-seqs 320 --max-num-batched-tokens 8192
        --max-model-len 25600 --gpu-memory-utilization 0.90 --dtype bfloat16 --tp N
  * per cell (B, T): fire B concurrent ``/v1/completions`` SSE streams. Each request sends a
    UNIQUE seeded random token-ID prompt (exact prompt_tokens, no tokenizer variance) with
    ``max_tokens=osl, temperature=0, stream=true, ignore_eos=true`` (verified at warmup by
    counting streamed events; fallback ``min_tokens=osl``). ``prompt = T - osl//2`` and
    ``osl = 384 + ceil(B*prompt/8192)`` (integer fixed point) so that (a) the steady window
    survives the chunked admission ramp (~B*prompt/8192 steps) and (b) the median in-window
    context sits at the nominal T.
  * a perf_counter timestamp is recorded per SSE content delta; for B >= --shard-threshold the
    client shards across --n-shards OS processes (per-shard wall-clock anchor for cross-shard
    alignment) and each shard samples its asyncio loop-lag (50 ms sleeper overshoot) — p99
    lag > 2 ms flags the cell ``check`` (an overloaded client loop fakes ITL).
  * steady window = [max_i(first_token_ts), min_i(last_token_ts)] (every request decoding, no
    prefill left); per-request p50 ITL over deltas inside the window (>= 64 in-window deltas
    per request required, else ``check``); cell decode_step_ms = median over the B per-request
    p50s; effective context_len = prompt + median in-window progress.

OUTPUT: an APPEND-ONLY raw per-request JSONL.gz (one line per request: wall-anchored event
timestamps) + an operator summary CSV. The AUTHORITATIVE grid CSV is produced by the
deterministic builder ``profiling/process/build_serving_decode_grid.py`` from the raw JSONL
(same ``summarize_cell`` below — the builder imports this module by path), with columns
    batch_size, context_len, decode_step_ms, validation_status,
    nominal_T, prompt_tokens, osl, n_samples, steady_window_s, ...diagnostics
of which ``simulator.kernel_step_cost.load_grid`` reads only the first four (the grid consumer
is agnostic to how cells were measured).

Example (tp4, h100 GPUs 4-7, per profiling/docs/h100_setup.md):
    CUDA_VISIBLE_DEVICES=4,5,6,7 TMPDIR=/data48/kevinlau/tmp \
      XDG_CACHE_HOME=/data48/kevinlau/tmp/.cache \
      ~/miniconda3/envs/vllm/bin/python serving_decode_grid.py \
      --model /data48/kevinlau/models/Llama-3.1-8B-Instruct --tensor-parallel-size 4 \
      --port 8791 --out-raw serving_decode_grid_H100x4_<date>.jsonl.gz \
      --out-summary serving_decode_grid_H100x4_<date>_summary.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import json
import math
import multiprocessing as mp
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

# aiohttp is needed only at MEASUREMENT time (on the GPU host). The local builder/test import
# of this module (for summarize_cell / the lattice) must work without it.


# ------------------------------------------------------------------------------------------------
# Pre-registered lattice (L11-h100multi.md). prompt/osl are an integer fixed point of
#   prompt = T - osl//2 ;  osl = 384 + ceil(B*prompt/CHUNK_BUDGET)
# starting from osl0 = 384 (deterministic; converges in < 10 iterations on all cells).
# ------------------------------------------------------------------------------------------------
CHUNK_BUDGET = 8192          # vLLM --max-num-batched-tokens (the admission ramp rate)
STEADY_MIN_STEPS = 384       # decode steps shared by ALL requests after the ramp
KV_POOL_TOKENS_TP4 = 2_116_352   # live tp4 pool (L8-verified at these engine flags)
CAP_FRAC = 0.95

# B -> nominal T list. The 26 pre-registered cells (16 + 4 + 3 + 3); the B=256/320 rows are the
# deterministic completion of the directive's truncated rows under its own cap rule
# B*(T+osl/2) <= 0.95*pool over its own T menu (see L11-h100multi.md). (160, 12288) is named by
# the directive ("~94.6% of pool"; strict fixed-point = 95.2%) and kept by name.
LATTICE_TP4: dict[int, tuple[int, ...]] = {
    1:   (512, 2048, 8192, 16384),
    8:   (512, 2048, 8192, 16384),
    32:  (512, 2048, 8192, 16384),
    80:  (512, 2048, 8192, 16384),
    160: (512, 2048, 8192, 12288),
    256: (512, 2048, 6144),
    320: (512, 2048, 4096),
}

T_MENU = (512, 2048, 4096, 6144, 8192, 12288, 16384)


def solve_cell(b: int, t: int, chunk_budget: int = CHUNK_BUDGET,
               steady: int = STEADY_MIN_STEPS) -> tuple[int, int]:
    """(prompt_tokens, osl) integer fixed point for cell (B, T). Deterministic."""
    osl = steady
    for _ in range(50):
        prompt = t - osl // 2
        new_osl = steady + math.ceil(b * prompt / chunk_budget)
        if new_osl == osl:
            return prompt, osl
        osl = new_osl
    return t - osl // 2, osl  # pragma: no cover (always converges in practice)


def lattice_cells(lattice: dict[int, tuple[int, ...]] = LATTICE_TP4,
                  pool_tokens: int = KV_POOL_TOKENS_TP4) -> list[dict]:
    """The pre-registered cell list with solved (prompt, osl) + the cap check."""
    cap = CAP_FRAC * pool_tokens
    cells = []
    for b in sorted(lattice):
        for t in lattice[b]:
            prompt, osl = solve_cell(b, t)
            kv_final = b * (prompt + osl)        # == B*(T + osl/2) up to integer rounding
            cells.append({
                "batch_size": b, "nominal_T": t, "prompt_tokens": prompt, "osl": osl,
                "kv_final_tokens": kv_final, "kv_frac_of_pool": round(kv_final / pool_tokens, 4),
                "exceeds_cap": kv_final > cap,
            })
    return cells


def build_lattice_for_pool(pool_tokens: int) -> dict[int, tuple[int, ...]]:
    """Generic lattice for a DIFFERENT pool (the conditional tp2 leg): the same B rows /
    T menu, keeping per row the directive's base Ts plus the largest menu Ts that satisfy
    the cap (mirrors how the tp4 rows were derived). Deterministic in pool_tokens."""
    cap = CAP_FRAC * pool_tokens
    out: dict[int, tuple[int, ...]] = {}
    for b in (1, 8, 32, 80, 160, 256, 320):
        ts = []
        for t in T_MENU:
            prompt, osl = solve_cell(b, t)
            if b * (prompt + osl) <= cap:
                ts.append(t)
        if ts:
            out[b] = tuple(ts)
    return out


# ------------------------------------------------------------------------------------------------
# Deterministic per-request prompts: unique seeded random token IDs (exact token counts).
# ------------------------------------------------------------------------------------------------
TOKEN_ID_LO, TOKEN_ID_HI = 1000, 30000   # plain-text band of the Llama-3.1 vocab


def prompt_token_ids(cell_b: int, nominal_t: int, req_idx: int, n_tokens: int) -> list[int]:
    # str seed: random.Random rejects tuples on Python >= 3.11
    rng = random.Random(f"{cell_b}:{nominal_t}:{req_idx}:serving_decode_grid")
    return [rng.randrange(TOKEN_ID_LO, TOKEN_ID_HI) for _ in range(n_tokens)]


# ------------------------------------------------------------------------------------------------
# Cell summary — THE pre-registered post-processing (shared with the builder, which imports it).
# Records: [{"req", "shard", "prompt_tokens", "osl", "t_first_wall", "deltas_ms": [...],
#            "lag_p99_ms"}]  (t_first_wall = wall-anchored first-token ts; event k>0 arrives at
#            t_first_wall + sum(deltas_ms[:k])/1e3).
# ------------------------------------------------------------------------------------------------
def _p99(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(0.99 * (len(s) - 1) + 0.999999))]


def summarize_cell(records: list[dict], min_samples: int = 64,
                   lag_limit_ms: float = 2.0) -> dict:
    """Deterministic cell row from raw per-request records (see module docstring)."""
    if not records:
        raise ValueError("empty cell")
    b = len(records)
    firsts, lasts, ev_walls = [], [], []
    for r in records:
        ts = [r["t_first_wall"]]
        for d in r["deltas_ms"]:
            ts.append(ts[-1] + d / 1e3)
        ev_walls.append(ts)
        firsts.append(ts[0])
        lasts.append(ts[-1])
    w_lo, w_hi = max(firsts), min(lasts)
    steady_window_s = w_hi - w_lo

    per_req_p50, per_req_progress, per_req_n = [], [], []
    coalesced = total_deltas = 0
    for ts in ev_walls:
        # delta k (between event k and k+1) is "in window" iff both endpoints are inside.
        din = [(ts[k + 1] - ts[k]) * 1e3 for k in range(len(ts) - 1)
               if ts[k] >= w_lo and ts[k + 1] <= w_hi]
        idx_in = [k + 1 for k in range(len(ts)) if w_lo <= ts[k] <= w_hi]  # token index (1-based)
        per_req_n.append(len(din))
        if din:
            per_req_p50.append(st.median(din))
            coalesced += sum(1 for d in din if d < 0.5)
            total_deltas += len(din)
        if idx_in:
            per_req_progress.append(st.median(idx_in))

    lag_p99 = max((float(r.get("lag_p99_ms") or 0.0) for r in records), default=0.0)
    status = "ok"
    if steady_window_s <= 0 or len(per_req_p50) < b or min(per_req_n) < min_samples:
        status = "check"
    if lag_p99 > lag_limit_ms:
        status = "check"

    prompt = int(records[0]["prompt_tokens"])
    progress = st.median(per_req_progress) if per_req_progress else 0.0
    return {
        "batch_size": b,
        "context_len": int(round(prompt + progress)),       # EFFECTIVE context (the grid axis)
        "decode_step_ms": round(st.median(per_req_p50), 4) if per_req_p50 else 0.0,
        "validation_status": status,
        "nominal_T": int(records[0].get("nominal_T", 0)),
        "prompt_tokens": prompt,
        "osl": int(records[0]["osl"]),
        "n_samples": min(per_req_n) if per_req_n else 0,    # MIN in-window deltas per request
        "steady_window_s": round(steady_window_s, 3),
        "median_inwindow_progress": round(progress, 1),
        "p99_loop_lag_ms": round(lag_p99, 3),
        "coalesce_frac": round(coalesced / total_deltas, 4) if total_deltas else 0.0,
        "itl_p50_spread_ms": round(max(per_req_p50) - min(per_req_p50), 4) if per_req_p50 else 0.0,
    }


SUMMARY_FIELDS = [
    "batch_size", "context_len", "decode_step_ms", "validation_status",
    "nominal_T", "prompt_tokens", "osl", "n_samples", "steady_window_s",
    "median_inwindow_progress", "p99_loop_lag_ms", "coalesce_frac", "itl_p50_spread_ms",
]


# ------------------------------------------------------------------------------------------------
# Async measurement worker (one OS process per shard; B < shard threshold -> a single shard).
# ------------------------------------------------------------------------------------------------
async def _stream_once(session, url: str, headers: dict, payload: dict) -> list[float]:
    """POST a streaming completion; perf_counter per SSE content delta (no per-event JSON parse:
    every completions data chunk is a text delta; the [DONE] sentinel ends the stream)."""
    ts: list[float] = []
    async with session.post(url, json=payload, headers=headers) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:300]}")
        async for raw in resp.content:
            if not raw.startswith(b"data:"):
                continue
            if raw[5:12].strip().startswith(b"[DONE]"):
                break
            ts.append(time.perf_counter())
    return ts


async def _loop_lag_sampler(stop: "asyncio.Event", out: list[float]) -> None:
    loop = asyncio.get_running_loop()
    while not stop.is_set():
        t0 = loop.time()
        await asyncio.sleep(0.05)
        out.append((loop.time() - t0 - 0.05) * 1e3)


async def _run_shard(req_specs: list[dict], port: int, api_key: str, model_name: str,
                     eos_mode: str, barrier) -> list[dict]:
    import aiohttp  # lazy: GPU-host runtime dependency only
    url = f"http://127.0.0.1:{port}/v1/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    timeout = aiohttp.ClientTimeout(total=3600, sock_read=600)
    connector = aiohttp.TCPConnector(limit=0)
    lag: list[float] = []
    stop = asyncio.Event()
    out: list[dict] = []
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # all shards block here until every process is connected & ready -> simultaneous fire
        if barrier is not None:
            await asyncio.get_running_loop().run_in_executor(None, barrier.wait)
        wall_anchor = time.time() - time.perf_counter()   # per-process wall<->perf_counter map
        # GC pauses are the dominant client-side lag source at high event rates (measurement
        # hygiene, run 1 -> run 2 fix): freeze startup garbage and disable collection for the
        # duration of the streams; re-enable after gather.
        import gc
        gc.collect()
        gc.freeze()
        gc.disable()
        sampler = asyncio.create_task(_loop_lag_sampler(stop, lag))

        async def one(spec: dict) -> dict:
            ids = prompt_token_ids(spec["cell_b"], spec["nominal_T"], spec["req"],
                                   spec["prompt_tokens"])
            payload = {"model": model_name, "prompt": ids, "max_tokens": spec["osl"],
                       "temperature": 0.0, "stream": True}
            payload["ignore_eos" if eos_mode == "ignore_eos" else "min_tokens"] = (
                True if eos_mode == "ignore_eos" else spec["osl"])
            t_send = time.perf_counter()
            ts = await _stream_once(session, url, headers, payload)
            if not ts:
                return {**{k: spec[k] for k in ("req", "prompt_tokens", "osl", "nominal_T")},
                        "t_first_wall": None, "deltas_ms": [], "n_events": 0,
                        "ttft_ms": None}
            return {**{k: spec[k] for k in ("req", "prompt_tokens", "osl", "nominal_T")},
                    "t_first_wall": round(ts[0] + wall_anchor, 6),
                    "deltas_ms": [round((ts[k + 1] - ts[k]) * 1e3, 3) for k in range(len(ts) - 1)],
                    "n_events": len(ts),
                    "ttft_ms": round((ts[0] - t_send) * 1e3, 2)}

        results = await asyncio.gather(*(one(s) for s in req_specs))
        stop.set()
        await sampler
        gc.enable()
        gc.collect()
    lag_p99 = round(_p99(lag), 3)
    for r in results:
        r["lag_p99_ms"] = lag_p99
        out.append(r)
    return out


def _shard_entry(req_specs, port, api_key, model_name, eos_mode, barrier, shard_id, out_path,
                 rt: bool = False):
    if rt:
        # SCHED_RR for the SHARD PROCESS ONLY (needs CAP_SYS_NICE / root): the GPU host's own
        # server threads otherwise preempt the client loop for ~2 ms CFS slices, tripping the
        # pre-registered p99 loop-lag flag while leaving the p50 ITL unchanged (run1 1-shard vs
        # run2 8-shard agree <=0.5%). The server keeps its normal policy (separate process).
        try:
            os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(10))
            print(f"shard {shard_id}: SCHED_RR set", flush=True)
        except (PermissionError, OSError) as e:
            print(f"shard {shard_id}: SCHED_RR unavailable ({e}); staying SCHED_OTHER", flush=True)
    res = asyncio.run(_run_shard(req_specs, port, api_key, model_name, eos_mode, barrier))
    for r in res:
        r["shard"] = shard_id
    Path(out_path).write_text(json.dumps(res))


def run_cell(cell: dict, port: int, api_key: str, model_name: str, eos_mode: str,
             shard_threshold: int, n_shards: int, tmp_dir: Path,
             rt_shards: bool = False) -> list[dict]:
    """Fire one (B, T) cell; returns per-request records (wall-anchored)."""
    b = cell["batch_size"]
    specs = [{"cell_b": b, "nominal_T": cell["nominal_T"], "req": i,
              "prompt_tokens": cell["prompt_tokens"], "osl": cell["osl"]}
             for i in range(b)]
    shards = n_shards if b >= shard_threshold else 1
    if shards == 1:
        recs = asyncio.run(_run_shard(specs, port, api_key, model_name, eos_mode, None))
        for r in recs:
            r["shard"] = 0
        return recs
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(shards)
    procs, paths = [], []
    for w in range(shards):
        chunk = specs[w::shards]
        path = tmp_dir / f"shard_{b}_{cell['nominal_T']}_{w}.json"
        p = ctx.Process(target=_shard_entry, args=(chunk, port, api_key, model_name,
                                                   eos_mode, barrier, w, str(path), rt_shards))
        p.start()
        procs.append(p)
        paths.append(path)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"shard process exited {p.exitcode}")
    recs: list[dict] = []
    for path in paths:
        recs.extend(json.loads(path.read_text()))
        path.unlink()
    recs.sort(key=lambda r: r["req"])
    return recs


# ------------------------------------------------------------------------------------------------
# Server lifecycle — launch/wait reused from serving_stage_split.py, flags per L11.
# ------------------------------------------------------------------------------------------------
def wait_health(port, timeout=600):
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


def launch_server(model, port, gpu_mem, max_model_len, api_key, log_path, tp,
                  max_num_seqs, max_num_batched_tokens):
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", model, "--served-model-name", "llama",
           "--host", "127.0.0.1", "--port", str(port),
           "--dtype", "bfloat16", "--gpu-memory-utilization", str(gpu_mem),
           "--max-model-len", str(max_model_len),
           "--tensor-parallel-size", str(tp),
           "--no-enable-prefix-caching",                  # measurement isolation (L11)
           "--max-num-seqs", str(max_num_seqs),
           "--max-num-batched-tokens", str(max_num_batched_tokens),
           "--api-key", api_key,
           "--no-enable-log-requests"]
    print("launching:", " ".join(cmd), flush=True)
    logf = open(log_path, "w")
    return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)


def kv_pool_from_log(log_path: str) -> int | None:
    """Parse vLLM's own 'GPU KV cache size: N tokens' line (kv_cache_utils)."""
    try:
        for line in Path(log_path).read_text(errors="replace").splitlines():
            if "GPU KV cache size:" in line:
                num = line.split("GPU KV cache size:")[1].split("tokens")[0]
                return int(num.strip().replace(",", ""))
    except Exception:
        pass
    return None


# ------------------------------------------------------------------------------------------------
# Warmup + ignore_eos verification (one unrecorded pass; L8/L3 first-cell outlier hygiene).
# ------------------------------------------------------------------------------------------------
def warmup_and_verify_eos(port: int, api_key: str, model_name: str) -> str:
    """8 short streams; returns the eos mode that yields the full requested osl."""
    for mode in ("ignore_eos", "min_tokens"):
        specs = [{"cell_b": 8, "nominal_T": 1024, "req": i, "prompt_tokens": 1024, "osl": 64}
                 for i in range(8)]
        try:
            recs = asyncio.run(_run_shard(specs, port, api_key, model_name, mode, None))
        except Exception as e:  # e.g. 400 on an unsupported field
            print(f"warmup: eos mode {mode!r} failed ({e}); trying fallback", flush=True)
            continue
        n = [r["n_events"] for r in recs]
        print(f"warmup: eos mode {mode!r} -> events per stream {n} (requested 64)", flush=True)
        if min(n) >= 61:  # SSE events may coalesce a few tokens; near-full = honored
            return mode
    raise SystemExit("neither ignore_eos nor min_tokens produced the requested output length")


# ------------------------------------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="/data48/kevinlau/models/Llama-3.1-8B-Instruct")
    ap.add_argument("--served-model-name", default="llama")
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--api-key", default="test")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=25600)
    ap.add_argument("--max-num-seqs", type=int, default=320)
    ap.add_argument("--max-num-batched-tokens", type=int, default=CHUNK_BUDGET)
    ap.add_argument("--tensor-parallel-size", type=int, default=4)
    ap.add_argument("--pool-tokens", type=int, default=None,
                    help="KV pool for lattice feasibility; default: tp4 pre-registered pool "
                         f"({KV_POOL_TOKENS_TP4}); for other tp, pass the live pool and the "
                         "lattice is re-derived by the same cap rule")
    ap.add_argument("--shard-threshold", type=int, default=160)
    ap.add_argument("--n-shards", type=int, default=4)
    ap.add_argument("--rt-shards", action="store_true",
                    help="SCHED_RR for shard processes (needs root/CAP_SYS_NICE); the server "
                         "keeps its normal policy. Counters the host's ~2 ms CFS preemptions "
                         "of the client loop (loop-lag flag hygiene).")
    ap.add_argument("--cells", default=None,
                    help="optional explicit 'B:T,B:T,...' subset (default: the lattice)")
    ap.add_argument("--out-raw", required=True, help="append-only per-request JSONL.gz")
    ap.add_argument("--out-summary", required=True, help="operator cell-summary CSV")
    ap.add_argument("--no-launch", action="store_true")
    ap.add_argument("--server-log", default="vllm_server_decode_grid.log")
    a = ap.parse_args()

    if a.pool_tokens is None and a.tensor_parallel_size == 4:
        lattice = LATTICE_TP4
        pool = KV_POOL_TOKENS_TP4
    else:
        pool = a.pool_tokens or KV_POOL_TOKENS_TP4
        lattice = build_lattice_for_pool(pool)
    cells = lattice_cells(lattice, pool)
    if a.cells:
        wanted = {tuple(int(x) for x in c.split(":")) for c in a.cells.split(",")}
        cells = [c for c in cells if (c["batch_size"], c["nominal_T"]) in wanted]
    over = [c for c in cells if c["exceeds_cap"]
            and not (c["batch_size"] == 160 and c["nominal_T"] == 12288)]
    if over:
        raise SystemExit(f"cells exceed the 0.95 pool cap: {over}")
    print(f"{len(cells)} cells; pool {pool} tokens; cap {CAP_FRAC}", flush=True)

    proc = None
    if not a.no_launch:
        proc = launch_server(a.model, a.port, a.gpu_mem, a.max_model_len, a.api_key,
                             a.server_log, a.tensor_parallel_size,
                             a.max_num_seqs, a.max_num_batched_tokens)
    try:
        if not wait_health(a.port):
            print(f"SERVER DID NOT BECOME HEALTHY -- see {a.server_log}", flush=True)
            sys.exit(1)
        live_pool = kv_pool_from_log(a.server_log)
        print(f"server healthy; live KV pool from log: {live_pool} tokens", flush=True)

        eos_mode = warmup_and_verify_eos(a.port, a.api_key, a.served_model_name)
        meta = {"_meta": True, "tool": "serving_decode_grid.py", "tp": a.tensor_parallel_size,
                "engine_flags": {"gpu_mem": a.gpu_mem, "max_model_len": a.max_model_len,
                                 "max_num_seqs": a.max_num_seqs,
                                 "max_num_batched_tokens": a.max_num_batched_tokens,
                                 "prefix_caching": False},
                "eos_mode": eos_mode, "live_kv_pool_tokens": live_pool,
                "lattice_pool_tokens": pool, "date": time.strftime("%Y-%m-%d"),
                "n_cells": len(cells)}

        tmp_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "serving_decode_grid_shards"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        out_raw = Path(a.out_raw)
        out_raw.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        with gzip.open(out_raw, "at") as raw_f:
            raw_f.write(json.dumps(meta) + "\n")
            for c in cells:
                t0 = time.time()
                recs = run_cell(c, a.port, a.api_key, a.served_model_name, eos_mode,
                                a.shard_threshold, a.n_shards, tmp_dir,
                                rt_shards=a.rt_shards)
                for r in recs:
                    raw_f.write(json.dumps({"cell": [c["batch_size"], c["nominal_T"]], **r}) + "\n")
                raw_f.flush()
                row = summarize_cell([r for r in recs if r["t_first_wall"] is not None])
                rows.append(row)
                print(f"  B={c['batch_size']:>3} T={c['nominal_T']:>5}  "
                      f"itl_p50={row['decode_step_ms']:>8.3f} ms  ctx_eff={row['context_len']:>5}  "
                      f"n>={row['n_samples']:>4}  win={row['steady_window_s']:>6.2f}s  "
                      f"lag_p99={row['p99_loop_lag_ms']:>5.2f}  [{row['validation_status']}]  "
                      f"({time.time()-t0:.0f}s)", flush=True)

        out_sum = Path(a.out_summary)
        out_sum.parent.mkdir(parents=True, exist_ok=True)
        with out_sum.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})
        print(f"\nwrote {out_raw} + {out_sum}", flush=True)
        print("authoritative grid CSV: python3 -m profiling.process.build_serving_decode_grid "
              f"--inputs {out_raw.name} ...", flush=True)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except Exception:
                proc.kill()


if __name__ == "__main__":
    main()
