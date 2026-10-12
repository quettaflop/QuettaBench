#!/usr/bin/env python3
"""Live-server (batch, context) grid of steady-state mid-stream ITL: launches vLLM, fires B concurrent completion streams per cell and records SSE timestamps to --out-raw.

`just export --only serving_decode_grid` builds the grid CSV from the raw runs with summarize_cell from here.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import gc
import gzip
import json
import math
import multiprocessing as mp
import os
import random
import statistics as st
import tempfile
import time
from pathlib import Path

from quettabench.probes.base import launch_vllm, run_recorded

# aiohttp exists only on the GPU host; summarize_cell and the lattice must import without it.


# prompt/osl are the integer fixed point of prompt = T - osl//2, osl = 384 + ceil(B*prompt/CHUNK_BUDGET), from osl0 = 384.
CHUNK_BUDGET = 8192  # vLLM --max-num-batched-tokens, the admission ramp rate
STEADY_MIN_STEPS = 384  # decode steps all requests share after the ramp
KV_POOL_TOKENS_TP4 = 2_116_352  # the live tp4 pool at these engine flags
CAP_FRAC = 0.95  # B*(T + osl/2) may fill at most this fraction of the KV pool

# B -> nominal T: 26 fixed cells, each under B*(T+osl/2) <= 0.95*pool except (160, 12288) at 95.2%.
LATTICE_TP4: dict[int, tuple[int, ...]] = {
    1: (512, 2048, 8192, 16384),
    8: (512, 2048, 8192, 16384),
    32: (512, 2048, 8192, 16384),
    80: (512, 2048, 8192, 16384),
    160: (512, 2048, 8192, 12288),
    256: (512, 2048, 6144),
    320: (512, 2048, 4096),
}

T_MENU = (512, 2048, 4096, 6144, 8192, 12288, 16384)


def solve_cell(b: int, t: int, chunk_budget: int = CHUNK_BUDGET, steady: int = STEADY_MIN_STEPS) -> tuple[int, int]:
    """(prompt_tokens, osl) integer fixed point for cell (B, T)."""
    osl = steady
    for _ in range(50):
        prompt = t - osl // 2
        new_osl = steady + math.ceil(b * prompt / chunk_budget)
        if new_osl == osl:
            return prompt, osl
        osl = new_osl
    return t - osl // 2, osl  # pragma: no cover


def lattice_cells(
    lattice: dict[int, tuple[int, ...]] = LATTICE_TP4, pool_tokens: int = KV_POOL_TOKENS_TP4
) -> list[dict]:
    """The fixed cell list with solved (prompt, osl) and the cap check."""
    cap = CAP_FRAC * pool_tokens
    cells = []
    for b in sorted(lattice):
        for t in lattice[b]:
            prompt, osl = solve_cell(b, t)
            kv_final = b * (prompt + osl)  # B*(T + osl/2) up to rounding
            cells.append(
                {
                    "batch_size": b,
                    "nominal_T": t,
                    "prompt_tokens": prompt,
                    "osl": osl,
                    "kv_final_tokens": kv_final,
                    "kv_frac_of_pool": round(kv_final / pool_tokens, 4),
                    "exceeds_cap": kv_final > cap,
                }
            )
    return cells


def build_lattice_for_pool(pool_tokens: int) -> dict[int, tuple[int, ...]]:
    """Lattice for another pool (the tp2 leg): the same B rows and T menu, keeping per row the base Ts plus the largest menu Ts under the cap, as the tp4 rows were derived."""
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


TOKEN_ID_LO, TOKEN_ID_HI = 1000, 30000  # plain-text band of the Llama-3.1 vocab


def prompt_token_ids(cell_b: int, nominal_t: int, req_idx: int, n_tokens: int) -> list[int]:
    # random.Random rejects tuple seeds on Python >= 3.11.
    rng = random.Random(f"{cell_b}:{nominal_t}:{req_idx}:serving_decode_grid")
    return [rng.randrange(TOKEN_ID_LO, TOKEN_ID_HI) for _ in range(n_tokens)]


# A record's event k>0 arrives at t_first_wall + sum(deltas_ms[:k])/1e3; the export builder imports summarize_cell.
def _p99(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(0.99 * (len(s) - 1) + 0.999999))]


def summarize_cell(records: list[dict], min_samples: int = 64, lag_limit_ms: float = 2.0) -> dict:
    """Cell row from raw per-request records."""
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
        # Delta k (event k to k+1) is in the window only if both ends are.
        din = [(ts[k + 1] - ts[k]) * 1e3 for k in range(len(ts) - 1) if ts[k] >= w_lo and ts[k + 1] <= w_hi]
        idx_in = [k + 1 for k in range(len(ts)) if w_lo <= ts[k] <= w_hi]  # 1-based token index
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
        "context_len": int(round(prompt + progress)),  # effective context, the grid axis
        "decode_step_ms": round(st.median(per_req_p50), 4) if per_req_p50 else 0.0,
        "validation_status": status,
        "nominal_T": int(records[0].get("nominal_T", 0)),
        "prompt_tokens": prompt,
        "osl": int(records[0]["osl"]),
        "n_samples": min(per_req_n) if per_req_n else 0,  # min in-window deltas per request
        "steady_window_s": round(steady_window_s, 3),
        "median_inwindow_progress": round(progress, 1),
        "p99_loop_lag_ms": round(lag_p99, 3),
        "coalesce_frac": round(coalesced / total_deltas, 4) if total_deltas else 0.0,
        "itl_p50_spread_ms": round(max(per_req_p50) - min(per_req_p50), 4) if per_req_p50 else 0.0,
    }


SUMMARY_FIELDS = [
    "batch_size",
    "context_len",
    "decode_step_ms",
    "validation_status",
    "nominal_T",
    "prompt_tokens",
    "osl",
    "n_samples",
    "steady_window_s",
    "median_inwindow_progress",
    "p99_loop_lag_ms",
    "coalesce_frac",
    "itl_p50_spread_ms",
]


async def _stream_once(session, url: str, headers: dict, payload: dict) -> list[float]:
    """POST a streaming completion and record perf_counter per SSE delta; every completions chunk is a text delta, so none is parsed."""
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


async def _loop_lag_sampler(stop: asyncio.Event, out: list[float]) -> None:
    loop = asyncio.get_running_loop()
    while not stop.is_set():
        t0 = loop.time()
        await asyncio.sleep(0.05)
        out.append((loop.time() - t0 - 0.05) * 1e3)


async def _run_shard(
    req_specs: list[dict], port: int, api_key: str, model_name: str, eos_mode: str, barrier
) -> list[dict]:
    import aiohttp  # GPU host only

    url = f"http://127.0.0.1:{port}/v1/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    timeout = aiohttp.ClientTimeout(total=3600, sock_read=600)
    connector = aiohttp.TCPConnector(limit=0)
    lag: list[float] = []
    stop = asyncio.Event()
    out: list[dict] = []
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Every shard waits here until all are connected, so they fire together.
        if barrier is not None:
            await asyncio.get_running_loop().run_in_executor(None, barrier.wait)
        wall_anchor = time.time() - time.perf_counter()
        # GC pauses dominate client-side lag at high event rates, so collection is off while streams run.
        gc.collect()
        gc.freeze()
        gc.disable()
        sampler = asyncio.create_task(_loop_lag_sampler(stop, lag))

        async def one(spec: dict) -> dict:
            ids = prompt_token_ids(spec["cell_b"], spec["nominal_T"], spec["req"], spec["prompt_tokens"])
            payload = {
                "model": model_name,
                "prompt": ids,
                "max_tokens": spec["osl"],
                "temperature": 0.0,
                "stream": True,
            }
            payload["ignore_eos" if eos_mode == "ignore_eos" else "min_tokens"] = (
                True if eos_mode == "ignore_eos" else spec["osl"]
            )
            t_send = time.perf_counter()
            ts = await _stream_once(session, url, headers, payload)
            if not ts:
                return {
                    **{k: spec[k] for k in ("req", "prompt_tokens", "osl", "nominal_T")},
                    "t_first_wall": None,
                    "deltas_ms": [],
                    "n_events": 0,
                    "ttft_ms": None,
                }
            return {
                **{k: spec[k] for k in ("req", "prompt_tokens", "osl", "nominal_T")},
                "t_first_wall": round(ts[0] + wall_anchor, 6),
                "deltas_ms": [round((ts[k + 1] - ts[k]) * 1e3, 3) for k in range(len(ts) - 1)],
                "n_events": len(ts),
                "ttft_ms": round((ts[0] - t_send) * 1e3, 2),
            }

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


def _shard_entry(req_specs, port, api_key, model_name, eos_mode, barrier, shard_id, out_path, rt: bool = False):
    if rt:
        # SCHED_RR (needs CAP_SYS_NICE) keeps the server's threads from preempting the client loop and tripping the p99 loop-lag flag.
        try:
            os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(10))
            print(f"shard {shard_id}: SCHED_RR set", flush=True)
        except (PermissionError, OSError) as e:
            print(f"shard {shard_id}: SCHED_RR unavailable ({e}); staying SCHED_OTHER", flush=True)
    res = asyncio.run(_run_shard(req_specs, port, api_key, model_name, eos_mode, barrier))
    for r in res:
        r["shard"] = shard_id
    Path(out_path).write_text(json.dumps(res))


def run_cell(
    cell: dict,
    port: int,
    api_key: str,
    model_name: str,
    eos_mode: str,
    shard_threshold: int,
    n_shards: int,
    tmp_dir: Path,
    rt_shards: bool = False,
) -> list[dict]:
    """Fire one (B, T) cell; returns per-request records (wall-anchored)."""
    b = cell["batch_size"]
    specs = [
        {
            "cell_b": b,
            "nominal_T": cell["nominal_T"],
            "req": i,
            "prompt_tokens": cell["prompt_tokens"],
            "osl": cell["osl"],
        }
        for i in range(b)
    ]
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
        p = ctx.Process(
            target=_shard_entry, args=(chunk, port, api_key, model_name, eos_mode, barrier, w, str(path), rt_shards)
        )
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


def launch_server(model, port, gpu_mem, max_model_len, api_key, run_dir, tp, max_num_seqs, max_num_batched_tokens):
    return launch_vllm(
        run_dir,
        model,
        port=port,
        tp=tp,
        gpu_util=gpu_mem,
        max_model_len=max_model_len,
        prefix_caching=False,  # each cell measures uncached prefill
        vllm_flags={
            "served_model_name": "llama",
            "host": "127.0.0.1",
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
            "api_key": api_key,
            "no_enable_log_requests": True,
        },
        timeout_s=600,
    )


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


def warmup_and_verify_eos(port: int, api_key: str, model_name: str) -> str:
    """8 short streams; returns the eos mode that yields the full requested osl."""
    for mode in ("ignore_eos", "min_tokens"):
        specs = [{"cell_b": 8, "nominal_T": 1024, "req": i, "prompt_tokens": 1024, "osl": 64} for i in range(8)]
        try:
            recs = asyncio.run(_run_shard(specs, port, api_key, model_name, mode, None))
        except Exception as e:  # e.g. 400 on an unsupported field
            print(f"warmup: eos mode {mode!r} failed ({e}); trying fallback", flush=True)
            continue
        n = [r["n_events"] for r in recs]
        print(f"warmup: eos mode {mode!r} -> events per stream {n} (requested 64)", flush=True)
        if min(n) >= 61:  # SSE events may coalesce a few tokens
            return mode
    raise SystemExit("neither ignore_eos nor min_tokens produced the requested output length")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default=None, help="model dir (Llama-3.1-8B-Instruct); needed unless --no-launch")
    ap.add_argument("--served-model-name", default="llama")
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--api-key", default="test")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=25600)
    ap.add_argument("--max-num-seqs", type=int, default=320)
    ap.add_argument("--max-num-batched-tokens", type=int, default=CHUNK_BUDGET)
    ap.add_argument("--tensor-parallel-size", type=int, default=4)
    ap.add_argument(
        "--pool-tokens",
        type=int,
        default=None,
        help="KV pool for lattice feasibility; default: the tp4 pool "
        f"({KV_POOL_TOKENS_TP4}); for other tp, pass the live pool and the "
        "lattice is re-derived by the same cap rule",
    )
    ap.add_argument("--shard-threshold", type=int, default=160)
    ap.add_argument("--n-shards", type=int, default=4)
    ap.add_argument(
        "--rt-shards",
        action="store_true",
        help="SCHED_RR for shard processes (needs root/CAP_SYS_NICE); the server "
        "keeps its normal policy. Counters the host's ~2 ms CFS preemptions "
        "of the client loop (loop-lag flag hygiene).",
    )
    ap.add_argument("--cells", default=None, help="optional explicit 'B:T,B:T,...' subset (default: the lattice)")
    ap.add_argument("--out-raw", required=True, help="append-only per-request JSONL.gz")
    ap.add_argument("--out-summary", required=True, help="operator cell-summary CSV")
    ap.add_argument("--no-launch", action="store_true")
    a = ap.parse_args(argv)
    if a.model is None and not a.no_launch:
        ap.error("--model is required to launch the server (or pass --no-launch)")

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
    over = [c for c in cells if c["exceeds_cap"] and not (c["batch_size"] == 160 and c["nominal_T"] == 12288)]
    if over:
        raise SystemExit(f"cells exceed the 0.95 pool cap: {over}")
    print(f"{len(cells)} cells; pool {pool} tokens; cap {CAP_FRAC}", flush=True)

    server = None
    if not a.no_launch:
        server = launch_server(
            a.model,
            a.port,
            a.gpu_mem,
            a.max_model_len,
            a.api_key,
            Path(a.out_raw).parent / "server",
            a.tensor_parallel_size,
            a.max_num_seqs,
            a.max_num_batched_tokens,
        )
    try:
        live_pool = kv_pool_from_log(server.results_dir / "server.log") if server else None
        print(f"server healthy; live KV pool from log: {live_pool} tokens", flush=True)

        eos_mode = warmup_and_verify_eos(a.port, a.api_key, a.served_model_name)
        meta = {
            "_meta": True,
            "tool": "serving_decode_grid.py",
            "tp": a.tensor_parallel_size,
            "engine_flags": {
                "gpu_mem": a.gpu_mem,
                "max_model_len": a.max_model_len,
                "max_num_seqs": a.max_num_seqs,
                "max_num_batched_tokens": a.max_num_batched_tokens,
                "prefix_caching": False,
            },
            "eos_mode": eos_mode,
            "live_kv_pool_tokens": live_pool,
            "lattice_pool_tokens": pool,
            "date": time.strftime("%Y-%m-%d"),
            "n_cells": len(cells),
        }

        out_raw = Path(a.out_raw)
        out_raw.parent.mkdir(parents=True, exist_ok=True)
        # Shard results go under this run's dir, never a shared tmp.
        tmp_dir = Path(tempfile.mkdtemp(dir=out_raw.parent, prefix="shards-"))
        rows = []
        with gzip.open(out_raw, "at") as raw_f:
            raw_f.write(json.dumps(meta) + "\n")
            for c in cells:
                t0 = time.time()
                recs = run_cell(
                    c,
                    a.port,
                    a.api_key,
                    a.served_model_name,
                    eos_mode,
                    a.shard_threshold,
                    a.n_shards,
                    tmp_dir,
                    rt_shards=a.rt_shards,
                )
                for r in recs:
                    raw_f.write(json.dumps({"cell": [c["batch_size"], c["nominal_T"]], **r}) + "\n")
                raw_f.flush()
                row = summarize_cell([r for r in recs if r["t_first_wall"] is not None])
                rows.append(row)
                print(
                    f"  B={c['batch_size']:>3} T={c['nominal_T']:>5}  "
                    f"itl_p50={row['decode_step_ms']:>8.3f} ms  ctx_eff={row['context_len']:>5}  "
                    f"n>={row['n_samples']:>4}  win={row['steady_window_s']:>6.2f}s  "
                    f"lag_p99={row['p99_loop_lag_ms']:>5.2f}  [{row['validation_status']}]  "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )

        out_sum = Path(a.out_summary)
        out_sum.parent.mkdir(parents=True, exist_ok=True)
        with out_sum.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})
        print(f"\nwrote {out_raw} + {out_sum}", flush=True)
        print(
            "authoritative grid CSV: just export --only serving_decode_grid "
            f"--results <dir holding {out_raw.name}> --dest <dir>",
            flush=True,
        )
    finally:
        if server is not None:
            server.stop()


if __name__ == "__main__":
    run_recorded(main, "serving_decode_grid", file_flag="--out-raw")
