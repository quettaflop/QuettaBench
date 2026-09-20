#!/usr/bin/env python3
"""Open-loop trace replay against a serving engine (TTFT/TPOT under arrivals).

The serving half of the trace story (static half: vmin_fit PROMPT_MODE=trace).
Their numbers answer different questions and never share a column. vLLM only for
now: QuettaServe lacks continuous batching, prefix caching, and routing, so its
comparable number is the static grid; an adapter with the same SERVE/SERVESUM
output drops in when those land.

Submits at arrival_ts/--speed (missing: back-to-back), greedy, max_tokens from
output_length. Emits per-request SERVE and aggregate SERVESUM. vLLM is imported
lazily so the module stays importable in CI without a GPU.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import synth_bench


def plan_arrivals(reqs, speed):
    """Submission offsets in seconds: arrival_ts scaled by --speed (first at 0),
    else all 0 (back-to-back)."""
    if not any("arrival_ts" in r for r in reqs):
        return [0.0] * len(reqs)
    t0 = min(r.get("arrival_ts", 0.0) for r in reqs)
    return [max(0.0, (r.get("arrival_ts", t0) - t0)) / max(speed, 1e-9) for r in reqs]


def _pct(sorted_vals, q):
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def summarize(records, wall_s):
    """Aggregate per-request records; single-token records skip tpot but still
    count for ttft and throughput."""
    if not records:
        raise ValueError("no completed requests")
    ttft = sorted(r["ttft_ms"] for r in records)
    tpot = sorted(r["tpot_ms"] for r in records if r["tpot_ms"] is not None)
    toks = sum(r["out_tokens"] for r in records)
    return {
        "n": len(records),
        "ttft_ms": {"p50": _pct(ttft, 0.50), "p95": _pct(ttft, 0.95)},
        "tpot_ms": ({"p50": _pct(tpot, 0.50), "p95": _pct(tpot, 0.95)}
                    if tpot else None),
        "req_s": len(records) / wall_s if wall_s > 0 else 0.0,
        "tok_s": toks / wall_s if wall_s > 0 else 0.0,
    }


async def _replay_vllm(reqs, offsets, args):
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        enable_prefix_caching=args.prefix_caching,
        trust_remote_code=True,
    ))
    t_start = time.perf_counter()
    records = []

    async def one(i, req, offset):
        delay = offset - (time.perf_counter() - t_start)
        if delay > 0:
            await asyncio.sleep(delay)
        out_len = int(req.get("output_length") or args.max_tokens)
        sp = SamplingParams(temperature=0.0, max_tokens=out_len, ignore_eos=True)
        t_submit = time.perf_counter()
        t_first, n_tok = None, 0
        async for out in engine.generate(
            {"prompt_token_ids": req["prompt_token_ids"]}, sp, request_id=str(i)
        ):
            n_tok = len(out.outputs[0].token_ids)
            if t_first is None and n_tok >= 1:
                t_first = time.perf_counter()
        t_done = time.perf_counter()
        ttft_ms = (t_first - t_submit) * 1e3 if t_first else (t_done - t_submit) * 1e3
        tpot_ms = ((t_done - t_first) * 1e3 / (n_tok - 1)) if t_first and n_tok > 1 else None
        rec = {"ttft_ms": ttft_ms, "tpot_ms": tpot_ms, "out_tokens": n_tok}
        records.append(rec)
        tp = f"{tpot_ms:.3f}" if tpot_ms is not None else "-"
        print(f"SERVE id={i} session={req.get('session_id', '-')} "
              f"in={req['prompt_len']} out={n_tok} ttft_ms={ttft_ms:.1f} tpot_ms={tp}",
              flush=True)
        return rec

    await asyncio.gather(*(one(i, r, o) for i, (r, o) in enumerate(zip(reqs, offsets))))
    wall = time.perf_counter() - t_start
    return records, wall


def main():
    ap = argparse.ArgumentParser(description="open-loop trace replay (serving latency)")
    ap.add_argument("trace", help="JSONL trace (synth_bench gen / Mooncake capture)")
    ap.add_argument("--engine", choices=("vllm",), default="vllm",
                    help="quettaserve adapter lands when it has continuous batching")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--prefix-caching", action="store_true",
                    help="off by default so both engines pay full prefill; on to measure the cache")
    ap.add_argument("--speed", type=float, default=1.0, help="arrival time compression factor")
    ap.add_argument("--max-tokens", type=int, default=128, help="when the trace has no output_length")
    ap.add_argument("--limit", type=int, help="replay only the first N requests")
    ap.add_argument("--json", help="write the aggregate summary to this path")
    args = ap.parse_args()

    reqs = synth_bench.load_trace(args.trace)
    if args.limit:
        reqs = reqs[: args.limit]
    offsets = plan_arrivals(reqs, args.speed)
    print(f"META trace={args.trace} reqs={len(reqs)} engine={args.engine} "
          f"speed={args.speed} prefix_caching={int(args.prefix_caching)}", flush=True)
    records, wall = asyncio.run(_replay_vllm(reqs, offsets, args))
    s = summarize(records, wall)
    tp = s["tpot_ms"]
    print(f"SERVESUM n={s['n']} ttft_ms_p50={s['ttft_ms']['p50']:.1f} "
          f"ttft_ms_p95={s['ttft_ms']['p95']:.1f} "
          + (f"tpot_ms_p50={tp['p50']:.3f} tpot_ms_p95={tp['p95']:.3f} " if tp else "tpot_ms_p50=- ")
          + f"req_s={s['req_s']:.3f} tok_s={s['tok_s']:.1f}", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(s, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
