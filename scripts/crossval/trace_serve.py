#!/usr/bin/env python3
"""Replay a request trace open loop and report per-request serving latency.

vLLM only: QuettaServe has no continuous batching, so its comparable number is
the static grid. Requests submit at arrival_ts/--speed, greedy, max_tokens from
the trace. Prints SERVE per request and SERVESUM at the end.
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


def _pcts(sorted_vals):
    return {f"p{q}": _pct(sorted_vals, q / 100) for q in (50, 90, 95, 99)}


def summarize(records, wall_s):
    """Aggregate per-request records; single-token records skip tpot."""
    if not records:
        raise ValueError("no completed requests")
    ttft = sorted(r["ttft_ms"] for r in records)
    tpot = sorted(r["tpot_ms"] for r in records if r["tpot_ms"] is not None)
    toks = sum(r["out_tokens"] for r in records)
    return {
        "n": len(records),
        "ttft_ms": _pcts(ttft),
        "tpot_ms": _pcts(tpot) if tpot else None,
        "req_s": len(records) / wall_s if wall_s > 0 else 0.0,
        "tok_s": toks / wall_s if wall_s > 0 else 0.0,
    }


def burst_recovery(records, sla_ttft_ms, window_s=5.0):
    """(recovery_s, peak_p99): time from the worst-latency window until p99 TTFT is
    back under the SLA target; recovery_s is None if it never returns."""
    if not records or sla_ttft_ms is None:
        return None, None
    rs = sorted(records, key=lambda r: r["arrival_s"])
    t0 = rs[0]["arrival_s"]
    buckets = {}
    for r in rs:
        buckets.setdefault(int((r["arrival_s"] - t0) // window_s), []).append(r["ttft_ms"])
    p99 = {w: _pct(sorted(v), 0.99) for w, v in buckets.items()}
    peak_w = max(p99, key=p99.get)
    for w in sorted(p99):
        if w > peak_w and p99[w] <= sla_ttft_ms:
            return (w - peak_w) * window_s, p99[peak_w]
    return None, p99[peak_w]


def _acceptance(reference_ids, generated_ids):
    """Matched leading-token length and rate between the trace's real assistant
    tokens and what the engine generated. Measured live, not from a golden file."""
    n = 0
    for a, b in zip(reference_ids, generated_ids):
        if a != b:
            break
        n += 1
    return n, (n / len(reference_ids) if reference_ids else 0.0)


async def _replay_vllm(reqs, offsets, args):
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=args.pp,
        enable_expert_parallel=args.ep,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        enable_prefix_caching=args.prefix_caching,
        trust_remote_code=True,
    ))
    t_start = time.perf_counter()
    records = []
    done = []  # requests the engine actually completed, for the workload-identity gate

    async def one(i, req, offset):
        delay = offset - (time.perf_counter() - t_start)
        if delay > 0:
            await asyncio.sleep(delay)
        out_len = int(req.get("output_length") or args.max_tokens)
        sp = SamplingParams(temperature=0.0, max_tokens=out_len, ignore_eos=True)
        t_submit = time.perf_counter()
        t_first, n_tok, gen_ids = None, 0, []
        async for out in engine.generate(
            {"prompt_token_ids": req["prompt_token_ids"]}, sp, request_id=str(i)
        ):
            gen_ids = out.outputs[0].token_ids
            n_tok = len(gen_ids)
            if t_first is None and n_tok >= 1:
                t_first = time.perf_counter()
        t_done = time.perf_counter()
        ttft_ms = (t_first - t_submit) * 1e3 if t_first else (t_done - t_submit) * 1e3
        tpot_ms = ((t_done - t_first) * 1e3 / (n_tok - 1)) if t_first and n_tok > 1 else None
        rec = {"ttft_ms": ttft_ms, "tpot_ms": tpot_ms, "out_tokens": n_tok, "arrival_s": offset}
        ref = req.get("output_token_ids")  # real assistant text, when the trace carries it
        if ref:
            rec["accept_len"], rec["accept_rate"] = _acceptance(ref, list(gen_ids))
        records.append(rec)
        done.append(req)
        tp = f"{tpot_ms:.3f}" if tpot_ms is not None else "-"
        print(f"SERVE id={i} session={req.get('session_id', '-')} "
              f"in={req['prompt_len']} out={n_tok} ttft_ms={ttft_ms:.1f} tpot_ms={tp}",
              flush=True)
        return rec

    await asyncio.gather(*(one(i, r, o) for i, (r, o) in enumerate(zip(reqs, offsets))))
    wall = time.perf_counter() - t_start
    return records, wall, done


def main():
    ap = argparse.ArgumentParser(description="open-loop trace replay (serving latency)")
    ap.add_argument("trace", help="JSONL trace (synth_bench gen / Mooncake capture)")
    ap.add_argument("--engine", choices=("vllm",), default="vllm",
                    help="quettaserve adapter lands when it has continuous batching")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1, help="pipeline parallel size")
    ap.add_argument("--ep", action="store_true", help="enable expert parallel (MoE)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--prefix-caching", action="store_true",
                    help="off by default so both engines pay full prefill; on to measure the cache")
    ap.add_argument("--speed", type=float, default=1.0, help="arrival time compression factor")
    ap.add_argument("--max-tokens", type=int, default=128, help="when the trace has no output_length")
    ap.add_argument("--limit", type=int, help="replay only the first N requests")
    ap.add_argument("--sla-ttft-ms", type=float, help="SLA target: max p99 TTFT ms")
    ap.add_argument("--sla-tpot-ms", type=float, help="SLA target: max p99 TPOT ms/token")
    ap.add_argument("--json", help="write the aggregate summary to this path")
    args = ap.parse_args()

    reqs = synth_bench.load_trace(args.trace)
    if args.limit:
        reqs = reqs[: args.limit]
    offsets = plan_arrivals(reqs, args.speed)
    sources = ",".join(sorted({str(r.get("source", "?")) for r in reqs}))
    text_mode = "real" if any(r.get("output_token_ids") for r in reqs) else "synthetic"
    print(f"META trace={args.trace} reqs={len(reqs)} engine={args.engine} "
          f"tp={args.tp} pp={args.pp} ep={int(args.ep)} "
          f"speed={args.speed} prefix_caching={int(args.prefix_caching)} "
          f"source={sources} text_mode={text_mode}", flush=True)
    expected = synth_bench.workload_hash(reqs)
    records, wall, done = asyncio.run(_replay_vllm(reqs, offsets, args))
    replayed = synth_bench.workload_hash(done)
    match = expected == replayed
    print(f"WORKLOADSUM expected={expected} replayed={replayed} match={int(match)} "
          f"n={len(done)}/{len(reqs)}", flush=True)
    if not match:
        print("WORKLOADVOID replayed workload differs from the trace; comparison is void",
              flush=True)
    s = summarize(records, wall)
    tp = s["tpot_ms"]
    qs = (50, 90, 95, 99)
    print(f"SERVESUM n={s['n']} "
          + " ".join(f"ttft_ms_p{q}={s['ttft_ms'][f'p{q}']:.1f}" for q in qs) + " "
          + (" ".join(f"tpot_ms_p{q}={tp[f'p{q}']:.3f}" for q in qs) + " " if tp else "tpot_ms_p50=- ")
          + f"req_s={s['req_s']:.3f} tok_s={s['tok_s']:.1f}", flush=True)
    if args.sla_ttft_ms or args.sla_tpot_ms:
        bound = []
        if args.sla_ttft_ms and s["ttft_ms"]["p99"] > args.sla_ttft_ms:
            bound.append("ttft_p99")
        if args.sla_tpot_ms and tp and tp["p99"] > args.sla_tpot_ms:
            bound.append("tpot_p99")
        print(f"SLASUM verdict={'PASS' if not bound else 'FAIL'} bound={','.join(bound) or 'none'} "
              f"ttft_p99={s['ttft_ms']['p99']:.1f}/{args.sla_ttft_ms or '-'} "
              f"tpot_p99={(tp['p99'] if tp else 0):.3f}/{args.sla_tpot_ms or '-'} "
              f"rate_req_s={s['req_s']:.3f}", flush=True)
    if args.sla_ttft_ms:
        rec_s, peak = burst_recovery(records, args.sla_ttft_ms)
        print(f"BURSTSUM peak_ttft_p99={peak:.1f} "
              f"recovery_s={'na' if rec_s is None else f'{rec_s:.1f}'} "
              f"target_ms={args.sla_ttft_ms}", flush=True)
    # Acceptance needs the trace's real assistant text; synthetic traces carry none,
    # so this reports off rather than a fake number.
    acc = [r for r in records if "accept_len" in r]
    if acc:
        print(f"SPECSUM spec=measured accept_len={sum(r['accept_len'] for r in acc)/len(acc):.2f} "
              f"accept_rate={sum(r['accept_rate'] for r in acc)/len(acc):.3f} n={len(acc)}", flush=True)
    else:
        print("SPECSUM spec=off note=needs-real-text-trace", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(s, indent=2, sort_keys=True))
    if not match:
        sys.exit(3)  # hard fail: the two runs did not process identical work


if __name__ == "__main__":
    main()
