#!/usr/bin/env python3
"""Replay a request trace open loop and report per-request serving latency.

vLLM only: QuettaServe has no continuous batching, so its comparable number is
the static grid. Requests submit at arrival_ts/--speed, greedy, max_tokens from
the trace. Prints SERVE per request and SERVESUM at the end.
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import synth_bench

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.workloads.arrival import burst_arrivals, poisson_arrivals, ramp_arrivals


def injected_arrivals(spec, n):
    """Arrival schedule for --arrival; None means keep the trace's own times.
    Specs: poisson:<rate>, ramp:<start>:<end>, burst:<n>x<size>@<gap_s>."""
    if spec == "trace":
        return None
    kind, _, rest = spec.partition(":")
    if kind == "poisson":
        return poisson_arrivals(n, float(rest))
    if kind == "ramp":
        a, b = rest.split(":")
        return ramp_arrivals(n, float(a), float(b))
    if kind == "burst":
        nb, rest2 = rest.split("x")
        size, gap = rest2.split("@")
        return burst_arrivals(n, int(nb), int(size), float(gap))
    raise ValueError(f"unknown arrival spec {spec!r}")


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


def integrate_power(samples_w, interval_s):
    """Rectangle-rule joules from per-sample total board watts. A missed sample
    shrinks the integral rather than crashing the run, so joules is a floor."""
    return sum(samples_w) * interval_s


def _power_sampler(samples, stop, interval_s=1.0):
    """Board power at 1 Hz over the devices in CUDA_VISIBLE_DEVICES (all GPUs
    when unset). Board scope includes idle draw and everything colocated on
    those devices, which is why ENERGYSUM is labeled gpu_board."""
    ids = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    cmd = ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"]
    if ids:
        cmd += ["-i", ids]
    while not stop.wait(interval_s):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True).stdout
            vals = [float(x) for x in out.split()]
            if vals:
                samples.append(sum(vals))
        except (OSError, ValueError):
            pass


def _acceptance(reference_ids, generated_ids):
    """Matched leading-token length and rate between the trace's real assistant
    tokens and what the engine generated. Measured live, not from a golden file."""
    n = 0
    for a, b in zip(reference_ids, generated_ids):
        if a != b:
            break
        n += 1
    return n, (n / len(reference_ids) if reference_ids else 0.0)


def sla_verdict(s, sla_ttft_ms, sla_tpot_ms):
    """Which p99 bound breaks at this operating point; empty means PASS."""
    bound = []
    if sla_ttft_ms and s["ttft_ms"]["p99"] > sla_ttft_ms:
        bound.append("ttft_p99")
    tp = s["tpot_ms"]
    if sla_tpot_ms and tp and tp["p99"] > sla_tpot_ms:
        bound.append("tpot_p99")
    return bound


def goodput_search(start=1.0, max_iters=6, tol=0.1):
    """Generator bisection for --goodput: yields the next speed, receives
    whether that probe met the SLA. Doubles from start to find the first FAIL,
    halves to find the first PASS, then bisects; stops once the bracket is
    within tol. Capped at max_iters probes so a sweep can never hold the GPU
    unbounded; without the cap a flat SLA response would double forever."""
    lo = hi = None
    speed = start
    for _ in range(max_iters):
        passed = yield speed
        if passed:
            lo = speed
        else:
            hi = speed
        if lo is not None and hi is not None:
            if (hi - lo) <= tol * hi:
                return
            speed = (lo + hi) / 2
        elif passed:
            speed = speed * 2
        else:
            speed = speed / 2


def _engine_args(args):
    from vllm.engine.arg_utils import AsyncEngineArgs
    return AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=args.pp,
        enable_expert_parallel=args.ep,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        enable_prefix_caching=args.prefix_caching,
        trust_remote_code=True,
    )


async def _serve(engine, reqs, offsets, args):
    """One replay pass on an existing engine. The engine is passed in so a
    goodput sweep reuses one model load across probes; an engine built in a
    previous event loop cannot be reused in a new asyncio.run."""
    from vllm import SamplingParams

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


def _run_goodput(args, reqs, expected):
    """Swept goodput: replay at bisected speeds on one engine until the SLA
    flips PASS to FAIL, then report the highest passing rate. Every probe runs
    the workload-identity gate; a mismatch voids the whole sweep because a
    partially completed probe would inflate the reported rate."""
    async def sweep():
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        engine = AsyncLLMEngine.from_engine_args(_engine_args(args))
        gen = goodput_search(start=args.speed)
        speed = next(gen)
        best, fail_bound, iters = None, None, 0
        while True:
            records, wall, done = await _serve(engine, reqs, plan_arrivals(reqs, speed), args)
            iters += 1
            replayed = synth_bench.workload_hash(done)
            if replayed != expected:
                print(f"WORKLOADSUM expected={expected} replayed={replayed} match=0 "
                      f"n={len(done)}/{len(reqs)}", flush=True)
                print("WORKLOADVOID replayed workload differs from the trace; sweep is void",
                      flush=True)
                sys.exit(3)
            s = summarize(records, wall)
            bound = sla_verdict(s, args.sla_ttft_ms, args.sla_tpot_ms)
            print(f"GOODPROBE speed={speed:g} verdict={'FAIL' if bound else 'PASS'} "
                  f"bound={','.join(bound) or 'none'} req_s={s['req_s']:.3f}", flush=True)
            if not bound and (best is None or s["req_s"] > best[1]):
                best = (speed, s["req_s"])
            if bound:
                fail_bound = ",".join(bound)
            try:
                speed = gen.send(not bound)
            except StopIteration:
                break
        return best, fail_bound, iters

    best, fail_bound, iters = asyncio.run(sweep())
    if best is None:
        print(f"GOODPUTSUM max_req_s=none bound={fail_bound or 'na'} iterations={iters} "
              f"note=no-passing-speed-within-budget", flush=True)
        sys.exit(1)
    print(f"GOODPUTSUM max_req_s={best[1]:.3f} speed={best[0]:g} "
          f"bound={fail_bound or 'none-within-range'} iterations={iters}", flush=True)


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
    ap.add_argument("--arrival", default="trace",
                    help="trace | poisson:<rate> | ramp:<start>:<end> | burst:<n>x<size>@<gap_s>")
    ap.add_argument("--max-tokens", type=int, default=128, help="when the trace has no output_length")
    ap.add_argument("--limit", type=int, help="replay only the first N requests")
    ap.add_argument("--sla-ttft-ms", type=float, help="SLA target: max p99 TTFT ms")
    ap.add_argument("--sla-tpot-ms", type=float, help="SLA target: max p99 TPOT ms/token")
    ap.add_argument("--goodput", action="store_true",
                    help="sweep --speed to the SLA PASS/FAIL boundary (max 6 probes)")
    ap.add_argument("--energy", action="store_true",
                    help="1 Hz board-power sampling over the replay; ENERGYSUM")
    ap.add_argument("--json", help="write the aggregate summary to this path")
    args = ap.parse_args()
    if args.goodput and not (args.sla_ttft_ms or args.sla_tpot_ms):
        ap.error("--goodput needs --sla-ttft-ms and/or --sla-tpot-ms")
    if args.energy and args.goodput:
        ap.error("--energy covers a single replay window; drop --goodput")
    if args.energy and not shutil.which("nvidia-smi"):
        ap.error("--energy needs nvidia-smi on PATH")

    reqs = synth_bench.load_trace(args.trace)
    if args.limit:
        reqs = reqs[: args.limit]
    injected = injected_arrivals(args.arrival, len(reqs))
    if injected is not None:
        # Injected schedules join the workload identity: arrival_ts is a hash
        # input, so trace-paced and injected runs can never share a hash.
        for r, t in zip(reqs, injected):
            r["arrival_ts"] = round(t, 6)
    offsets = plan_arrivals(reqs, args.speed)
    sources = ",".join(sorted({str(r.get("source", "?")) for r in reqs}))
    text_mode = "real" if any(r.get("output_token_ids") for r in reqs) else "synthetic"
    print(f"META trace={args.trace} reqs={len(reqs)} engine={args.engine} "
          f"tp={args.tp} pp={args.pp} ep={int(args.ep)} "
          f"speed={args.speed} arrival={args.arrival} "
          f"prefix_caching={int(args.prefix_caching)} "
          f"source={sources} text_mode={text_mode}", flush=True)
    expected = synth_bench.workload_hash(reqs)
    if args.goodput:
        _run_goodput(args, reqs, expected)
        return

    stop, samples = threading.Event(), []
    sampler = threading.Thread(target=_power_sampler, args=(samples, stop), daemon=True)

    async def _single():
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        engine = AsyncLLMEngine.from_engine_args(_engine_args(args))
        if args.energy:
            sampler.start()  # after the model load so joules cover the replay only
        out = await _serve(engine, reqs, offsets, args)
        stop.set()
        return out

    records, wall, done = asyncio.run(_single())
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
        bound = sla_verdict(s, args.sla_ttft_ms, args.sla_tpot_ms)
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
    if args.energy:
        sampler.join(timeout=2.0)
        joules = integrate_power(samples, 1.0)
        out_toks = sum(r["out_tokens"] for r in records)
        ids = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        print(f"ENERGYSUM joules={joules:.0f} "
              f"joules_per_token={joules / max(out_toks, 1):.3f} scope=gpu_board "
              f"devices={len(ids.split(',')) if ids else 'all'} samples={len(samples)}",
              flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(s, indent=2, sort_keys=True))
    if not match:
        sys.exit(3)  # hard fail: the two runs did not process identical work


if __name__ == "__main__":
    main()
