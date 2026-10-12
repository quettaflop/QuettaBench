"""The measurement procedure: send a Trace's requests through a Driver and collect one RequestResult row per request.

Closed loop holds mode.concurrency slots; open loop fires each request at its arrival. SessionDispatcher feeds each reply into the session's later turns; run_grid times a GridTrace cell by cell. The metrics.core gates and the workload identity gate (exit 3 on a hash mismatch) run after.
"""

import asyncio
import dataclasses
import itertools
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, LocalEntryNotFoundError

from ..config.schema import ConfigError, WorkloadCfg
from ..engines.drivers import PrefixCacheResetError
from ..metrics.core import (
    BenchmarkSummary,
    acceptance,
    aggregate,
    aggregate_per_turn,
    annotate_multi_turn_cache_estimate,
    annotate_request_observability,
    burst_recovery,
    first_error,
    print_multi_turn_summary,
    print_summary,
    sla_verdict,
    success_rate_gate,
    summarize_serve,
    transient_gate,
    usage_gate,
)
from ..metrics.results import goodprobe_line, grid_event_line, print_legacy_grid_lines, print_legacy_lines
from ..metrics.step_fit import GIB, fit, fit_run
from ..workloads.arrival import offered_rate, parse_arrival, rebase_arrivals
from ..workloads.base import Request, workload_hash
from ..workloads.sessions import SessionTrace, _sleep_until_elapsed, inter_turn_wait_s

TRACE_REQUEST_ID_PREFIX = "agenticbench"
IDENTITY_EXIT = 3
DEFAULT_WORKLOAD = WorkloadCfg()
RESERVE_GIB = 5.0  # per GPU, held back from the KV budget for activations and graphs
TEXT_TOKENS = 64  # mode.returns=text: greedy tokens decoded per cell


class RunAborted(Exception):
    """The run cannot produce a measurement; the message is the ABORT line."""


@asynccontextmanager
async def no_limit():
    yield


@dataclass
class Summary:
    """One run: the aggregate (None for a grid run), the rows in summary order, the per-turn breakdown (multi-turn only) and extra run.json sections."""
    core: BenchmarkSummary
    rows: list
    turns: list = field(default_factory=list)
    sections: dict = field(default_factory=dict)
    status: str = "ok"
    reason: str | None = None

    def to_dict(self):
        """run.json's summary; aggregate and per_turn percentiles are linear, serve/sla/burst nearest-rank."""
        core = {**self.core.to_dict(), "percentile_method": "linear"} if self.core else {}
        out = {**core, **self.sections}
        if self.turns:
            out["per_turn"] = [{**t.to_dict(), "percentile_method": "linear"}
                               for t in self.turns]
        return out


def make_trace_request_id(
    *,
    profile_name: str,
    concurrency: int,
    session_id: int,
    turn_index: int,
    request_index: int,
) -> str:
    return (
        f"{TRACE_REQUEST_ID_PREFIX}__p={profile_name}"
        f"__c={concurrency}"
        f"__t={turn_index}"
        f"__s={session_id}"
        f"__i={request_index}"
    )


def trace_name(trace) -> str:
    meta = trace.source_meta()
    return meta.get("profile") or meta.get("path", "")


def sampling(mode, request) -> dict:
    """Sampling fields for one request; exact_output_length pins min_tokens."""
    return {
        "ignore_eos": mode.ignore_eos or mode.exact_output_length,
        "temperature": mode.temperature,
        "seed": mode.sampling_seed,
        "min_tokens": (request.max_tokens or 0) if mode.exact_output_length else 0,
    }


def arrival_offsets(trace, workload) -> list:
    """Submission offsets for a single-turn trace. An injected schedule is stamped on arrival_ts first so it joins the workload hash; "trace" keeps the trace's own times, rebased to the first."""
    reqs = trace.requests()
    injected = parse_arrival(workload.arrival, len(reqs), workload.seed)
    if injected is None:
        return rebase_arrivals([r.arrival_ts for r in reqs], workload.speed)
    for r, t in zip(reqs, injected):
        r.arrival_ts = round(t, 6)
    return [t / workload.speed for t in injected]


async def warmup(trace, driver, mode, count, multi_turn):
    """Send warmup requests over the run's own session, so its connections exist before timing; abort when none succeeds."""
    if multi_turn and not mode.reset_prefix_cache:
        # Multi-turn warmup duplicates measured prompts rather than consuming them.
        print("WARNING: multi-turn warmup replays prompts the measured run will also send,")
        print("         so the cache starts warm and turn-1 TTFT is understated. Pass")
        print("         --reset-prefix-cache (or --warmup 0) to measure a cold first turn.")
    print(f"Warming up with {count} profile requests at concurrency {mode.concurrency}...")
    slots = asyncio.Semaphore(mode.warmup_concurrency or mode.concurrency)

    async def one(request):
        async with slots:
            return await driver.send(request, **sampling(mode, request))

    results = await asyncio.gather(*(one(r) for r in trace.warmup_requests(count)),
                                   return_exceptions=True)
    ok = sum(1 for r in results if not isinstance(r, BaseException) and r is not None and r.success)
    print(f"Warmup done: {ok}/{len(results)} succeeded.")
    if ok == 0:
        raise RunAborted("ABORT: every warmup request failed. The server is not serving "
                         "this profile; check the server log before benchmarking.")


def emitter(mode, writer, probe=None):
    """emit(request, result): stamp the TTFT basis, score acceptance against the trace's reply when the driver returned tokens, write the row."""
    basis = "arrival" if mode.load == "open" else "slot"
    tag = {} if probe is None else {"probe": probe}

    def emit(request, result):
        result.ttft_basis = basis
        if request.output_token_ids and result.generated_token_ids is not None:
            result.accepted_tokens = acceptance(request.output_token_ids,
                                                result.generated_token_ids)[0]
        writer.append(result, **tag)

    return emit


async def replay(trace, driver, mode, emit, offsets, request_id):
    """Single-turn dispatch; returns ([(request, result)] by index, start time). The first mode.discard_first requests stay out of the summary."""
    open_loop = mode.load == "open"
    slots = asyncio.Semaphore(mode.concurrency)
    reqs = trace.requests()
    rows = [None] * len(reqs)
    start = time.perf_counter()

    async def dispatch(i, request, at):
        delay = at - (time.perf_counter() - start)
        if delay > 0:
            await asyncio.sleep(delay)
        dispatched = time.perf_counter() - start
        async with (no_limit() if open_loop else slots):
            granted = time.perf_counter() - start
            result = await driver.send(request, request_id=request_id(i, 0, i),
                                       **sampling(mode, request))
        annotate_request_observability(
            result, request_index=i, request=request, scheduled_at_s=at,
            dispatch_started_at_s=dispatched, semaphore_acquired_at_s=granted,
            completed_at_s=time.perf_counter() - start)
        if i < mode.discard_first:
            result.excluded_from_summary = True
        emit(request, result)
        rows[i] = (request, result)

    await asyncio.gather(*(dispatch(i, r, t) for i, (r, t) in enumerate(zip(reqs, offsets))))
    fail = sum(1 for _, r in rows if not r.success)
    if fail > 0 and fail >= len(reqs) * 0.9:
        raise RunAborted(f"ABORT: {fail}/{len(reqs)} requests failed "
                         f"({fail / len(reqs) * 100:.0f}%). "
                         f"Server may not be functional. Check server logs.")
    return rows, start


def server_counters(driver):
    """Spec-decode counters of the driver's server, or None."""
    counters = getattr(driver, "spec_counters", None)
    return counters() if counters else None


async def measure(trace, driver, mode, emit, workload, request_id):
    """Warm, reset, dispatch: (results, results_by_turn | None, [(request, result)], duration, server SPECSUM | None). Spec counters are read around dispatch only, so warmup is not in the window."""
    multi = isinstance(trace, SessionTrace)
    offsets = None if multi else arrival_offsets(trace, workload)
    sessions = None
    if multi:
        sessions = SessionDispatcher(trace, mode, workload,
                                     lambda r, **kw: driver.send(r, **kw, **sampling(mode, r)),
                                     request_id, emit)
        sessions.announce()
    async with driver.session(mode.concurrency, mode.load == "open"):
        if mode.warmup > 0:
            await warmup(trace, driver, mode, mode.warmup, multi)
        # After warmup, so the run starts cold whatever warmup left behind.
        if mode.reset_prefix_cache:
            print(f"Prefix cache: {await driver.reset()}")
        before = server_counters(driver)
        if multi:
            results, by_turn, pairs, start = await sessions.run()
        else:
            pairs, start = await replay(trace, driver, mode, emit, offsets, request_id)
            results, by_turn = [r for _, r in pairs], None
        duration = time.perf_counter() - start
        spec = server_spec_section(before, server_counters(driver) if before else None)
    return results, by_turn, pairs, duration, spec


def serve_records(pairs, open_loop):
    """Serving records from the successful counted rows: TTFT (whole latency when no token came), wall-clock TPOT, output tokens, and the burst bucket time (the slot grant under a closed loop)."""
    out = []
    for _, r in pairs:
        if not r.success or r.excluded_from_summary:
            continue
        n = r.output_tokens
        arrival = r.scheduled_at_s if open_loop and r.scheduled_at_s is not None \
            else r.semaphore_acquired_at_s
        out.append({"ttft_ms": (r.ttft or r.e2el) * 1e3, "out_tokens": n, "arrival_s": arrival,
                    "tpot_ms": (r.e2el - r.ttft) * 1e3 / (n - 1) if r.ttft and n > 1 else None})
    return out


def spec_section(pairs):
    """SPECSUM: mean accepted leading tokens and rate over the scored rows."""
    scored = [(r.accepted_tokens, len(q.output_token_ids)) for q, r in pairs
              if r.accepted_tokens is not None]
    if not scored:
        return {"spec": "off"}
    return {"spec": "measured", "n": len(scored),
            "accept_len": sum(a for a, _ in scored) / len(scored),
            "accept_rate": sum(a / n for a, n in scored) / len(scored)}


def server_spec_section(before, after):
    """SPECSUM from the server's counter deltas as vllm bench serve defines them: accept_rate = accepted / draft tokens, accept_len = 1 + accepted / drafts. None without both scrapes or a draft in the window."""
    if before is None or after is None:
        return None
    d = {key: int(after[key] - before[key]) for key in after}
    if d["drafts"] <= 0 or d["draft_tokens"] <= 0:
        return None
    return {"spec": "measured", "source": "server_metrics", **d,
            "accept_rate": d["accepted_tokens"] / d["draft_tokens"],
            "accept_len": 1 + d["accepted_tokens"] / d["drafts"], "n": d["drafts"]}


def crossval_sections(pairs, duration, mode, server_spec=None):
    """serve (nearest-rank), sla, burst and spec; spec prefers the server's counters over the rows'."""
    out = {"spec": server_spec or spec_section(pairs)}
    records = serve_records(pairs, mode.load == "open")
    if not records:
        return out
    s = out["serve"] = summarize_serve(records, duration, "nearest")
    if mode.sla_ttft_ms or mode.sla_tpot_ms:
        bound = sla_verdict(s, mode.sla_ttft_ms, mode.sla_tpot_ms)
        out["sla"] = {"verdict": "FAIL" if bound else "PASS", "bound": bound,
                      "ttft_p99": s["ttft_ms"]["p99"],
                      "tpot_p99": s["tpot_ms"]["p99"] if s["tpot_ms"] else None,
                      "rate_req_s": s["req_s"], "percentile_method": s["percentile_method"],
                      "sla_ttft_ms": mode.sla_ttft_ms, "sla_tpot_ms": mode.sla_tpot_ms}
    if mode.sla_ttft_ms:
        recovery, peak = burst_recovery(records, mode.sla_ttft_ms)
        out["burst"] = {"peak_ttft_p99": peak, "recovery_s": recovery,
                        "target_ms": mode.sla_ttft_ms, "percentile_method": "nearest"}
    return out


def goodput_search(start=1.0, max_iters=6, tol=0.1):
    """Generator bisection over speed: send() whether the last probe met the SLA. Doubles to the first FAIL, halves to the first PASS, then bisects within tol; at most max_iters probes."""
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


def sweep(trace, driver, mode, writer, workload, request_id):
    """--goodput-sweep: one measure() per probe speed (warmup on the first only), stopping at the first probe that breaks workload identity. Returns the last measure() result and the goodput section."""
    gen, probes, best, fail_bound = goodput_search(start=workload.speed), [], None, []
    speed = next(gen)
    while True:
        probe_mode = mode if not probes else dataclasses.replace(mode, warmup=0)
        out = asyncio.run(measure(trace, driver, probe_mode, emitter(mode, writer, len(probes)),
                                  dataclasses.replace(workload, speed=speed), request_id))
        section = {"max_req_s": None, "speed": None, "fail_bound": fail_bound, "probes": probes}
        if not identity(trace, served(out[2]))["match"]:
            return out, {**section, "iterations": len(probes) + 1}
        s = summarize_serve(serve_records(out[2], True), out[3], "nearest")
        bound = sla_verdict(s, mode.sla_ttft_ms, mode.sla_tpot_ms)
        probe = {"speed": speed, "verdict": "FAIL" if bound else "PASS", "bound": bound,
                 "req_s": s["req_s"]}
        print(goodprobe_line(probe), flush=True)
        probes.append(probe)
        if not bound and (best is None or s["req_s"] > best["req_s"]):
            best = probe
        if bound:
            fail_bound = bound
        try:
            speed = gen.send(not bound)
        except StopIteration:
            break
    return out, {"max_req_s": best and best["req_s"], "speed": best and best["speed"],
                 "fail_bound": fail_bound, "iterations": len(probes), "probes": probes}


def stop(writer, summary, status, reason, code):
    """Close run.json with status and exit with code."""
    writer.finish(summary, status=status, reason=reason)
    sys.exit(code)


def enforce(gate, writer, summary):
    """Print a metrics.core gate's message; on abort close the run and exit 1."""
    verdict, message = gate
    if message:
        print(message)
    if verdict == "abort":
        stop(writer, summary.to_dict(), "aborted", message, 1)


def served(pairs):
    """The requests the identity gate hashes: every one dispatched, failed or not, so the hash moves only when what was sent differs from the trace."""
    return [q for q, _ in pairs]


def identity(trace, served) -> dict:
    """The served requests hashed the way the trace hashes itself."""
    expected, replayed = trace.workload_hash(), workload_hash(served)
    return {"expected": expected, "replayed": replayed, "match": expected == replayed,
            "n": len(served), "of": len(trace.requests())}


def run(trace, driver, mode, writer, workload=DEFAULT_WORKLOAD) -> Summary:
    """One measured run of trace through driver; a grid run reads its engine facts from writer.cfg."""
    if mode.grid:
        record, steps, rows = run_grid(trace, driver, writer.cfg, writer)
        summary = Summary(None, rows, sections={"grid": record})
        writer.finish(summary.to_dict(), steps=steps)
        return summary
    name = trace_name(trace)

    def request_id(session, turn, index):
        if not mode.trace_request_ids:
            return None
        return make_trace_request_id(profile_name=name, concurrency=mode.concurrency,
                                     session_id=session, turn_index=turn, request_index=index)

    goodput = None
    try:
        if mode.goodput_sweep:
            measured, goodput = sweep(trace, driver, mode, writer, workload, request_id)
        else:
            measured = asyncio.run(measure(trace, driver, mode, emitter(mode, writer), workload,
                                           request_id))
    except PrefixCacheResetError as e:
        # Going on would measure the warm cache the reset exists to prevent.
        print(f"ABORT: prefix cache reset failed: {e}")
        stop(writer, None, "aborted", f"prefix cache reset failed: {e}", 1)
    except RunAborted as e:
        print(e)
        stop(writer, None, "aborted", str(e), 1)
    results, by_turn, pairs, duration, server_spec = measured

    core = aggregate(
        results=[r for r in results if r is not None], duration_s=duration,
        model=driver.model or "", profile=name, concurrency=mode.concurrency,
        load_mode=f"{mode.load}-loop",
        target_rate=offered_rate(workload.arrival) if mode.load == "open" else 0.0,
        warmup_concurrency=(mode.warmup_concurrency or mode.concurrency) if mode.warmup else 0)
    summary = Summary(core, results)
    if by_turn is not None:
        summary.turns = aggregate_per_turn(by_turn)
        print_multi_turn_summary(summary.turns, core)
    else:
        print_summary(core)
        enforce(transient_gate(len(trace.requests()), mode.concurrency), writer, summary)
    enforce(success_rate_gate(core, mode.min_success_rate), writer, summary)
    enforce(usage_gate(core, mode.allow_missing_usage), writer, summary)

    ident = summary.sections["workload"] = identity(trace, served(pairs))
    summary.sections.update({"goodput": goodput} if goodput else
                            crossval_sections(pairs, duration, mode, server_spec))
    print_legacy_lines(summary.sections)
    if not ident["match"]:
        stop(writer, summary.to_dict(), "failed", "workload identity", IDENTITY_EXIT)
    if goodput and goodput["max_req_s"] is None:
        stop(writer, summary.to_dict(), "aborted", "no passing speed within budget", 1)
    if core.failed_requests:
        summary.status = "partial"
        summary.reason = f"{core.failed_requests} of {core.num_requests} requests failed"
    writer.finish(summary.to_dict(), status=summary.status, reason=summary.reason)
    return summary


def model_dir(model):
    """engine.model when it is a local dir, else that Hugging Face repo id's snapshot in the local HF cache; never downloads."""
    if os.path.isdir(model):
        return model
    try:
        return snapshot_download(model, local_files_only=True)
    except (HFValidationError, LocalEntryNotFoundError):
        raise ConfigError(f"engine.model={model!r} is no local dir and not in the Hugging Face cache; "
                          f"download it (huggingface-cli download {model}) or set engine.model to "
                          "its snapshot dir") from None


def kv_bytes_per_token(engine):
    """engine.kv_bytes, else the dense formula over config.json, which counts every layer; hybrid and MLA stacks must set kv_bytes."""
    if engine.kv_bytes:
        return int(engine.kv_bytes)
    with open(os.path.join(model_dir(engine.model), "config.json")) as f:
        cfg = json.load(f)
    cfg = cfg.get("text_config", cfg)
    hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    hd += cfg.get("qk_rope_head_dim") or 0
    return 2 * cfg["num_hidden_layers"] * cfg["num_key_value_heads"] * hd * 2


def graph_mode(engine):
    return "FULL" if engine.cudagraph else "NOGRAPH"


def meta(cfg, gpu, version, eval_pos, steps):
    """META key -> value, in the grid log's order; expert_parallel only when on."""
    e, h = cfg.engine, cfg.hardware
    out = {"gpu": gpu, "vllm": version, "model": e.model_name, "dtype": e.dtype,
           "grid": cfg.mode.grid, "mode": graph_mode(e), "eval_pos": eval_pos,
           "step_points": list(steps),
           "nccl_algo": h.collective.get("NCCL_ALGO") or "auto",
           "nccl_proto": h.collective.get("NCCL_PROTO") or "auto",
           "link_profile": h.link_profile, "prompt_mode": cfg.workload.prompt_mode,
           "tp": e.tp, "pp": e.pp, "quant": e.quant or e.dtype}
    if e.ep:
        out["expert_parallel"] = 1
    return out


def capacity(engine, gpu_bytes, kv_b):
    """(free GiB for KV across tp GPUs, KV tokens that fit; 0 when none)."""
    gpu_gib = gpu_bytes / GIB
    free_gib = engine.tp * (gpu_gib * engine.gpu_util) - engine.weights_gib - engine.tp * RESERVE_GIB
    return free_gib, int(free_gib * GIB / kv_b) if free_gib > 0 else 0


def measure_cell(driver, writer, mode, steps, prompts, cell):
    """The cell's batches in order: text run when asked, warm run at max(steps), then the fit passes; returns the fit rows."""
    def batch(step, phase, attempt=None, **extra):
        requests = [Request(ids, len(ids), output_length=step) for ids in prompts]
        rows = list(driver.run(requests, cell={**cell, "step_point": step, "phase": phase,
                                               "attempt": attempt}))
        for r in rows:
            r.excluded_from_summary = phase != "fit"
            writer.append(r, **{k: f(r) for k, f in extra.items()})
        return rows

    if mode.returns == "text":
        # Routing-parity audit: greedy ids must match the engine's DS_TOKEN_TRACE.
        batch(TEXT_TOKENS, "text", token_ids=lambda r: r.generated_token_ids[:TEXT_TOKENS],
              text=lambda r: r.generated_text)
    if mode.returns == "logits":
        score(driver, writer, prompts, cell)
    batch(max(steps), "warm")
    out = []
    for attempt in range(mode.vmin_attempts):
        walls = []
        for step in steps:
            rows = batch(step, "fit", attempt)
            out += rows
            walls.append(rows[0].batch_wall_s)
        if fit(steps, walls)[2] >= mode.fit_r2_stop:
            break
    return out


def score(driver, writer, prompts, cell):
    """One scoring pass over the cell's prompts; rows carry logits, no time."""
    requests = [Request(ids, len(ids)) for ids in prompts]
    for r in driver.run(requests, cell={**cell, "step_point": None, "phase": "logits",
                                        "attempt": None}, returns="logits"):
        r.excluded_from_summary = True
        writer.append(r)


def score_grid(trace, driver, cfg, writer):
    """One scoring pass per cell for a score-only driver; maxlen SKIPs apply and nothing is timed."""
    record, over = {"events": []}, set(trace.cells_over_maxlen)
    for ctx, bs in trace.cells:
        if (ctx, bs) in over:
            event = {"event": "skip_maxlen", "ctx": ctx, "bs": bs}
            record["events"].append(event)
            print(grid_event_line(record, event), flush=True)
        else:
            score(driver, writer, trace.cell_prompts(ctx, bs)[0],
                  {"ctx": ctx, "bs": bs, "tp": cfg.engine.tp})
    return record, None, []


def run_bench(driver, cfg, writer):
    """driver.cells past the SKIP rules, measured in one driver.run call.

    A KV SKIP needs engine.kv_bytes, engine.weights_gib and driver.device()'s memory; a cell needs ctx + timing_steps + headroom tokens per sequence. steps passes the measured rows through.
    """
    engine, hw, grid = cfg.engine, cfg.hardware, cfg.grid
    device = driver.device() if engine.kv_bytes and engine.weights_gib is not None else None
    cap = capacity(engine, device[1], engine.kv_bytes)[1] if device and device[1] else 0
    over = set(grid.cells_over_maxlen)
    record, kept = {"bench_log": str(driver.log_path), "events": []}, []
    for ctx, bs, spec in driver.cells:
        need = (ctx + hw.timing_steps + hw.max_seq_headroom) * bs
        if (ctx, bs) in over:
            event = {"event": "skip_maxlen", "ctx": ctx, "bs": bs}
        elif cap and need > cfg.mode.cap_frac * cap:
            event = {"event": "skip_kv", "ctx": ctx, "bs": bs, "need": need, "cap": cap}
        else:
            kept.append((ctx, bs, spec))
            continue
        record["events"].append(event)
        print(grid_event_line(record, event), flush=True)
    rows = list(driver.run(kept, step_points=grid.step_points))
    for row in rows:
        writer.append(row)
    return record, fit_run(rows, None, None, None), rows


def run_grid(trace, driver, cfg, writer):
    """One grid run; returns (record, steps, rows): record feeds the legacy lines and run.json's grid section, steps is fit_run over the fit rows."""
    if getattr(driver, "measures_steps", False):
        return run_bench(driver, cfg, writer)
    if getattr(driver, "scores_only", False):
        return score_grid(trace, driver, cfg, writer)
    mode, engine = cfg.mode, cfg.engine
    eval_pos = mode.eval_pos if mode.eval_pos is not None else cfg.hardware.timing_steps // 2
    steps = trace.step_points
    kv_label = engine.kv_dtype if engine.kv_dtype != "auto" else engine.dtype
    gpu, gpu_bytes = driver.device()
    run = {"meta": meta(cfg, gpu, driver.version, eval_pos, steps), "kv": kv_label, "events": []}
    if trace.trace is not None:
        reqs = trace.trace
        run["trace"] = {"path": cfg.workload.source, "reqs": len(reqs.requests()),
                        "workload_hash": reqs.workload_hash()}
    shown = print_legacy_grid_lines(run)

    kv_b = kv_bytes_per_token(engine)
    # amd-smi, neuron-ls and tpu report no memory size, so those grids skip the KV SKIP.
    free_gib, cap = capacity(engine, gpu_bytes, kv_b) if gpu_bytes else (None, 0)
    run["capacity"] = {"mode": graph_mode(engine), "tp": engine.tp, "dtype": engine.dtype,
                       "kv_bytes": kv_b, "free_gib": free_gib, "kv_tokens": cap}
    shown = print_legacy_grid_lines(run, shown)
    driver.warmup()
    run["warm_ok"] = True
    shown = print_legacy_grid_lines(run, shown)

    over, top, rows = set(trace.cells_over_maxlen), max(steps), []
    for ctx, bs in trace.cells:
        need = (ctx + top) * bs
        if (ctx, bs) in over:
            run["events"].append({"event": "skip_maxlen", "ctx": ctx, "bs": bs})
        elif cap and need > mode.cap_frac * cap:
            run["events"].append({"event": "skip_kv", "ctx": ctx, "bs": bs, "need": need,
                                  "cap": cap})
        else:
            # Checked at the first cell that needs it, so an all-SKIP run never asks.
            if trace.prompt_mode == "corpus" and not trace.corpus_path:
                reason = "prompt_mode=corpus needs XVAL_CORPUS pointing at a text file"
                writer.finish(None, status="aborted", reason=reason)
                sys.exit(reason)
            prompts, cycled = trace.cell_prompts(ctx, bs)
            if cycled:
                run["events"].append({"event": "cycled", "ctx": ctx, "bs": bs})
                shown = print_legacy_grid_lines(run, shown)
            cell_rows = measure_cell(driver, writer, mode, steps, prompts,
                                     {"ctx": ctx, "bs": bs, "tp": engine.tp})
            rows += cell_rows
            (step,) = fit_run(cell_rows, eval_pos, kv_b, engine.weights_gib)
            run["events"].append({"event": "result", "ctx": ctx, "bs": bs, "step": step})
        shown = print_legacy_grid_lines(run, shown)
    run["done"] = True
    print_legacy_grid_lines(run, shown)
    return run, fit_run(rows, eval_pos, kv_b, engine.weights_gib), rows


class SessionDispatcher:
    """One multi-turn replay. send is the driver call with sampling bound; request_id(session, turn, index) names a request or returns None; emit takes every result as it finishes."""

    def __init__(self, trace, mode, workload, send, request_id, emit):
        self.trace, self.mode, self.workload = trace, mode, workload
        self.send, self.request_id, self.emit = send, request_id, emit
        self.sessions = trace.sessions()
        self.max_turns = getattr(trace, "max_turns", max(len(s) for s in self.sessions))
        self.pacing = trace.pacing
        self.open_loop = mode.load == "open"

    def announce(self):
        if not self.open_loop and self.pacing.pacing == "interleaved":
            print("WARNING: --turn-pacing interleaved is a turn-aligned herd "
                  "(global barrier per turn).")
            print("         TTFT/TPOT spikes at turn boundaries are schedule "
                  "artifacts, not production traffic.")
            print("         Default is per-session. Inter-turn waits are ignored "
                  "under interleaved.")
            if self.pacing.use_recorded_arrivals:
                print("WARNING: --use-recorded-arrivals is ignored under "
                      "--turn-pacing interleaved (everyone starts turn 0 together).")

    async def run(self):
        """(results in turn order, results_by_turn, [(request, result)] served, start)."""
        self.slots = asyncio.Semaphore(self.mode.concurrency)
        self.by_turn = {i: [] for i in range(self.max_turns)}
        self.served = []
        self.prev_context, self.prev_output = {}, {}
        self.dead, self.warned = set(), False
        self.start = time.perf_counter()
        if self.open_loop:
            await self._open()
        elif self.pacing.pacing == "per-session":
            await self._per_session()
        else:
            await self._interleaved()
        flat = []
        for turn in sorted(self.by_turn):
            for r in self.by_turn[turn]:
                if r.turn_index is None:
                    r.turn_index = turn
                flat.append(r)
        return flat, self.by_turn, self.served, self.start

    async def _dispatch(self, request, index, scheduled_at_s=None):
        sid, turn = request.session_id, request.turn

        def elapsed():
            return time.perf_counter() - self.start

        dispatched = elapsed()
        async with (no_limit() if self.open_loop else self.slots):
            granted = elapsed()
            result = await self.send(request, capture_text=True,
                                     request_id=self.request_id(sid, turn, index))
        annotate_request_observability(
            result, request_index=index, request=request, scheduled_at_s=scheduled_at_s,
            dispatch_started_at_s=dispatched, semaphore_acquired_at_s=granted,
            completed_at_s=elapsed())
        annotate_multi_turn_cache_estimate(
            result, session_id=sid, turn_index=turn,
            previous_context_tokens=self.prev_context.get(sid, 0),
            cache_block_size=self.mode.prefix_cache_block_size,
            previous_output_tokens=self.prev_output.get(sid, 0),
            # Placeholders mean the engine reply is spliced into later prompts.
            reply_in_cache=bool(self.trace.replies.get(sid)))
        self.served.append((request, result))
        self.emit(request, result)
        return result

    def _record(self, request, result):
        sid = request.session_id
        self.by_turn[request.turn].append(result)
        if result.success and result.input_tokens > 0:
            self.prev_context[sid] = int(result.input_tokens)
            self.prev_output[sid] = int(result.output_tokens or 0)
        if result.success:
            self._feed_reply_back(sid, request.turn, result)
        else:
            self.dead.add(sid)

    def _feed_reply_back(self, sid, turn, result):
        replies = self.trace.replies.get(sid)
        if not replies:
            if not self.warned:
                self.warned = True
                print("WARNING: this dataset has no assistant placeholders; "
                      "replies are not fed back (trace replay keeps recorded "
                      "assistant turns).", flush=True)
            return
        if result.generated_text and turn < len(replies):
            replies[turn]["content"] = result.generated_text

    def _wait_s(self, request):
        p = self.pacing
        return inter_turn_wait_s(request, tool_wait_ms=p.tool_wait_ms,
                                 human_wait_ms=p.human_wait_ms,
                                 use_recorded_waits=p.use_recorded_waits)

    def _print_waits(self):
        p = self.pacing
        if p.tool_wait_ms or p.human_wait_ms or p.use_recorded_waits:
            print(f"  Inter-turn waits: tool_wait_ms={p.tool_wait_ms:g} "
                  f"human_wait_ms={p.human_wait_ms:g} "
                  f"recorded={'on' if p.use_recorded_waits else 'off'}")

    async def _turns(self, turns, index_of, arrive_at_s=None):
        """One session's turns in order; stop at the first failure."""
        last = len(turns) - 1
        for request in turns:
            result = await self._dispatch(request, index_of(request),
                                          arrive_at_s if request.turn == 0 else None)
            self._record(request, result)
            if not result.success:
                break
            wait_s = self._wait_s(request)
            if wait_s > 0 and request.turn < last:
                await asyncio.sleep(wait_s)

    async def _open(self):
        n = len(self.sessions)
        if self.pacing.use_recorded_arrivals:
            arrivals = [s[0].arrival_ts or 0.0 for s in self.sessions]
            print(f"  Open loop: {n} sessions at recorded "
                  f"arrival_time_ms; turns sequential within a session, no barrier")
        else:
            spec = self.workload.arrival
            arrivals = [t / self.workload.speed
                        for t in parse_arrival(spec, n, self.workload.seed)]
            print(f"  Open loop: {n} sessions arriving at "
                  f"{offered_rate(spec)} sess/s ({spec.partition(':')[0]}); "
                  f"turns sequential within a session, no barrier")
        self._print_waits()
        counter = itertools.count()

        async def session(turns, arrive_at_s):
            await _sleep_until_elapsed(self.start, arrive_at_s)
            await self._turns(turns, lambda r: next(counter), arrive_at_s)

        await asyncio.gather(*[session(s, t) for s, t in zip(self.sessions, arrivals)])
        done = [r for v in self.by_turn.values() for r in v if r is not None]
        if done and not any(r.success for r in done):
            raise RunAborted(f"ABORT: all {len(done)} requests failed. First error: "
                             f"{first_error(done)}")

    async def _per_session(self):
        index = {(r.session_id, r.turn): i
                 for i, r in enumerate(r for s in self.sessions for r in s)}
        print(f"  Per-session pacing: {len(self.sessions)} sessions, "
              f"{len(index)} requests, up to {self.mode.concurrency} in flight "
              f"(no cross-session turn barrier)")
        self._print_waits()
        recorded = self.pacing.use_recorded_arrivals
        if recorded:
            print("  Session starts: recorded arrival_time_ms "
                  "(wait is outside the concurrency slot)")

        async def session(turns):
            if recorded:
                await _sleep_until_elapsed(self.start, turns[0].arrival_ts or 0.0)
            await self._turns(turns, lambda r: index[(r.session_id, r.turn)])

        await asyncio.gather(*[session(s) for s in self.sessions])
        results = [r for v in self.by_turn.values() for r in v]
        if results and not any(r is not None and r.success for r in results):
            raise RunAborted(f"ABORT: All {len(results)} requests failed. First error: "
                             f"{first_error(results)}")

    async def _interleaved(self):
        for turn in range(self.max_turns):
            batch = [s[turn] for s in self.sessions
                     if s[0].session_id not in self.dead and turn < len(s)]
            if not batch:
                continue
            print(f"  Turn {turn + 1}/{self.max_turns}: dispatching {len(batch)} requests...")
            offset = sum(len(v) for v in self.by_turn.values())
            completed = await asyncio.gather(*[
                self._dispatch(r, offset + i) for i, r in enumerate(batch)])
            if not any(r is not None and r.success for r in completed):
                raise RunAborted(f"ABORT: All {len(completed)} requests in turn {turn + 1} "
                                 f"failed. First error: {first_error(completed)}")
            for request, result in zip(batch, completed):
                self._record(request, result)
