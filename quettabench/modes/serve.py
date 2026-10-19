"""Send a Trace's requests through a Driver; one RequestResult row per request."""

import asyncio
import collections
import dataclasses
import gc
import hashlib
import itertools
import json
import os
import sys
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, LocalEntryNotFoundError

from ..config.schema import ConfigError, WorkloadCfg
from ..engines.drivers import PrefixCacheResetError
from ..metrics.results import goodput_probe_line, grid_event_line, print_grid_tag_lines, print_summary_tag_lines
from ..metrics.step_fit import GIB, fit, fit_run
from ..metrics.summary import (
    MIN_RECYCLED_WINDOW_S,
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
    steady_section,
    success_rate_gate,
    summarize_serve,
    transient_gate,
    usage_gate,
    withhold_latency,
)
from ..workloads.arrival import offered_rate, parse_arrival, rebase_arrivals
from ..workloads.base import Request
from ..workloads.plan import freeze, sent_digest, spliced_at
from ..workloads.sessions import PLAY_RULES, SessionTrace, inter_turn_wait_s, play, sleep_until_elapsed

TRACE_REQUEST_ID_PREFIX = "agenticbench"
IDENTITY_EXIT = 3
DEFAULT_WORKLOAD = WorkloadCfg()
RESERVE_GIB = 5.0  # Per GPU, held back from the KV budget for activations and graphs.
TEXT_TOKENS = 64


class RunAborted(Exception):
    """The message is the ABORT line."""


@asynccontextmanager
async def no_limit():
    yield


@dataclass
class Summary:
    core: BenchmarkSummary | None
    rows: list
    turns: list = field(default_factory=list)
    sections: dict = field(default_factory=dict)
    status: str = "ok"
    reason: str | None = None

    def to_dict(self):
        """aggregate and per_turn percentiles are linear; serve/sla/burst nearest-rank."""
        core = {**self.core.to_dict(), "percentile_method": "linear"} if self.core else {}
        out = {**core, **self.sections}
        if self.turns:
            out["per_turn"] = [{**t.to_dict(), "percentile_method": "linear"} for t in self.turns]
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
    """exact_output_length pins min_tokens; chunk_usage is passed only when on, for the drivers that take it."""
    out = {
        "ignore_eos": mode.ignore_eos or mode.exact_output_length,
        "temperature": mode.temperature,
        "seed": mode.sampling_seed,
        "min_tokens": (request.max_tokens or 0) if mode.exact_output_length else 0,
    }
    return {**out, "chunk_usage": True} if mode.chunk_usage else out


def stamp_arrivals(trace, workload):
    """Stamps an injected schedule on arrival_ts, before the plan is frozen, so it joins the workload hash; returns
    the injected times, or None when the trace keeps its own."""
    reqs = trace.requests()
    injected = parse_arrival(workload.arrival, len(reqs), workload.seed)
    for r, t in zip(reqs, injected or ()):
        r.arrival_ts = round(t, 6)
    return injected


def arrival_offsets(trace, workload) -> list:
    injected = stamp_arrivals(trace, workload)
    if injected is None:
        return rebase_arrivals([r.arrival_ts for r in trace.requests()], workload.speed)
    return [t / workload.speed for t in injected]


def enforce_first_chunk(result):
    """A reply with no chunk carrying choices has no TTFT and no token timing: a failed request, not one timed by
    its e2el."""
    if result.success and result.ttft is None:
        result.success = False
        result.error = "the reply carried no chunk with choices"
        result.error_kind = "empty_reply"


def enforce_output_length(mode, request, result):
    """exact_output_length: a reply shorter than planned is a failed request, not a faster one."""
    if mode.exact_output_length and result.success and result.output_tokens != request.max_tokens:
        result.success = False
        result.error = f"{result.output_tokens} output tokens, {request.max_tokens} planned"
        result.error_kind = "short_output"


async def warmup(trace, driver, mode, count, multi_turn):
    """So connections exist before timing; abort when none succeeds."""
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

    results = await asyncio.gather(*(one(r) for r in trace.warmup_requests(count)), return_exceptions=True)
    ok = sum(1 for r in results if not isinstance(r, BaseException) and r is not None and r.success)
    print(f"Warmup done: {ok}/{len(results)} succeeded.")
    if ok == 0:
        raise RunAborted(
            "ABORT: every warmup request failed. The server is not serving "
            "this profile; check the server log before benchmarking."
        )


def emitter(mode, writer, probe=None):
    basis = "arrival" if mode.load == "open" else "slot"
    tag = {} if probe is None else {"probe": probe}

    def emit(request, result):
        result.ttft_basis = basis
        if request.output_token_ids and result.generated_token_ids is not None:
            result.accepted_tokens = acceptance(request.output_token_ids, result.generated_token_ids)[0]
        writer.append(result, **tag)

    return emit


async def replay(trace, driver, mode, emit, offsets, request_id, executed):
    """The first mode.discard_first requests stay out of the summary. executed gets each request's sent_digest just
    before it goes out."""
    open_loop = mode.load == "open"
    slots = asyncio.Semaphore(mode.concurrency)
    reqs = trace.requests()
    rows: list = [None] * len(reqs)
    start = time.perf_counter()

    async def dispatch(i, request, at):
        delay = at - (time.perf_counter() - start)
        if delay > 0:
            await asyncio.sleep(delay)
        dispatched = time.perf_counter() - start
        async with no_limit() if open_loop else slots:
            granted = time.perf_counter() - start
            executed[i] = sent_digest(request)
            result = await driver.send(request, request_id=request_id(i, 0, i), **sampling(mode, request))
        annotate_request_observability(
            result,
            request_index=i,
            request=request,
            scheduled_at_s=at,
            dispatch_started_at_s=dispatched,
            semaphore_acquired_at_s=granted,
            completed_at_s=time.perf_counter() - start,
            run_start=start,
        )
        enforce_first_chunk(result)
        enforce_output_length(mode, request, result)
        if i < mode.discard_first:
            result.excluded_from_summary = True
        emit(request, result)
        rows[i] = (request, result)

    await asyncio.gather(*(dispatch(i, r, t) for i, (r, t) in enumerate(zip(reqs, offsets))))
    fail = sum(1 for _, r in rows if not r.success)
    if fail > 0 and fail >= len(reqs) * 0.9:
        raise RunAborted(
            f"ABORT: {fail}/{len(reqs)} requests failed "
            f"({fail / len(reqs) * 100:.0f}%). "
            f"Server may not be functional. Check server logs."
        )
    return rows, start


def server_counters(driver):
    counters = getattr(driver, "spec_decode_counters", None)
    return counters() if counters else None


LOOP_LAG_INTERVAL_S = 0.01
# Steady TTFT is withheld above this window p99 loop lag: every timestamp in the window is read off that loop.
LOOP_LAG_LIMIT_MS = 5.0


def loop_lag_ms(samples) -> dict | None:
    """p50, p99 and max of (time, lag) samples in ms, each weighted by the wall time it spans (the interval plus its
    lag), so a 1 s stall weighs 1 s rather than one sample."""
    if not samples:
        return None
    spans = sorted((lag, LOOP_LAG_INTERVAL_S + lag) for _, lag in samples)
    total = sum(w for _, w in spans)

    def at(q):
        acc = 0.0
        for lag, w in spans:
            acc += w
            if acc >= q * total:
                return lag * 1000
        return spans[-1][0] * 1000

    return {"50": at(0.5), "99": at(0.99), "max": spans[-1][0] * 1000}


class ClientHealth:
    """Samples how late the event loop runs and times gc passes while a run measures: start() before the pass,
    stop() after, section() for run.json. samples holds (perf_counter, lag in s) for slicing to a window."""

    def __init__(self):
        self.samples: list[tuple[float, float]] = []
        self.gc_s: list[tuple[int, float]] = []
        self._gc_start: float | None = None
        self._task: asyncio.Task | None = None

    def _gc(self, phase, info):
        if phase == "start":
            self._gc_start = time.perf_counter()
        elif self._gc_start is not None:
            self.gc_s.append((info["generation"], time.perf_counter() - self._gc_start))
            self._gc_start = None

    async def _sample(self):
        loop = asyncio.get_running_loop()
        while True:
            t = loop.time()
            await asyncio.sleep(LOOP_LAG_INTERVAL_S)
            self.samples.append((time.perf_counter(), loop.time() - t - LOOP_LAG_INTERVAL_S))

    def start(self):
        gc.callbacks.append(self._gc)
        self._task = asyncio.get_running_loop().create_task(self._sample())

    async def stop(self):
        gc.callbacks.remove(self._gc)
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task

    def section(self) -> dict:
        out: dict = {"loop_lag_samples": len(self.samples), "loop_lag_ms": loop_lag_ms(self.samples)}
        pauses = [s * 1000 for _, s in self.gc_s]
        out.update(
            gc_passes=len(pauses),
            gc_gen2_passes=sum(1 for g, _ in self.gc_s if g == 2),
            gc_pause_ms={"max": max(pauses, default=0.0), "total": sum(pauses)},
        )
        return out


def client_line(c: dict) -> str:
    lag = c["loop_lag_ms"] or {"99": 0.0, "max": 0.0}
    return (
        f"Client: event loop late by {lag['99']:.1f} ms at p99, {lag['max']:.1f} ms at most; "
        f"{c['gc_passes']} gc passes ({c['gc_gen2_passes']} full), longest {c['gc_pause_ms']['max']:.1f} ms"
    )


@dataclass
class Measurement:
    """What one measured pass produced, before any gate or summary reads it."""

    results: list
    by_turn: dict | None
    pairs: list
    duration: float
    spec_decode: dict | None
    session_times: dict | None = None
    client: dict | None = None
    # Recycled replay: the plays started, each a salted copy of a session's turns, and the deadline.
    plays: list | None = None
    deadline_s: float | None = None
    # (seconds since the run's start, loop lag in s), for the steady window's slice.
    loop_lag: list | None = None
    # Each request's sent_digest at dispatch, keyed as the plan keys it, in the order they went out.
    executed: dict = field(default_factory=dict)


async def measure(trace, driver, mode, emit, workload, request_id) -> Measurement:
    multi = isinstance(trace, SessionTrace)
    executed: dict = {}
    offsets = None if multi else arrival_offsets(trace, workload)
    sessions: SessionDispatcher | None = None
    if multi:
        sessions = SessionDispatcher(
            trace, mode, workload, lambda r, **kw: driver.send(r, **kw, **sampling(mode, r)), request_id, emit
        )
        sessions.announce()
    async with driver.session(mode.concurrency, mode.load == "open"):
        if mode.warmup > 0:
            await warmup(trace, driver, mode, mode.warmup, multi)
        # After warmup, so the run starts cold whatever warmup left behind.
        if mode.reset_prefix_cache:
            print(f"Prefix cache: {await driver.reset()}")
        before = server_counters(driver)
        health = ClientHealth()
        health.start()
        try:
            if sessions is not None:
                results, by_turn, pairs, start = await sessions.run(executed)
            else:
                pairs, start = await replay(trace, driver, mode, emit, offsets, request_id, executed)
                results, by_turn = [r for _, r in pairs], None
            duration = time.perf_counter() - start
        finally:
            await health.stop()
        spec = server_spec_decode_section(before, server_counters(driver) if before else None)
    if sessions is None:
        return Measurement(results, by_turn, pairs, duration, spec, client=health.section(), executed=executed)
    return Measurement(
        results,
        by_turn,
        pairs,
        duration,
        spec,
        sessions.session_times,
        health.section(),
        sessions.plays or None,
        sessions.deadline_s,
        [(t - start, lag) for t, lag in health.samples],
        executed,
    )


def withhold_on_client_lag(steady: dict, loop_lag: list) -> None:
    """Adds the loop lag inside a valid steady window and withholds its TTFT when p99 is over LOOP_LAG_LIMIT_MS: a
    late loop reads every first chunk late."""
    inside = [(t, lag) for t, lag in loop_lag if steady["start_s"] <= t <= steady["end_s"]]
    lag = steady["client_loop_lag_ms"] = loop_lag_ms(inside)
    steady["client_loop_lag_samples"] = len(inside)
    if lag and lag["99"] > LOOP_LAG_LIMIT_MS:
        steady["ttft_ms"] = None
        steady["withheld"] = {
            "ttft_ms": f"client event loop late by {lag['99']:.1f} ms at p99 in the window, over {LOOP_LAG_LIMIT_MS:g} ms"
        }


def steady_line(s: dict) -> str:
    if not s["valid"]:
        return f"Steady window: none ({s['reason']}); whole-run numbers include ramp and drain"
    tl = s["timeline"]["output_tok_s"]
    withheld = "".join(f"; {k} withheld: {why}" for k, why in s.get("withheld", {}).items())
    return (
        f"Steady window: {s['start_s']:.1f} s to {s['end_s']:.1f} s of the run, all {s['agents']} agents busy: "
        f"{s['output_token_throughput']:.1f} out tok/s (per-bin min {tl['min']:.1f}, max {tl['max']:.1f}), "
        f"{s['sessions_per_hour']:.1f} sessions/h, {s['requests_inside']} requests fully inside{withheld}"
    )


def cold_start_line(s: dict) -> str | None:
    """The cold-start TTFT beside the steady window: each agent's first turn, then everything before the window."""
    parts = []
    for label, t in (
        ("first turns", s.get("first_turn_ttft_ms")),
        ("before the window", (s.get("ramp") or {}).get("ttft_ms")),
    ):
        if t:
            parts.append(f"{label} p50 {t[50]:.1f} ms, p99 {t[99]:.1f} ms (n={t['n']})")
    return f"Cold start TTFT: {'; '.join(parts)}" if parts else None


def serve_records(pairs, open_loop):
    out = []
    for _, r in pairs:
        if not r.success or r.excluded_from_summary:
            continue
        n = r.output_tokens
        arrival = r.scheduled_at_s if open_loop and r.scheduled_at_s is not None else r.semaphore_acquired_at_s
        out.append(
            {
                "ttft_ms": r.ttft * 1e3,
                "index": r.request_index,
                "out_tokens": n,
                "arrival_s": arrival,
                "tpot_ms": (r.e2el - r.ttft) * 1e3 / (n - 1) if r.ttft and n > 1 else None,
            }
        )
    return out


def spec_decode_section(pairs):
    scored = [(r.accepted_tokens, len(q.output_token_ids)) for q, r in pairs if r.accepted_tokens is not None]
    if not scored:
        return {"spec": "off"}
    return {
        "spec": "measured",
        "n": len(scored),
        "accept_len": sum(a for a, _ in scored) / len(scored),
        "accept_rate": sum(a / n for a, n in scored) / len(scored),
    }


def server_spec_decode_section(before, after):
    """As vllm bench serve defines them; None without both scrapes or a draft in the window."""
    if before is None or after is None:
        return None
    d = {key: int(after[key] - before[key]) for key in after}
    if d["drafts"] <= 0 or d["draft_tokens"] <= 0:
        return None
    return {
        "spec": "measured",
        "source": "server_metrics",
        **d,
        "accept_rate": d["accepted_tokens"] / d["draft_tokens"],
        "accept_len": 1 + d["accepted_tokens"] / d["drafts"],
        "n": d["drafts"],
    }


def failed_count(pairs) -> int:
    """Failed requests among those the summary keeps."""
    return sum(1 for _, r in pairs if not r.success and not r.excluded_from_summary)


def summary_sections(pairs, duration, mode, server_spec_decode=None, sessions=False, withheld=False):
    """spec prefers the server's counters over the rows'. sessions: a session run, whose steady TTFT is
    sections.steady's, not the requests past the first C indexes. withheld: a run with failed requests, whose
    latency percentiles and burst recovery are left out."""
    out = {"spec_decode": server_spec_decode or spec_decode_section(pairs)}
    records = serve_records(pairs, mode.load == "open")
    if not records:
        return out
    # In a session run C caps agents, not requests in flight, and request indexes are not dispatch order.
    closed = mode.concurrency if mode.load == "closed" and not sessions else 0
    s = out["serve"] = summarize_serve(records, duration, "nearest", closed)
    if withheld:
        s.update(ttft_ms=None, tpot_ms=None, **({"steady_ttft_ms": None} if "steady_ttft_ms" in s else {}))
    if mode.sla_ttft_ms or mode.sla_tpot_ms:
        failed = failed_count(pairs)
        bound = sla_verdict(s, mode.sla_ttft_ms, mode.sla_tpot_ms, failed)
        out["sla"] = {
            "verdict": "FAIL" if bound else "PASS",
            "bound": bound,
            "failed_requests": failed,
            "ttft_p99": None if failed else s["ttft_ms"]["p99"],
            "tpot_p99": s["tpot_ms"]["p99"] if s["tpot_ms"] and not failed else None,
            "rate_req_s": s["req_s"],
            "percentile_method": s["percentile_method"],
            "sla_ttft_ms": mode.sla_ttft_ms,
            "sla_tpot_ms": mode.sla_tpot_ms,
        }
    if mode.sla_ttft_ms and not withheld:
        recovery, peak = burst_recovery(records, mode.sla_ttft_ms)
        out["burst"] = {
            "peak_ttft_p99": peak,
            "recovery_s": recovery,
            "target_ms": mode.sla_ttft_ms,
            "percentile_method": "nearest",
        }
    return out


def goodput_search(start=1.0, max_iters=6, tol=0.1):
    """Generator bisection over speed: send() whether the last probe met the SLA."""
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


def sweep(trace, driver, mode, writer, workload, request_id, plan):
    """Stops at the first probe that breaks workload identity."""
    gen, best, best_out = goodput_search(start=workload.speed), None, None
    probes: list = []
    fail_bound: list = []
    speed = next(gen)
    while True:
        probe_mode = mode if not probes else dataclasses.replace(mode, warmup=0)
        out = asyncio.run(
            measure(
                trace,
                driver,
                probe_mode,
                emitter(mode, writer, len(probes)),
                dataclasses.replace(workload, speed=speed),
                request_id,
            )
        )
        section = {"max_req_s": None, "speed": None, "fail_bound": fail_bound, "probes": probes}
        if not identity(plan, out.executed, plan.planned_keys(out.executed))["match"]:
            return out, {**section, "iterations": len(probes) + 1}
        s = summarize_serve(serve_records(out.pairs, True), out.duration, "nearest")
        bound = sla_verdict(s, mode.sla_ttft_ms, mode.sla_tpot_ms, failed_count(out.pairs))
        probe = {"speed": speed, "verdict": "FAIL" if bound else "PASS", "bound": bound, "req_s": s["req_s"]}
        print(goodput_probe_line(probe), flush=True)
        probes.append(probe)
        if not bound and (best is None or s["req_s"] > best["req_s"]):
            best, best_out = probe, out
        if bound:
            fail_bound = bound
        try:
            speed = gen.send(not bound)
        except StopIteration:
            break
    # The best passing probe's own requests stand for the run; with none, the last probe's.
    return best_out or out, {
        "max_req_s": best and best["req_s"],
        "speed": best and best["speed"],
        "fail_bound": fail_bound,
        "iterations": len(probes),
        "probes": probes,
    }


def stop(writer, summary, status, reason, code):
    writer.finish(summary, status=status, reason=reason)
    sys.exit(code)


def enforce(gate, writer, summary):
    verdict, message = gate
    if message:
        print(message)
    if verdict == "abort":
        stop(writer, summary.to_dict(), "aborted", message, 1)


def turns_not_sent(planned, executed, pairs) -> list:
    """The planned (session, turn) keys after each session's first failed turn: a failed turn ends its session, so
    these were never sent."""
    failed_at: dict = {}
    for q, r in pairs:
        if not r.success:
            failed_at[q.session_id] = min(q.turn, failed_at.get(q.session_id, q.turn))
    return [k for k in planned if k not in executed and k[0] in failed_at and k[1] > failed_at[k[0]]]


def key_row(key) -> dict:
    return {"session": key[0], "turn": key[1]} if isinstance(key, tuple) else {"request": key}


def digests_hash(digests) -> str:
    """One hash over request digests, whatever their order."""
    h = hashlib.sha256()
    for d in sorted(digests):
        h.update(d.encode())
    return h.hexdigest()[:16]


def identity(plan, executed, planned, not_sent=()) -> dict:
    """Checks every request as it was sent (executed: its digest at dispatch) against the plan frozen before the
    run. Turns a failed turn left unsent (not_sent) are listed, not mismatches; a request that differs, was not
    planned or was never sent fails the check, and the first one is named. plan_hash is the hash runs compare on."""
    skipped = set(not_sent)
    mismatch = None
    for key, digest in executed.items():
        want = plan.digest(key)
        if want != digest:
            mismatch = {**key_row(key), "reason": "not in the plan" if want is None else "differs from the plan"}
            break
    missing = next((k for k in planned if k not in executed and k not in skipped), None)
    if mismatch is None and missing is not None:
        mismatch = {**key_row(missing), "reason": "planned but never sent"}
    out = {
        "plan_hash": plan.plan_hash,
        "expected": digests_hash(plan.digest(k) for k in planned if k not in skipped),
        "executed_hash": digests_hash(executed.values()),
        "match": mismatch is None,
        "n": len(executed),
        "of": len(planned),
    }
    if mismatch:
        out["first_mismatch"] = mismatch
    if not_sent:
        out["turns_not_sent"] = [key_row(k) for k in not_sent]
    return out


def run(trace, driver, mode, writer, workload=DEFAULT_WORKLOAD) -> Summary:
    if mode.grid:
        record, steps, rows = run_grid(trace, driver, writer.cfg, writer)
        summary = Summary(None, rows, sections={"grid": record})
        writer.finish(summary.to_dict(), steps=steps)
        return summary
    name = trace_name(trace)

    def request_id(session, turn, index):
        if not mode.trace_request_ids:
            return None
        return make_trace_request_id(
            profile_name=name, concurrency=mode.concurrency, session_id=session, turn_index=turn, request_index=index
        )

    multi = isinstance(trace, SessionTrace)
    if not multi:
        stamp_arrivals(trace, workload)
    # Frozen before anything is sent: the check below compares each request as sent with this, not with the
    # Request objects the run itself sends.
    plan = freeze(trace, recycled=multi and mode.duration_s > 0, seed=workload.seed, salt=workload.salt)
    goodput = None
    try:
        if mode.goodput_sweep:
            measured, goodput = sweep(trace, driver, mode, writer, workload, request_id, plan)
        else:
            measured = asyncio.run(measure(trace, driver, mode, emitter(mode, writer), workload, request_id))
    except PrefixCacheResetError as e:
        # Going on would measure the warm cache the reset exists to prevent.
        print(f"ABORT: prefix cache reset failed: {e}")
        stop(writer, None, "aborted", f"prefix cache reset failed: {e}", 1)
    except RunAborted as e:
        print(e)
        stop(writer, None, "aborted", str(e), 1)
    results, by_turn, pairs, duration = measured.results, measured.by_turn, measured.pairs, measured.duration
    server_spec_decode, session_times, client = measured.spec_decode, measured.session_times, measured.client
    plays = measured.plays
    if plays:
        planned_turns = {p[0].session_id: len(p) for p in plays}
    elif multi:
        planned_turns = {s[0].session_id: len(s) for s in trace.sessions()}
    else:
        planned_turns = None
    planned = plan.planned_keys(measured.executed)
    not_sent = turns_not_sent(planned, measured.executed, pairs) if multi else []

    core = aggregate(
        results=[r for r in results if r is not None],
        duration_s=duration,
        model=driver.model or "",
        profile=name,
        concurrency=mode.concurrency,
        load_mode=f"{mode.load}-loop",
        target_rate=offered_rate(workload.arrival) if mode.load == "open" else 0.0,
        warmup_concurrency=(mode.warmup_concurrency or mode.concurrency) if mode.warmup else 0,
        planned_turns=planned_turns,
    )
    summary = Summary(core, results)
    failed = core.failed_requests + len(not_sent)
    withheld = f"{failed} requests failed or went unsent; the rest leave out the slowest" if failed else None
    if by_turn is not None:
        summary.turns = aggregate_per_turn(by_turn)
    if withheld:
        withhold_latency(core, *summary.turns)
        summary.sections["withheld"] = {"latency_percentiles": withheld}
    if by_turn is not None:
        print_multi_turn_summary(summary.turns, core)
    else:
        print_summary(core)
        enforce(transient_gate(len(trace.requests()), mode.concurrency), writer, summary)
    enforce(success_rate_gate(core, mode.min_success_rate, len(not_sent)), writer, summary)
    enforce(usage_gate(core, mode.allow_missing_usage), writer, summary)

    ident = summary.sections["workload"] = identity(plan, measured.executed, planned, not_sent)
    if plays:
        stratum_of = getattr(trace, "stratum_of", None)
        summary.sections["plays"] = {
            "sessions": len(trace.sessions()),
            "plays": len(plays),
            "duration_s": measured.deadline_s,
            "salt": workload.salt,
            "play_rule": PLAY_RULES[workload.salt],
            # Plays per stratum: a stratified order keeps these within one of each other per pass.
            "plays_by_stratum": dict(
                collections.Counter(stratum_of[int(str(p[0].session_id).split("#")[0])] for p in plays)
            )
            if stratum_of
            else None,
        }
    # What was replayed, for a reader of run.json alone: the file and its digest, and the sessions in order.
    manifest = getattr(trace, "manifest", None)
    summary.sections["trace"] = {
        **trace.source_meta(),
        "sample": list(getattr(trace, "sample", [])),
        "manifest": manifest,
    }
    if manifest and manifest.get("comparable") is False:
        print(f"WARNING: {manifest['file']} is not the file data/MANIFEST.json lists; this run is not comparable")
    summary.sections.update(
        {"goodput": goodput}
        if goodput
        else summary_sections(pairs, duration, mode, server_spec_decode, multi, withheld is not None)
    )
    print_summary_tag_lines(summary.sections)
    if client is not None:
        summary.sections["client"] = client
        print(client_line(client), flush=True)
    if session_times is not None and mode.load == "closed":
        steady = summary.sections["steady"] = steady_section(
            results, session_times, mode.concurrency, end_s=measured.deadline_s, min_window_s=MIN_RECYCLED_WINDOW_S
        )
        if steady["valid"] and measured.loop_lag is not None:
            withhold_on_client_lag(steady, measured.loop_lag)
        if withheld:
            for key in ("ttft_ms", "tpot_ms", "chunk_gap_ms", "interactivity_p90_tok_s", "first_turn_ttft_ms", "ramp"):
                if key in steady:
                    steady[key] = None
            steady.setdefault("withheld", {})["latency"] = withheld
        print(steady_line(steady), flush=True)
        if cold := cold_start_line(steady):
            print(cold, flush=True)
    if not ident["match"]:
        stop(writer, summary.to_dict(), "failed", "workload identity", IDENTITY_EXIT)
    if goodput and goodput["max_req_s"] is None:
        stop(writer, summary.to_dict(), "aborted", "no passing speed within budget", 1)
    if core.failed_requests:
        summary.status = "partial"
        summary.reason = f"{core.failed_requests} of {core.num_requests} requests failed"
        if not_sent:
            summary.reason += f"; {len(not_sent)} later turns of their sessions not sent"
    writer.finish(summary.to_dict(), status=summary.status, reason=summary.reason)
    return summary


# The files that name a local checkpoint's weights: its architecture and its shard map.
WEIGHT_FILES = ("config.json", "model.safetensors.index.json")


def weights_revision(model, revision=None, cache_dir=None) -> tuple[str | None, str | None]:
    """The weights a server loading model resolves to now, and why when that cannot be told. A local dir is a sha256
    over WEIGHT_FILES; a Hugging Face id is the snapshot cached under cache_dir (hardware.hf_home) for revision
    (extra.vllm.revision), refs/main without one, which another tool's download can move."""
    if os.path.isdir(model):
        files = [f for f in WEIGHT_FILES if os.path.isfile(os.path.join(model, f))]
        if not files:
            return None, f"{model} has neither {' nor '.join(WEIGHT_FILES)} to hash"
        h = hashlib.sha256()
        for f in files:
            h.update(f.encode() + b"\0" + Path(model, f).read_bytes())
        return f"sha256:{h.hexdigest()[:16]}", None
    try:
        return os.path.basename(
            snapshot_download(model, revision=revision, cache_dir=cache_dir, local_files_only=True)
        ), None
    except HFValidationError:
        return None, f"{model} is neither a directory nor a Hugging Face id"
    except LocalEntryNotFoundError:
        at = f"@{revision}" if revision else ""
        return None, f"{model}{at} is not in the Hugging Face cache {cache_dir or '(the default)'}"


def model_dir(model):
    """Never downloads."""
    if os.path.isdir(model):
        return model
    try:
        return snapshot_download(model, local_files_only=True)
    except (HFValidationError, LocalEntryNotFoundError):
        raise ConfigError(
            f"engine.model={model!r} is no local dir and not in the Hugging Face cache; "
            f"download it (huggingface-cli download {model}) or set engine.model to "
            "its snapshot dir"
        ) from None


def resolve_kv_bytes_per_token(engine):
    """Hybrid and MLA stacks must set kv_bytes_per_token."""
    if engine.kv_bytes_per_token:
        return int(engine.kv_bytes_per_token)
    with open(os.path.join(model_dir(engine.model), "config.json")) as f:
        cfg = json.load(f)
    cfg = cfg.get("text_config", cfg)
    hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    hd += cfg.get("qk_rope_head_dim") or 0
    return 2 * cfg["num_hidden_layers"] * cfg["num_key_value_heads"] * hd * 2


def graph_mode(engine):
    return "FULL" if engine.cudagraph else "NOGRAPH"


def meta(cfg, gpu, version, eval_pos, steps):
    """expert_parallel only when on."""
    e, h = cfg.engine, cfg.hardware
    out = {
        "gpu": gpu,
        "vllm": version,
        "model": e.model_name,
        "dtype": e.dtype,
        "grid": cfg.mode.grid,
        "mode": graph_mode(e),
        "eval_pos": eval_pos,
        "step_points": list(steps),
        "nccl_algo": h.collective.get("NCCL_ALGO") or "auto",
        "nccl_proto": h.collective.get("NCCL_PROTO") or "auto",
        "link_profile": h.link_profile,
        "prompt_mode": cfg.workload.prompt_mode,
        "tp": e.tp,
        "pp": e.pp,
        "quant": e.quant or e.dtype,
    }
    if e.ep:
        out["expert_parallel"] = 1
    return out


def capacity(engine, gpu_bytes, kv_bytes_per_token):
    gpu_gib = gpu_bytes / GIB
    free_gib = engine.tp * (gpu_gib * engine.gpu_util) - engine.weights_gib - engine.tp * RESERVE_GIB
    return free_gib, int(free_gib * GIB / kv_bytes_per_token) if free_gib > 0 else 0


def measure_cell(driver, writer, mode, steps, prompts, cell):

    def batch(step, phase, attempt=None, **extra):
        requests = [Request(ids, len(ids), output_length=step) for ids in prompts]
        rows = list(driver.run(requests, cell={**cell, "step_point": step, "phase": phase, "attempt": attempt}))
        for r in rows:
            r.excluded_from_summary = phase != "fit"
            writer.append(r, **{k: f(r) for k, f in extra.items()})
        return rows

    if mode.return_outputs == "text":
        # Routing-parity audit: greedy ids must match the engine's DS_TOKEN_TRACE.
        batch(
            TEXT_TOKENS,
            "text",
            token_ids=lambda r: r.generated_token_ids[:TEXT_TOKENS],
            text=lambda r: r.generated_text,
        )
    if mode.return_outputs == "logits":
        score(driver, writer, prompts, cell)
    batch(max(steps), "warm")
    out = []
    for attempt in range(mode.fit_attempts):
        walls = []
        for step in steps:
            rows = batch(step, "fit", attempt)
            out += rows
            walls.append(rows[0].batch_wall_s)
        if fit(steps, walls)[2] >= mode.fit_r2_stop:
            break
    return out


def score(driver, writer, prompts, cell):
    requests = [Request(ids, len(ids)) for ids in prompts]
    for r in driver.run(
        requests, cell={**cell, "step_point": None, "phase": "logits", "attempt": None}, return_outputs="logits"
    ):
        r.excluded_from_summary = True
        writer.append(r)


def score_grid(trace, driver, cfg, writer):
    """maxlen SKIPs apply and nothing is timed."""
    record: dict = {"events": []}
    over = set(trace.cells_over_maxlen)
    for ctx, bs in trace.cells:
        if (ctx, bs) in over:
            event = {"event": "skip_maxlen", "ctx": ctx, "bs": bs}
            record["events"].append(event)
            print(grid_event_line(record, event), flush=True)
        else:
            score(driver, writer, trace.cell_prompts(ctx, bs)[0], {"ctx": ctx, "bs": bs, "tp": cfg.engine.tp})
    return record, None, []


def run_bench(driver, cfg, writer):
    """A KV SKIP needs engine.kv_bytes_per_token, engine.weights_gib and device memory."""
    engine, hw, grid = cfg.engine, cfg.hardware, cfg.grid
    device = driver.device() if engine.kv_bytes_per_token and engine.weights_gib is not None else None
    cap = capacity(engine, device[1], engine.kv_bytes_per_token)[1] if device and device[1] else 0
    over = set(grid.cells_over_maxlen)
    record: dict = {"bench_log": str(driver.log_path), "events": []}
    kept: list = []
    for ctx, bs, spec in driver.cells:
        need = (ctx + hw.timing_steps + hw.max_seq_headroom) * bs
        if (ctx, bs) in over:
            event = {"event": "skip_maxlen", "ctx": ctx, "bs": bs}
        elif cap and need > cfg.mode.max_kv_fraction * cap:
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
    """record feeds the grid stdout lines and run.json; steps is fit_run over fit rows."""
    if getattr(driver, "measures_steps", False):
        return run_bench(driver, cfg, writer)
    if getattr(driver, "scores_only", False):
        return score_grid(trace, driver, cfg, writer)
    mode, engine = cfg.mode, cfg.engine
    eval_pos = mode.eval_decode_pos if mode.eval_decode_pos is not None else cfg.hardware.timing_steps // 2
    steps = trace.step_points
    kv_label = engine.kv_cache_dtype if engine.kv_cache_dtype != "auto" else engine.dtype
    gpu, gpu_bytes = driver.device()
    run = {"meta": meta(cfg, gpu, driver.version, eval_pos, steps), "kv": kv_label, "events": []}
    if trace.trace is not None:
        reqs = trace.trace
        run["trace"] = {
            "path": cfg.workload.source,
            "reqs": len(reqs.requests()),
            "workload_hash": reqs.workload_hash(),
        }
    shown = print_grid_tag_lines(run)

    kv_bytes_per_token = resolve_kv_bytes_per_token(engine)
    # amd-smi, neuron-ls and tpu report no memory size, so those grids skip the KV SKIP.
    free_gib, cap = capacity(engine, gpu_bytes, kv_bytes_per_token) if gpu_bytes else (None, 0)
    run["capacity"] = {
        "mode": graph_mode(engine),
        "tp": engine.tp,
        "dtype": engine.dtype,
        "kv_bytes_per_token": kv_bytes_per_token,
        "free_gib": free_gib,
        "kv_tokens": cap,
    }
    shown = print_grid_tag_lines(run, shown)
    driver.warmup()
    run["warm_ok"] = True
    shown = print_grid_tag_lines(run, shown)

    over, top, rows = set(trace.cells_over_maxlen), max(steps), []
    for ctx, bs in trace.cells:
        need = (ctx + top) * bs
        if (ctx, bs) in over:
            run["events"].append({"event": "skip_maxlen", "ctx": ctx, "bs": bs})
        elif cap and need > mode.max_kv_fraction * cap:
            run["events"].append({"event": "skip_kv", "ctx": ctx, "bs": bs, "need": need, "cap": cap})
        else:
            # Checked at the first cell that needs it, so an all-SKIP run never asks.
            if trace.prompt_mode == "corpus" and not trace.corpus_path:
                reason = "prompt_mode=corpus needs QBENCH_CORPUS pointing at a text file"
                writer.finish(None, status="aborted", reason=reason)
                sys.exit(reason)
            prompts, cycled = trace.cell_prompts(ctx, bs)
            if cycled:
                run["events"].append({"event": "cycled", "ctx": ctx, "bs": bs})
                shown = print_grid_tag_lines(run, shown)
            cell_rows = measure_cell(driver, writer, mode, steps, prompts, {"ctx": ctx, "bs": bs, "tp": engine.tp})
            rows += cell_rows
            (step,) = fit_run(cell_rows, eval_pos, kv_bytes_per_token, engine.weights_gib)
            run["events"].append({"event": "result", "ctx": ctx, "bs": bs, "step": step})
        shown = print_grid_tag_lines(run, shown)
    run["done"] = True
    print_grid_tag_lines(run, shown)
    return run, fit_run(rows, eval_pos, kv_bytes_per_token, engine.weights_gib), rows


class SessionDispatcher:
    """request_id names a request or returns None; emit takes each result."""

    def __init__(self, trace, mode, workload, send, request_id, emit):
        self.trace, self.mode, self.workload = trace, mode, workload
        self.send, self.request_id, self.emit = send, request_id, emit
        self.sessions = trace.sessions()
        self.max_turns = getattr(trace, "max_turns", max(len(s) for s in self.sessions))
        self.pacing = trace.pacing
        self.open_loop = mode.load == "open"

    def announce(self):
        if not self.open_loop and self.pacing.pacing == "interleaved":
            print("WARNING: --turn-pacing interleaved is a turn-aligned herd (global barrier per turn).")
            print("         TTFT/TPOT spikes at turn boundaries are schedule artifacts, not production traffic.")
            print("         Default is per-session. Inter-turn waits are ignored under interleaved.")
            if self.pacing.use_recorded_arrivals:
                print(
                    "WARNING: --use-recorded-arrivals is ignored under "
                    "--turn-pacing interleaved (everyone starts turn 0 together)."
                )

    async def run(self, executed: dict):
        """executed gets each request's sent_digest just before it goes out, keyed (session, turn)."""
        self.executed = executed
        self.slots = asyncio.Semaphore(self.mode.concurrency)
        self.by_turn: dict[int, list] = {i: [] for i in range(self.max_turns)}
        self.served: list = []
        self.prev_context: dict = {}
        self.prev_output: dict = {}
        self.dead: set = set()
        self.warned = False
        # session_id -> (arrived, admitted, finished), seconds since start; the steady window reads these
        self.session_times: dict = {}
        self.plays: list = []
        self.deadline_s: float | None = None
        self.start = time.perf_counter()
        if self.mode.duration_s > 0:
            await self._recycled()
        elif self.open_loop:
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
        # Per-session pacing admits agents, not requests: an admitted agent sends its next turn at once.
        gated = not self.open_loop and self.pacing.pacing == "interleaved"
        async with self.slots if gated else no_limit():
            granted = elapsed()
            self.executed[sid, turn] = sent_digest(request, spliced_at(request, self.trace.replies))
            result = await self.send(request, capture_text=True, request_id=self.request_id(sid, turn, index))
        annotate_request_observability(
            result,
            request_index=index,
            request=request,
            scheduled_at_s=scheduled_at_s,
            dispatch_started_at_s=dispatched,
            semaphore_acquired_at_s=granted,
            completed_at_s=elapsed(),
            run_start=self.start,
        )
        enforce_first_chunk(result)
        enforce_output_length(self.mode, request, result)
        annotate_multi_turn_cache_estimate(
            result,
            session_id=sid,
            turn_index=turn,
            previous_context_tokens=self.prev_context.get(sid, 0),
            cache_block_size=self.mode.prefix_cache_block_size,
            previous_output_tokens=self.prev_output.get(sid, 0),
            # Placeholders mean the engine reply is spliced into later prompts.
            reply_in_cache=bool(self.trace.replies.get(sid)),
        )
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
                print(
                    "WARNING: this dataset has no assistant placeholders; "
                    "replies are not fed back (trace replay keeps recorded "
                    "assistant turns).",
                    flush=True,
                )
            return
        if result.generated_text and turn < len(replies):
            replies[turn]["content"] = result.generated_text

    def _wait_s(self, request):
        p = self.pacing
        return inter_turn_wait_s(
            request, tool_wait_ms=p.tool_wait_ms, human_wait_ms=p.human_wait_ms, use_recorded_waits=p.use_recorded_waits
        )

    def _print_waits(self):
        p = self.pacing
        if p.tool_wait_ms or p.human_wait_ms or p.use_recorded_waits:
            print(
                f"  Inter-turn waits: tool_wait_ms={p.tool_wait_ms:g} "
                f"human_wait_ms={p.human_wait_ms:g} "
                f"recorded={'on' if p.use_recorded_waits else 'off'}"
            )

    async def _turns(self, turns, index_of, arrive_at_s=None):
        last = len(turns) - 1
        for request in turns:
            result = await self._dispatch(request, index_of(request), arrive_at_s if request.turn == 0 else None)
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
            print(
                f"  Open loop: {n} sessions at recorded arrival_time_ms; turns sequential within a session, no barrier"
            )
        else:
            spec = self.workload.arrival
            offsets = parse_arrival(spec, n, self.workload.seed)
            assert offsets is not None, "the loader refuses arrival=trace without recorded arrivals"
            arrivals = [t / self.workload.speed for t in offsets]
            print(
                f"  Open loop: {n} sessions arriving at "
                f"{offered_rate(spec)} sess/s ({spec.partition(':')[0]}); "
                f"turns sequential within a session, no barrier"
            )
        self._print_waits()
        counter = itertools.count()

        async def session(turns, arrive_at_s):
            await sleep_until_elapsed(self.start, arrive_at_s)
            await self._turns(turns, lambda r: next(counter), arrive_at_s)

        await asyncio.gather(*[session(s, t) for s, t in zip(self.sessions, arrivals)])
        done = [r for v in self.by_turn.values() for r in v if r is not None]
        if done and not any(r.success for r in done):
            raise RunAborted(f"ABORT: all {len(done)} requests failed. First error: {first_error(done)}")

    async def _per_session(self):
        index = {(r.session_id, r.turn): i for i, r in enumerate(r for s in self.sessions for r in s)}
        print(
            f"  Per-session pacing: {len(self.sessions)} sessions, {len(index)} requests, "
            f"{self.mode.concurrency} agents at once; an agent runs its session's turns back to back, "
            f"waits included, and the next session starts when one finishes"
        )
        self._print_waits()
        recorded = self.pacing.use_recorded_arrivals
        if recorded:
            print("  Session starts: recorded arrival_time_ms, then the wait for a free agent")
        agents = asyncio.Semaphore(self.mode.concurrency)

        def elapsed():
            return time.perf_counter() - self.start

        async def session(turns):
            arrived = turns[0].arrival_ts or 0.0
            if recorded:
                await sleep_until_elapsed(self.start, arrived)
            async with agents:
                admitted = elapsed()
                await self._turns(turns, lambda r: index[(r.session_id, r.turn)])
                self.session_times[turns[0].session_id] = (arrived if recorded else admitted, admitted, elapsed())

        await asyncio.gather(*[session(s) for s in self.sessions])
        results = [r for v in self.by_turn.values() for r in v]
        if results and not any(r is not None and r.success for r in results):
            raise RunAborted(f"ABORT: All {len(results)} requests failed. First error: {first_error(results)}")

    async def _recycled(self):
        """mode.duration_s: each agent draws the next session from a cycle over the sample, plays it to the end and
        draws again until the deadline; every play is salted (play())."""
        if self.open_loop or self.pacing.pacing != "per-session":
            raise ConfigError("mode.duration_s needs closed-loop, per-session pacing")
        if self.trace.live_replies:
            raise ConfigError("mode.duration_s needs recorded replies; this trace fills its replies from the engine")
        n, agents, self.deadline_s = len(self.sessions), self.mode.concurrency, self.mode.duration_s
        print(
            f"  Recycled replay: {n} sessions in a cycle, {agents} agents for {self.deadline_s:g} s; a play started "
            "before the deadline runs to its end, and every play carries its own salt"
        )
        self._print_waits()
        draws, indexes = itertools.count(), itertools.count()

        def elapsed():
            return time.perf_counter() - self.start

        async def agent():
            while elapsed() < self.deadline_s:
                k = next(draws)
                turns = play(self.sessions[k % n], self.workload.seed, k // n, self.workload.salt)
                self.plays.append(turns)
                admitted = elapsed()
                await self._turns(turns, lambda r: next(indexes))
                self.session_times[turns[0].session_id] = (admitted, admitted, elapsed())

        await asyncio.gather(*[agent() for _ in range(agents)])

    async def _interleaved(self):
        for turn in range(self.max_turns):
            batch = [s[turn] for s in self.sessions if s[0].session_id not in self.dead and turn < len(s)]
            if not batch:
                continue
            print(f"  Turn {turn + 1}/{self.max_turns}: dispatching {len(batch)} requests...")
            offset = sum(len(v) for v in self.by_turn.values())
            completed = await asyncio.gather(*[self._dispatch(r, offset + i) for i, r in enumerate(batch)])
            if not any(r is not None and r.success for r in completed):
                raise RunAborted(
                    f"ABORT: All {len(completed)} requests in turn {turn + 1} "
                    f"failed. First error: {first_error(completed)}"
                )
            for request, result in zip(batch, completed):
                self._record(request, result)
