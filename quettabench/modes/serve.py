"""THE measurement procedure: send a Trace's requests through a Driver and
collect one RequestResult row per request.

run(trace, driver, mode, writer, workload) warms the engine on the trace's
warmup requests, resets the prefix cache when asked, then replays:

  closed loop  mode.concurrency slots; a request's TTFT clock starts at its
               slot grant (ttft_basis "slot")
  open loop    no cap; each request fires at its arrival from
               workload.arrival / workload.speed (ttft_basis "arrival")

A SessionTrace goes through SessionDispatcher (modes/_sessions_dispatch.py),
which feeds each reply into the session's later turns. With mode.grid set
the trace is a GridTrace and run_grid (modes/_grid.py) times it cell by
cell instead; its per-step numbers are the run's steps section. Every row goes to the
writer as it finishes. Then the gates from metrics.core (exit 1 on abort)
and the workload identity gate: the served requests must hash to
trace.workload_hash(), else the run is "failed" and exits 3.

quettabench/runner.py builds the trace, driver and writer and calls run.
"""

import asyncio
import dataclasses
import sys
import time
from dataclasses import dataclass, field

from ..config.schema import WorkloadCfg
from ..engines.drivers import PrefixCacheResetError, server_base_url
from ..metrics.core import (
    BenchmarkSummary,
    acceptance,
    aggregate,
    aggregate_per_turn,
    annotate_request_observability,
    burst_recovery,
    print_multi_turn_summary,
    print_summary,
    sla_verdict,
    success_rate_gate,
    summarize_serve,
    transient_gate,
    usage_gate,
)
from ..metrics.results import goodprobe_line, print_legacy_lines
from ..profiling.native import spec_decode_counters
from ..workloads.arrival import offered_rate, parse_arrival, rebase_arrivals
from ..workloads.base import workload_hash
from ..workloads.sessions import SessionTrace
from ._grid import run_grid
from ._sessions_dispatch import RunAborted, SessionDispatcher, no_limit

WORKLOAD_SCHEMA_VERSION = "distributional-synthetic-v1"
TRACE_REQUEST_ID_PREFIX = "agenticbench"
IDENTITY_EXIT = 3  # legacy/trace_serve.py's exit code for a workload mismatch
DEFAULT_WORKLOAD = WorkloadCfg()


@dataclass
class Summary:
    """One run: the aggregate (None for a grid run), the rows in summary
    order, the per-turn breakdown (multi-turn only) and the extra run.json
    sections."""
    core: BenchmarkSummary
    rows: list
    turns: list = field(default_factory=list)
    sections: dict = field(default_factory=dict)
    status: str = "ok"
    reason: str | None = None

    def to_dict(self):
        out = {**(self.core.to_dict() if self.core else {}), **self.sections}
        if self.turns:
            out["per_turn"] = [t.to_dict() for t in self.turns]
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
    """Submission offsets for a single-turn trace. An injected schedule is
    stamped on the requests' arrival_ts first, so it joins the workload hash
    (trace-paced and injected runs never share one); "trace" keeps the
    trace's own times, rebased to the first."""
    reqs = trace.requests()
    injected = parse_arrival(workload.arrival, len(reqs), workload.seed)
    if injected is None:
        return rebase_arrivals([r.arrival_ts for r in reqs], workload.speed)
    for r, t in zip(reqs, injected):
        r.arrival_ts = round(t, 6)
    return [t / workload.speed for t in injected]


async def warmup(trace, driver, mode, count, multi_turn):
    """Send count warmup requests at the warmup width; discard the results,
    abort when none succeeds. Warmup goes over the run's own session so its
    TCP connections exist before timing starts."""
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
    """emit(request, result): stamp the TTFT basis, score acceptance against the
    trace's reference reply when the driver returned tokens, write the row."""
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
    """Single-turn dispatch; returns ([(request, result)] by index, start time).
    The first mode.discard_first requests are held out of the summary."""
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
    """Spec-decode counters of the server an HTTP driver talks to; None for
    any other driver, or a server without them."""
    url = getattr(driver, "url", None)
    return spec_decode_counters(server_base_url(url)) if url else None


async def measure(trace, driver, mode, emit, workload, request_id):
    """Warm, reset, dispatch: (results, results_by_turn | None, [(request, result)],
    duration, server SPECSUM | None). The server's spec-decode counters are read
    just before and just after dispatch, so warmup is not in the window."""
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
    """trace_serve's records from the successful, counted rows: TTFT (whole
    latency when no token came), wall-clock TPOT, output tokens, and the
    arrival burst windows bucket by (the slot grant under a closed loop)."""
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
    """SPECSUM from the server's counter deltas over the measured window, as
    vllm bench serve defines them: accept_rate = accepted / draft tokens,
    accept_len = 1 + accepted / drafts (its qwen_tp_bench.json numbers obey
    both). None without both scrapes or without a draft in the window."""
    if before is None or after is None:
        return None
    d = {key: int(after[key] - before[key]) for key in after}
    if d["drafts"] <= 0 or d["draft_tokens"] <= 0:
        return None
    return {"spec": "measured", "source": "server_metrics", **d,
            "accept_rate": d["accepted_tokens"] / d["draft_tokens"],
            "accept_len": 1 + d["accepted_tokens"] / d["drafts"], "n": d["drafts"]}


def crossval_sections(pairs, duration, mode, server_spec=None):
    """serve (nearest-rank, as SERVESUM always was), sla, burst and spec;
    spec is the server's counters when it reported them, else the rows'."""
    out = {"spec": server_spec or spec_section(pairs)}
    records = serve_records(pairs, mode.load == "open")
    if not records:
        return out
    s = out["serve"] = summarize_serve(records, duration, "nearest")
    if mode.sla_ttft_ms or mode.sla_tpot_ms:
        bound = sla_verdict(s, mode.sla_ttft_ms, mode.sla_tpot_ms)
        out["sla"] = {"verdict": "FAIL" if bound else "PASS", "bound": bound,
                      "ttft_p99": s["ttft_ms"]["p99"],
                      "tpot_p99": s["tpot_ms"]["p99"] if s["tpot_ms"] else 0,
                      "rate_req_s": s["req_s"],
                      "sla_ttft_ms": mode.sla_ttft_ms, "sla_tpot_ms": mode.sla_tpot_ms}
    if mode.sla_ttft_ms:
        recovery, peak = burst_recovery(records, mode.sla_ttft_ms)
        out["burst"] = {"peak_ttft_p99": peak, "recovery_s": recovery,
                        "target_ms": mode.sla_ttft_ms}
    return out


def goodput_search(start=1.0, max_iters=6, tol=0.1):
    """Generator bisection over speed: yields the next speed, receives whether
    that probe met the SLA. Doubles from start to the first FAIL, halves to
    the first PASS, then bisects until the bracket is within tol; at most
    max_iters probes (legacy/trace_serve.py goodput_search)."""
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
    """--goodput-sweep: one measure() per probe speed (warmup on the first
    only), rows tagged with their probe. Stops at the first probe whose
    served set breaks the workload identity. Returns the last probe's
    measure() result and the goodput section."""
    gen, probes, best, fail_bound = goodput_search(start=workload.speed), [], None, []
    speed = next(gen)
    while True:
        probe_mode = mode if not probes else dataclasses.replace(mode, warmup=0)
        out = asyncio.run(measure(trace, driver, probe_mode, emitter(mode, writer, len(probes)),
                                  dataclasses.replace(workload, speed=speed), request_id))
        section = {"max_req_s": None, "speed": None, "fail_bound": fail_bound, "probes": probes}
        if not identity(trace, [q for q, _ in out[2]])["match"]:
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


def identity(trace, served) -> dict:
    """The served requests hashed the way the trace hashes itself."""
    expected, replayed = trace.workload_hash(), workload_hash(served)
    return {"expected": expected, "replayed": replayed, "match": expected == replayed,
            "n": len(served), "of": len(trace.requests())}


def run(trace, driver, mode, writer, workload=DEFAULT_WORKLOAD) -> Summary:
    """One measured run of trace through driver; see the module docstring.
    A grid run reads its engine facts from writer.cfg, the run's Config."""
    if mode.grid:
        record, steps, rows = run_grid(trace, driver, writer.cfg, writer)
        summary = Summary(None, rows, sections={"grid": record})
        writer.finish(summary.to_dict(), steps=steps)
        return summary
    if mode.goodput_sweep and (mode.load != "open" or not (mode.sla_ttft_ms or mode.sla_tpot_ms)):
        raise ValueError("goodput sweep needs open load and mode.sla_ttft_ms and/or sla_tpot_ms")
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
        # Fatal by design: going on would measure the warm cache the reset prevents.
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

    # Single-turn: only requests that got a result count as served, so a
    # dropped request moves the hash.
    served = [q for q, r in pairs if by_turn is not None or r.success]
    ident = summary.sections["workload"] = identity(trace, served)
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
