"""THE measurement procedure: send a Trace's requests through a Driver and
collect one RequestResult row per request.

run(trace, driver, mode, writer, workload) warms the engine on the trace's
warmup requests, resets the prefix cache when asked, then replays:

  closed loop  mode.concurrency slots; a request's TTFT clock starts at its
               slot grant (ttft_basis "slot")
  open loop    no cap; each request fires at its arrival from
               workload.arrival / workload.speed (ttft_basis "arrival")

A SessionTrace goes through SessionDispatcher (modes/_sessions_dispatch.py),
which feeds each reply into the session's later turns. Every row goes to the
writer as it finishes. Then the gates from metrics.core (exit 1 on abort)
and the workload identity gate: the served requests must hash to
trace.workload_hash(), else the run is "failed" and exits 3.

main(argv) is the old runner CLI (python -m quettabench.modes.serve ...),
translated by modes/legacy_cli.py; commit 7 removes it.
"""

import asyncio
import sys
import time
from dataclasses import dataclass, field

from ..config.schema import WorkloadCfg
from ..engines.drivers import PrefixCacheResetError
from ..metrics.core import (
    BenchmarkSummary,
    aggregate,
    aggregate_per_turn,
    annotate_request_observability,
    print_multi_turn_summary,
    print_summary,
    success_rate_gate,
    transient_gate,
    usage_gate,
)
from ..metrics.results import write_legacy_per_turn, write_legacy_results
from ..workloads.arrival import offered_rate, parse_arrival, rebase_arrivals
from ..workloads.base import workload_hash
from ..workloads.sessions import SessionTrace
from ._sessions_dispatch import RunAborted, SessionDispatcher, no_limit
from .legacy_cli import apply_mode, check_args, get_args, list_profiles, translate

WORKLOAD_SCHEMA_VERSION = "distributional-synthetic-v1"
TRACE_REQUEST_ID_PREFIX = "agenticbench"
IDENTITY_EXIT = 3  # legacy/trace_serve.py's exit code for a workload mismatch
DEFAULT_WORKLOAD = WorkloadCfg()


@dataclass
class Summary:
    """One run: the aggregate, the rows in summary order, the per-turn
    breakdown (multi-turn only) and the extra run.json sections."""
    core: BenchmarkSummary
    rows: list
    turns: list = field(default_factory=list)
    sections: dict = field(default_factory=dict)
    status: str = "ok"
    reason: str | None = None

    def to_dict(self):
        out = {**self.core.to_dict(), **self.sections}
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


async def replay(trace, driver, mode, writer, offsets, request_id):
    """Single-turn dispatch; returns ([(request, result)] by index, start time).
    The first mode.discard_first requests are held out of the summary."""
    open_loop = mode.load == "open"
    basis = "arrival" if open_loop else "slot"
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
        result.ttft_basis = basis
        if i < mode.discard_first:
            result.excluded_from_summary = True
        writer.append(result)
        rows[i] = (request, result)

    await asyncio.gather(*(dispatch(i, r, t) for i, (r, t) in enumerate(zip(reqs, offsets))))
    fail = sum(1 for _, r in rows if not r.success)
    if fail > 0 and fail >= len(reqs) * 0.9:
        raise RunAborted(f"ABORT: {fail}/{len(reqs)} requests failed "
                         f"({fail / len(reqs) * 100:.0f}%). "
                         f"Server may not be functional. Check server logs.")
    return rows, start


async def measure(trace, driver, mode, writer, workload, request_id):
    """Warm, reset, dispatch: (results, results_by_turn | None, served, duration)."""
    multi = isinstance(trace, SessionTrace)
    offsets = None if multi else arrival_offsets(trace, workload)
    sessions = None
    if multi:
        sessions = SessionDispatcher(trace, mode, workload,
                                     lambda r, **kw: driver.send(r, **kw, **sampling(mode, r)),
                                     request_id, writer)
        sessions.announce()
    async with driver.session(mode.concurrency, mode.load == "open"):
        if mode.warmup > 0:
            await warmup(trace, driver, mode, mode.warmup, multi)
        # After warmup, so the run starts cold whatever warmup left behind.
        if mode.reset_prefix_cache:
            print(f"Prefix cache: {await driver.reset()}")
        if multi:
            results, by_turn, served, start = await sessions.run()
        else:
            rows, start = await replay(trace, driver, mode, writer, offsets, request_id)
            results, by_turn, served = [r for _, r in rows], None, [q for q, _ in rows]
    return results, by_turn, served, time.perf_counter() - start


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
    """One measured run of trace through driver; see the module docstring."""
    name = trace_name(trace)

    def request_id(session, turn, index):
        if not mode.trace_request_ids:
            return None
        return make_trace_request_id(profile_name=name, concurrency=mode.concurrency,
                                     session_id=session, turn_index=turn, request_index=index)

    try:
        results, by_turn, served, duration = asyncio.run(
            measure(trace, driver, mode, writer, workload, request_id))
    except PrefixCacheResetError as e:
        # Fatal by design: going on would measure the warm cache the reset prevents.
        print(f"ABORT: prefix cache reset failed: {e}")
        stop(writer, None, "aborted", f"prefix cache reset failed: {e}", 1)
    except RunAborted as e:
        print(e)
        stop(writer, None, "aborted", str(e), 1)

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

    summary.sections["workload"] = ident = identity(trace, served)
    if not ident["match"]:
        print(f"WORKLOADSUM expected={ident['expected']} replayed={ident['replayed']} "
              f"match=0 n={ident['n']}/{ident['of']}", flush=True)
        print("WORKLOADVOID replayed workload differs from the trace; comparison is void",
              flush=True)
        stop(writer, summary.to_dict(), "failed", "workload identity", IDENTITY_EXIT)
    if core.failed_requests:
        summary.status = "partial"
        summary.reason = f"{core.failed_requests} of {core.num_requests} requests failed"
    writer.finish(summary.to_dict(), status=summary.status, reason=summary.reason)
    return summary


def main(argv=None) -> None:
    """The old runner CLI: parse, check, translate to Config/Trace/Driver, run, save."""
    args = get_args(argv)
    if args.list_profiles:
        list_profiles(args)
        sys.exit(0)
    check_args(args)
    apply_mode(args)
    config, cfg, trace, driver, writer = translate(args)
    summary = run(trace, driver, cfg.mode, writer, cfg.workload)
    write_legacy_results(summary.core, summary.rows, args.output, config)
    if summary.turns:
        write_legacy_per_turn(summary.turns, args.output, config)


if __name__ == "__main__":
    main()
