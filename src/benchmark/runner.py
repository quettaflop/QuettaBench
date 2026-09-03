"""
Benchmark runner — orchestrates a full benchmark run.

Usage:
    python -m src.benchmark.runner \
        --url http://localhost:8000/v1/chat/completions \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --backend vllm \
        --profile chat-singleturn \
        --concurrency 10 \
        --num-requests 100 \
        --tensor-parallel-size 1 \
        --api-key test \
        --output results/run_001.json
"""

import asyncio
import argparse
import itertools
import json
import sys
import time
import os
from pathlib import Path

from contextlib import asynccontextmanager

from .metrics import (
    aggregate,
    aggregate_per_turn,
    annotate_multi_turn_cache_estimate,
    annotate_request_observability,
    print_summary,
    print_multi_turn_summary,
)
from .server_control import PrefixCacheResetError, reset_prefix_cache
from ..workloads.profiles import get_profile
from ..workloads.dataset import make_dataset
from ..workloads.arrival import make_arrival_times


SUPPORTED_BACKENDS = ["openai", "vllm", "sglang"]
# v5: greedy-by-default + seed; failed requests keep partial timings and error_kind.
BENCHMARK_SCHEMA_VERSION = 5
WORKLOAD_SCHEMA_VERSION = "distributional-synthetic-v1"
TRACE_REQUEST_ID_PREFIX = "agenticbench"

# Closed loop: a fixed population of `concurrency` in-flight requests, each
# replaced only when the previous one finishes. Offered load is a consequence of
# server speed, so the server can never fall behind.
# Open loop: requests arrive on a clock at --target-rate regardless of whether
# the server keeps up, so queues can grow without bound. This is the one that
# exposes saturation.
LOAD_MODES = ("closed-loop", "open-loop")
OPEN_LOOP_ARRIVALS = ("poisson", "ramp")
TURN_PACINGS = ("per-session", "interleaved")


def inter_turn_wait_s(
    request,
    *,
    tool_wait_ms: float = 0.0,
    human_wait_ms: float = 0.0,
    use_recorded_waits: bool = False,
) -> float:
    """Seconds to sleep after a turn before the next turn in the same session.

    Defaults are zero (auto-mode: no human approval pause, ignore recorded
    tool gaps). ``--use-recorded-waits`` adds ``tool_wait_after_ms`` and
    ``human_wait_after_ms`` from the turn metadata when present.
    """
    recorded_tool = 0.0
    recorded_human = 0.0
    if use_recorded_waits and request is not None:
        metadata = getattr(request, "metadata", None) or {}
        recorded_tool = float(metadata.get("tool_wait_after_ms") or 0.0)
        recorded_human = float(metadata.get("human_wait_after_ms") or 0.0)
    total_ms = tool_wait_ms + human_wait_ms + recorded_tool + recorded_human
    return max(0.0, total_ms / 1000.0)


async def _sleep_until_elapsed(benchmark_start: float, target_s: float) -> None:
    delay = target_s - (time.perf_counter() - benchmark_start)
    if delay > 0:
        await asyncio.sleep(delay)


@asynccontextmanager
async def _no_limit():
    """Stand-in for the concurrency semaphore when running open loop."""
    yield


async def _warmup_with_profile(
    session,
    backend,
    dataset,
    *,
    url: str,
    model: str,
    api_key: str,
    concurrency: int,
    count: int,
    ignore_eos: bool,
    temperature: float,
    sampling_seed: int | None,
    exact_output_length: bool,
) -> tuple[int, int]:
    """Warm the server with real profile requests at a given batch width.

    `concurrency` here is the WARMUP width, which is not always the run's
    --concurrency: under open loop that flag caps nothing, so warmup sized from
    it can miss the batch shapes the run actually reaches. Engines that JIT
    kernels per shape (DeepSeek V4's TileLang MHC kernels, Triton) then compile
    mid-measurement, which inflates TTFT and pushes the TPOT mean far above its
    median. See --warmup-concurrency.

    The old warmup sent a handful of "Hello" prompts at max_tokens=10 over a
    throwaway ClientSession. That warms nothing the benchmark then measures:
    not the first large-ISL prefill, not decode at real batch width, and not
    the benchmark session's own TCP connections -- so those one-time costs
    landed inside the first measured requests' TTFT instead.

    Requests are drawn from the live dataset, so for single-turn profiles the
    warmup prompts are consumed and the measured run sees different ones.
    Results are discarded, but failures are counted: a warmup where nothing
    succeeds means the server is not usable and the run should not proceed.

    Returns (successful, attempted).
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def one():
        request = dataset.get_next_request()
        async with semaphore:
            return await backend.send_request(
                session=session,
                url=url,
                model=model,
                messages=request.messages,
                max_tokens=request.max_tokens,
                api_key=api_key,
                ignore_eos=ignore_eos or exact_output_length,
                temperature=temperature,
                seed=sampling_seed,
                min_tokens=request.max_tokens if exact_output_length else 0,
            )

    results = await asyncio.gather(*[one() for _ in range(count)], return_exceptions=True)
    ok = sum(
        1 for r in results
        if not isinstance(r, BaseException) and r is not None and r.success
    )
    return ok, len(results)


def _warn_if_transient_dominated(num_requests: int, concurrency: int) -> None:
    """Warn when the run is too short to contain a steady-state regime.

    A closed-loop run is one synchronized wave of `concurrency` requests, then
    backfill as each completes, then a drain to zero. Settling takes several
    generations of turnover (one generation == `concurrency` completions).
    Below ~4 the run is all startup convoy and drain, and -- importantly --
    discarding cannot fix it, because there is no settled middle left to keep.
    """
    if concurrency <= 1 or num_requests <= 0:
        return
    per_slot = num_requests / concurrency
    if per_slot >= 4:
        return
    print(f"WARNING: only {per_slot:.1f} requests per concurrency slot "
          f"({num_requests} requests at concurrency {concurrency}).")
    print(f"         The run is dominated by the startup wave and the drain tail, "
          f"with no")
    print(f"         steady-state middle. --discard-first cannot recover it; raise "
          f"--num-requests")
    print(f"         to at least {concurrency * 8} for a settled measurement.")


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


async def run_benchmark(
    url: str,
    model: str,
    profile_name: str,
    concurrency: int,
    num_requests: int,
    backend_name: str = "vllm",
    api_key: str = "test",
    arrival_pattern: str = "steady",
    target_rate: float = 10.0,
    warmup_requests: int = 3,
    seed: int = 42,
    timeout: int = 120,
    ignore_eos: bool = False,
    max_context_tokens: int | None = None,
    context_safety_margin_tokens: int = 256,
    trace_request_ids: bool = False,
    load_mode: str = "closed-loop",
    temperature: float = 0.0,
    sampling_seed: int | None = None,
    exact_output_length: bool = False,
    reset_prefix_cache_first: bool = False,
    warmup_concurrency: int | None = None,
):
    """
    Run a benchmark and return (results, duration).

    load_mode="closed-loop" holds `concurrency` requests in flight and dispatches
    everything at t=0; load_mode="open-loop" lets arrivals fire on the schedule
    from `arrival_pattern`/`target_rate` with no client-side cap, so an
    overloaded server builds a real queue instead of back-pressuring the client.
    """
    import aiohttp
    from ..engines import get_backend

    backend = get_backend(backend_name)
    open_loop = load_mode == "open-loop"
    profile = get_profile(profile_name)
    dataset = make_dataset(
        profile,
        max_context_tokens=max_context_tokens,
        random_seed=seed,
        context_safety_margin_tokens=context_safety_margin_tokens,
        tokenizer_name=model,
    )
    arrival_times = make_arrival_times(
        pattern=arrival_pattern,
        num_requests=num_requests,
        concurrency=concurrency,
        target_rate=target_rate,
        seed=seed,
    )

    # Open loop must not cap connections: a connector limit would silently
    # become the concurrency cap we just removed.
    connector = aiohttp.TCPConnector(limit=0 if open_loop else concurrency + 10)
    client_timeout = aiohttp.ClientTimeout(total=timeout)

    async with aiohttp.ClientSession(connector=connector, timeout=client_timeout) as session:
        # Warmup: real profile requests at the real concurrency, over THIS
        # session so its TCP connections are established before timing starts.
        if warmup_requests > 0:
            print(f"Warming up with {warmup_requests} profile requests "
                  f"at concurrency {concurrency}...")
            ok, attempted = await _warmup_with_profile(
                session, backend, dataset,
                url=url, model=model, api_key=api_key,
                concurrency=(warmup_concurrency or concurrency), count=warmup_requests,
                ignore_eos=ignore_eos, temperature=temperature,
                sampling_seed=sampling_seed,
                exact_output_length=exact_output_length,
            )
            print(f"Warmup done: {ok}/{attempted} succeeded.")
            if ok == 0:
                print("ABORT: every warmup request failed. The server is not serving "
                      "this profile; check the server log before benchmarking.")
                sys.exit(1)

        # Reset AFTER warmup so the run starts from a genuinely cold cache
        # regardless of what warmup left behind.
        if reset_prefix_cache_first:
            status = await reset_prefix_cache(session, url, backend_name, api_key, timeout)
            print(f"Prefix cache: {status}")

        # Schedule requests
        semaphore = asyncio.Semaphore(concurrency)
        results = [None] * num_requests
        benchmark_start = time.perf_counter()

        async def dispatch(i: int, dispatch_time: float):
            now = time.perf_counter() - benchmark_start
            delay = dispatch_time - now
            if delay > 0:
                await asyncio.sleep(delay)

            request = dataset.get_next_request()
            dispatch_started_at_s = time.perf_counter() - benchmark_start
            async with (_no_limit() if open_loop else semaphore):
                semaphore_acquired_at_s = time.perf_counter() - benchmark_start
                result = await backend.send_request(
                    session=session,
                    url=url,
                    model=model,
                    messages=request.messages,
                    max_tokens=request.max_tokens,
                    api_key=api_key,
                    ignore_eos=ignore_eos or exact_output_length,
                    temperature=temperature,
                    seed=sampling_seed,
                    min_tokens=request.max_tokens if exact_output_length else 0,
                    request_id=(
                        make_trace_request_id(
                            profile_name=profile_name,
                            concurrency=concurrency,
                            session_id=i,
                            turn_index=0,
                            request_index=i,
                        )
                        if trace_request_ids
                        else None
                    ),
                )
            completed_at_s = time.perf_counter() - benchmark_start
            annotate_request_observability(
                result,
                request_index=i,
                request=request,
                scheduled_at_s=dispatch_time,
                dispatch_started_at_s=dispatch_started_at_s,
                semaphore_acquired_at_s=semaphore_acquired_at_s,
                completed_at_s=completed_at_s,
            )
            results[i] = result

        tasks = [dispatch(i, t) for i, t in enumerate(arrival_times)]
        await asyncio.gather(*tasks)

        ok = sum(1 for r in results if r is not None and r.success)
        fail = num_requests - ok
        if fail > 0 and fail >= num_requests * 0.9:
            print(
                f"ABORT: {fail}/{num_requests} requests failed "
                f"({fail / num_requests * 100:.0f}%). "
                f"Server may not be functional. Check server logs."
            )
            sys.exit(1)

    benchmark_duration = time.perf_counter() - benchmark_start
    return results, benchmark_duration


def _first_error(results) -> str:
    """First recorded error text, for abort messages that name a cause."""
    for r in results:
        if r is not None and not r.success and r.error:
            return str(r.error)
    return "no error text recorded"


async def run_multi_turn_benchmark(
    url: str,
    model: str,
    profile_name: str,
    concurrency: int,
    backend_name: str = "vllm",
    api_key: str = "test",
    warmup_requests: int = 3,
    timeout: int = 120,
    ignore_eos: bool = False,
    max_context_tokens: int | None = None,
    context_safety_margin_tokens: int = 256,
    seed: int = 42,
    cache_block_size: int | None = 16,
    num_sessions: int | None = None,
    source_session_ids: list[str] | None = None,
    max_turn_index: int | None = None,
    trace_request_ids: bool = False,
    turn_pacing: str = "per-session",
    temperature: float = 0.0,
    sampling_seed: int | None = None,
    exact_output_length: bool = False,
    reset_prefix_cache_first: bool = False,
    warmup_concurrency: int | None = None,
    load_mode: str = "closed-loop",
    arrival_pattern: str = "steady",
    target_rate: float = 1.0,
    tool_wait_ms: float = 0.0,
    human_wait_ms: float = 0.0,
    use_recorded_waits: bool = False,
    use_recorded_arrivals: bool = False,
):
    """Run a multi-turn benchmark.

    CLOSED LOOP (default): ``turn_pacing="per-session"`` runs a session's turns
    back to back; sessions share the in-flight semaphore. ``interleaved``
    barriers every turn across sessions (turn-aligned herd; not production
    traffic — TTFT/TPOT spikes at turn boundaries are schedule artifacts).

    OPEN LOOP: sessions arrive at ``target_rate`` sess/s (or recorded
    ``arrival_time_ms``); turns are sequential within a session, with no
    barrier and no concurrency cap.

    Inter-turn sleeps (``tool_wait_ms`` + ``human_wait_ms``, plus recorded
    waits if requested) apply to per-session and open-loop only. They default
    to zero. Interleaved ignores them.

    Later prompts contain the engine's own reply. Trace replay keeps recorded
    assistant text. ``exact_output_length`` pins min_tokens=max_tokens.

    Returns (all_results, results_by_turn, duration).
    """
    if turn_pacing not in TURN_PACINGS:
        raise ValueError(
            f"turn_pacing must be one of {TURN_PACINGS}, got {turn_pacing!r}"
        )
    import aiohttp
    from ..engines import get_backend

    from ..workloads.dataset import (
        DistributionalMultiTurnDataset,
        ShareGPTMultiTurnDataset,
        TrajectoryMultiTurnDataset,
    )

    backend = get_backend(backend_name)
    open_loop = load_mode == "open-loop"
    profile = get_profile(profile_name)
    dataset = make_dataset(
        profile,
        max_context_tokens=max_context_tokens,
        random_seed=seed,
        context_safety_margin_tokens=context_safety_margin_tokens,
        num_sessions=num_sessions,
        tokenizer_name=model,
        source_session_ids=source_session_ids,
    )

    if not isinstance(dataset, (DistributionalMultiTurnDataset, ShareGPTMultiTurnDataset, TrajectoryMultiTurnDataset)):
        raise ValueError(f"Profile '{profile_name}' does not use a multi-turn dataset")

    sessions = dataset.sessions
    if not sessions:
        raise ValueError("No multi-turn sessions loaded — check ShareGPT dataset and filter bounds")

    max_turns = max(len(s.turns) for s in sessions)
    print(f"Loaded {len(sessions)} sessions, max {max_turns} turns per session")
    if not open_loop and turn_pacing == "interleaved":
        print("WARNING: --turn-pacing interleaved is a turn-aligned herd "
              "(global barrier per turn).")
        print("         TTFT/TPOT spikes at turn boundaries are schedule "
              "artifacts, not production traffic.")
        print("         Default is per-session. Inter-turn waits are ignored "
              "under interleaved.")
        if use_recorded_arrivals:
            print("WARNING: --use-recorded-arrivals is ignored under "
                  "--turn-pacing interleaved (everyone starts turn 0 together).")

    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    client_timeout = aiohttp.ClientTimeout(total=timeout)

    async with aiohttp.ClientSession(connector=connector, timeout=client_timeout) as session_http:
        # Warmup
        if warmup_requests > 0:
            # Multi-turn datasets serve get_next_request() from a flat iterator
            # that is independent of `sessions`, so warmup prompts DUPLICATE
            # measured ones rather than consuming them. That pre-warms the
            # prefix cache for the measured run unless it is reset afterwards.
            if not reset_prefix_cache_first:
                print("WARNING: multi-turn warmup replays prompts the measured run will "
                      "also send,")
                print("         so the cache starts warm and turn-1 TTFT is understated. "
                      "Pass")
                print("         --reset-prefix-cache (or --warmup 0) to measure a cold "
                      "first turn.")
            print(f"Warming up with {warmup_requests} profile requests "
                  f"at concurrency {concurrency}...")
            ok, attempted = await _warmup_with_profile(
                session_http, backend, dataset,
                url=url, model=model, api_key=api_key,
                concurrency=(warmup_concurrency or concurrency), count=warmup_requests,
                ignore_eos=ignore_eos, temperature=temperature,
                sampling_seed=sampling_seed,
                exact_output_length=exact_output_length,
            )
            print(f"Warmup done: {ok}/{attempted} succeeded.")
            if ok == 0:
                print("ABORT: every warmup request failed. The server is not serving "
                      "this profile; check the server log before benchmarking.")
                sys.exit(1)

        # Reset AFTER warmup so the run starts cold. Intra-run prefix reuse --
        # the thing multi-turn profiles exist to measure -- is unaffected.
        if reset_prefix_cache_first:
            status = await reset_prefix_cache(session_http, url, backend_name, api_key, timeout)
            print(f"Prefix cache: {status}")

        semaphore = asyncio.Semaphore(concurrency)
        # results_by_turn[turn_idx] = list of RequestResult
        results_by_turn: dict[int, list] = {i: [] for i in range(max_turns)}
        previous_context_by_session: dict[int, int] = {}
        previous_output_by_session: dict[int, int] = {}
        sessions_by_id = {s.session_id: s for s in sessions}
        nonlocal_state = {"warned": False}
        benchmark_start = time.perf_counter()

        async def dispatch(
            session_id: int,
            request,
            t_idx: int,
            previous_context_tokens: int,
            request_index: int,
            scheduled_at_s: float | None = None,
        ):
            dispatch_started_at_s = time.perf_counter() - benchmark_start
            async with (_no_limit() if open_loop else semaphore):
                semaphore_acquired_at_s = time.perf_counter() - benchmark_start
                result = await backend.send_request(
                    session=session_http,
                    url=url,
                    model=model,
                    messages=request.messages,
                    max_tokens=request.max_tokens,
                    api_key=api_key,
                    ignore_eos=ignore_eos or exact_output_length,
                    capture_text=True,
                    temperature=temperature,
                    seed=sampling_seed,
                    min_tokens=request.max_tokens if exact_output_length else 0,
                    request_id=(
                        make_trace_request_id(
                            profile_name=profile_name,
                            concurrency=concurrency,
                            session_id=session_id,
                            turn_index=t_idx,
                            request_index=request_index,
                        )
                        if trace_request_ids
                        else None
                    ),
                )
            completed_at_s = time.perf_counter() - benchmark_start
            annotate_request_observability(
                result,
                request_index=request_index,
                request=request,
                scheduled_at_s=scheduled_at_s,
                dispatch_started_at_s=dispatch_started_at_s,
                semaphore_acquired_at_s=semaphore_acquired_at_s,
                completed_at_s=completed_at_s,
            )
            annotate_multi_turn_cache_estimate(
                result,
                session_id=session_id,
                turn_index=t_idx,
                previous_context_tokens=previous_context_tokens,
                cache_block_size=cache_block_size,
                previous_output_tokens=previous_output_by_session.get(session_id, 0),
                # Placeholders mean the engine reply is spliced into later prompts.
                reply_in_cache=bool(
                    getattr(sessions_by_id.get(session_id), "assistant_messages", None)),
            )
            return session_id, t_idx, result

        def _last_turn(n_turns: int) -> int:
            last = n_turns - 1
            return last if max_turn_index is None else min(last, max_turn_index)

        def _wait_s(request) -> float:
            return inter_turn_wait_s(
                request,
                tool_wait_ms=tool_wait_ms,
                human_wait_ms=human_wait_ms,
                use_recorded_waits=use_recorded_waits,
            )

        apply_inter_turn_waits = open_loop or turn_pacing == "per-session"

        dead_sessions: set[int] = set()

        def _record(sid: int, t_idx: int, result) -> None:
            results_by_turn[t_idx].append(result)
            if result is not None and result.success and result.input_tokens > 0:
                previous_context_by_session[sid] = int(result.input_tokens)
                previous_output_by_session[sid] = int(result.output_tokens or 0)
            if result is not None and result.success:
                _feed_reply_back(sid, t_idx, result)
            else:
                dead_sessions.add(sid)

        def _feed_reply_back(sid: int, t_idx: int, result) -> None:
            session = sessions_by_id.get(sid)
            if session is None:
                return
            if not getattr(session, "assistant_messages", None):
                if not nonlocal_state["warned"]:
                    nonlocal_state["warned"] = True
                    print("WARNING: this dataset has no assistant placeholders; "
                          "replies are not fed back (trace replay keeps recorded "
                          "assistant turns).",
                          flush=True)
                return
            text = result.generated_text
            if not text:
                return
            if t_idx < len(session.assistant_messages):
                session.assistant_messages[t_idx]["content"] = text

        if open_loop:
            if use_recorded_arrivals:
                arrivals = [s.arrival_time_ms / 1000.0 for s in sessions]
                print(f"  Open loop: {len(sessions)} sessions at recorded "
                      f"arrival_time_ms; turns sequential within a session, "
                      f"no barrier")
            else:
                arrivals = make_arrival_times(
                    pattern=arrival_pattern,
                    num_requests=len(sessions),
                    concurrency=concurrency,
                    target_rate=target_rate,
                    seed=seed,
                )
                print(f"  Open loop: {len(sessions)} sessions arriving at "
                      f"{target_rate} sess/s ({arrival_pattern}); "
                      f"turns sequential within a session, no barrier")
            if apply_inter_turn_waits and (
                tool_wait_ms or human_wait_ms or use_recorded_waits
            ):
                print(f"  Inter-turn waits: tool_wait_ms={tool_wait_ms:g} "
                      f"human_wait_ms={human_wait_ms:g} "
                      f"recorded={'on' if use_recorded_waits else 'off'}")
            counter = itertools.count()

            async def run_open_session(conv_session, arrive_at_s: float):
                await _sleep_until_elapsed(benchmark_start, arrive_at_s)
                last = _last_turn(len(conv_session.turns))
                for t_idx, request in enumerate(conv_session.turns):
                    if t_idx > last:
                        break
                    _sid, _t, result = await dispatch(
                        conv_session.session_id,
                        request,
                        t_idx,
                        previous_context_by_session.get(conv_session.session_id, 0),
                        request_index=next(counter),
                        scheduled_at_s=arrive_at_s if t_idx == 0 else None,
                    )
                    _record(_sid, _t, result)
                    if result is None or not result.success:
                        break
                    wait_s = _wait_s(request)
                    if wait_s > 0 and t_idx < last:
                        await asyncio.sleep(wait_s)

            await asyncio.gather(*[
                run_open_session(cs, t) for cs, t in zip(sessions, arrivals)
            ])
            done = [r for v in results_by_turn.values() for r in v if r is not None]
            if done and not any(r.success for r in done):
                print(f"ABORT: all {len(done)} requests failed. First error: "
                      f"{_first_error(done)}")
                sys.exit(1)
        elif turn_pacing == "per-session":
            index_of: dict[tuple[int, int], int] = {}
            nxt = 0
            for conv_session in sessions:
                for t_idx in range(_last_turn(len(conv_session.turns)) + 1):
                    index_of[(conv_session.session_id, t_idx)] = nxt
                    nxt += 1
            print(f"  Per-session pacing: {len(sessions)} sessions, "
                  f"{nxt} requests, up to {concurrency} in flight "
                  f"(no cross-session turn barrier)")
            if apply_inter_turn_waits and (
                tool_wait_ms or human_wait_ms or use_recorded_waits
            ):
                print(f"  Inter-turn waits: tool_wait_ms={tool_wait_ms:g} "
                      f"human_wait_ms={human_wait_ms:g} "
                      f"recorded={'on' if use_recorded_waits else 'off'}")
            if use_recorded_arrivals:
                print("  Session starts: recorded arrival_time_ms "
                      "(wait is outside the concurrency slot)")

            async def run_session(conv_session) -> None:
                sid = conv_session.session_id
                if use_recorded_arrivals:
                    await _sleep_until_elapsed(
                        benchmark_start, conv_session.arrival_time_ms / 1000.0
                    )
                last = _last_turn(len(conv_session.turns))
                for t_idx in range(last + 1):
                    request = conv_session.turns[t_idx]
                    _sid, _t, result = await dispatch(
                        sid,
                        request,
                        t_idx,
                        previous_context_by_session.get(sid, 0),
                        request_index=index_of[(sid, t_idx)],
                    )
                    _record(_sid, _t, result)
                    if result is None or not result.success:
                        break
                    wait_s = _wait_s(request)
                    if wait_s > 0 and t_idx < last:
                        await asyncio.sleep(wait_s)

            await asyncio.gather(*[run_session(s) for s in sessions])
            done = sum(len(v) for v in results_by_turn.values())
            ok = sum(1 for v in results_by_turn.values() for r in v
                     if r is not None and r.success)
            if done and ok == 0:
                print(f"ABORT: All {done} requests failed. First error: "
                      f"{_first_error(r for v in results_by_turn.values() for r in v)}")
                sys.exit(1)
        else:
            for turn_idx in range(max_turns):
                if max_turn_index is not None and turn_idx > max_turn_index:
                    break
                turn_requests = []
                for conv_session in sessions:
                    if conv_session.session_id in dead_sessions:
                        continue
                    if turn_idx < len(conv_session.turns):
                        turn_requests.append(
                            (conv_session.session_id, conv_session.turns[turn_idx])
                        )

                if not turn_requests:
                    continue

                print(f"  Turn {turn_idx + 1}/{max_turns}: "
                      f"dispatching {len(turn_requests)} requests...")

                request_offset = sum(len(v) for v in results_by_turn.values())
                tasks = [
                    dispatch(
                        sid,
                        req,
                        turn_idx,
                        previous_context_by_session.get(sid, 0),
                        request_index=request_offset + i,
                    )
                    for i, (sid, req) in enumerate(turn_requests)
                ]
                completed = await asyncio.gather(*tasks)

                turn_ok = sum(1 for _, _, r in completed if r is not None and r.success)
                turn_fail = len(completed) - turn_ok
                if turn_fail == len(completed):
                    print(
                        f"ABORT: All {len(completed)} requests in turn "
                        f"{turn_idx + 1} failed. First error: "
                        f"{_first_error(r for _, _, r in completed)}"
                    )
                    sys.exit(1)

                for sid, t_idx, result in completed:
                    _record(sid, t_idx, result)

    benchmark_duration = time.perf_counter() - benchmark_start

    # Flatten results, tagging each with turn_index
    all_results = []
    for turn_idx in sorted(results_by_turn.keys()):
        for r in results_by_turn[turn_idx]:
            if r.turn_index is None:
                r.turn_index = turn_idx
            all_results.append(r)

    return all_results, results_by_turn, benchmark_duration


def _check_success_rate(summary, min_rate: float):
    """Exit with error if success rate is below the minimum threshold."""
    if summary.num_requests == 0:
        print(f"ABORT: No requests completed. Minimum success rate: {min_rate:.0%}")
        sys.exit(1)
    rate = summary.successful_requests / summary.num_requests
    if rate < min_rate:
        kinds = ", ".join(f"{k}={v}" for k, v in sorted(summary.error_kinds.items()))
        print(f"ABORT: Success rate {rate:.1%} below minimum {min_rate:.0%} "
              f"({summary.successful_requests}/{summary.num_requests}); failures: {kinds}")
        sys.exit(1)


def _run(coro):
    """asyncio.run with a clean exit for cache-reset failures.

    A failed reset is fatal by design: continuing would silently produce the
    warm-cache contamination the flag exists to prevent.
    """
    try:
        return asyncio.run(coro)
    except PrefixCacheResetError as e:
        coro.close()
        print(f"ABORT: prefix cache reset failed: {e}")
        sys.exit(1)


def _check_usage_reported(summary, allow_missing: bool):
    """Exit if no response carried a usage block.

    Without usage, input_tokens/output_tokens stay 0 for every request, so the
    summary records 0 tok/s and 0 total tokens while still looking like a valid
    run. That silently poisons any downstream ISL/OSL or throughput comparison.
    """
    if summary.successful_requests == 0 or summary.usage_reported_requests > 0:
        return
    if allow_missing:
        print("WARNING: no usage block in any response; token counts and throughputs "
              "are zero-filled. Continuing because --allow-missing-usage was passed.")
        return
    print("ABORT: no response carried a usage block, so every token count and "
          "throughput in this run would be 0 rather than measured. The server must "
          "honour stream_options.include_usage. Pass --allow-missing-usage to record "
          "the run anyway (latency metrics stay valid; token metrics do not).")
    sys.exit(1)


def _load_source_session_ids(path: str | None) -> list[str] | None:
    if not path:
        return None
    source_path = Path(path)
    ids = []
    seen = set()
    for line in source_path.read_text(encoding="utf-8").splitlines():
        source_session_id = line.strip()
        if not source_session_id or source_session_id.startswith("#"):
            continue
        if source_session_id in seen:
            continue
        ids.append(source_session_id)
        seen.add(source_session_id)
    if not ids:
        print(f"Error: --source-session-ids-file had no usable IDs: {path}")
        sys.exit(1)
    return ids


def save_results(summary, results, output_path: str, config: dict):
    """Save summary + per-request data to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # Stamp the schema version so result files are self-describing (v4 =
    # fixed streaming client + coverage-guarded tpot). setdefault keeps any
    # explicitly provided value.
    config = dict(config)
    config.setdefault("benchmark_schema_version", BENCHMARK_SCHEMA_VERSION)
    output = {
        "config": config,
        "summary": summary.to_dict(),
        "per_request": [
            {
                "success": r.success,
                "ttft_ms": round(r.ttft * 1000, 2) if r.ttft else None,
                "tpot_ms": round(r.tpot * 1000, 2) if r.tpot else None,
                "itl_ms": [round(t * 1000, 2) for t in r.itl] if r.itl else [],
                "e2el_ms": round(r.e2el * 1000, 2) if r.e2el else None,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "error": r.error,
                **({"error_kind": r.error_kind} if r.error_kind is not None else {}),
                "usage_reported": r.usage_reported,
                **({"excluded_from_summary": True} if r.excluded_from_summary else {}),
                **({"request_index": r.request_index}
                   if r.request_index is not None else {}),
                **({"max_tokens_requested": r.max_tokens_requested}
                   if r.max_tokens_requested is not None else {}),
                **({"message_count": r.message_count}
                   if r.message_count is not None else {}),
                **({"prompt_chars": r.prompt_chars}
                   if r.prompt_chars is not None else {}),
                **({"scheduled_at_ms": round(r.scheduled_at_s * 1000, 2)}
                   if r.scheduled_at_s is not None else {}),
                **({"dispatch_started_at_ms": round(r.dispatch_started_at_s * 1000, 2)}
                   if r.dispatch_started_at_s is not None else {}),
                **({"semaphore_acquired_at_ms": round(r.semaphore_acquired_at_s * 1000, 2)}
                   if r.semaphore_acquired_at_s is not None else {}),
                **({"completed_at_ms": round(r.completed_at_s * 1000, 2)}
                   if r.completed_at_s is not None else {}),
                **({"client_schedule_delay_ms": round(r.client_schedule_delay_s * 1000, 2)}
                   if r.client_schedule_delay_s is not None else {}),
                **({"client_queue_wait_ms": round(r.client_queue_wait_s * 1000, 2)}
                   if r.client_queue_wait_s is not None else {}),
                **({"client_request_wall_ms": round(r.client_request_wall_s * 1000, 2)}
                   if r.client_request_wall_s is not None else {}),
                **({"session_id": r.session_id} if r.session_id is not None else {}),
                **({"turn_index": r.turn_index} if r.turn_index is not None else {}),
                **({"previous_context_tokens": r.previous_context_tokens}
                   if r.previous_context_tokens is not None else {}),
                **({"total_context_tokens": r.total_context_tokens}
                   if r.total_context_tokens is not None else {}),
                **({"new_prefill_tokens": r.new_prefill_tokens}
                   if r.new_prefill_tokens is not None else {}),
                **({"cached_context_tokens": r.cached_context_tokens}
                   if r.cached_context_tokens is not None else {}),
                **({"cache_hit_rate": round(r.cache_hit_rate, 4)}
                   if r.cache_hit_rate is not None else {}),
                **({"cache_estimate_source": r.cache_estimate_source}
                   if r.cache_estimate_source is not None else {}),
                **({"cache_block_size": r.cache_block_size}
                   if r.cache_block_size is not None else {}),
                **({"block_aligned_cached_context_tokens": r.block_aligned_cached_context_tokens}
                   if r.block_aligned_cached_context_tokens is not None else {}),
                **({"block_aligned_new_prefill_tokens": r.block_aligned_new_prefill_tokens}
                   if r.block_aligned_new_prefill_tokens is not None else {}),
                **({"block_aligned_cache_hit_rate": round(r.block_aligned_cache_hit_rate, 4)}
                   if r.block_aligned_cache_hit_rate is not None else {}),
                **({"uncached_prefix_tail_tokens": r.uncached_prefix_tail_tokens}
                   if r.uncached_prefix_tail_tokens is not None else {}),
                **({"total_context_blocks": r.total_context_blocks}
                   if r.total_context_blocks is not None else {}),
                **({"cached_context_blocks": r.cached_context_blocks}
                   if r.cached_context_blocks is not None else {}),
                **({"new_prefill_blocks": r.new_prefill_blocks}
                   if r.new_prefill_blocks is not None else {}),
                **({"request_metadata": r.request_metadata}
                   if r.request_metadata else {}),
            }
            for r in results if r is not None
        ],
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {output_path}")


def _resolve_cache_state(cli_value: str, profile) -> tuple[str, str]:
    if cli_value != "auto":
        return cli_value, "cli"
    return (
        "expected_on" if profile.prefix_caching_required else "expected_off",
        "profile_default",
    )


def _resolve_tri_state(cli_value: str) -> tuple[str, str]:
    if cli_value != "auto":
        return cli_value, "cli"
    return "unknown", "not_reported"


def resolve_multi_turn_num_sessions(
    profile,
    concurrency: int,
    override: int | None = None,
) -> tuple[int, str]:
    """Multi-turn runs need enough sessions to saturate the requested concurrency."""
    if profile.mode != "multi-turn":
        return profile.num_sessions, "profile_default"

    if override is not None:
        effective_num_sessions = max(override, concurrency)
        if effective_num_sessions == override:
            return effective_num_sessions, "cli"
        return effective_num_sessions, "cli_concurrency_floor"

    effective_num_sessions = max(profile.num_sessions, concurrency)
    if effective_num_sessions == profile.num_sessions:
        return effective_num_sessions, "profile_default"
    return effective_num_sessions, "concurrency_floor"


def normalize_dashboard_scope(scope: str) -> str:
    if scope in {"latest", "synthetic", "synthetic-distributional", "synthetic_distributional"}:
        return "synthetic_distributional"
    if scope in {"archive", "trace_replay"}:
        return "trace_replay"
    if scope in {"current", "canonical", "fixed", "fixed-grid", "mse", "archived"}:
        return "archived"
    return scope


def get_args():
    parser = argparse.ArgumentParser(description="inference-benchmark runner")
    parser.add_argument("--url", required=False, help="Server endpoint URL")
    parser.add_argument("--model", required=False)
    parser.add_argument("--backend", default="vllm", choices=SUPPORTED_BACKENDS,
                        help="Backend type (vllm/sglang/openai → /v1/chat/completions)")
    parser.add_argument("--profile", default="chat-singleturn", help="Workload profile name")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--multi-turn-sessions", type=int, default=None,
                        help="Override number of multi-turn sessions to load/sample. Floored at --concurrency.")
    parser.add_argument("--max-turn-index", type=int, default=None,
                        help="For multi-turn runs, stop after this zero-based turn index.")
    parser.add_argument("--source-session-ids-file", default=None,
                        help="Validation mode: source-lock distributional multi-turn sampling to these source_session_id values.")
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--api-key", default="test")
    parser.add_argument("--arrival", default="steady", choices=["steady", "poisson", "ramp"])
    parser.add_argument("--turn-pacing", default="per-session",
                        choices=list(TURN_PACINGS),
                        help="closed-loop multi-turn scheduling: 'per-session' (default) "
                             "runs a session's turns back to back with no cross-session "
                             "barrier; 'interleaved' is a turn-aligned herd (global "
                             "barrier per turn — schedule artifacts in TTFT/TPOT, not "
                             "production traffic). Ignored under --open-loop.")
    parser.add_argument("--tool-wait-ms", type=float, default=0.0,
                        help="Fixed sleep after each turn before the next turn in the "
                             "same session (ms). Default 0 (auto-mode / no tool gap). "
                             "Ignored under --turn-pacing interleaved.")
    parser.add_argument("--human-wait-ms", type=float, default=0.0,
                        help="Extra sleep after each turn for a human-approval gap (ms). "
                             "Default 0. Added to --tool-wait-ms. Ignored under "
                             "--turn-pacing interleaved.")
    parser.add_argument("--use-recorded-waits", action="store_true",
                        help="Add per-turn tool_wait_after_ms / human_wait_after_ms from "
                             "the trajectory (tool_ms / human_ms aliases accepted). "
                             "Default off. Still adds the CLI --tool-wait-ms / "
                             "--human-wait-ms values.")
    parser.add_argument("--use-recorded-arrivals", action="store_true",
                        help="Start each session at its recorded arrival_time_ms instead "
                             "of t=0 (closed-loop) or a generated Poisson/ramp "
                             "(open-loop). Default off. Ignored under interleaved.")
    parser.add_argument("--target-rate", type=float, default=10.0, help="req/s for poisson/ramp")
    load_group = parser.add_mutually_exclusive_group()
    load_group.add_argument(
        "--closed-loop", dest="load_mode", action="store_const", const="closed-loop",
        help="Closed loop (default): hold --concurrency requests in flight, each replaced "
             "only when the previous finishes. Offered load is capped by server speed, so "
             "the server can never fall behind. Implies --arrival steady. Multi-turn "
             "defaults to --turn-pacing per-session.")
    load_group.add_argument(
        "--open-loop", dest="load_mode", action="store_const", const="open-loop",
        help="Open loop: fire arrivals on a clock at --target-rate with NO client-side "
             "concurrency cap, so an overloaded server queues instead of back-pressuring "
             "the client. Requires --arrival poisson|ramp. Single-turn: independent "
             "requests. Multi-turn: session starts are open-loop; turns stay sequential "
             "inside each session.")
    parser.set_defaults(load_mode="closed-loop")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature. Default 0.0 (greedy) so output lengths "
                             "are reproducible across runs.")
    parser.add_argument("--sampling-seed", type=int, default=None,
                        help="Seed forwarded to the server sampler. Defaults to --seed; "
                             "pass -1 to omit it entirely.")
    parser.add_argument("--exact-output-length", action="store_true",
                        help="Pin generated length to the profile's per-request max_tokens "
                             "(sends min_tokens=max_tokens and ignore_eos). Makes measured OSL "
                             "equal planned OSL instead of wherever the model chose to stop.")
    parser.add_argument("--reset-prefix-cache", action="store_true",
                        help="POST the server's prefix-cache reset endpoint before the run. "
                             "Sweeps replay byte-identical prompts across cells, so without "
                             "this every cell after the first is measured warm.")
    parser.add_argument("--warmup-concurrency", type=int, default=None, metavar="N",
                        help="Batch width to warm at. Defaults to --concurrency under "
                             "closed loop (where that IS the in-flight cap), and to "
                             "min(--num-requests, 64) under open loop (where it is not). "
                             "Engines that JIT kernels per batch shape compile mid-run for "
                             "any width warmup missed.")
    parser.add_argument("--discard-first", type=int, default=0, metavar="N",
                        help="Hold the first N requests out of the summary (they stay in "
                             "per_request). Removes the startup wave. Only meaningful once "
                             "--num-requests is several times --concurrency; at 2x there is "
                             "no settled middle left to keep. Single-turn only.")
    parser.add_argument("--allow-missing-usage", action="store_true",
                        help="Permit a run where no response carried a usage block. Token "
                             "counts and throughputs will be zero-filled, not measured.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output", default="results/latest.json")
    parser.add_argument("--max-context-tokens", type=int, default=None,
                        help="Optional cap for distributional multi-turn synthetic prompt context")
    parser.add_argument("--context-safety-margin-tokens", type=int, default=256,
                        help="Reserved token headroom under --max-context-tokens for output and tokenizer mismatch")
    parser.add_argument("--prefix-cache-block-size", type=int, default=16,
                        help="KV prefix-cache block size in tokens for block-aligned cache estimates")
    parser.add_argument("--prefix-caching-state", choices=["auto", "on", "off", "unknown"],
                        default="auto",
                        help="Metadata only: actual server prefix-cache state when known")
    parser.add_argument("--chunked-prefill", choices=["auto", "on", "off", "unknown"],
                        default="auto",
                        help="Metadata only: actual server chunked-prefill state when known")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Metadata only: server --max-model-len")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None,
                        help="Metadata only: server GPU memory utilization target")
    parser.add_argument("--tensor-parallel-size", type=int, default=None,
                        help="Required: server tensor parallel size. Recorded in the result "
                             "config and used to compose the parallelism label (1gpu/tp/tp+ep).")
    parser.add_argument("--dtype", default=None,
                        help="Metadata only: server compute dtype")
    parser.add_argument("--kv-cache-dtype", default=None,
                        help="Metadata only: server KV-cache dtype")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None,
                        help="Metadata only: server max_num_batched_tokens")
    parser.add_argument("--max-num-seqs", type=int, default=None,
                        help="Metadata only: server max_num_seqs")
    parser.add_argument("--ignore-eos", action="store_true",
                        help="Pass ignore_eos=true to vLLM (needed for FP8 models with random token workloads)")
    parser.add_argument("--mode", choices=["stress-test", "single-turn", "multi-turn"],
                        help="Benchmark mode (sets profile defaults and required flags). "
                             "Use --profile for a specific profile within a mode.")
    parser.add_argument(
        "--scope",
        choices=[
            "synthetic_distributional",
            "synthetic-distributional",
            "trace_replay",
            "archived",
            "synthetic",
            "latest",
            "current",
            "canonical",
            "archive",
            "fixed",
            "fixed-grid",
            "mse",
            "moe_ep",
        ],
        default=None,
        help="Dashboard scope override (default: *-synth→synthetic_distributional, active→archived, inactive→trace_replay)",
    )
    parser.add_argument("--min-success-rate", type=float, default=0.75, dest="min_success_rate",
                        help="Minimum success rate (0.0-1.0). Runs below this threshold exit with an error. Default: 0.75")
    parser.add_argument("--list-profiles", action="store_true", help="List available profiles and exit")
    parser.add_argument("--include-inactive", action="store_true",
                        help="With --list-profiles, include legacy/inactive profiles")
    parser.add_argument("--trace-request-ids", action="store_true",
                        help="Send stable request_id/X-Request-Id values for vLLM engine tracing.")
    parser.add_argument("--agent-type", type=str, default=None, help="Filter profiles by agent type")
    parser.add_argument("--turn-style", type=str, default=None, help="Filter profiles by turn style")
    parser.add_argument("--serving-style", type=str, default=None, help="Filter profiles by serving style")
    parser.add_argument("--data-source", type=str, default=None, help="Filter profiles by data source")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.list_profiles:
        from ..workloads.profiles import filter_profiles, PROFILES, AGENT_TYPES, TURN_STYLES, SERVING_STYLES, DATA_SOURCES
        filtered = filter_profiles(
            agent_type=args.agent_type,
            turn_style=args.turn_style,
            serving_style=args.serving_style,
            data_source=args.data_source,
            include_inactive=args.include_inactive,
        )
        print(f"\n{'Name':<30} {'Agent Type':<18} {'Turn Style':<14} {'Serving':<20} {'Data Source':<12} {'ISL':<6} {'OSL':<6}")
        print("-" * 110)
        for name, p in sorted(filtered.items()):
            print(f"{name:<30} {p.agent_type:<18} {p.turn_style:<14} {p.serving_style:<20} {p.data_source:<12} {p.isl_tokens:<6} {p.osl_tokens:<6}")
        inactive_note = " including inactive" if args.include_inactive else ""
        print(f"\n{len(filtered)} profiles shown{inactive_note} (of {len(PROFILES)} total)")
        if any([args.agent_type, args.turn_style, args.serving_style, args.data_source]):
            active = []
            if args.agent_type: active.append(f"agent_type={args.agent_type}")
            if args.turn_style: active.append(f"turn_style={args.turn_style}")
            if args.serving_style: active.append(f"serving_style={args.serving_style}")
            if args.data_source: active.append(f"data_source={args.data_source}")
            print(f"Filters: {', '.join(active)}")
        sys.exit(0)

    # --url and --model are required for actual benchmark runs
    if not args.url or not args.model:
        print("Error: --url and --model are required for benchmark runs.")
        print("Use --list-profiles to browse profiles without a server.")
        sys.exit(1)

    # --tensor-parallel-size is load-bearing (it composes the parallelism label
    # recorded in the result config), so refuse to guess it. Silently defaulting
    # to 1 would mislabel every multi-GPU sweep.
    if args.tensor_parallel_size is None:
        print("Error: --tensor-parallel-size is required for benchmark runs "
              "(pass 1 for single-GPU serving).")
        sys.exit(1)
    if args.tensor_parallel_size < 1:
        print(f"Error: --tensor-parallel-size must be >= 1, got {args.tensor_parallel_size}.")
        sys.exit(1)

    # Load mode and arrival pattern have to agree. "steady" means every request
    # is scheduled at t=0 and the semaphore paces them, which is the definition
    # of closed loop; a rate process only means anything with the cap removed.
    if args.load_mode == "open-loop":
        if args.use_recorded_arrivals:
            # Recorded arrival_time_ms replaces the generated pattern, so the
            # poisson|ramp requirement (and --target-rate) does not apply.
            if args.arrival in OPEN_LOOP_ARRIVALS:
                print("WARNING: --use-recorded-arrivals replays recorded "
                      "arrival_time_ms; --arrival and --target-rate are ignored.")
        elif args.arrival not in OPEN_LOOP_ARRIVALS:
            print(f"Error: --open-loop requires --arrival {'|'.join(OPEN_LOOP_ARRIVALS)}, "
                  f"got '{args.arrival}'. 'steady' schedules everything at t=0, which is "
                  f"closed-loop by construction. Multi-turn trajectory replays can "
                  f"exempt this with --use-recorded-arrivals.")
            sys.exit(1)
        elif args.target_rate <= 0:
            print(f"Error: --open-loop requires a positive --target-rate, got {args.target_rate}.")
            sys.exit(1)
    elif args.arrival in OPEN_LOOP_ARRIVALS:
        print(f"Error: --arrival {args.arrival} describes open-loop offered load, but the run "
              f"is closed-loop, so --concurrency would still cap it. Pass --open-loop to "
              f"remove the cap, or use --arrival steady.")
        sys.exit(1)

    if args.tool_wait_ms < 0 or args.human_wait_ms < 0:
        print("Error: --tool-wait-ms and --human-wait-ms must be >= 0.")
        sys.exit(1)

    # Warmup width. Under closed loop --concurrency is the real in-flight cap, so
    # it is the right width. Under open loop it caps nothing, so sizing warmup
    # from it can miss the batch shapes the run reaches -- default to the upper
    # bound on in-flight (num_requests), capped to keep warmup affordable.
    if args.warmup_concurrency is None:
        if args.load_mode == "open-loop":
            args.warmup_concurrency = max(1, min(args.num_requests, 64))
        else:
            args.warmup_concurrency = args.concurrency
    elif args.warmup_concurrency < 1:
        print(f"Error: --warmup-concurrency must be >= 1, got {args.warmup_concurrency}.")
        sys.exit(1)

    # -1 is the explicit opt-out; otherwise the sampler shares the workload seed
    # so identical prompts produce identical generations.
    if args.sampling_seed is None:
        args.sampling_seed = args.seed
    elif args.sampling_seed < 0:
        args.sampling_seed = None

    if args.mode:
        if args.mode == "multi-turn":
            print("NOTE: multi-turn mode requires server launched with --enable-prefix-caching (vLLM)")
            if args.profile == "chat-singleturn":  # default — override for multi-turn
                args.profile = "chat-multiturn"
        if args.mode == "stress-test":
            if not args.ignore_eos:
                print("NOTE: stress-test mode auto-enables --ignore-eos (required for FP8 models)")
                args.ignore_eos = True
            if args.profile == "chat-singleturn":  # default — override for stress-test
                args.profile = "random-1k"
        if args.mode == "single-turn":
            print("NOTE: single-turn mode requires server launched with --enable-prefix-caching (vLLM)")
            print("      or radix cache (SGLang default). See scripts/launch_server.sh")

    profile = get_profile(args.profile)
    profile_name = profile.name
    if args.discard_first < 0:
        print(f"Error: --discard-first must be >= 0, got {args.discard_first}.")
        sys.exit(1)
    if args.discard_first and profile.mode == "multi-turn":
        print(f"Error: --discard-first is not supported for multi-turn profile "
              f"'{profile_name}'. Every turn has its own barrier transient, so a global "
              f"first-N has no consistent meaning; use the per-turn breakdown instead.")
        sys.exit(1)
    if args.discard_first >= args.num_requests and profile.mode != "multi-turn":
        print(f"Error: --discard-first {args.discard_first} would hold out every request "
              f"of {args.num_requests}.")
        sys.exit(1)
    if args.multi_turn_sessions is not None and args.multi_turn_sessions <= 0:
        print("Error: --multi-turn-sessions must be positive when provided.")
        sys.exit(1)
    effective_num_sessions, num_sessions_source = resolve_multi_turn_num_sessions(
        profile,
        args.concurrency,
        args.multi_turn_sessions,
    )
    source_session_ids = _load_source_session_ids(args.source_session_ids_file)
    if source_session_ids is not None and profile.dataset != "distributional-multi-turn":
        print("--source-session-ids-file is only valid for distributional multi-turn profiles.")
        sys.exit(1)
    if source_session_ids is not None:
        effective_num_sessions = len(source_session_ids)
        num_sessions_source = "source_session_ids_file"
    prefix_caching_state, prefix_caching_state_source = _resolve_cache_state(
        args.prefix_caching_state,
        profile,
    )
    chunked_prefill_state, chunked_prefill_state_source = _resolve_tri_state(
        args.chunked_prefill,
    )
    scope = args.scope
    if scope is None:
        scope = "synthetic_distributional" if profile_name.endswith("-synth") else ("archived" if profile.active else "trace_replay")
    else:
        scope = normalize_dashboard_scope(scope)
    # Expert parallelism is enabled by the launcher via ENABLE_EP (moe_ep scope).
    # Record it explicitly so EP-on runs are labelled in the result data itself,
    # not just inferred from the scope. ep_size mirrors the launcher's --ep-size $TP.
    enable_ep = str(os.environ.get("ENABLE_EP", "")).strip().lower() in {"1", "true", "on", "yes"}
    ep_size = args.tensor_parallel_size if enable_ep else 1
    # Canonical parallelism-strategy label composed from the active axes:
    # "1gpu" (single GPU), "tp", "tp+ep". Mirrors compile_sweep.parallelism_label.
    _par_axes = (["tp"] if args.tensor_parallel_size > 1 else []) + (["ep"] if enable_ep else [])
    parallelism = "+".join(_par_axes) or "1gpu"
    config = {
        **vars(args),
        "profile": profile_name,
        "mode": args.mode or profile.mode,
        "dashboard_scope": scope,
        "enable_ep": enable_ep,
        "ep_size": ep_size,
        "parallelism": parallelism,
        "turn_pacing": args.turn_pacing,
        "tool_wait_ms": args.tool_wait_ms,
        "human_wait_ms": args.human_wait_ms,
        "use_recorded_waits": args.use_recorded_waits,
        "use_recorded_arrivals": args.use_recorded_arrivals,
        "reply_feedback": True,
        "load_mode": args.load_mode,
        "sampling": {
            "temperature": args.temperature,
            "sampling_seed": args.sampling_seed,
            "exact_output_length": args.exact_output_length,
            "ignore_eos": args.ignore_eos or args.exact_output_length,
            "output_length_controlled": args.exact_output_length,
        },
        "prefix_cache_reset_before_run": args.reset_prefix_cache,
        "warmup_style": "profile_shaped_at_concurrency" if args.warmup else "none",
        "discard_first": args.discard_first,
        "warmup_concurrency": args.warmup_concurrency,
        "requests_per_concurrency_slot": (
            round(args.num_requests / args.concurrency, 2) if args.concurrency else None
        ),
        "profile_metadata": {
            "dataset": profile.dataset,
            "agent_type": profile.agent_type,
            "turn_style": profile.turn_style,
            "serving_style": profile.serving_style,
            "data_source": profile.data_source,
            "active": profile.active,
            "prefix_caching_required": profile.prefix_caching_required,
            "isl_tokens": profile.isl_tokens,
            "osl_tokens": profile.osl_tokens,
            "min_turns": profile.min_turns,
            "max_turns": profile.max_turns,
            "num_sessions": effective_num_sessions,
            "profile_num_sessions": profile.num_sessions,
            "num_sessions_source": num_sessions_source,
            "source_session_ids_count": (
                len(source_session_ids) if source_session_ids is not None else None
            ),
        },
        "prediction_metadata": {
            "prefix_caching_state": prefix_caching_state,
            "prefix_caching_state_source": prefix_caching_state_source,
            "prefix_cache_block_size": args.prefix_cache_block_size,
            "chunked_prefill": chunked_prefill_state,
            "chunked_prefill_source": chunked_prefill_state_source,
            "max_context_tokens": args.max_context_tokens,
            "context_safety_margin_tokens": args.context_safety_margin_tokens,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.dtype,
            "kv_cache_dtype": args.kv_cache_dtype,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "logical_cache_estimate": "previous_prompt_tokens",
            "block_aligned_cache_estimate": "floor(previous_context / block_size) * block_size",
            "engine_cache_telemetry": "not_available",
        },
    }

    if profile.mode == "multi-turn":
        all_results, results_by_turn, duration = _run(run_multi_turn_benchmark(
            url=args.url,
            model=args.model,
            profile_name=profile_name,
            concurrency=args.concurrency,
            backend_name=args.backend,
            api_key=args.api_key,
            warmup_requests=args.warmup,
            timeout=args.timeout,
            ignore_eos=args.ignore_eos,
            max_context_tokens=args.max_context_tokens,
            context_safety_margin_tokens=args.context_safety_margin_tokens,
            seed=args.seed,
            cache_block_size=args.prefix_cache_block_size,
            num_sessions=effective_num_sessions,
            source_session_ids=source_session_ids,
            turn_pacing=args.turn_pacing,
            max_turn_index=args.max_turn_index,
            trace_request_ids=args.trace_request_ids,
            load_mode=args.load_mode,
            arrival_pattern=args.arrival,
            target_rate=args.target_rate,
            tool_wait_ms=args.tool_wait_ms,
            human_wait_ms=args.human_wait_ms,
            use_recorded_waits=args.use_recorded_waits,
            use_recorded_arrivals=args.use_recorded_arrivals,
            temperature=args.temperature,
            sampling_seed=args.sampling_seed,
            exact_output_length=args.exact_output_length,
            reset_prefix_cache_first=args.reset_prefix_cache,
            warmup_concurrency=args.warmup_concurrency,
        ))

        summary = aggregate(
            results=[r for r in all_results if r is not None],
            duration_s=duration,
            model=args.model,
            profile=profile_name,
            concurrency=args.concurrency,
            load_mode=args.load_mode,
            target_rate=args.target_rate if args.load_mode == "open-loop" else 0.0,
            warmup_concurrency=args.warmup_concurrency if args.warmup else 0,
        )

        turn_summaries = aggregate_per_turn(results_by_turn)
        print_multi_turn_summary(turn_summaries, summary)
        _check_success_rate(summary, args.min_success_rate)
        _check_usage_reported(summary, args.allow_missing_usage)
        save_results(summary, all_results, args.output, config)

        # Also save per-turn breakdown
        turn_output = args.output.replace(".json", "_per_turn.json")
        import json as json_mod
        from pathlib import Path as PathMod
        PathMod(turn_output).parent.mkdir(parents=True, exist_ok=True)
        with open(turn_output, "w") as f:
            json_mod.dump({
                "config": config,
                "per_turn": [ts.to_dict() for ts in turn_summaries],
            }, f, indent=2)
        print(f"Per-turn results saved to: {turn_output}")

    else:
        if (args.use_recorded_arrivals or args.use_recorded_waits
                or args.tool_wait_ms or args.human_wait_ms):
            print("WARNING: --use-recorded-arrivals / --use-recorded-waits / "
                  "--tool-wait-ms / --human-wait-ms apply to multi-turn "
                  "profiles only; ignoring them.")
        results, duration = _run(run_benchmark(
            url=args.url,
            model=args.model,
            profile_name=profile_name,
            concurrency=args.concurrency,
            num_requests=args.num_requests,
            backend_name=args.backend,
            api_key=args.api_key,
            arrival_pattern=args.arrival,
            target_rate=args.target_rate,
            warmup_requests=args.warmup,
            seed=args.seed,
            timeout=args.timeout,
            ignore_eos=args.ignore_eos,
            max_context_tokens=args.max_context_tokens,
            context_safety_margin_tokens=args.context_safety_margin_tokens,
            trace_request_ids=args.trace_request_ids,
            load_mode=args.load_mode,
            temperature=args.temperature,
            sampling_seed=args.sampling_seed,
            exact_output_length=args.exact_output_length,
            reset_prefix_cache_first=args.reset_prefix_cache,
            warmup_concurrency=args.warmup_concurrency,
        ))

        # Hold out the startup wave by dispatch order. Requests keep their data
        # in per_request; they simply stop feeding the summary.
        if args.discard_first:
            for r in results[:args.discard_first]:
                if r is not None:
                    r.excluded_from_summary = True

        summary = aggregate(
            results=[r for r in results if r is not None],
            duration_s=duration,
            model=args.model,
            profile=profile_name,
            concurrency=args.concurrency,
            load_mode=args.load_mode,
            target_rate=args.target_rate if args.load_mode == "open-loop" else 0.0,
            warmup_concurrency=args.warmup_concurrency if args.warmup else 0,
        )

        print_summary(summary)
        _warn_if_transient_dominated(args.num_requests, args.concurrency)
        _check_success_rate(summary, args.min_success_rate)
        _check_usage_reported(summary, args.allow_missing_usage)
        save_results(summary, results, args.output, config)
