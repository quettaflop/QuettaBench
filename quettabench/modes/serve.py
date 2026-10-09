"""
Benchmark runner — orchestrates a full benchmark run.

Usage:
    python -m quettabench.modes.serve \
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
import dataclasses
import itertools
import sys
import time

from contextlib import asynccontextmanager

from ..metrics.core import (
    aggregate,
    aggregate_per_turn,
    annotate_multi_turn_cache_estimate,
    annotate_request_observability,
    first_error,
    print_summary,
    print_multi_turn_summary,
    success_rate_gate,
    transient_gate,
    usage_gate,
)
from ..metrics.results import write_legacy_per_turn, write_legacy_results
from ..engines.drivers import PrefixCacheResetError
from ..workloads.profiles import get_profile
from ..workloads.dataset import make_dataset
from ..workloads.sessions import (
    _sleep_until_elapsed,
    inter_turn_wait_s,
)
from ..workloads.arrival import make_arrival_times


from .legacy_cli import TURN_PACINGS, apply_mode, build_config, check_args, get_args, list_profiles
WORKLOAD_SCHEMA_VERSION = "distributional-synthetic-v1"
TRACE_REQUEST_ID_PREFIX = "agenticbench"



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
    Trace datasets are rewound by the caller afterwards.
    Results are discarded, but failures are counted: a warmup where nothing
    succeeds means the server is not usable and the run should not proceed.

    Returns (successful, attempted).
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def one():
        request = dataset.get_next_request()
        token_id_kwargs = {}
        if request.metadata.get("prompt_token_ids"):
            token_id_kwargs["prompt_token_ids"] = request.metadata["prompt_token_ids"]
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
                **token_id_kwargs,
            )

    results = await asyncio.gather(*[one() for _ in range(count)], return_exceptions=True)
    ok = sum(
        1 for r in results
        if not isinstance(r, BaseException) and r is not None and r.success
    )
    return ok, len(results)


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
    workload_file: str = "",
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
    if workload_file:
        profile = dataclasses.replace(profile, file_path=workload_file)
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
            # Rewind trace datasets so request i still maps to record i.
            reset = getattr(dataset, "reset", None)
            if callable(reset):
                reset()

        # Reset AFTER warmup so the run starts from a genuinely cold cache
        # regardless of what warmup left behind.
        if reset_prefix_cache_first:
            status = await backend.reset(session, url, api_key, timeout)
            print(f"Prefix cache: {status}")

        # Schedule requests
        semaphore = asyncio.Semaphore(concurrency)
        results = [None] * num_requests
        benchmark_start = time.perf_counter()

        async def dispatch(i: int, dispatch_time: float):
            # Fetch before sleeping: pre-await code runs in creation order,
            # so request i gets record i, which trace replay relies on.
            request = dataset.get_next_request()
            now = time.perf_counter() - benchmark_start
            delay = dispatch_time - now
            if delay > 0:
                await asyncio.sleep(delay)

            token_id_kwargs = {}
            if request.metadata.get("prompt_token_ids"):
                token_id_kwargs["prompt_token_ids"] = request.metadata["prompt_token_ids"]
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
                    **token_id_kwargs,
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

    connector = aiohttp.TCPConnector(limit=0 if open_loop else concurrency + 10)
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
            status = await backend.reset(session_http, url, api_key, timeout)
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
                      f"{first_error(done)}")
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
                      f"{first_error(r for v in results_by_turn.values() for r in v)}")
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
                        f"{first_error(r for _, _, r in completed)}"
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


def _enforce(gate) -> None:
    """Print a metrics.core gate's message; exit 1 when its verdict is abort."""
    verdict, message = gate
    if message:
        print(message)
    if verdict == "abort":
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


def run_multi(config, args, profile_name, num_sessions, source_session_ids) -> None:
    """Multi-turn run: dispatch, summarize per turn, gate, save both files."""
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
        num_sessions=num_sessions,
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
    _enforce(success_rate_gate(summary, args.min_success_rate))
    _enforce(usage_gate(summary, args.allow_missing_usage))
    write_legacy_results(summary, all_results, args.output, config)
    write_legacy_per_turn(turn_summaries, args.output, config)


def run_single(config, args, profile_name) -> None:
    """Single-turn run: dispatch, hold out --discard-first, gate, save."""
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
        workload_file=args.workload_file,
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
    _enforce(transient_gate(args.num_requests, args.concurrency))
    _enforce(success_rate_gate(summary, args.min_success_rate))
    _enforce(usage_gate(summary, args.allow_missing_usage))
    write_legacy_results(summary, results, args.output, config)


def main(argv=None) -> None:
    """The old runner CLI: parse, check, run the profile's mode, save."""
    args = get_args(argv)
    if args.list_profiles:
        list_profiles(args)
        sys.exit(0)
    check_args(args)
    apply_mode(args)
    config, profile, num_sessions, source_session_ids = build_config(args)
    if profile.mode == "multi-turn":
        run_multi(config, args, profile.name, num_sessions, source_session_ids)
    else:
        run_single(config, args, profile.name)


if __name__ == "__main__":
    main()
