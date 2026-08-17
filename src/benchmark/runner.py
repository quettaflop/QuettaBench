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
        --api-key test \
        --output results/run_001.json

    # TRT-LLM (point URL at /generate_stream):
    python -m src.benchmark.runner \
        --url http://localhost:8000/generate_stream \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --backend trtllm \
        --profile chat-singleturn \
        --concurrency 10 \
        --num-requests 100
"""

import asyncio
import argparse
import json
import sys
import time
import os
from pathlib import Path

from .metrics import (
    aggregate,
    aggregate_per_turn,
    annotate_multi_turn_cache_estimate,
    annotate_request_observability,
    print_summary,
    print_multi_turn_summary,
)
from ..workloads.profiles import get_profile
from ..workloads.dataset import make_dataset
from ..workloads.arrival import make_arrival_times


SUPPORTED_BACKENDS = ["openai", "vllm", "sglang", "trtllm"]
# v4 (2026-07-22): streaming client counts reasoning_content/reasoning/tool_calls
# deltas as token events, and tpot falls back to wall-clock when ITL chunk
# coverage is incomplete. v3 and earlier vllm gpt-oss results carry inter-chunk
# tpot (see QuettaSim tools/GT_QUALITY_FLAGS.md Finding 1).
BENCHMARK_SCHEMA_VERSION = 4
WORKLOAD_SCHEMA_VERSION = "distributional-synthetic-v1"
TRACE_REQUEST_ID_PREFIX = "agenticbench"


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
):
    """
    Run a benchmark and return (results, duration).
    """
    import aiohttp
    from ..engines import get_backend

    backend = get_backend(backend_name)
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

    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    client_timeout = aiohttp.ClientTimeout(total=timeout)

    async with aiohttp.ClientSession(connector=connector, timeout=client_timeout) as session:
        # Warmup
        if warmup_requests > 0:
            print(f"Warming up with {warmup_requests} requests...")
            await backend.run_warmup(url, model, api_key, warmup_requests, timeout)
            print("Warmup done.")

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
            async with semaphore:
                semaphore_acquired_at_s = time.perf_counter() - benchmark_start
                result = await backend.send_request(
                    session=session,
                    url=url,
                    model=model,
                    messages=request.messages,
                    max_tokens=request.max_tokens,
                    api_key=api_key,
                    ignore_eos=ignore_eos,
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
    """The first recorded error, for an abort message that names a cause.

    Both abort paths used to say only "Server may not be functional", which is a
    guess and was wrong the one time it mattered: a client-side NameError in the
    streaming loop was caught by `send_request`'s blanket `except`, turned into
    `success=False`, and reported as a server problem. The error text is already
    on every RequestResult; printing it costs nothing and points at the right half
    of the system.
    """
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
    turn_pacing: str = "interleaved",
    pin_output_tokens: bool = True,
):
    """
    Run a multi-turn benchmark. Two scheduling modes, measuring different things.

    ``turn_pacing="interleaved"`` (default, unchanged behaviour): dispatch every
    session's turn N, wait for all of them, then turn N+1.

        [A1, B1, C1] -> barrier -> [A2, B2, C2] -> barrier -> ...

    A session's turns are separated by every other session's turn, which forces KV
    eviction in between and so measures prefix reuse under memory pressure. It is
    also a *barrier*: every turn is a synchronised herd of `concurrency` requests, so
    queueing delay — and therefore TTFT — grows with concurrency at every turn.

    ``turn_pacing="per-session"``: each session runs its own turns back to back, and
    sessions run concurrently against the semaphore.

        A1 -> A2 -> A3 ...   concurrently with   B1 -> B2 -> B3 ...

    This is what a real chat client does: turn N+1 is sent when *that* session's
    reply arrives, not when every other session has finished. Consequences to be
    aware of before comparing the two modes:

      * A session's KV is usually still resident when its next turn arrives, so this
        measures *warm* prefix reuse rather than reuse after eviction.
      * There is no herd after turn 0. Arrivals are throttled by completions, so the
        server rarely sees more than a couple of concurrent prefills and TTFT stops
        growing with concurrency — the load shows up in TPOT and E2EL instead. A
        near-flat TTFT curve here is the mode working, not a broken run.

    The mode is recorded in the result metadata, because the two are not comparable
    and a number without it cannot be placed.

    **The assistant turns are the engine's own output.** Whatever it generates for
    turn N is what turn N+1 sends back, because that is what a chat client does and
    what the engine already holds KV for -- so turn N+1 recomputes only the new user
    message. There is no option to send invented assistant text instead: the engine
    never generated it, so it is not in the prefix cache, and every turn then
    recomputes a reply a real client would have had cached.
    actually produced goes back into the transcript, which is what a chat client
    sends and what the engine already holds KV for — so turn N+1 recomputes only
    the new user message.

    Measured on B200/Llama-3.1-8B chat, C=1: 41.7 tokens recomputed per turn, against
    230 when the assistant text was invented -- and 41.7 is the new user message, all
    of it. Agentic profiles move only 2-8%, because there most of a turn's new prefill
    stands in for a tool result, which a real agent loop must prefill too: nothing
    generated it either. That is the one place invented text belongs, and it is still
    used for it.

    The cost is determinism. Planned context lengths assume the reply is exactly
    ``output_tokens`` long; a real reply is whatever the model produced, and stops
    at EOS unless ``ignore_eos``. So the *shape* of the workload stops being
    reproducible across models and becomes a property of the run — which is why
    this is off by default, and why the per-request records report the context that
    actually happened rather than the plan.

    Returns (results_by_turn, duration) where results_by_turn is a dict
    mapping turn_index → list[RequestResult].
    """
    if turn_pacing not in ("interleaved", "per-session"):
        raise ValueError(
            f"turn_pacing must be 'interleaved' or 'per-session', got {turn_pacing!r}"
        )
    import aiohttp
    from ..engines import get_backend

    from ..workloads.dataset import (
        DistributionalMultiTurnDataset,
        ShareGPTMultiTurnDataset,
        TrajectoryMultiTurnDataset,
    )

    backend = get_backend(backend_name)
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

    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    client_timeout = aiohttp.ClientTimeout(total=timeout)

    async with aiohttp.ClientSession(connector=connector, timeout=client_timeout) as session_http:
        # Warmup
        if warmup_requests > 0:
            print(f"Warming up with {warmup_requests} requests...")
            await backend.run_warmup(url, model, api_key, warmup_requests, timeout)
            print("Warmup done.")

        semaphore = asyncio.Semaphore(concurrency)
        # results_by_turn[turn_idx] = list of RequestResult
        results_by_turn: dict[int, list] = {i: [] for i in range(max_turns)}
        previous_context_by_session: dict[int, int] = {}
        # How many tokens the engine generated for this session's previous turn.
        # With reply feedback that output is part of the reusable prefix, because
        # the engine still holds its KV.
        previous_output_by_session: dict[int, int] = {}
        sessions_by_id = {s.session_id: s for s in sessions}
        # Whether feedback actually happened, so the result can say so rather than
        # leaving it to be inferred from the flag that was asked for.
        nonlocal_state = {"warned": False}
        benchmark_start = time.perf_counter()

        async def dispatch(
            session_id: int,
            request,
            t_idx: int,
            previous_context_tokens: int,
            request_index: int,
        ):
            dispatch_started_at_s = time.perf_counter() - benchmark_start
            async with semaphore:
                semaphore_acquired_at_s = time.perf_counter() - benchmark_start
                result = await backend.send_request(
                    session=session_http,
                    url=url,
                    model=model,
                    messages=request.messages,
                    max_tokens=request.max_tokens,
                    api_key=api_key,
                    ignore_eos=ignore_eos,
                    capture_text=True,
                    min_tokens=request.max_tokens if pin_output_tokens else None,
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
                scheduled_at_s=None,
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
                # Per session, not per run: whether the reply is reusable depends on
                # whether it was the engine's own, and a trace-replay dataset's is not.
                reply_in_cache=bool(
                    getattr(sessions_by_id.get(session_id), "assistant_messages", None)),
            )
            return session_id, t_idx, result

        def _last_turn(n_turns: int) -> int:
            """Highest turn index to run, honouring --max-turn-index."""
            last = n_turns - 1
            return last if max_turn_index is None else min(last, max_turn_index)

        # A session whose turn failed has no reply for it, so the transcript from
        # that point on is missing an assistant message and every later prompt would
        # be short by it. per-session already breaks its own loop; interleaved needs
        # to be told.
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
            """Put the engine's own reply where the synthetic one was.

            The assistant dict is shared by every later turn's message list
            (`list(messages)` is a shallow copy), so one assignment fixes the whole
            remaining transcript. Both pacing modes reach here before the next turn
            is dispatched: per-session because a session's turns are sequential,
            interleaved because the barrier waits for all of turn N first.

            An empty reply is left alone rather than written through. A model can
            legitimately return nothing, and replacing planned text with "" would
            silently shorten every subsequent prompt — a workload change disguised
            as a fidelity improvement.
            """
            session = sessions_by_id.get(sid)
            if session is None:
                return
            if not getattr(session, "assistant_messages", None):
                # Degrade, loudly and exactly once. Raising would make every
                # trajectory profile unrunnable now that this is the default; saying
                # nothing is worse, because reply feedback doing nothing looks
                # identical to it working -- the only symptom is prefill numbers 4.6x
                # too high, which is what the flag exists to remove.
                if not nonlocal_state["warned"]:
                    nonlocal_state["warned"] = True
                    print("WARNING: reply feedback is on but this dataset exposes no "
                          "assistant placeholders, so replies are NOT fed back. Its "
                          "assistant turns come from a trace, where substituting the "
                          "model's output would stop it replaying the trace.",
                          flush=True)
                return
            text = result.generated_text
            if not text:
                return
            if t_idx < len(session.assistant_messages):
                session.assistant_messages[t_idx]["content"] = text

        if turn_pacing == "interleaved":
            # All sessions' turn N, barrier, then turn N+1.
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
        else:
            # per-session: one coroutine per session, its turns strictly sequential.
            # No barrier between sessions, so a fast session pulls ahead and the
            # cohort desynchronises — which is the point.
            #
            # request_index is assigned up front rather than from a running count of
            # completions: with sessions interleaving freely there is no deterministic
            # completion order, and an index derived from one would not be stable
            # across runs of the same seed.
            index_of: dict[tuple[int, int], int] = {}
            nxt = 0
            for conv_session in sessions:
                for t_idx in range(_last_turn(len(conv_session.turns)) + 1):
                    index_of[(conv_session.session_id, t_idx)] = nxt
                    nxt += 1
            print(f"  Per-session pacing: {len(sessions)} sessions, "
                  f"{nxt} requests, up to {concurrency} in flight")

            async def run_session(conv_session) -> None:
                sid = conv_session.session_id
                for t_idx in range(_last_turn(len(conv_session.turns)) + 1):
                    # Read the running context *now*: it is set by this session's own
                    # previous turn, which has already completed because this loop is
                    # sequential.
                    _sid, _t, result = await dispatch(
                        sid,
                        conv_session.turns[t_idx],
                        t_idx,
                        previous_context_by_session.get(sid, 0),
                        request_index=index_of[(sid, t_idx)],
                    )
                    _record(_sid, _t, result)
                    if result is None or not result.success:
                        # The next turn's prompt is this reply's continuation; without
                        # it the session cannot be extended, so stop rather than
                        # sending a turn whose prefix never existed.
                        break

            await asyncio.gather(*[run_session(s) for s in sessions])

            done = sum(len(v) for v in results_by_turn.values())
            ok = sum(1 for v in results_by_turn.values() for r in v
                     if r is not None and r.success)
            if done and ok == 0:
                print(f"ABORT: All {done} requests failed. First error: "
                      f"{_first_error(r for v in results_by_turn.values() for r in v)}")
                sys.exit(1)

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
        print(f"ABORT: Success rate {rate:.1%} below minimum {min_rate:.0%} "
              f"({summary.successful_requests}/{summary.num_requests})")
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
                        help="Backend type (vllm/sglang/openai → /v1/chat/completions, trtllm → /generate_stream)")
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
    parser.add_argument("--turn-pacing", default="interleaved",
                        choices=["interleaved", "per-session"],
                        help="multi-turn scheduling. 'interleaved' (default) "
                             "barriers every turn across sessions, forcing KV "
                             "eviction between a session's turns. 'per-session' "
                             "runs each session's turns back to back with no "
                             "barrier, as a real chat client does. Not "
                             "comparable to each other; see "
                             "run_multi_turn_benchmark.")
    parser.add_argument("--no-pin-output-tokens", dest="pin_output_tokens",
                        action="store_false",
                        help="stop sending min_tokens = max_tokens. Pinning is the "
                             "default: without it the model ends each turn wherever "
                             "it likes, so the planned context lengths describe a run "
                             "that did not happen and two runs of one config are not "
                             "the same workload (195.9 asked vs 185.7 produced on "
                             "B200/Llama-3.1-8B). Per turn, from that turn's own "
                             "sample, so the distribution over output lengths is "
                             "unchanged -- only the early stop is removed.")
    parser.add_argument("--target-rate", type=float, default=10.0, help="req/s for poisson/ramp")
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
                        help="Metadata only: server tensor parallel size")
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
        # Recorded because each changes what the benchmark measures, and an
        # unrecorded knob makes old results unidentifiable: telling whether an
        # earlier run was barriered took the commit date of the flag that
        # introduced pacing, since the snapshot did not say.
        "turn_pacing": args.turn_pacing,
        "pin_output_tokens": args.pin_output_tokens,
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
        all_results, results_by_turn, duration = asyncio.run(run_multi_turn_benchmark(
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
            pin_output_tokens=args.pin_output_tokens,
            max_turn_index=args.max_turn_index,
            trace_request_ids=args.trace_request_ids,
        ))

        summary = aggregate(
            results=[r for r in all_results if r is not None],
            duration_s=duration,
            model=args.model,
            profile=profile_name,
            concurrency=args.concurrency,
        )

        turn_summaries = aggregate_per_turn(results_by_turn)
        print_multi_turn_summary(turn_summaries, summary)
        _check_success_rate(summary, args.min_success_rate)
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
        results, duration = asyncio.run(run_benchmark(
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
        ))

        summary = aggregate(
            results=[r for r in results if r is not None],
            duration_s=duration,
            model=args.model,
            profile=profile_name,
            concurrency=args.concurrency,
        )

        print_summary(summary)
        _check_success_rate(summary, args.min_success_rate)
        save_results(summary, results, args.output, config)
