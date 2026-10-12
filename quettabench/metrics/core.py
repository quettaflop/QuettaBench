"""Aggregates request results into BenchmarkSummary: TTFT, TPOT, ITL and E2EL percentiles over successful requests, failures counted separately, input and output token throughput reported apart."""

import re
import statistics
from dataclasses import dataclass, field


@dataclass
class RequestResult:
    """Per-request benchmark result, shared by every backend."""
    success: bool
    ttft: float | None = None          # seconds to first token
    itl: list = field(default_factory=list)  # inter-token latencies (seconds)
    e2el: float | None = None          # end-to-end latency (seconds)
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None
    # A timeout means the client cut the stream, so the slowest requests are missing from the latency tail.
    error_kind: str | None = None      # "timeout" | "http_error" | "client_error"
    # Without a usage block the token counts are 0 and throughputs are meaningless.
    usage_reported: bool = False
    # Held out of the summary by --discard-first; still kept in per_request.
    excluded_from_summary: bool = False

    # Captured only when asked, to splice the engine reply into later turns; not written to the result JSON.
    generated_text: str | None = None
    # First-token rule: the chat endpoint's "first_payload_chunk" or completions' "first_nonempty_text".
    ttft_rule: str | None = None
    # TTFT start: "slot" (closed loop, the concurrency slot grant) or "arrival" (open loop, the scheduled arrival).
    ttft_basis: str | None = None
    # Leading generated tokens equal to the trace's reference reply; None without output_token_ids and returned tokens.
    accepted_tokens: int | None = None
    # Generated token ids, when the driver returns them. Not written.
    generated_token_ids: list | None = None
    # {ctx, bs, tp, step_point, phase, attempt} for grid runs, else None.
    cell: dict | None = None
    # Wall time of the one generate() the whole batch ran in; every row of the batch carries it.
    batch_wall_s: float | None = None
    # mode.returns=logits: {argmax, top2_gap, logprob}, entry i scoring prompt token i + 1.
    logits: dict | None = None

    # Client-side request shape and timing.
    request_index: int | None = None
    max_tokens_requested: int | None = None
    message_count: int | None = None
    prompt_chars: int | None = None
    scheduled_at_s: float | None = None
    dispatch_started_at_s: float | None = None
    semaphore_acquired_at_s: float | None = None
    completed_at_s: float | None = None
    client_schedule_delay_s: float | None = None
    client_queue_wait_s: float | None = None
    client_request_wall_s: float | None = None

    turn_index: int | None = None      # multi-turn: which turn (0-indexed)
    session_id: int | None = None      # multi-turn: conversation/session id

    # Multi-turn cache estimate from per-session prompt-token deltas, not engine-reported prefix-cache hits.
    previous_context_tokens: int | None = None
    total_context_tokens: int | None = None
    new_prefill_tokens: int | None = None
    cached_context_tokens: int | None = None
    cache_hit_rate: float | None = None
    cache_estimate_source: str | None = None
    cache_block_size: int | None = None
    block_aligned_cached_context_tokens: int | None = None
    block_aligned_new_prefill_tokens: int | None = None
    block_aligned_cache_hit_rate: float | None = None
    uncached_prefix_tail_tokens: int | None = None
    total_context_blocks: int | None = None
    cached_context_blocks: int | None = None
    new_prefill_blocks: int | None = None
    request_metadata: dict = field(default_factory=dict)

    @property
    def tpot(self) -> float | None:
        """Mean ITL, excluding the first token; falls back to (e2el - ttft) / (output_tokens - 1) when ITL is missing or covers under 80% of the tokens (multi-token chunks make it inter-chunk)."""
        wall_clock_ok = (
            self.e2el is not None and self.ttft is not None
            and self.output_tokens > 1
        )
        if self.itl:
            incomplete = (
                self.output_tokens > 1
                and len(self.itl) < 0.8 * (self.output_tokens - 1)
            )
            if incomplete and wall_clock_ok:
                return (self.e2el - self.ttft) / (self.output_tokens - 1)
            return sum(self.itl) / len(self.itl)
        if wall_clock_ok:
            decode_time = self.e2el - self.ttft
            return decode_time / (self.output_tokens - 1)
        return None


def annotate_request_observability(
    result: RequestResult,
    *,
    request_index: int | None,
    request,
    scheduled_at_s: float | None,
    dispatch_started_at_s: float,
    semaphore_acquired_at_s: float,
    completed_at_s: float,
) -> RequestResult:
    """Attach client-side request shape and scheduling metadata."""
    result.request_index = request_index
    result.max_tokens_requested = int(getattr(request, "max_tokens", 0) or 0)
    messages = getattr(request, "messages", None) or ()
    result.message_count = len(messages)
    result.prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
    result.scheduled_at_s = scheduled_at_s
    result.dispatch_started_at_s = dispatch_started_at_s
    result.semaphore_acquired_at_s = semaphore_acquired_at_s
    result.completed_at_s = completed_at_s
    result.client_schedule_delay_s = (
        dispatch_started_at_s - scheduled_at_s
        if scheduled_at_s is not None
        else None
    )
    result.client_queue_wait_s = max(0.0, semaphore_acquired_at_s - dispatch_started_at_s)
    result.client_request_wall_s = max(0.0, completed_at_s - semaphore_acquired_at_s)

    metadata = getattr(request, "metadata", None)
    if metadata:
        result.request_metadata = dict(metadata)

    return result


def annotate_multi_turn_cache_estimate(
    result: RequestResult,
    session_id: int,
    turn_index: int,
    previous_context_tokens: int,
    cache_block_size: int | None = None,
    previous_output_tokens: int = 0,
    reply_in_cache: bool = False,
) -> RequestResult:
    """Attach the cache estimate to a multi-turn result; with reply_in_cache the previous turn's output counts as reusable prefix, otherwise the prefix stops at the previous prompt."""
    result.session_id = session_id
    result.turn_index = turn_index
    result.previous_context_tokens = max(0, int(previous_context_tokens or 0))

    if not result.success or result.input_tokens <= 0:
        result.cache_estimate_source = "unavailable"
        return result

    total_context = int(result.input_tokens)
    reusable = result.previous_context_tokens
    if reply_in_cache:
        reusable += max(0, int(previous_output_tokens or 0))
    cached_context = min(reusable, total_context)
    new_prefill = max(0, total_context - cached_context)

    result.total_context_tokens = total_context
    result.cached_context_tokens = cached_context
    result.new_prefill_tokens = new_prefill
    result.cache_hit_rate = cached_context / total_context if total_context > 0 else 0.0
    result.cache_estimate_source = ("previous_prompt_and_reply_tokens"
                                   if reply_in_cache else "previous_prompt_tokens")

    if cache_block_size is not None and cache_block_size > 0:
        block_size = int(cache_block_size)
        aligned_cached = (cached_context // block_size) * block_size
        aligned_new_prefill = max(0, total_context - aligned_cached)
        result.cache_block_size = block_size
        result.block_aligned_cached_context_tokens = aligned_cached
        result.block_aligned_new_prefill_tokens = aligned_new_prefill
        result.block_aligned_cache_hit_rate = (
            aligned_cached / total_context if total_context > 0 else 0.0
        )
        result.uncached_prefix_tail_tokens = cached_context - aligned_cached
        result.total_context_blocks = (total_context + block_size - 1) // block_size
        result.cached_context_blocks = aligned_cached // block_size
        result.new_prefill_blocks = (
            (aligned_new_prefill + block_size - 1) // block_size
            if aligned_new_prefill > 0
            else 0
        )
    return result


@dataclass
class BenchmarkSummary:
    """Aggregated metrics for a benchmark run."""

    model: str = ""
    profile: str = ""
    concurrency: int = 0
    # Closed loop caps in flight at `concurrency`; open loop only sizes warmup with it.
    load_mode: str = ""
    # Open-loop offered rate: sessions/s for multi-turn, requests/s for single-turn.
    target_rate: float = 0.0
    sessions_completed: int = 0
    session_throughput: float = 0.0
    warmup_concurrency: int = 0     # batch width the warmup actually exercised
    num_requests: int = 0
    duration_s: float = 0.0

    # duration_s, or the span of the kept requests under --discard-first.
    measured_window_s: float = 0.0
    excluded_requests: int = 0

    successful_requests: int = 0
    failed_requests: int = 0
    # Above 0 means the latency tail is censored: the slowest requests were cut off and are in no percentile.
    timeout_requests: int = 0
    error_kinds: dict = field(default_factory=dict)
    failed_with_partial_ttft: int = 0
    # 0 means every token count and throughput is zero-filled, not measured.
    usage_reported_requests: int = 0

    # Over total wall clock; an open-loop run below saturation measures the offered rate here.
    request_throughput: float = 0.0     # req/s
    input_token_throughput: float = 0.0  # input tok/s
    output_token_throughput: float = 0.0  # output tok/s
    total_token_throughput: float = 0.0   # (input + output) tok/s

    # Over busy time (union of in-flight intervals), excluding idle wall clock.
    busy_time_s: float = 0.0
    busy_request_throughput: float = 0.0
    busy_input_token_throughput: float = 0.0
    busy_output_token_throughput: float = 0.0
    busy_total_token_throughput: float = 0.0

    # Time-weighted in-flight count over the busy window; far below concurrency means the run mostly drained.
    mean_inflight_requests: float = 0.0
    max_inflight_requests: int = 0
    # Mean in-flight over each half of the run; rising means the queue is growing.
    mean_inflight_first_half: float = 0.0
    mean_inflight_second_half: float = 0.0

    total_input_tokens: int = 0
    total_output_tokens: int = 0

    mean_ttft_ms: float = 0.0
    median_ttft_ms: float = 0.0
    p90_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0

    mean_tpot_ms: float = 0.0
    median_tpot_ms: float = 0.0
    p90_tpot_ms: float = 0.0
    p99_tpot_ms: float = 0.0

    # Every inter-token latency pooled across requests.
    mean_itl_ms: float = 0.0
    median_itl_ms: float = 0.0
    p90_itl_ms: float = 0.0
    p99_itl_ms: float = 0.0

    mean_e2el_ms: float = 0.0
    median_e2el_ms: float = 0.0
    p90_e2el_ms: float = 0.0
    p99_e2el_ms: float = 0.0

    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "errors"}
        d["errors"] = self.errors[:10]  # cap error list in JSON
        return d


def _busy_time_and_load(results) -> tuple[float, float, int]:
    """(busy_time_s, mean_inflight_requests, max_inflight_requests); busy time is the union of the in-flight intervals and mean in-flight is sum(durations) / busy_time. Failed requests count because they occupied the server."""
    intervals = [
        (r.semaphore_acquired_at_s, r.completed_at_s)
        for r in results
        if r.semaphore_acquired_at_s is not None
        and r.completed_at_s is not None
        and r.completed_at_s > r.semaphore_acquired_at_s
    ]
    if not intervals:
        return 0.0, 0.0, 0

    intervals.sort()
    busy = 0.0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start > cur_end:  # idle gap
            busy += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    busy += cur_end - cur_start

    total_request_time = sum(end - start for start, end in intervals)
    mean_inflight = total_request_time / busy if busy > 0 else 0.0

    events = [(s, 1) for s, _ in intervals] + [(e, -1) for _, e in intervals]
    events.sort(key=lambda ev: (ev[0], ev[1]))
    inflight = 0
    max_inflight = 0
    for _, delta in events:
        inflight += delta
        max_inflight = max(max_inflight, inflight)

    return busy, mean_inflight, max_inflight


def _inflight_halves(results) -> tuple[float, float]:
    """Time-weighted mean in-flight over the first and second half of the run; rising means the queue is growing, falling means the run is drain-dominated."""
    intervals = [
        (r.semaphore_acquired_at_s, r.completed_at_s)
        for r in results
        if r.semaphore_acquired_at_s is not None
        and r.completed_at_s is not None
        and r.completed_at_s > r.semaphore_acquired_at_s
    ]
    if len(intervals) < 2:
        return 0.0, 0.0

    t0 = min(s for s, _ in intervals)
    t1 = max(e for _, e in intervals)
    if t1 <= t0:
        return 0.0, 0.0
    tm = (t0 + t1) / 2.0

    def mean_over(a: float, b: float) -> float:
        span = b - a
        if span <= 0:
            return 0.0
        area = sum(max(0.0, min(e, b) - max(s, a)) for s, e in intervals)
        return area / span

    return mean_over(t0, tm), mean_over(tm, t1)


PERCENTILE_METHODS = ("nearest", "linear")


def percentile(values, p: float, method: str) -> float:
    """p-th percentile (0-100) of values.

    "nearest" is sorted[min(n - 1, int(p / 100 * n))] and raises ValueError on no values; "linear" interpolates between ranks at p / 100 * (n - 1) and returns 0.0 on no values. The two disagree on most inputs, so every caller names its method."""
    if method not in PERCENTILE_METHODS:
        raise ValueError(f"unknown percentile method {method!r}; expected one of {PERCENTILE_METHODS}")
    sorted_data = sorted(values)
    n = len(sorted_data)
    if method == "nearest":
        if not n:
            raise ValueError("nearest-rank percentile of no values")
        return sorted_data[min(n - 1, int(p / 100 * n))]
    if not n:
        return 0.0
    idx = (p / 100) * (n - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        return sorted_data[-1]
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


def pcts(values, method: str) -> dict:
    """{"p50", "p90", "p95", "p99"} percentiles of values by method."""
    return {f"p{q}": percentile(values, q, method) for q in (50, 90, 95, 99)}


def summarize_serve(records, wall_s: float, method: str) -> dict:
    """Serving summary of replay records ({ttft_ms, tpot_ms, out_tokens}).

    n, ttft/tpot percentiles by method, req/s and output tok/s over wall_s.
    Records with tpot_ms None (single-token outputs) are left out of tpot; tpot_ms is None when no record has one. Raises on no records.
    """
    if not records:
        raise ValueError("no completed requests")
    ttft = [r["ttft_ms"] for r in records]
    tpot = [r["tpot_ms"] for r in records if r["tpot_ms"] is not None]
    toks = sum(r["out_tokens"] for r in records)
    return {
        "n": len(records),
        "percentile_method": method,
        "ttft_ms": pcts(ttft, method),
        "tpot_ms": pcts(tpot, method) if tpot else None,
        "req_s": len(records) / wall_s if wall_s > 0 else 0.0,
        "tok_s": toks / wall_s if wall_s > 0 else 0.0,
    }


def sla_verdict(summary: dict, sla_ttft_ms, sla_tpot_ms) -> list[str]:
    """Names of the p99 bounds summary breaks ("ttft_p99", "tpot_p99"); empty means PASS.

    A bound of None or 0 is not checked; tpot is skipped when the summary has none.
    """
    bound = []
    if sla_ttft_ms and summary["ttft_ms"]["p99"] > sla_ttft_ms:
        bound.append("ttft_p99")
    tp = summary["tpot_ms"]
    if sla_tpot_ms and tp and tp["p99"] > sla_tpot_ms:
        bound.append("tpot_p99")
    return bound


def acceptance(reference_ids, generated_ids) -> tuple[int, float]:
    """Matched leading-token length and rate between the trace's real assistant tokens and what the engine generated."""
    n = 0
    for a, b in zip(reference_ids, generated_ids):
        if a != b:
            break
        n += 1
    return n, (n / len(reference_ids) if reference_ids else 0.0)


def burst_recovery(records, sla_ttft_ms, window_s: float = 5.0):
    """(recovery_s, peak_p99) over arrival windows of window_s seconds.

    Records ({arrival_s, ttft_ms}) are bucketed by arrival from the first one, each bucket's p99 TTFT nearest-rank. recovery_s is the time from the worst bucket to the first later one at or under the SLA, None if none is; (None, None) with no records or no SLA."""
    if not records or sla_ttft_ms is None:
        return None, None
    rs = sorted(records, key=lambda r: r["arrival_s"])
    t0 = rs[0]["arrival_s"]
    buckets = {}
    for r in rs:
        buckets.setdefault(int((r["arrival_s"] - t0) // window_s), []).append(r["ttft_ms"])
    p99 = {w: percentile(v, 99, "nearest") for w, v in buckets.items()}
    peak_w = max(p99, key=p99.get)
    for w in sorted(p99):
        if w > peak_w and p99[w] <= sla_ttft_ms:
            return (w - peak_w) * window_s, p99[peak_w]
    return None, p99[peak_w]


def aggregate(
    results,
    duration_s: float,
    model: str = "",
    profile: str = "",
    concurrency: int = 0,
    load_mode: str = "",
    target_rate: float = 0.0,
    warmup_concurrency: int = 0,
) -> BenchmarkSummary:
    """Aggregate RequestResults into a BenchmarkSummary; requests held out by --discard-first are left out of every statistic, busy time included."""
    excluded = sum(1 for r in results if r.excluded_from_summary)
    results = [r for r in results if not r.excluded_from_summary]

    summary = BenchmarkSummary(
        model=model,
        profile=profile,
        concurrency=concurrency,
        num_requests=len(results),
        duration_s=duration_s,
        excluded_requests=excluded,
        load_mode=load_mode,
        target_rate=target_rate,
        warmup_concurrency=warmup_concurrency,
    )

    ttfts = []
    tpots = []
    itls = []
    e2els = []

    for r in results:
        if r.success:
            summary.successful_requests += 1
            summary.total_input_tokens += r.input_tokens
            summary.total_output_tokens += r.output_tokens
            if r.usage_reported:
                summary.usage_reported_requests += 1

            if r.ttft is not None:
                ttfts.append(r.ttft * 1000)  # ms
            if r.tpot is not None:
                tpots.append(r.tpot * 1000)
            if r.itl:
                itls.extend(t * 1000 for t in r.itl)
            if r.e2el is not None:
                e2els.append(r.e2el * 1000)
        else:
            # Failed requests stay out of every percentile; they are counted by kind so a censored tail shows.
            summary.failed_requests += 1
            kind = r.error_kind or "unknown"
            summary.error_kinds[kind] = summary.error_kinds.get(kind, 0) + 1
            if kind == "timeout":
                summary.timeout_requests += 1
            if r.ttft is not None:
                summary.failed_with_partial_ttft += 1
            if r.error:
                summary.errors.append(r.error)

    # With held-out requests the window is the span of the kept ones.
    window_s = duration_s
    if excluded:
        spans = [
            (r.semaphore_acquired_at_s, r.completed_at_s)
            for r in results
            if r.semaphore_acquired_at_s is not None and r.completed_at_s is not None
        ]
        if spans:
            window_s = max(e for _, e in spans) - min(s for s, _ in spans)
    summary.measured_window_s = window_s

    # Multi-turn open loop offers sessions, so the served rate to compare is session completions, not turns.
    session_ids = {r.session_id for r in results if r.session_id is not None}
    summary.sessions_completed = len(session_ids)
    if session_ids and window_s > 0:
        summary.session_throughput = len(session_ids) / window_s

    if window_s > 0:
        summary.request_throughput = summary.successful_requests / window_s
        summary.input_token_throughput = summary.total_input_tokens / window_s
        summary.output_token_throughput = summary.total_output_tokens / window_s
        summary.total_token_throughput = (
            summary.total_input_tokens + summary.total_output_tokens
        ) / window_s

    busy_s, mean_inflight, max_inflight = _busy_time_and_load(results)
    summary.busy_time_s = busy_s
    summary.mean_inflight_requests = mean_inflight
    summary.max_inflight_requests = max_inflight
    first_half, second_half = _inflight_halves(results)
    summary.mean_inflight_first_half = first_half
    summary.mean_inflight_second_half = second_half
    if busy_s > 0:
        summary.busy_request_throughput = summary.successful_requests / busy_s
        summary.busy_input_token_throughput = summary.total_input_tokens / busy_s
        summary.busy_output_token_throughput = summary.total_output_tokens / busy_s
        summary.busy_total_token_throughput = (
            summary.total_input_tokens + summary.total_output_tokens
        ) / busy_s

    if ttfts:
        summary.mean_ttft_ms = statistics.mean(ttfts)
        summary.median_ttft_ms = statistics.median(ttfts)
        summary.p90_ttft_ms = percentile(ttfts, 90, "linear")
        summary.p99_ttft_ms = percentile(ttfts, 99, "linear")

    if tpots:
        summary.mean_tpot_ms = statistics.mean(tpots)
        summary.median_tpot_ms = statistics.median(tpots)
        summary.p90_tpot_ms = percentile(tpots, 90, "linear")
        summary.p99_tpot_ms = percentile(tpots, 99, "linear")

    if itls:
        summary.mean_itl_ms = statistics.mean(itls)
        summary.median_itl_ms = statistics.median(itls)
        summary.p90_itl_ms = percentile(itls, 90, "linear")
        summary.p99_itl_ms = percentile(itls, 99, "linear")

    if e2els:
        summary.mean_e2el_ms = statistics.mean(e2els)
        summary.median_e2el_ms = statistics.median(e2els)
        summary.p90_e2el_ms = percentile(e2els, 90, "linear")
        summary.p99_e2el_ms = percentile(e2els, 99, "linear")

    return summary


@dataclass
class TurnSummary:
    """Per-turn metrics for multi-turn benchmarks."""
    turn_index: int
    num_requests: int = 0
    successful: int = 0
    mean_ttft_ms: float = 0.0
    median_ttft_ms: float = 0.0
    p90_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    mean_tpot_ms: float = 0.0
    median_tpot_ms: float = 0.0
    mean_e2el_ms: float = 0.0
    median_e2el_ms: float = 0.0
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0
    median_input_tokens: float = 0.0
    median_output_tokens: float = 0.0
    avg_new_prefill_tokens: float = 0.0
    median_new_prefill_tokens: float = 0.0
    avg_cached_context_tokens: float = 0.0
    median_cached_context_tokens: float = 0.0
    avg_cache_hit_rate: float = 0.0
    median_cache_hit_rate: float = 0.0
    avg_block_aligned_new_prefill_tokens: float = 0.0
    median_block_aligned_new_prefill_tokens: float = 0.0
    avg_block_aligned_cached_context_tokens: float = 0.0
    median_block_aligned_cached_context_tokens: float = 0.0
    avg_block_aligned_cache_hit_rate: float = 0.0
    median_block_aligned_cache_hit_rate: float = 0.0
    avg_uncached_prefix_tail_tokens: float = 0.0
    median_uncached_prefix_tail_tokens: float = 0.0
    median_client_queue_wait_ms: float = 0.0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def aggregate_per_turn(results_by_turn: dict[int, list]) -> list[TurnSummary]:
    """Aggregate metrics per turn for multi-turn benchmarks."""
    summaries = []
    for turn_idx in sorted(results_by_turn.keys()):
        results = results_by_turn[turn_idx]
        if not results:
            continue

        ts = TurnSummary(turn_index=turn_idx)
        ts.num_requests = len(results)

        ttfts = []
        tpots = []
        e2els = []
        input_toks = []
        output_toks = []
        new_prefill_toks = []
        cached_context_toks = []
        cache_hit_rates = []
        block_new_prefill_toks = []
        block_cached_context_toks = []
        block_cache_hit_rates = []
        uncached_prefix_tail_toks = []
        client_queue_waits = []

        for r in results:
            if r.success:
                ts.successful += 1
                if r.ttft is not None:
                    ttfts.append(r.ttft * 1000)
                if r.tpot is not None:
                    tpots.append(r.tpot * 1000)
                if r.e2el is not None:
                    e2els.append(r.e2el * 1000)
                input_toks.append(r.input_tokens)
                output_toks.append(r.output_tokens)
                if r.new_prefill_tokens is not None:
                    new_prefill_toks.append(r.new_prefill_tokens)
                if r.cached_context_tokens is not None:
                    cached_context_toks.append(r.cached_context_tokens)
                if r.cache_hit_rate is not None:
                    cache_hit_rates.append(r.cache_hit_rate)
                if r.block_aligned_new_prefill_tokens is not None:
                    block_new_prefill_toks.append(r.block_aligned_new_prefill_tokens)
                if r.block_aligned_cached_context_tokens is not None:
                    block_cached_context_toks.append(r.block_aligned_cached_context_tokens)
                if r.block_aligned_cache_hit_rate is not None:
                    block_cache_hit_rates.append(r.block_aligned_cache_hit_rate)
                if r.uncached_prefix_tail_tokens is not None:
                    uncached_prefix_tail_toks.append(r.uncached_prefix_tail_tokens)
                if r.client_queue_wait_s is not None:
                    client_queue_waits.append(r.client_queue_wait_s * 1000)

        if ttfts:
            ts.mean_ttft_ms = statistics.mean(ttfts)
            ts.median_ttft_ms = statistics.median(ttfts)
            ts.p90_ttft_ms = percentile(ttfts, 90, "linear")
            ts.p99_ttft_ms = percentile(ttfts, 99, "linear")
        if tpots:
            ts.mean_tpot_ms = statistics.mean(tpots)
            ts.median_tpot_ms = statistics.median(tpots)
        if e2els:
            ts.mean_e2el_ms = statistics.mean(e2els)
            ts.median_e2el_ms = statistics.median(e2els)
        if input_toks:
            ts.avg_input_tokens = statistics.mean(input_toks)
            ts.median_input_tokens = statistics.median(input_toks)
        if output_toks:
            ts.avg_output_tokens = statistics.mean(output_toks)
            ts.median_output_tokens = statistics.median(output_toks)
        if new_prefill_toks:
            ts.avg_new_prefill_tokens = statistics.mean(new_prefill_toks)
            ts.median_new_prefill_tokens = statistics.median(new_prefill_toks)
        if cached_context_toks:
            ts.avg_cached_context_tokens = statistics.mean(cached_context_toks)
            ts.median_cached_context_tokens = statistics.median(cached_context_toks)
        if cache_hit_rates:
            ts.avg_cache_hit_rate = statistics.mean(cache_hit_rates)
            ts.median_cache_hit_rate = statistics.median(cache_hit_rates)
        if block_new_prefill_toks:
            ts.avg_block_aligned_new_prefill_tokens = statistics.mean(block_new_prefill_toks)
            ts.median_block_aligned_new_prefill_tokens = statistics.median(block_new_prefill_toks)
        if block_cached_context_toks:
            ts.avg_block_aligned_cached_context_tokens = statistics.mean(block_cached_context_toks)
            ts.median_block_aligned_cached_context_tokens = statistics.median(block_cached_context_toks)
        if block_cache_hit_rates:
            ts.avg_block_aligned_cache_hit_rate = statistics.mean(block_cache_hit_rates)
            ts.median_block_aligned_cache_hit_rate = statistics.median(block_cache_hit_rates)
        if uncached_prefix_tail_toks:
            ts.avg_uncached_prefix_tail_tokens = statistics.mean(uncached_prefix_tail_toks)
            ts.median_uncached_prefix_tail_tokens = statistics.median(uncached_prefix_tail_toks)
        if client_queue_waits:
            ts.median_client_queue_wait_ms = statistics.median(client_queue_waits)

        summaries.append(ts)
    return summaries


# Run gates return (verdict, message): verdict is "pass", "warn" or "abort", message is None on pass.

def success_rate_gate(summary, min_rate: float) -> tuple[str, str | None]:
    """Abort when no request completed or the success rate is below min_rate."""
    if summary.num_requests == 0:
        return "abort", f"ABORT: No requests completed. Minimum success rate: {min_rate:.0%}"
    rate = summary.successful_requests / summary.num_requests
    if rate < min_rate:
        kinds = ", ".join(f"{k}={v}" for k, v in sorted(summary.error_kinds.items()))
        return "abort", (f"ABORT: Success rate {rate:.1%} below minimum {min_rate:.0%} "
                         f"({summary.successful_requests}/{summary.num_requests}); failures: {kinds}")
    return "pass", None


def usage_gate(summary, allow_missing: bool) -> tuple[str, str | None]:
    """Abort (warn with allow_missing) when no successful response carried usage, since every token count and throughput would read 0."""
    if summary.successful_requests == 0 or summary.usage_reported_requests > 0:
        return "pass", None
    if allow_missing:
        return "warn", ("WARNING: no usage block in any response; token counts and throughputs "
                        "are zero-filled. Continuing because --allow-missing-usage was passed.")
    return "abort", ("ABORT: no response carried a usage block, so every token count and "
                     "throughput in this run would be 0 rather than measured. The server must "
                     "honour stream_options.include_usage. Pass --allow-missing-usage to record "
                     "the run anyway (latency metrics stay valid; token metrics do not).")


def transient_gate(num_requests: int, concurrency: int) -> tuple[str, str | None]:
    """Warn when a closed-loop run has under 4 requests per concurrency slot: it is all startup wave and drain, and --discard-first cannot fix it."""
    if concurrency <= 1 or num_requests <= 0:
        return "pass", None
    per_slot = num_requests / concurrency
    if per_slot >= 4:
        return "pass", None
    return "warn", "\n".join([
        f"WARNING: only {per_slot:.1f} requests per concurrency slot "
        f"({num_requests} requests at concurrency {concurrency}).",
        "         The run is dominated by the startup wave and the drain tail, with no",
        "         steady-state middle. --discard-first cannot recover it; raise --num-requests",
        f"         to at least {concurrency * 8} for a settled measurement.",
    ])


def first_error(results) -> str:
    """First recorded error text, for abort messages that name a cause."""
    for r in results:
        if r is not None and not r.success and r.error:
            return str(r.error)
    return "no error text recorded"


def print_multi_turn_summary(turn_summaries: list, overall: BenchmarkSummary) -> None:
    """Print per-turn metrics table for multi-turn benchmarks."""
    print_summary(overall)
    print(f"{'=' * 98}")
    print(" Per-Turn Breakdown (prefix cache effect visible in TTFT trend)")
    print(f"{'=' * 98}")
    print(f" {'Turn':>4}  {'Reqs':>5}  {'Avg ISL':>8}  {'New p50':>8}  {'Cache p50':>9}  "
          f"{'Hit p50':>7}  {'TTFT p50':>9}  {'TTFT p90':>9}  {'TPOT p50':>9}  {'E2EL p50':>9}")
    print(f" {'─' * 4}  {'─' * 5}  {'─' * 8}  {'─' * 8}  {'─' * 9}  {'─' * 7}  "
          f"{'─' * 9}  {'─' * 9}  {'─' * 9}  {'─' * 9}")
    for ts in turn_summaries:
        print(f" {ts.turn_index + 1:>4}  {ts.successful:>5}  {ts.avg_input_tokens:>8.0f}  "
              f"{ts.median_new_prefill_tokens:>8.0f}  {ts.median_cached_context_tokens:>9.0f}  "
              f"{ts.median_cache_hit_rate * 100:>6.0f}%  "
              f"{ts.median_ttft_ms:>8.1f}ms  {ts.p90_ttft_ms:>8.1f}ms  "
              f"{ts.median_tpot_ms:>8.1f}ms  {ts.median_e2el_ms:>8.1f}ms")
    print(f"{'=' * 98}\n")


def print_summary(s: BenchmarkSummary) -> None:
    """Print a formatted benchmark summary to stdout."""
    print(f"\n{'=' * 52}")
    print(f" Benchmark Results: {s.profile} | concurrency={s.concurrency}")
    print(f"{'=' * 52}")
    print(f" Model:                    {s.model}")
    print(f" Duration:                 {s.duration_s:.2f}s")
    print(f" Requests:                 {s.successful_requests} ok / {s.failed_requests} failed")
    if s.excluded_requests:
        print(f" Held out (--discard-first): {s.excluded_requests} request(s); "
              f"window {s.measured_window_s:.2f}s")
    busy_pct = (f" ({100.0 * s.busy_time_s / s.measured_window_s:.1f}% of window)"
                if s.measured_window_s > 0 else "")
    print(f" Busy time:                {s.busy_time_s:.2f}s{busy_pct}")
    print(f" Mean in-flight requests:  {s.mean_inflight_requests:.1f} "
          f"(peak {s.max_inflight_requests}; "
          f"1st half {s.mean_inflight_first_half:.1f} -> "
          f"2nd half {s.mean_inflight_second_half:.1f})")
    print(f" Request throughput:       {s.request_throughput:.2f} req/s "
          f"(busy: {s.busy_request_throughput:.2f})")
    print(f" Input token throughput:   {s.input_token_throughput:.0f} tok/s "
          f"(busy: {s.busy_input_token_throughput:.0f})")
    print(f" Output token throughput:  {s.output_token_throughput:.0f} tok/s "
          f"(busy: {s.busy_output_token_throughput:.0f})")
    print(f" Total token throughput:   {s.total_token_throughput:.0f} tok/s "
          f"(busy: {s.busy_total_token_throughput:.0f})")
    # Mean in-flight against --concurrency only means something under closed loop; open loop asks whether the server kept up with the offered rate.
    if s.load_mode == "open-loop" and s.target_rate > 0:
        if s.sessions_completed:
            served, unit = s.session_throughput, "sess/s"
        else:
            served, unit = s.request_throughput, "req/s"
        # Two signals: in-flight growth catches overload only while arrivals continue, and the served/offered ratio reads slightly low from the final drain, so 0.75 leaves room for a healthy run.
        growing = (s.mean_inflight_first_half > 0.5
                   and s.mean_inflight_second_half > 1.5 * s.mean_inflight_first_half)
        rate_short = served < 0.75 * s.target_rate
        print(f" NOTE: offered {s.target_rate:.2f} {unit}, served {served:.2f} {unit} "
              f"({served / s.target_rate * 100:.0f}% of offered; mean in-flight "
              f"{s.mean_inflight_requests:.1f}).")
        if rate_short or growing:
            why = []
            if rate_short:
                why.append("served rate fell short of offered")
            if growing:
                why.append("in-flight grew through the run")
            print(f"       The server did NOT keep up ({'; '.join(why)}).")
            print("       Arrivals queued and the latency figures include that backlog.")
            print("       This is a saturation point, not a steady-state latency figure.")
        else:
            print("       In-flight stayed bounded, so the run is arrival-limited: this")
            print("       measures latency at that rate, NOT capacity. Raise")
            print("       --target-rate to find saturation.")
    elif s.concurrency and s.mean_inflight_requests < 0.7 * s.concurrency:
        print(f" NOTE: mean in-flight ({s.mean_inflight_requests:.1f}) is well below "
              f"concurrency ({s.concurrency}).")
        print("       The server was idle or draining for much of the run, so the")
        print("       throughput above reflects the schedule, not its capacity.")

    # Growing in-flight means arrivals outran departures, so the averages depend on when measurement stopped.
    _f, _s2 = s.mean_inflight_first_half, s.mean_inflight_second_half
    if _f > 0.5 and _s2 > 1.5 * _f:
        print(f" WARNING: in-flight grew {_f:.1f} -> {_s2:.1f} across the run "
              f"({_s2 / _f:.1f}x).")
        print("          The queue never stabilised, so these averages depend on run")
        print("          length and are not a steady-state measurement. Lower the")
        print("          offered load, or run long enough to reach equilibrium.")

    # Steady decode has TPOT mean near median; a mean well above it means intermittent stalls.
    if s.median_tpot_ms > 0 and s.mean_tpot_ms > 2.0 * s.median_tpot_ms:
        print(f" WARNING: TPOT mean ({s.mean_tpot_ms:.1f} ms) is "
              f"{s.mean_tpot_ms / s.median_tpot_ms:.1f}x its median "
              f"({s.median_tpot_ms:.1f} ms).")
        print("          Decode was intermittently stalled rather than uniformly slow.")
        print("          Usual causes: kernel JIT at an unwarmed batch shape, scheduler")
        print("          preemption, or host stalls. The median is the reliable figure.")

    # Kernel JIT compiles per batch shape, so a width beyond what warmup exercised compiles during measurement.
    if s.warmup_concurrency and s.max_inflight_requests > s.warmup_concurrency:
        print(f" WARNING: peak in-flight ({s.max_inflight_requests}) exceeded the warmup "
              f"width ({s.warmup_concurrency}).")
        print("          Batch shapes past the warmup width may have compiled during")
        print(f"          measurement. Re-run with --warmup-concurrency "
              f">= {s.max_inflight_requests}.")
    print(f"{'─' * 52}")
    print(f" TTFT  mean/p50/p90/p99:   {s.mean_ttft_ms:.1f} / {s.median_ttft_ms:.1f} / {s.p90_ttft_ms:.1f} / {s.p99_ttft_ms:.1f} ms")
    print(f" TPOT  mean/p50/p90/p99:   {s.mean_tpot_ms:.1f} / {s.median_tpot_ms:.1f} / {s.p90_tpot_ms:.1f} / {s.p99_tpot_ms:.1f} ms")
    print(f" ITL   mean/p50/p90/p99:   {s.mean_itl_ms:.1f} / {s.median_itl_ms:.1f} / {s.p90_itl_ms:.1f} / {s.p99_itl_ms:.1f} ms")
    print(f" E2EL  mean/p50/p90/p99:   {s.mean_e2el_ms:.1f} / {s.median_e2el_ms:.1f} / {s.p90_e2el_ms:.1f} / {s.p99_e2el_ms:.1f} ms")
    print(f"{'=' * 52}\n")
    if s.timeout_requests:
        pct = 100.0 * s.timeout_requests / s.num_requests if s.num_requests else 0.0
        print(f" WARNING: {s.timeout_requests} request(s) ({pct:.1f}%) hit the client "
              f"timeout and were cut off.")
        print("          The percentiles above EXCLUDE them, so the latency tail is")
        print("          censored and biased low. Raise --timeout or lower the load")
        print("          before treating this cell as ground truth.\n")
    if s.successful_requests and not s.usage_reported_requests:
        print(" WARNING: no successful request carried a usage block. Token counts")
        print("          and all throughput figures above are zero-filled, not measured.\n")
    if s.error_kinds:
        kinds = ", ".join(f"{k}={v}" for k, v in sorted(s.error_kinds.items()))
        print(f" Failure kinds: {kinds}")
        if s.failed_with_partial_ttft:
            print(f" ({s.failed_with_partial_ttft} failed request(s) recorded a partial "
                  f"TTFT; see per_request in the result JSON)")
    if s.errors:
        print(f" Errors ({len(s.errors)} total, first {min(3,len(s.errors))}):")
        for e in s.errors[:3]:
            print(f"   {e}")
        print()


# vLLM's speculative-decoding counters; they exist only on a server with spec decoding.
SPEC_COUNTERS = {"accepted_tokens": "vllm:spec_decode_num_accepted_tokens",
                 "draft_tokens": "vllm:spec_decode_num_draft_tokens",
                 "drafts": "vllm:spec_decode_num_drafts"}

# name{labels} value [timestamp]; label values may hold escaped quotes.
SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+\S+)?$')
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse_prometheus(text):
    """{name: value} for unlabeled samples, {(name, ((label, value), ...)): value} for labeled ones with labels sorted; comments, blank lines and lines that do not parse are skipped."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = SAMPLE.match(line)
        if not m:
            continue
        name, labels, value = m.groups()
        try:
            value = float(value)
        except ValueError:
            continue
        key = (name, tuple(sorted(LABEL.findall(labels)))) if labels else name
        out[key] = value
    return out


def counter_total(metrics, name):
    """Sum of a counter over every label set, counting both X and X_total; None when absent."""
    names = (name, name + "_total")
    vals = [v for k, v in metrics.items() if (k[0] if isinstance(k, tuple) else k) in names]
    return sum(vals) if vals else None


def spec_counts(metrics):
    """{accepted_tokens, draft_tokens, drafts} from parsed /metrics, each summed over engines; None when a counter is missing."""
    counts = {key: counter_total(metrics, name) for key, name in SPEC_COUNTERS.items()}
    return None if None in counts.values() else counts
