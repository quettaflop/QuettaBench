"""Aggregate request results into BenchmarkSummary; failures are counted apart."""

import re
import statistics
from dataclasses import dataclass, field

# Under this a gap is two tokens stamped by one event-loop wakeup, not a decode step.
ITL_COALESCED_MS = 0.05
# Above this share of gaps the client, not the server, shaped the ITL tail.
ITL_COALESCED_WARN_FRAC = 0.01


@dataclass
class RequestResult:
    success: bool
    ttft: float | None = None
    itl: list = field(default_factory=list)
    e2el: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None
    # A timeout means the client cut the stream: the slowest requests are missing from the tail.
    error_kind: str | None = None  # "timeout" | "http_error" | "client_error"
    # Without a usage block token counts are 0 and throughputs are meaningless.
    usage_reported: bool = False
    # Held out of the summary by --discard-first; still kept in per_request.
    excluded_from_summary: bool = False

    # Captured only on request, to splice the reply into later turns; not written to run.json.
    generated_text: str | None = None
    # "first_payload_chunk" (chat) or "first_nonempty_text" (completions).
    ttft_rule: str | None = None
    # "slot" (closed loop, concurrency slot grant) or "arrival" (open loop, scheduled arrival).
    ttft_basis: str | None = None
    # Leading tokens equal to the trace's reference reply; None without returned tokens.
    accepted_tokens: int | None = None
    # Not written to run.json.
    generated_token_ids: list | None = None
    # None outside grid runs.
    cell: dict | None = None
    # Every row of the batch carries the wall time of the one generate().
    batch_wall_s: float | None = None
    # mode.return_outputs=logits; entry i scores prompt token i + 1.
    logits: dict | None = None

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

    turn_index: int | None = None
    session_id: int | None = None

    # From per-session prompt-token deltas, not engine-reported prefix-cache hits.
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
        """Falls back to (e2el - ttft) / (output_tokens - 1) when ITL covers under 80% of tokens."""
        decode_time = None
        if self.e2el is not None and self.ttft is not None and self.output_tokens > 1:
            decode_time = self.e2el - self.ttft
        if self.itl:
            incomplete = self.output_tokens > 1 and len(self.itl) < 0.8 * (self.output_tokens - 1)
            if incomplete and decode_time is not None:
                return decode_time / (self.output_tokens - 1)
            return sum(self.itl) / len(self.itl)
        if decode_time is not None:
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
    result.request_index = request_index
    result.max_tokens_requested = int(getattr(request, "max_tokens", 0) or 0)
    messages = getattr(request, "messages", None) or ()
    result.message_count = len(messages)
    result.prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
    result.scheduled_at_s = scheduled_at_s
    result.dispatch_started_at_s = dispatch_started_at_s
    result.semaphore_acquired_at_s = semaphore_acquired_at_s
    result.completed_at_s = completed_at_s
    result.client_schedule_delay_s = dispatch_started_at_s - scheduled_at_s if scheduled_at_s is not None else None
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
    """With reply_in_cache the previous output counts as reusable prefix."""
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
    result.cache_estimate_source = "previous_prompt_and_reply_tokens" if reply_in_cache else "previous_prompt_tokens"

    if cache_block_size is not None and cache_block_size > 0:
        block_size = int(cache_block_size)
        aligned_cached = (cached_context // block_size) * block_size
        aligned_new_prefill = max(0, total_context - aligned_cached)
        result.cache_block_size = block_size
        result.block_aligned_cached_context_tokens = aligned_cached
        result.block_aligned_new_prefill_tokens = aligned_new_prefill
        result.block_aligned_cache_hit_rate = aligned_cached / total_context if total_context > 0 else 0.0
        result.uncached_prefix_tail_tokens = cached_context - aligned_cached
        result.total_context_blocks = (total_context + block_size - 1) // block_size
        result.cached_context_blocks = aligned_cached // block_size
        result.new_prefill_blocks = (
            (aligned_new_prefill + block_size - 1) // block_size if aligned_new_prefill > 0 else 0
        )
    return result


@dataclass
class BenchmarkSummary:
    model: str = ""
    profile: str = ""
    concurrency: int = 0
    # Closed loop caps in flight at `concurrency`; open loop only sizes warmup with it.
    load_mode: str = ""
    # Sessions/s for multi-turn, requests/s for single-turn.
    target_rate: float = 0.0
    sessions_completed: int = 0
    session_throughput: float = 0.0
    warmup_concurrency: int = 0
    num_requests: int = 0
    duration_s: float = 0.0

    # duration_s, or the span of the kept requests under --discard-first.
    measured_window_s: float = 0.0
    excluded_requests: int = 0

    successful_requests: int = 0
    failed_requests: int = 0
    # Above 0 the latency tail is censored and in no percentile.
    timeout_requests: int = 0
    error_kinds: dict = field(default_factory=dict)
    failed_with_partial_ttft: int = 0
    # 0 means token counts and throughputs are zero-filled, not measured.
    usage_reported_requests: int = 0

    # Over total wall clock; below saturation this is the offered rate.
    request_throughput: float = 0.0  # req/s
    input_token_throughput: float = 0.0  # tok/s
    output_token_throughput: float = 0.0  # tok/s
    total_token_throughput: float = 0.0  # tok/s

    # Over busy time (union of in-flight intervals), excluding idle wall clock.
    busy_time_s: float = 0.0
    busy_request_throughput: float = 0.0
    busy_input_token_throughput: float = 0.0
    busy_output_token_throughput: float = 0.0
    busy_total_token_throughput: float = 0.0

    # Far below concurrency means the run mostly drained.
    mean_inflight_requests: float = 0.0
    max_inflight_requests: int = 0
    # Rising means the queue is growing.
    mean_inflight_first_half: float = 0.0
    mean_inflight_second_half: float = 0.0

    total_input_tokens: int = 0
    total_output_tokens: int = 0

    mean_ttft_ms: float = 0.0
    median_ttft_ms: float = 0.0
    p90_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    # Closed loop only: drops the first `concurrency` requests, which start together and queue.
    steady_median_ttft_ms: float | None = None
    steady_p99_ttft_ms: float | None = None

    mean_tpot_ms: float = 0.0
    median_tpot_ms: float = 0.0
    p90_tpot_ms: float = 0.0
    p99_tpot_ms: float = 0.0

    mean_itl_ms: float = 0.0
    median_itl_ms: float = 0.0
    p90_itl_ms: float = 0.0
    p99_itl_ms: float = 0.0
    itl_coalesced_frac: float = 0.0

    mean_e2el_ms: float = 0.0
    median_e2el_ms: float = 0.0
    p90_e2el_ms: float = 0.0
    p99_e2el_ms: float = 0.0

    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "errors"}
        d["errors"] = self.errors[:10]
        return d


def _busy_time_and_load(results) -> tuple[float, float, int]:
    """Failed requests count: they occupied the server."""
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
        if start > cur_end:
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
    """Falling means drain dominates."""
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
    """ "nearest" raises on no values, "linear" returns 0.0; they disagree on most inputs."""
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
    return {f"p{q}": percentile(values, q, method) for q in (50, 90, 95, 99)}


def steady_ttfts(indexed_ttfts, concurrency: int) -> list | None:
    """TTFTs past the first `concurrency` dispatched; None when fewer than `concurrency` remain."""
    rest = [t for i, t in indexed_ttfts if i >= concurrency]
    return rest if len(rest) >= concurrency else None


def summarize_serve(records, wall_s: float, method: str, closed_concurrency: int = 0) -> dict:
    """closed_concurrency > 0 adds steady-state TTFT percentiles; records then carry their dispatch index."""
    if not records:
        raise ValueError("no completed requests")
    ttft = [r["ttft_ms"] for r in records]
    tpot = [r["tpot_ms"] for r in records if r["tpot_ms"] is not None]
    toks = sum(r["out_tokens"] for r in records)
    out = {
        "n": len(records),
        "percentile_method": method,
        "ttft_ms": pcts(ttft, method),
        "tpot_ms": pcts(tpot, method) if tpot else None,
        "req_s": len(records) / wall_s if wall_s > 0 else 0.0,
        "tok_s": toks / wall_s if wall_s > 0 else 0.0,
    }
    if closed_concurrency:
        steady = steady_ttfts(((r["index"], r["ttft_ms"]) for r in records), closed_concurrency)
        out["steady_ttft_ms"] = pcts(steady, method) if steady else None
    return out


def sla_verdict(summary: dict, sla_ttft_ms, sla_tpot_ms) -> list[str]:
    """A bound of None or 0 is not checked; empty means PASS."""
    bound = []
    if sla_ttft_ms and summary["ttft_ms"]["p99"] > sla_ttft_ms:
        bound.append("ttft_p99")
    tp = summary["tpot_ms"]
    if sla_tpot_ms and tp and tp["p99"] > sla_tpot_ms:
        bound.append("tpot_p99")
    return bound


def acceptance(reference_ids, generated_ids) -> tuple[int, float]:
    n = 0
    for a, b in zip(reference_ids, generated_ids):
        if a != b:
            break
        n += 1
    return n, (n / len(reference_ids) if reference_ids else 0.0)


def burst_recovery(records, sla_ttft_ms, window_s: float = 5.0):
    """recovery_s is the time from the worst bucket back under the SLA, else None."""
    if not records or sla_ttft_ms is None:
        return None, None
    rs = sorted(records, key=lambda r: r["arrival_s"])
    t0 = rs[0]["arrival_s"]
    buckets: dict[int, list[float]] = {}
    for r in rs:
        buckets.setdefault(int((r["arrival_s"] - t0) // window_s), []).append(r["ttft_ms"])
    p99 = {w: percentile(v, 99, "nearest") for w, v in buckets.items()}
    peak_w = max(p99, key=lambda w: p99[w])
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
    """Requests held out by --discard-first are in no statistic, busy time included."""
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
    steady_pairs = []
    tpots = []
    itls: list[float] = []
    e2els = []

    for pos, r in enumerate(results):
        if r.success:
            summary.successful_requests += 1
            summary.total_input_tokens += r.input_tokens
            summary.total_output_tokens += r.output_tokens
            if r.usage_reported:
                summary.usage_reported_requests += 1

            if r.ttft is not None:
                ttfts.append(r.ttft * 1000)
                steady_pairs.append((r.request_index if r.request_index is not None else pos, r.ttft * 1000))
            if r.tpot is not None:
                tpots.append(r.tpot * 1000)
            if r.itl:
                itls.extend(t * 1000 for t in r.itl)
            if r.e2el is not None:
                e2els.append(r.e2el * 1000)
        else:
            # Failed requests stay out of percentiles; counted by kind so a censored tail shows.
            summary.failed_requests += 1
            kind = r.error_kind or "unknown"
            summary.error_kinds[kind] = summary.error_kinds.get(kind, 0) + 1
            if kind == "timeout":
                summary.timeout_requests += 1
            if r.ttft is not None:
                summary.failed_with_partial_ttft += 1
            if r.error:
                summary.errors.append(r.error)

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

    # Multi-turn open loop offers sessions, so the served rate counts session completions.
    session_ids = {r.session_id for r in results if r.session_id is not None}
    summary.sessions_completed = len(session_ids)
    if session_ids and window_s > 0:
        summary.session_throughput = len(session_ids) / window_s

    if window_s > 0:
        summary.request_throughput = summary.successful_requests / window_s
        summary.input_token_throughput = summary.total_input_tokens / window_s
        summary.output_token_throughput = summary.total_output_tokens / window_s
        summary.total_token_throughput = (summary.total_input_tokens + summary.total_output_tokens) / window_s

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
        summary.busy_total_token_throughput = (summary.total_input_tokens + summary.total_output_tokens) / busy_s

    if ttfts:
        summary.mean_ttft_ms = statistics.mean(ttfts)
        summary.median_ttft_ms = statistics.median(ttfts)
        summary.p90_ttft_ms = percentile(ttfts, 90, "linear")
        summary.p99_ttft_ms = percentile(ttfts, 99, "linear")
        steady = steady_ttfts(steady_pairs, concurrency) if load_mode == "closed-loop" else None
        if steady:
            summary.steady_median_ttft_ms = statistics.median(steady)
            summary.steady_p99_ttft_ms = percentile(steady, 99, "linear")

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
        summary.itl_coalesced_frac = sum(t < ITL_COALESCED_MS for t in itls) / len(itls)

    if e2els:
        summary.mean_e2el_ms = statistics.mean(e2els)
        summary.median_e2el_ms = statistics.median(e2els)
        summary.p90_e2el_ms = percentile(e2els, 90, "linear")
        summary.p99_e2el_ms = percentile(e2els, 99, "linear")

    return summary


@dataclass
class TurnSummary:
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


# Gates return (verdict, message): "pass", "warn" or "abort"; message None on pass.


def success_rate_gate(summary, min_rate: float) -> tuple[str, str | None]:
    if summary.num_requests == 0:
        return "abort", f"ABORT: No requests completed. Minimum success rate: {min_rate:.0%}"
    rate = summary.successful_requests / summary.num_requests
    if rate < min_rate:
        kinds = ", ".join(f"{k}={v}" for k, v in sorted(summary.error_kinds.items()))
        return "abort", (
            f"ABORT: Success rate {rate:.1%} below minimum {min_rate:.0%} "
            f"({summary.successful_requests}/{summary.num_requests}); failures: {kinds}"
        )
    return "pass", None


def usage_gate(summary, allow_missing: bool) -> tuple[str, str | None]:
    """Every token count and throughput would read 0."""
    if summary.successful_requests == 0 or summary.usage_reported_requests > 0:
        return "pass", None
    if allow_missing:
        return "warn", (
            "WARNING: no usage block in any response; token counts and throughputs "
            "are zero-filled. Continuing because --allow-missing-usage was passed."
        )
    return "abort", (
        "ABORT: no response carried a usage block, so every token count and "
        "throughput in this run would be 0 rather than measured. The server must "
        "honour stream_options.include_usage. Pass --allow-missing-usage to record "
        "the run anyway (latency metrics stay valid; token metrics do not)."
    )


def transient_gate(num_requests: int, concurrency: int) -> tuple[str, str | None]:
    """It is all startup wave and drain."""
    if concurrency <= 1 or num_requests <= 0:
        return "pass", None
    per_slot = num_requests / concurrency
    if per_slot >= 4:
        return "pass", None
    return "warn", "\n".join(
        [
            f"WARNING: only {per_slot:.1f} requests per concurrency slot "
            f"({num_requests} requests at concurrency {concurrency}).",
            "         The run is dominated by the startup wave and the drain tail, with no",
            "         steady-state middle. --discard-first cannot recover it; raise --num-requests",
            f"         to at least {concurrency * 8} for a settled measurement.",
        ]
    )


def first_error(results) -> str:
    for r in results:
        if r is not None and not r.success and r.error:
            return str(r.error)
    return "no error text recorded"


def print_multi_turn_summary(turn_summaries: list, overall: BenchmarkSummary) -> None:
    print_summary(overall)
    print(f"{'=' * 98}")
    print(" Per-Turn Breakdown (prefix cache effect visible in TTFT trend)")
    print(f"{'=' * 98}")
    print(
        f" {'Turn':>4}  {'Reqs':>5}  {'Avg ISL':>8}  {'New p50':>8}  {'Cache p50':>9}  "
        f"{'Hit p50':>7}  {'TTFT p50':>9}  {'TTFT p90':>9}  {'TPOT p50':>9}  {'E2EL p50':>9}"
    )
    print(
        f" {'─' * 4}  {'─' * 5}  {'─' * 8}  {'─' * 8}  {'─' * 9}  {'─' * 7}  {'─' * 9}  {'─' * 9}  {'─' * 9}  {'─' * 9}"
    )
    for ts in turn_summaries:
        print(
            f" {ts.turn_index + 1:>4}  {ts.successful:>5}  {ts.avg_input_tokens:>8.0f}  "
            f"{ts.median_new_prefill_tokens:>8.0f}  {ts.median_cached_context_tokens:>9.0f}  "
            f"{ts.median_cache_hit_rate * 100:>6.0f}%  "
            f"{ts.median_ttft_ms:>8.1f}ms  {ts.p90_ttft_ms:>8.1f}ms  "
            f"{ts.median_tpot_ms:>8.1f}ms  {ts.median_e2el_ms:>8.1f}ms"
        )
    print(f"{'=' * 98}\n")


def print_summary(s: BenchmarkSummary) -> None:
    print(f"\n{'=' * 52}")
    print(f" Benchmark Results: {s.profile} | concurrency={s.concurrency}")
    print(f"{'=' * 52}")
    print(f" Model:                    {s.model}")
    print(f" Duration:                 {s.duration_s:.2f}s")
    print(f" Requests:                 {s.successful_requests} ok / {s.failed_requests} failed")
    if s.excluded_requests:
        print(f" Held out (--discard-first): {s.excluded_requests} request(s); window {s.measured_window_s:.2f}s")
    busy_pct = f" ({100.0 * s.busy_time_s / s.measured_window_s:.1f}% of window)" if s.measured_window_s > 0 else ""
    print(f" Busy time:                {s.busy_time_s:.2f}s{busy_pct}")
    print(
        f" Mean in-flight requests:  {s.mean_inflight_requests:.1f} "
        f"(peak {s.max_inflight_requests}; "
        f"1st half {s.mean_inflight_first_half:.1f} -> "
        f"2nd half {s.mean_inflight_second_half:.1f})"
    )
    print(f" Request throughput:       {s.request_throughput:.2f} req/s (busy: {s.busy_request_throughput:.2f})")
    print(
        f" Input token throughput:   {s.input_token_throughput:.0f} tok/s (busy: {s.busy_input_token_throughput:.0f})"
    )
    print(
        f" Output token throughput:  {s.output_token_throughput:.0f} tok/s (busy: {s.busy_output_token_throughput:.0f})"
    )
    print(
        f" Total token throughput:   {s.total_token_throughput:.0f} tok/s (busy: {s.busy_total_token_throughput:.0f})"
    )
    if s.load_mode == "open-loop" and s.target_rate > 0:
        if s.sessions_completed:
            served, unit = s.session_throughput, "sess/s"
        else:
            served, unit = s.request_throughput, "req/s"
        # Overload shows as in-flight growth or served under 0.75x offered.
        growing = s.mean_inflight_first_half > 0.5 and s.mean_inflight_second_half > 1.5 * s.mean_inflight_first_half
        rate_short = served < 0.75 * s.target_rate
        print(
            f" NOTE: offered {s.target_rate:.2f} {unit}, served {served:.2f} {unit} "
            f"({served / s.target_rate * 100:.0f}% of offered; mean in-flight "
            f"{s.mean_inflight_requests:.1f})."
        )
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
        print(f" NOTE: mean in-flight ({s.mean_inflight_requests:.1f}) is well below concurrency ({s.concurrency}).")
        print("       The server was idle or draining for much of the run, so the")
        print("       throughput above reflects the schedule, not its capacity.")

    # Growing in-flight means arrivals outran departures.
    _f, _s2 = s.mean_inflight_first_half, s.mean_inflight_second_half
    if _f > 0.5 and _s2 > 1.5 * _f:
        print(f" WARNING: in-flight grew {_f:.1f} -> {_s2:.1f} across the run ({_s2 / _f:.1f}x).")
        print("          The queue never stabilised, so these averages depend on run")
        print("          length and are not a steady-state measurement. Lower the")
        print("          offered load, or run long enough to reach equilibrium.")

    # Steady decode has TPOT mean near median; a higher mean means stalls.
    if s.median_tpot_ms > 0 and s.mean_tpot_ms > 2.0 * s.median_tpot_ms:
        print(
            f" WARNING: TPOT mean ({s.mean_tpot_ms:.1f} ms) is "
            f"{s.mean_tpot_ms / s.median_tpot_ms:.1f}x its median "
            f"({s.median_tpot_ms:.1f} ms)."
        )
        print("          Decode was intermittently stalled rather than uniformly slow.")
        print("          Usual causes: kernel JIT at an unwarmed batch shape, scheduler")
        print("          preemption, or host stalls. The median is the reliable figure.")

    # Kernel JIT compiles per batch shape: a width beyond warmup compiles during measurement.
    if s.warmup_concurrency and s.max_inflight_requests > s.warmup_concurrency:
        print(
            f" WARNING: peak in-flight ({s.max_inflight_requests}) exceeded the warmup width ({s.warmup_concurrency})."
        )
        print("          Batch shapes past the warmup width may have compiled during")
        print(f"          measurement. Re-run with --warmup-concurrency >= {s.max_inflight_requests}.")
    print(f"{'─' * 52}")
    print(
        f" TTFT  mean/p50/p90/p99:   {s.mean_ttft_ms:.1f} / {s.median_ttft_ms:.1f} / {s.p90_ttft_ms:.1f} / {s.p99_ttft_ms:.1f} ms"
    )
    if s.steady_median_ttft_ms is not None:
        print(f" TTFT  steady p50/p99:     {s.steady_median_ttft_ms:.1f} / {s.steady_p99_ttft_ms:.1f} ms")
    print(
        f" TPOT  mean/p50/p90/p99:   {s.mean_tpot_ms:.1f} / {s.median_tpot_ms:.1f} / {s.p90_tpot_ms:.1f} / {s.p99_tpot_ms:.1f} ms"
    )
    print(
        f" ITL   mean/p50/p90/p99:   {s.mean_itl_ms:.1f} / {s.median_itl_ms:.1f} / {s.p90_itl_ms:.1f} / {s.p99_itl_ms:.1f} ms"
    )
    print(
        f" E2EL  mean/p50/p90/p99:   {s.mean_e2el_ms:.1f} / {s.median_e2el_ms:.1f} / {s.p90_e2el_ms:.1f} / {s.p99_e2el_ms:.1f} ms"
    )
    print(f"{'=' * 52}\n")
    if s.itl_coalesced_frac > ITL_COALESCED_WARN_FRAC:
        print(f" WARNING: {s.itl_coalesced_frac:.1%} of inter-token gaps are under {ITL_COALESCED_MS} ms.")
        print("          The client event loop fell behind and stamped several tokens together,")
        print("          so ITL and TPOT are distorted. Lower the concurrency or add client processes.\n")
    if s.timeout_requests:
        pct = 100.0 * s.timeout_requests / s.num_requests if s.num_requests else 0.0
        print(f" WARNING: {s.timeout_requests} request(s) ({pct:.1f}%) hit the client timeout and were cut off.")
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
            print(
                f" ({s.failed_with_partial_ttft} failed request(s) recorded a partial "
                f"TTFT; see per_request in the result JSON)"
            )
    if s.errors:
        print(f" Errors ({len(s.errors)} total, first {min(3, len(s.errors))}):")
        for e in s.errors[:3]:
            print(f"   {e}")
        print()


# Only on a server with spec decoding.
SPEC_COUNTERS = {
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens",
    "drafts": "vllm:spec_decode_num_drafts",
}

# name{labels} value [timestamp]; label values may hold escaped quotes.
SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+\S+)?$")
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse_prometheus(text):
    """Unparsable lines are skipped."""
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
    """Counts both X and X_total."""
    names = (name, name + "_total")
    vals = [v for k, v in metrics.items() if (k[0] if isinstance(k, tuple) else k) in names]
    return sum(vals) if vals else None


def spec_decode_counts(metrics):
    counts = {key: counter_total(metrics, name) for key, name in SPEC_COUNTERS.items()}
    return None if None in counts.values() else counts


# Event k>0 arrives at t_first_wall + sum(deltas_ms[:k])/1e3.
def summarize_cell(records: list[dict], min_samples: int = 64, lag_limit_ms: float = 2.0) -> dict:
    if not records:
        raise ValueError("empty cell")
    b = len(records)
    firsts, lasts, ev_walls = [], [], []
    for r in records:
        ts = [r["t_first_wall"]]
        for d in r["deltas_ms"]:
            ts.append(ts[-1] + d / 1e3)
        ev_walls.append(ts)
        firsts.append(ts[0])
        lasts.append(ts[-1])
    w_lo, w_hi = max(firsts), min(lasts)
    steady_window_s = w_hi - w_lo

    per_req_p50, per_req_progress, per_req_n = [], [], []
    coalesced = total_deltas = 0
    for ts in ev_walls:
        # Delta k (event k to k+1) is in the window only if both ends are.
        din = [(ts[k + 1] - ts[k]) * 1e3 for k in range(len(ts) - 1) if ts[k] >= w_lo and ts[k + 1] <= w_hi]
        idx_in = [k + 1 for k in range(len(ts)) if w_lo <= ts[k] <= w_hi]  # 1-based token index
        per_req_n.append(len(din))
        if din:
            per_req_p50.append(statistics.median(din))
            coalesced += sum(1 for d in din if d < 0.5)
            total_deltas += len(din)
        if idx_in:
            per_req_progress.append(statistics.median(idx_in))

    lag_p99 = max((float(r.get("lag_p99_ms") or 0.0) for r in records), default=0.0)
    status = "ok"
    if steady_window_s <= 0 or len(per_req_p50) < b or min(per_req_n) < min_samples:
        status = "check"
    if lag_p99 > lag_limit_ms:
        status = "check"

    prompt = int(records[0]["prompt_tokens"])
    progress = statistics.median(per_req_progress) if per_req_progress else 0.0
    return {
        "batch_size": b,
        "context_len": int(round(prompt + progress)),  # effective context, the grid axis
        "decode_step_ms": round(statistics.median(per_req_p50), 4) if per_req_p50 else 0.0,
        "validation_status": status,
        "nominal_T": int(records[0].get("nominal_T", 0)),
        "prompt_tokens": prompt,
        "osl": int(records[0]["osl"]),
        "n_samples": min(per_req_n) if per_req_n else 0,  # min in-window deltas per request
        "steady_window_s": round(steady_window_s, 3),
        "median_inwindow_progress": round(progress, 1),
        "p99_loop_lag_ms": round(lag_p99, 3),
        "coalesce_frac": round(coalesced / total_deltas, 4) if total_deltas else 0.0,
        "itl_p50_spread_ms": round(max(per_req_p50) - min(per_req_p50), 4) if per_req_p50 else 0.0,
    }


SUMMARY_FIELDS = [
    "batch_size",
    "context_len",
    "decode_step_ms",
    "validation_status",
    "nominal_T",
    "prompt_tokens",
    "osl",
    "n_samples",
    "steady_window_s",
    "median_inwindow_progress",
    "p99_loop_lag_ms",
    "coalesce_frac",
    "itl_p50_spread_ms",
]
