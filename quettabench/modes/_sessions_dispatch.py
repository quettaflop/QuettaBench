"""Multi-turn dispatch for serve: a SessionTrace's turns sent in session order.

Three schedules, picked by load and the trace's pacing:

  closed, per-session  each session runs its turns back to back; sessions
                       share the concurrency slots; no cross-session barrier
  closed, interleaved  a barrier after every turn index across sessions (a
                       turn-aligned herd; TTFT/TPOT spikes at turn
                       boundaries are schedule artifacts); waits ignored
  open                 sessions arrive on the arrival schedule (or their
                       recorded arrival); turns sequential within a session,
                       no barrier, no cap

A session whose turn fails stops there. A successful turn's reply is
written into the session's assistant placeholder (trace.replies), so later
turns carry the engine's own text; a trace without placeholders keeps its
recorded replies.

Kept apart from serve.py so the procedure there stays under 500 lines;
this is the session half of "drive requests and collect rows".
"""

import asyncio
import itertools
import time
from contextlib import asynccontextmanager

from ..metrics.core import (
    annotate_multi_turn_cache_estimate,
    annotate_request_observability,
    first_error,
)
from ..workloads.arrival import offered_rate, parse_arrival
from ..workloads.sessions import _sleep_until_elapsed, inter_turn_wait_s


class RunAborted(Exception):
    """The run cannot produce a measurement; the message is the ABORT line."""


@asynccontextmanager
async def no_limit():
    """Stand-in for the concurrency semaphore when running open loop."""
    yield


class SessionDispatcher:
    """One multi-turn replay. send(request, request_id=, capture_text=) is
    the driver call with sampling bound; request_id(session, turn, index)
    names a request or returns None; every result goes to writer.append."""

    def __init__(self, trace, mode, workload, send, request_id, writer):
        self.trace, self.mode, self.workload = trace, mode, workload
        self.send, self.request_id, self.writer = send, request_id, writer
        self.sessions = trace.sessions()
        self.max_turns = getattr(trace, "max_turns", max(len(s) for s in self.sessions))
        self.pacing = trace.pacing
        self.open_loop = mode.load == "open"
        self.basis = "arrival" if self.open_loop else "slot"

    def announce(self):
        """Before warmup: say what interleaved pacing does to the numbers."""
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
        """(results in turn order, results_by_turn, served requests, start time)."""
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
        result.ttft_basis = self.basis
        self.served.append(request)
        self.writer.append(result)
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
