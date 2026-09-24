"""One timing primitive for every kernel probe: CUDA events, in one of two modes.

    graph  -- capture the op into a CUDA graph and time replay (see _graph.py).
              No per-launch dispatch, which is how vLLM runs DECODE. Reduce: MIN
              over replays (a decode step runs its kernels warm and back-to-back;
              matches the tables' min-on-load semantics).
    eager  -- time each call as launched, dispatch included, which is how vLLM
              runs PREFILL (dynamic shapes are not graphed). Reduce: MEDIAN over
              calls (dispatch jitter is part of the cost; min would drop it).

Tables timed in graph mode go under kernel_data/graph/, eager ones under eager/.
NCU (_ncu.py) is kept only as a cross-check of the graph numbers, not as a source.
"""
from __future__ import annotations

import statistics as st

import torch

MODES = ("graph", "eager")
REDUCE = {"graph": "min", "eager": "median"}


def eager_time_us(fn, *, reps: int = 50, warmup: int = 10) -> float:
    """Median CUDA-event wall-clock of one call (us), dispatch included."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)  # ms -> us
    return st.median(ts)


FLUSH_BYTES = int(__import__("os").environ.get("PROBE_FLUSH_L2_BYTES", "0"))   # cold-L2 graph timing (see _graph)


def time_us(fn, *, graph: bool, reps: int = 50, warmup: int = 10) -> float:
    """``graph=True`` -> graph-replay min; ``graph=False`` -> eager median.
    PROBE_FLUSH_L2_BYTES=<n> in the environment makes graph timing cold-L2 (_graph)."""
    if graph:
        from _graph import graph_time_us  # noqa: PLC0415
        return graph_time_us(fn, warmup=warmup, reps=reps, reduce=min, flush_bytes=FLUSH_BYTES)
    return eager_time_us(fn, reps=reps, warmup=warmup)


def mode_name(graph: bool) -> str:
    return "graph" if graph else "eager"
