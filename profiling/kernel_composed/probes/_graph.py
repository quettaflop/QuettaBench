"""Shared CUDA-graph-replay timing — the graphed-decode-faithful method when NCU
counters are unavailable (e.g. hosts with RmProfilingAdminOnly=1, where no
container setting can unlock them).

vLLM's decode is CUDA-graphed: kernels replay with no per-launch CPU dispatch,
so the faithful decode number is pure kernel time. NCU measures that by
counter; capturing the op into a torch.cuda.CUDAGraph and timing replay()
measures the same thing by construction (H200 spot-check, vLLM fused_experts
at 8 decode tokens: eager 182us -> graph replay 78.5us — the 57%
delta is exactly the dispatch cost graphs eliminate). Replay timing even keeps
real back-to-back kernel behaviour that per-kernel NCU isolation loses.

Prefill stays eager in serving, so prefill grids keep plain eager (CUDA-event) timing.
"""
from __future__ import annotations

import statistics as st

import torch


def graph_time_us(fn, *, warmup: int = 10, reps: int = 50, reduce=min) -> float:
    """Capture ``fn`` into a CUDA graph and time replay (us, ``reduce`` over reps).

    ``fn`` must be shape-static with all inputs pre-allocated (same contract as
    vLLM's decode capture). Eager warmup runs first so autotuners (triton
    fused_moe configs, cuBLAS heuristics) pick their kernels OUTSIDE capture.

    Each capture uses its own private memory pool: the pool is retired when the
    graph is deleted at the end, so sweeping thousands of cells does not
    accumulate pools. (Sharing one pool handle across captures looks thriftier
    but is invalid — the handle dies with the first graph and the next capture
    trips a CUDACachingAllocator use_count assert.)
    """
    for _ in range(3):
        fn()
    # One cudaGraphLaunch costs ~6us; a real decode step launches ONE graph for
    # the whole step, so per-kernel launch is ~0. Amortize by capturing the op
    # `iters` times per graph and dividing -- iters chosen from a quick eager
    # estimate so big ops (where 6us is noise anyway) don't blow pool memory.
    torch.cuda.synchronize()
    s0, e0 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s0.record(); fn(); e0.record(); torch.cuda.synchronize()
    est_us = s0.elapsed_time(e0) * 1000.0
    iters = 32 if est_us < 30.0 else (8 if est_us < 200.0 else 1)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()

    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)  # ms -> us
    del g
    return reduce(ts) / iters


def graph_median_us(fn, **kw) -> float:
    return graph_time_us(fn, reduce=st.median, **kw)
