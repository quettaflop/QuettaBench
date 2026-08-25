#!/usr/bin/env python3
"""Shared NCU timing for the kernel probes (the STANDARD; matches gemm_probe --ncu).

Runs a self-contained python snippet under NCU, reads per-kernel
`gpu__time_duration.sum`, and returns the min-over-invocations SUM of the op's
kernels. Multi-kernel ops (GDN fires ~8 Triton sub-kernels per call; FA3 fires
~3) are handled by summing each invocation's kernels and taking the min warm
invocation.

Kernel isolation: the snippet's ONLY non-op kernels are torch tensor setup
(randn / fill / arange / neg), which are all ATen `at::` kernels; the op's own
kernels (FLA Triton `chunk_*`/`l2norm_*`/...; cutlass `flash::...`) never contain
`at::`. So excluding `at::` isolates the op across both ops and all three GPUs.
"""
from __future__ import annotations

import csv
import io
import subprocess
import sys


def ncu_kernel_durations(inner: str, ncu_bin: str) -> list[tuple[str, float]]:
    """[(kernel_name, us)] for every profiled kernel launch, in launch order."""
    cmd = [ncu_bin, "--csv", "--metrics", "gpu__time_duration.sum",
           "--target-processes", "all", sys.executable, "-c", inner]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    lines = out.splitlines()
    hdr = next((i for i, l in enumerate(lines) if l.startswith('"ID"')), None)
    if hdr is None:
        return []
    res: list[tuple[str, float]] = []
    for r in csv.DictReader(io.StringIO("\n".join(lines[hdr:]))):
        if r.get("Metric Name") != "gpu__time_duration.sum":
            continue
        v = float(r["Metric Value"].replace(",", ""))
        us = v / 1000.0 if r.get("Metric Unit") == "ns" else v
        res.append((r.get("Kernel Name", ""), us))
    return res


def ncu_op_us(inner: str, ncu_bin: str, reps: int,
              exclude: tuple[str, ...] = ("at::",)) -> float | None:
    """Pure-kernel us of ONE op call: the snippet calls the op `reps` times; sum
    the op's kernels (drop `exclude` setup kernels) per invocation, min over the
    invocations (the warmest = steady state). None if NCU produced no table or the
    op-kernel count isn't a clean multiple of reps (autotune leak -> tune filter)."""
    durs = [us for kn, us in ncu_kernel_durations(inner, ncu_bin)
            if not any(t in kn for t in exclude)]
    if not durs or reps <= 0 or len(durs) % reps != 0:
        return None
    n = len(durs) // reps               # op sub-kernels per invocation
    groups = [sum(durs[i * n:(i + 1) * n]) for i in range(reps)]
    return min(groups)
