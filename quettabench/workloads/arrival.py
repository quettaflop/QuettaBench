"""Arrival schedules: parse_arrival turns a spec (trace, steady, poisson:<r>, ramp:<a>:<b>, burst:<n>x<size>@<gap>) into offsets in seconds from the start of the run."""

import math
import random


def poisson_arrivals(num_requests: int, target_rate: float, seed: int = 42) -> list[float]:
    """Cumulative arrival times in seconds of a Poisson process at target_rate requests per second."""
    rng = random.Random(seed)
    times = []
    t = 0.0
    for _ in range(num_requests):
        inter_arrival = -math.log(1.0 - rng.random()) / target_rate
        t += inter_arrival
        times.append(t)
    return times


def ramp_arrivals(num_requests: int, start_rate: float, end_rate: float, seed: int = 42) -> list[float]:
    """Cumulative arrival times with the rate rising linearly from start_rate to end_rate."""
    rng = random.Random(seed)
    times = []
    t = 0.0
    for i in range(num_requests):
        fraction = i / max(num_requests - 1, 1)
        rate = start_rate + fraction * (end_rate - start_rate)
        inter_arrival = -math.log(1.0 - rng.random()) / rate
        t += inter_arrival
        times.append(t)
    return times


def burst_arrivals(num_requests: int, n_bursts: int, burst_size: int, gap_s: float) -> list[float]:
    """Arrival times for n_bursts groups of burst_size requests, gap_s seconds apart; raises ValueError when num_requests exceeds n_bursts * burst_size."""
    if num_requests > n_bursts * burst_size:
        raise ValueError(
            f"burst spec covers {n_bursts * burst_size} requests but the trace has "
            f"{num_requests}; raise the spec or use --limit")
    times = []
    for b in range(n_bursts):
        take = min(burst_size, num_requests - len(times))
        times.extend([b * gap_s] * take)
        if len(times) >= num_requests:
            break
    return times


def positive(rate, spec):
    """rate when it is a finite number > 0 (nan, inf and <= 0 are refused)."""
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError(f"arrival {spec!r}: rate must be a finite number > 0")
    return rate


def parse_arrival(spec: str, n: int, seed: int = 42) -> list[float] | None:
    """Arrival offsets in seconds for n requests from a spec; trace returns None to keep the trace's own times, steady is all at 0."""
    if spec == "trace":
        return None
    if spec == "steady":
        return [0.0] * n
    kind, _, rest = spec.partition(":")
    if kind == "poisson":
        return poisson_arrivals(n, positive(float(rest), spec), seed)
    if kind == "ramp":
        a, b = rest.split(":")
        return ramp_arrivals(n, positive(float(a), spec), positive(float(b), spec), seed)
    if kind == "burst":
        nb, rest2 = rest.split("x")
        size, gap = rest2.split("@")
        nb, size, gap = int(nb), int(size), float(gap)
        if nb <= 0 or size <= 0 or not math.isfinite(gap) or gap < 0:
            raise ValueError(f"arrival {spec!r}: bursts and size must be > 0, gap finite and >= 0")
        return burst_arrivals(n, nb, size, gap)
    raise ValueError(f"unknown arrival spec {spec!r}")


def offered_rate(spec: str) -> float:
    """The rate a spec offers, req/s: poisson's rate, ramp's end rate, else 0.0."""
    kind, _, rest = spec.partition(":")
    if kind == "poisson":
        return float(rest)
    if kind == "ramp":
        return float(rest.split(":")[1])
    return 0.0


def rebase_arrivals(ts: list, speed: float) -> list[float]:
    """Arrival times shifted by the earliest and divided by speed; a missing time counts as 0 and submits at once, no times at all gives all 0."""
    if all(t is None for t in ts):
        return [0.0] * len(ts)
    t0 = min(0.0 if t is None else t for t in ts)
    return [max(0.0, (t0 if t is None else t) - t0) / max(speed, 1e-9) for t in ts]
