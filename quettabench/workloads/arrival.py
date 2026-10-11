"""Request arrival schedules, one spec syntax for every caller.

parse_arrival turns a spec string (trace | steady | poisson:<r> |
ramp:<a>:<b> | burst:<n>x<size>@<gap>) into offsets in seconds from the
start of the run. serve.py's --arrival/--target-rate pair is translated into
that syntax by arrival_spec.
"""

import math
import random

ARRIVAL_PATTERNS = ["steady", "poisson", "ramp"]


def poisson_arrivals(num_requests: int, target_rate: float, seed: int = 42) -> list[float]:
    """
    Poisson process: inter-arrival times are exponentially distributed.

    Args:
        num_requests: total number of requests to schedule
        target_rate: average requests per second
        seed: random seed for reproducibility

    Returns:
        List of cumulative arrival times (seconds from t=0)
    """
    rng = random.Random(seed)
    times = []
    t = 0.0
    for _ in range(num_requests):
        # Exponential inter-arrival time: mean = 1/rate
        inter_arrival = -math.log(1.0 - rng.random()) / target_rate
        t += inter_arrival
        times.append(t)
    return times


def ramp_arrivals(num_requests: int, start_rate: float, end_rate: float, seed: int = 42) -> list[float]:
    """
    Linearly increasing arrival rate from start_rate to end_rate.
    Useful for finding saturation point.

    Returns list of cumulative arrival times.
    """
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
    """
    Injected bursts: n_bursts groups of burst_size requests land at the same
    instant, gap_s seconds apart. Deterministic (no randomness).

    The spec must cover the trace: raises ValueError when num_requests exceeds
    n_bursts * burst_size, because silently spilling extra requests into a tail
    would blur the recovery window the burst exists to measure.

    Returns list of cumulative arrival times.
    """
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
    if rate <= 0:
        raise ValueError(f"arrival {spec!r}: rate must be > 0")
    return rate


def parse_arrival(spec: str, n: int, seed: int = 42) -> list[float] | None:
    """Arrival offsets in seconds for n requests from one spec string:
    trace (None: keep the trace's own times), steady (all at t=0),
    poisson:<rate>, ramp:<start>:<end>, burst:<n>x<size>@<gap_s>."""
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
        return burst_arrivals(n, int(nb), int(size), float(gap))
    raise ValueError(f"unknown arrival spec {spec!r}")


def offered_rate(spec: str) -> float:
    """The rate a spec offers, req/s: poisson's rate, ramp's end rate, else 0.0."""
    kind, _, rest = spec.partition(":")
    if kind == "poisson":
        return float(rest)
    if kind == "ramp":
        return float(rest.split(":")[1])
    return 0.0


def arrival_spec(pattern: str, target_rate: float) -> str:
    """serve.py's --arrival/--target-rate pair as a parse_arrival spec; ramp
    climbs from a tenth of the rate to the rate."""
    if pattern == "steady":
        return "steady"
    if pattern == "poisson":
        return f"poisson:{target_rate}"
    if pattern == "ramp":
        return f"ramp:{target_rate * 0.1}:{target_rate}"
    raise ValueError(f"Unknown arrival pattern: '{pattern}'. Use 'steady', 'poisson', or 'ramp'.")


def make_arrival_times(
    pattern: str,
    num_requests: int,
    concurrency: int = 10,
    target_rate: float = 10.0,
    seed: int = 42,
) -> list[float]:
    """serve.py's entry: translate the pair, then parse_arrival. concurrency
    is unused (steady paces by semaphore, not by schedule)."""
    return parse_arrival(arrival_spec(pattern, target_rate), num_requests, seed)


def rebase_arrivals(ts: list, speed: float) -> list[float]:
    """Submission offsets: arrival times shifted by the earliest and divided by
    speed. A missing time takes part in the shift as 0 and submits at once;
    no times at all gives all 0 (trace_serve's plan_arrivals, unchanged)."""
    if all(t is None for t in ts):
        return [0.0] * len(ts)
    t0 = min(0.0 if t is None else t for t in ts)
    return [max(0.0, (t0 if t is None else t) - t0) / max(speed, 1e-9) for t in ts]
