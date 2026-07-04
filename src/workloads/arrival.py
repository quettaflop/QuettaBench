"""
Request arrival pattern generators.

Supports:
- "steady": fixed concurrency (InferenceX default — kept for comparison)
- "poisson": Poisson process arrivals at target rate (most realistic for web services)

Each generator yields timestamps (seconds from benchmark start) at which
to dispatch each request.
"""

import random
import math
from typing import Iterator


def steady_arrivals(num_requests: int, concurrency: int) -> list[float]:
    """
    Fixed concurrency: dispatch requests in batches of `concurrency`.
    Requests within a batch start simultaneously (t=0 offset within batch).
    Returns list of dispatch times (seconds).

    This matches llm-bench and InferenceX default behavior.
    """
    times = []
    # All requests at t=0 (fire-and-forget concurrency control via semaphore)
    for i in range(num_requests):
        times.append(0.0)
    return times


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


def make_arrival_times(
    pattern: str,
    num_requests: int,
    concurrency: int = 10,
    target_rate: float = 10.0,
    seed: int = 42,
) -> list[float]:
    """
    Factory: create arrival times for the given pattern.

    Args:
        pattern: "steady", "poisson", or "ramp"
        num_requests: number of requests
        concurrency: used by "steady" pattern
        target_rate: requests/sec for "poisson" and "ramp"
        seed: random seed
    """
    if pattern == "steady":
        return steady_arrivals(num_requests, concurrency)
    elif pattern == "poisson":
        return poisson_arrivals(num_requests, target_rate, seed)
    elif pattern == "ramp":
        return ramp_arrivals(num_requests, target_rate * 0.1, target_rate, seed)
    else:
        raise ValueError(f"Unknown arrival pattern: '{pattern}'. Use 'steady', 'poisson', or 'ramp'.")
