"""Per-cell step fit: decode cost per step from timings at a few step counts.

A grid cell is timed at each step point (tokens generated); the slope of
time vs steps is the marginal decode cost, the intercept the fixed cost
(prefill and launch).
"""


def fit(xs, ys):
    """Least-squares line through (xs, ys): (slope, intercept, r2).

    r2 is 0.0 when ys has no spread.
    """
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    inter = my - slope * mx
    ss_res = sum((y - (inter + slope * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    return slope, inter, r2


def ms_at_eval(ms_slope, attn_share, eval_pos, mean_steps, ctx):
    """Slope moved from the fit centroid (ctx + mean_steps) to ctx + eval_pos.

    Only the KV-bound share of a step (attn_share) grows with KV length, so
    the shift is attn_share * (eval_pos - mean_steps) / (ctx + mean_steps):
    a sub-2% correction that vanishes as ctx grows.
    """
    return ms_slope * (1 + attn_share * (eval_pos - mean_steps) / (ctx + mean_steps))
