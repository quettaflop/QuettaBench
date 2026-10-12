"""Per-cell step fit: the slope of batch time against step count is the decode cost per step, the intercept the fixed cost (prefill and launch)."""

GIB = 1 << 30


def fit(xs, ys):
    """Least-squares line through (xs, ys) as (slope, intercept, r2); r2 is 0.0 when ys has no spread."""
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
    """Slope moved from the fit centroid (ctx + mean_steps) to ctx + eval_pos; only the KV-bound share (attn_share) of a step grows with KV length."""
    return ms_slope * (1 + attn_share * (eval_pos - mean_steps) / (ctx + mean_steps))


def _get(row, name):
    """A row field from a RequestResult or from a BenchDriver dict row."""
    return row.get(name) if isinstance(row, dict) else getattr(row, name, None)


def fit_run(rows, eval_pos, kv_bytes, weights_gib):
    """One step entry per grid cell, in order of first appearance.

    A row with measured_ms_per_step passes through as {ctx, bs, tp, ms_per_step, measured: True}. Other rows count only in phase "fit": per attempt the first row's batch wall at each step point is fit against the step points, and the highest r2 wins, the earliest on a tie."""
    cells = {}
    for row in rows:
        cell = _get(row, "cell")
        if cell is None:
            continue
        key = (cell["ctx"], cell["bs"], cell["tp"])
        measured = _get(row, "measured_ms_per_step")
        if measured is not None:
            cells[key] = {"ctx": key[0], "bs": key[1], "tp": key[2], "ms_per_step": measured, "measured": True}
        elif cell.get("phase") == "fit":
            walls = cells.setdefault(key, {}).setdefault(cell["attempt"], {})
            walls.setdefault(cell["step_point"], _get(row, "batch_wall_s"))
    return [
        cell if cell.get("measured") else _fit_cell(key, cell, eval_pos, kv_bytes, weights_gib)
        for key, cell in cells.items()
    ]


def _fit_cell(key, attempts, eval_pos, kv_bytes, weights_gib):
    ctx, bs, tp = key
    best = None  # (r2, slope, intercept, xs, ys)
    for walls in attempts.values():
        xs, ys = list(walls), list(walls.values())
        slope, inter, r2 = fit(xs, ys)
        if best is None or r2 > best[0]:
            best = (r2, slope, inter, xs, ys)
    r2, slope, inter, xs, ys = best
    mean_steps = sum(xs) / len(xs)
    kv_gib = ctx * bs * kv_bytes / GIB
    attn = kv_gib / (kv_gib + weights_gib)
    ms = slope * 1e3
    return {
        "ctx": ctx,
        "bs": bs,
        "tp": tp,
        "ms_per_step": ms,
        "ms_at_eval": ms_at_eval(ms, attn, eval_pos, mean_steps, ctx),
        "eval_pos": eval_pos,
        "mean_steps": mean_steps,
        "r2": r2,
        "prefill_s": inter,
        "kv_gib": kv_gib,
        "attn_share": attn,
        "step_points": xs,
        "raw": ys,
        "attempts": len(attempts),
    }
