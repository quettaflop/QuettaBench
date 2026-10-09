"""serve.run's grid branch: decode ms/step per (ctx, bs) cell, as the slope
of batch wall time over a few step points (tokens generated).

run_grid prints META and gates the cells on KV capacity: a cell needing
more than mode.cap_frac of the KV tokens left after weights and a 5 GiB
per-GPU reserve is SKIPped, as is one past max_model_len. After one
warmup it measures each cell: a warm run at the top step point, then up
to mode.vmin_attempts passes over the step points, stopping at the first
pass that fits r2 >= mode.fit_r2_stop. Every batch goes through the
driver, every row through the writer; RESULT is metrics.step_fit.fit_run
over the cell's rows. Lines print through results.print_legacy_grid_lines
as the run advances, byte for byte what modes/vmin_fit.py printed.

A driver that times whole cells in its own process (BenchDriver, the
QuettaServe cargo bench) goes through run_bench instead: its rows are
already ms per step, so there is no capacity gate, warmup or fit.

A sibling of serve.py the way _sessions_dispatch.py is, to keep serve.py
one screenful per mechanism.
"""

import json
import os
import sys

from ..metrics.core import RequestResult
from ..metrics.results import print_legacy_grid_lines
from ..metrics.step_fit import GIB, fit, fit_run
from ..workloads.base import Request

RESERVE_GIB = 5.0   # per GPU, held back from the KV budget for activations and graphs
TEXT_TOKENS = 64    # mode.returns=text: greedy tokens decoded per cell


def kv_bytes_per_token(engine):
    """engine.kv_bytes, else the dense formula over the model's config.json.
    The formula counts every layer; hybrid and MLA stacks must set kv_bytes."""
    if engine.kv_bytes:
        return int(engine.kv_bytes)
    with open(os.path.join(engine.model, "config.json")) as f:
        cfg = json.load(f)
    cfg = cfg.get("text_config", cfg)
    hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    hd += cfg.get("qk_rope_head_dim") or 0
    return 2 * cfg["num_hidden_layers"] * cfg["num_key_value_heads"] * hd * 2


def graph_mode(engine):
    return "FULL" if engine.cudagraph else "NOGRAPH"


def meta(cfg, gpu, version, eval_pos, steps):
    """META key -> value, in vmin_fit's order; expert_parallel only when on."""
    e, h = cfg.engine, cfg.hardware
    out = {"gpu": gpu, "vllm": version, "model": e.model_name, "dtype": e.dtype,
           "grid": cfg.mode.grid, "mode": graph_mode(e), "eval_pos": eval_pos,
           "step_points": list(steps),
           "nccl_algo": h.collective.get("NCCL_ALGO") or "auto",
           "nccl_proto": h.collective.get("NCCL_PROTO") or "auto",
           "link_profile": h.link_profile, "prompt_mode": cfg.workload.prompt_mode,
           "tp": e.tp, "pp": e.pp, "quant": e.quant or e.dtype}
    if e.ep:
        out["expert_parallel"] = 1
    return out


def capacity(engine, gpu_bytes, kv_b):
    """(free GiB for KV across tp GPUs, KV tokens that fit; 0 when none)."""
    gpu_gib = gpu_bytes / GIB
    free_gib = engine.tp * (gpu_gib * engine.gpu_util) - engine.weights_gib - engine.tp * RESERVE_GIB
    return free_gib, int(free_gib * GIB / kv_b) if free_gib > 0 else 0


def measure_cell(driver, writer, mode, steps, prompts, cell):
    """The cell's batches in vmin_fit's order: the text run when asked, the
    warm run at max(steps), then the fit passes. Returns the fit rows."""
    def batch(step, phase, attempt=None, **extra):
        requests = [Request(ids, len(ids), output_length=step) for ids in prompts]
        rows = list(driver.run(requests, cell={**cell, "step_point": step, "phase": phase,
                                               "attempt": attempt}))
        for r in rows:
            r.excluded_from_summary = phase != "fit"
            writer.append(r, **{k: f(r) for k, f in extra.items()})
        return rows

    if mode.returns == "text":
        # Routing-parity audit: greedy ids must match the engine's DS_TOKEN_TRACE.
        batch(TEXT_TOKENS, "text", token_ids=lambda r: r.generated_token_ids[:TEXT_TOKENS],
              text=lambda r: r.generated_text)
    batch(max(steps), "warm")
    out = []
    for attempt in range(mode.vmin_attempts):
        walls = []
        for step in steps:
            rows = batch(step, "fit", attempt)
            out += rows
            walls.append(rows[0].batch_wall_s)
        if fit(steps, walls)[2] >= mode.fit_r2_stop:
            break
    return out


def run_bench(driver, writer):
    """Every cell of driver.cells through the bench; each row is written
    with its measured ms per step, and steps passes them through."""
    rows = list(driver.run(driver.cells))
    for row in rows:
        writer.append(RequestResult(success=True, cell=row["cell"]),
                      **{k: v for k, v in row.items() if k != "cell"})
    return {"bench_log": str(driver.log_path)}, fit_run(rows, None, None, None), rows


def run_grid(trace, driver, cfg, writer):
    """One grid run; returns (record, steps, rows). record is what the
    legacy lines print from and the run.json grid section; steps is
    fit_run over every fit row."""
    if getattr(driver, "cells", None):
        return run_bench(driver, writer)
    mode, engine = cfg.mode, cfg.engine
    eval_pos = mode.eval_pos if mode.eval_pos is not None else cfg.hardware.timing_steps // 2
    steps = trace.step_points
    kv_label = engine.kv_dtype if engine.kv_dtype != "auto" else engine.dtype
    gpu, gpu_bytes = driver.device()
    run = {"meta": meta(cfg, gpu, driver.version, eval_pos, steps), "kv": kv_label, "events": []}
    if trace.trace is not None:
        reqs = trace.trace
        run["trace"] = {"path": cfg.workload.source, "reqs": len(reqs.requests()),
                        "workload_hash": reqs.workload_hash()}
    shown = print_legacy_grid_lines(run)

    kv_b = kv_bytes_per_token(engine)
    free_gib, cap = capacity(engine, gpu_bytes, kv_b)
    run["capacity"] = {"mode": graph_mode(engine), "tp": engine.tp, "dtype": engine.dtype,
                       "kv_bytes": kv_b, "free_gib": free_gib, "kv_tokens": cap}
    shown = print_legacy_grid_lines(run, shown)
    driver.warmup()
    run["warm_ok"] = True
    shown = print_legacy_grid_lines(run, shown)

    over, top, rows = set(trace.cells_over_maxlen), max(steps), []
    for ctx, bs in trace.cells:
        need = (ctx + top) * bs
        if (ctx, bs) in over:
            run["events"].append({"event": "skip_maxlen", "ctx": ctx, "bs": bs})
        elif cap and need > mode.cap_frac * cap:
            run["events"].append({"event": "skip_kv", "ctx": ctx, "bs": bs, "need": need,
                                  "cap": cap})
        else:
            # Checked at the first cell that needs it, so an all-SKIP run never asks.
            if trace.prompt_mode == "corpus" and not trace.corpus_path:
                reason = "prompt_mode=corpus needs XVAL_CORPUS pointing at a text file"
                writer.finish(None, status="aborted", reason=reason)
                sys.exit(reason)
            prompts, cycled = trace.cell_prompts(ctx, bs)
            if cycled:
                run["events"].append({"event": "cycled", "ctx": ctx, "bs": bs})
                shown = print_legacy_grid_lines(run, shown)
            cell_rows = measure_cell(driver, writer, mode, steps, prompts,
                                     {"ctx": ctx, "bs": bs, "tp": engine.tp})
            rows += cell_rows
            (step,) = fit_run(cell_rows, eval_pos, kv_b, engine.weights_gib)
            run["events"].append({"event": "result", "ctx": ctx, "bs": bs, "step": step})
        shown = print_legacy_grid_lines(run, shown)
    run["done"] = True
    print_legacy_grid_lines(run, shown)
    return run, fit_run(rows, eval_pos, kv_b, engine.weights_gib), rows
