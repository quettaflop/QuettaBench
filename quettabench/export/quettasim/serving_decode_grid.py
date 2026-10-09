#!/usr/bin/env python3
# quettabench/export/quettasim/serving_decode_grid.py
"""Deterministic builder: raw serving-decode-grid JSONL.gz run(s) -> grid CSV.

The measurement client (``quettabench/probes/serving_decode_grid.py``) writes an
APPEND-ONLY raw per-request JSONL.gz per session (wall-anchored SSE event timestamps). This
builder re-derives every cell row from the raw events with the SAME pre-registered
post-processing (``summarize_cell``, imported from the client module — single source of truth):
steady window = [max first-token, min last-token]; per-request p50 mid-stream ITL inside the
window (>= 64 in-window deltas per request, else ``check``); cell decode_step_ms = median over
the per-request p50s; effective context_len = prompt + median in-window progress. Where a cell
appears in several inputs, the LATEST input (list order = chronological) wins — the
``build_decode_grid.py`` merge rule.

Output CSV columns (L11 pre-registration):
    batch_size, context_len, decode_step_ms, validation_status,
    nominal_T, prompt_tokens, osl, n_samples, steady_window_s, ...diagnostics
``simulator.kernel_step_cost.load_grid`` reads only the first four; extras are harmless
(DictReader). The grid consumer is agnostic to how cells were measured.

Deterministic (no RNG). ``run`` groups the results dirs' serving_decode_grid_<tag>_<date>
.jsonl[.gz] runs by tag, oldest date first, and writes dest/results/serving_decode_grid_<tag>_<latest
date>.csv.
"""
from __future__ import annotations

import csv
import gzip
import json
import re
from pathlib import Path
from typing import Iterable

from quettabench.probes import serving_decode_grid as _client  # aiohttp is imported lazily there

summarize_cell = _client.summarize_cell
SUMMARY_FIELDS = _client.SUMMARY_FIELDS
OUT_FIELDS = SUMMARY_FIELDS + ["effective_context_len", "source_file"]


def read_run(path: Path) -> tuple[dict, dict[tuple[int, int], list[dict]]]:
    """One raw JSONL.gz -> (meta, {(B, nominal_T): [request records]})."""
    meta: dict = {}
    cells: dict[tuple[int, int], list[dict]] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("_meta"):
                meta = d
                continue
            key = (int(d["cell"][0]), int(d["cell"][1]))
            cells.setdefault(key, []).append(d)
    return meta, cells


_LAG_UPGRADE_SPREAD = 0.005      # the L11 cross-pass reproducibility standard (<=0.5%)
_LAG_UPGRADE_HEADROOM = 1.5      # latest-pass p99 loop lag must be <= 1.5x the flag limit
LAG_UPGRADE_FIELD = "lag_reproducibility_upgrade"


def build(inputs: list[Path], snap_tolerance: float = 0.02,
          lag_upgrade: bool = False) -> list[dict]:
    """Merge runs (latest wins per cell) and summarize.

    ``snap_tolerance`` (L11 builder amendment, decided PRE-GATE on consumer-structure grounds,
    no validation data consulted): the lattice's ``prompt = T - osl/2`` design targets an
    effective context == nominal T at the steady-window middle, and the measured effectives land
    within ~1.5% of nominal — but as raw floats they are UNIQUE per (B, T) cell, which makes
    every ``load_grid`` t-axis column single-B (ragged), so the consumer's bilinear corners mix
    measured cells with analytic fills almost everywhere, diluting the measurement. The builder
    therefore SNAPS ``context_len`` to the cell's nominal T when the measured effective is
    within ``snap_tolerance`` of it (the by-construction case; rows align into a proper B x T
    grid) and keeps the measured effective in the ``effective_context_len`` diagnostic column.
    A deviation beyond the tolerance means the steady window was asymmetric/disturbed: the cell
    keeps its measured effective AND is flagged ``check``.

    ``lag_upgrade`` (L13 builder amendment, DEFAULT OFF — every pre-L13 artifact regenerates
    byte-identically; decided on consumer-structure grounds before the L13 grid wiring gate):
    the p99-loop-lag ``check`` flag exists to catch CLIENT-side measurement distortion, and the
    L11 pass-2 finding was that cross-topology reproducibility falsifies that distortion
    ("values reproduce <=0.5% vs run 1 -> the flags are client-loop scheduling noise ... not
    measurement distortion", L11-h100multi). ``load_grid`` drops non-``ok`` rows, so a
    marginally-over-limit lag flag (e.g. 2.01 vs the 2.0 ms limit on a host whose CPUs also run
    the tensor-parallel server) silently replaces a reproducible MEASURED cell with the analytic
    fill the grid exists to correct. With ``lag_upgrade=True`` a flagged cell is upgraded to
    ``ok`` iff ALL deterministic criteria hold:
      * the cell was measured in >= 2 input passes (independent client topologies);
      * EVERY pass is hygiene-clean (no empty-stream drops, in-window deltas >= min_samples,
        positive steady window) — i.e. the only flag reason anywhere is loop lag;
      * the final (latest-wins) row's flag is lag-only, with the snap check passing;
      * cross-pass decode_step_ms spread <= 0.5% of the adopted (latest) value;
      * the adopted pass's p99 loop lag is <= 1.5x the flag limit (a genuinely lag-disturbed
        run cannot be upgraded by an agreeing pair — it must be re-measured clean).
    Upgraded rows carry the evidence (pass count, spread, per-pass lags) in the
    ``lag_reproducibility_upgrade`` diagnostic column."""
    import inspect
    sig = inspect.signature(summarize_cell)
    min_samples = sig.parameters["min_samples"].default
    lag_limit_ms = sig.parameters["lag_limit_ms"].default

    history: dict[tuple[int, int], list[tuple[list[dict], str]]] = {}
    for path in inputs:
        meta, cells = read_run(path)
        for key, recs in cells.items():
            if key in history:
                print(f"cell {key}: superseded by {path.name}")
            history.setdefault(key, []).append((recs, path.name))
    rows = []
    for key in sorted(history):
        recs, src = history[key][-1]
        ok = [r for r in recs if r.get("t_first_wall") is not None]
        dropped = len(recs) - len(ok)
        row = summarize_cell(ok)
        if dropped:
            row["validation_status"] = "check"
            print(f"cell {key}: {dropped} empty-stream request(s) -> check")
        eff = row["context_len"]
        nominal = row["nominal_T"]
        row["effective_context_len"] = eff
        snap_ok = nominal > 0 and abs(eff - nominal) / nominal <= snap_tolerance
        if snap_ok:
            row["context_len"] = nominal
        else:
            row["validation_status"] = "check"
            print(f"cell {key}: effective context {eff} deviates >{snap_tolerance:.0%} "
                  f"from nominal {nominal} -> kept + check")
        if (lag_upgrade and row["validation_status"] == "check" and not dropped and snap_ok
                and row["n_samples"] >= min_samples and row["steady_window_s"] > 0
                and len(history[key]) >= 2):
            per_pass = []
            for recs_i, name_i in history[key]:
                ok_i = [r for r in recs_i if r.get("t_first_wall") is not None]
                if len(ok_i) != len(recs_i):
                    per_pass = None
                    break
                row_i = summarize_cell(ok_i)
                if row_i["n_samples"] < min_samples or row_i["steady_window_s"] <= 0:
                    per_pass = None
                    break
                per_pass.append((row_i["decode_step_ms"], row_i["p99_loop_lag_ms"], name_i))
            if per_pass is not None:
                vals = [v for v, _, _ in per_pass]
                spread = (max(vals) - min(vals)) / vals[-1]
                if (spread <= _LAG_UPGRADE_SPREAD
                        and per_pass[-1][1] <= _LAG_UPGRADE_HEADROOM * lag_limit_ms):
                    row["validation_status"] = "ok"
                    row[LAG_UPGRADE_FIELD] = (
                        f"passes={len(per_pass)} spread={spread * 100:.3f}% "
                        f"lags={'/'.join(f'{l:.3f}' for _, l, _ in per_pass)}")
                    print(f"cell {key}: lag-only flag upgraded to ok "
                          f"({row[LAG_UPGRADE_FIELD]})")
        row["source_file"] = src
        rows.append(row)
    return rows


RUN_NAME = re.compile(r"serving_decode_grid_(?P<tag>.+)_(?P<date>[^_]+)\.jsonl(\.gz)?$")


def write_grid(rows: list[dict], out: Path, lag_upgrade: bool = False) -> None:
    out_fields = OUT_FIELDS + ([LAG_UPGRADE_FIELD] if lag_upgrade else [])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in out_fields})
    n_ok = sum(1 for r in rows if r["validation_status"] == "ok")
    print(f"wrote {out} ({len(rows)} cells, {n_ok} ok / {len(rows) - n_ok} check)")


def run(results_dirs: Iterable[Path], dest: Path, lag_upgrade: bool = False) -> list[Path]:
    """Write dest/results/serving_decode_grid_<tag>_<date>.csv, one per tag found."""
    runs: dict[str, list[tuple[str, Path]]] = {}
    for root in results_dirs:
        for path in sorted(Path(root).rglob("serving_decode_grid_*.jsonl*")):
            m = RUN_NAME.match(path.name)
            if m:
                runs.setdefault(m["tag"], []).append((m["date"], path))
    written = []
    for tag, dated in runs.items():
        dated.sort()
        rows = build([p for _, p in dated], lag_upgrade=lag_upgrade)
        out = Path(dest) / "results" / f"serving_decode_grid_{tag}_{dated[-1][0]}.csv"
        write_grid(rows, out, lag_upgrade)
        written.append(out)
    return written
