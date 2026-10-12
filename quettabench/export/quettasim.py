"""qbench export: results dirs in, QuettaSim's files out; the one route out of QuettaBench.

kernel_tables builds QuettaSim's kernel tables from the probe runs' rows.jsonl into its data root (--kernel-dest). The other builders write fitted constants and grids under --dest; every export writes a manifest of the runs it read.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import functools
import gzip
import inspect
import json
import os
import re
import shlex
import shutil
import statistics
import tempfile
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from quettabench.metrics.results import RUN_MARKERS, bench_sha, write_atomic
from quettabench.probes import serving_decode_grid as _client

TP_DEGREES = (2, 4)
# The finish time of a run with none, or none that parses: older than any.
NEVER = datetime.min.replace(tzinfo=timezone.utc)


def clearing(path: Path, root: Path) -> bool:
    """path sits in a .clearing-* dir under root, left by an interrupted runner.clear_run_dir."""
    return any(part.startswith(".clearing-") for part in Path(path).relative_to(root).parts)


@functools.cache
def run_dirs(root: Path) -> dict[Path, str | None]:
    """{run dir: why export leaves it out, or None when kept} for every run dir under root: run.json missing, unparsable, or a run_status other than ok.

    A server/ dir under a run dir is part of that run; a .clearing-* dir is no run. Cached per root; export() clears it before each builder.
    """
    runs = {}
    dirs = sorted({p.parent for marker in RUN_MARKERS for p in Path(root).rglob(marker) if not clearing(p, root)})
    for run_dir in dirs:
        if run_dir.name == "server" and run_dir.parent in dirs:
            continue
        path = run_dir / "run.json"
        if not path.exists():
            runs[run_dir] = "no run.json"
            continue
        try:
            status = json.loads(path.read_text())["run_status"]["status"]
        except (OSError, ValueError, KeyError, TypeError) as e:
            runs[run_dir] = f"run.json unreadable ({type(e).__name__})"
            continue
        runs[run_dir] = None if status == "ok" else f"run_status {status}"
    return runs


def skipped_runs(root: Path) -> dict[Path, str]:
    """{run dir: why} for every run dir under root that export leaves out."""
    return {d: why for d, why in run_dirs(Path(root)).items() if why}


def owner(path: Path, root: Path, runs: dict[Path, str | None]) -> Path | None:
    """The run dir path belongs to: path itself or its nearest run-dir ancestor up to root, else None."""
    for d in (path, *path.parents):
        if d in runs:
            return d
        if d == root:
            return None
    return None


def files(root: Path, pattern: str, recursive: bool = True) -> list[Path]:
    """Paths under root matching pattern, sorted, without those in a .clearing-* dir or a run dir export leaves out."""
    root = Path(root)
    runs = run_dirs(root)
    hits = root.rglob(pattern) if recursive else root.glob(pattern)
    return [p for p in sorted(hits) if not clearing(p, root) and runs.get(owner(p, root, runs)) is None]


def finished(run_dir: Path | None) -> datetime:
    """provenance.pulled (a remote run, on this machine's clock) or else provenance.finished as a UTC instant; NEVER when neither parses."""
    try:
        provenance = json.loads((Path(run_dir) / "run.json").read_text())["provenance"]
        when = provenance["pulled"] if "pulled" in provenance else provenance["finished"]
        return datetime.fromisoformat(when).astimezone(timezone.utc)
    except (OSError, ValueError, KeyError, TypeError):
        return NEVER


def find(results_dirs: Iterable[Path], name: str) -> Path | None:
    """The file called `name` whose ok run finished last, over every results dir; ties go to the earlier results dir, then the first path."""
    best = None
    for rank, root in enumerate(map(Path, results_dirs)):
        runs = run_dirs(root)
        for path in files(root, name):
            key = (finished(owner(path, root, runs)), -rank)
            if best is None or key > best[0]:
                best = (key, path)
    return best[1] if best else None


def stage_split_legs(results_dirs: Iterable[Path], gpu: str) -> list[tuple[int, Path, Path]]:
    """(tp, tp1 csv, tpN csv) for every tp degree with both stage-split legs, each the last-finished one."""
    tp1 = find(results_dirs, f"serving_stage_split_{gpu}.csv")
    legs = []
    for tp in TP_DEGREES:
        tpn = find(results_dirs, f"serving_stage_split_{gpu}_tp{tp}.csv")
        if tp1 and tpn:
            legs.append((tp, tp1, tpn))
    return legs


# A row's timing -> QuettaSim's kernel_data dir (kernel_composed/provenance.py MODES).
DIRS = {
    "graph_replay": "graph",
    "ncu": "graph",
    "amortized_wallclock": "graph",
    "eager": "eager",
    "host_wallclock": "eager",
}
# The indent and ending of a probe's JSON table, as its own file was written.
JSON_STYLE = {"elementwise": (2, ""), "gemm": (2, "\n"), "moe": (1, "\n")}
# The reduce each probe takes over its timings, for graph_replay and eager tables.
REDUCE = {
    "gemm": "min",
    "moe": "min",
    "elementwise": "median",
    "flash_attn": "median",
    "cross_attn": "median",
    "gdn": "median",
}
# The key a collectives row replaces a dest row on.
COLLECTIVE_KEY = ("op", "world", "bytes")
GEOMETRY = ("n_heads", "n_kv_heads", "head_dim")
# (probe, kind) -> key columns of a CSV row; flash_attn and cross_attn replace every row of a head config.
ROW_KEYS = {
    ("flash_attn", "flash_attn"): GEOMETRY,
    ("flash_attn", "fa3_prefill"): GEOMETRY,
    ("cross_attn", "fa3_cross"): GEOMETRY,
    ("gemm", "gemm"): ("M", "N", "K"),
    ("gemm", "gemm_wide"): ("M", "N", "K"),
    ("moe", "moe"): ("tokens", "n_experts", "top_k", "intermediate", "hidden"),
    ("reparallel", "reparallel"): ("gpu", "world", "tp_from", "tp_to", "total_weight_gb"),
    ("kv_transfer", "kv_transfer"): (
        "gpu",
        "kv_heads_per_rank",
        "head_dim",
        "block_size",
        "layers",
        "layout",
        "tokens",
    ),
}
# Keyed tables whose union is sorted on these columns.
SORTED = {
    ("reparallel", "reparallel"): ROW_KEYS[("reparallel", "reparallel")],
    ("kv_transfer", "kv_transfer"): (
        "gpu",
        "kv_heads_per_rank",
        "layout",
        "tokens",
        "head_dim",
        "block_size",
        "layers",
    ),
}
# Probes that live outside quettabench/probes/, by their sidecar tool.
TOOLS = {"kv_transfer": "quettabench.profiling.kv_transfer"}
# Probes whose run without --append replaces the whole table.
RESET_WITHOUT_APPEND = ("gemm", "moe")
# (probe, kind) whose JSON entries union across runs (gemm's fp8_act_quant).
JSON_UNION = {("gemm", "elementwise")}
# When a dest table finished, as a source: before every run.
OLDEST = datetime.min.replace(tzinfo=timezone.utc)


def _record(run_dir: Path) -> tuple[str | None, list[str], str | None]:
    """(probe, its args, the day it finished) from run_dir/run.json; probe None for a run that is no probe run."""
    doc = json.loads((run_dir / "run.json").read_text())
    mode = (doc.get("config") or {}).get("mode") or {}
    probe = mode.get("probe") if mode.get("mode") == "probe" else None
    try:
        args = shlex.split(mode.get("probe_args") or "")
    except ValueError as e:
        raise SystemExit(f"{run_dir}: probe_args do not parse ({e})") from None
    finished = (doc.get("provenance") or {}).get("finished")
    return probe, args, finished[:10] if isinstance(finished, str) else None


def _gpu(args: list[str]) -> str | None:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--gpu-label", "--gpu")
    return ap.parse_known_args(args)[0].gpu_label


def _sidecar(rel: Path, by_probe: dict[str, list[Source]], runs: set[Path]) -> dict:
    """rel's <file>.meta.json over the runs whose rows it holds; runs timed in two modes make mode unknown."""
    probes, modes, gpus, reduces, days = set(), set(), set(), set(), []
    for probe, sources in by_probe.items():
        for _, table, run, _ in (s for s in sources if s[2] in runs):
            _, args, day = _record(run)
            for mode in {"reference"} if rel.stem.endswith("_nccl_vs_vllm") else table.timings:
                modes.add(mode)
                reduces.add(
                    "min" if mode == "ncu" else REDUCE.get(probe) if mode in ("graph_replay", "eager") else None
                )
            probes.add(probe)
            gpus.add(_gpu(args))
            days.append(day or "")
    return {
        "schema": 1,
        "table": rel.as_posix(),
        "dir": rel.parts[0],
        "gpu": gpus.pop() if len(gpus) == 1 else None,
        "mode": modes.pop() if len(modes) == 1 else "unknown",
        "tool": " + ".join(TOOLS.get(p, f"quettabench.probes.{p}") for p in sorted(probes)),
        "reduce": reduces.pop() if len(reduces) == 1 else None,
        "measured": max(days, default="") or None,
        "notes": "" if len(modes) <= 1 else f"rows timed as {', '.join(sorted(modes))}",
        "provenance": "qbench export kernel_tables; source runs in kernel_data/qbench_manifest.json",
    }


def _keep_sidecar(meta: dict, table: Path, kept: bool) -> dict:
    """meta merged over the sidecar dest has beside table: a null field in meta keeps the old value and notes join.

    When its rows were kept, measured spans both, tool names both, and a different mode makes mode unknown.
    """
    path = table.with_name(table.name + ".meta.json")
    try:
        prior = json.loads(path.read_text()) if path.is_file() else {}
    except ValueError:
        prior = None
    if not isinstance(prior, dict):
        raise SystemExit(f"{path}: malformed JSON (not an object)")
    notes = "; ".join(n for n in dict.fromkeys((prior.get("notes"), meta["notes"])) if n)
    meta = {k: v for k, v in meta.items() if v is not None or prior.get(k) is None}
    if not kept:
        return {**prior, **meta, "notes": notes}
    days = sorted(d for d in (*str(prior.get("measured") or "").split(" .. "), meta.get("measured")) if d)
    tools = [t for t in (*str(prior.get("tool") or "").split(" + "), *meta["tool"].split(" + ")) if t]
    mode = meta["mode"] if prior.get("mode") in (None, meta["mode"]) else "unknown"
    return {
        **prior,
        **meta,
        "mode": mode,
        "notes": notes,
        "tool": " + ".join(dict.fromkeys(tools)),
        "measured": " .. ".join(dict.fromkeys(days[:1] + days[-1:])) or None,
    }


@dataclasses.dataclass
class Rows:
    """One run's rows of one table from its rows.jsonl: each row's key and value cells, the probe and how the rows were timed."""

    probe: str
    timings: set[str]
    cells: list[dict]


Source = tuple[datetime, "Path | Rows", Path, bool]  # (finished, dest table or rows, run dir, --append)


class Lost(SystemExit):
    """The export would drop a dest row or column; replace (--replace-tables) replaces just the tables that stop with it."""


def _rows(run: Path) -> list[dict]:
    """run's rows.jsonl, one ProbeRow dict per line; stops the export on a line that is not one."""
    rows = []
    for n, line in enumerate((run / "rows.jsonl").read_text().splitlines(), 1):
        try:
            row = json.loads(line)
            if row["timing"] not in DIRS or not all(isinstance(row[f], dict) for f in ("key", "value")):
                raise ValueError
            rows.append(row)
        except (ValueError, KeyError, TypeError):
            raise SystemExit(f"{run}/rows.jsonl line {n} malformed") from None
    return rows


def found(results_dirs: Iterable[Path]) -> dict[Path, dict[str, list[Source]]]:
    """{<dir>/<kind>/<file>: {probe: its ok runs' rows, in finished order}}."""
    tables: dict[Path, dict[str, list[Source]]] = {}
    for root in map(Path, results_dirs):
        for path in files(root, "rows.jsonl"):
            run = path.parent
            if not (run / "run.json").is_file() or _record(run)[0] is None:
                continue
            args = _record(run)[1]
            mine: dict[Path, Rows] = {}
            for row in _rows(run):
                rows = mine.setdefault(Path(DIRS[row["timing"]], row["table"]), Rows(row["probe"], set(), []))
                rows.timings.add(row["timing"])
                rows.cells.append({**row["key"], **row["value"]})
            for rel, rows in mine.items():
                tables.setdefault(rel, {}).setdefault(rows.probe, []).append(
                    (finished(run), rows, run, "--append" in args)
                )
    for by_probe in tables.values():
        for sources in by_probe.values():
            sources.sort(key=lambda s: (s[0], str(s[2])))
    return tables


def _runs_used(probe: str, rel: Path, sources: list[Source]) -> list[Source]:
    """The runs a probe's table is built from: every run for a keyed table (from the last one without --append, for gemm and moe), else the last to finish."""
    kind = rel.parts[-2]
    if (probe, kind) in JSON_UNION and rel.suffix == ".json":
        return sources
    if (probe, kind) not in ROW_KEYS or rel.suffix != ".csv":
        return sources[-1:]
    if probe in RESET_WITHOUT_APPEND:
        resets = [i for i, s in enumerate(sources) if not s[3]]
        return sources[resets[-1] :] if resets else sources
    return sources


def _key(row: dict, key: tuple[str, ...]) -> tuple:
    """row's key, numbers parsed with int(); an empty op is all_reduce, as QuettaSim's collectives loader reads it."""
    return tuple(
        (row.get("op") or "all_reduce")
        if c == "op"
        else row[c]
        if c in ("gpu", "layout")
        else float(row[c])
        if c == "total_weight_gb"
        else int(row[c])
        for c in key
    )


def _newline(table: Path | Rows) -> str:
    """The line ending of table's first line; a run's rows get csv's own."""
    if isinstance(table, Rows):
        return "\r\n"
    with table.open("rb") as f:
        return "\r\n" if f.readline().endswith(b"\r\n") else "\n"


def _write_csv(out: Path, fields: list[str], rows: Iterable[dict], like: Path | Rows) -> None:
    """rows under fields, with like's line ending."""
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="", lineterminator=_newline(like))
        w.writeheader()
        w.writerows(rows)


def _read_csv(rel: Path, table: Path | Rows, run: Path, key: tuple[str, ...] = ()) -> tuple[list[str], list[dict]]:
    """table's header and rows as csv reads them; stops the export on a row with a missing or extra cell or an empty or unparsable key."""
    if isinstance(table, Rows):
        fields = list(dict.fromkeys(c for cells in table.cells for c in cells))
        rows = [{c: "" if cells.get(c) is None else str(cells[c]) for c in fields} for cells in table.cells]
    else:
        with table.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fields = list(reader.fieldnames or [])
    if not fields:
        raise SystemExit(f"{rel.as_posix()}: {run} has no header (empty CSV)")
    for n, row in enumerate(rows, 1):
        try:
            if None in row or None in row.values() or not all(row.get(c) for c in key if c != "op"):
                raise ValueError
            _key(row, key)
        except ValueError:
            raise SystemExit(f"{rel.as_posix()}: {run} row {n} malformed") from None
    return fields, rows


def _read_json(rel: Path, table: Path | Rows, run: Path) -> object:
    if isinstance(table, Rows):
        return {cells["entry"]: cells["value"] for cells in table.cells}
    try:
        return json.loads(table.read_text())
    except ValueError:
        raise SystemExit(f"{rel.as_posix()}: {run} malformed JSON") from None


def _union_csv(
    rel: Path, sources: list[Source], key: tuple[str, ...], out: Path, old: Source | None = None
) -> list[Path]:
    """Rows of the dest table old, then of every source oldest first, each source replacing earlier rows on the keys it holds. Returns the run dirs with a row kept.

    Within a run the last row on a key wins; old keeps repeated keys (QuettaSim's loaders reduce them) unless its table is in SORTED.
    """
    order = next((cols for k, cols in SORTED.items() if ROW_KEYS[k] == key), None)
    fields: list[str] = []
    kept: list[tuple[dict, Path]] = []
    olds: list[dict] = []
    newer: set[tuple] = set()
    for source in [old, *sources] if old else sources:
        _, table, run, _ = source
        head, rows = _read_csv(rel, table, run, key)
        if key != GEOMETRY and (source is not old or order):
            rows = list({_key(r, key): r for r in rows}.values())
        fields += [c for c in head if c not in fields]
        mine = {_key(r, key) for r in rows}
        if source is old:
            olds = rows
        else:
            newer |= mine
        kept = [(r, d) for r, d in kept if _key(r, key) not in mine]
        kept += [(r, run) for r in rows]
    if order:
        kept.sort(key=lambda rd: _key(rd[0], order))
    _write_csv(out, fields, (r for r, _ in kept), (old or sources[0])[1])
    if old:
        _lost(rel, old, out, key, newer, olds)
    return list(dict.fromkeys(d for _, d in kept))


def _union_json(rel: Path, sources: list[Source], out: Path) -> list[Path]:
    """Entries of every source in finished order, the later run winning a key."""
    merged: dict[str, tuple[object, Path]] = {}
    sources = sorted(sources, key=lambda s: (s[0], str(s[2])))
    for _, table, run, _ in sources:
        entries = _read_json(rel, table, run)
        if not isinstance(entries, dict):
            raise SystemExit(f"{rel.as_posix()}: {run} malformed JSON (not an object)")
        merged.update({k: (v, run) for k, v in entries.items()})
    indent, end = _json_style(sources[0][1])
    out.write_text(json.dumps({k: v for k, (v, _) in merged.items()}, indent=indent) + end)
    return list(dict.fromkeys(run for _, run in merged.values()))


def _json_style(table: Path | Rows) -> tuple[int | None, str]:
    """(indent, ending) of a dest JSON table, or of the probe's own JSON."""
    if isinstance(table, Rows):
        return JSON_STYLE[table.probe]
    text = table.read_text()
    lines = text.splitlines()
    indent = len(lines[1]) - len(lines[1].lstrip(" ")) if len(lines) > 1 else None
    return indent, "\n" * text.endswith("\n")


def _merge_collectives(
    rel: Path, collective: Source | None, vllm: Source | None, out: Path, old: Source | None = None
) -> list[Path]:
    """Collectives table: old's rows, then the collective run's replacing on COLLECTIVE_KEY, then vllm_allreduce's all_reduce rows replacing every earlier all_reduce row of their world.

    Beside a vllm_allreduce run the collective run's all_reduce rows are left out. Collective rows fill an empty dtype_bytes (2) and path (nccl).
    """
    sources = [(s, _read_csv(rel, s[1], s[2], COLLECTIVE_KEY)) for s in (old, collective, vllm) if s]
    fields: list[str] = []
    for _, (head, _) in sources:
        fields += [c for c in head if c not in fields]
    fill = {c: v for c, v in (("dtype_bytes", "2"), ("path", "nccl")) if c in fields}
    kept: list[tuple[dict, Path]] = []
    newer: set[tuple] = set()
    for source, (_, rows) in sources:
        if source is collective:
            rows = [
                {**r, **{c: r.get(c) or v for c, v in fill.items()}}
                for r in rows
                if not (vllm and _key(r, ("op",)) == ("all_reduce",))
            ]
        if source is vllm:
            rows = [r for r in rows if _key(r, ("op",)) == ("all_reduce",)]
            worlds = {_key(r, ("world",)) for r in rows}
            mine = {
                _key(r, COLLECTIVE_KEY)
                for r, _ in kept
                if _key(r, ("op",)) == ("all_reduce",) and _key(r, ("world",)) in worlds
            }
        else:
            mine = set()
        mine |= {_key(r, COLLECTIVE_KEY) for r in rows}
        if source is not old:
            newer |= mine
        kept = [(r, d) for r, d in kept if _key(r, COLLECTIVE_KEY) not in mine]
        kept += [(r, source[2]) for r in rows]
    _write_csv(out, fields, (r for r, _ in kept), sources[0][0][1])
    if old:
        _lost(rel, old, out, COLLECTIVE_KEY, newer)
    return list(dict.fromkeys(d for _, d in kept))


def _check(rel: Path, by_probe: dict[str, list[Source]]) -> None:
    """Stop on a file two probes write that kernel_tables does not merge."""
    probes = sorted(by_probe)
    if len(probes) > 1 and (probes, rel.suffix) not in (
        (["collective", "vllm_allreduce"], ".csv"),
        (["elementwise", "gemm"], ".json"),
    ):
        raise SystemExit(
            f"{rel.as_posix()}: written by probes {probes}; kernel_tables "
            "merges only collective+vllm_allreduce and elementwise+gemm"
        )


def _lost(
    rel: Path, old: Source, out: Path, key: tuple[str, ...], newer: set[tuple], rows: list[dict] | None = None
) -> None:
    """Stop (Lost) when out lacks a column of old or drops one of its rows, cell for cell, unless a newer run re-measured that key."""
    head, olds = _read_csv(rel, old[1], old[1], key)
    fields, kept = _read_csv(rel, out, out, key)
    stop = "; give --replace-tables to replace each table that stops this way"
    gone = [c for c in head if c not in fields]
    if gone:
        raise Lost(f"{rel.as_posix()}: the export would drop column {', '.join(gone)} of {old[1]}{stop}")
    cols = sorted({*head, *fields})
    have = Counter(tuple(r.get(c) or "" for c in cols) for r in kept)
    rows = olds if rows is None else rows
    left = 0
    for row in rows:
        cells = tuple(row.get(c) or "" for c in cols)
        if key and _key(row, key) in newer:
            left += 1
        elif have[cells]:
            have[cells] -= 1
            left += 1
    if left < len(rows):
        raise Lost(f"{rel.as_posix()}: the export would keep {left} of its {len(rows)} rows in {old[1]}{stop}")


def _write(rel: Path, by_probe: dict[str, list[Source]], out: Path, old: Source | None = None) -> list[Path]:
    """Build one dest file over the dest table old, if any; returns the run dirs (and old's path) whose rows it holds."""
    used = {probe: _runs_used(probe, rel, sources) for probe, sources in by_probe.items()}
    probes = sorted(used)
    olds = [old] if old else []
    tp_table = rel.parts[-2] == "collectives" and not rel.stem.endswith("_nccl_vs_vllm") and rel.suffix == ".csv"
    if tp_table and set(probes) <= {"collective", "vllm_allreduce"}:
        return _merge_collectives(
            rel, *(used[p][-1] if p in used else None for p in ("collective", "vllm_allreduce")), out, old
        )
    if probes == ["elementwise", "gemm"]:
        return _union_json(rel, olds + used["elementwise"] + used["gemm"], out)
    (probe,) = probes
    sources = used[probe]
    key = ROW_KEYS.get((probe, rel.parts[-2]), ())
    if old and rel.suffix == ".json":
        return _union_json(rel, olds + sources, out)
    if old and rel.suffix == ".csv":
        reset = probe in RESET_WITHOUT_APPEND and not all(s[3] for s in sources)
        if key and not reset:
            return _union_csv(rel, sources, key, out, old)
        runs = _write(rel, by_probe, out)
        newer = {_key(r, key) for s in sources if key for r in _read_csv(rel, s[1], s[2], key)[1]}
        _lost(rel, old, out, key, newer)
        return runs
    if len(sources) == 1 and rel.suffix == ".csv":
        _, table, run, _ = sources[0]
        _write_csv(out, *_read_csv(rel, table, run, key), table)
        return [run]
    if rel.suffix == ".json":
        return _union_json(rel, sources, out)
    return _union_csv(rel, sources, key, out)


def run_kernel_tables(
    results_dirs: Iterable[Path], dest: Path, replace: bool = False
) -> tuple[list[Path], dict[str, list[str]]]:
    """Build every table and sidecar in a temp dir under dest, then replace the dest files, so a stop leaves dest as it was. With replace, tables that would stop with Lost are rebuilt without dest's rows.

    Returns the files written and {<dir>/<kind>/<file>: run dirs with rows in it, in finished order} for the manifest.
    """
    tables = sorted(found(results_dirs).items())
    for rel, by_probe in tables:
        _check(rel, by_probe)
    if not tables:
        return [], {}
    used = {}
    dest = Path(dest)
    made = next((d for d in reversed((dest, *dest.parents)) if not d.exists()), None)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=dest, prefix=".kernel_tables-") as tmp:
            for rel, by_probe in tables:
                out = Path(tmp) / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                table = dest / "kernel_data" / rel
                old = (OLDEST, table, table, True) if table.is_file() else None
                try:
                    runs = set(_write(rel, by_probe, out, old))
                except Lost:
                    if not replace:
                        raise
                    old = None
                    runs = set(_write(rel, by_probe, out))
                meta = _sidecar(rel, by_probe, runs)
                if old:
                    meta = _keep_sidecar(meta, table, table in runs)
                meta = json.dumps(meta, indent=1, sort_keys=True)
                out.with_name(out.name + ".meta.json").write_text(meta + "\n")
                order = sorted((s[0], str(s[2])) for sources in by_probe.values() for s in sources if s[2] in runs)
                used[rel.as_posix()] = (["adopted from dest"] if table in runs else []) + list(
                    dict.fromkeys(run for _, run in order)
                )
            written = []
            for rel, by_probe in tables:
                out = dest / "kernel_data" / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                for name in (rel.name, rel.name + ".meta.json"):
                    os.replace(Path(tmp) / rel.parent / name, out.with_name(name))
                    written.append(out.with_name(name))
                print(f"{rel.as_posix()} <- {', '.join(sorted(by_probe))}")
    except BaseException:
        if made is not None:
            shutil.rmtree(made, ignore_errors=True)
        raise
    return written, used


NCCL_PHYSICS_BAND_MS_PER_1K = (1.0, 3.0)  # external-lit all-reduce band, hidden=4096 bf16


def _span_new_rate(path: Path) -> dict:
    """OLS prefill_span_ms ~ floor + a·new + b·cached over the stage-split rows."""
    rows = list(csv.DictReader(path.open()))
    X = [(1.0, float(r["new"]), float(r["cached"])) for r in rows]
    y = [float(r["prefill_span_ms"]) for r in rows]
    n = len(X)
    xtx = [[sum(X[k][i] * X[k][j] for k in range(n)) for j in range(3)] for i in range(3)]
    xty = [sum(X[k][i] * y[k] for k in range(n)) for i in range(3)]
    m = [xtx[i] + [xty[i]] for i in range(3)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        for r in range(3):
            if r != col and m[col][col]:
                f = m[r][col] / m[col][col]
                m[r] = [a - f * b for a, b in zip(m[r], m[col])]
    beta = [m[i][3] / m[i][i] for i in range(3)]
    return {"floor_ms": beta[0], "new_ms_per_tok": beta[1], "cached_ms_per_tok": beta[2], "n_rows": n}


def build_tpn(n: int, tpn_csv: Path, out_json: Path, tp1_csv: Path, gpu: str = "H100", host: str = "h100") -> None:
    """Like-for-like stage-split estimate at tp n: comm_total = prefill_span.new(tpN) - span.new(tp1)/N, the per-new-token cost above the 1/N GEMM share, pinned per config as prefill_tp_comm_ms_per_token. Each GPU uses its own tp1 leg."""
    for p in (tp1_csv, tpn_csv):
        if not p.exists():
            raise SystemExit(f"missing {p} -- pull both stage-split legs first")
    tp1 = _span_new_rate(tp1_csv)
    tpn = _span_new_rate(tpn_csv)
    comm = tpn["new_ms_per_tok"] - tp1["new_ms_per_tok"] / float(n)
    payload = {
        "schema": "prefill_tp_comm.v1",
        "gpu": f"{gpu}x{n}",
        "model": "Llama-3.1-8B",
        "tp": n,
        "method": (
            "like-for-like multiprocess api_server stage-split pair (same script both "
            f"legs); comm_total = prefill_span.new(tp{n}) − prefill_span.new(tp1)/{n}"
        ),
        "tp1_fit": {k: round(v, 6) if isinstance(v, float) else v for k, v in tp1.items()},
        f"tp{n}_fit": {k: round(v, 6) if isinstance(v, float) else v for k, v in tpn.items()},
        "constants": {"prefill_tp_comm_ms_per_token_total": comm},
        "fallback_it_replaces_ms_per_token": None,
        "_notes": (
            f"TOTAL tp{n} comm per new token (NOT per-extra-rank). Pinned in the "
            f"deployment JSON key prefill_tp_comm_ms_per_token of the {gpu}x{n} config; "
            "every config without the pin keeps PREFILL_TP_COMM_MS_PER_TOKEN*(tp-1). "
            "Regenerate: just export. Sources: "
            f"serving_stage_split.py --tensor-parallel-size {{1,{n}}} on {host}."
        ),
    }
    try:  # informational; QuettaSim may not have the constant
        import simulator.ttft_queue_sim as _sim

        payload["fallback_it_replaces_ms_per_token"] = _sim.PREFILL_TP_COMM_MS_PER_TOKEN * (n - 1)
    except ImportError:
        pass
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    fallback = payload["fallback_it_replaces_ms_per_token"]
    print(
        f"tp1 span.new {tp1['new_ms_per_tok'] * 1e3:.3f} ms/1k | tp{n} "
        f"{tpn['new_ms_per_tok'] * 1e3:.3f} -> comm_total {comm * 1e3:.4f} ms/1k "
        f"(fallback it replaces: {'n/a' if fallback is None else f'{fallback * 1e3:.4f}'} ms/1k)"
    )
    print(f"wrote {out_json}")


def build_tp2(tp1_csv: Path, tp2_csv: Path, out_json: Path) -> None:
    """H100 tp1 vs tp2: comm = span.new(tp2) - span.new(tp1)/2."""
    tp1 = _span_new_rate(tp1_csv)
    tp2 = _span_new_rate(tp2_csv)
    comm = tp2["new_ms_per_tok"] - tp1["new_ms_per_tok"] / 2.0
    in_band = NCCL_PHYSICS_BAND_MS_PER_1K[0] <= comm * 1e3 <= NCCL_PHYSICS_BAND_MS_PER_1K[1] + 1.0
    payload = {
        "schema": "prefill_tp_comm.v1",
        "gpu": "H100",
        "model": "Llama-3.1-8B",
        "method": (
            "like-for-like multiprocess api_server stage-split pair (same script both "
            "legs); comm = prefill_span.new(tp2) − prefill_span.new(tp1)/2"
        ),
        "tp1_fit": {k: round(v, 6) if isinstance(v, float) else v for k, v in tp1.items()},
        "tp2_fit": {k: round(v, 6) if isinstance(v, float) else v for k, v in tp2.items()},
        "constants": {"PREFILL_TP_COMM_MS_PER_TOKEN": comm},
        "nccl_physics_band_ms_per_1k": list(NCCL_PHYSICS_BAND_MS_PER_1K),
        "within_physics_band_plus_1": in_band,
        "retired_backed_out_remainder": 0.00585,
        "_notes": (
            "Replaces the instrumentation-inconsistent backed-out remainder 5.85 ms/1k "
            "(audit-v2 G3): the like-for-like pair removes the multiprocess-vs-in-process "
            "mismatch, and the residual lands at the top of the NCCL all-reduce physics "
            "band. Regenerate: just export. Sources: "
            "serving_stage_split.py --tensor-parallel-size {1,2} on h100."
        ),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"tp1 span.new {tp1['new_ms_per_tok'] * 1e3:.3f} ms/1k | tp2 {tp2['new_ms_per_tok'] * 1e3:.3f} "
        f"-> comm {comm * 1e3:.4f} ms/1k (retired remainder 5.85; physics band "
        f"{NCCL_PHYSICS_BAND_MS_PER_1K})"
    )
    print(f"wrote {out_json}")


def run_tp_comm(results_dirs: Iterable[Path], dest: Path) -> list[Path]:
    """Write dest/kernels/prefill_tp_comm_<gpu>[x<tp>].json for every gpu/tp whose legs exist."""
    written = []
    for gpu, host in (("H100", "h100"), ("RTX3090", "3090")):
        for tp, tp1_csv, tpn_csv in stage_split_legs(results_dirs, gpu):
            if gpu == "H100" and tp == 2:
                out_json = Path(dest) / "kernels" / "prefill_tp_comm_H100.json"
                build_tp2(tp1_csv, tpn_csv, out_json)
            else:
                out_json = Path(dest) / "kernels" / f"prefill_tp_comm_{gpu}x{tp}.json"
                build_tpn(tp, tpn_csv, out_json, tp1_csv, gpu=gpu, host=host)
            written.append(out_json)
    return written


TRACES = [
    "vllm_engine_step_trace_swe_c40_t12_benchmark_serving_wall.jsonl",
    "vllm_engine_step_trace_swe_c80_t12_benchmark_serving_wall.jsonl",
    "vllm_engine_step_trace_swe_c320_t2_benchmark_serving_wall.jsonl",
    "vllm_engine_step_trace_terminal_c80_t16_benchmark_serving_wall.jsonl",
]

# Fixed before measuring; SCHED_PRIOR only gates eligibility and never enters the output.
SCHED_PRIOR_MS = 5.7
FULL_BATCH_FRAC = 0.9
WARMUP_STEP_ID = 5
BW_DOMINANCE = 2.0  # bw_roofline_ms >= BW_DOMINANCE * SCHED_PRIOR_MS
WALL_TRIM = (0.5, 2.0)  # vs per-trace median of the candidate set
PREFILL_MIN_TOKENS = 1024
UTIL_FLOPS_MIN_N = 5


def _f(v) -> float:
    return float(v) if v not in ("", None) else 0.0


def _i(v) -> int:
    return int(float(v)) if v not in ("", None) else 0


def load_trace(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def reconstruct_contexts(steps: list[dict]) -> list[dict | None]:
    """Per decode-only step, {"ctx_total", "batch"}: context tokens across the decode batch, or None for a mixed step or one with an unknown prompt.

    Context is the first-seen prompt_tokens plus the steps the request has already decoded.
    """
    prompt_tokens: dict[str, int] = {}
    generated: dict[str, int] = {}
    out: list[dict | None] = []
    for s in steps:
        ect = s.get("engine_cache_truth")
        if ect:
            try:
                truth = json.loads(ect)
            except (TypeError, ValueError):
                truth = {}
            for req in truth.get("requests", []):
                rid, pt = req.get("request_id"), req.get("prompt_tokens")
                if rid and pt and rid not in prompt_tokens:
                    prompt_tokens[rid] = int(pt)
        decode_ids = [r for r in (s.get("decode_request_ids") or "").split() if r]
        is_decode_only = (
            _i(s.get("decode_batch")) > 0
            and _i(s.get("prefill_tokens")) == 0
            and str(s.get("model_executed")) == "true"
        )
        if is_decode_only:
            if any(r not in prompt_tokens for r in decode_ids):
                out.append(None)
            else:
                ctx = sum(prompt_tokens[r] + generated.get(r, 0) for r in decode_ids)
                out.append({"ctx_total": ctx, "batch": len(decode_ids)})
        else:
            out.append(None)
        for r in decode_ids:
            generated[r] = generated.get(r, 0) + 1
    return out


def quartile_medians(pairs: list[tuple[float, float]]) -> list[dict]:
    """pairs = (bw_roofline_ms, ratio); medians per byte-scale quartile."""
    if not pairs:
        return []
    pairs = sorted(pairs)
    out = []
    n = len(pairs)
    for q in range(4):
        chunk = pairs[q * n // 4 : (q + 1) * n // 4] or pairs[-1:]
        out.append(
            {
                "bw_roofline_ms_range": [round(chunk[0][0], 2), round(chunk[-1][0], 2)],
                "n": len(chunk),
                "util_bw_median": round(statistics.median(r for _, r in chunk), 4),
            }
        )
    return out


def build_roofline_utils(gpu: str, params_path: Path, trace_paths: dict[str, Path], out_path: Path) -> None:
    params = json.loads(params_path.read_text())
    n_params = float(params["n_params"])
    bpp = float(params["bytes_per_param"])
    kv_bpt = float(params["kv_bytes_per_token"])
    peak_flops = float(params["peak_flops_per_s"])
    peak_bw = float(params["peak_bw_bytes_per_s"])

    per_trace: dict[str, dict] = {}
    bw_pool: list[tuple[float, float]] = []  # (bw_roofline_ms, ratio_wall)
    bw_pool_net: list[float] = []  # ratio vs (wall - SCHED_PRIOR)
    flops_pool: list[float] = []
    flops_pool_engine: list[float] = []
    sched_pool: list[float] = []

    for name in TRACES:
        path = trace_paths[name]
        steps = load_trace(path)
        ctxs = reconstruct_contexts(steps)
        key = name.split("_benchmark")[0].replace("vllm_engine_step_trace_", "")
        tr: dict = {"steps": len(steps)}

        max_batch = max((c["batch"] for c in ctxs if c), default=0)
        candidates = []
        excluded_unknown_ctx = 0
        for s, c in zip(steps, ctxs):
            if (
                _i(s.get("decode_batch")) > 0
                and _i(s.get("prefill_tokens")) == 0
                and str(s.get("model_executed")) == "true"
                and c is None
            ):
                excluded_unknown_ctx += 1
            if c is None:
                continue
            if c["batch"] < FULL_BATCH_FRAC * max_batch:
                continue
            if _i(s.get("step_id")) <= WARMUP_STEP_ID:
                continue
            bw_ms = (bpp * n_params + c["ctx_total"] * kv_bpt + c["batch"] * kv_bpt) / peak_bw * 1e3
            if bw_ms < BW_DOMINANCE * SCHED_PRIOR_MS:
                continue
            wall = _f(s.get("engine_step_wall_ms"))
            if wall <= 0:
                continue
            candidates.append((bw_ms, wall))
        trimmed = []
        if candidates:
            med_wall = statistics.median(w for _, w in candidates)
            lo, hi = WALL_TRIM[0] * med_wall, WALL_TRIM[1] * med_wall
            for bw_ms, wall in candidates:
                if lo <= wall <= hi:
                    trimmed.append((bw_ms, wall))
        ratios = [(bw, bw / w) for bw, w in trimmed]
        bw_pool.extend(ratios)
        bw_pool_net.extend(bw / max(w - SCHED_PRIOR_MS, 1e-9) for bw, w in trimmed)
        tr["util_bw"] = {
            "n_candidates": len(candidates),
            "n_after_trim": len(trimmed),
            "n_outliers_trimmed": len(candidates) - len(trimmed),
            "n_decode_steps_unknown_ctx": excluded_unknown_ctx,
            "max_decode_batch": max_batch,
            "median": round(statistics.median(r for _, r in ratios), 4) if ratios else None,
        }

        fl_cand = []
        for s in steps:
            if not (
                _i(s.get("prefill_tokens")) >= PREFILL_MIN_TOKENS
                and _i(s.get("decode_batch")) == 0
                and str(s.get("model_executed")) == "true"
            ):
                continue
            m = _i(s.get("prefill_tokens"))
            submit = _f(s.get("model_submit_wall_ms"))
            engine = _f(s.get("engine_step_wall_ms"))
            if submit <= 0:
                continue
            fl_cand.append((m, submit, engine))
        fl_trim = []
        if fl_cand:
            med_pt = statistics.median(sub / m for m, sub, _ in fl_cand)
            lo, hi = WALL_TRIM[0] * med_pt, WALL_TRIM[1] * med_pt
            fl_trim = [(m, sub, eng) for m, sub, eng in fl_cand if lo <= sub / m <= hi]
        fl_ratios = [2.0 * n_params * m / peak_flops * 1e3 / sub for m, sub, _ in fl_trim]
        flops_pool.extend(fl_ratios)
        flops_pool_engine.extend(2.0 * n_params * m / peak_flops * 1e3 / eng for m, _, eng in fl_trim if eng > 0)
        tr["util_flops"] = {
            "n_candidates": len(fl_cand),
            "n_after_trim": len(fl_trim),
            "median": round(statistics.median(fl_ratios), 4) if fl_ratios else None,
        }

        gaps = []
        for s in steps:
            if (
                _i(s.get("decode_batch")) == 1
                and _i(s.get("prefill_tokens")) == 0
                and str(s.get("model_executed")) == "true"
            ):
                gaps.append(_f(s.get("engine_step_wall_ms")) - _f(s.get("model_submit_wall_ms")))
        sched_pool.extend(gaps)
        tr["sched"] = {
            "n": len(gaps),
            "median": round(statistics.median(gaps), 4) if gaps else None,
        }
        per_trace[key] = tr

    util_bw = statistics.median(r for _, r in bw_pool) if bw_pool else None
    util_flops = statistics.median(flops_pool) if len(flops_pool) >= UTIL_FLOPS_MIN_N else None
    sched = statistics.median(sched_pool) if sched_pool else None

    out = {
        "gpu": gpu,
        "recipe": "profiling/docs/defit_log_entries/L6-utils.md (pre-registered 2026-06-10; "
        "implemented by quettabench/export/quettasim/roofline_utils.py)",
        "params_source": params_path.name,
        "util_bw": round(util_bw, 4) if util_bw is not None else None,
        "util_flops": round(util_flops, 4) if util_flops is not None else None,
        "scheduler_overhead_ms_per_step": round(sched, 4) if sched is not None else None,
        "n_steps": {
            "util_bw": len(bw_pool),
            "util_flops": len(flops_pool),
            "sched": len(sched_pool),
        },
        "diagnostics": {
            "util_bw_net_of_sched_prior": round(statistics.median(bw_pool_net), 4) if bw_pool_net else None,
            "util_bw_net_note": "same steps, ratio vs (engine_wall - 5.7ms): the "
            "ttft_queue_sim.py:762 convention (sched added separately). Headline uses the "
            "full engine wall: closed_form_tpot adds no sched term to tpot.",
            "util_flops_vs_engine_wall": round(statistics.median(flops_pool_engine), 4) if flops_pool_engine else None,
            "util_flops_thinness": "pure-prefill steps are rare in decode-heavy serving "
            "traces; prefill_gemm_util_H100.json (R1 measured curve) remains the better "
            "prefill-util source.",
            "util_bw_by_byte_quartile": quartile_medians(bw_pool),
        },
        "per_trace": per_trace,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {out_path}")
    print(
        json.dumps(
            {k: out[k] for k in ("util_bw", "util_flops", "scheduler_overhead_ms_per_step", "n_steps")}, indent=2
        )
    )


def run_roofline_utils(results_dirs: Iterable[Path], dest: Path) -> list[Path]:
    """Write dest/kernels/roofline_utils_<gpu>.json for every gpu with its params and all traces."""
    written = []
    for gpu in ("H100", "A100"):
        params_path = find(results_dirs, f"roofline_params_{gpu}_llama31_8b.json")
        trace_paths = {name: find(results_dirs, name) for name in TRACES}
        if params_path is None or None in trace_paths.values():
            print(f"SKIP roofline_utils {gpu}: params or pinned traces not in the results dirs")
            continue
        out_path = Path(dest) / "kernels" / f"roofline_utils_{gpu}.json"
        build_roofline_utils(gpu, params_path, trace_paths, out_path)
        written.append(out_path)
    return written


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


_LAG_UPGRADE_SPREAD = 0.005  # cross-pass reproducibility bound
_LAG_UPGRADE_HEADROOM = 1.5  # adopted pass's p99 loop lag <= 1.5x the flag limit
LAG_UPGRADE_FIELD = "lag_reproducibility_upgrade"


def build_decode_grid(inputs: list[Path], snap_tolerance: float = 0.02, lag_upgrade: bool = False) -> list[dict]:
    """Merge runs (latest wins per cell) and summarize.

    snap_tolerance: measured effective contexts land within ~1.5% of nominal T but are unique floats per cell, which would make every load_grid t-axis column ragged. Within the tolerance context_len snaps to T and the measured value goes to effective_context_len; beyond it the cell is flagged check.

    lag_upgrade (default off): load_grid drops non-ok rows, so a marginal loop-lag flag (2.01 vs a 2.0 ms limit) replaces a reproducible measured cell with the analytic fill. A lag-only flagged cell becomes ok when all hold:
      * measured in >= 2 input passes (independent client topologies);
      * every pass is otherwise hygiene-clean (no empty-stream drops, enough in-window samples, positive steady window);
      * the latest row's flag is lag-only and its snap check passes;
      * cross-pass decode_step_ms spread <= 0.5% of the adopted value;
      * the adopted pass's p99 loop lag <= 1.5x the flag limit.
    Upgraded rows carry the evidence in lag_reproducibility_upgrade.
    """
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
            print(
                f"cell {key}: effective context {eff} deviates >{snap_tolerance:.0%} "
                f"from nominal {nominal} -> kept + check"
            )
        if (
            lag_upgrade
            and row["validation_status"] == "check"
            and not dropped
            and snap_ok
            and row["n_samples"] >= min_samples
            and row["steady_window_s"] > 0
            and len(history[key]) >= 2
        ):
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
                if spread <= _LAG_UPGRADE_SPREAD and per_pass[-1][1] <= _LAG_UPGRADE_HEADROOM * lag_limit_ms:
                    row["validation_status"] = "ok"
                    row[LAG_UPGRADE_FIELD] = (
                        f"passes={len(per_pass)} spread={spread * 100:.3f}% "
                        f"lags={'/'.join(f'{lag:.3f}' for _, lag, _ in per_pass)}"
                    )
                    print(f"cell {key}: lag-only flag upgraded to ok ({row[LAG_UPGRADE_FIELD]})")
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


def run_serving_decode_grid(results_dirs: Iterable[Path], dest: Path, lag_upgrade: bool = False) -> list[Path]:
    """Write dest/results/serving_decode_grid_<tag>_<date>.csv, one per tag found."""
    runs: dict[str, list[tuple[str, Path]]] = {}
    for root in results_dirs:
        for path in files(root, "serving_decode_grid_*.jsonl*"):
            m = RUN_NAME.match(path.name)
            if m:
                runs.setdefault(m["tag"], []).append((m["date"], path))
    written = []
    for tag, dated in runs.items():
        dated.sort()
        rows = build_decode_grid([p for _, p in dated], lag_upgrade=lag_upgrade)
        out = Path(dest) / "results" / f"serving_decode_grid_{tag}_{dated[-1][0]}.csv"
        write_grid(rows, out, lag_upgrade)
        written.append(out)
    return written


BUILDERS = {
    "tp_comm": run_tp_comm,
    "roofline_utils": run_roofline_utils,
    "serving_decode_grid": run_serving_decode_grid,
    "kernel_tables": run_kernel_tables,
}


def source_runs(results_dirs: list[Path]) -> list[dict]:
    """Every exported run.json under the results dirs (run dir, name, status, start), so a table names its runs; skipped runs are printed instead."""
    runs = []
    for root in results_dirs:
        skipped = skipped_runs(root)
        for run_dir, why in skipped.items():
            print(f"skip {run_dir}: {why}")
        for path in files(root, "run.json"):
            doc = json.loads(path.read_text())
            runs.append(
                {
                    "run": str(path.parent),
                    "run_name": doc["config"].get("run_name"),
                    "status": doc["run_status"]["status"],
                    "started": doc["provenance"].get("started"),
                }
            )
    return runs


def manifest(results_dirs: list[Path]) -> dict:
    return {
        "source_dirs": [str(d) for d in results_dirs],
        "runs": source_runs(results_dirs),
        "bench_sha": bench_sha(),
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def export(
    results_dirs: Iterable[Path],
    dest: Path | None = None,
    only: Iterable[str] | None = None,
    kernel_dest: Path | None = None,
    replace_tables: bool = False,
) -> list[Path]:
    """Run the builders (all, or the named ones) over the results dirs; return the files written.

    kernel_tables writes under kernel_dest (QuettaSim's data root) and merges into its tables unless replace_tables; the other builders write under dest. Each runs only when its dest is given, and naming one without its dest stops the export.
    dest gets manifest.json, and kernel_dest gets kernel_data/qbench_manifest.json when kernel_tables found a table: inputs, runs, bench commit and time, and per kernel table the run dirs behind it. Tables this export did not write keep their entries; a table that kept dest rows names the old manifest's runs first, else "adopted from dest". A kernel manifest that does not parse, or lacks a kernel_tables object, stops the export with kernel_dest as it was.
    """
    results_dirs = [Path(d) for d in results_dirs]
    names = list(only) if only else list(BUILDERS)
    unknown = [n for n in names if n not in BUILDERS]
    if unknown:
        raise SystemExit(f"unknown builder {unknown}; choose from {sorted(BUILDERS)}")
    if only:
        if "kernel_tables" in names and kernel_dest is None:
            raise SystemExit("kernel_tables writes under --kernel-dest (QuettaSim's data root); give it")
        if any(n != "kernel_tables" for n in names) and dest is None:
            raise SystemExit("the builders other than kernel_tables write under --dest; give it")
    names = [n for n in names if (kernel_dest if n == "kernel_tables" else dest) is not None]
    if not names:
        raise SystemExit("give --dest, --kernel-dest or both")
    if "kernel_tables" in names:
        path = Path(kernel_dest) / "kernel_data" / "qbench_manifest.json"
        kernel = manifest(results_dirs)
        try:
            prior = json.loads(path.read_text()) if path.is_file() else {"kernel_tables": {}}
        except ValueError:
            prior = None
        if not isinstance(prior, dict) or not isinstance(prior.get("kernel_tables"), dict):
            raise SystemExit(f"{path}: malformed manifest")
    written: list[Path] = []
    tables = None
    for name in names:
        run_dirs.cache_clear()
        if name == "kernel_tables":
            paths, tables = run_kernel_tables(results_dirs, Path(kernel_dest), replace_tables)
            written.extend(paths)
        else:
            written.extend(BUILDERS[name](results_dirs, Path(dest)))
    run_dirs.cache_clear()
    if any(n != "kernel_tables" for n in names):
        Path(dest).mkdir(parents=True, exist_ok=True)
        (Path(dest) / "manifest.json").write_text(json.dumps(manifest(results_dirs), indent=2) + "\n")
    if tables:
        old = prior["kernel_tables"]
        new = {}
        for rel, runs in tables.items():
            entry = old.get(rel)
            before = entry.get("runs") if isinstance(entry, dict) else None
            if runs[:1] == ["adopted from dest"] and isinstance(before, list) and before:
                runs = list(dict.fromkeys(before + runs[1:]))
            new[rel] = {"runs": runs, "bench_sha": kernel["bench_sha"], "exported_at": kernel["exported_at"]}
        write_atomic(path, json.dumps({**prior, **kernel, "kernel_tables": {**old, **new}}, indent=2) + "\n")
    return written


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="qbench export", description=__doc__)
    ap.add_argument(
        "--results",
        type=Path,
        nargs="+",
        default=[Path("results")],
        help="results dirs the builders read; a named table comes from the ok run that "
        "finished last, a tie from the earlier dir",
    )
    ap.add_argument(
        "--dest",
        type=Path,
        help="where every builder but kernel_tables writes (kernels/, results/, "
        "manifest.json); those builders run only when it is given",
    )
    ap.add_argument(
        "--kernel-dest",
        type=Path,
        help="QuettaSim's data root ($QUETTASIM_DATA, e.g. ../QuettaSim/data): "
        "kernel_tables writes its kernel_data/ (with qbench_manifest.json) there, "
        "and runs only when given",
    )
    ap.add_argument(
        "--replace-tables",
        action="store_true",
        help="replace the kernel tables (and sidecars) --kernel-dest holds where merging "
        "into them would stop, dropping their rows; the others still merge",
    )
    ap.add_argument("--only", nargs="+", choices=sorted(BUILDERS), help="run just these builders (default: all)")
    a = ap.parse_args(argv)
    for path in export(a.results, a.dest, a.only, kernel_dest=a.kernel_dest, replace_tables=a.replace_tables):
        print(path)


if __name__ == "__main__":
    main()
