"""Measured kernel tables -> dest/kernel_data/<dir>/<kind>/<file>, dir graph or eager.

That is the tree QuettaSim's kernel_composed backend reads
(engine/backends/kernel_composed_cost.py kernel_artifact_paths, under the
data root $QUETTASIM_DATA, default the checkout's data/), so dest is that
data root, given to `qbench export` as --kernel-dest. A probe run writes
<method>/<kind>/<file> under its run dir: results/probes/<gpu>/<probe>/<run-id>/,
or results/<run>/<probe>/ for a `qbench run` probe. Method ncu goes to
graph/, cuda_event to eager/ (DIRS). Each table gets the <file>.meta.json
sidecar QuettaSim's tools/kernel_manifest.py checks: its mode
(kernel_composed/provenance.py MODES, from the probe, its flags and the
kind), tool, reduce, gpu and the date it was measured. Each run holds only
the rows it measured; _inputs.files leaves out failed and unfinished runs.

Per dest file, over every ok run of the probe(s) that wrote it, in the
order the runs finished (run.json provenance.pulled, else provenance.finished):
- a table with a row key (ROW_KEYS, the key each probe's old upsert
  replaced on) is the union of the runs' rows, and on a key the run that
  finished last wins; gemm and moe runs without --append drop the rows of
  runs that finished before them, as they once rewrote the whole file;
- any other table is the file of the run that finished last;
- one run's table is copied byte for byte.
A keyed table's key columns compare as numbers (int, total_weight_gb a
float; gpu and layout as text; an empty op is all_reduce), and within one
run's table the last row on a key wins.
A table dest already holds is the oldest source, so its rows stay unless
a run holds their key (QuettaSim's tables were upserted in place); its
repeated rows on a key all stay (QuettaSim's gemm and moe loaders reduce
them), but for reparallel and kv_transfer, whose upserts held a dict, its
last row on a key wins. Its header's columns join the union, a JSON
table's entries join it too, and a merged table keeps its line ending
(JSON its indent and final newline).
Every merge over a dest CSV is checked: each dest row is in the output
cell for cell or a run re-measured its key, and each dest column is in
the output. Where that fails (a table with no row key, a gemm or moe run
without --append, a run's table without a dest column), the export stops
naming the file and its row or column counts, unless replace is given
(--replace-tables), which replaces just those tables and their sidecars as
built; every other table still merges. Its sidecar keeps the fields qbench
does not write and the ones qbench leaves null, joins qbench's notes to
its own, and, when its rows were kept, spans measured over both
("<old> .. <new>"), names both tools (" + ") and, timed in another mode
than the runs, makes mode unknown.
Every table is built and checked in a temp dir under dest first: a row
with too many or too few cells, an empty or unparsable key, or JSON that
does not parse (or, for a union, is no object) stops the export with dest
as it was; then each file replaces its dest file.
Two probes write one dest file in two cases, and they merge:
- collective and vllm_allreduce, graph/collectives/<gpu>_tp<N>.csv: the
  collective probe's rows (path nccl), without its all_reduce rows when a
  vllm_allreduce run is there too, replacing earlier rows on (op, world,
  bytes); then vllm_allreduce's all_reduce rows, which replace every
  all_reduce row of their world, as its upsert did; the header is dest's
  and the runs' columns, so an export over its own table changes nothing;
- gemm and elementwise, <dir>/elementwise/<gpu>.json: the elementwise
  probe's latest table, with every gemm run's entries (fp8_act_quant)
  unioned over it, the later finished run winning a key.
Any other file two probes write stops the export.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from quettabench.export.quettasim import _inputs

# A probe run's method dir -> QuettaSim's kernel_data dir (kernel_composed/provenance.py DIRS).
DIRS = {"ncu": "graph", "cuda_event": "eager"}
# eager/ kinds timed by host clocks around a whole operation (mode host_wallclock).
HOST_KINDS = ("reparallel", "kv_transfer")
# The reduce each probe takes over its timings, for graph_replay and eager tables.
REDUCE = {"gemm": "min", "moe": "min", "elementwise": "median", "flash_attn": "median",
          "cross_attn": "median", "gdn": "median"}
# The key a collectives row replaces a dest row on.
COLLECTIVE_KEY = ("op", "world", "bytes")
GEOMETRY = ("n_heads", "n_kv_heads", "head_dim")
# (probe, kind) -> the columns a CSV row is keyed on: flash_attn and
# cross_attn replace every row of a head config, the others one row.
ROW_KEYS = {
    ("flash_attn", "flash_attn"): GEOMETRY,
    ("flash_attn", "fa3_prefill"): GEOMETRY,
    ("cross_attn", "fa3_cross"): GEOMETRY,
    ("gemm", "gemm"): ("M", "N", "K"),
    ("gemm", "gemm_wide"): ("M", "N", "K"),
    ("moe", "moe"): ("tokens", "n_experts", "top_k", "intermediate", "hidden"),
    ("reparallel", "reparallel"): ("gpu", "world", "tp_from", "tp_to", "total_weight_gb"),
    ("kv_transfer", "kv_transfer"): ("gpu", "kv_heads_per_rank", "head_dim", "block_size", "layers",
                                     "layout", "tokens"),
}
# Keyed tables whose union is sorted, on these columns, as their old upserts wrote them.
SORTED = {("reparallel", "reparallel"): ROW_KEYS[("reparallel", "reparallel")],
          ("kv_transfer", "kv_transfer"): ("gpu", "kv_heads_per_rank", "layout", "tokens",
                                           "head_dim", "block_size", "layers")}
# Probes that live outside quettabench/probes/, by their sidecar tool.
TOOLS = {"kv_transfer": "quettabench.profiling.kv_transfer"}
# Probes whose run without --append replaced the whole table.
RESET_WITHOUT_APPEND = ("gemm", "moe")
# (probe, kind) whose JSON entries union across runs (gemm's fp8_act_quant).
JSON_UNION = {("gemm", "elementwise")}
# When a dest table finished, as a source: before every run.
OLDEST = datetime.min.replace(tzinfo=timezone.utc)


def _record(run_dir: Path) -> tuple[str | None, list[str], str | None]:
    """(probe, its args, the day it finished) from run_dir/run.json; probe None
    for a run that is no probe run."""
    doc = json.loads((run_dir / "run.json").read_text())
    mode = (doc.get("config") or {}).get("mode") or {}
    probe = mode.get("probe") if mode.get("mode") == "probe" else None
    try:
        args = shlex.split(mode.get("probe_args") or "")
    except ValueError as e:
        raise SystemExit(f"{run_dir}: probe_args do not parse ({e})") from None
    finished = (doc.get("provenance") or {}).get("finished")
    return probe, args, finished[:10] if isinstance(finished, str) else None


def _mode(rel: Path, args: list[str]) -> str:
    """QuettaSim's mode for a table one run wrote with args."""
    where, kind = rel.parts[0], rel.parts[1]
    if where == "eager":
        return "host_wallclock" if kind in HOST_KINDS else "eager"
    if kind == "collectives":
        return "reference" if rel.stem.endswith("_nccl_vs_vllm") else "amortized_wallclock"
    return "ncu" if "--ncu" in args else "graph_replay"


def _gpu(args: list[str]) -> str | None:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--gpu-label", "--gpu")
    return ap.parse_known_args(args)[0].gpu_label


def _sidecar(rel: Path, by_probe: dict[str, list[Source]], runs: set[Path]) -> dict:
    """rel's <file>.meta.json over the runs whose rows it holds; runs timed in
    two modes make mode unknown."""
    probes, modes, gpus, reduces, days = set(), set(), set(), set(), []
    for probe, sources in by_probe.items():
        for run in (s[2] for s in sources if s[2] in runs):
            _, args, day = _record(run)
            mode = _mode(rel, args)
            probes.add(probe)
            modes.add(mode)
            gpus.add(_gpu(args))
            reduces.add("min" if mode == "ncu" else REDUCE.get(probe)
                        if mode in ("graph_replay", "eager") else None)
            days.append(day or "")
    return {"schema": 1, "table": rel.as_posix(), "dir": rel.parts[0],
            "gpu": gpus.pop() if len(gpus) == 1 else None,
            "mode": modes.pop() if len(modes) == 1 else "unknown",
            "tool": " + ".join(TOOLS.get(p, f"quettabench.probes.{p}") for p in sorted(probes)),
            "reduce": reduces.pop() if len(reduces) == 1 else None,
            "measured": max(days, default="") or None,
            "notes": "" if len(modes) <= 1 else f"rows timed as {', '.join(sorted(modes))}",
            "provenance": "qbench export kernel_tables; source runs in kernel_data/qbench_manifest.json"}


def _keep_sidecar(meta: dict, table: Path, kept: bool) -> dict:
    """meta over the sidecar dest has beside table: its other fields stay, a
    field meta leaves null keeps its value and meta's notes join its notes.
    When its rows were kept, measured spans both, tool names both, and a mode
    other than meta's makes mode unknown."""
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
    return {**prior, **meta, "mode": mode, "notes": notes, "tool": " + ".join(dict.fromkeys(tools)),
            "measured": " .. ".join(dict.fromkeys(days[:1] + days[-1:])) or None}


Source = tuple[datetime, Path, Path, bool]  # (finished, table, its run dir, ran with --append)


class Lost(SystemExit):
    """The export would drop a dest row or column; replace (--replace-tables)
    replaces just the tables that stop with it."""


def found(results_dirs: Iterable[Path]) -> dict[Path, dict[str, list[Source]]]:
    """{<dir>/<kind>/<file>: {probe: its ok runs' tables, in finished order}}."""
    tables: dict[Path, dict[str, list[Source]]] = {}
    for root in map(Path, results_dirs):
        runs = _inputs.run_dirs(root)
        for method, where in DIRS.items():
            for path in _inputs.files(root, f"{method}/*/*"):
                run = _inputs.owner(path, root, runs)
                if not path.is_file() or run is None or not (run / "run.json").is_file():
                    continue
                rel = path.relative_to(run).parts
                probe, args, _ = _record(run)
                if probe is None or rel[-3] != method or rel[:-3] not in ((), (probe,)):
                    continue
                source = (_inputs.finished(run), path, run, "--append" in args)
                tables.setdefault(Path(where, *rel[-2:]), {}).setdefault(probe, []).append(source)
    for by_probe in tables.values():
        for sources in by_probe.values():
            sources.sort(key=lambda s: (s[0], str(s[2])))
    return tables


def _runs_used(probe: str, rel: Path, sources: list[Source]) -> list[Source]:
    """The runs whose rows a probe's table is built from: all of them for a
    keyed table (from the last run without --append on, for gemm and moe),
    else the run that finished last."""
    kind = rel.parts[-2]
    if (probe, kind) in JSON_UNION and rel.suffix == ".json":
        return sources
    if (probe, kind) not in ROW_KEYS or rel.suffix != ".csv":
        return sources[-1:]
    if probe in RESET_WITHOUT_APPEND:
        resets = [i for i, s in enumerate(sources) if not s[3]]
        return sources[resets[-1]:] if resets else sources
    return sources


def _key(row: dict, key: tuple[str, ...]) -> tuple:
    """row's key, numbers parsed as the old upserts' int() read them; an
    empty op is all_reduce, as QuettaSim's collectives loader reads it."""
    return tuple((row.get("op") or "all_reduce") if c == "op" else row[c] if c in ("gpu", "layout")
                 else float(row[c]) if c == "total_weight_gb" else int(row[c]) for c in key)


def _newline(table: Path) -> str:
    """The line ending of table's first line."""
    with table.open("rb") as f:
        return "\r\n" if f.readline().endswith(b"\r\n") else "\n"


def _write_csv(out: Path, fields: list[str], rows: Iterable[dict], like: Path) -> None:
    """rows under fields, with like's line ending."""
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="", lineterminator=_newline(like))
        w.writeheader()
        w.writerows(rows)


def _read_csv(rel: Path, table: Path, run: Path,
              key: tuple[str, ...] = ()) -> tuple[list[str], list[dict]]:
    """table's header and rows; stops the export on a row with a missing or
    extra cell or a key that is empty or does not parse."""
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


def _read_json(rel: Path, table: Path, run: Path) -> object:
    try:
        return json.loads(table.read_text())
    except ValueError:
        raise SystemExit(f"{rel.as_posix()}: {run} malformed JSON") from None


def _union_csv(rel: Path, sources: list[Source], key: tuple[str, ...], out: Path,
               old: Source | None = None) -> list[Path]:
    """Rows of the dest table old, then of every source, oldest first; a
    source's rows replace the earlier rows on each key it holds. Within a
    run's table the last row on a key wins; old keeps its repeated rows on a
    key (QuettaSim's loaders reduce them) unless the table's old upsert held
    its rows in a dict (SORTED). Returns the run dirs with a row kept."""
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
    text = sources[0][1].read_text()
    lines = text.splitlines()
    indent = len(lines[1]) - len(lines[1].lstrip(" ")) if len(lines) > 1 else None
    out.write_text(json.dumps({k: v for k, (v, _) in merged.items()}, indent=indent)
                   + "\n" * text.endswith("\n"))
    return list(dict.fromkeys(run for _, run in merged.values()))


def _merge_collectives(rel: Path, collective: Source | None, vllm: Source | None, out: Path,
                       old: Source | None = None) -> list[Path]:
    """The dest table old's rows, then the collective run's, each replacing
    the earlier rows on its COLLECTIVE_KEY, then vllm_allreduce's all_reduce
    rows, which replace every earlier all_reduce row of their world, as its
    upsert did. Beside a vllm_allreduce run the collective run's all_reduce
    rows are left out. The header is old's and the runs' columns; the
    collective run's rows fill an empty dtype_bytes (2) and path (nccl)."""
    sources = [(s, _read_csv(rel, s[1], s[2], COLLECTIVE_KEY)) for s in (old, collective, vllm) if s]
    fields: list[str] = []
    for _, (head, _) in sources:
        fields += [c for c in head if c not in fields]
    fill = {c: v for c, v in (("dtype_bytes", "2"), ("path", "nccl")) if c in fields}
    kept: list[tuple[dict, Path]] = []
    newer: set[tuple] = set()
    for source, (_, rows) in sources:
        if source is collective:
            rows = [{**r, **{c: r.get(c) or v for c, v in fill.items()}} for r in rows
                    if not (vllm and _key(r, ("op",)) == ("all_reduce",))]
        if source is vllm:
            rows = [r for r in rows if _key(r, ("op",)) == ("all_reduce",)]
            worlds = {_key(r, ("world",)) for r in rows}
            mine = {_key(r, COLLECTIVE_KEY) for r, _ in kept
                    if _key(r, ("op",)) == ("all_reduce",) and _key(r, ("world",)) in worlds}
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
    if len(probes) > 1 and (probes, rel.suffix) not in ((["collective", "vllm_allreduce"], ".csv"),
                                                        (["elementwise", "gemm"], ".json")):
        raise SystemExit(f"{rel.as_posix()}: written by probes {probes}; kernel_tables "
                         "merges only collective+vllm_allreduce and elementwise+gemm")


def _lost(rel: Path, old: Source, out: Path, key: tuple[str, ...], newer: set[tuple],
          rows: list[dict] | None = None) -> None:
    """Stop (Lost) when out lacks a column of the dest table old, or drops one
    of its rows (rows, else all of old's): each must be in out, cell for
    cell, unless a run re-measured its key (newer)."""
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


def _write(rel: Path, by_probe: dict[str, list[Source]], out: Path,
           old: Source | None = None) -> list[Path]:
    """Build one dest file (_check passed) over the dest table old, when there
    is one; returns the run dirs (and old's path) whose rows it holds."""
    used = {probe: _runs_used(probe, rel, sources) for probe, sources in by_probe.items()}
    probes = sorted(used)
    olds = [old] if old else []
    tp_table = rel.parts[-2] == "collectives" and not rel.stem.endswith("_nccl_vs_vllm") and rel.suffix == ".csv"
    if tp_table and set(probes) <= {"collective", "vllm_allreduce"}:
        return _merge_collectives(rel, *(used[p][-1] if p in used else None
                                         for p in ("collective", "vllm_allreduce")), out, old)
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
    if len(sources) == 1:
        _, table, run, _ = sources[0]
        if rel.suffix == ".json":
            _read_json(rel, table, run)
        elif rel.suffix == ".csv":
            _read_csv(rel, table, run, key)
        shutil.copyfile(table, out)
        return [run]
    if rel.suffix == ".json":
        return _union_json(rel, sources, out)
    return _union_csv(rel, sources, key, out)


def run(results_dirs: Iterable[Path], dest: Path,
        replace: bool = False) -> tuple[list[Path], dict[str, list[str]]]:
    """Build every table and its sidecar in a temp dir under dest, over the
    tables dest holds (or, given replace, without the ones that would stop
    with Lost), then replace the dest files, so a stop
    leaves the tables in dest as they were (and no dest when this run made
    it). Returns the files written and, for
    the manifest, {<dir>/<kind>/<file>: the run dirs with rows in it, in
    finished order}."""
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
                order = sorted((s[0], str(s[2])) for sources in by_probe.values() for s in sources
                               if s[2] in runs)
                used[rel.as_posix()] = (["adopted from dest"] if table in runs else []) + list(
                    dict.fromkeys(run for _, run in order))
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
