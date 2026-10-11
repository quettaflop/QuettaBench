"""Measured kernel tables -> dest/kernel_data/<method>/<kind>/<file>.

That is the tree QuettaSim's kernel_composed backend reads
(engine/backends/kernel_composed_cost.py kernel_artifact_paths, under the
data root $QUETTASIM_DATA, default the checkout's data/), so dest is that
data root, given to `qbench export` as --kernel-dest. A probe run writes <method>/<kind>/<file> (method ncu or
cuda_event) under its run dir: results/probes/<gpu>/<probe>/<run-id>/, or
results/<run>/<probe>/ for a `qbench run` probe. Each run holds only the
rows it measured; _inputs.files leaves out failed and unfinished runs.

Per dest file, over every ok run of the probe(s) that wrote it, in the
order the runs finished (run.json provenance.finished):
- a table with a row key (ROW_KEYS, the key each probe's old upsert
  replaced on) is the union of the runs' rows, and on a key the run that
  finished last wins; gemm and moe runs without --append drop the rows of
  runs that finished before them, as they once rewrote the whole file;
- any other table is the file of the run that finished last;
- one run's table is copied byte for byte.
A keyed table's key columns compare as numbers (int, total_weight_gb a
float), and within one run's table the last row on a key wins.
Every table is built and checked in a temp dir under dest first: a row
with too many or too few cells, an empty or unparsable key, or JSON that
does not parse (or, for a union, is no object) stops the export with dest
as it was; then each file replaces its dest file.
Two probes write one dest file in two cases, and they merge:
- collective and vllm_allreduce, ncu/collectives/<gpu>_tp<N>.csv: the
  collective probe's rows other than all_reduce (path nccl), then
  vllm_allreduce's all_reduce rows, as vllm_allreduce's upsert did;
- gemm and elementwise, <method>/elementwise/<gpu>.json: the elementwise
  probe's latest table, with every gemm run's entries (fp8_act_quant)
  unioned over it, the later finished run winning a key.
Any other file two probes write stops the export.
"""
from __future__ import annotations

import csv
import json
import os
import shlex
import shutil
import tempfile
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from quettabench.export.quettasim import _inputs

METHODS = ("ncu", "cuda_event")
COLLECTIVE_FIELDS = ("op", "world", "bytes", "latency_us", "dtype_bytes", "path")
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
}
# Probes whose run without --append replaced the whole table.
RESET_WITHOUT_APPEND = ("gemm", "moe")
# (probe, kind) whose JSON entries union across runs (gemm's fp8_act_quant).
JSON_UNION = {("gemm", "elementwise")}


def _record(run_dir: Path) -> tuple[str | None, bool]:
    """(probe, ran with --append) from run_dir/run.json; probe None for a run
    that is no probe run."""
    doc = json.loads((run_dir / "run.json").read_text())
    mode = (doc.get("config") or {}).get("mode") or {}
    probe = mode.get("probe") if mode.get("mode") == "probe" else None
    return probe, "--append" in shlex.split(mode.get("probe_args") or "")


Source = tuple[datetime, Path, Path, bool]  # (finished, table, its run dir, ran with --append)


def found(results_dirs: Iterable[Path]) -> dict[Path, dict[str, list[Source]]]:
    """{<method>/<kind>/<file>: {probe: its ok runs' tables, in finished order}}."""
    tables: dict[Path, dict[str, list[Source]]] = {}
    for root in map(Path, results_dirs):
        runs = _inputs.run_dirs(root)
        for method in METHODS:
            for path in _inputs.files(root, f"{method}/*/*"):
                run = _inputs.owner(path, root, runs)
                if not path.is_file() or run is None or not (run / "run.json").is_file():
                    continue
                rel = path.relative_to(run).parts
                probe, append = _record(run)
                if probe is None or rel[-3] != method or rel[:-3] not in ((), (probe,)):
                    continue
                source = (_inputs.finished(run), path, run, append)
                tables.setdefault(Path(*rel[-3:]), {}).setdefault(probe, []).append(source)
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
    """row's key, numbers parsed as the old upserts' int() read them."""
    return tuple(row[c] if c == "gpu" else float(row[c]) if c == "total_weight_gb" else int(row[c])
                 for c in key)


def _read_csv(rel: Path, table: Path, run: Path,
              key: tuple[str, ...] = ()) -> tuple[list[str], list[dict]]:
    """table's header and rows; stops the export on a row with a missing or
    extra cell or a key that is empty or does not parse."""
    with table.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for n, row in enumerate(rows, 1):
        try:
            if None in row or None in row.values() or not all(row.get(c) for c in key):
                raise ValueError
            _key(row, key)
        except ValueError:
            raise SystemExit(f"{rel.as_posix()}: {run} row {n} malformed") from None
    return list(reader.fieldnames or []), rows


def _read_json(rel: Path, table: Path, run: Path) -> object:
    try:
        return json.loads(table.read_text())
    except ValueError:
        raise SystemExit(f"{rel.as_posix()}: {run} malformed JSON") from None


def _union_csv(rel: Path, sources: list[Source], key: tuple[str, ...], out: Path) -> list[Path]:
    """Rows of every source, oldest first; a source's rows replace the earlier
    rows on each key it holds. Returns the run dirs with a row kept."""
    fields: list[str] = []
    kept: list[tuple[dict, Path]] = []
    for _, table, run, _ in sources:
        head, rows = _read_csv(rel, table, run, key)
        if key != GEOMETRY:
            rows = list({_key(r, key): r for r in rows}.values())
        fields += [c for c in head if c not in fields]
        mine = {_key(r, key) for r in rows}
        kept = [(r, d) for r, d in kept if _key(r, key) not in mine]
        kept += [(r, run) for r in rows]
    if key == ROW_KEYS[("reparallel", "reparallel")]:
        kept.sort(key=lambda rd: _key(rd[0], key))
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(r for r, _ in kept)
    return list(dict.fromkeys(d for _, d in kept))


def _union_json(rel: Path, sources: list[Source], out: Path) -> list[Path]:
    """Entries of every source in finished order, the later run winning a key."""
    merged: dict[str, tuple[object, Path]] = {}
    for _, table, run, _ in sorted(sources, key=lambda s: (s[0], str(s[2]))):
        entries = _read_json(rel, table, run)
        if not isinstance(entries, dict):
            raise SystemExit(f"{rel.as_posix()}: {run} malformed JSON (not an object)")
        merged.update({k: (v, run) for k, v in entries.items()})
    out.write_text(json.dumps({k: v for k, (v, _) in merged.items()}, indent=2) + "\n")
    return list(dict.fromkeys(run for _, run in merged.values()))


def _merge_collectives(rel: Path, collective: Source, vllm: Source, out: Path) -> list[Path]:
    kept = [r for r in _read_csv(rel, collective[1], collective[2])[1]
            if (r.get("op") or "all_reduce") != "all_reduce"]
    ours = [r for r in _read_csv(rel, vllm[1], vllm[2])[1]
            if (r.get("op") or "all_reduce") == "all_reduce"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLLECTIVE_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in kept:
            w.writerow({**r, "dtype_bytes": r.get("dtype_bytes") or 2, "path": r.get("path") or "nccl"})
        w.writerows(ours)
    return [s[2] for s, rows in ((collective, kept), (vllm, ours)) if rows]


def _check(rel: Path, by_probe: dict[str, list[Source]]) -> None:
    """Stop on a file two probes write that kernel_tables does not merge."""
    probes = sorted(by_probe)
    if len(probes) > 1 and (probes, rel.suffix) not in ((["collective", "vllm_allreduce"], ".csv"),
                                                        (["elementwise", "gemm"], ".json")):
        raise SystemExit(f"{rel.as_posix()}: written by probes {probes}; kernel_tables "
                         "merges only collective+vllm_allreduce and elementwise+gemm")


def _write(rel: Path, by_probe: dict[str, list[Source]], out: Path) -> list[Path]:
    """Build one dest file (_check passed); returns the run dirs whose rows it holds."""
    used = {probe: _runs_used(probe, rel, sources) for probe, sources in by_probe.items()}
    probes = sorted(used)
    if probes == ["collective", "vllm_allreduce"]:
        return _merge_collectives(rel, used["collective"][-1], used["vllm_allreduce"][-1], out)
    if probes == ["elementwise", "gemm"]:
        return _union_json(rel, used["elementwise"] + used["gemm"], out)
    (probe,) = probes
    sources = used[probe]
    key = ROW_KEYS.get((probe, rel.parts[-2]), ())
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


def run(results_dirs: Iterable[Path], dest: Path) -> tuple[list[Path], dict[str, list[str]]]:
    """Build every table in a temp dir under dest, then replace the dest files,
    so a stop leaves the tables in dest as they were. Returns the files
    written and, for the manifest, {<method>/<kind>/<file>: the run dirs
    with rows in it, in finished order}."""
    tables = sorted(found(results_dirs).items())
    for rel, by_probe in tables:
        _check(rel, by_probe)
    if not tables:
        return [], {}
    used = {}
    Path(dest).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest, prefix=".kernel_tables-") as tmp:
        for rel, by_probe in tables:
            out = Path(tmp) / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            runs = set(_write(rel, by_probe, out))
            order = sorted((s[0], str(s[2])) for sources in by_probe.values() for s in sources
                           if s[2] in runs)
            used[rel.as_posix()] = list(dict.fromkeys(run for _, run in order))
        written = []
        for rel, by_probe in tables:
            out = Path(dest) / "kernel_data" / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            os.replace(Path(tmp) / rel, out)
            print(f"{rel.as_posix()} <- {', '.join(sorted(by_probe))}")
            written.append(out)
    return written, used
