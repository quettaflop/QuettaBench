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
import shlex
import shutil
from collections.abc import Iterable
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


Source = tuple[str, Path, Path, bool]  # (finished, table, its run dir, ran with --append)


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
    if (probe, kind) in JSON_UNION:
        return sources
    if (probe, kind) not in ROW_KEYS or rel.suffix != ".csv":
        return sources[-1:]
    if probe in RESET_WITHOUT_APPEND:
        resets = [i for i, s in enumerate(sources) if not s[3]]
        return sources[resets[-1]:] if resets else sources
    return sources


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _union_csv(sources: list[Source], key: tuple[str, ...], out: Path) -> list[Path]:
    """Rows of every source, oldest first; a source's rows replace the earlier
    rows on each key it holds. Returns the run dirs with a row kept."""
    fields: list[str] = []
    kept: list[tuple[dict, Path]] = []
    for _, table, run, _ in sources:
        head, rows = _read_csv(table)
        fields += [c for c in head if c not in fields]
        mine = {tuple(r.get(c) for c in key) for r in rows}
        kept = [(r, d) for r, d in kept if tuple(r.get(c) for c in key) not in mine]
        kept += [(r, run) for r in rows]
    if key == ROW_KEYS[("reparallel", "reparallel")]:
        kept.sort(key=lambda rd: (rd[0]["gpu"], int(rd[0]["world"]), int(rd[0]["tp_from"]),
                                  int(rd[0]["tp_to"]), float(rd[0]["total_weight_gb"])))
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(r for r, _ in kept)
    return list(dict.fromkeys(d for _, d in kept))


def _union_json(sources: list[Source], out: Path) -> list[Path]:
    """Entries of every source in finished order, the later run winning a key."""
    merged: dict[str, tuple[object, Path]] = {}
    for _, table, run, _ in sorted(sources, key=lambda s: (s[0], str(s[2]))):
        merged.update({k: (v, run) for k, v in json.loads(table.read_text()).items()})
    out.write_text(json.dumps({k: v for k, (v, _) in merged.items()}, indent=2) + "\n")
    return list(dict.fromkeys(run for _, run in merged.values()))


def _merge_collectives(collective: Source, vllm: Source, out: Path) -> list[Path]:
    with collective[1].open(newline="") as f:
        kept = [r for r in csv.DictReader(f) if (r.get("op") or "all_reduce") != "all_reduce"]
    with vllm[1].open(newline="") as f:
        ours = [r for r in csv.DictReader(f) if (r.get("op") or "all_reduce") == "all_reduce"]
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
        return _merge_collectives(used["collective"][-1], used["vllm_allreduce"][-1], out)
    if probes == ["elementwise", "gemm"]:
        return _union_json(used["elementwise"] + used["gemm"], out)
    (probe,) = probes
    sources = used[probe]
    if len(sources) == 1:
        shutil.copyfile(sources[0][1], out)
        return [sources[0][2]]
    if rel.suffix == ".json":
        return _union_json(sources, out)
    return _union_csv(sources, ROW_KEYS[(probe, rel.parts[-2])], out)


def run(results_dirs: Iterable[Path], dest: Path) -> tuple[list[Path], dict[str, list[str]]]:
    """Write every table, after checking all of them merge, so a stop leaves
    dest untouched. Returns the files written and, for the manifest,
    {<method>/<kind>/<file>: the run dirs with rows in it, in finished order}."""
    tables = sorted(found(results_dirs).items())
    for rel, by_probe in tables:
        _check(rel, by_probe)
    written, used = [], {}
    for rel, by_probe in tables:
        out = Path(dest) / "kernel_data" / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        runs = set(_write(rel, by_probe, out))
        order = sorted((s[0], str(s[2])) for sources in by_probe.values() for s in sources
                       if s[2] in runs)
        used[rel.as_posix()] = list(dict.fromkeys(run for _, run in order))
        print(f"{rel.as_posix()} <- {', '.join(sorted(by_probe))}")
        written.append(out)
    return written, used
