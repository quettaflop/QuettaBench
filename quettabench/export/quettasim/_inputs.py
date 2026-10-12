"""Find the result files the QuettaSim builders read."""
from __future__ import annotations

import functools
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from quettabench.metrics.results import RUN_MARKERS

TP_DEGREES = (2, 4)
# The finish time of a run with none, or none that parses: older than any.
NEVER = datetime.min.replace(tzinfo=timezone.utc)


@functools.cache
def run_dirs(root: Path) -> dict[Path, str | None]:
    """{run dir: why export leaves it out, None when kept} for every run dir
    under root (root included): run.json missing (a run that never finished),
    unparsable, or a run_status other than ok. A server/ dir under a run dir
    (a probe's launched server, engine.json there) is part of that run; a
    .clearing-* dir (runner.clear_run_dir cut short) is none.
    Cached per root; export() clears it before each builder."""
    runs = {}
    dirs = sorted({p.parent for marker in RUN_MARKERS for p in Path(root).rglob(marker)
                   if not any(part.startswith(".clearing-") for part in p.parts)})
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
    """The run dir path belongs to: path itself when it is a run dir, else its
    nearest ancestor (up to root) that is one, None when it sits in none (a
    table without a run.json)."""
    for d in (path, *path.parents):
        if d in runs:
            return d
        if d == root:
            return None
    return None


def files(root: Path, pattern: str, recursive: bool = True) -> list[Path]:
    """Paths under root matching pattern, sorted, without those whose run dir
    (owner) export leaves out."""
    root = Path(root)
    runs = run_dirs(root)
    hits = root.rglob(pattern) if recursive else root.glob(pattern)
    return [p for p in sorted(hits) if runs.get(owner(p, root, runs)) is None]


def quettasim_symbol_gone(symbol: str) -> None:
    """Stop a builder whose QuettaSim anchor did not import, with one line instead
    of a traceback. QuettaSim main removed simulator/ttft_queue_sim.py and has no
    constant with the same meaning."""
    raise SystemExit(f"QuettaSim main has no {symbol} (simulator.ttft_queue_sim was removed); "
                     "this builder needs a new anchor before it can run")


def finished(run_dir: Path | None) -> datetime:
    """run_dir/run.json's provenance.finished as a UTC instant, NEVER for no
    run dir or no time that parses."""
    try:
        when = json.loads((Path(run_dir) / "run.json").read_text())["provenance"]["finished"]
        return datetime.fromisoformat(when).astimezone(timezone.utc)
    except (OSError, ValueError, KeyError, TypeError):
        return NEVER


def find(results_dirs: Iterable[Path], name: str) -> Path | None:
    """The file called `name` whose ok run finished last (provenance.finished),
    over every results dir; a file in no run dir counts as finished never.
    A tie goes to the earlier results dir, then the first path."""
    best = None
    for rank, root in enumerate(map(Path, results_dirs)):
        runs = run_dirs(root)
        for path in files(root, name):
            key = (finished(owner(path, root, runs)), -rank)
            if best is None or key > best[0]:
                best = (key, path)
    return best[1] if best else None


def find_dir(results_dirs: Iterable[Path], name: str) -> Path | None:
    """First directory called `name` under the results dirs, searched in list
    order, without one that is itself, or sits in, a run export leaves out."""
    for root in results_dirs:
        hit = next((p for p in files(root, name) if p.is_dir()), None)
        if hit is not None:
            return hit
    return None


def stage_split_legs(results_dirs: Iterable[Path], gpu: str) -> list[tuple[int, Path, Path]]:
    """(tp, tp1 csv, tpN csv) for every tp degree whose two stage-split legs
    are present, each leg the one find picks (its run finished last)."""
    tp1 = find(results_dirs, f"serving_stage_split_{gpu}.csv")
    legs = []
    for tp in TP_DEGREES:
        tpn = find(results_dirs, f"serving_stage_split_{gpu}_tp{tp}.csv")
        if tp1 and tpn:
            legs.append((tp, tp1, tpn))
    return legs
