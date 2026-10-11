"""Find the result files the QuettaSim builders read."""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

TP_DEGREES = (2, 4)
# Files only a run dir holds; a dir with one of them is a run dir.
RUN_MARKERS = ("run.json", "requests.jsonl", "engine.json")


def skipped_runs(root: Path) -> dict[Path, str]:
    """{run dir: why} for every run dir under root (root included) that export
    leaves out: run.json missing (a run that never finished), unparsable, or
    a run_status other than ok."""
    skipped = {}
    dirs = sorted({p.parent for marker in RUN_MARKERS for p in Path(root).rglob(marker)})
    for run_dir in dirs:
        path = run_dir / "run.json"
        if not path.exists():
            skipped[run_dir] = "no run.json"
            continue
        try:
            status = json.loads(path.read_text())["run_status"]["status"]
        except (OSError, ValueError, KeyError, TypeError) as e:
            skipped[run_dir] = f"run.json unreadable ({type(e).__name__})"
            continue
        if status != "ok":
            skipped[run_dir] = f"run_status {status}"
    return skipped


def files(root: Path, pattern: str, recursive: bool = True) -> list[Path]:
    """Paths under root matching pattern, sorted, without those inside a
    run dir skipped_runs leaves out."""
    skipped = skipped_runs(root)
    hits = Path(root).rglob(pattern) if recursive else Path(root).glob(pattern)
    return [p for p in sorted(hits) if not any(p.is_relative_to(d) for d in skipped)]


def quettasim_symbol_gone(symbol: str) -> None:
    """Stop a builder whose QuettaSim anchor did not import, with one line instead
    of a traceback. QuettaSim main removed simulator/ttft_queue_sim.py and has no
    constant with the same meaning."""
    raise SystemExit(f"QuettaSim main has no {symbol} (simulator.ttft_queue_sim was removed); "
                     "this builder needs a new anchor before it can run")


def find(results_dirs: Iterable[Path], name: str) -> Path | None:
    """First file called `name` under the results dirs, searched in list order."""
    for root in results_dirs:
        hit = next(iter(files(root, name)), None)
        if hit is not None:
            return hit
    return None


def find_dir(results_dirs: Iterable[Path], name: str) -> Path | None:
    """First directory called `name` under the results dirs, searched in list order."""
    for root in results_dirs:
        hit = next((p for p in files(root, name) if p.is_dir()), None)
        if hit is not None:
            return hit
    return None


def stage_split_legs(results_dirs: Iterable[Path], gpu: str) -> list[tuple[int, Path, Path]]:
    """(tp, tp1 csv, tpN csv) for every tp degree whose two stage-split legs are present."""
    tp1 = find(results_dirs, f"serving_stage_split_{gpu}.csv")
    legs = []
    for tp in TP_DEGREES:
        tpn = find(results_dirs, f"serving_stage_split_{gpu}_tp{tp}.csv")
        if tp1 and tpn:
            legs.append((tp, tp1, tpn))
    return legs
