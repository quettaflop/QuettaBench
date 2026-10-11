"""Find the result files the QuettaSim builders read."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

TP_DEGREES = (2, 4)


def quettasim_symbol_gone(symbol: str) -> None:
    """Stop a builder whose QuettaSim anchor did not import, with one line instead
    of a traceback. QuettaSim main removed simulator/ttft_queue_sim.py and has no
    constant with the same meaning."""
    raise SystemExit(f"QuettaSim main has no {symbol} (simulator.ttft_queue_sim was removed); "
                     "this builder needs a new anchor before it can run")


def find(results_dirs: Iterable[Path], name: str) -> Path | None:
    """First file called `name` under the results dirs, searched in list order."""
    for root in results_dirs:
        hit = next(iter(sorted(Path(root).rglob(name))), None)
        if hit is not None:
            return hit
    return None


def find_dir(results_dirs: Iterable[Path], name: str) -> Path | None:
    """First directory called `name` under the results dirs, searched in list order."""
    for root in results_dirs:
        hit = next((p for p in sorted(Path(root).rglob(name)) if p.is_dir()), None)
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
