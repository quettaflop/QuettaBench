# quettabench/export/quettasim/prefill_floor.py
"""Measured conc=1 prefill floor (ms) per deployment.

y-intercept of TTFT vs new-prefill tokens when a request is admitted immediately.
Needs QuettaSim/agentic-serve on PYTHONPATH (`configs.loader`, `simulator.ramp_tpot`) for the
deployment list; each deployment's bench dir is looked up in the results dirs. Writes
dest/kernels/prefill_floor_llama31_8b.json (see ``run``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from quettabench.export.quettasim._inputs import find_dir

OUT_NAME = "prefill_floor_llama31_8b.json"
PROFILES = [
    "swebench-multiturn-synth", "osworld-multiturn-synth",
    "terminalbench-multiturn-synth", "chat-multiturn-synth",
]
# conc levels with negligible server-side prefill contention (one or few in flight). c1 is the
# clean anchor; we pool c1 only (c>=5 already shows prefill batching, see the H100x2 trace).
CLEAN_CONCS = [1]
MAX_QUEUE_WAIT_MS = 1.0       # admitted immediately (no client-side admission wait)


def _clean_points(bench_root: Path) -> list[tuple[float, float]]:
    """(new_prefill_tokens, ttft_ms) for immediately-admitted conc=1 requests across profiles."""
    pts: list[tuple[float, float]] = []
    for profile in PROFILES:
        for conc in CLEAN_CONCS:
            f = bench_root / f"{profile}_conc{conc}.json"
            if not f.exists():
                continue
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            for r in d.get("per_request", []):
                if float(r.get("client_queue_wait_ms", 9.9)) > MAX_QUEUE_WAIT_MS:
                    continue
                ttft = r.get("ttft_ms")
                new = r.get("new_prefill_tokens")
                if ttft is None or new is None:
                    continue
                pts.append((float(new), float(ttft)))
    return pts


def _floor_from_points(pts: list[tuple[float, float]]) -> dict | None:
    """Min TTFT over immediately-admitted conc=1 requests (hardware/launch floor)."""
    if len(pts) < 5:
        return None
    ttfts = [t for _, t in pts]
    return {
        "floor_ms": round(float(min(ttfts)), 3),
        "method": "min_clean_c1",
        "n_clean": len(pts),
    }


def run(results_dirs: Iterable[Path], dest: Path) -> list[Path]:
    try:
        from configs.loader import all_deployments
        from simulator.ramp_tpot import _gpu_slug
    except ImportError as e:
        raise SystemExit(
            "prefill_floor needs QuettaSim/agentic-serve on PYTHONPATH "
            f"(could not import {e.name!r}). See profiling/README.md."
        ) from e
    out: dict[str, dict] = {}
    for dep in all_deployments():
        if getattr(dep, "model", None) != "Llama-3.1-8B":
            continue
        if not getattr(dep, "ground_truth", False):
            continue
        root = find_dir(results_dirs, dep.bench_dir)
        if root is None:
            continue
        pts = _clean_points(root)
        rec = _floor_from_points(pts)
        if rec is None:
            print(f"SKIP {dep.gpu_key}: too few clean conc1 points ({len(pts)})")
            continue
        rec["gpu_key"] = dep.gpu_key
        rec["tensor_parallel"] = getattr(dep, "tp", None)
        out[_gpu_slug(dep.gpu_key)] = rec
        print(f"{dep.gpu_key:18s} (tp{rec['tensor_parallel']}) floor={rec['floor_ms']:6.2f} ms "
              f"[{rec['method']}, n={rec['n_clean']}]")
    out_path = Path(dest) / "kernels" / OUT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {len(out)} config floors -> {out_path}")
    return [out_path]
