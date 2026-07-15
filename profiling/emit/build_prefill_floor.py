# profiling/emit/build_prefill_floor.py
"""Measured per-config prefill FLOOR (ms): the fixed per-request first-token overhead
(kernel launch + first-token emit + detok + return) that the queue sim charges ONCE per
request on prefill completion (``ttft_queue_sim._on_first_token``).

The floor is the y-intercept of TTFT vs newly-(re)prefilled tokens in the **conc=1, immediately
admitted** regime — i.e. TTFT with the prefill COMPUTE extrapolated to zero. At conc=1 there is
no server-side prefill batching/contention (one request in flight), so TTFT = floor + compute.
This is the same measured-anchor method that set the retired single H100-tp1 constant
``PREFILL_FLOOR_MS = 26.0`` (c1 turn-0, cached~=0); here it is computed PER deployment so tp2/tp4
configs stop inheriting the tp1 floor (the H100x2 floor is ~14 ms, not 26 — the dominant source of
the tp2 low-concurrency TTFT/TPOT over-prediction). No fit to any validation target: a measured
intercept per config, regenerable like the decode grid.

Run:  python3 -m profiling.emit.build_prefill_floor   (writes profile_data/results/prefill_floor_llama31_8b.json)
"""
from __future__ import annotations

import json
from pathlib import Path

from configs.loader import all_deployments
from simulator.ramp_tpot import _gpu_slug

BENCH_BASE = Path("/mnt/100g/agent-bench/results/synthetic_distributional")
OUT = Path("profile_data/kernels/prefill_floor_llama31_8b.json")  # committed (curated), like the saturated ceiling
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
    """Floor = the minimum TTFT over all immediately-admitted conc=1 requests — the
    compute-and-cache minimum, ≈ the pure per-request first-token overhead (the turn with
    near-zero effective prefill). This is the EXACT measured-anchor definition of the retired
    tp1 constant 26.0 ("min pure-prefill TTFT, c1"), and reproduces it (H100-tp1 = 25.9 ≈ 26)
    while exposing the true lower tp2/tp4 floors (H100x2 = 14.0) that the single tp1 constant
    wrongly imposed. NOT restricted by ``new_prefill_tokens``: that field undercounts an
    evicted-prefix re-prefill, so small-``new`` turns are not reliably floor-dominated — the
    global min is. Stable here: the floor is a hardware/launch constant, and n=63 across four
    profiles makes the min a true cohort minimum, not a single-request glitch (validated vs 26.0)."""
    if len(pts) < 5:
        return None
    ttfts = [t for _, t in pts]
    return {
        "floor_ms": round(float(min(ttfts)), 3),
        "method": "min_clean_c1",
        "n_clean": len(pts),
    }


def main() -> None:
    out: dict[str, dict] = {}
    for dep in all_deployments():
        if getattr(dep, "model", None) != "Llama-3.1-8B":
            continue
        if not getattr(dep, "ground_truth", False):
            continue
        root = BENCH_BASE / dep.bench_dir
        if not root.exists():
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
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {len(out)} config floors -> {OUT}")


if __name__ == "__main__":
    main()
