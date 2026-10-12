#!/usr/bin/env python3
"""Repair stored results where inter-chunk gaps inflated tpot_ms (reasoning deltas without content were skipped).

For a successful request with output_tokens > 1, non-empty itl_ms and len(itl_ms) < 0.8 * (output_tokens - 1), tpot_ms becomes (e2el_ms - ttft_ms) / (output_tokens - 1).
Summary and per-turn stats are recomputed, changed files get a tpot_repair stamp, and each directory gets a _tpot_repair.json manifest. Dry run unless --apply.
"""

import argparse
import glob
import json
import os
import statistics
from datetime import datetime, timezone

COVERAGE_THRESHOLD = 0.8
RULE = "len(itl)<0.8*(out-1) -> tpot=(e2el-ttft)/(out-1)"
REASON = "QuettaSim tools/GT_QUALITY_FLAGS.md Finding 1 (inter-chunk tpot)"


def repaired_tpot(row):
    """Corrected tpot_ms for a row the coverage guard fires on, else None."""
    if not row.get("success"):
        return None
    out = row.get("output_tokens") or 0
    itl = row.get("itl_ms") or []
    e2el, ttft = row.get("e2el_ms"), row.get("ttft_ms")
    if out <= 1 or not itl or e2el is None or ttft is None:
        return None
    if len(itl) >= COVERAGE_THRESHOLD * (out - 1):
        return None
    return round((e2el - ttft) / (out - 1), 2)


def percentile(values, p):
    """Linear-interpolated p-th percentile, as metrics.core.percentile(..., "linear") on non-empty values."""
    s = sorted(values)
    idx = p / 100 * (len(s) - 1)
    lo = int(idx)
    if lo + 1 >= len(s):
        return s[-1]
    return s[lo] * (1 - (idx - lo)) + s[lo + 1] * (idx - lo)


def summary_tpot_stats(rows):
    tpots = [r["tpot_ms"] for r in rows
             if r.get("success") and isinstance(r.get("tpot_ms"), (int, float))]
    if not tpots:
        return {}
    return {
        "mean_tpot_ms": statistics.mean(tpots),
        "median_tpot_ms": statistics.median(tpots),
        "p90_tpot_ms": percentile(tpots, 90),
        "p99_tpot_ms": percentile(tpots, 99),
    }


def per_turn_tpot_stats(rows):
    """turn_index -> {mean_tpot_ms, median_tpot_ms} from repaired rows."""
    by_turn = {}
    for r in rows:
        if not r.get("success") or not isinstance(r.get("tpot_ms"), (int, float)):
            continue
        by_turn.setdefault(int(r.get("turn_index") or 0), []).append(r["tpot_ms"])
    return {t: {"mean_tpot_ms": statistics.mean(v),
                "median_tpot_ms": statistics.median(v)}
            for t, v in by_turn.items()}


def repair_cell(path, apply, stamp):
    with open(path) as f:
        data = json.load(f)
    if "tpot_repair" in data:
        return {"status": "already_repaired", "rows_changed": 0}
    rows = data.get("per_request") or []
    changed = 0
    max_ratio = 1.0
    for row in rows:
        new = repaired_tpot(row)
        if new is None:
            continue
        old = row.get("tpot_ms")
        if old is not None and new > 0:
            max_ratio = max(max_ratio, old / new)
        if old != new:
            row["tpot_ms"] = new
            changed += 1
    if changed == 0:
        return {"status": "clean", "rows_changed": 0}

    data["summary"].update(summary_tpot_stats(rows))
    data["tpot_repair"] = stamp
    if apply:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    pt_path = path.replace(".json", "_per_turn.json")
    pt_changed = False
    if os.path.exists(pt_path):
        with open(pt_path) as f:
            pt = json.load(f)
        stats = per_turn_tpot_stats(rows)
        for item in pt.get("per_turn") or []:
            s = stats.get(int(item.get("turn_index") or 0))
            if s:
                item.update(s)
                pt_changed = True
        if pt_changed:
            pt["tpot_repair"] = stamp
            if apply:
                with open(pt_path, "w") as f:
                    json.dump(pt, f, indent=2)

    return {"status": "repaired", "rows_changed": changed,
            "max_ratio": round(max_ratio, 2), "per_turn_updated": pt_changed}


def repair_dir(dirpath, apply):
    stamp = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rule": RULE,
        "reason": REASON,
    }
    cells = sorted(f for f in glob.glob(os.path.join(dirpath, "*_conc*.json"))
                   if "_per_turn" not in f)
    report = {"dir": dirpath, "cells_scanned": len(cells), "cells_changed": 0,
              "rows_changed": 0, "max_ratio": 1.0, "already_repaired": 0}
    for c in cells:
        r = repair_cell(c, apply, stamp)
        if r["status"] == "repaired":
            report["cells_changed"] += 1
            report["rows_changed"] += r["rows_changed"]
            report["max_ratio"] = max(report["max_ratio"], r.get("max_ratio", 1.0))
        elif r["status"] == "already_repaired":
            report["already_repaired"] += 1
    if apply and report["cells_changed"]:
        manifest = dict(stamp, **{k: report[k] for k in
                                  ("cells_scanned", "cells_changed", "rows_changed",
                                   "max_ratio")})
        with open(os.path.join(dirpath, "_tpot_repair.json"), "w") as f:
            json.dump(manifest, f, indent=2)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+", help="result directories to scan")
    ap.add_argument("--apply", action="store_true",
                    help="write changes (default: dry run)")
    args = ap.parse_args()
    mode = "APPLY" if args.apply else "DRY-RUN"
    total_rows = 0
    print(f"[{mode}] {len(args.dirs)} dirs")
    for d in args.dirs:
        rep = repair_dir(d, args.apply)
        total_rows += rep["rows_changed"]
        print(f"  {rep['dir']:70s} cells {rep['cells_changed']:3d}/{rep['cells_scanned']:3d}"
              f"  rows {rep['rows_changed']:6d}  max_ratio {rep['max_ratio']:6.2f}"
              + (f"  (already {rep['already_repaired']})" if rep["already_repaired"] else ""))
    print(f"[{mode}] total rows changed: {total_rows}")


if __name__ == "__main__":
    main()
