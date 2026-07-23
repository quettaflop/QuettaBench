#!/usr/bin/env python3
"""Offline repair of the inter-chunk tpot bug in stored benchmark results.

gpt-oss on vLLM streams reasoning tokens as content-less deltas; the old
client skipped them, so mean(itl_ms) measured inter-CHUNK gaps and inflated
per-request tpot_ms by up to ~13x (QuettaSim tools/GT_QUALITY_FLAGS.md,
Finding 1). The wall-clock quantities (e2el_ms, ttft_ms, output_tokens) were
measured correctly, so the derived tpot is repairable offline.

Repair rule (same coverage guard as the fixed metrics.RequestResult.tpot):
for each successful request with output_tokens > 1, non-empty itl_ms, and
len(itl_ms) < 0.8 * (output_tokens - 1):
    tpot_ms := (e2el_ms - ttft_ms) / (output_tokens - 1)

Everything else in the file is left byte-identical. When any row changes,
the summary tpot stats and the *_per_turn.json sibling's per-turn tpot
fields are recomputed from the repaired rows, and a top-level
"tpot_repair" provenance stamp is added to both files. Untouched files get
no stamp. A per-directory _tpot_repair.json manifest records counts.

Usage:
    python3 scripts/repair_tpot_coverage.py DIR [DIR ...]          # dry run
    python3 scripts/repair_tpot_coverage.py --apply DIR [DIR ...]
"""

import argparse
import glob
import json
import os
import statistics
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.benchmark.metrics import _percentile  # noqa: E402

COVERAGE_THRESHOLD = 0.8
RULE = "len(itl)<0.8*(out-1) -> tpot=(e2el-ttft)/(out-1)"
REASON = "QuettaSim tools/GT_QUALITY_FLAGS.md Finding 1 (inter-chunk tpot)"


def repaired_tpot(row):
    """Return the corrected tpot_ms for a row the guard fires on, else None."""
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


def summary_tpot_stats(rows):
    tpots = [r["tpot_ms"] for r in rows
             if r.get("success") and isinstance(r.get("tpot_ms"), (int, float))]
    if not tpots:
        return {}
    return {
        "mean_tpot_ms": statistics.mean(tpots),
        "median_tpot_ms": statistics.median(tpots),
        "p90_tpot_ms": _percentile(tpots, 90),
        "p99_tpot_ms": _percentile(tpots, 99),
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

    # Sibling per-turn aggregate file, if present.
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
    ap.add_argument("dirs", nargs="+")
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
