#!/usr/bin/env python3
"""Export qbench per-op probe measurements as an LLMServingSim profiler tree.

The sim consumes profiler/perf/<HW>/<org>/<model>/<variant>/tp<N>/{dense.csv,
attention.csv, per_sequence.csv, moe.csv, skew_fit.csv} plus a variant-level
meta.yaml. This compiles qbench probe records into that exact contract, with
coverage checked against a reference tree's key set: any cell the probes do
not cover is printed as MISSING (full list in a gaps file) and the export
exits 2. Nothing is interpolated and no reference time is ever copied; a
silently filled cell would poison the sim-vs-live comparison these tables
exist to make, so gaps must fail loudly here.

Probe interchange, one JSON object per line, merged across --probes files:
  {"table":"dense","tp":1,"layer":"act_fn","tokens":7,"time_us":3.34}
  {"table":"attention","tp":1,"prefill_chunk":0,"kv_prefill":0,
   "n_decode":1,"kv_decode":16,"time_us":11.6}
  {"table":"per_sequence","tp":1,"layer":"lm_head","sequences":4,"time_us":250.1}
  {"table":"moe","tp":1,"tokens":64,"activated_experts":8,"time_us":88.0}
  {"table":"skew_fit","tp":1,"pc":0,"n_label":"n<=128","skew_rate_label":"sr<=15%",
   "kv_big_label":"kvB<=1k","kp_label":"kp=0","alpha":0.1252,"n_samples":4}
Duplicate keys take the median. vmin_fit slope grids are aggregates and can
not be decomposed into per-op cells, so they are deliberately not read.
"""

import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

# table -> (key columns, value columns), matching the sim's CSV headers.
TABLES = {
    "dense": (("layer", "tokens"), ("time_us",)),
    "attention": (("prefill_chunk", "kv_prefill", "n_decode", "kv_decode"), ("time_us",)),
    "per_sequence": (("layer", "sequences"), ("time_us",)),
    "moe": (("tokens", "activated_experts"), ("time_us",)),
    "skew_fit": (("pc", "n_label", "skew_rate_label", "kv_big_label", "kp_label"),
                 ("alpha", "n_samples")),
}
MISSING_STDOUT_CAP = 40


def read_reference(ref_dir, tps=None):
    """Required cells per (tp, table): the reference tree's exact key tuples,
    in reference row order so exports diff cleanly against it."""
    ref_dir = Path(ref_dir)
    out = {}
    for tp_dir in sorted(ref_dir.glob("tp*")):
        tp = int(tp_dir.name[2:])
        if tps and tp not in tps:
            continue
        for name, (keys, _) in TABLES.items():
            p = tp_dir / f"{name}.csv"
            if not p.exists():
                continue
            with open(p) as fh:
                rows = list(csv.DictReader(fh))
            out[(tp, name)] = [tuple(str(r[k]) for k in keys) for r in rows]
    if not out:
        sys.exit(f"no tp*/<table>.csv under reference {ref_dir}")
    return out


def load_probes(paths):
    """Merge probe JSONL files into {(tp, table, keytuple): {value_col: median}}."""
    acc = {}
    for path in paths:
        for ln, line in enumerate(open(path), 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            name = rec.get("table")
            if name not in TABLES:
                sys.exit(f"{path}:{ln}: unknown table {name!r}")
            keys, vals = TABLES[name]
            try:
                kt = tuple(str(rec[k]) for k in keys)
                vv = {v: float(rec[v]) for v in vals}
            except KeyError as e:
                sys.exit(f"{path}:{ln}: {name} record missing field {e}")
            acc.setdefault((int(rec["tp"]), name, kt), []).append(vv)
    return {k: {v: statistics.median(x[v] for x in xs) for v in xs[0]}
            for k, xs in acc.items()}


def _fmt(v):
    return f"{int(v)}" if float(v).is_integer() else f"{v:g}"


def export(required, probes, out_variant_dir):
    """Write CSVs for every fully covered (tp, table); return the missing cells."""
    missing = []
    for (tp, name), cells in required.items():
        keys, vals = TABLES[name]
        rows, holes = [], []
        for kt in cells:
            hit = probes.get((tp, name, kt))
            if hit is None:
                holes.append(kt)
            else:
                rows.append(list(kt) + [_fmt(hit[v]) for v in vals])
        if holes:
            missing.extend((tp, name, keys, kt) for kt in holes)
            continue
        d = Path(out_variant_dir) / f"tp{tp}"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{name}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(list(keys) + list(vals))
            w.writerows(rows)
    return missing


def write_meta(ref_dir, out_variant_dir, probe_paths, tps, first_token_overhead_us):
    """meta.yaml: the reference's engine/grid contract with qbench provenance.
    The sim reads grid fields from here; dropping them breaks its lookup, so
    they are copied while every provenance field is replaced."""
    import yaml
    ref = yaml.safe_load((Path(ref_dir) / "meta.yaml").read_text())
    ref["profiler_version"] = "qbench-lss-export 1.0"
    ref["profiled_at"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    ref["source"] = "qbench"
    ref["tp_degrees"] = sorted(tps)
    ref["probe_files"] = [
        {"path": str(p), "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()}
        for p in probe_paths
    ]
    if first_token_overhead_us is not None:
        ref["first_token_overhead_us"] = float(first_token_overhead_us)
    (Path(out_variant_dir) / "meta.yaml").write_text(yaml.safe_dump(ref, sort_keys=False))


def main():
    ap = argparse.ArgumentParser(description="qbench probes -> LLMServingSim perf tree")
    ap.add_argument("--probes", nargs="*", default=[], help="probe JSONL files")
    ap.add_argument("--reference", required=True,
                    help="reference variant dir (defines the required key set)")
    ap.add_argument("--out", required=True, help="output root (profiler/perf tree)")
    ap.add_argument("--tp", type=int, action="append",
                    help="restrict to these tp degrees (default: all in reference)")
    ap.add_argument("--first-token-overhead-us", type=float,
                    help="additive per-request TTFT constant from lss_first_token.py")
    args = ap.parse_args()

    ref = Path(args.reference)
    hw, org, model, variant = ref.parts[-4:]
    required = read_reference(ref, tps=set(args.tp) if args.tp else None)
    probes = load_probes(args.probes)
    out_variant = Path(args.out) / "profiler" / "perf" / hw / org / model / variant
    missing = export(required, probes, out_variant)
    if missing:
        gaps = Path(args.out) / "gaps.txt"
        gaps.parent.mkdir(parents=True, exist_ok=True)
        with open(gaps, "w") as fh:
            for tp, name, keys, kt in missing:
                fh.write(f"MISSING table={name} tp={tp} "
                         + " ".join(f"{k}={v}" for k, v in zip(keys, kt)) + "\n")
        for tp, name, keys, kt in missing[:MISSING_STDOUT_CAP]:
            print(f"MISSING table={name} tp={tp} "
                  + " ".join(f"{k}={v}" for k, v in zip(keys, kt)))
        print(f"GAPS {len(missing)} uncovered cells; full list in {gaps}")
        print("fill them with the qbench kernel_composed probes "
              "(feat/quettasim-probes); this tool never interpolates")
        sys.exit(2)
    tps = sorted({tp for (tp, _n) in required})
    write_meta(ref, out_variant, args.probes, tps, args.first_token_overhead_us)
    n = sum(len(c) for c in required.values())
    print(f"EXPORTED {out_variant} tables={len(required)} cells={n} tp={tps}")


if __name__ == "__main__":
    main()
