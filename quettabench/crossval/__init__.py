"""crossval <ours> <theirs>: compare two measured runs, reading files only.

Each side is a results/<run>/ dir, an old baselines/*.json, or an engine
bench log, told apart by content (a dir, a json object, anything else).
Prints the comparison text (crossval/compare.py, the old table.py output)
and writes the Report as json to results/xval/<ours>_vs_<theirs>.json, or
to --json OUT. Agreement rows appear when both sides are run dirs whose
rows returned text or logits; --floor adds the slice BAND verdict
(crossval/agreement.py). Exits 1, after the json is written, when the
FIDELITY or the BAND verdict is FAIL.

The facts a bench log does not state (NCCL pins, quant, serving style) and
XVAL_ALLOW_METHOD_MISMATCH come from the environment through env_map.
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

from quettabench.config import env_map
from quettabench.crossval import agreement, compare, reader

REPO = Path(__file__).resolve().parents[2]
XVAL_DIR = REPO / "results" / "xval"
ENV = ("XVAL_QUANT", "XVAL_SERVING_STYLE", "NCCL_ALGO", "NCCL_PROTO",
       "XVAL_ALLOW_METHOD_MISMATCH")


def read_side(path, values):
    """The Side at path: run dir, baseline json, or bench log."""
    path = Path(path)
    if path.is_dir():
        return reader.read_run(path)
    if path.read_text().lstrip().startswith("{"):
        return reader.read_baseline(path)
    return reader.read_bench_log(path, reader.engine_facts(values))


def report_json(report):
    """The Report as plain json-able data."""
    return json.loads(json.dumps(dataclasses.asdict(report), default=str))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="crossval", description=__doc__.splitlines()[0])
    ap.add_argument("ours", help="results/<run>/ dir, baselines/*.json, or bench log")
    ap.add_argument("theirs", help="results/<run>/ dir, baselines/*.json, or bench log")
    ap.add_argument("--allow-method-mismatch", action="store_true",
                    help="ratios even when the two sides time different quantities")
    ap.add_argument("--floor", action="append", default=[], metavar="RUN",
                    help="a reference run at the engine's dtype; theirs is then the anchor "
                         "and ours passes BAND within the worst floor run's argmax flips")
    ap.add_argument("--json", type=Path, default=None,
                    help="report path (default results/xval/<ours>_vs_<theirs>.json)")
    args = ap.parse_args(argv)
    for path in (args.ours, args.theirs, *args.floor):
        if not Path(path).exists():
            sys.exit(f"no results at {path}")
    if args.floor and not all(Path(p).is_dir() for p in (args.ours, args.theirs, *args.floor)):
        sys.exit("--floor compares logits runs: ours, theirs and every floor must be run dirs")

    values = env_map.read_env(only=ENV)
    ours = read_side(args.ours, values)
    if not ours.rows and not Path(args.ours).is_dir():
        print("".join(n + "\n" for n in ours.notes), end="")
        sys.exit(f"no engine rows parsed from {args.ours}")
    theirs = read_side(args.theirs, values)
    allow = args.allow_method_mismatch or values.get("crossval.allow_method_mismatch", False)
    report = compare.pair(ours, theirs, allow, now=time.time())
    if Path(args.ours).is_dir() and Path(args.theirs).is_dir():
        ours_rows, theirs_rows = reader.read_rows(args.ours), reader.read_rows(args.theirs)
        report.agreement = agreement.compare_runs(ours_rows, theirs_rows)
        if args.floor:
            report.agreement.append(agreement.band(
                ours_rows, theirs_rows, [reader.read_rows(f) for f in args.floor]))
    print(compare.render(report), end="")

    a, b = (Path(p).name if Path(p).is_dir() else Path(p).stem for p in (args.ours, args.theirs))
    out = args.json or XVAL_DIR / f"{a}_vs_{b}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report_json(report), indent=2) + "\n")
    if not report.rows and not report.agreement:
        sys.exit("no overlapping (ctx, bs, tp) points")
    if any(r["kind"] in ("logit_summary", "band") and r["verdict"] == "FAIL"
           for r in report.agreement):
        sys.exit(1)
