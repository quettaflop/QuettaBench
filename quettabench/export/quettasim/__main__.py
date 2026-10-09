"""python3 -m quettabench.export.quettasim --results results/<run> ... --dest <dir>"""
from __future__ import annotations

import argparse
from pathlib import Path

from quettabench.export.quettasim import BUILDERS, export


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="qbench export", description=__doc__)
    ap.add_argument("--results", type=Path, nargs="+", default=[Path("results")],
                    help="results dirs the builders read, later dirs searched after earlier ones")
    ap.add_argument("--dest", type=Path, required=True,
                    help="where the QuettaSim files land (kernels/, results/, manifest.json)")
    ap.add_argument("--only", nargs="+", choices=sorted(BUILDERS),
                    help="run just these builders (default: all)")
    a = ap.parse_args(argv)
    for path in export(a.results, a.dest, a.only):
        print(path)


if __name__ == "__main__":
    main()
