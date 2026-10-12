"""python3 -m quettabench.export.quettasim --results results/<run> ... --dest <dir>
[--kernel-dest <QuettaSim data root> [--replace-tables]]"""
from __future__ import annotations

import argparse
from pathlib import Path

from quettabench.export.quettasim import BUILDERS, export


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="qbench export", description=__doc__)
    ap.add_argument("--results", type=Path, nargs="+", default=[Path("results")],
                    help="results dirs the builders read; a named table comes from the ok run that "
                         "finished last, a tie from the earlier dir")
    ap.add_argument("--dest", type=Path,
                    help="where every builder but kernel_tables writes (kernels/, results/, "
                         "manifest.json); those builders run only when it is given")
    ap.add_argument("--kernel-dest", type=Path,
                    help="QuettaSim's data root ($QUETTASIM_DATA, e.g. ../QuettaSim/data): "
                         "kernel_tables writes its kernel_data/ (with qbench_manifest.json) there, "
                         "and runs only when given")
    ap.add_argument("--replace-tables", action="store_true",
                    help="replace the kernel tables (and sidecars) --kernel-dest holds where merging "
                         "into them would stop, dropping their rows; the others still merge")
    ap.add_argument("--only", nargs="+", choices=sorted(BUILDERS),
                    help="run just these builders (default: all)")
    a = ap.parse_args(argv)
    for path in export(a.results, a.dest, a.only, kernel_dest=a.kernel_dest,
                       replace_tables=a.replace_tables):
        print(path)


if __name__ == "__main__":
    main()
