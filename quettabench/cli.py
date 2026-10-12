"""qbench: the one command line. `run` is parsed here; profile, xval and export pass the rest of argv to that module's main(argv).

Config errors exit 1 with a one-line message."""

import argparse
import importlib
import sys

from quettabench import runner
from quettabench.config.schema import ConfigError
from quettabench.engines.base import UnsupportedField

DOORS = {
    "profile": "quettabench.profiler",
    "xval": "quettabench.crossval",
    "export": "quettabench.export.quettasim",
}


def parser():
    """run's flags, and one help line per delegated door."""
    ap = argparse.ArgumentParser(prog="qbench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="door", metavar="{run," + ",".join(DOORS) + "}")
    run = sub.add_parser("run", help="quettabench.runner: one run yaml to results/<run>/")
    run.add_argument("run_yaml", help="configs/runs/<name>.yaml")
    run.add_argument("--set", dest="overrides", action="append", default=[], metavar="K=V",
                     help="override one Config path, highest precedence (repeatable)")
    run.add_argument("--dry-run", action="store_true",
                     help="print each Config with its origins and LaunchSpec; start nothing")
    run.add_argument("--results-dir", default="results", help="parent of <run>/ (default results)")
    run.add_argument("--allow-unverified", action="store_true",
                     help="run a model whose yaml says verified: false (bring-up only)")
    run.add_argument("--overwrite", action="store_true",
                     help="empty a results/<run>/ that already holds a run before running")
    for door, module in DOORS.items():
        sub.add_parser(door, help=f"{module}.main", add_help=False)
    return ap


def delegate(door, argv):
    """module.main(argv) for a door."""
    return importlib.import_module(DOORS[door]).main(argv)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in DOORS:
        return delegate(argv[0], argv[1:])
    ap = parser()
    a = ap.parse_args(argv)
    if a.door != "run":
        ap.print_help()
        return 2
    try:
        return runner.main(a.run_yaml, a.overrides, a.dry_run, a.results_dir,
                           a.allow_unverified, a.overwrite)
    except (ConfigError, UnsupportedField) as e:
        sys.exit(str(e))


if __name__ == "__main__":
    sys.exit(main())
