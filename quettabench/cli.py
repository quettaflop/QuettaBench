"""qbench: the one command line.

  qbench run <run.yaml> [--set k=v ...] [--dry-run] [--results-dir DIR] [--allow-unverified]
             [--overwrite]
                                                      quettabench.runner
  qbench profile  --run results/<run> --source ...   quettabench.profiler
  qbench crossval ...                                 quettabench.crossval
  qbench export   --results ... --dest DIR            quettabench.export.quettasim
  qbench provision <run.yaml>... [--dry-run]          quettabench.provision

run is parsed here: --set is repeatable and its values are yaml scalars,
--results-dir is the parent of results/<run>/. A config the loader refuses,
or a value the engine adapter has no flag for (UnsupportedField, such as an
afd serving_style, which no adapter launches), exits 1 with that one-line
message; every config error shares exit 1. Every other door hands the rest of argv
unchanged to its module's main(argv); the module owns its own flags and
--help, and is imported only when chosen, so `qbench export` never imports
the profiler. `python3 -m quettabench.cli` works from a checkout without
installing.
"""

import argparse
import importlib
import sys

from quettabench import runner
from quettabench.config.schema import ConfigError
from quettabench.engines.base import UnsupportedField

# door -> module whose main(argv) takes the remaining arguments
DOORS = {
    "profile": "quettabench.profiler",
    "crossval": "quettabench.crossval",
    "export": "quettabench.export.quettasim.__main__",
    "provision": "quettabench.provision",
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
    """module.main(argv) for a door; a module without main() says so and exits."""
    main = getattr(importlib.import_module(DOORS[door]), "main", None)
    if main is None:
        sys.exit(f"qbench {door}: {DOORS[door]} has no main() yet")
    return main(argv)


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
