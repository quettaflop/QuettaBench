"""qbench CLI: `run` is parsed here, other commands get the rest of argv."""

import argparse
import importlib
import sys

from quettabench import runner
from quettabench.config.schema import ConfigError
from quettabench.engines.base import UnsupportedField

SUBCOMMANDS = {
    "profile": "quettabench.profiler",
    "compare": "quettabench.compare",
    "export": "quettabench.export.quettasim",
}


def parser():
    ap = argparse.ArgumentParser(
        prog="qbench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="subcommand", metavar="{run," + ",".join(SUBCOMMANDS) + "}")
    run = sub.add_parser("run", help="quettabench.runner: one run yaml to results/<run>/")
    run.add_argument("run_yaml", help="configs/runs/<name>.yaml")
    run.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="K=V",
        help="override one Config path, highest precedence (repeatable)",
    )
    run.add_argument(
        "--dry-run", action="store_true", help="print each Config with its origins and LaunchSpec; start nothing"
    )
    run.add_argument("--results-dir", default="results", help="parent of <run>/ (default results)")
    run.add_argument(
        "--allow-unverified", action="store_true", help="run a model whose yaml says verified: false (bring-up only)"
    )
    run.add_argument(
        "--overwrite", action="store_true", help="empty a results/<run>/ that already holds a run before running"
    )
    for subcommand, module in SUBCOMMANDS.items():
        sub.add_parser(subcommand, help=f"{module}.main", add_help=False)
    return ap


def delegate(subcommand, argv):
    return importlib.import_module(SUBCOMMANDS[subcommand]).main(argv)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in SUBCOMMANDS:
        return delegate(argv[0], argv[1:])
    ap = parser()
    a = ap.parse_args(argv)
    if a.subcommand != "run":
        ap.print_help()
        return 2
    try:
        return runner.main(a.run_yaml, a.overrides, a.dry_run, a.results_dir, a.allow_unverified, a.overwrite)
    except (ConfigError, UnsupportedField) as e:
        sys.exit(str(e))


if __name__ == "__main__":
    sys.exit(main())
