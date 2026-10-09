"""qbench: the one command line.

  qbench profile  --run results/<run> --source ...   quettabench.profiler
  qbench crossval ...                                 quettabench.crossval
  qbench export   --results ... --dest DIR            quettabench.export.quettasim

Each door hands the rest of argv unchanged to its module's main(argv); the
module owns its own flags and --help. Doors are imported only when chosen,
so `qbench export` never imports the profiler. `python3 -m quettabench.cli`
works from a checkout without installing.
"""

import argparse
import importlib
import sys

# door -> module whose main(argv) takes the remaining arguments
DOORS = {
    "profile": "quettabench.profiler",
    "crossval": "quettabench.crossval",
    "export": "quettabench.export.quettasim.__main__",
}


def parser():
    """The top-level help: one line per door."""
    ap = argparse.ArgumentParser(prog="qbench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="door", metavar="{" + ",".join(DOORS) + "}")
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
    parser().parse_args(argv)
    parser().print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
