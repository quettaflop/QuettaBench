"""Output arguments every kernel probe shares."""
from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_OUT = Path("results") / "probes"


def add_output_args(ap: argparse.ArgumentParser) -> None:
    """Add --gpu-label and --out; --out defaults to results/probes/<gpu-label>."""
    ap.add_argument("--gpu-label", "--gpu", required=True, help="GPU name the tables are keyed by, e.g. A100/H100")
    ap.add_argument("--out", "--out-dir", dest="out_dir", default=None,
                    help="directory the grids are written under (default: results/probes/<gpu-label>)")


def out_root(args: argparse.Namespace) -> Path:
    return Path(args.out_dir) if args.out_dir else DEFAULT_OUT / args.gpu_label
