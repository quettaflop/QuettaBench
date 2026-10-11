"""Output arguments every kernel probe shares, and the run record each writes."""
from __future__ import annotations

import argparse
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

from quettabench.config.schema import Config, EngineCfg, HardwareCfg, ModeCfg, WorkloadCfg
from quettabench.metrics.results import RunWriter

DEFAULT_OUT = Path("results") / "probes"


def add_output_args(ap: argparse.ArgumentParser) -> None:
    """Add --gpu-label and --out; --out defaults to results/probes/<gpu-label>."""
    ap.add_argument("--gpu-label", "--gpu", required=True, help="GPU name the tables are keyed by, e.g. A100/H100")
    ap.add_argument("--out", "--out-dir", dest="out_dir", default=None,
                    help="directory the grids are written under (default: results/probes/<gpu-label>)")


def out_root(args: argparse.Namespace) -> Path:
    return Path(args.out_dir) if args.out_dir else DEFAULT_OUT / args.gpu_label


def record_dir(name: str, argv: list[str], file_flag: str | None = None) -> Path | None:
    """Where a probe's run.json goes: <out_root>/<name>/, or with file_flag
    (a probe that names its output file) <that file's dir>/<name>/. None when
    argv names no output (no --gpu-label, or --out -)."""
    ap = argparse.ArgumentParser(add_help=False)
    if file_flag:
        ap.add_argument(file_flag, dest="file")
        known, _ = ap.parse_known_args(argv)
        return Path(known.file).parent / name if known.file else None
    ap.add_argument("--gpu-label", "--gpu")
    ap.add_argument("--out", "--out-dir", dest="out_dir")
    known, _ = ap.parse_known_args(argv)
    if not known.gpu_label or known.out_dir == "-":
        return None
    return out_root(known) / name


def run_recorded(main, name: str, file_flag: str | None = None) -> None:
    """main() with its run recorded: <record_dir>/run.json through RunWriter,
    holding the probe and its argv as config (mode=probe), provenance and
    run_status ok, or failed (aborted on Ctrl-C) with the exception, which is
    raised on. A main that returns False (a torchrun rank other than 0)
    writes nothing."""
    argv = sys.argv[1:]
    run_dir = record_dir(name, argv, file_flag)
    started = datetime.now(timezone.utc).isoformat()
    primary, status, reason = None, "ok", None
    try:
        primary = main()
    except SystemExit as e:
        if e.code not in (None, 0):
            status, reason = "failed", str(e.code)
        raise
    except KeyboardInterrupt:
        status, reason = "aborted", "interrupted"
        raise
    except Exception as e:
        status, reason = "failed", f"{type(e).__name__}: {e}"
        raise
    finally:
        if run_dir is not None and primary is not False:
            cfg = Config(EngineCfg(), WorkloadCfg(), ModeCfg(mode="probe", probe=name,
                                                             probe_args=shlex.join(argv)),
                         HardwareCfg(), run_name=f"probe-{name}")
            writer = RunWriter(run_dir, cfg)
            writer.provenance["started"] = started
            writer.finish({"probe": name, "argv": argv}, status=status, reason=reason)
