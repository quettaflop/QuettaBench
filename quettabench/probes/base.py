"""Output arguments every kernel probe shares, the run record each writes,
and the one way a live-server probe starts vLLM (launch_vllm)."""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

from quettabench.config import env_map
from quettabench.config.schema import Config, EngineCfg, HardwareCfg, ModeCfg, WorkloadCfg
from quettabench.metrics.results import RunWriter, exit_status, redact

DEFAULT_OUT = Path("results") / "probes"
# A run id: the run's start in UTC to the microsecond, so a probe's run dirs sort by start.
RUN_ID = re.compile(r"\d{8}T\d{6}\.\d{6}Z")
# The Config of the `qbench run` (mode: probe) running this probe, if any.
_RUN_CFG: Config | None = None


@contextlib.contextmanager
def run_config(cfg: Config):
    """Hand the run's Config to launch_vllm for the duration of a probe's main."""
    global _RUN_CFG
    _RUN_CFG = cfg
    try:
        yield
    finally:
        _RUN_CFG = None


def add_output_args(ap: argparse.ArgumentParser) -> None:
    """Add --gpu-label, --out and --run-id; --out defaults to
    results/probes/<gpu-label>, and each probe writes under its own
    <out>/<probe>/<run-id>/ (out_root)."""
    ap.add_argument("--gpu-label", "--gpu", required=True, help="GPU name the tables are keyed by, e.g. A100/H100")
    ap.add_argument("--out", "--out-dir", dest="out_dir", default=None,
                    help="parent of the probe's own <probe>/ dir (default: results/probes/<gpu-label>)")
    ap.add_argument("--run-id", default=None,
                    help="this run's own dir under <out>/<probe>/; the probe CLI sets it to the "
                         "start time, and without it the tables go into <out>/<probe>/ itself")


def run_id(now: datetime) -> str:
    """The run dir name for a run started at now (RUN_ID)."""
    return now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def out_root(args: argparse.Namespace, name: str) -> Path:
    """<out>/<name>/<run-id>/: the dir a probe's tables and its run.json go
    under; <out>/<name>/ when no run id is given (a `qbench run` probe, whose
    --out is already its own run dir)."""
    root = (Path(args.out_dir) if args.out_dir else DEFAULT_OUT / args.gpu_label) / name
    run = getattr(args, "run_id", None)
    if run is None:
        return root
    if not RUN_ID.fullmatch(run):
        raise SystemExit(f"--run-id {run!r} is not a run id like 20261011T120000.000000Z")
    return root / run


def run_ok(run_dir: Path) -> bool:
    """Whether run_dir/run.json records an ok run."""
    try:
        return json.loads((run_dir / "run.json").read_text())["run_status"]["status"] == "ok"
    except (OSError, ValueError, KeyError, TypeError):
        return False


def prior_table(args: argparse.Namespace, name: str, path: Path) -> Path | None:
    """The table a probe that upserts merges its new rows into: path itself
    when there is no run id, else the same file in this probe's newest
    earlier run dir whose run.json is ok, so a failed or unfinished run's
    rows never carry forward. None when there is none."""
    root = out_root(args, name)
    if getattr(args, "run_id", None) is None:
        return path if path.exists() else None
    rel = Path(path).relative_to(root)
    earlier = sorted((d for d in root.parent.glob("*") if RUN_ID.fullmatch(d.name)
                      and d.name < args.run_id), reverse=True)
    return next((d / rel for d in earlier if (d / rel).is_file() and run_ok(d)), None)


def record_dir(name: str, argv: list[str], file_flag: str | None = None) -> Path | None:
    """Where a probe's run.json goes: out_root, the dir its tables are written
    under, or with file_flag (a probe that names its output file) that file's
    dir. None when argv names no output (no --gpu-label, or --out -)."""
    ap = argparse.ArgumentParser(add_help=False)
    if file_flag:
        ap.add_argument(file_flag, dest="file")
        known, _ = ap.parse_known_args(argv)
        return Path(known.file).parent if known.file else None
    ap.add_argument("--gpu-label", "--gpu")
    ap.add_argument("--out", "--out-dir", dest="out_dir")
    ap.add_argument("--run-id")
    known, _ = ap.parse_known_args(argv)
    if not known.gpu_label or known.out_dir == "-":
        return None
    return out_root(known, name)


def run_recorded(main, name: str, file_flag: str | None = None, now: datetime | None = None) -> None:
    """main(argv) with its run recorded: <record_dir>/run.json through RunWriter,
    holding the probe and its argv as config (mode=probe), provenance and
    run_status ok, or failed (aborted on Ctrl-C) with the exception, which is
    raised on. A probe without file_flag gets --run-id from now (the start,
    taken here once when not given), so each run has its own dir. Under
    torchrun only RANK 0 records, whatever its main does; a main that returns
    False writes nothing either."""
    now = now or datetime.now(timezone.utc)
    argv = sys.argv[1:] if file_flag else ["--run-id", run_id(now), *sys.argv[1:]]
    run_dir = record_dir(name, argv, file_flag) if env_map.torchrun_rank() in (None, 0) else None
    primary, status, reason = None, "ok", None
    try:
        primary = main(argv)
    except BaseException as e:
        status, reason = exit_status(e)
        raise
    finally:
        if run_dir is not None and primary is not False:
            cfg = Config(EngineCfg(), WorkloadCfg(), ModeCfg(mode="probe", probe=name,
                                                             probe_args=shlex.join(argv)),
                         HardwareCfg(), run_name=f"probe-{name}")
            writer = RunWriter(run_dir, cfg)
            writer.provenance["started"] = now.isoformat()
            writer.finish({"probe": name, "argv": redact(argv)}, status=status, reason=reason)


def launch_vllm(run_dir: Path, model: str, *, port: int, max_model_len: int, prefix_caching: bool,
                tp: int = 1, gpu_util: float = 0.9, vllm_flags: dict | None = None,
                timeout_s: int = 1800):
    """Start vLLM as `qbench run` does: a Config through the vllm adapter's
    translate, then launcher.start into run_dir (engine.json, server.log).
    vllm_flags (served_model_name, api_key, ...) go in as engine.extra.vllm.
    The hardware is the run's (run_config) when a `qbench run` started the
    probe, else the defaults; the server is this python's venv vllm unless
    that hardware names a provision.py. Returns the Handle; the caller
    stops it."""
    # Imported here: base.py ships to remote kernel-probe hosts, which have
    # no engine stack; only the live-server probes, run locally, launch.
    from quettabench.engines import launcher, vllm

    engine = EngineCfg(model=model, tp=tp, dtype="bfloat16", port=port, gpu_util=gpu_util,
                       max_model_len=max_model_len, prefix_caching=prefix_caching,
                       extra={"vllm": dict(vllm_flags or {})})
    hardware = _RUN_CFG.hardware if _RUN_CFG else HardwareCfg()
    hardware = dataclasses.replace(hardware, provision={"py": sys.executable, **hardware.provision})
    cfg = Config(engine, WorkloadCfg(), ModeCfg(), hardware, run_name=Path(run_dir).name)
    spec = vllm.ADAPTER.translate(cfg, run_dir)
    return launcher.start(spec, run_dir, timeout_s=timeout_s)
