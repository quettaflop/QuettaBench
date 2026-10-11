"""Output arguments every kernel probe shares, the run record each writes,
and the one way a live-server probe starts vLLM (launch_vllm)."""
from __future__ import annotations

import argparse
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

from quettabench.config import env_map
from quettabench.config.schema import Config, EngineCfg, HardwareCfg, ModeCfg, WorkloadCfg
from quettabench.metrics.results import RunWriter, exit_status, redact

DEFAULT_OUT = Path("results") / "probes"


def add_output_args(ap: argparse.ArgumentParser) -> None:
    """Add --gpu-label and --out; --out defaults to results/probes/<gpu-label>,
    and each probe writes under its own <out>/<probe>/ (out_root)."""
    ap.add_argument("--gpu-label", "--gpu", required=True, help="GPU name the tables are keyed by, e.g. A100/H100")
    ap.add_argument("--out", "--out-dir", dest="out_dir", default=None,
                    help="parent of the probe's own <probe>/ dir (default: results/probes/<gpu-label>)")


def out_root(args: argparse.Namespace, name: str) -> Path:
    """<out>/<name>/: the dir a probe's tables and its run.json go under."""
    return (Path(args.out_dir) if args.out_dir else DEFAULT_OUT / args.gpu_label) / name


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
    known, _ = ap.parse_known_args(argv)
    if not known.gpu_label or known.out_dir == "-":
        return None
    return out_root(known, name)


def run_recorded(main, name: str, file_flag: str | None = None) -> None:
    """main() with its run recorded: <record_dir>/run.json through RunWriter,
    holding the probe and its argv as config (mode=probe), provenance and
    run_status ok, or failed (aborted on Ctrl-C) with the exception, which is
    raised on. Under torchrun only RANK 0 records, whatever its main does;
    a main that returns False writes nothing either."""
    argv = sys.argv[1:]
    run_dir = record_dir(name, argv, file_flag) if env_map.torchrun_rank() in (None, 0) else None
    started = datetime.now(timezone.utc).isoformat()
    primary, status, reason = None, "ok", None
    try:
        primary = main()
    except BaseException as e:
        status, reason = exit_status(e)
        raise
    finally:
        if run_dir is not None and primary is not False:
            cfg = Config(EngineCfg(), WorkloadCfg(), ModeCfg(mode="probe", probe=name,
                                                             probe_args=shlex.join(argv)),
                         HardwareCfg(), run_name=f"probe-{name}")
            writer = RunWriter(run_dir, cfg)
            writer.provenance["started"] = started
            writer.finish({"probe": name, "argv": redact(argv)}, status=status, reason=reason)


def launch_vllm(run_dir: Path, model: str, *, port: int, max_model_len: int, prefix_caching: bool,
                tp: int = 1, gpu_util: float = 0.9, vllm_flags: dict | None = None,
                timeout_s: int = 1800):
    """Start vLLM as `qbench run` does: a Config through the vllm adapter's
    translate, then launcher.start into run_dir (engine.json, server.log).
    vllm_flags (served_model_name, api_key, ...) go in as engine.extra.vllm.
    The server is this python's venv vllm. Returns the Handle; the caller
    stops it."""
    # Imported here: base.py ships to remote kernel-probe hosts, which have
    # no engine stack; only the live-server probes, run locally, launch.
    from quettabench.engines import launcher, vllm

    engine = EngineCfg(model=model, tp=tp, dtype="bfloat16", port=port, gpu_util=gpu_util,
                       max_model_len=max_model_len, prefix_caching=prefix_caching,
                       extra={"vllm": dict(vllm_flags or {})})
    cfg = Config(engine, WorkloadCfg(), ModeCfg(), HardwareCfg(provision={"py": sys.executable}),
                 run_name=Path(run_dir).name)
    spec = vllm.ADAPTER.translate(cfg, run_dir)
    return launcher.start(spec, run_dir, timeout_s=timeout_s)
