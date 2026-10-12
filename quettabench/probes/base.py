"""What every kernel probe shares: ProbeRow, the Probe ABC, the output arguments, the run record, launch_vllm for live-server probes, and remote_probe's ssh and rsync."""

from __future__ import annotations

import abc
import argparse
import contextlib
import dataclasses
import re
import shlex
import subprocess
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from quettabench.config import env_map
from quettabench.config.schema import Config, EngineCfg, HardwareCfg, ModeCfg, WorkloadCfg
from quettabench.metrics.results import RunWriter, exit_status, redact, stamp_pulled

DEFAULT_OUT = Path("results") / "probes"
# Run start in UTC to the microsecond, so a probe's run dirs sort by start.
RUN_ID = re.compile(r"\d{8}T\d{6}\.\d{6}Z")
# QuettaSim's mode words; export maps each to kernel_data's graph/ or eager/.
TIMINGS = ("graph_replay", "ncu", "amortized_wallclock", "eager", "host_wallclock")
# The Config of the `qbench run` (mode: probe) running this probe, if any.
_RUN_CFG: Config | None = None
# The writer of the run recording this probe (run_recorded or `qbench run`), if any.
_WRITER: RunWriter | None = None


@dataclasses.dataclass(frozen=True)
class ProbeRow:
    """One measured row of a QuettaSim table: table is <kind>/<file> (gemm/H200.csv), columns are key then value, timing picks the dir. A JSON table's row is key {"entry": name}, value {"value": ...}."""

    probe: str
    table: str
    key: dict
    value: dict
    timing: str

    def __post_init__(self):
        if self.timing not in TIMINGS:
            raise ValueError(f"{self.probe}: timing {self.timing!r} is not one of {TIMINGS}")


def emit(row: ProbeRow) -> None:
    """Append row to the recording run's rows.jsonl."""
    if _WRITER is None:
        raise RuntimeError(
            f"{row.probe}: no run records its rows; run it as a probe (python -m, just probe) or through qbench run"
        )
    _WRITER.probe_row(row)


class Probe(abc.ABC):
    """grid() yields points, run_point() times one into a ProbeRow (None to skip), run() emits each row as measured and returns the count."""

    @abc.abstractmethod
    def grid(self) -> Iterable:
        """The points to time, in order."""

    @abc.abstractmethod
    def run_point(self, point) -> ProbeRow | None:
        """One point's row, or None when it is skipped."""

    def run(self) -> int:
        n = 0
        for point in self.grid():
            row = self.run_point(point)
            if row is not None:
                emit(row)
                n += 1
        return n


@contextlib.contextmanager
def run_config(cfg: Config, writer: RunWriter | None = None):
    """Hand the run's Config to launch_vllm, and its writer to emit, for the duration of a probe's main."""
    global _RUN_CFG, _WRITER
    _RUN_CFG, _WRITER = cfg, writer
    try:
        yield
    finally:
        _RUN_CFG, _WRITER = None, None


def add_output_args(ap: argparse.ArgumentParser) -> None:
    """Add --gpu-label, --out (default results/probes/<gpu-label>) and --run-id; each run records under <out>/<probe>/<run-id>/. Flags are taken only as written, so export reads probe_args as the probe did."""
    ap.allow_abbrev = False
    ap.add_argument("--gpu-label", "--gpu", required=True, help="GPU name the tables are keyed by, e.g. A100/H100")
    ap.add_argument(
        "--out",
        "--out-dir",
        dest="out_dir",
        default=None,
        help="parent of the probe's own <probe>/ dir (default: results/probes/<gpu-label>)",
    )
    ap.add_argument(
        "--run-id",
        default=None,
        help="this run's own dir under <out>/<probe>/; the probe CLI sets it to the "
        "start time, and without it the run records into <out>/<probe>/ itself",
    )


def run_id(now: datetime) -> str:
    """The run dir name for a run started at now (RUN_ID)."""
    return now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def out_root(args: argparse.Namespace, name: str) -> Path:
    """<out>/<name>/<run-id>/, or <out>/<name>/ with no run id."""
    root = (Path(args.out_dir) if args.out_dir else DEFAULT_OUT / args.gpu_label) / name
    run = getattr(args, "run_id", None)
    if run is None:
        return root
    if not RUN_ID.fullmatch(run):
        raise SystemExit(f"--run-id {run!r} is not a run id like 20261011T120000.000000Z")
    return root / run


def record_dir(name: str, argv: list[str], file_flag: str | None = None) -> Path | None:
    """Where a probe's run.json goes: out_root, or with file_flag that file's dir; None when argv names no output (no --gpu-label, or --out -)."""
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
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
    """main(argv) with its run recorded through RunWriter in <record_dir>/run.json, with main's rows in rows.jsonl beside it; an exception marks the run failed (aborted on Ctrl-C) and is re-raised.

    A probe without file_flag gets --run-id from now, so each run has its own dir. The writer opens before main, so a killed run leaves no run.json and export skips it. Under torchrun only rank 0 records, and a main returning False records nothing.
    """
    global _WRITER
    now = now or datetime.now(timezone.utc)
    argv = sys.argv[1:] if file_flag else ["--run-id", run_id(now), *sys.argv[1:]]
    run_dir = record_dir(name, argv, file_flag) if env_map.torchrun_rank() in (None, 0) else None
    writer = None
    if run_dir is not None:
        cfg = Config(
            EngineCfg(),
            WorkloadCfg(),
            ModeCfg(mode="probe", probe=name, probe_args=shlex.join(argv)),
            HardwareCfg(),
            run_name=f"probe-{name}",
        )
        try:
            writer = RunWriter(run_dir, cfg, exclusive=True)
        except FileExistsError:
            raise SystemExit(f"{run_dir} already exists; each probe run makes its own dir") from None
        writer.provenance["started"] = now.isoformat()
    primary, status, reason = None, "ok", None
    _WRITER = writer
    try:
        primary = main(argv)
    except BaseException as e:
        status, reason = exit_status(e)
        raise
    finally:
        _WRITER = None
        if writer is not None and primary is False:
            writer.discard()
        elif writer is not None:
            writer.finish({"probe": name, "argv": redact(argv)}, status=status, reason=reason)


def launch_vllm(
    run_dir: Path,
    model: str,
    *,
    port: int,
    max_model_len: int,
    prefix_caching: bool,
    tp: int = 1,
    gpu_util: float = 0.9,
    vllm_flags: dict | None = None,
    timeout_s: int = 1800,
):
    """Start vLLM as `qbench run` does: a Config through the vllm adapter's translate, then launcher.start into run_dir. vllm_flags go in as engine.extra.vllm.

    The hardware is run_config's when `qbench run` started the probe, else the defaults. Returns the Handle; the caller stops it.
    """
    # Imported here: remote kernel-probe hosts have no engine stack.
    from quettabench.engines import launcher, vllm

    engine = EngineCfg(
        model=model,
        tp=tp,
        dtype="bfloat16",
        port=port,
        gpu_util=gpu_util,
        max_model_len=max_model_len,
        prefix_caching=prefix_caching,
        extra={"vllm": dict(vllm_flags or {})},
    )
    hardware = _RUN_CFG.hardware if _RUN_CFG else HardwareCfg()
    hardware = dataclasses.replace(hardware, provision={"py": sys.executable, **hardware.provision})
    cfg = Config(engine, WorkloadCfg(), ModeCfg(), hardware, run_name=Path(run_dir).name)
    spec = vllm.ADAPTER.translate(cfg, run_dir)
    return launcher.start(spec, run_dir, timeout_s=timeout_s)


# What remote_probe ships: the probes and what base.py imports for run.json.
SHIPPED = (
    "quettabench/__init__.py",
    "quettabench/probes/",
    "quettabench/config/__init__.py",
    "quettabench/config/schema.py",
    "quettabench/config/env_map.py",
    "quettabench/metrics/__init__.py",
    "quettabench/metrics/results.py",
)
REMOTE_DIR = re.compile(r"/tmp/qbench-probes\.[A-Za-z0-9]{6,}")
NAME = re.compile(r"[a-z][a-z0-9_]*")
GPU = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# Flags remote_probe sets itself; a probe arg naming or abbreviating one is refused.
OWN_FLAGS = ("--out", "--out-dir", "--run-id", "--gpu-label", "--gpu")


def _own_flag(name):
    """name is one of OWN_FLAGS or, past "--", a prefix an abbreviating parser would expand to one."""
    return name in OWN_FLAGS or (name.startswith("--") and len(name) > 2 and any(f.startswith(name) for f in OWN_FLAGS))


def _remote_args(argv):
    ap = argparse.ArgumentParser(prog="just remote-probe")
    ap.add_argument("--host", required=True, help="ssh host or alias (REMOTE_HOST)")
    ap.add_argument("--py", default="python3", help="the remote python (PYTHON_BIN)")
    ap.add_argument("--gpu-id", default="0", help="the remote GPU (GPU_ID)")
    ap.add_argument("--gpu-ids", default="", help="the remote GPUs under torchrun (GPU_IDS)")
    ap.add_argument("--nproc", type=int, default=1, help="ranks; above 1 runs torchrun (NPROC)")
    ap.add_argument("--ncu-bin", default="", help="the remote ncu (NCU_BIN)")
    ap.add_argument("--probe-args", default="", help="the probe's own flags, one string")
    ap.add_argument("name")
    ap.add_argument("gpu")
    ap.add_argument("probe_argv", nargs="*", help="the probe's own flags, after --")
    a = ap.parse_args(argv)
    if not a.host or a.host.startswith("-") or not NAME.fullmatch(a.name) or not GPU.fullmatch(a.gpu):
        ap.error("needs --host (set REMOTE_HOST, not starting with -), a probe module name and a GPU label")
    a.probe_argv = [*shlex.split(a.probe_args), *a.probe_argv]
    if any(_own_flag(arg.split("=")[0]) for arg in a.probe_argv):
        ap.error(f"remote_probe sets {', '.join(OWN_FLAGS)} itself; leave them out of the probe args")
    return a


def _remote_python(py):
    """py quoted for the remote shell, a leading ~/ left for it to expand."""
    return "~/" + shlex.quote(py[2:]) if py.startswith("~/") else shlex.quote(py)


def _finish_here(run_dir: Path) -> None:
    """Stamp provenance.pulled on this machine's clock so export orders runs on one clock; warn and leave the run as pulled when that fails."""
    path = run_dir / "run.json"
    if not path.is_file():
        return
    try:
        stamp_pulled(path)
    except (ValueError, TypeError, OSError) as e:
        print(f"{run_dir}: run.json not stamped pulled ({type(e).__name__}: {e})", file=sys.stderr)


def remote_probe(argv=None, run=subprocess.run, now=None) -> int:
    """Run a probe in a remote `mktemp -d` dir with --run-id from this machine's clock, always pull its out/ into results/probes/<gpu>/ without overwriting, stamp it, then remove that dir.

    Returns the remote exit code; a failed or interrupted pull keeps the remote dir and says where it is.
    """
    a = _remote_args(argv)
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", a.host]
    rsh = "ssh -o BatchMode=yes -o ConnectTimeout=30"
    made = run(
        [*ssh, "d=$(mktemp -d /tmp/qbench-probes.XXXXXX) && mkdir -p $d/out && echo $d"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    # The last line that is a dir: a login banner may come first.
    dirs = [line.strip() for line in made.splitlines() if REMOTE_DIR.fullmatch(line.strip())]
    if not dirs:
        raise SystemExit(f"remote mktemp gave {made!r}, not a /tmp/qbench-probes.XXXXXX dir")
    rdir = dirs[-1]
    py = _remote_python(a.py)
    if a.nproc > 1:
        launch = (
            f"CUDA_VISIBLE_DEVICES={shlex.quote(a.gpu_ids or a.gpu_id)} {py} "
            f"-m torch.distributed.run --nproc_per_node={a.nproc} -m quettabench.probes.{a.name}"
        )
    else:
        ncu = f"NCU_BIN={shlex.quote(a.ncu_bin)} " if a.ncu_bin else ""
        launch = f"CUDA_VISIBLE_DEVICES={shlex.quote(a.gpu_id)} {ncu}{py} -m quettabench.probes.{a.name}"
    rid = run_id(now or datetime.now(timezone.utc))
    command = (
        f"cd {rdir} && {launch} --gpu-label {a.gpu} --out {rdir}/out --run-id {rid} {shlex.join(a.probe_argv)}"
    ).rstrip()
    started, pulled, status = False, None, None
    try:
        run(["rsync", "-a", "--relative", "-e", rsh, *SHIPPED, f"{a.host}:{rdir}/"], check=True)
        started = True
        status = run([*ssh, command]).returncode
    finally:
        # Pulled even when the run failed or was interrupted; removed only once pulled.
        if started:
            local = Path("results") / "probes" / a.gpu
            local.mkdir(parents=True, exist_ok=True)
            try:
                pulled = run(
                    ["rsync", "-a", "--ignore-existing", "-e", rsh, f"{a.host}:{rdir}/out/", f"{local}/"]
                ).returncode
            finally:
                if pulled != 0:
                    print(
                        f"pulling {a.host}:{rdir}/out/ failed ({pulled}); the remote dir {a.host}:{rdir} is kept",
                        file=sys.stderr,
                    )
            if not pulled:
                _finish_here(local / a.name / rid)
        if not pulled:
            run([*ssh, f"rm -rf {rdir}"])
    if pulled:
        raise SystemExit(f"remote dir kept: {a.host}:{rdir} (probe exit {status})")
    return status


if __name__ == "__main__":
    if sys.argv[1:2] != ["remote-probe"]:
        raise SystemExit("usage: python -m quettabench.probes.base remote-probe ...")
    raise SystemExit(remote_probe(sys.argv[2:]))
