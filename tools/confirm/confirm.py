"""Check a box before paying for GPU time: config, data, a client self-test against a fake server, then a smoke
run on the real server (--server).
"""

import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

from quettabench import runner
from quettabench.config import loader
from quettabench.modes.serve import LOOP_LAG_LIMIT_MS, first_start_stagger_s

REPO = Path(__file__).resolve().parents[2]
# Fake server timing: 30 ms to the first token, 20 ms a token, four processes so request parsing never lands on the
# client's account.
FAKE_TIMING = ("--ttft-a", "0.03", "--ttft-b", "2e-7", "--tpot", "0.02", "--workers", "4", "--no-hash")
# Enough sessions to exercise every gate in under a minute.
SMOKE_SESSIONS = 8
# First turns only, so plays end soon.
SMOKE_TURNS = 3
# A fake server that has not answered /health by then failed to start; it needs a second or two.
FAKE_SERVER_START_S = 30
# Long enough for every agent to be playing and the loop-lag p99 to settle at that load.
SELFTEST_DEFAULT_S = 180.0
# Past the estimated opening, so the window lasts well over the 300 s a recycled run needs to publish one.
WINDOW_MARGIN_S = 600.0
# One loop stall this long in the self-test is a client that cannot be trusted at this agent count.
STALL_LIMIT_MS = 50.0


def first_point(run_yaml) -> list[str]:
    """--set lines pinning a sweep yaml's matrix to its first point, so one run is confirmed at a time. A user's own
    --set for an axis, placed after these, wins."""
    raw = yaml.safe_load(Path(run_yaml).read_text()) or {}
    return [f"{axis}={json.dumps([choices[0]])}" for axis, choices in (raw.get("matrix") or {}).items()]


def phase_sets(phase, base, **values) -> list[str]:
    return [*base, *(f"{key}={value}" for key, value in values.items()), f"run_name=confirm-{phase}"]


def qbench(run_yaml, sets, results, log, allow_unverified, dry=False) -> int:
    """qbench as the real run runs it: its own process with the repo on its path, stdout and stderr kept in log."""
    cmd = [sys.executable, "-m", "quettabench.cli", "run", str(run_yaml), "--results-dir", str(results), "--overwrite"]
    cmd += ["--dry-run"] if dry else []
    cmd += ["--allow-unverified"] if allow_unverified else []
    for item in sets:
        cmd += ["--set", item]
    path = os.pathsep.join(p for p in (str(REPO), os.environ.get("PYTHONPATH")) if p)
    with open(log, "w") as out:
        out.write("$ " + " ".join(cmd) + "\n")
        out.flush()
        return subprocess.run(
            cmd, env=dict(os.environ, PYTHONPATH=path), stdout=out, stderr=subprocess.STDOUT
        ).returncode


def run_dir_of(results, phase):
    """The phase's run dir: qbench adds the matrix point's suffixes to run_name, so it is found by prefix."""
    dirs = [d for d in Path(results).glob(f"confirm-{phase}*") if (d / "run.json").exists()]
    return max(dirs, key=lambda d: d.stat().st_mtime) if dirs else None


def run_files(run_dir):
    run = json.loads((Path(run_dir) / "run.json").read_text())
    rows_path = Path(run_dir) / "requests.jsonl"
    rows = (
        [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()] if rows_path.exists() else []
    )
    return run, rows


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def healthy(port) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
            return True
    except OSError:
        return False


class FakeServer:
    """tools.confirm.fake_server in its own process for the with block, serving the one model name the yaml asks
    for; SIGINT on exit lets it flush its log."""

    def __init__(self, port, served, log, timing=None):
        self.port = port
        # The server runs in the repo, so its log is named by an absolute path; its own output goes beside it.
        self.log = Path(log).resolve()
        self.cmd = [sys.executable, "-m", "tools.confirm.fake_server", "--port", str(port), "--model", served]
        self.cmd += ["--log", str(self.log), *(timing or FAKE_TIMING)]

    def __enter__(self):
        self.out = open(self.log.with_suffix(".log"), "w")
        self.proc = subprocess.Popen(self.cmd, cwd=REPO, stdout=self.out, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + FAKE_SERVER_START_S
        while not healthy(self.port):
            if self.proc.poll() is not None:
                self.out.close()
                raise RuntimeError(f"fake server exited with {self.proc.returncode}; see {self.out.name}")
            if time.monotonic() > deadline:
                self.__exit__()
                raise RuntimeError(
                    f"fake server not healthy on port {self.port} after {FAKE_SERVER_START_S} s; see {self.out.name}"
                )
            time.sleep(0.1)
        return self

    def __exit__(self, *exc):
        self.proc.send_signal(signal.SIGINT)
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.out.close()


def client_gates(run) -> list[tuple[str, bool, str]]:
    """The client's own health at the agent count, which is what the fake server is for: nothing failed and the event
    loop kept up. Latency validity is a note beside these (client_notes), not a gate: the fake server's 30 ms TTFT
    weighs a client stall about ten times more than a real server's does."""
    summary = run.get("summary") or {}
    lag = (summary.get("client") or {}).get("loop_lag_ms") or {}
    p99, worst = lag.get("99", math.inf), lag.get("max", math.inf)
    failed, total = summary.get("failed_requests", 0), summary.get("num_requests", 0)
    return [
        ("no failed request", total > 0 and not failed, f"{failed} of {total} failed"),
        (f"event loop p99 lag under {LOOP_LAG_LIMIT_MS:g} ms", p99 <= LOOP_LAG_LIMIT_MS, f"p99 {p99:.1f} ms"),
        (f"no loop stall over {STALL_LIMIT_MS:g} ms", worst <= STALL_LIMIT_MS, f"worst {worst:.1f} ms"),
    ]


def client_notes(run) -> list[str]:
    """What the self-test says about latency validity, for the record: withheld on the fake server does not mean
    withheld on a real one, whose first token takes far longer than any client stall."""
    summary = run.get("summary") or {}
    status = run.get("run_status") or {}
    validity = summary.get("validity") or {}
    valid = validity.get("latency_valid") is True
    note = f"run status {status.get('status')}; latency {'valid' if valid else 'withheld'} on the fake server"
    reasons = "; ".join(validity.get("reasons") or [])
    return [note + (f": {reasons}" if reasons and not valid else "")]


def run_gates(run) -> list[tuple[str, bool, str]]:
    """The run-level gates the real run applies: it finished ok, nothing failed, and latency stands."""
    summary = run.get("summary") or {}
    status = run.get("run_status") or {}
    validity = summary.get("validity") or {}
    failed, total = summary.get("failed_requests", 0), summary.get("num_requests", 0)
    return [
        ("run finished ok", status.get("status") == "ok", f"{status.get('status')}: {status.get('reason')}"),
        ("no failed request", total > 0 and not failed, f"{failed} of {total} failed"),
        (
            "latency valid",
            validity.get("latency_valid") is True,
            "; ".join(validity.get("reasons") or []) or "client stalls under the bias bound",
        ),
    ]


def smoke_gates(run, rows) -> tuple[list[tuple[str, bool, str]], list[str]]:
    """Every gate the real run applies, on a short replay against the real server: streams complete with usage, each
    reply is its planned length, each request is the plan's. A server that reports no cached-token counts is a
    warning, not a failure: cache hit would be unknown, never wrong."""
    summary = run.get("summary") or {}
    ok = [r for r in rows if r.get("success")]
    with_usage = sum(1 for r in ok if r.get("usage_reported"))
    exact = sum(1 for r in ok if r.get("output_tokens") == r.get("max_tokens_requested"))
    identity = summary.get("workload") or {}
    mismatch = identity.get("first_mismatch")
    gates = run_gates(run) + [
        ("every reply carries usage", bool(rows) and with_usage == len(rows), f"{with_usage} of {len(rows)}"),
        ("every reply is its planned length", bool(rows) and exact == len(rows), f"{exact} of {len(rows)}"),
        (
            "every request as planned",
            identity.get("match") is True,
            json.dumps(mismatch) if mismatch else f"{identity.get('n')} of {identity.get('of')}",
        ),
    ]
    warnings = []
    if summary.get("server_cache_hit_rate") is None:
        warnings.append(
            "the server reports no cached-token counts, so cache hit will be unknown "
            "(vLLM: --enable-prompt-tokens-details)"
        )
    return gates, warnings


def session_s(turns, ttft_s, tpot_s, waits) -> float:
    """How long one session takes an agent at these latencies: each turn's first token, its remaining tokens, and
    the recorded wait after every turn but the last, as the dispatcher sleeps it."""
    total = 0.0
    for i, turn in enumerate(turns):
        total += ttft_s + max((turn.max_tokens or 1) - 1, 0) * tpot_s
        if waits and i < len(turns) - 1:
            meta = getattr(turn, "metadata", None) or {}
            total += ((meta.get("tool_wait_after_ms") or 0.0) + (meta.get("human_wait_after_ms") or 0.0)) / 1000.0
    return total


def window_opens_s(sessions, agents, ttft_s, tpot_s, waits) -> float:
    """When the steady window could open at this agent count: agent a starts a stagger later than agent a - 1 and
    plays session a of the cycle, and the window opens when the last of them finishes. At the smoke's latencies
    this is a lower bound; a server with more agents on it is slower per token."""
    stagger = first_start_stagger_s(agents)
    return max(a * stagger + session_s(sessions[a % len(sessions)], ttft_s, tpot_s, waits) for a in range(agents))


def phase_config(args, results, base) -> dict:
    code = qbench(args.run_yaml, base, results, results / "config.log", args.allow_unverified, dry=True)
    point = " ".join(base) or "the yaml as it is"
    return {"passed": code == 0, "detail": f"dry run exit {code} on {point}; see config.log"}


def phase_data(cfg, allow_unverified) -> tuple[dict, object]:
    """Builds the trace as the real run does: the manifest check and the sample, which refuses a file with fewer
    sessions than agents."""
    try:
        trace = runner.build_trace(cfg, allow_unverified=allow_unverified)
    except Exception as e:  # ConfigError or a missing file; the message says what to fix
        return {"passed": False, "detail": str(e)}, None
    meta = trace.source_meta()
    manifest = getattr(trace, "manifest", None) or {}
    name = Path(manifest.get("file") or cfg.workload.source).name
    warnings = []
    if not manifest.get("listed"):
        warnings.append(f"{name} is not in data/MANIFEST.json; a run on it is not comparable")
    elif manifest.get("comparable") is False:
        warnings.append(f"{name} is not the manifest's file; a run on it is not comparable")
    kept = {
        k: meta[k] for k in ("sessions", "sessions_cut", "sessions_dropped", "sessions_dropped_no_waits") if k in meta
    }
    detail = f"{name}: {kept.get('sessions')} sessions in the sample, {kept.get('sessions_dropped', 0)} dropped, "
    detail += f"{kept.get('sessions_cut', 0)} cut at the context limit"
    return {"passed": True, "detail": detail, "warnings": warnings, "trace": kept}, trace


def phase_client(args, results, base, served, timing=None) -> dict:
    port = free_port()
    sets = phase_sets(
        "client",
        base,
        **{
            "hardware.launcher": "attach",
            "engine.port": port,
            "mode.concurrency": args.agents,
            "mode.duration_s": args.selftest_s,
            "mode.reset_prefix_cache": "true",
        },
    )
    if args.selftest_turns is not None:
        # The quick form: the first turns only, so plays end soon; their prompts are smaller than late turns'.
        sets.append(f"workload.max_turn_index={args.selftest_turns - 1}")
    with FakeServer(port, served, results / "fake-server.jsonl", timing):
        code = qbench(args.run_yaml, sets, results, results / "client.log", args.allow_unverified)
    run_dir = run_dir_of(results, "client")
    if run_dir is None:
        return {"passed": False, "detail": f"qbench exit {code} and no run.json; see client.log"}
    run, _ = run_files(run_dir)
    gates = client_gates(run)
    summary = run.get("summary") or {}
    turns = f", {args.selftest_turns} turns a play" if args.selftest_turns is not None else ""
    detail = f"{args.agents} agents for {args.selftest_s:g} s{turns} on the fake server: {summary.get('num_requests')} "
    detail += f"requests, {summary.get('max_inflight_requests')} in flight at the peak"
    out = {"passed": all(g[1] for g in gates), "detail": detail, "gates": gates, "notes": client_notes(run)}
    return {**out, "run_dir": str(run_dir)}


def phase_server(port, served) -> dict:
    endpoint = f"http://127.0.0.1:{port}"
    try:
        with urllib.request.urlopen(endpoint + "/health", timeout=5) as resp:
            healthy_ = resp.status == 200
        with urllib.request.urlopen(endpoint + "/v1/models", timeout=5) as resp:
            models = [m.get("id") for m in json.load(resp).get("data", [])]
    except (OSError, ValueError, AttributeError) as e:
        return {"passed": False, "detail": f"{endpoint}: {e}"}
    try:
        with urllib.request.urlopen(endpoint + "/version", timeout=5) as resp:
            version = (json.load(resp) or {}).get("version")
    except (OSError, ValueError, AttributeError):
        version = None
    detail = f"{endpoint} serves {models}, version {version}"
    if served not in models:
        detail += f"; the yaml asks for {served}"
    return {"passed": healthy_ and served in models, "detail": detail}


def phase_smoke(args, results, base) -> tuple[dict, dict | None]:
    sessions = min(args.agents, SMOKE_SESSIONS)
    sets = phase_sets(
        "smoke",
        base,
        **{
            "hardware.launcher": "attach",
            "engine.port": args.server,
            "mode.concurrency": sessions,
            "mode.duration_s": 0,
            "workload.num_sessions": sessions,
            "workload.max_turn_index": SMOKE_TURNS - 1,
        },
    )
    code = qbench(args.run_yaml, sets, results, results / "smoke.log", args.allow_unverified)
    run_dir = run_dir_of(results, "smoke")
    if run_dir is None:
        return {"passed": False, "detail": f"qbench exit {code} and no run.json; see smoke.log"}, None
    run, rows = run_files(run_dir)
    gates, warnings = smoke_gates(run, rows)
    summary = run.get("summary") or {}
    cache = summary.get("server_cache_hit_rate")
    detail = f"{sessions} sessions of {SMOKE_TURNS} turns on the real server: {len(rows)} requests, TTFT p50 "
    detail += f"{summary.get('median_ttft_ms') or 0:.0f} ms, TPOT p50 {summary.get('median_tpot_ms') or 0:.1f} ms, "
    detail += "cache hit unknown" if cache is None else f"cache hit {cache:.1%}"
    out = {"passed": all(g[1] for g in gates), "detail": detail, "gates": gates, "warnings": warnings}
    return {**out, "run_dir": str(run_dir)}, run


def phase_sizing(args, cfg, trace, smoke_run) -> dict:
    summary = smoke_run.get("summary") or {}
    ttft_ms, tpot_ms = summary.get("median_ttft_ms"), summary.get("median_tpot_ms")
    if not ttft_ms or not tpot_ms:
        return {"passed": True, "detail": "the smoke measured no latency, so there is nothing to size"}
    waits = cfg.workload.use_recorded_waits
    opens = window_opens_s(trace.sessions(), args.agents, ttft_ms / 1000, tpot_ms / 1000, waits)
    advised = math.ceil((opens + WINDOW_MARGIN_S) / 60) * 60
    detail = f"at the smoke's latencies the steady window opens after about {opens:,.0f} s at {args.agents} agents, "
    detail += f"a lower bound; mode.duration_s is {cfg.mode.duration_s:g}, {advised:,} or more is advised"
    return {"passed": True, "detail": detail, "opens_s": opens, "duration_s_advised": advised}


def served_name(cfg) -> str:
    extra = cfg.engine.extra.get(cfg.engine.name) or {}
    return str(extra.get("served_model_name") or cfg.engine.model)


def parser():
    p = argparse.ArgumentParser(
        prog="python -m tools.confirm", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run_yaml", help="the run yaml the real run will use")
    p.add_argument("--agents", type=int, required=True, help="the agent count the real run will use")
    p.add_argument("--set", dest="sets", action="append", default=[], metavar="K=V", help="as for qbench run")
    p.add_argument("--allow-unverified", action="store_true", help="as for qbench run")
    p.add_argument("--server", type=int, metavar="PORT", help="the real server's port, once it answers")
    p.add_argument("--results-dir", default="results/confirm", help="where every phase's run and log lands")
    p.add_argument(
        "--selftest-s", type=float, default=SELFTEST_DEFAULT_S, help="how long the client runs on the fake server"
    )
    p.add_argument(
        "--selftest-turns",
        type=int,
        metavar="N",
        help="the quick form of the client phase: plays end after their first N turns",
    )
    p.add_argument("--workdir", default=str(REPO), help="where configs/ and data/ are (default: the repo)")
    return p


def parse_args(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.agents < 1:
        p.error("--agents must be at least 1")
    if args.selftest_s <= 0:
        p.error("--selftest-s must be greater than 0")
    if args.selftest_turns is not None and args.selftest_turns < 1:
        p.error("--selftest-turns must be at least 1, or left out for whole plays")
    if args.server is not None and not 1 <= args.server <= 65535:
        p.error("--server must be a port from 1 to 65535")
    if not Path(args.workdir).is_dir():
        p.error(f"--workdir {args.workdir} is not a directory; name the one holding configs/ and data/")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    os.chdir(args.workdir)
    results = Path(args.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    base = [*first_point(args.run_yaml), *args.sets]
    report: dict = {}

    def record_phase(name, outcome) -> bool:
        report[name] = outcome
        print(f"{name:<8}{'PASS' if outcome['passed'] else 'FAIL'}  {outcome['detail']}")
        for gate, passed, why in outcome.get("gates", []):
            print(f"        {'ok ' if passed else 'NO '} {gate}: {why}")
        for warning in outcome.get("warnings", []):
            print(f"        warning: {warning}")
        for note in outcome.get("notes", []):
            print(f"        note: {note}")
        (results / "confirm.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
        if not outcome["passed"]:
            print(f"CONFIRM FAIL at {name}")
        return outcome["passed"]

    if not record_phase("config", phase_config(args, results, base)):
        return 1
    sets = [*base, "hardware.launcher=attach", f"mode.concurrency={args.agents}"]
    sets += ["allow_unverified=true"] if args.allow_unverified else []
    cfg = loader.load(args.run_yaml, sets)[0]
    outcome, trace = phase_data(cfg, args.allow_unverified)
    if not record_phase("data", outcome):
        return 1
    served = served_name(cfg)
    try:
        outcome = phase_client(args, results, base, served)
    except RuntimeError as e:
        outcome = {"passed": False, "detail": str(e)}
    if not record_phase("client", outcome):
        return 1
    if args.server is not None:
        if not record_phase("server", phase_server(args.server, served)):
            return 1
        outcome, smoke_run = phase_smoke(args, results, base)
        if not record_phase("smoke", outcome):
            return 1
        record_phase("sizing", phase_sizing(args, cfg, trace, smoke_run))
    print("CONFIRM PASS")
    return 0
