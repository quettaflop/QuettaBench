"""qbench run: a run yaml to results/<run>/, one Config per matrix point."""

import ast
import dataclasses
import importlib
import json
import os
import re
import shlex
import shutil
import sys
import uuid
from pathlib import Path

from quettabench.config import env_map, loader
from quettabench.config.schema import OTHER_ACCELERATORS, ConfigError, device_count
from quettabench.engines import base, launcher
from quettabench.metrics.results import RUN_MARKERS, RunWriter, redact
from quettabench.modes import serve
from quettabench.probes import base as probes_base
from quettabench.workloads.grid import GridTrace
from quettabench.workloads.live import (
    PROFILES,
    MooncakeTrace,
    PromptJsonlTrace,
    RealTrace,
    ShareGPTTrace,
    SmokeTrace,
    get_profile,
)
from quettabench.workloads.sessions import (
    ShareGPTSessionTrace,
    TrajectorySessionTrace,
    TurnPacing,
    resolve_multi_turn_num_sessions,
)

# Probes mode: probe refuses (check_probe): torchrun's, and those whose output flag names a file rather than the run dir.
TORCHRUN_PROBES = ("collective", "reparallel", "vllm_allreduce")
FILE_OUT_PROBES = ("serving_decode_grid", "serving_stage_split")

TRAJECTORY_DATASETS = ("swebench-multi-turn", "terminalbench-multi-turn", "osworld-multi-turn")

# Room left between a trajectory turn's prompt plus output and the profile's isl_tokens window.
CONTEXT_SAFETY_MARGIN = 256


def single_turn_trace(profile, seed):
    """The undrawn single-turn trace a profile's dataset names."""
    d = profile.dataset
    if d == "test":
        return SmokeTrace()
    if d == "mooncake-trace":
        return MooncakeTrace(profile.file_path, seed=seed)
    if d == "jsonl":
        return PromptJsonlTrace(profile.file_path, seed=seed)
    if d == "sharegpt":
        return ShareGPTTrace(
            num_prompts=1000,
            seed=seed,
            system_prompt=profile.system_prompt,
            max_isl_tokens=profile.isl_tokens,
            max_osl_tokens=profile.osl_tokens,
            context_safety_margin_tokens=CONTEXT_SAFETY_MARGIN,
        )
    if d == "sharegpt-multi-turn" or d in TRAJECTORY_DATASETS:
        raise ValueError(f"Profile '{profile.name}' is multi-turn; run it with workload.kind=sessions")
    raise ValueError(f"Unknown dataset type: {d}")


def session_trace(profile, num_sessions, seed, **trace_kw):
    """A multi-turn profile's sessions as a LoadedSessionTrace."""
    if profile.dataset == "sharegpt-multi-turn":
        return ShareGPTSessionTrace(
            min_turns=profile.min_turns,
            max_turns=profile.max_turns,
            num_sessions=num_sessions,
            seed=seed,
            system_prompt=profile.system_prompt,
            max_isl_tokens=profile.isl_tokens,
            max_osl_tokens=profile.osl_tokens,
            profile=profile,
            **trace_kw,
        )
    if profile.dataset in TRAJECTORY_DATASETS:
        return TrajectorySessionTrace(
            profile.file_path,
            min_turns=profile.min_turns,
            max_turns=profile.max_turns,
            num_sessions=num_sessions,
            seed=seed,
            max_isl_tokens=profile.isl_tokens,
            max_osl_tokens=profile.osl_tokens,
            context_safety_margin_tokens=CONTEXT_SAFETY_MARGIN,
            profile=profile,
            **trace_kw,
        )
    raise ValueError(f"Profile '{profile.name}' does not use a multi-turn dataset")


def build_trace(cfg, tokenizer=None):
    """The run's Trace: a GridTrace for mode.grid, else workload.source as a profile name or a replay JSONL path."""
    w = cfg.workload
    if cfg.mode.grid is not None:
        return GridTrace.from_config(cfg, tokenizer=tokenizer)
    if w.source not in PROFILES:
        return RealTrace.from_jsonl(w.source, seed=w.seed)
    profile = get_profile(w.source)
    if w.kind == "sessions":
        pacing = TurnPacing(w.pacing, w.tool_wait_ms, w.human_wait_ms, w.use_recorded_waits, w.use_recorded_arrivals)
        sessions, _ = resolve_multi_turn_num_sessions(profile, cfg.mode.concurrency)
        return session_trace(
            profile, sessions, w.seed, warmup=cfg.mode.warmup, pacing=pacing, max_turn_index=w.max_turn_index
        )
    trace = single_turn_trace(profile, w.seed)
    if (
        w.arrival == "trace"
        and isinstance(trace, MooncakeTrace)
        and trace.lap_ms == 0
        and w.num_requests > len(trace.records)
    ):
        # Every lap would land on the first lap's times, and a lap length is not invented.
        raise ConfigError(
            f"workload.arrival=trace: {profile.name}'s timestamps span 0 ms, so "
            f"{w.num_requests} requests over its {len(trace.records)} records "
            "wrap with no lap length; lower workload.num_requests or use "
            "another arrival"
        )
    return trace.draw(w.num_requests, cfg.mode.warmup, profile)


def spec_lines(spec):
    """One `spec.<field> = value` line per LaunchSpec field, then one `cell ctx= bs= = <shell line>` per bench cell; secrets redacted."""
    lines = [f"spec.{k} = {v!r}" for k, v in redact(dataclasses.asdict(spec)).items() if k != "cells"]
    return lines + [
        f"cell ctx={ctx} bs={bs} = {shlex.join(map(str, redact(cell.child_argv(cell.bench_argv))))}"
        for ctx, bs, cell in spec.cells
    ]


def in_docker(cfg, adapter):
    """Whether the server runs in docker; refused when nothing is launched, since the driver would then run the engine on the host."""
    if cfg.hardware.launcher != "docker":
        return False
    if not adapter.needs_server(cfg):
        raise ConfigError(
            f"{cfg.run_name}: hardware.launcher=docker needs a launched server; "
            "a grid run with mode.returns builds vLLM in this process"
        )
    return True


def dry_run(cfg, out):
    print(f"== {cfg.run_name}")
    print(cfg.dump(origins=True, scrub=redact))
    if cfg.mode.mode != "probe":
        adapter = base.get(cfg.engine.name)
        spec = adapter.translate(cfg, out)
        print("\n".join(spec_lines(spec)))
        print(f"launch = {'server' if adapter.needs_server(cfg) else 'none, the driver runs it'}")
        if in_docker(cfg, adapter) and spec.server_argv:
            # The dry run shows 0..n-1 unless visible_devices is set; a real run takes idle GPUs.
            gpus = cfg.hardware.visible_devices or ",".join(str(i) for i in range(device_count(cfg.engine)))
            argv = redact(launcher.docker_argv(spec, cfg, gpus, out))
            print(f"docker = {shlex.join(map(str, argv))}")


def is_probe(name):
    """Whether quettabench.probes.<name> has a top-level main, read from source so no probe (or torch) is imported."""
    module = sys.modules.get(f"quettabench.probes.{name}")
    if module is not None:
        return callable(getattr(module, "main", None))
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name or ""):
        return False
    path = Path(probes_base.__file__).with_name(f"{name}.py")
    if not path.is_file():
        return False
    return any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in ast.parse(path.read_text()).body)


def check_probe(cfg):
    """Refuse a probe `mode: probe` cannot run: no such module with a main, one that needs torchrun ranks, or one whose output flag names a file."""
    name = cfg.mode.probe
    if cfg.hardware.backend in OTHER_ACCELERATORS:
        raise ConfigError(
            f"{cfg.run_name}: the kernel probes time CUDA kernels (CUDA graphs, "
            f"ncu, CUDA events); hardware.backend={cfg.hardware.backend} has none"
        )
    if not is_probe(name):
        raise ConfigError(f"{cfg.run_name}: mode.probe {name!r} is not a module under quettabench/probes/ with a main")
    if name in TORCHRUN_PROBES:
        raise ConfigError(
            f"{cfg.run_name}: probe {name} runs one process per rank under "
            "torchrun; launch it with `just probe` or torchrun, not mode: probe"
        )
    if name in FILE_OUT_PROBES:
        raise ConfigError(
            f"{cfg.run_name}: probe {name} takes output file paths, not a run dir --out; run it with `just probe`"
        )


def run_probe(cfg, out):
    """The probe's main with its flags and --out set to the run dir, on the run's GPUs; its rows go to rows.jsonl."""
    check_probe(cfg)
    argv = [*shlex.split(cfg.mode.probe_args), "--out", str(out)]
    writer = RunWriter(out, cfg)
    try:
        gpus = cfg.hardware.visible_devices or launcher.idle_gpus(device_count(cfg.engine), cfg.hardware.backend)
        env_map.export({"CUDA_VISIBLE_DEVICES": gpus})
        writer.provenance["launch_env"] = {"CUDA_VISIBLE_DEVICES": gpus}
        probe = importlib.import_module(f"quettabench.probes.{cfg.mode.probe}")
        with probes_base.run_config(cfg, writer):
            probe.main(argv)
    except BaseException as e:
        writer.fail(e)
        raise
    writer.finish({"probe": cfg.mode.probe, "argv": redact(argv)})


def with_child_env(spec, pins, drop=()):
    """spec with pins added to its env and every bench cell's env, and the names in drop (exported by an earlier in-process point) unset in the child."""
    cells = tuple((ctx, bs, with_child_env(cell, pins, drop)) for ctx, bs, cell in spec.cells)
    env_drop = (*spec.env_drop, *sorted(set(drop) - set(pins) - set(spec.env_drop)))
    return dataclasses.replace(spec, env={**spec.env, **pins}, env_drop=env_drop, cells=cells)


def in_process(cfg, adapter, spec):
    """Whether the driver runs the engine in this process: no server and no bench child."""
    return not adapter.needs_server(cfg) and not spec.bench_argv


def refuse_mixed_in_process(cfgs):
    """In-process points must agree on CUDA_VISIBLE_DEVICES and collective pins, since CUDA reads the first point's for the whole process."""
    seen, unnamed, points = set(), False, 0
    for cfg in cfgs:
        adapter = None if cfg.mode.mode == "probe" else base.get(cfg.engine.name)
        if adapter is None or in_process(cfg, adapter, adapter.translate(cfg)):
            hw = cfg.hardware
            points += 1
            unnamed = unnamed or not hw.visible_devices
            seen.add(
                (
                    hw.visible_devices or device_count(cfg.engine),
                    tuple(sorted((k, v) for k, v in hw.collective.items() if v)),
                )
            )
    if len(seen) > 1:
        raise ConfigError(
            "in-process matrix points differ in GPUs or collective pins; "
            "one process cannot change them, so run each point on its own"
        )
    if points > 1 and unnamed:
        raise ConfigError(
            f"{points} in-process matrix points share this process; "
            "set hardware.visible_devices so they run on the same GPUs"
        )


def run_one(cfg, out, exported=None):
    """One Config through launch, drive, serve and stop; exported collects env names in-process points put into this process."""
    exported = set() if exported is None else exported
    if cfg.mode.mode == "probe":
        return run_probe(cfg, out)
    # Opened first, so a run that raises anywhere below still leaves run.json.
    writer = RunWriter(out, cfg)
    try:
        serve_one(cfg, out, exported, writer)
    except BaseException as e:
        writer.fail(e)
        raise


def serve_one(cfg, out, exported, writer):
    """run_one's body; the server is stopped whatever happens."""
    adapter = base.get(cfg.engine.name)
    spec = adapter.translate(cfg, out)
    gpus = cfg.hardware.visible_devices or launcher.idle_gpus(device_count(cfg.engine), cfg.hardware.backend)
    backend = "docker" if in_docker(cfg, adapter) else "local"
    pins = {k: v for k, v in cfg.hardware.collective.items() if v}
    visible, unset = launcher.visible_env(cfg.hardware.backend, gpus)
    # docker gives cuda its GPUs by --gpus; other accelerators read this env.
    if backend == "local" or cfg.hardware.backend in OTHER_ACCELERATORS:
        pins.update(visible)
    if in_process(cfg, adapter, spec):
        exported.update(env_map.export(pins))
    spec = with_child_env(spec, pins, {*exported, *unset})
    writer.provenance["launch_env"] = redact(spec.env)
    # Build the trace first so a bad workload fails before a long model load.
    trace = build_trace(cfg) if adapter.needs_server(cfg) else None
    handle = (
        launcher.start(spec, out, backend=backend, cfg=cfg, gpus=gpus, timeout_s=cfg.hardware.ready_timeout_s)
        if adapter.needs_server(cfg)
        else launcher.NoServer(out)
    )
    try:
        driver = adapter.driver(spec, handle, trace)
        writer.bind(handle, driver)
        if trace is None:
            trace = build_trace(cfg, tokenizer=getattr(driver, "tokenizer", None))
        serve.run(trace, driver, cfg.mode, writer, cfg.workload)
    finally:
        handle.stop()


def run_failed(out):
    """Whether out/run.json records a failed run."""
    try:
        return json.loads((out / "run.json").read_text())["run_status"]["status"] == "failed"
    except (OSError, ValueError, KeyError, TypeError):
        return False


def clear_run_dir(out, overwrite, results_dir):
    """Refuse a run dir that already holds files since requests.jsonl appends; with overwrite, or a failed run.json, empty it."""
    if out.is_symlink():
        raise ConfigError(f"{out} is a symlink; qbench empties only a run dir it made")
    if not out.is_dir() or not any(out.iterdir()):
        return
    if out.resolve().parent != Path(results_dir).resolve():
        raise ConfigError(f"{out} is not a run dir directly under {results_dir}")
    if not any((out / marker).exists() for marker in RUN_MARKERS):
        raise ConfigError(
            f"{out} holds no {', '.join(RUN_MARKERS[:-1])} or {RUN_MARKERS[-1]}; "
            "qbench did not make it, so it is never emptied"
        )
    if not (overwrite or run_failed(out)):
        raise ConfigError(f"{out} already holds a run; pass --overwrite to replace it")
    # os.rename moves the entry itself, never a symlink's target.
    aside = out.parent / f".clearing-{uuid.uuid4().hex}-{out.name}"
    os.rename(out, aside)
    if aside.is_symlink() or not aside.is_dir():
        os.rename(aside, out)
        raise ConfigError(f"{out} became a symlink or a file while being emptied; left as it is")
    # Markers first: an rmtree cut short leaves no dir a results scan reads as a run.
    for marker in RUN_MARKERS:
        (aside / marker).unlink(missing_ok=True)
    shutil.rmtree(aside)


def main(run_yaml, overrides=(), dry=False, results_dir="results", allow_unverified=False, overwrite=False):
    """Every Config of run_yaml, in matrix order; results/<run_name>/ each."""
    sets = [*overrides, *(["allow_unverified=true"] if allow_unverified else [])]
    cfgs = loader.load(run_yaml, sets)
    for cfg in cfgs:
        if cfg.mode.mode == "probe":
            check_probe(cfg)
    if not dry:
        refuse_mixed_in_process(cfgs)
        for cfg in cfgs:
            clear_run_dir(Path(results_dir) / cfg.run_name, overwrite, results_dir)
    exported = set()
    for cfg in cfgs:
        out = Path(results_dir) / cfg.run_name
        if dry:
            dry_run(cfg, out)
        else:
            run_one(cfg, out, exported)
    return 0
