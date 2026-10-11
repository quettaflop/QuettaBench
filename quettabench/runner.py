"""qbench run: a run yaml to results/<run>/, the same sequence every time.

load the yaml (+ --set, + env through env_map) into one Config per matrix
point, then per Config: translate it to a LaunchSpec, add the collective
pins and the GPUs (hardware.visible_devices, else the first idle
device_count) to the child env of the server or each bench cell (docker
gets the GPUs through --gpus instead), launch the server unless the
adapter drives the engine itself (in-process grid, cargo bench), get the
driver, build the Trace (before the launch when a server is launched, since
its driver has no tokenizer), and let modes/serve.py run trace
through driver into a RunWriter; the server is stopped whatever happens.
mode: probe runs quettabench.probes.<name> with --out results/<run>/
instead and records it in the same run.json. --dry-run prints each Config
with its origins and its LaunchSpec (a bench run: each cell's env and
bench_argv as one shell line), and starts nothing. hardware.launcher=docker
runs the server in engine.image and needs a launched server (a grid run sets
engine.driver=http); --dry-run then also prints the docker command.
"""

import dataclasses
import importlib
import shlex
from pathlib import Path

from quettabench import workloads
from quettabench.config import env_map, loader
from quettabench.config.schema import ConfigError
from quettabench.config.validate import device_count
from quettabench.engines import base, launcher
from quettabench.metrics.results import RunWriter, redact
from quettabench.modes import serve

# What a run writes into results/<run>/ itself; --overwrite removes these.
RUN_FILES = ("run.json", "requests.jsonl", "engine.json", "server.log")


def spec_lines(spec):
    """One `spec.<field> = value` line per LaunchSpec field, then one
    `cell ctx= bs= = <shell line>` per bench cell: its env and bench_argv
    as the child runs them."""
    lines = [f"spec.{k} = {v!r}" for k, v in dataclasses.asdict(spec).items() if k != "cells"]
    return lines + [f"cell ctx={ctx} bs={bs} = {shlex.join(cell.child_argv(cell.bench_argv))}"
                    for ctx, bs, cell in spec.cells]


def in_docker(cfg, adapter):
    """Whether the server runs in docker; refused when nothing is launched,
    since the driver would then run the engine on the host."""
    if cfg.hardware.launcher != "docker":
        return False
    if not adapter.needs_server(cfg):
        raise ConfigError(f"{cfg.run_name}: hardware.launcher=docker needs a launched server; "
                          "a grid run sets engine.driver=http")
    return True


def dry_run(cfg, out):
    print(f"== {cfg.run_name}")
    print(cfg.dump(origins=True))
    if cfg.mode.mode != "probe":
        adapter = base.get(cfg.engine.name)
        spec = adapter.translate(cfg, out)
        print("\n".join(spec_lines(spec)))
        print(f"launch = {'server' if adapter.needs_server(cfg) else 'none, the driver runs it'}")
        if in_docker(cfg, adapter) and spec.server_argv:
            # the run takes idle GPUs; the dry run shows 0..n-1 unless visible_devices is set
            gpus = cfg.hardware.visible_devices or ",".join(
                str(i) for i in range(device_count(cfg.engine)))
            print(f"docker = {shlex.join(launcher.docker_argv(spec, cfg, gpus, out))}")


def run_probe(cfg, out):
    """The probe's main with its own flags and --out set to the run dir; a
    probe that raises leaves run.json failed (aborted on Ctrl-C)."""
    argv = [*shlex.split(cfg.mode.probe_args), "--out", str(out)]
    writer = RunWriter(out, cfg)
    try:
        importlib.import_module(f"quettabench.probes.{cfg.mode.probe}").main(argv)
    except BaseException as e:
        writer.fail(e)
        raise
    writer.finish({"probe": cfg.mode.probe, "argv": redact(argv)})


def with_child_env(spec, pins):
    """spec with pins added to its env and to every bench cell's env."""
    cells = tuple((ctx, bs, with_child_env(cell, pins)) for ctx, bs, cell in spec.cells)
    return dataclasses.replace(spec, env={**spec.env, **pins}, cells=cells)


def in_process(cfg, adapter, spec):
    """Whether the driver runs the engine in this process: no server and no bench child."""
    return not adapter.needs_server(cfg) and not spec.bench_argv


def refuse_mixed_in_process(cfgs):
    """One process cannot change its CUDA_VISIBLE_DEVICES or collective pins
    between matrix points, so in-process points must agree on them."""
    seen = set()
    for cfg in cfgs:
        if cfg.mode.mode == "probe":
            continue
        adapter = base.get(cfg.engine.name)
        if in_process(cfg, adapter, adapter.translate(cfg)):
            hw = cfg.hardware
            seen.add((hw.visible_devices or device_count(cfg.engine),
                      tuple(sorted((k, v) for k, v in hw.collective.items() if v))))
    if len(seen) > 1:
        raise ConfigError("in-process matrix points differ in GPUs or collective pins; "
                          "one process cannot change them, so run each point on its own")


def run_one(cfg, out):
    """One Config through launch, drive, serve and stop."""
    if cfg.mode.mode == "probe":
        return run_probe(cfg, out)
    adapter = base.get(cfg.engine.name)
    spec = adapter.translate(cfg, out)
    gpus = cfg.hardware.visible_devices or launcher.idle_gpus(device_count(cfg.engine),
                                                              cfg.hardware.backend)
    backend = "docker" if in_docker(cfg, adapter) else "local"
    pins = {k: v for k, v in cfg.hardware.collective.items() if v}
    if backend == "local":
        pins["CUDA_VISIBLE_DEVICES"] = gpus
    if in_process(cfg, adapter, spec):
        env_map.export(pins)
    spec = with_child_env(spec, pins)
    # Opened first, so a run that raises anywhere below still leaves run.json.
    writer = RunWriter(out, cfg, launch_env=spec.env)
    try:
        # A launched server's driver has no tokenizer, so its trace is built
        # first: a bad workload fails before a long model load, not after.
        trace = workloads.build(cfg) if adapter.needs_server(cfg) else None
        handle = (launcher.start(spec, out, backend=backend, cfg=cfg, gpus=gpus,
                                 timeout_s=cfg.hardware.ready_timeout_s)
                  if adapter.needs_server(cfg) else launcher.NoServer(out))
        try:
            driver = adapter.driver(spec, handle, trace)
            writer.bind(handle, driver)
            if trace is None:
                trace = workloads.build(cfg, tokenizer=getattr(driver, "tokenizer", None))
            serve.run(trace, driver, cfg.mode, writer, cfg.workload)
        finally:
            handle.stop()
    except BaseException as e:
        writer.fail(e)
        raise


def clear_run_dir(out, overwrite):
    """Refuse a run dir that already holds files, since requests.jsonl
    appends; with overwrite, remove the old run's own files instead."""
    if not out.is_dir() or not any(out.iterdir()):
        return
    if not overwrite:
        raise ConfigError(f"{out} already holds a run; pass --overwrite to replace it")
    for name in RUN_FILES:
        (out / name).unlink(missing_ok=True)


def main(run_yaml, overrides=(), dry=False, results_dir="results", allow_unverified=False,
         overwrite=False):
    """Every Config of run_yaml, in matrix order; results/<run_name>/ each."""
    sets = [*overrides, *(["allow_unverified=true"] if allow_unverified else [])]
    cfgs = loader.load(run_yaml, sets)
    if not dry:
        refuse_mixed_in_process(cfgs)
        for cfg in cfgs:
            clear_run_dir(Path(results_dir) / cfg.run_name, overwrite)
    for cfg in cfgs:
        out = Path(results_dir) / cfg.run_name
        if dry:
            dry_run(cfg, out)
        else:
            run_one(cfg, out)
    return 0
