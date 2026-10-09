"""qbench run: a run yaml to results/<run>/, the same sequence every time.

load the yaml (+ --set, + env through env_map) into one Config per matrix
point, then per Config: translate it to a LaunchSpec, put the collective
env and the GPUs (hardware.visible_devices, else the first idle tp * pp)
into this process for the engine to inherit, launch the server unless the
adapter drives the engine itself (in-process grid, cargo bench), get the
driver, build the Trace, and let modes/serve.py run trace
through driver into a RunWriter; the server is stopped whatever happens.
mode: probe runs quettabench.probes.<name> with --out results/<run>/
instead and records it in the same run.json. --dry-run prints each Config
with its origins and its LaunchSpec, and starts nothing.
"""

import dataclasses
import importlib
import shlex
from pathlib import Path

from quettabench import workloads
from quettabench.config import env_map, loader
from quettabench.engines import base, launcher
from quettabench.metrics.results import RunWriter
from quettabench.modes import serve


def spec_lines(spec):
    """One `spec.<field> = value` line per LaunchSpec field."""
    return [f"spec.{k} = {v!r}" for k, v in dataclasses.asdict(spec).items()]


def dry_run(cfg):
    print(f"== {cfg.run_name}")
    print(cfg.dump(origins=True))
    if cfg.mode.mode != "probe":
        adapter = base.get(cfg.engine.name)
        print("\n".join(spec_lines(adapter.translate(cfg))))
        print(f"launch = {'server' if adapter.needs_server(cfg) else 'none, the driver runs it'}")


def run_probe(cfg, out):
    """The probe's main with its own flags and --out set to the run dir."""
    argv = [*shlex.split(cfg.mode.probe_args), "--out", str(out)]
    writer = RunWriter(out, cfg)
    importlib.import_module(f"quettabench.probes.{cfg.mode.probe}").main(argv)
    writer.finish({"probe": cfg.mode.probe, "argv": argv})


def run_one(cfg, out):
    """One Config through launch, drive, serve and stop."""
    if cfg.mode.mode == "probe":
        return run_probe(cfg, out)
    adapter = base.get(cfg.engine.name)
    spec = adapter.translate(cfg)
    gpus = cfg.hardware.visible_devices or launcher.idle_gpus(cfg.engine.tp * cfg.engine.pp,
                                                              cfg.hardware.backend)
    env_map.export({**cfg.hardware.collective, "CUDA_VISIBLE_DEVICES": gpus})
    handle = launcher.start(spec, out) if adapter.needs_server(cfg) else launcher.NoServer(out)
    try:
        driver = adapter.driver(spec, handle)
        trace = workloads.build(cfg, tokenizer=getattr(driver, "tokenizer", None))
        writer = RunWriter(out, cfg, handle=handle, driver=driver, launch_env=spec.env)
        serve.run(trace, driver, cfg.mode, writer, cfg.workload)
    finally:
        handle.stop()


def main(run_yaml, overrides=(), dry=False, results_dir="results", allow_unverified=False):
    """Every Config of run_yaml, in matrix order; results/<run_name>/ each."""
    sets = [*overrides, *(["allow_unverified=true"] if allow_unverified else [])]
    for cfg in loader.load(run_yaml, sets):
        if dry:
            dry_run(cfg)
        else:
            run_one(cfg, Path(results_dir) / cfg.run_name)
    return 0
