"""QuettaServe adapter: the cargo bench argv and the per-cell env bench.sh exports.

QuettaServe cannot listen yet, so there is no server_argv; the spec runs the
crate's bench test once per grid cell. Everything here is
scripts/crossval/bench.sh restated as data, with its env reads moved to
Config fields:

  <P>_CKPT              engine.model
  <P>_WORLD             engine.tp (bench.sh: <P>_WORLD, default the workload tp)
  <P>_TIMING_STEPS      hardware.timing_steps
  <P>_MAX_SEQ           ctx + hardware.timing_steps + hardware.max_seq_headroom
  <P>_CFG (moe)         engine.extra.qs_cfg
  <P>_BENCH_BIN         engine.extra.bench_bin (prebuilt test binary)
  QS_DIR                engine.extra.qs_dir (cwd of the bench; bench.sh default ".")
  QS_VLLM_GDN_DIR (gdn) engine.extra.gdn_dir
  QS_KERNELS, QS_CUSTOM_AR, <P>_MODE (gdn overrides)
                        engine.extra.qs_kernels, .qs_custom_ar, .qs_mode

Per-cell values come from a `cell=(ctx, bs)` argument to translate rather
than a for_cell(spec, ...) helper: they are a pure function of (cfg, ctx,
bs) and translate already holds cfg, while a helper would need the prefix,
steps and headroom carried again on the spec. translate(cfg) of a grid run
lists every cell's spec in spec.cells, the loop bench.sh ran.
"""

from pathlib import Path

from quettabench.config.schema import ConfigError
from quettabench.engines.base import (
    NSYS,
    EngineAdapter,
    LaunchSpec,
    UnsupportedField,
    register,
)
from quettabench.engines.drivers import BenchDriver

# bench.sh's pass pattern, per test (workloads.json "ok").
OK_BY_TEST = {
    "decode_loop": r"^LOOP ",
    "batch_bench": r"^\[batch_bench\] .*layers=all",
}


def moe_rule(p, extra, world):
    """bench.sh:18-22: the moe bench needs <P>_CFG and must not see <P>_LAYERS."""
    if not extra.get("qs_cfg"):
        raise ConfigError(f"quettaserve moe: set engine.extra.qs_cfg (bench.sh {p}_CFG)")
    return {f"{p}_CFG": extra["qs_cfg"]}, (f"{p}_LAYERS",), ""


def gdn_rule(p, extra, world):
    """bench.sh:47-61: exact kernels only when a vLLM GDN cubin dir is given
    (the loader panics without it), custom all-reduce at world >= 2, eager
    mode under exact kernels, CUDA graphs under rust kernels."""
    gdn_dir = extra.get("gdn_dir")
    kernels = extra.get("qs_kernels") or ("exact" if gdn_dir else "rust")
    env = {"QS_KERNELS": kernels}
    if gdn_dir:
        env["QS_VLLM_GDN_DIR"] = gdn_dir
    if world >= 2:
        env["QS_CUSTOM_AR"] = str(extra.get("qs_custom_ar") or "1")
    mode = extra.get("qs_mode") or ("eager" if kernels == "exact" else "graph")
    env[f"{p}_MODE"] = mode
    header = (f" kern={kernels} ar={env.get('QS_CUSTOM_AR', '0')} mode={mode}"
              f" gdn_dir={gdn_dir or 'none'}")
    return env, (), header


FAMILY_RULES = {
    "moe": moe_rule,
    "gdn": gdn_rule,
    "dense": lambda p, extra, world: ({}, (), ""),
}


class QuettaServeAdapter(EngineAdapter):
    name = "quettaserve"
    supports = frozenset({"model", "tp", "extra", "instrumentation"})

    def translate(self, cfg, cell=None):
        engine, hw = cfg.engine, cfg.hardware
        bench = engine.extra.get("quettaserve")
        if not bench:
            raise ConfigError("quettaserve: the model yaml has no quettaserve: block")
        if engine.model is None:
            raise ConfigError(f"quettaserve: engine.model is required ({bench['env_prefix']}_CKPT)")
        if engine.instrumentation == "torch":
            raise UnsupportedField("quettaserve", "instrumentation=torch", "no torch in the bench")
        if bench["family"] not in FAMILY_RULES:
            raise ConfigError(f"quettaserve: unknown family {bench['family']!r}")
        p, world, steps = bench["env_prefix"], engine.tp, hw.timing_steps
        algo = hw.collective.get("NCCL_ALGO") or "auto"
        proto = hw.collective.get("NCCL_PROTO") or "auto"
        env = {"XVAL_LINK_PROFILE": hw.link_profile}
        env.update({k: v for k, v in hw.collective.items() if v})
        family_env, drop, header_tail = FAMILY_RULES[bench["family"]](p, engine.extra, world)
        env.update(family_env)
        env.update({f"{p}_CKPT": engine.model, f"{p}_WORLD": str(world),
                    f"{p}_TIMING_STEPS": str(steps)})
        header = ""
        if cell is not None:
            ctx, bs = cell
            max_seq = ctx + steps + hw.max_seq_headroom
            env.update({f"{p}_BATCH": str(bs), f"{p}_BENCH_PROMPT": str(ctx),
                        f"{p}_MAX_SEQ": str(max_seq)})
            header = (f">>> ctx={ctx} bs={bs} world={world} max_seq={max_seq} "
                      f"link={hw.link_profile} nccl={algo}/{proto}{header_tail}")
        binary = engine.extra.get("bench_bin")
        argv = ([binary, "--nocapture"] if binary else
                ["cargo", "test", "-p", bench["crate"], "--test", bench["test"],
                 "--release", "--", "--nocapture"])
        if engine.instrumentation == "nsys":
            argv = [a.format(out=f"results/{cfg.run_name}/nsys") for a in NSYS] + argv
        cells = () if cell is not None or cfg.grid is None else tuple(
            (ctx, bs, self.translate(cfg, (ctx, bs))) for ctx, bs in cfg.grid.cells)
        return LaunchSpec(
            adapter=self.name,
            bench_argv=tuple(argv),
            env=env,
            env_drop=drop,
            cwd=engine.extra.get("qs_dir") or ".",
            instrumentation=engine.instrumentation,
            ok_pattern=OK_BY_TEST[bench["test"]],
            log_header=header,
            cells=cells,
        )

    def needs_server(self, cfg):
        """QuettaServe cannot listen yet: the bench is its own process per cell."""
        return False

    def driver(self, spec, handle):
        """Bench rows logged to <results_dir>/bench.log, as bench.sh's out-log."""
        return BenchDriver(Path(handle.results_dir) / "bench.log", spec.cells)


ADAPTER = register(QuettaServeAdapter())
