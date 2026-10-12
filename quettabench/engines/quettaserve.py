"""QuettaServe adapter: one cargo bench run per grid cell; the engine cannot listen yet."""

from pathlib import Path

from quettabench.config.schema import ConfigError
from quettabench.engines.base import (
    EngineAdapter,
    LaunchSpec,
    UnsupportedField,
    nsys_argv,
    register,
)
from quettabench.engines.drivers import BenchDriver

OK_BY_TEST = {
    "decode_loop": r"^LOOP ",
    "batch_bench": r"^\[batch_bench\] .*layers=all",
}


def moe_rule(p, extra, world):
    """The moe bench needs <P>_CFG and must not see an inherited <P>_LAYERS."""
    if not extra.get("qs_cfg"):
        raise ConfigError(f"quettaserve moe: set engine.extra.qs_cfg ({p}_CFG)")
    return {f"{p}_CFG": extra["qs_cfg"]}, (f"{p}_LAYERS",), ""


def gdn_rule(p, extra, world):
    """Exact kernels need a vLLM GDN cubin dir; eager under exact, CUDA graphs under rust."""
    gdn_dir = extra.get("gdn_dir")
    kernels = extra.get("qs_kernels") or ("exact" if gdn_dir else "rust")
    env = {"QS_KERNELS": kernels}
    if gdn_dir:
        env["QS_VLLM_GDN_DIR"] = gdn_dir
    if world >= 2:
        env["QS_CUSTOM_AR"] = str(extra.get("qs_custom_ar") or "1")
    mode = extra.get("qs_mode") or ("eager" if kernels == "exact" else "graph")
    env[f"{p}_MODE"] = mode
    header = f" kern={kernels} ar={env.get('QS_CUSTOM_AR', '0')} mode={mode} gdn_dir={gdn_dir or 'none'}"
    return env, (), header


FAMILY_RULES = {
    "moe": moe_rule,
    "gdn": gdn_rule,
    "dense": lambda p, extra, world: ({}, (), ""),
}


class QuettaServeAdapter(EngineAdapter):
    name = "quettaserve"
    # max_model_len has no bench flag; the grid loop reads it to SKIP cells past it.
    supports = frozenset({"model", "tp", "extra", "instrumentation", "layers", "max_model_len"})

    def translate(self, cfg, run_dir=None, cell=None):
        self.refuse_unsupported(cfg)
        engine, hw = cfg.engine, cfg.hardware
        bench = engine.extra.get("quettaserve")
        if not bench:
            raise ConfigError("quettaserve: the model yaml has no quettaserve: block")
        if engine.model is None:
            raise ConfigError(f"quettaserve: engine.model is required ({bench['env_prefix']}_CKPT)")
        if engine.instrumentation == "torch":
            raise UnsupportedField("quettaserve", "instrumentation=torch", "no torch in the bench")
        if engine.serving_style != "aggregated":
            raise UnsupportedField(
                "quettaserve", f"serving_style={engine.serving_style}", "the bench runs one aggregated world"
            )
        if bench["family"] not in FAMILY_RULES:
            raise ConfigError(f"quettaserve: unknown family {bench['family']!r}")
        p, world, steps = bench["env_prefix"], engine.tp, hw.timing_steps
        algo = hw.collective.get("NCCL_ALGO") or "auto"
        proto = hw.collective.get("NCCL_PROTO") or "auto"
        env = {"XVAL_LINK_PROFILE": hw.link_profile}
        env.update({k: v for k, v in hw.collective.items() if v})
        family_env, drop, header_tail = FAMILY_RULES[bench["family"]](p, engine.extra, world)
        env.update(family_env)
        ok = OK_BY_TEST[bench["test"]]
        if engine.layers is not None:
            if bench["test"] != "batch_bench":
                raise UnsupportedField("quettaserve", f"layers={engine.layers}", f"only batch_bench reads {p}_LAYERS")
            env[f"{p}_LAYERS"] = str(engine.layers)
            drop = tuple(d for d in drop if d != f"{p}_LAYERS")
            ok = ok.replace("layers=all", f"layers={engine.layers}")
        env.update({f"{p}_CKPT": engine.model, f"{p}_WORLD": str(world), f"{p}_TIMING_STEPS": str(steps)})
        header = ""
        if cell is not None:
            ctx, bs = cell
            max_seq = ctx + steps + hw.max_seq_headroom
            env.update({f"{p}_BATCH": str(bs), f"{p}_BENCH_PROMPT": str(ctx), f"{p}_MAX_SEQ": str(max_seq)})
            header = (
                f">>> ctx={ctx} bs={bs} world={world} max_seq={max_seq} "
                f"link={hw.link_profile} nccl={algo}/{proto}{header_tail}"
            )
        binary = engine.extra.get("bench_bin")
        argv = (
            [binary, "--nocapture"]
            if binary
            else ["cargo", "test", "-p", bench["crate"], "--test", bench["test"], "--release", "--", "--nocapture"]
        )
        if engine.instrumentation == "nsys":
            name = "nsys" if cell is None else f"nsys-{cell[0]}x{cell[1]}"
            argv = nsys_argv(cfg, run_dir, name) + argv
        cells = (
            ()
            if cell is not None or cfg.grid is None
            else tuple((ctx, bs, self.translate(cfg, run_dir, (ctx, bs))) for ctx, bs in cfg.grid.cells)
        )
        return LaunchSpec(
            adapter=self.name,
            bench_argv=tuple(argv),
            env=env,
            env_drop=drop,
            cwd=engine.extra.get("qs_dir") or ".",
            instrumentation=engine.instrumentation,
            ok_pattern=ok,
            log_header=header,
            cells=cells,
        )

    def needs_server(self, cfg):
        return False

    def driver(self, spec, handle, trace=None):
        return BenchDriver(Path(handle.results_dir) / "bench.log", spec.cells)


ADAPTER = register(QuettaServeAdapter())
