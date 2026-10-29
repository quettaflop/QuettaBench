"""QuettaServe adapter: a serve run launches api and its dist secondaries; a grid run is one cargo bench run per cell."""

from pathlib import Path

from quettabench.config.schema import ConfigError
from quettabench.engines.base import (
    EngineAdapter,
    LaunchSpec,
    UnsupportedField,
    nsys_argv,
    register,
)
from quettabench.engines.drivers import BenchDriver, HttpDriver
from quettabench.engines.vllm import chat_requests, flag, switch, unless

SERVE_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "quettaserve-serve.sh"
# EngineCfg field -> scripts/quettaserve-serve.sh argv fragment.
SERVE_FIELDS = {
    "model": flag("--model"),
    "tp": flag("--tp"),
    "pp": flag("--pp"),
    "port": flag("--http-port"),
    "prefix_caching": switch("--prefix-cache"),
    "kv_cache_dtype": unless("auto", "--kv-cache-dtype"),
    "max_model_len": unless(None, "--max-context-length"),
}
# Flags the table or the script sets; engine.extra.quettaserve_serve may not repeat them.
SERVE_TAKEN = frozenset(
    {"--bin-dir", "--model", "--tp", "--pp", "--http-port", "--prefix-cache", "--kv-cache-dtype"}
    | {"--max-context-length", "--stagger", "--gpu", "--tokenizer", "--port", "--num-nodes", "--tensor-parallel"}
    | {"--http-addr"}
)
SERVE_ENV = {"RUST_LOG": "info"}  # api and dist log only errors without it
# api and dist read these when their flag is absent; dropped so the argv is the whole server config.
SERVE_ENV_DROP = (
    "DATABASE_URL",
    "KV_CACHE_DTYPE",
    "MAX_NUM_BATCHED_TOKENS",
    "PREFILL_CHUNK_SIZE",
    "KV_MEMORY_FRACTION",
)
SCALARS = (str, int, float, bool)

OK_BY_TEST = {
    "decode_loop": r"^LOOP ",
    "batch_bench": r"^\[batch_bench\] .*layers=all",
}


def moe_rule(p, extra, world):
    """The moe bench needs <P>_CFG and must not see an inherited <P>_LAYERS."""
    if not extra.get("quettaserve_config"):
        raise ConfigError(f"quettaserve moe: set engine.extra.quettaserve_config ({p}_CFG)")
    return {f"{p}_CFG": extra["quettaserve_config"]}, (f"{p}_LAYERS",), ""


def gdn_rule(p, extra, world):
    """Exact kernels need a vLLM GDN cubin dir; eager under exact, CUDA graphs under rust."""
    gdn_dir = extra.get("gdn_dir")
    kernels = extra.get("quettaserve_kernels") or ("exact" if gdn_dir else "rust")
    env = {"QS_KERNELS": kernels}
    if gdn_dir:
        env["QS_VLLM_GDN_DIR"] = gdn_dir
    if world >= 2:
        env["QS_CUSTOM_AR"] = str(extra.get("quettaserve_custom_allreduce") or "1")
    mode = extra.get("quettaserve_mode") or ("eager" if kernels == "exact" else "graph")
    env[f"{p}_MODE"] = mode
    header = f" kern={kernels} ar={env.get('QS_CUSTOM_AR', '0')} mode={mode} gdn_dir={gdn_dir or 'none'}"
    return env, (), header


FAMILY_RULES = {
    "moe": moe_rule,
    "gdn": gdn_rule,
    "dense": lambda p, extra, world: ({}, (), ""),
}


def serve_flags(flags):
    """engine.extra.quettaserve_serve as flags; the script gives --kv-memory-fraction to dist, the rest to api."""
    if not isinstance(flags, dict):
        raise UnsupportedField("quettaserve", "extra.quettaserve_serve", "expected a mapping of flag: value")
    argv = []
    for key, value in flags.items():
        name = "--" + key.replace("_", "-")
        if name in SERVE_TAKEN or not isinstance(value, SCALARS):
            why = "a table field" if name in SERVE_TAKEN else "value must be a scalar"
            raise UnsupportedField("quettaserve", f"extra.quettaserve_serve.{key}", why)
        argv += [name] if value is True else [] if value is False else [name, str(value)]
    return argv


# max_model_len has no bench flag; the grid loop reads it to SKIP cells past it.
BENCH_SUPPORTS = frozenset({"model", "tp", "extra", "instrumentation", "layers", "max_model_len"})
SERVE_SUPPORTS = frozenset(SERVE_FIELDS) | {"extra"}


class QuettaServeAdapter(EngineAdapter):
    name = "quettaserve"
    supports = BENCH_SUPPORTS | SERVE_SUPPORTS

    def translate(self, cfg, run_dir=None, cell=None):
        if self.needs_server(cfg):
            return self.serve(cfg)
        self.refuse_unsupported(cfg, BENCH_SUPPORTS)
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
            cwd=engine.extra.get("quettaserve_dir") or ".",
            instrumentation=engine.instrumentation,
            ok_pattern=ok,
            log_header=header,
            cells=cells,
        )

    def serve(self, cfg):
        """api on GPU 0 beside one dist secondary per GPU, through scripts/quettaserve-serve.sh."""
        self.refuse_unsupported(cfg, SERVE_SUPPORTS)
        engine = cfg.engine
        if engine.model is None:
            raise ConfigError("quettaserve: engine.model is required to launch")
        # The script refuses a dir without api and dist.
        bin_dir = engine.extra.get("quettaserve_bin_dir") or Path(
            engine.extra.get("quettaserve_dir") or ".", "target/release"
        )
        argv = [str(SERVE_SCRIPT), "--bin-dir", str(bin_dir)]
        for name, to_argv in SERVE_FIELDS.items():
            argv += to_argv(getattr(engine, name))
        flags = engine.extra.get("quettaserve_serve", {})
        argv += serve_flags(flags)
        return LaunchSpec(
            adapter=self.name,
            server_argv=tuple(argv),
            env=dict(SERVE_ENV),
            env_drop=SERVE_ENV_DROP,
            health_path="/health",
            served_name=str(flags.get("served_model_name") or engine.model),
            port=engine.port,
            timeout_s=cfg.mode.timeout,
        )

    def driver(self, spec, handle, trace=None):
        if not spec.server_argv:
            return BenchDriver(Path(handle.results_dir) / "bench.log", spec.cells)
        chat = chat_requests(trace)
        return HttpDriver(
            "chat" if chat else "completions",
            backend_name=self.name,
            url=handle.endpoint + ("/v1/chat/completions" if chat else "/v1/completions"),
            model=spec.served_name,
            timeout=spec.timeout_s,
            accelerator=getattr(handle, "backend", "cuda"),
        )


ADAPTER = register(QuettaServeAdapter())
