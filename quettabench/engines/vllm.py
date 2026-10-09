"""vLLM adapter: server argv and in-process kwargs from one field table.

No vLLM version is pinned (tests/golden/REPORT.md), so every flag and kwarg
here is one a launch on real GPUs already used:

  argv   scripts/launch_server.sh:86-96 and scripts/sweep_all_profiles.sh:109-119
         (python -m vllm.entrypoints.openai.api_server, --model, --port,
         --max-model-len, --gpu-memory-utilization, --dtype,
         --tensor-parallel-size, --enable-chunked-prefill,
         --enable-prefix-caching, --trust-remote-code), sweep_all_profiles.sh:102
         (--enable-expert-parallel), sweep_all_profiles.sh:19,26 (env)
  kwargs quettabench/modes/vmin_fit.py LLM(...) and legacy/trace_serve.py
         AsyncEngineArgs(...) as they stood at 708cdbe
  nsys   scripts/crossval/prof.sh:16

A field whose value has no flag in those launches lands in
server_unsupported: the launcher refuses the spec, the in-process path still
builds. Today that is pp > 1, kv_dtype, quant and cudagraph=False (the
sweeps' --enforce-eager is a different switch, not proven equal to
cudagraph_mode NONE). serving_style other than aggregated and
instrumentation=torch have no flag or kwarg anywhere and raise.

The two sides differ by design in their constants: the server gets
--enable-chunked-prefill and --trust-remote-code as both scripts pass them;
the kwargs get disable_log_stats and trust_remote_code as vmin_fit passes them.
"""

from quettabench.config.schema import ConfigError
from quettabench.engines.base import (
    NSYS,
    EngineAdapter,
    LaunchSpec,
    UnsupportedField,
    register,
)

SERVER_CMD = ("-m", "vllm.entrypoints.openai.api_server")
SERVER_FLAGS = ("--enable-chunked-prefill", "--trust-remote-code")
SERVER_ENV = {
    "VLLM_SERVER_DEV_MODE": "1",                      # mounts /reset_prefix_cache
    "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1",  # sizes KV after graph capture
}
KWARGS = {"disable_log_stats": True, "trust_remote_code": True}


def flag(name):
    return lambda v: [name, str(v)]


def switch(name):
    return lambda v: [name] if v else []


def only(default):
    """No flag exists: the default value adds nothing, any other is unsupported."""
    return lambda v: [] if v == default else None


def kw(key):
    return lambda v: {key: v}


def kw_when(key, keep):
    return lambda v: {key: v} if keep(v) else {}


def nothing(v):
    return {}


def aggregated_only(v):
    if v != "aggregated":
        raise UnsupportedField("vllm", f"serving_style={v}", "no launch splits prefill and decode")
    return {}


# EngineCfg field -> (argv fragment, kwargs fragment). An argv builder that
# returns None means no flag carries that value.
FIELDS = {
    "model":          (flag("--model"),                    kw("model")),
    "port":           (flag("--port"),                     nothing),
    "tp":             (flag("--tensor-parallel-size"),     kw("tensor_parallel_size")),
    "pp":             (only(1),                            kw("pipeline_parallel_size")),
    "ep":             (switch("--enable-expert-parallel"), kw_when("enable_expert_parallel", bool)),
    "dtype":          (flag("--dtype"),                    kw("dtype")),
    "kv_dtype":       (only("auto"),                       kw_when("kv_cache_dtype", lambda v: v != "auto")),
    "quant":          (only(None),                         kw_when("quantization", bool)),
    "max_model_len":  (flag("--max-model-len"),            kw("max_model_len")),
    "gpu_util":       (flag("--gpu-memory-utilization"),   kw("gpu_memory_utilization")),
    "prefix_caching": (switch("--enable-prefix-caching"),  kw("enable_prefix_caching")),
    "cudagraph":      (only(True),                         lambda v: {} if v else
                       {"compilation_config": {"cudagraph_mode": "NONE"}}),
    "serving_style":  (only("aggregated"),                 aggregated_only),
}
SCALARS = (str, int, float, bool)


def extra_fragments(extra, taken):
    """argv and kwargs for engine.extra.vllm: {kwarg_name: scalar}, appended last."""
    if not isinstance(extra, dict):
        raise UnsupportedField("vllm", "extra.vllm", "expected a mapping of flag: value")
    argv, kwargs = [], {}
    for key, value in extra.items():
        if not isinstance(value, SCALARS) or key in taken:
            raise UnsupportedField("vllm", f"extra.vllm.{key}",
                                   "a table field" if key in taken else "value must be a scalar")
        name = "--" + key.replace("_", "-")
        argv += [name] if value is True else [] if value is False else [name, str(value)]
        kwargs[key] = value
    return argv, kwargs


class VllmAdapter(EngineAdapter):
    name = "vllm"
    supports = frozenset(FIELDS) | {"extra", "instrumentation"}

    def translate(self, cfg):
        engine = cfg.engine
        if engine.model is None:
            raise ConfigError("vllm: engine.model is required to launch")
        if engine.instrumentation == "torch":
            raise UnsupportedField("vllm", "instrumentation=torch", "no profiler env var in the repo")
        argv, kwargs, unsupported = [], dict(KWARGS), []
        for name, (to_argv, to_kwargs) in FIELDS.items():
            value = getattr(engine, name)
            part = to_argv(value)
            if part is None:
                unsupported.append(f"{name}={value}")
            else:
                argv += part
            kwargs.update(to_kwargs(value))
        extra_argv, extra_kwargs = extra_fragments(engine.extra.get("vllm", {}), kwargs)
        kwargs.update(extra_kwargs)
        py = cfg.hardware.provision.get("py") or "python3"
        server = [py, *SERVER_CMD, *argv, *SERVER_FLAGS, *extra_argv]
        if engine.instrumentation == "nsys":
            out = f"results/{cfg.run_name}/nsys"
            server = [a.format(out=out) for a in NSYS] + server
        return LaunchSpec(
            adapter=self.name,
            server_argv=() if unsupported else tuple(server),
            server_unsupported=tuple(unsupported),
            engine_kwargs=kwargs,
            env=dict(SERVER_ENV),
            health_path="/health",
            port=engine.port,
            instrumentation=engine.instrumentation,
        )

    def driver(self, spec, handle):
        raise NotImplementedError("vllm drivers land with engines/drivers.py")


ADAPTER = register(VllmAdapter())
