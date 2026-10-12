"""vLLM adapter: one field table gives the `vllm serve` argv and the in-process vllm.LLM kwargs.

quant adds no server flag, since the server reads it from the checkpoint; a slice run is engine.model naming a truncated checkpoint dir.
"""

from pathlib import Path

from quettabench.config.schema import ConfigError
from quettabench.engines.base import (
    EngineAdapter,
    LaunchSpec,
    UnsupportedField,
    nsys_argv,
    register,
)
from quettabench.engines.drivers import HttpDriver, InProcessDriver
from quettabench.workloads.sessions import SessionTrace

SERVER_CMD = "serve"
SERVER_FLAGS = ("--enable-chunked-prefill", "--trust-remote-code")
SERVER_ENV = {
    "VLLM_SERVER_DEV_MODE": "1",                      # mounts /reset_prefix_cache
    "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1",  # sizes KV after graph capture
}
KWARGS = {"disable_log_stats": True, "trust_remote_code": True}
# extra.vllm.api_key goes in this env var: vLLM's startup log would print --api-key
API_KEY_ENV = "VLLM_API_KEY"


def flag(name):
    return lambda v: [name, str(v)]


def switch(name):
    return lambda v: [name] if v else []


def switch_off(name):
    return lambda v: [] if v else [name]


def unless(default, name):
    """A flag only when the value differs from the engine's own default."""
    return lambda v: [] if v == default else [name, str(v)]


def kw(key):
    return lambda v: {key: v}


def kw_when(key, keep):
    return lambda v: {key: v} if keep(v) else {}


def nothing(v):
    return {}


def aggregated_only(v):
    if v != "aggregated":
        raise UnsupportedField("vllm", f"serving_style={v}", "no launch splits the GPUs into pools")
    return {}


# EngineCfg field -> (argv fragment, kwargs fragment).
FIELDS = {
    "model":          (lambda v: [],                       kw("model")),  # positional
    "port":           (flag("--port"),                     nothing),
    "tp":             (flag("--tensor-parallel-size"),     kw("tensor_parallel_size")),
    "pp":             (unless(1, "--pipeline-parallel-size"), kw("pipeline_parallel_size")),
    "ep":             (switch("--enable-expert-parallel"), kw_when("enable_expert_parallel", bool)),
    "dtype":          (flag("--dtype"),                    kw("dtype")),
    "kv_dtype":       (unless("auto", "--kv-cache-dtype"), kw_when("kv_cache_dtype", lambda v: v != "auto")),
    "quant":          (lambda v: [],                       kw_when("quantization", bool)),  # server reads it from the checkpoint
    "max_model_len":  (flag("--max-model-len"),            kw("max_model_len")),
    "gpu_util":       (flag("--gpu-memory-utilization"),   kw("gpu_memory_utilization")),
    "prefix_caching": (lambda v: ["--enable-prefix-caching" if v else "--no-enable-prefix-caching"],
                       kw("enable_prefix_caching")),
    "cudagraph":      (switch_off("--enforce-eager"),      lambda v: {} if v else
                       {"compilation_config": {"cudagraph_mode": "NONE"}}),
    "serving_style":  (lambda v: [],                       aggregated_only),
}
SCALARS = (str, int, float, bool)


def extra_fragments(extra, taken):
    """argv and kwargs for engine.extra.vllm: {kwarg_name: scalar}, appended last; api_key is left to the env (API_KEY_ENV)."""
    if not isinstance(extra, dict):
        raise UnsupportedField("vllm", "extra.vllm", "expected a mapping of flag: value")
    argv, kwargs = [], {}
    for key, value in extra.items():
        if key == "api_key":
            continue
        if not isinstance(value, SCALARS) or key in taken:
            raise UnsupportedField("vllm", f"extra.vllm.{key}",
                                   "a table field" if key in taken else "value must be a scalar")
        name = "--" + key.replace("_", "-")
        argv += [name] if value is True else [] if value is False else [name, str(value)]
        kwargs[key] = value
    return argv, kwargs


class VllmAdapter(EngineAdapter):
    name = "vllm"
    supports = frozenset(FIELDS) | {"extra", "instrumentation", "image"}

    def translate(self, cfg, run_dir=None):
        self.refuse_unsupported(cfg)
        engine = cfg.engine
        if engine.model is None:
            raise ConfigError("vllm: engine.model is required to launch")
        if engine.instrumentation == "torch":
            raise UnsupportedField("vllm", "instrumentation=torch", "no profiler env var in the repo")
        if engine.layers is not None:
            raise UnsupportedField("vllm", f"layers={engine.layers}",
                                   "slice runs load a truncated checkpoint dir as engine.model")
        argv, kwargs = [], dict(KWARGS)
        for name, (to_argv, to_kwargs) in FIELDS.items():
            value = getattr(engine, name)
            argv += to_argv(value)
            kwargs.update(to_kwargs(value))
        extra = engine.extra.get("vllm", {})
        extra_argv, extra_kwargs = extra_fragments(extra, kwargs)
        kwargs.update(extra_kwargs)
        env = dict(SERVER_ENV)
        if extra.get("api_key") is not None:
            env[API_KEY_ENV] = str(extra["api_key"])
        py = cfg.hardware.provision.get("py")
        vllm_bin = str(Path(py).with_name("vllm")) if py else "vllm"
        server = [vllm_bin, SERVER_CMD, engine.model, *argv, *SERVER_FLAGS, *extra_argv]
        if engine.instrumentation == "nsys":
            server = nsys_argv(cfg, run_dir) + server
        return LaunchSpec(
            adapter=self.name,
            server_argv=tuple(server),
            engine_kwargs=kwargs,
            env=env,
            health_path="/health",
            port=engine.port,
            image=engine.image,
            instrumentation=engine.instrumentation,
            timeout_s=cfg.mode.timeout,
        )

    def needs_server(self, cfg):
        """A serve run, or a grid run that only times. A grid run with mode.returns builds vllm.LLM itself: only that path gives token ids, text or logits (tests/parity/slope_gate.json passed timing over HTTP)."""
        return cfg.mode.grid is None or cfg.mode.returns == "none"

    def driver(self, spec, handle, trace=None):
        """InProcessDriver when nothing was launched (a grid run with mode.returns), else the launched server with the model it was started on: its /v1/chat/completions for chat requests, /v1/completions otherwise, sending the server's api key when it was started with one."""
        if handle.endpoint is None:
            return InProcessDriver(spec)
        endpoint = "chat" if chat_requests(trace) else "completions"
        path = "/v1/chat/completions" if endpoint == "chat" else "/v1/completions"
        return HttpDriver(endpoint, url=handle.endpoint + path, model=spec.engine_kwargs["model"],
                          api_key=spec.env.get(API_KEY_ENV, "test"), timeout=spec.timeout_s,
                          accelerator=getattr(handle, "backend", "cuda"))


def chat_requests(trace):
    """Whether trace is chat: multi-turn sessions, whose turns carry the whole conversation and take the engine's reply text, or requests that carry messages and no token ids. The completions body would send only the last user message of such a request."""
    if trace is None:
        return False
    if isinstance(trace, SessionTrace):
        return True
    requests = trace.requests()
    return bool(requests) and all(r.messages and not r.prompt_token_ids for r in requests)


ADAPTER = register(VllmAdapter())
