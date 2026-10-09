"""Launch adapter for SGLang's OpenAI-compatible server (sglang.launch_server).

Same shape as the TRT-LLM adapter: the benchmark client is engine-agnostic, so
an SGLang endpoint is measured through the exact same TTFT/TPOT path as vLLM and
the workload-identity gate holds across engines. That cross-engine determinism
is the claim a competitor cannot make. Unlike TRT-LLM, SGLang installs as a
normal package with no multi-hour engine build, so this launch runs directly.
"""

DEFAULT_PORT = 30000
HEALTH_PATH = "/health"


def launch_command(model, port=DEFAULT_PORT, tp=1, host="0.0.0.0"):
    """argv to serve a model over SGLang's OpenAI-compatible API. tp shards the
    model across GPUs; the caller pins CUDA_VISIBLE_DEVICES to the pool."""
    return ["python3", "-m", "sglang.launch_server", "--model-path", model,
            "--tp", str(tp), "--host", host, "--port", str(port)]


def health_url(port=DEFAULT_PORT, host="127.0.0.1"):
    """The readiness endpoint the benchmark polls before sending requests."""
    return f"http://{host}:{port}{HEALTH_PATH}"
