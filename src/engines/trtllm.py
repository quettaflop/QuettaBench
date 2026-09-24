"""Launch adapter for TensorRT-LLM's OpenAI-compatible server (trtllm-serve).

The client side is already engine-agnostic: openai_chat / openai_completions
speak to any OpenAI-compatible server, so a trtllm-serve endpoint is benchmarked
through the exact same metric path as vLLM. This module only builds the server
launch -- the serve command, the health endpoint, and the engine-directory
convention.

The one-time TensorRT engine build is multi-hour and compiles for the exact GPU,
so it is NOT automated here: build_command returns the command to run by hand
and the operator builds the engine before serving (documented in AGENTS.md).
"""

HEALTH_PATH = "/health"
DEFAULT_PORT = 8000


def launch_command(model, engine_dir, port=DEFAULT_PORT, tp=1):
    """argv to serve a prebuilt TensorRT engine over an OpenAI-compatible API.
    engine_dir must already hold a built engine (see build_command); serving a
    missing engine dir fails in trtllm-serve rather than silently falling back."""
    return ["trtllm-serve", model, "--engine_dir", engine_dir,
            "--tp_size", str(tp), "--host", "0.0.0.0", "--port", str(port)]


def build_command(checkpoint_dir, engine_dir, tp=1):
    """The one-time engine build. Returned for the operator to run, never
    launched automatically: it takes hours and is specific to the GPU."""
    return ["trtllm-build", "--checkpoint_dir", checkpoint_dir,
            "--output_dir", engine_dir, "--tp_size", str(tp)]


def health_url(port=DEFAULT_PORT, host="127.0.0.1"):
    """The readiness endpoint the benchmark polls before sending requests."""
    return f"http://{host}:{port}{HEALTH_PATH}"
