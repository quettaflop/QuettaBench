"""
Backend registry. Maps backend name → (send_request, run_warmup) functions.

Supported backends:
  openai       — any OpenAI-compatible /v1/chat/completions (vLLM, SGLang, lmdeploy)
  vllm         — alias for openai
  sglang       — alias for openai
  trtllm       — NVIDIA TensorRT-LLM /generate_stream
"""

from . import openai_chat, trtllm

_BACKENDS = {
    "openai": openai_chat,
    "vllm": openai_chat,
    "sglang": openai_chat,
    "trtllm": trtllm,
}

SUPPORTED_BACKENDS = list(_BACKENDS.keys())


def get_backend(name: str):
    """
    Return the backend module for the given name.

    Usage:
        backend = get_backend("vllm")
        result = await backend.send_request(session, url, model, messages, max_tokens)
        await backend.run_warmup(url, model, api_key, num_requests)
    """
    name = name.lower()
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Choose from: {SUPPORTED_BACKENDS}")
    return _BACKENDS[name]
