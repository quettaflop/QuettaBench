"""
Backend registry. Maps backend name → module exposing send_request().

Supported backends:
  openai       — any OpenAI-compatible /v1/chat/completions (vLLM, SGLang, lmdeploy)
  vllm         — alias for openai
  sglang       — alias for openai
  vllm-completions — /v1/completions, takes token id or raw text prompts
"""

from . import openai_chat
from . import openai_completions

_BACKENDS = {
    "openai": openai_chat,
    "vllm": openai_chat,
    "sglang": openai_chat,
    "vllm-completions": openai_completions,
}

SUPPORTED_BACKENDS = list(_BACKENDS.keys())


def get_backend(name: str):
    """
    Return the backend module for the given name.

    Usage:
        backend = get_backend("vllm")
        result = await backend.send_request(session, url, model, messages, max_tokens)
    """
    name = name.lower()
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Choose from: {SUPPORTED_BACKENDS}")
    return _BACKENDS[name]
