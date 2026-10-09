"""Engine adapters, plus serve.py's --backend names mapped onto vllm's HTTP client.

sglang resolves to the same client as vllm: the request path is identical,
which is what the old backend registry did too.
"""

from quettabench.engines import (  # noqa: F401  (registers the adapters)
    quettaserve,
    reference,
    vllm,
)
from quettabench.engines.drivers import HttpDriver

# --backend name -> (adapter, HTTP endpoint)
BACKENDS = {
    "openai": ("vllm", "chat"),
    "vllm": ("vllm", "chat"),
    "sglang": ("vllm", "chat"),
    "vllm-completions": ("vllm", "completions"),
}

SUPPORTED_BACKENDS = list(BACKENDS)


def get_backend(name: str, **server):
    """HttpDriver for a --backend name; server (url, model, api_key, timeout)
    binds it for send/run/reset."""
    name = name.lower()
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Choose from: {SUPPORTED_BACKENDS}")
    return HttpDriver(BACKENDS[name][1], name, **server)
