"""
Server-side state control between benchmark cells.

A sweep launches one server and then runs every (profile, concurrency) cell
against it. The workload is deterministic -- the dataset seed is fixed and the
distributional sampler seeds each message from its label -- so cell N+1 replays
byte-identical prompts that cell N just served. With prefix caching enabled the
later cell reads them straight out of the KV cache and its TTFT is measured
warm, while the first cell paid the cold price. That gradient is an artifact of
sweep ordering, not of concurrency.

Resetting the prefix cache immediately before each run removes the gradient
while keeping the workload identical across cells. Intra-run reuse (the point
of the multi-turn profiles) is unaffected.
"""

from typing import Optional
from urllib.parse import urlsplit, urlunsplit


# Endpoint verified against the vLLM build in use:
# vllm/entrypoints/serve/cache/api_router.py -> @router.post("/reset_prefix_cache")
_RESET_ENDPOINTS = {
    "vllm": "/reset_prefix_cache",
    "openai": "/reset_prefix_cache",
}


class PrefixCacheResetError(RuntimeError):
    """Raised when a requested cache reset could not be performed."""


def server_base_url(url: str) -> str:
    """Strip the request path off an endpoint URL, keeping scheme://host:port."""
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        raise PrefixCacheResetError(f"Cannot derive server base URL from {url!r}")
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


def reset_endpoint_for(backend_name: str) -> Optional[str]:
    """Return the cache-reset path for a backend, or None if it has none known."""
    return _RESET_ENDPOINTS.get(backend_name.lower())


async def reset_prefix_cache(
    session,
    url: str,
    backend_name: str,
    api_key: str = "test",
    timeout: int = 60,
) -> str:
    """
    Reset the server's prefix cache. Returns a short status string for logging.

    Raises PrefixCacheResetError if the backend has no known reset endpoint or
    the server rejects the call -- silently continuing would produce exactly the
    cross-cell contamination this exists to prevent.
    """
    import aiohttp

    endpoint = reset_endpoint_for(backend_name)
    if endpoint is None:
        raise PrefixCacheResetError(
            f"--reset-prefix-cache is not supported for backend '{backend_name}': "
            f"no verified reset endpoint. Known: {sorted(_RESET_ENDPOINTS)}. "
            f"Re-run without the flag (and expect warm-cache bias across cells), "
            f"or add the backend's endpoint once it has been confirmed."
        )

    reset_url = f"{server_base_url(url)}{endpoint}"
    try:
        async with session.post(
            reset_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status != 200:
                body = (await resp.text())[:200]
                raise PrefixCacheResetError(
                    f"POST {reset_url} returned HTTP {resp.status}: {body}"
                )
    except PrefixCacheResetError:
        raise
    except Exception as e:  # network/timeout
        raise PrefixCacheResetError(f"POST {reset_url} failed: {e}") from e

    return f"prefix cache reset via {reset_url}"
