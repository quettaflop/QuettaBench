"""OpenAI-compatible legacy completions backend.

Posts to /v1/completions. The prompt is a token id list when provided
(trace replay needs exact tokens, no chat template), else the last user
message as raw text. Streaming and errors mirror openai_chat.
"""

import asyncio
import json
import time
from typing import Optional

import aiohttp

from ..benchmark.metrics import RequestResult


def resolve_prompt(prompt_token_ids, messages):
    """Token ids when present, else the last user message text."""
    if prompt_token_ids:
        return prompt_token_ids
    for m in reversed(messages or []):
        if m.get("role") == "user" and m.get("content"):
            return m["content"]
    return None


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    messages: list,
    max_tokens: int,
    temperature: float = 0.0,
    api_key: str = "test",
    extra_headers: Optional[dict] = None,
    ignore_eos: bool = False,
    request_id: Optional[str] = None,
    seed: Optional[int] = None,
    min_tokens: int = 0,
    prompt_token_ids: Optional[list[int]] = None,
) -> RequestResult:
    """Send one streaming completion from token ids or raw text."""
    prompt = resolve_prompt(prompt_token_ids, messages)
    if prompt is None:
        return RequestResult(
            success=False,
            e2el=0.0,
            error="completions backend needs prompt_token_ids or a user message",
            error_kind="client_error",
        )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if extra_headers:
        headers.update(extra_headers)
    if request_id:
        headers["X-Request-Id"] = request_id

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        payload["ignore_eos"] = True
    if request_id:
        payload["request_id"] = request_id
    if seed is not None:
        payload["seed"] = seed
    if min_tokens > 0:
        payload["min_tokens"] = min_tokens

    start_time = time.perf_counter()
    ttft = None
    itl = []
    last_token_time = None
    input_tokens = 0
    output_tokens = 0
    usage_reported = False

    try:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                return RequestResult(
                    success=False,
                    e2el=time.perf_counter() - start_time,
                    error=f"HTTP {resp.status}: {body[:200]}",
                    error_kind="http_error",
                )

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8").strip()

                if not line.startswith("data:"):
                    continue

                data_str = line[len("data:"):].strip()

                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                now = time.perf_counter()

                if chunk.get("usage"):
                    usage = chunk["usage"]
                    input_tokens = usage.get("prompt_tokens", 0)
                    output_tokens = usage.get("completion_tokens", 0)
                    usage_reported = True

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                # A chunk with empty text is not a token event.
                if choices[0].get("text"):
                    if ttft is None:
                        ttft = now - start_time
                        last_token_time = now
                    else:
                        itl.append(now - last_token_time)
                        last_token_time = now

        e2el = time.perf_counter() - start_time
        return RequestResult(
            success=True,
            ttft=ttft,
            itl=itl,
            e2el=e2el,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usage_reported=usage_reported,
        )

    except asyncio.CancelledError:
        raise
    except Exception as e:
        # Keep partial timings, a truncated tail is still evidence.
        is_timeout = isinstance(e, asyncio.TimeoutError)
        return RequestResult(
            success=False,
            ttft=ttft,
            itl=itl,
            e2el=time.perf_counter() - start_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usage_reported=usage_reported,
            error=f"timeout after {time.perf_counter() - start_time:.1f}s" if is_timeout else str(e),
            error_kind="timeout" if is_timeout else "client_error",
        )
