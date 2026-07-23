"""
OpenAI-compatible chat completions backend.

Works with: vLLM, SGLang, lmdeploy, and any server exposing
/v1/chat/completions with SSE streaming.
"""

import asyncio
import json
import time
from typing import Optional

import aiohttp

from ..benchmark.metrics import RequestResult


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    messages: list,
    max_tokens: int,
    temperature: float = 1.0,
    api_key: str = "test",
    extra_headers: Optional[dict] = None,
    ignore_eos: bool = False,
    request_id: Optional[str] = None,
) -> RequestResult:
    """
    Send a single streaming chat completion request and record metrics.

    Parses SSE stream: each `data: {...}` line whose choices[0].delta carries a
    generated payload (content, reasoning_content/reasoning, or tool_calls) is
    one token. First = TTFT. Subsequent = ITL entries.
    """
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
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        payload["ignore_eos"] = True
    if request_id:
        payload["request_id"] = request_id

    start_time = time.perf_counter()
    ttft = None
    itl = []
    last_token_time = None
    input_tokens = 0
    output_tokens = 0

    try:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                return RequestResult(
                    success=False,
                    e2el=time.perf_counter() - start_time,
                    error=f"HTTP {resp.status}: {body[:200]}",
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

                # Token counts from final chunk (stream_options: include_usage)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                    input_tokens = usage.get("prompt_tokens", 0)
                    output_tokens = usage.get("completion_tokens", 0)

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                # A chunk is a token event if it carries ANY generated payload,
                # not just `content`. Reasoning models (e.g. gpt-oss harmony on
                # vLLM) stream analysis-channel tokens as `reasoning_content`
                # (older) / `reasoning` (newer) with content=None, and tool-call
                # argument deltas arrive under `tool_calls` with content=None.
                # Skipping those coalesced whole reasoning phases into single
                # ITL gaps and inflated tpot up to ~13x (see QuettaSim
                # tools/GT_QUALITY_FLAGS.md, Finding 1).
                has_payload = (
                    delta.get("content") is not None
                    or delta.get("reasoning_content") is not None
                    or delta.get("reasoning") is not None
                    or bool(delta.get("tool_calls"))
                )
                if not has_payload:
                    continue

                # Real token received
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
        )

    except asyncio.CancelledError:
        raise
    except Exception as e:
        return RequestResult(
            success=False,
            e2el=time.perf_counter() - start_time,
            error=str(e),
        )


async def run_warmup(
    url: str,
    model: str,
    api_key: str,
    num_requests: int = 3,
    timeout: int = 60,
) -> None:
    """Send warmup requests and discard results."""
    warmup_messages = [{"role": "user", "content": "Hello"}]
    connector = aiohttp.TCPConnector(limit=num_requests)
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as session:
        tasks = [
            send_request(session, url, model, warmup_messages, max_tokens=10, api_key=api_key)
            for _ in range(num_requests)
        ]
        await asyncio.gather(*tasks)
