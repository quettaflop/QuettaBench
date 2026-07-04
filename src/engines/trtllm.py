"""
NVIDIA TensorRT-LLM backend.

TRT-LLM exposes a different API from OpenAI:
  - Endpoint: /generate_stream
  - Request: {"text_input": "...", "max_tokens": N, "stream": true, "accumulate_tokens": true}
  - Response: SSE chunks with {"text_output": "..."}
  - No chat template applied server-side — we must format messages ourselves.
  - No usage field — token counts estimated via whitespace split (approximate).

TODO: pass tokenizer for accurate token counting.
"""

import asyncio
import json
import time
from typing import Optional

import aiohttp

from ..benchmark.metrics import RequestResult


def _format_messages(messages: list) -> str:
    """
    Format a messages list into a single text string for TRT-LLM.

    Uses a simple role-prefix format. For production use, apply the model's
    actual chat template via tokenizer.apply_chat_template().
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    parts.append("Assistant:")  # prompt the model to respond
    return "\n".join(parts)


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
) -> RequestResult:
    """
    Send a single streaming request to TRT-LLM and record metrics.

    URL should point to /generate_stream endpoint, e.g.:
        http://localhost:8000/generate_stream
    """
    headers = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)

    text_input = _format_messages(messages)

    payload = {
        "text_input": text_input,
        "max_tokens": max_tokens,
        "temperature": max(temperature, 0.01),  # TRT-LLM may not accept 0.0
        "top_p": 1.0,
        "stream": True,
        "accumulate_tokens": True,
    }

    start_time = time.perf_counter()
    ttft = None
    itl = []
    last_token_time = None
    generated_text = ""

    try:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                return RequestResult(
                    success=False,
                    e2el=time.perf_counter() - start_time,
                    error=f"HTTP {resp.status}: {body[:200]}",
                )

            async for chunk_bytes in resp.content:
                chunk_bytes = chunk_bytes.strip()
                if not chunk_bytes:
                    continue

                # TRT-LLM SSE: "data: {...}" or just "{...}"
                line = chunk_bytes.decode("utf-8")
                if line.startswith("data:"):
                    line = line[len("data:"):].strip()

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text_output = data.get("text_output", "")
                if not text_output:
                    continue

                now = time.perf_counter()
                generated_text = text_output  # accumulate_tokens=True means this is the full text so far

                if ttft is None:
                    ttft = now - start_time
                    last_token_time = now
                else:
                    itl.append(now - last_token_time)
                    last_token_time = now

        e2el = time.perf_counter() - start_time

        # Estimate token counts (no usage field from TRT-LLM)
        # Rough approximation: words ≈ tokens. Replace with tokenizer for accuracy.
        input_tokens = len(text_input.split())
        output_tokens = len(generated_text.split())

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
            send_request(session, url, model, warmup_messages, max_tokens=10)
            for _ in range(num_requests)
        ]
        await asyncio.gather(*tasks)
