"""How work reaches an engine: one Driver per way of driving it.

InProcessDriver builds vLLM inside this process from spec.engine_kwargs.
BenchDriver runs a QuettaServe bench per grid cell, as bench.sh does, and
reads its rows with crossval/table.py's parsers. HttpDriver streams
OpenAI-compatible requests at a launched server. Its two
endpoints keep their own first-token rules, recorded on every result as
ttft_rule:

  chat         first_payload_chunk  the first SSE chunk whose delta carries
                                    content, reasoning(_content) or tool_calls
  completions  first_nonempty_text  the first SSE chunk whose choices[0].text
                                    is non-empty

The stream bodies are openai_chat.send_request and
openai_completions.send_request moved unchanged; only the header and payload
building they shared byte for byte is factored out.
"""

import asyncio
import json
import re
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

import aiohttp

from quettabench.crossval import table
from quettabench.metrics.core import RequestResult


class Driver(ABC):
    @abstractmethod
    def run(self, requests, *, concurrency, **kwargs):
        """Yield one RequestResult per request."""


def _headers(api_key, extra_headers, request_id):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if extra_headers:
        headers.update(extra_headers)
    if request_id:
        headers["X-Request-Id"] = request_id
    return headers


def _payload(model, prompt, max_tokens, temperature, ignore_eos, request_id, seed, min_tokens):
    """prompt is {"messages": ...} for chat or {"prompt": ...} for completions."""
    payload = {
        "model": model,
        **prompt,
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
    return payload


async def _stream_chat(
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
    capture_text: bool = False,
) -> RequestResult:
    """
    Send a single streaming chat completion request and record metrics.

    Parses SSE stream: each `data: {...}` line whose choices[0].delta carries a
    generated payload (content, reasoning_content/reasoning, or tool_calls) is
    one token. First = TTFT. Subsequent = ITL entries.

    Sampling is caller-controlled: `temperature` defaults to greedy and `seed`
    is forwarded, so re-running the same workload produces the same output
    lengths. `min_tokens` (with ignore_eos) pins the generated length exactly.

    On failure the partial timings collected before the stream died are kept on
    the result rather than discarded -- a timeout on the slowest request is
    exactly the datapoint you need to see.
    """
    headers = _headers(api_key, extra_headers, request_id)
    payload = _payload(model, {"messages": messages}, max_tokens, temperature, ignore_eos,
                       request_id, seed, min_tokens)

    start_time = time.perf_counter()
    ttft = None
    itl = []
    last_token_time = None
    input_tokens = 0
    output_tokens = 0
    usage_reported = False
    text_parts: list[str] = []

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

                # Token counts from final chunk (stream_options: include_usage)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                    input_tokens = usage.get("prompt_tokens", 0)
                    output_tokens = usage.get("completion_tokens", 0)
                    usage_reported = True

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

                if capture_text:
                    # Transcript is content only; reasoning/tool_calls are not sent back.
                    piece = delta.get("content")
                    if piece:
                        text_parts.append(piece)

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
            usage_reported=usage_reported,
            generated_text="".join(text_parts) if capture_text else None,
        )

    except asyncio.CancelledError:
        raise
    except Exception as e:
        # Keep whatever the stream already delivered. A timeout truncates the
        # slowest requests, so discarding their partial TTFT/ITL would erase
        # the only evidence that the tail was cut off.
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


def resolve_prompt(prompt_token_ids, messages):
    """Token ids when present, else the last user message text."""
    if prompt_token_ids:
        return prompt_token_ids
    for m in reversed(messages or []):
        if m.get("role") == "user" and m.get("content"):
            return m["content"]
    return None


async def _stream_completions(
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

    headers = _headers(api_key, extra_headers, request_id)
    payload = _payload(model, {"prompt": prompt}, max_tokens, temperature, ignore_eos,
                       request_id, seed, min_tokens)

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


# endpoint -> (stream body, first-token rule it implements)
ENDPOINTS = {
    "chat": (_stream_chat, "first_payload_chunk"),
    "completions": (_stream_completions, "first_nonempty_text"),
}

# Endpoint verified against the vLLM builds in use (0.19.1 and 0.26.0):
#   0.19.1: vllm/entrypoints/serve/cache/api_router.py
#   0.26.0: vllm/entrypoints/serve/dev/cache/api_router.py
# In BOTH, the router is mounted only under dev mode:
#   def attach_router(app):
#       if not envs.VLLM_SERVER_DEV_MODE: return
# so the server must be launched with VLLM_SERVER_DEV_MODE=1 or this 404s.
_RESET_ENDPOINTS = {
    "vllm": "/reset_prefix_cache",
    "vllm-completions": "/reset_prefix_cache",
    "openai": "/reset_prefix_cache",
}

_DEV_MODE_HINT = (
    "The endpoint is gated behind dev mode in every vLLM build checked "
    "(0.19.1 and 0.26.0): serve/**/cache/api_router.py returns early from "
    "attach_router() unless VLLM_SERVER_DEV_MODE=1. Relaunch the server with "
    "VLLM_SERVER_DEV_MODE=1 in its environment."
)


class PrefixCacheResetError(RuntimeError):
    """Raised when a requested cache reset could not be performed."""


def server_base_url(url: str) -> str:
    """Strip the request path off an endpoint URL, keeping scheme://host:port."""
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        raise PrefixCacheResetError(f"Cannot derive server base URL from {url!r}")
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


class HttpDriver(Driver):
    """OpenAI-compatible streaming client for one endpoint of a launched server.

    backend_name is serve.py's --backend; it picks the cache-reset endpoint.
    url, model, api_key and timeout (seconds, for every request and the
    reset) bind the driver to one server; send() and run() use them over
    the aiohttp session opened by session().
    """

    version = None  # the server is not asked for one

    def __init__(self, endpoint, backend_name="vllm", *, url=None, model=None,
                 api_key="test", timeout=60):
        self.endpoint, self.backend_name = endpoint, backend_name
        self._stream, self.ttft_rule = ENDPOINTS[endpoint]
        self.url, self.model, self.api_key, self.timeout = url, model, api_key, timeout
        self._http = None

    async def send_request(self, *args, **kwargs) -> RequestResult:
        result = await self._stream(*args, **kwargs)
        result.ttft_rule = self.ttft_rule
        return result

    @asynccontextmanager
    async def session(self, concurrency, open_loop=False):
        """One aiohttp session for a run. Open loop must not cap connections:
        a connector limit would silently become the concurrency cap."""
        connector = aiohttp.TCPConnector(limit=0 if open_loop else concurrency + 10)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as http:
            self._http = http
            try:
                yield http
            finally:
                self._http = None

    async def send(self, request, *, ignore_eos=False, temperature=0.0, seed=None,
                   min_tokens=0, request_id=None, capture_text=False) -> RequestResult:
        """Stream one Request: token ids when it has them, else its messages.
        capture_text is passed only when set; the completions body has none."""
        extra = {}
        if request.prompt_token_ids:
            extra["prompt_token_ids"] = request.prompt_token_ids
        if capture_text:
            extra["capture_text"] = True
        return await self.send_request(
            session=self._http, url=self.url, model=self.model, messages=request.messages,
            max_tokens=request.max_tokens, api_key=self.api_key, ignore_eos=ignore_eos,
            temperature=temperature, seed=seed, min_tokens=min_tokens, request_id=request_id,
            **extra)

    def run(self, requests, *, concurrency, **sampling):
        """Closed loop at `concurrency` over requests; yields results in request order."""
        async def all_of_them():
            slots = asyncio.Semaphore(concurrency)

            async def one(request):
                async with slots:
                    return await self.send(request, **sampling)

            async with self.session(concurrency):
                return await asyncio.gather(*(one(r) for r in requests))

        yield from asyncio.run(all_of_them())

    async def reset(self, session=None, url=None, api_key=None, timeout=None) -> str:
        """
        Reset the server's prefix cache. Returns a short status string for logging.
        Arguments left out come from the bound server and its open session.

        Raises PrefixCacheResetError if the backend has no known reset endpoint or
        the server rejects the call -- silently continuing would produce exactly the
        cross-cell contamination this exists to prevent.
        """
        session = self._http if session is None else session
        url = self.url if url is None else url
        api_key = self.api_key if api_key is None else api_key
        timeout = self.timeout if timeout is None else timeout
        backend_name = self.backend_name
        endpoint = _RESET_ENDPOINTS.get(backend_name.lower())
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
                    hint = f" {_DEV_MODE_HINT}" if resp.status == 404 else ""
                    raise PrefixCacheResetError(
                        f"POST {reset_url} returned HTTP {resp.status}: {body}.{hint}"
                    )
        except PrefixCacheResetError:
            raise
        except Exception as e:  # network/timeout
            raise PrefixCacheResetError(f"POST {reset_url} failed: {e}") from e

        return f"prefix cache reset via {reset_url}"


class BenchFailed(RuntimeError):
    """A bench cell exited non-zero or printed no result line."""


class BenchDriver(Driver):
    """bench.sh's cell loop: truncate the log, then per cell write the `>>>`
    header, run bench_argv under the cell's env in its cwd, tee stdout and
    stderr into the log, and check the exit code and the pass pattern.
    Rows come from table.ours_points on the finished log, so they are what
    the table reads."""

    def __init__(self, log_path):
        self.log_path = Path(log_path)

    def run(self, cell_specs, *, concurrency=1, **kwargs):
        """cell_specs: (ctx, bs, translate(cfg, (ctx, bs))) per cell; yields rows."""
        self.log_path.write_text("")
        missing = []
        for ctx, bs, spec in cell_specs:
            if not self._run_cell(ctx, bs, spec):
                self._log(f"FAIL ctx={ctx} bs={bs} (no bench result line; see {self.log_path})")
                missing.append((ctx, bs))
        if missing:
            raise BenchFailed(f"{len(missing)} cell(s) failed: {missing}")
        yield from self.rows()

    def rows(self):
        """One dict per (ctx, bs, tp) table.py found in the log."""
        points, method = table.ours_points(self.log_path)
        for (ctx, bs, tp), (ms, group, kv) in points.items():
            yield {"ctx": ctx, "bs": bs, "tp": tp, "ms_per_step": ms, "group": group,
                   "kv": kv, "method": method}

    def _log(self, line):
        print(line, file=sys.stderr, flush=True)
        with self.log_path.open("a") as fh:
            fh.write(line + "\n")

    def _run_cell(self, ctx, bs, spec):
        """True when the cell's output has a line matching spec.ok_pattern."""
        if spec.bench_argv[0] == "cargo" and not (Path(spec.cwd) / spec.bench_argv[3]
                                                  / "Cargo.toml").exists():
            raise BenchFailed(f"no {spec.bench_argv[3]} crate under {spec.cwd!r}; "
                              "set engine.extra.qs_dir or engine.extra.bench_bin")
        self._log(spec.log_header)
        ok = re.compile(spec.ok_pattern)
        passed = False
        proc = subprocess.Popen(spec.child_argv(spec.bench_argv), cwd=spec.cwd, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with self.log_path.open("a") as fh:
            for line in proc.stdout:
                sys.stdout.write(line)
                fh.write(line)
                passed = passed or bool(ok.search(line))
        rc = proc.wait()
        if rc != 0:
            self._log(f"FAIL ctx={ctx} bs={bs} (bench exited {rc}; see {self.log_path})")
            raise BenchFailed(f"bench exited {rc}")
        return passed


class InProcessDriver(Driver):
    """vLLM in this process, built from spec.engine_kwargs on first use.

    Grid runs only, and only until the HTTP-vs-in-process parity gate
    (QBENCH_STRUCTURE.md section 5) passes; a pass deletes this class.
    vllm is imported inside the builders because it exists only on GPU boxes.
    """

    def __init__(self, spec):
        self.spec = spec

    def llm(self):
        """vmin_fit's engine: vllm.LLM(**engine_kwargs)."""
        from vllm import LLM
        return LLM(**self.spec.engine_kwargs)

    def async_engine(self):
        """trace_serve's engine: AsyncLLMEngine from AsyncEngineArgs(**engine_kwargs)."""
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        return AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**self.spec.engine_kwargs))

    def run(self, requests, *, concurrency, **kwargs):
        raise NotImplementedError("vmin_fit and trace_serve drive the engine themselves for now")
