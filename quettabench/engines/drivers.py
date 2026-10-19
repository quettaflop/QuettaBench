"""Drivers that deliver a Trace's requests to an engine."""

import asyncio
import contextlib
import functools
import http.client
import json
import os
import re
import signal
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import aiohttp

from quettabench import compare
from quettabench.config import env_map
from quettabench.config.schema import OTHER_ACCELERATORS, ConfigError
from quettabench.engines import launcher
from quettabench.engines.base import EngineAdapter, LaunchSpec, register
from quettabench.metrics.summary import RequestResult, parse_prometheus, spec_decode_counts
from quettabench.workloads.base import Request
from quettabench.workloads.grid import distinct_ids


class Driver(ABC):
    @abstractmethod
    def run(self, *args, **kwargs):
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


def _payload(model, prompt, max_tokens, temperature, ignore_eos, request_id, seed, min_tokens, tools=None):
    payload = {
        "model": model,
        **prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        payload["tools"] = tools
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
    extra_headers: dict | None = None,
    ignore_eos: bool = False,
    request_id: str | None = None,
    seed: int | None = None,
    min_tokens: int = 0,
    capture_text: bool = False,
    tools: list | None = None,
) -> RequestResult:
    """Each chunk with content, reasoning or tool_calls counts as a token."""
    headers = _headers(api_key, extra_headers, request_id)
    payload = _payload(
        model, {"messages": messages}, max_tokens, temperature, ignore_eos, request_id, seed, min_tokens, tools
    )

    start_time = time.perf_counter()
    ttft = None
    itl: list[float] = []
    last_token_time = start_time
    input_tokens = 0
    output_tokens = 0
    usage_reported = False
    done = False
    server_cached_tokens = None
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

                data_str = line[len("data:") :].strip()

                if data_str == "[DONE]":
                    done = True
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if chunk.get("error"):
                    return RequestResult(
                        success=False,
                        ttft=ttft,
                        itl=itl,
                        e2el=time.perf_counter() - start_time,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        usage_reported=usage_reported,
                        error=f"server error in stream: {json.dumps(chunk['error'])[:200]}",
                        error_kind="server_error",
                    )

                now = time.perf_counter()

                if chunk.get("usage"):
                    usage = chunk["usage"]
                    input_tokens = usage.get("prompt_tokens", 0)
                    output_tokens = usage.get("completion_tokens", 0)
                    server_cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                    usage_reported = True

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                # vllm bench serve's rule: every chunk with choices is a token event, the finish chunk included,
                # so decode the server buffered (a tool call its parser emits whole) is in the request's time.
                if capture_text:
                    piece = choices[0].get("delta", {}).get("content")
                    if piece:
                        text_parts.append(piece)

                if ttft is None:
                    ttft = now - start_time
                    last_token_time = now
                else:
                    itl.append(now - last_token_time)
                    last_token_time = now

        # The request ends at its last chunk with choices; the usage chunk, [DONE] and the close come after.
        stream_closed = time.perf_counter() - start_time
        e2el = last_token_time - start_time if ttft is not None else stream_closed
        if not done:
            return RequestResult(
                success=False,
                ttft=ttft,
                itl=itl,
                e2el=e2el,
                stream_closed_s=stream_closed,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                usage_reported=usage_reported,
                error="stream ended without [DONE]; the reply is incomplete",
                error_kind="incomplete_stream",
            )
        return RequestResult(
            success=True,
            ttft=ttft,
            itl=itl,
            e2el=e2el,
            stream_closed_s=stream_closed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usage_reported=usage_reported,
            server_cached_tokens=server_cached_tokens,
            generated_text="".join(text_parts) if capture_text else None,
        )

    except asyncio.CancelledError:
        raise
    except Exception as e:
        # A timeout truncates the slowest requests; partial timings are the only evidence.
        is_timeout = isinstance(e, asyncio.TimeoutError)
        return RequestResult(
            success=False,
            ttft=ttft,
            itl=itl,
            e2el=time.perf_counter() - start_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usage_reported=usage_reported,
            server_cached_tokens=server_cached_tokens,
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
    extra_headers: dict | None = None,
    ignore_eos: bool = False,
    request_id: str | None = None,
    seed: int | None = None,
    min_tokens: int = 0,
    prompt_token_ids: list[int] | None = None,
    tools: list | None = None,
) -> RequestResult:
    if tools:
        return RequestResult(
            success=False, e2el=0.0, error="completions backend cannot send a tool schema", error_kind="client_error"
        )
    prompt = resolve_prompt(prompt_token_ids, messages)
    if prompt is None:
        return RequestResult(
            success=False,
            e2el=0.0,
            error="completions backend needs prompt_token_ids or a user message",
            error_kind="client_error",
        )

    headers = _headers(api_key, extra_headers, request_id)
    payload = _payload(model, {"prompt": prompt}, max_tokens, temperature, ignore_eos, request_id, seed, min_tokens)

    start_time = time.perf_counter()
    ttft = None
    itl: list[float] = []
    last_token_time = start_time
    input_tokens = 0
    output_tokens = 0
    usage_reported = False
    done = False
    last_choices_time = start_time
    server_cached_tokens = None

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

                data_str = line[len("data:") :].strip()

                if data_str == "[DONE]":
                    done = True
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if chunk.get("error"):
                    return RequestResult(
                        success=False,
                        ttft=ttft,
                        itl=itl,
                        e2el=time.perf_counter() - start_time,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        usage_reported=usage_reported,
                        error=f"server error in stream: {json.dumps(chunk['error'])[:200]}",
                        error_kind="server_error",
                    )

                now = time.perf_counter()

                if chunk.get("usage"):
                    usage = chunk["usage"]
                    input_tokens = usage.get("prompt_tokens", 0)
                    output_tokens = usage.get("completion_tokens", 0)
                    server_cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                    usage_reported = True

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                last_choices_time = now
                # A chunk with empty text is not a token event (QuettaServe sends one before its first token).
                if choices[0].get("text"):
                    if ttft is None:
                        ttft = now - start_time
                        last_token_time = now
                    else:
                        itl.append(now - last_token_time)
                        last_token_time = now

        # The request ends at its last chunk with choices, as vllm bench serve counts it; the usage chunk, [DONE]
        # and the close come after.
        stream_closed = time.perf_counter() - start_time
        e2el = last_choices_time - start_time if ttft is not None else stream_closed
        if not done:
            return RequestResult(
                success=False,
                ttft=ttft,
                itl=itl,
                e2el=e2el,
                stream_closed_s=stream_closed,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                usage_reported=usage_reported,
                error="stream ended without [DONE]; the reply is incomplete",
                error_kind="incomplete_stream",
            )
        return RequestResult(
            success=True,
            ttft=ttft,
            itl=itl,
            e2el=e2el,
            stream_closed_s=stream_closed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usage_reported=usage_reported,
            server_cached_tokens=server_cached_tokens,
        )

    except asyncio.CancelledError:
        raise
    except Exception as e:
        is_timeout = isinstance(e, asyncio.TimeoutError)
        return RequestResult(
            success=False,
            ttft=ttft,
            itl=itl,
            e2el=time.perf_counter() - start_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usage_reported=usage_reported,
            server_cached_tokens=server_cached_tokens,
            error=f"timeout after {time.perf_counter() - start_time:.1f}s" if is_timeout else str(e),
            error_kind="timeout" if is_timeout else "client_error",
        )


# endpoint -> (stream body, first-token rule)
ENDPOINTS: dict[str, tuple[Callable[..., Awaitable[RequestResult]], str]] = {
    "chat": (_stream_chat, "first_choices_chunk"),
    "completions": (_stream_completions, "first_nonempty_text"),
}

# vLLM mounts the reset router only with VLLM_SERVER_DEV_MODE=1; otherwise 404.
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
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        raise ValueError(f"cannot derive a server base URL from {url!r}")
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


class GridBatchFailed(RuntimeError):
    """A grid batch over HTTP lost a request or decoded the wrong length."""


class HttpDriver(Driver):
    def __init__(
        self, endpoint, backend_name="vllm", *, url=None, model=None, api_key="test", timeout=60, accelerator="cuda"
    ):
        self.endpoint, self.backend_name, self.accelerator = endpoint, backend_name, accelerator
        self._stream, self.ttft_rule = ENDPOINTS[endpoint]  # which chunk counts as the first token, recorded per result
        self.url, self.model, self.api_key, self.timeout = url, model, api_key, timeout
        self._http = None
        self._cell = None  # Last grid batch; a new one resets the cache.

    def device(self):
        return launcher.accelerator(self.accelerator)

    @functools.cached_property
    def version(self):
        """Asked once; None without a bound url or an answer."""
        if not self.url:
            return None
        try:
            with launcher.urlopen(f"{server_base_url(self.url)}/version", timeout=10) as resp:
                return json.loads(resp.read())["version"]
        except (OSError, http.client.HTTPException, ValueError, KeyError, TypeError):
            return None

    async def send_request(self, *args, **kwargs) -> RequestResult:
        result = await self._stream(*args, **kwargs)
        result.ttft_rule = self.ttft_rule
        return result

    @asynccontextmanager
    async def session(self, concurrency, open_loop=False):
        """Open loop gets no connection limit, which would silently cap its concurrency."""
        connector = aiohttp.TCPConnector(limit=0 if open_loop else concurrency + 10)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as http:
            self._http = http
            try:
                yield http
            finally:
                self._http = None

    async def send(
        self,
        request,
        *,
        ignore_eos=False,
        temperature=0.0,
        seed=None,
        min_tokens=0,
        request_id=None,
        capture_text=False,
    ) -> RequestResult:
        extra = {}
        if request.prompt_token_ids:
            extra["prompt_token_ids"] = request.prompt_token_ids
        if capture_text:
            extra["capture_text"] = True
        if request.tools:
            extra["tools"] = request.tools
        return await self.send_request(
            session=self._http,
            url=self.url,
            model=self.model,
            messages=request.messages,
            max_tokens=request.max_tokens,
            api_key=self.api_key,
            ignore_eos=ignore_eos,
            temperature=temperature,
            seed=seed,
            min_tokens=min_tokens,
            request_id=request_id,
            **extra,
        )

    def run(self, requests, *, concurrency=None, cell=None, **sampling):
        """With a cell, one grid batch."""
        if cell is not None:
            yield from self.grid_batch(requests, cell)
            return

        async def all_of_them():
            slots = asyncio.Semaphore(concurrency)

            async def one(request):
                async with slots:
                    return await self.send(request, **sampling)

            async with self.session(concurrency):
                return await asyncio.gather(*(one(r) for r in requests))

        yield from asyncio.run(all_of_them())

    def timed_batch(self, requests, cell_key=None):
        """Greedy, ignoring EOS; a new cell_key resets the prefix cache first."""

        async def go():
            async with self.session(len(requests)):
                if cell_key is not None and cell_key != self._cell:
                    await self.reset()
                    self._cell = cell_key
                t0 = time.perf_counter()
                results = await asyncio.gather(*(self.send(r, ignore_eos=True) for r in requests))
                return time.perf_counter() - t0, results

        return asyncio.run(go())

    def spec_decode_counters(self):
        if not self.url:
            return None
        try:
            with launcher.urlopen(f"{server_base_url(self.url)}/metrics", timeout=10) as resp:
                return spec_decode_counts(parse_prometheus(resp.read().decode()))
        except (OSError, http.client.HTTPException, ValueError):
            return None

    def warmup(self):
        """The time is discarded."""
        self.timed_batch([Request(distinct_ids(256), 256, output_length=20)])

    def grid_batch(self, requests, cell):
        """Every row carries the batch wall as e2el."""
        n = cell["step_point"]
        wall, results = self.timed_batch(requests, (cell["ctx"], cell["bs"]))
        for res in results:
            if not res.success or not res.usage_reported or res.output_tokens != n:
                why = (
                    res.error
                    if not res.success
                    else "no usage reported"
                    if not res.usage_reported
                    else f"{res.output_tokens} tokens decoded"
                )
                raise GridBatchFailed(f"ctx={cell['ctx']} bs={cell['bs']} step={n}: {why}")
        for r in requests:
            yield RequestResult(
                success=True,
                e2el=wall,
                batch_wall_s=wall,
                input_tokens=r.prompt_len,
                output_tokens=n,
                ttft_basis="batch",
                ttft_rule=self.ttft_rule,
                cell=dict(cell),
            )

    async def reset(self, session=None, url=None, api_key=None, timeout=None) -> str:
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
                    raise PrefixCacheResetError(f"POST {reset_url} returned HTTP {resp.status}: {body}.{hint}")
        except PrefixCacheResetError:
            raise
        except aiohttp.ClientConnectionError as e:
            raise ConnectionError(f"POST {reset_url}: {e}") from e
        except Exception as e:
            raise PrefixCacheResetError(f"POST {reset_url} failed: {e}") from e

        return f"prefix cache reset via {reset_url}"


class BenchFailed(RuntimeError):
    """A bench cell exited non-zero or printed no result line."""


class BenchDriver(Driver):
    measures_steps = True  # rows are ms per step already

    def __init__(self, log_path, cells=()):
        self.log_path = Path(log_path)
        self.cells = cells

    def device(self):
        return launcher.accelerator("cuda")

    def run(self, cell_specs, *, concurrency=1, step_points=(), **kwargs):
        """step_points go unused: the bench sets its own timing_steps."""
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
        """Already ms per step, so step_fit passes it through."""
        points, method, notes = compare.bench_log_points(self.log_path)
        for note in notes:
            print(note)
        for (ctx, bs, tp), (ms, group, kv) in points.items():
            yield {
                "cell": {"ctx": ctx, "bs": bs, "tp": tp, "step_point": None},
                "measured_ms_per_step": ms,
                "group": group,
                "kv": kv,
                "method": method,
            }

    def _log(self, line):
        print(line, file=sys.stderr, flush=True)
        with self.log_path.open("a") as fh:
            fh.write(line + "\n")

    def _run_cell(self, ctx, bs, spec):
        if spec.bench_argv[0] == "cargo" and not (Path(spec.cwd) / spec.bench_argv[3] / "Cargo.toml").exists():
            raise BenchFailed(
                f"no {spec.bench_argv[3]} crate under {spec.cwd!r}; set engine.extra.quettaserve_dir or engine.extra.bench_bin"
            )
        self._log(spec.log_header)
        ok = re.compile(spec.ok_pattern)
        passed = False
        proc = subprocess.Popen(
            spec.child_argv(spec.bench_argv),
            cwd=spec.cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        assert proc.stdout is not None
        try:
            with self.log_path.open("a") as fh:
                for line in proc.stdout:
                    sys.stdout.write(line)
                    fh.write(line)
                    passed = passed or bool(ok.search(line))
            rc = proc.wait()
        finally:
            # Kill the whole group so no child (cargo's) keeps the GPUs.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        if rc != 0:
            self._log(f"FAIL ctx={ctx} bs={bs} (bench exited {rc}; see {self.log_path})")
            raise BenchFailed(f"bench exited {rc}")
        return passed


class InProcessDriver(Driver):
    """Only for mode.return_outputs: only this path returns token ids, text or logits."""

    def __init__(self, spec):
        self.spec = spec
        self._llm = None

    @property
    def version(self):
        import vllm

        return vllm.__version__

    def device(self):
        import torch

        return torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory

    def llm(self):
        """Built after spec.env is applied, so KV is sized under the env a server gets."""
        if self._llm is None:
            env_map.export(self.spec.env)
            from vllm import LLM

            self._llm = LLM(**self.spec.engine_kwargs)
        return self._llm

    def tokenizer(self):
        return self.llm().get_tokenizer()

    def generate(self, prompts, max_tokens):
        """Greedy, ignoring EOS, so every prompt decodes exactly max_tokens."""
        from vllm import SamplingParams

        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
        t0 = time.perf_counter()
        outs = self.llm().generate(prompts, sp, use_tqdm=False)
        return time.perf_counter() - t0, outs

    def warmup(self):
        """The time is discarded."""
        self.generate([{"prompt_token_ids": distinct_ids(256)}], 20)

    def score(self, prompts):
        from vllm import SamplingParams

        params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=2)
        return self.llm().generate([{"prompt_token_ids": ids} for ids in prompts], params, use_tqdm=False)

    def run(self, requests, *, cell, return_outputs="none", concurrency=None, **kwargs):
        """return_outputs="logits" scores instead and times nothing."""
        if return_outputs == "logits":
            outs = self.score([r.prompt_token_ids for r in requests])
            for r, out in zip(requests, outs):
                yield RequestResult(
                    success=True,
                    input_tokens=r.prompt_len,
                    cell=dict(cell),
                    logits=prompt_logprob_summary(r.prompt_token_ids, out.prompt_logprobs),
                )
            return
        n = cell["step_point"]
        wall, outs = self.generate([{"prompt_token_ids": r.prompt_token_ids} for r in requests], n)
        for r, out in zip(requests, outs):
            completion = out.outputs[0]
            yield RequestResult(
                success=True,
                e2el=wall,
                batch_wall_s=wall,
                input_tokens=r.prompt_len,
                output_tokens=n,
                ttft_basis="batch",
                cell=dict(cell),
                generated_token_ids=list(completion.token_ids),
                generated_text=getattr(completion, "text", None),
            )


def truncated_config(model, layers):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    if layers > config.num_hidden_layers:
        raise ValueError(
            f"reference: engine.layers={layers} exceeds the checkpoint's {config.num_hidden_layers} layers"
        )
    config.num_hidden_layers = layers
    if isinstance(getattr(config, "layer_types", None), list):
        config.layer_types = config.layer_types[:layers]
    return config


def prompt_logprob_summary(ids, prompt_logprobs):
    """Entry i scores ids[i + 1]."""
    out: dict[str, list] = {"argmax": [], "top2_gap": [], "logprob": []}
    for pos in range(1, len(ids)):
        entry = prompt_logprobs[pos]
        by_rank = {lp.rank: (token, lp.logprob) for token, lp in entry.items()}
        out["argmax"].append(by_rank[1][0])
        out["top2_gap"].append(by_rank[1][1] - by_rank[2][1])
        out["logprob"].append(entry[ids[pos]].logprob)
    return out


class ReferenceDriver(Driver):
    """One teacher-forced pass per prompt on cuda:0 in engine_kwargs["dtype"]."""

    scores_only = True

    def __init__(self, spec):
        self.spec = spec
        self._model = None

    @property
    def version(self):
        import transformers

        return transformers.__version__

    def tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.spec.engine_kwargs["model"], trust_remote_code=True)

    def model(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM

            kw = self.spec.engine_kwargs
            load = {"torch_dtype": getattr(torch, kw["dtype"]), "trust_remote_code": True}
            if kw.get("layers"):
                load["config"] = truncated_config(kw["model"], kw["layers"])
            self._model = AutoModelForCausalLM.from_pretrained(kw["model"], **load).to("cuda:0")
            self._model.eval()
            if kw.get("layers") and len(self._model.model.layers) != kw["layers"]:
                raise RuntimeError(
                    f"reference: built {len(self._model.model.layers)} decoder "
                    f"layers, engine.layers asked for {kw['layers']}"
                )
        return self._model

    def score(self, ids):
        import torch

        with torch.no_grad():
            t = torch.tensor([ids], device="cuda:0")
            logits = self.model()(t).logits[0].float()
            logprobs = torch.log_softmax(logits, dim=-1)
            top2 = logits.topk(2, dim=-1)
            preds = top2.indices[:, 0].tolist()
            gaps = (top2.values[:, 0] - top2.values[:, 1]).tolist()
            true_lp = [float(logprobs[i, ids[i + 1]]) for i in range(len(ids) - 1)]
        return {"argmax": preds[:-1], "top2_gap": gaps[:-1], "logprob": true_lp}

    def run(self, requests, *, cell, **kwargs):
        for r in requests:
            yield RequestResult(
                success=True, input_tokens=r.prompt_len, cell=dict(cell), logits=self.score(r.prompt_token_ids)
            )


class ReferenceAdapter(EngineAdapter):
    """No server; the driver loads engine_kwargs {model, dtype, layers}."""

    name = "reference"
    supports = frozenset({"model", "dtype", "layers"})
    benchmarked = False

    def translate(self, cfg, run_dir=None):
        self.refuse_unsupported(cfg)
        engine = cfg.engine
        if engine.model is None:
            raise ConfigError("reference: engine.model is required")
        if engine.dtype == "auto":
            raise ConfigError("reference: engine.dtype must name a torch dtype, e.g. bfloat16")
        if cfg.mode.grid is None:
            raise ConfigError("reference: scores grid prompts only; set mode.grid")
        if cfg.hardware.backend in OTHER_ACCELERATORS:
            raise ConfigError(
                f"reference: runs transformers on cuda:0; hardware.backend={cfg.hardware.backend} has none"
            )
        if engine.layers is not None and engine.layers < 1:
            raise ConfigError(f"reference: engine.layers must be >= 1, got {engine.layers}")
        kwargs = {"model": engine.model, "dtype": engine.dtype}
        if engine.layers is not None:
            kwargs["layers"] = engine.layers
        return LaunchSpec(adapter=self.name, engine_kwargs=kwargs)

    def needs_server(self, cfg):
        return False

    def driver(self, spec, handle, trace=None):
        return ReferenceDriver(spec)


REFERENCE = register(ReferenceAdapter())
