"""Fake OpenAI-compatible streaming server with a fixed timing model; confirm runs the client against it before a GPU
hour is spent. Its timing is fixed by flags, so a client gate that fails here failed on the client.
"""

import argparse
import asyncio
import hashlib
import json
import math
import multiprocessing
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from aiohttp import web


def compact_json(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def digest(obj):
    """Short fingerprint of a message, tool list or prompt, so two requests' contents can be compared from the log."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()[:16]


@dataclass
class Plan:
    """When and what one reply emits: TTFT, the tokens each token chunk carries, and the gaps between them."""

    ttft: float
    chunks: list[int]
    gaps: list[float]

    @property
    def tokens(self):
        return sum(self.chunks)


@dataclass
class Request:
    """What the server read from one POST."""

    index: int
    path: str
    body: dict
    t_entry: float
    t_body: float
    chat: bool
    prompt_chars: int
    prompt_tokens: int
    max_tokens: int
    turn: int
    record: dict = field(default_factory=dict)


def prompt_size(body, chat):
    """Prompt size that drives the TTFT model."""
    if chat:
        chars = len(compact_json(body.get("messages", [])))
        if body.get("tools"):
            chars += len(compact_json(body["tools"]))
        return chars, max(1, math.ceil(chars / 4))
    prompt = body.get("prompt", "")
    if isinstance(prompt, list):
        return len(prompt), max(1, len(prompt))
    return len(prompt), max(1, math.ceil(len(prompt) / 4))


class FakeServer:
    """The handlers plus per-request timing and the chunk log."""

    def __init__(self, args):
        self.args = args
        self.model = args.model
        self.created = int(time.time())
        self.count = 0
        self.records = []
        self.counters = {"prompt_tokens": 0, "generation_tokens": 0, "success": 0, "running": 0}
        self.log_path = Path(args.log) if args.log else None
        if self.log_path:
            self.log_path.write_text("")

    def timing_plan_for(self, req):
        a = self.args
        ttft = a.ttft_a + a.ttft_b * req.prompt_chars
        return Plan(ttft, [1] * req.max_tokens, [a.tpot] * (req.max_tokens - 1))

    async def health(self, request):
        return web.Response(text="")

    async def version(self, request):
        return web.json_response({"version": "fake-0"})

    async def models(self, request):
        card = {"id": self.model, "object": "model", "created": self.created, "owned_by": "fake"}
        card["max_model_len"] = self.args.max_model_len
        return web.json_response({"object": "list", "data": [card]})

    async def reset(self, request):
        self.records.append({"path": request.path, "t_entry": time.perf_counter()})
        return web.Response(text="")

    async def metrics(self, request):
        c = self.counters
        lines = [
            f'vllm:prompt_tokens_total{{model_name="{self.model}"}} {float(c["prompt_tokens"])}',
            f'vllm:generation_tokens_total{{model_name="{self.model}"}} {float(c["generation_tokens"])}',
            f'vllm:request_success_total{{model_name="{self.model}"}} {float(c["success"])}',
            f'vllm:num_requests_running{{model_name="{self.model}"}} {float(c["running"])}',
            f'vllm:num_requests_waiting{{model_name="{self.model}"}} 0.0',
        ]
        return web.Response(text="\n".join(lines) + "\n")

    async def complete(self, request):
        t_entry = time.perf_counter()
        raw = await request.read()
        t_body = time.perf_counter()
        body = json.loads(raw)
        chat = request.path.endswith("/chat/completions")
        chars, tokens = prompt_size(body, chat)
        turn = sum(1 for m in body.get("messages", []) if m.get("role") == "assistant") if chat else 0
        max_tokens = body.get("max_completion_tokens") or body.get("max_tokens") or self.args.default_max_tokens
        req = Request(self.count, request.path, body, t_entry, t_body, chat, chars, tokens, int(max_tokens), turn)
        self.count += 1
        plan = self.timing_plan_for(req)
        req.record = self.start_record(req, request, plan)
        self.counters["running"] += 1
        try:
            if not body.get("stream"):
                return await self.reply_unstreamed(req, plan)
            return await self.stream(req, request, plan)
        finally:
            self.counters["running"] -= 1
            req.record["t_end"] = time.perf_counter()
            self.records.append(req.record)

    def start_record(self, req, request, plan):
        messages = req.body.get("messages", [])
        rec = {
            "i": req.index,
            "rid": f"fake-{req.index}" if not self.args.worker_id else f"fake-w{self.args.worker_id}-{req.index}",
            "path": req.path,
            "xrid": request.headers.get("X-Request-Id"),
            "t_entry": req.t_entry,
            "t_body": req.t_body,
            "turn": req.turn,
            "n_msgs": len(messages),
            "keys": sorted(k for k in req.body if k not in ("messages", "prompt", "tools")),
            "params": {k: v for k, v in req.body.items() if k not in ("messages", "prompt", "tools")},
            "prompt_chars": req.prompt_chars,
            "prompt_tokens": req.prompt_tokens,
            "max_tokens": req.max_tokens,
            "plan_ttft": plan.ttft,
            "plan_tokens": plan.tokens,
            "status": 200,
            "emits": [],
            "kinds": [],
            "disconnected": False,
        }
        if self.args.no_hash:
            return rec
        rec["msg_sha"] = [digest(m) for m in messages]
        rec["tools_sha"] = digest(req.body["tools"]) if req.body.get("tools") else None
        if not req.chat:
            rec["prompt_sha"] = digest(req.body.get("prompt"))
        return rec

    async def reply_unstreamed(self, req, plan):
        await sleep_until(req.t_entry + plan.ttft + sum(plan.gaps))
        text = "tok " * plan.tokens
        rid = req.record["rid"]
        if req.chat:
            choice = {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "length"}
        else:
            choice = {"index": 0, "text": text, "finish_reason": "length"}
        usage = {"prompt_tokens": req.prompt_tokens, "completion_tokens": plan.tokens}
        usage["total_tokens"] = req.prompt_tokens + plan.tokens
        self.count_tokens(req, plan.tokens)
        kind = "chat.completion" if req.chat else "text_completion"
        return web.json_response({"id": rid, "object": kind, "model": self.model, "choices": [choice], "usage": usage})

    def count_tokens(self, req, generated):
        self.counters["prompt_tokens"] += req.prompt_tokens
        self.counters["generation_tokens"] += generated
        self.counters["success"] += 1

    async def stream(self, req, request, plan):
        resp = web.StreamResponse(
            status=200, headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
        )
        await resp.prepare(request)
        rec = req.record
        rid = rec["rid"]
        sent_tokens = 0
        rec["chunk_tokens"], rec["chunk_chars"] = [], []
        options = req.body.get("stream_options") or {}
        continuous = bool(options.get("continuous_usage_stats"))

        async def emit(kind, payload, tokens=0, chars=0):
            await resp.write(f"data: {payload}\n\n".encode())
            rec["emits"].append(time.perf_counter())
            rec["kinds"].append(kind)
            rec["chunk_tokens"].append(tokens)
            rec["chunk_chars"].append(chars)

        def with_usage(chunk, done):
            """vLLM's continuous_usage_stats: every chunk carries the cumulative completion_tokens."""
            if continuous:
                usage = {"prompt_tokens": req.prompt_tokens, "completion_tokens": done}
                chunk["usage"] = usage | {"total_tokens": req.prompt_tokens + done}
            return compact_json(chunk)

        try:
            target = req.t_entry + plan.ttft
            await sleep_until(target)
            if req.chat:
                role = self.chunk(rid, req, {"role": "assistant", "content": ""})
                await emit("role", with_usage(role, 0))
            for i, k in enumerate(plan.chunks):
                if i > 0:
                    target += plan.gaps[i - 1]
                    await sleep_until(target)
                last = i == len(plan.chunks) - 1
                sent_tokens += k
                delta = {"content": "tok " * k}
                body = with_usage(self.chunk(rid, req, delta, "length" if last else None), sent_tokens)
                await emit("finish" if last else "token", body, k, 4 * k)
            usage = {"prompt_tokens": req.prompt_tokens, "completion_tokens": sent_tokens}
            usage["total_tokens"] = req.prompt_tokens + sent_tokens
            usage["prompt_tokens_details"] = {"cached_tokens": 0}
            head = {"id": rid, "object": self.object_name(req), "created": self.created, "model": self.model}
            await emit("usage", compact_json(head | {"choices": [], "usage": usage}))
            await emit("done", "[DONE]")
            await resp.write_eof()
            self.count_tokens(req, sent_tokens)
        except (ConnectionResetError, ConnectionError, asyncio.CancelledError):
            rec["disconnected"] = True
        return resp

    def object_name(self, req):
        return "chat.completion.chunk" if req.chat else "text_completion"

    def chunk(self, rid, req, delta, finish=None):
        if req.chat:
            choice = {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}
        else:
            choice = {"index": 0, "text": delta.get("content", ""), "logprobs": None, "finish_reason": finish}
        return {
            "id": rid,
            "object": self.object_name(req),
            "created": self.created,
            "model": self.model,
            "choices": [choice],
        }

    async def flusher(self, app):
        async def loop():
            while True:
                await asyncio.sleep(1.0)
                self.flush()

        task = asyncio.create_task(loop())
        yield
        task.cancel()
        self.flush()

    def flush(self):
        if not self.log_path or not self.records:
            return
        records, self.records = self.records, []
        with open(self.log_path, "a") as f:
            for rec in records:
                f.write(compact_json(rec) + "\n")


async def sleep_until(target):
    delay = target - time.perf_counter()
    if delay > 0:
        await asyncio.sleep(delay)


def build_app(args):
    server = FakeServer(args)
    app = web.Application(client_max_size=1 << 30)
    app.router.add_get("/health", server.health)
    app.router.add_get("/version", server.version)
    app.router.add_get("/v1/models", server.models)
    app.router.add_get("/metrics", server.metrics)
    app.router.add_route("*", "/reset_prefix_cache", server.reset)
    app.router.add_post("/v1/chat/completions", server.complete)
    app.router.add_post("/v1/completions", server.complete)
    app.cleanup_ctx.append(server.flusher)
    return app


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18999)
    p.add_argument("--model", default="Qwen/Qwen3.8-Flash-Next", help="the one name /v1/models serves")
    p.add_argument("--max-model-len", type=int, default=1_000_000)
    p.add_argument("--ttft-a", type=float, default=0.05, help="TTFT intercept, seconds")
    p.add_argument("--ttft-b", type=float, default=0.0, help="TTFT seconds per prompt char")
    p.add_argument("--tpot", type=float, default=0.01, help="seconds between token chunks")
    p.add_argument("--default-max-tokens", type=int, default=16, help="output length when a request names none")
    p.add_argument("--log", help="emit log, one JSON line per request")
    p.add_argument(
        "--no-hash", action="store_true", help="log no message, tool or prompt digests (less work per request)"
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="processes sharing the port (SO_REUSEPORT); worker k > 0 logs to <log>.w<k>",
    )
    p.add_argument("--worker-id", type=int, default=0, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.workers < 1:
        p.error("--workers must be at least 1")
    return args


def serve(args):
    reuse = args.workers > 1
    app = build_app(args)
    web.run_app(app, host=args.host, port=args.port, access_log=None, print=None, backlog=4096, reuse_port=reuse)


def run_worker(argv, k):
    """One more process on the same port, logging to <log>.w<k>."""
    args = parse_args(argv)
    args.worker_id = k
    if args.log:
        args.log = f"{args.log}.w{k}"
    serve(args)


def main(argv=None):
    """--workers N: N - 1 more processes share the port; on SIGINT this one stops them so every log is flushed."""
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    ctx = multiprocessing.get_context("spawn")
    others = [ctx.Process(target=run_worker, args=(argv, k)) for k in range(1, args.workers)]
    for p in others:
        p.start()
    try:
        serve(args)
    finally:
        for p in others:
            if p.pid is not None:
                os.kill(p.pid, signal.SIGINT)
        for p in others:
            p.join(30)


if __name__ == "__main__":
    main()
