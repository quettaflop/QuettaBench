"""HttpDriver, BenchDriver, InProcessDriver, reference scorer and driver choice."""

import asyncio
import contextlib
import io
import json
import os
import sys
import tempfile
import time
import types
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest import mock

import aiohttp

from quettabench import cli, compare
from quettabench.config import loader
from quettabench.engines import drivers, launcher, quettaserve, vllm as vllm_adapter
from quettabench.engines.base import LaunchSpec
from quettabench.engines.drivers import (
    BenchDriver,
    BenchFailed,
    GridBatchFailed,
    HttpDriver,
    InProcessDriver,
    PrefixCacheResetError,
    ReferenceDriver,
)
from quettabench.metrics.results import RunWriter, jsonl_row
from quettabench.metrics.step_fit import fit_run
from quettabench.metrics.summary import RequestResult
from quettabench.modes import serve
from quettabench.workloads.base import Request
from tests.contracts import helpers
from tests.golden._harness import FIXTURES, REPO, compare_path, fresh_import, grid_argv, run_grid_cli


class _Resp:
    def __init__(self, lines, status=200):
        self.status, self._lines = status, lines

    @property
    def content(self):
        async def gen():
            for line in self._lines:
                yield line

        return gen()

    async def text(self):
        return "stub"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Session:
    def __init__(self, chunks, status=200, error=None):
        self.lines = [b"data: " + json.dumps(c).encode() + b"\n" for c in chunks]
        self.lines.append(b"data: [DONE]\n")
        self.status, self.posts, self.error = status, [], error

    def post(self, url, headers=None, json=None, timeout=None):
        self.posts.append((url, json))
        if self.error:
            raise self.error
        return _Resp(self.lines, self.status)


def send(endpoint, chunks, **kw):
    session = _Session(chunks)
    result = asyncio.run(
        HttpDriver(endpoint).send_request(
            session=session,
            url="http://stub/v1/x",
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=4,
            **kw,
        )
    )
    return result, session.posts[0][1]


class TtftRules(unittest.TestCase):
    def test_chat_fires_on_a_reasoning_chunk(self):
        r, payload = send(
            "chat", [{"choices": [{"delta": {"reasoning": "r"}}]}, {"choices": [{"delta": {"content": "a"}}]}]
        )
        self.assertEqual((r.ttft_rule, len(r.itl)), ("first_payload_chunk", 1))
        self.assertEqual(payload["messages"][0]["content"], "hi")

    def test_completions_skips_empty_text(self):
        r, payload = send(
            "completions",
            [{"choices": [{"text": ""}]}, {"choices": [{"text": "a"}]}, {"choices": [{"text": "b"}]}],
            prompt_token_ids=[1, 2],
        )
        self.assertEqual((r.ttft_rule, len(r.itl)), ("first_nonempty_text", 1))
        self.assertEqual(payload["prompt"], [1, 2])

    def test_completions_rejects_capture_text_as_before(self):
        with self.assertRaises(TypeError):
            send("completions", [], capture_text=True)

    def test_completions_without_a_prompt_is_a_client_error(self):
        r = asyncio.run(
            HttpDriver("completions").send_request(session=None, url="u", model="m", messages=[], max_tokens=1)
        )
        self.assertEqual((r.success, r.error_kind, r.ttft_rule), (False, "client_error", "first_nonempty_text"))


class Reset(unittest.TestCase):
    def test_reset_posts_to_the_dev_mode_endpoint(self):
        session = _Session([])
        status = asyncio.run(HttpDriver("chat", "vllm").reset(session, "http://h:8000/v1/completions"))
        self.assertEqual(status, "prefix cache reset via http://h:8000/reset_prefix_cache")

    def test_reset_has_no_sglang_endpoint(self):
        with self.assertRaisesRegex(PrefixCacheResetError, "backend 'sglang'"):
            asyncio.run(HttpDriver("chat", "sglang").reset(_Session([]), "http://h:1/v1/x"))

    def test_reset_404_names_dev_mode(self):
        with self.assertRaisesRegex(PrefixCacheResetError, "VLLM_SERVER_DEV_MODE=1"):
            asyncio.run(HttpDriver("chat", "vllm").reset(_Session([], status=404), "http://h:1/v1/x"))

    def test_a_bad_url_is_not_a_reset_error(self):
        with self.assertRaisesRegex(ValueError, "'nowhere'"):
            asyncio.run(HttpDriver("chat", "vllm").reset(_Session([]), "nowhere"))

    def test_a_refused_connection_raises_with_the_url(self):
        with self.assertRaisesRegex(ConnectionError, "POST http://127.0.0.1:9/reset_prefix_cache"):
            asyncio.run(
                HttpDriver("chat", "vllm", url="http://127.0.0.1:9/v1/x").reset(
                    _Session([], error=aiohttp.ClientConnectionError("refused"))
                )
            )


class BoundDriver(unittest.TestCase):
    def driver(self, endpoint="completions", chunks=({"choices": [{"text": "a"}]},)):
        drv = HttpDriver(
            endpoint, "vllm-completions", url="http://h:8000/v1/completions", model="m", api_key="k", timeout=9
        )
        self.stub = _Session(list(chunks))

        @asynccontextmanager
        async def session(concurrency, open_loop=False):
            drv._http = self.stub
            yield self.stub

        return drv, session

    def test_version_asks_the_server_once(self):
        drv, _ = self.driver()
        reply = mock.MagicMock()
        reply.__enter__.return_value = io.BytesIO(b'{"version": "0.30.1"}')
        with mock.patch.object(drivers.launcher, "urlopen", return_value=reply) as urlopen:
            self.assertEqual((drv.version, drv.version), ("0.30.1", "0.30.1"))
        urlopen.assert_called_once_with("http://h:8000/version", timeout=10)

    def test_send_routes_token_ids_and_sampling(self):
        drv, session = self.driver()

        async def go():
            async with session(1):
                return await drv.send(
                    Request([5, 6], 2, output_length=3), ignore_eos=True, seed=7, min_tokens=3, request_id="r1"
                )

        result = asyncio.run(go())
        url, body = self.stub.posts[0]
        self.assertEqual(url, "http://h:8000/v1/completions")
        self.assertEqual(
            (
                body["prompt"],
                body["max_tokens"],
                body["seed"],
                body["min_tokens"],
                body["ignore_eos"],
                body["request_id"],
            ),
            ([5, 6], 3, 7, 3, True, "r1"),
        )
        self.assertEqual(result.ttft_rule, "first_nonempty_text")

    def test_chat_send_uses_messages_and_captures_text_when_asked(self):
        drv, session = self.driver("chat", [{"choices": [{"delta": {"content": "hey"}}]}])
        msgs = [{"role": "user", "content": "hi"}]

        async def go():
            async with session(1):
                return await drv.send(Request([], 0, output_length=2, messages=msgs), capture_text=True)

        result = asyncio.run(go())
        self.assertEqual(self.stub.posts[0][1]["messages"], msgs)
        self.assertEqual(result.generated_text, "hey")

    def test_a_request_with_a_tool_schema_sends_it_and_one_without_sends_none(self):
        drv, session = self.driver("chat", [{"choices": [{"delta": {"content": "x"}}]}])
        tools = [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}]
        msgs = [{"role": "user", "content": "hi"}]

        async def go():
            async with session(1):
                await drv.send(Request([], 0, output_length=2, messages=msgs, tools=tools))
                await drv.send(Request([], 0, output_length=2, messages=msgs))

        asyncio.run(go())
        self.assertEqual(self.stub.posts[0][1]["tools"], tools)
        self.assertNotIn("tools", self.stub.posts[1][1])

    def test_completions_refuses_a_tool_schema(self):
        drv, session = self.driver("completions", [{"choices": [{"text": "x"}]}])
        request = Request([], 0, output_length=2, messages=[{"role": "user", "content": "hi"}], tools=[{}])

        async def go():
            async with session(1):
                return await drv.send(request)

        result = asyncio.run(go())
        self.assertEqual((result.success, result.error_kind), (False, "client_error"))
        self.assertIn("cannot send a tool schema", result.error)

    def test_run_yields_in_request_order(self):
        drv, session = self.driver()
        with mock.patch.object(drv, "session", session):
            out = list(drv.run([Request([i], 1, output_length=1) for i in range(5)], concurrency=2))
        self.assertEqual(len(out), 5)
        self.assertEqual(sorted(b["prompt"] for _, b in self.stub.posts), [[i] for i in range(5)])
        self.assertTrue(all(r.success for r in out))

    def test_reset_defaults_to_the_bound_server(self):
        drv, session = self.driver()

        async def go():
            async with session(1):
                return await drv.reset()

        self.assertEqual(asyncio.run(go()), "prefix cache reset via http://h:8000/reset_prefix_cache")


class GridOverHttp(unittest.TestCase):
    def driver(self, endpoint):
        return HttpDriver("completions", url=endpoint + "/v1/completions", model="m", timeout=9)

    def cell(self, ctx, bs, step, attempt=0):
        return {"ctx": ctx, "bs": bs, "tp": 1, "step_point": step, "phase": "fit", "attempt": attempt}

    def batch(self, drv, ctx, bs, step):
        requests = [Request([i] * ctx, ctx, output_length=step) for i in range(bs)]
        return list(drv.run(requests, cell=self.cell(ctx, bs, step)))

    def test_rows_have_the_in_process_shape(self):
        with helpers.serve() as (endpoint, posts):
            rows = self.batch(self.driver(endpoint), 4, 3, 5)
        sent = [b for path, b in posts if path == "/v1/completions"]
        self.assertEqual(len(sent), 3)
        for body in sent:
            self.assertEqual(
                (body["max_tokens"], body["ignore_eos"], body["temperature"], len(body["prompt"])), (5, True, 0.0, 4)
            )
        self.assertEqual(len(rows), 3)
        for r in rows:
            self.assertEqual(
                (r.success, r.e2el, r.ttft, r.output_tokens, r.input_tokens, r.ttft_basis),
                (True, r.batch_wall_s, None, 5, 4, "batch"),
            )
            self.assertEqual(r.cell, self.cell(4, 3, 5))
        self.assertEqual(len({r.batch_wall_s for r in rows}), 1)

    def test_cache_resets_once_per_cell(self):
        with helpers.serve() as (endpoint, posts):
            drv = self.driver(endpoint)
            drv.warmup()
            for ctx, bs, step in ((4, 2, 3), (4, 2, 6), (8, 1, 3)):
                self.batch(drv, ctx, bs, step)
        paths = [path for path, _ in posts]
        self.assertEqual(
            paths,
            [
                "/v1/completions",
                "/reset_prefix_cache",
                *["/v1/completions"] * 4,
                "/reset_prefix_cache",
                "/v1/completions",
            ],
        )

    def test_step_fit_reads_the_rows_unchanged(self):
        with helpers.serve() as (endpoint, _):
            drv = self.driver(endpoint)
            rows = [r for step in (3, 6, 9) for r in self.batch(drv, 4, 2, step)]
        (step,) = fit_run(rows, 4, 131072, 15.0)
        self.assertEqual((step["ctx"], step["bs"], step["step_points"]), (4, 2, [3, 6, 9]))
        self.assertEqual(step["raw"], [rows[0].batch_wall_s, rows[2].batch_wall_s, rows[4].batch_wall_s])

    def test_a_short_decode_fails_the_batch(self):
        with helpers.serve() as (endpoint, _):
            drv = self.driver(endpoint)
            ok = drv._stream

            async def short(*args, **kw):
                result = await ok(*args, **kw)
                result.output_tokens -= 1
                return result

            drv._stream = short
            with self.assertRaisesRegex(GridBatchFailed, "ctx=4 bs=2 step=5: 4 tokens decoded"):
                self.batch(drv, 4, 2, 5)

    def test_a_batch_without_usage_fails(self):
        with helpers.serve() as (endpoint, _):
            drv = self.driver(endpoint)
            ok = drv._stream

            async def no_usage(*args, **kw):
                result = await ok(*args, **kw)
                result.usage_reported = False
                return result

            drv._stream = no_usage
            with self.assertRaisesRegex(GridBatchFailed, "step=5: no usage reported"):
                self.batch(drv, 4, 2, 5)


send_request = HttpDriver("chat", "openai").send_request


def _chat_sse(chunks: list[dict]) -> list[bytes]:
    lines = [b"data: " + json.dumps(c).encode() + b"\n" for c in chunks]
    return lines + [b"data: [DONE]\n"]


def _delta(content=None, **extra) -> dict:
    d = dict(extra)
    d["content"] = content
    return {"choices": [{"delta": d}]}


class _ChatResp:
    def __init__(self, lines, status=200):
        self.status = status
        self.content = self._Lines(lines)

    class _Lines:
        def __init__(self, lines):
            self._lines = lines

        def __aiter__(self):
            async def gen():
                for line in self._lines:
                    yield line

            return gen()

    async def text(self):
        return "stub"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _ChatSession:
    def __init__(self, lines, status=200):
        self._lines, self._status = lines, status
        self.payload = None

    def post(self, url, headers=None, json=None):
        self.payload = json
        return _ChatResp(self._lines, self._status)


def _run(session):
    return asyncio.run(
        send_request(
            session=session,
            url="http://stub/v1/chat/completions",
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=8,
            capture_text=True,
        )
    )


class PinnedOutputTests(unittest.TestCase):
    def test_min_tokens_is_sent_when_given(self):
        session = _ChatSession(_chat_sse([_delta("a")]))
        asyncio.run(
            send_request(
                session=session,
                url="http://stub/v1/chat/completions",
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=120,
                min_tokens=120,
            )
        )
        self.assertEqual(session.payload["min_tokens"], 120)
        self.assertEqual(session.payload["max_tokens"], 120)

    def test_absent_unless_asked(self):
        session = _ChatSession(_chat_sse([_delta("a")]))
        asyncio.run(
            send_request(
                session=session,
                url="http://stub/v1/chat/completions",
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=120,
            )
        )
        self.assertNotIn("min_tokens", session.payload)

    def test_each_turn_keeps_its_own_length(self):
        for planned in (37, 204, 91):
            session = _ChatSession(_chat_sse([_delta("a")]))
            asyncio.run(
                send_request(
                    session=session,
                    url="http://stub/v1/chat/completions",
                    model="m",
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=planned,
                    min_tokens=planned,
                )
            )
            self.assertEqual(session.payload["min_tokens"], planned)


class CaptureTextTests(unittest.TestCase):
    def test_content_deltas_are_joined(self):
        r = _run(
            _ChatSession(
                _chat_sse(
                    [
                        _delta("Hel"),
                        _delta("lo"),
                        _delta(" there"),
                        {"choices": [{"delta": {}}], "usage": {"prompt_tokens": 4, "completion_tokens": 3}},
                    ]
                )
            )
        )
        self.assertTrue(r.success, msg=r.error)
        self.assertEqual(r.generated_text, "Hello there")

    def test_billed_cached_tokens_come_from_usage_and_reach_the_row(self):
        usage = {"prompt_tokens": 4, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 3}}
        r = _run(_ChatSession(_chat_sse([_delta("a"), {"choices": [{"delta": {}}], "usage": usage}])))
        self.assertEqual(r.server_cached_tokens, 3)
        self.assertEqual(jsonl_row(r)["server_cached_tokens"], 3)
        plain = _run(_ChatSession(_chat_sse([_delta("a")])))
        self.assertIsNone(plain.server_cached_tokens)
        self.assertNotIn("server_cached_tokens", jsonl_row(plain))

    def test_reasoning_and_tool_calls_are_not_transcript_content(self):
        r = _run(
            _ChatSession(
                _chat_sse(
                    [
                        _delta(None, reasoning_content="thinking"),
                        _delta("answer"),
                        _delta(None, tool_calls=[{"index": 0}]),
                    ]
                )
            )
        )
        self.assertTrue(r.success, msg=r.error)
        self.assertEqual(r.generated_text, "answer")
        # three payload-bearing chunks: one ttft plus two inter-token gaps
        self.assertEqual(len(r.itl), 2)

    def test_capture_off_returns_none_and_still_measures(self):
        session = _ChatSession(_chat_sse([_delta("a"), _delta("b")]))
        r = asyncio.run(
            send_request(
                session=session,
                url="http://stub/v1/chat/completions",
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=8,
            )
        )
        self.assertTrue(r.success, msg=r.error)
        self.assertIsNone(r.generated_text)
        self.assertIsNotNone(r.ttft)

    def test_empty_reply_is_not_an_error(self):
        r = _run(_ChatSession(_chat_sse([_delta(None, tool_calls=[{"index": 0}])])))
        self.assertTrue(r.success, msg=r.error)
        self.assertEqual(r.generated_text, "")

    def test_non_200_is_a_failure_with_the_status_in_it(self):
        r = _run(_ChatSession([], status=503))
        self.assertFalse(r.success)
        self.assertIn("503", str(r.error))


# aiohttp is imported lazily so metric-only tests run without it.
try:
    import aiohttp
    from aiohttp import web

    HAVE_AIOHTTP = True
except ModuleNotFoundError:
    HAVE_AIOHTTP = False


class TestTpotCoverageGuard(unittest.TestCase):
    def test_full_coverage_keeps_mean_itl(self):
        r = RequestResult(success=True, ttft=0.05, e2el=1.05, itl=[0.01] * 99, output_tokens=100)
        self.assertAlmostEqual(r.tpot, 0.01)

    def test_sparse_coverage_falls_back_to_wall_clock(self):
        # gpt-oss-on-vllm signature: 54 chunks for 269 tokens, one giant gap.
        itl = [2.505] + [0.0104] * 53
        r = RequestResult(success=True, ttft=0.25, e2el=3.32, itl=itl, output_tokens=269)
        # Wall clock gives ~11.5 ms/token; mean(itl) would say ~57.
        self.assertAlmostEqual(r.tpot, (3.32 - 0.25) / 268)
        self.assertLess(r.tpot, 0.013)

    def test_short_outputs_keep_mean_itl(self):
        r = RequestResult(success=True, ttft=0.05, e2el=0.10, itl=[0.011, 0.012], output_tokens=3)
        self.assertAlmostEqual(r.tpot, (0.011 + 0.012) / 2)

    def test_no_itl_falls_back(self):
        r = RequestResult(success=True, ttft=0.4, e2el=2.4, itl=[], output_tokens=201)
        self.assertAlmostEqual(r.tpot, 2.0 / 200)

    def test_no_itl_no_ttft_is_none(self):
        r = RequestResult(success=True, e2el=2.4, output_tokens=201)
        self.assertIsNone(r.tpot)

    def test_sparse_coverage_without_wall_clock_keeps_mean(self):
        r = RequestResult(success=True, itl=[0.5, 0.01], output_tokens=100)
        self.assertAlmostEqual(r.tpot, 0.255)


def _sse(payload) -> bytes:
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


def _chunk(delta, finish_reason=None):
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}


async def _serve_and_request(chunks):
    from quettabench.engines.drivers import HttpDriver

    send_request = HttpDriver("chat", "openai").send_request

    async def handler(request):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for c in chunks:
            await resp.write(_sse(c))
        await resp.write(b"data: [DONE]\n\n")
        return resp

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        async with aiohttp.ClientSession() as session:
            return await send_request(
                session,
                f"http://127.0.0.1:{port}/v1/chat/completions",
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=16,
            )
    finally:
        await runner.cleanup()


@unittest.skipUnless(HAVE_AIOHTTP, "aiohttp not installed")
class TestReasoningChunksCounted(unittest.TestCase):
    def test_reasoning_content_chunks_are_token_events(self):
        # gpt-oss harmony shape: role chunk, reasoning deltas, then content deltas.
        chunks = (
            [_chunk({"role": "assistant", "content": ""})]
            + [_chunk({"reasoning_content": "r"}) for _ in range(3)]
            + [_chunk({"content": "t"}) for _ in range(4)]
            + [_chunk({}, finish_reason="stop")]
            + [{"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 8}}]
        )
        r = asyncio.run(_serve_and_request(chunks))
        self.assertTrue(r.success)
        # 8 token events (1 role + 3 reasoning + 4 content): ttft + 7 ITLs.
        self.assertIsNotNone(r.ttft)
        self.assertEqual(len(r.itl), 7)
        self.assertEqual(r.output_tokens, 8)

    def test_newer_reasoning_field_and_tool_calls_count(self):
        chunks = (
            [_chunk({"role": "assistant", "content": ""})]
            + [_chunk({"reasoning": "r"}) for _ in range(2)]
            + [_chunk({"tool_calls": [{"index": 0, "function": {"arguments": "{"}}]}) for _ in range(2)]
            + [_chunk({"content": "t"})]
        )
        r = asyncio.run(_serve_and_request(chunks))
        self.assertTrue(r.success)
        self.assertEqual(len(r.itl), 5)

    def test_plain_content_stream_unchanged(self):
        chunks = (
            [_chunk({"role": "assistant", "content": ""})]
            + [_chunk({"content": "t"}) for _ in range(5)]
            + [_chunk({}, finish_reason="stop")]
        )
        r = asyncio.run(_serve_and_request(chunks))
        self.assertTrue(r.success)
        self.assertEqual(len(r.itl), 5)


FAKE = Path(__file__).with_name("fake_bench.py")


def cell_spec(ctx, bs, **env):
    return LaunchSpec(
        adapter="quettaserve",
        bench_argv=(sys.executable, str(FAKE)),
        cwd=".",
        env={"QW_BENCH_PROMPT": str(ctx), "QW_BATCH": str(bs), "QW_WORLD": "1", **env},
        ok_pattern=r"^LOOP ",
        log_header=f">>> ctx={ctx} bs={bs} world=1",
    )


def alive(pid):
    """A zombie waiting for its reaper counts as gone."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ")[1][0] != "Z"
    except (OSError, IndexError):
        return False


def as_points(rows):
    return {
        (r["cell"]["ctx"], r["cell"]["bs"], r["cell"]["tp"]): (r["measured_ms_per_step"], r["group"], r["kv"])
        for r in rows
    }


class GoldenLogs(unittest.TestCase):
    def test_rows_equal_the_reader_parse(self):
        for name in ("bench_loop.log", "bench_criterion.log"):
            with self.subTest(log=name):
                path = FIXTURES / name
                rows = list(BenchDriver(path).rows())
                points, method, _ = compare.bench_log_points(path)
                self.assertTrue(points)
                self.assertEqual(as_points(rows), points)
                self.assertEqual({r["method"] for r in rows}, {method})


class CellLoop(unittest.TestCase):
    def setUp(self):
        self.log = Path(tempfile.mkdtemp()) / "bench.log"

    def test_two_cells_log_and_rows(self):
        rows = list(BenchDriver(self.log).run([(1024, 1, cell_spec(1024, 1)), (1024, 16, cell_spec(1024, 16))]))
        self.assertEqual(
            as_points(rows), {(1024, 1, 1): (3.5, "decode_loop", "fp16"), (1024, 16, 1): (3.5, "decode_loop", "fp16")}
        )
        lines = self.log.read_text().splitlines()
        self.assertEqual(lines[0], ">>> ctx=1024 bs=1 world=1")
        self.assertIn("bench chatter", lines)

    def test_missing_result_line_fails_after_all_cells(self):
        with self.assertRaisesRegex(BenchFailed, r"1 cell\(s\) failed"):
            list(BenchDriver(self.log).run([(1, 1, cell_spec(1, 1, FAKE_BENCH_SILENT="1")), (2, 1, cell_spec(2, 1))]))
        text = self.log.read_text()
        self.assertIn("FAIL ctx=1 bs=1 (no bench result line", text)
        self.assertIn("LOOP ctx=2 bs=1", text)

    def test_nonzero_exit_stops_the_sweep(self):
        with self.assertRaisesRegex(BenchFailed, "exited 3"):
            list(BenchDriver(self.log).run([(1, 1, cell_spec(1, 1, FAKE_BENCH_EXIT="3")), (2, 1, cell_spec(2, 1))]))
        self.assertNotIn("ctx=2", self.log.read_text())

    def test_an_interrupted_cell_kills_its_bench(self):
        started, real = [], drivers.subprocess.Popen

        def popen(*args, **kwargs):
            started.append(real(*args, **kwargs))
            return started[-1]

        with (
            mock.patch.object(drivers.subprocess, "Popen", side_effect=popen),
            mock.patch.object(drivers.sys, "stdout") as out,
            self.assertRaises(KeyboardInterrupt),
        ):
            out.write.side_effect = KeyboardInterrupt
            list(BenchDriver(self.log).run([(1, 1, cell_spec(1, 1, FAKE_BENCH_SLEEP="60"))]))
        (proc,) = started
        self.assertIsNotNone(proc.poll())

    def test_an_interrupted_cell_kills_the_bench_children(self):
        pid_file = self.log.with_name("child.pid")
        spec = cell_spec(1, 1, FAKE_BENCH_SLEEP="60", FAKE_BENCH_CHILD=str(pid_file))
        with mock.patch.object(drivers.sys, "stdout") as out, self.assertRaises(KeyboardInterrupt):
            out.write.side_effect = KeyboardInterrupt
            list(BenchDriver(self.log).run([(1, 1, spec)]))
        pid = int(pid_file.read_text())
        deadline = time.monotonic() + 5
        while alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(alive(pid))

    def test_a_finished_cell_leaves_no_bench_child(self):
        """A bench that exits on its own can leave a child (cargo's) holding the GPUs."""
        pid_file = self.log.with_name("child.pid")
        rows = list(BenchDriver(self.log).run([(1, 1, cell_spec(1, 1, FAKE_BENCH_CHILD=str(pid_file)))]))
        self.assertEqual(len(rows), 1)
        pid = int(pid_file.read_text())
        deadline = time.monotonic() + 5
        while alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(alive(pid))


KWARGS = {"model": "m", "tensor_parallel_size": 2}
CELL = {"ctx": 8, "bs": 3, "tp": 1, "step_point": 5, "phase": "fit", "attempt": 0}


class Builders(unittest.TestCase):
    def test_llm_gets_the_kwargs_on_first_use(self):
        with compare_path(stubs=True):
            vllm = fresh_import("vllm")
            vllm.reset()
            driver = InProcessDriver(LaunchSpec(adapter="vllm", engine_kwargs=KWARGS))
            self.assertEqual(vllm.CALLS, [])
            driver.llm()
            self.assertEqual(vllm.CALLS, [("LLM", KWARGS)])

    def test_llm_is_built_once(self):
        with compare_path(stubs=True):
            vllm = fresh_import("vllm")
            vllm.reset()
            driver = InProcessDriver(LaunchSpec(adapter="vllm", engine_kwargs=KWARGS))
            self.assertIs(driver.llm(), driver.llm())
            self.assertEqual(vllm.CALLS, [("LLM", KWARGS)])

    def test_spec_env_is_in_the_process_before_llm(self):
        env = {"VLLM_SERVER_DEV_MODE": "1", "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1"}
        seen = {}
        with compare_path(stubs=True), mock.patch.dict(os.environ, {"VLLM_SERVER_DEV_MODE": "0"}, clear=True):
            vllm = fresh_import("vllm")
            vllm.reset()
            real = vllm.LLM

            def llm(**kw):
                seen.update({k: os.environ.get(k) for k in env})
                return real(**kw)

            with mock.patch.object(vllm, "LLM", llm):
                InProcessDriver(LaunchSpec(adapter="vllm", engine_kwargs=KWARGS, env=env)).llm()
        self.assertEqual(seen, env)


class GridBatch(unittest.TestCase):
    def run_batch(self, seconds):
        with compare_path(stubs=True):
            vllm = fresh_import("vllm")
            vllm.reset()
            vllm.generate_seconds = lambda prompt_len, batch, max_tokens, call_no: seconds
            driver = InProcessDriver(LaunchSpec(adapter="vllm", engine_kwargs=KWARGS))
            requests = [Request([7] * 8, 8, output_length=5) for _ in range(3)]
            clock = types.SimpleNamespace(perf_counter=vllm.CLOCK.perf_counter)
            with mock.patch.object(drivers, "time", clock):
                driver.warmup()
                rows = list(driver.run(requests, cell=CELL))
            return rows, vllm.CLOCK.generate_calls

    def test_one_generate_shared_wall(self):
        rows, calls = self.run_batch(0.25)
        self.assertEqual(calls, 2)
        self.assertEqual(len(rows), 3)
        for r in rows:
            self.assertEqual((r.e2el, r.batch_wall_s, r.ttft), (0.25, 0.25, None))
            self.assertEqual((r.output_tokens, r.input_tokens, r.ttft_basis), (5, 8, "batch"))
            self.assertEqual(r.cell, CELL)
            self.assertEqual(r.generated_token_ids, [0, 1, 2, 3, 4])
        self.assertIsNot(rows[0].cell, CELL)


LOGITS_CELL = {"ctx": 5, "bs": 2, "tp": 1, "step_point": None, "phase": "logits", "attempt": None}
PROMPTS = [[3, 1, 4, 1, 5], [9, 2, 6, 5, 3]]


def requests():
    return [Request(ids, len(ids)) for ids in PROMPTS]


class Halves(unittest.TestCase):
    def vllm_rows(self):
        vllm = fresh_import("vllm")
        vllm.reset()
        driver = InProcessDriver(LaunchSpec(adapter="vllm", engine_kwargs={"model": "m"}))
        with mock.patch.object(vllm.LLM, "generate", autospec=True, side_effect=vllm.LLM.generate) as generate:
            rows = list(driver.run(requests(), cell=LOGITS_CELL, return_outputs="logits"))
        (_, prompts, params), kwargs = generate.call_args
        self.assertEqual(params.kwargs, {"temperature": 0, "max_tokens": 1, "prompt_logprobs": 2})
        self.assertEqual(prompts, [{"prompt_token_ids": ids} for ids in PROMPTS])
        self.assertEqual(kwargs, {"use_tqdm": False})
        return rows

    def reference_rows(self):
        transformers = fresh_import("transformers")
        torch = fresh_import("torch")
        transformers.CALLS.clear()
        driver = ReferenceDriver(LaunchSpec(adapter="reference", engine_kwargs={"model": "/w", "dtype": "bfloat16"}))
        rows = list(driver.run(requests(), cell=LOGITS_CELL))
        self.assertEqual(
            transformers.CALLS,
            [
                (
                    "AutoModelForCausalLM.from_pretrained",
                    "/w",
                    {"torch_dtype": torch.bfloat16, "trust_remote_code": True},
                ),
                ("to", "cuda:0"),
                ("eval",),
            ],
        )
        return rows

    def test_layers_cut_the_config_as_slice_truncate_did(self):
        with compare_path(stubs=True):
            transformers = fresh_import("transformers")
            torch = fresh_import("torch")
            transformers.CALLS.clear()
            driver = ReferenceDriver(
                LaunchSpec(adapter="reference", engine_kwargs={"model": "/w", "dtype": "float32", "layers": 2})
            )
            list(driver.run(requests(), cell=LOGITS_CELL))
            (_, path, kwargs) = transformers.CALLS[1]
            self.assertEqual(transformers.CALLS[0], ("AutoConfig.from_pretrained", "/w", {"trust_remote_code": True}))
            self.assertEqual((path, kwargs["torch_dtype"]), ("/w", torch.float32))
            self.assertEqual((kwargs["config"].num_hidden_layers, kwargs["config"].layer_types), (2, ["a", "b"]))
            self.assertEqual(len(driver.model().model.layers), 2)
            deep = ReferenceDriver(
                LaunchSpec(adapter="reference", engine_kwargs={"model": "/w", "dtype": "float32", "layers": 5})
            )
            with self.assertRaisesRegex(ValueError, "exceeds the checkpoint's 4 layers"):
                deep.model()

    def test_summaries_line_up(self):
        with compare_path(stubs=True):
            ours, ref = self.vllm_rows(), self.reference_rows()
        for o, r, ids in zip(ours, ref, PROMPTS):
            self.assertEqual(o.cell, LOGITS_CELL)
            self.assertEqual(o.logits["argmax"], [t + 1 for t in ids[:-1]])
            self.assertEqual(r.logits["argmax"], o.logits["argmax"])
            self.assertEqual(len(r.logits["logprob"]), len(ids) - 1)
            for a, b in zip(o.logits["logprob"] + o.logits["top2_gap"], r.logits["logprob"] + r.logits["top2_gap"]):
                self.assertAlmostEqual(a, b, places=12)
        summary = compare.logit(
            [{"cell": LOGITS_CELL, "logits": o.logits} for o in ours],
            [{"cell": LOGITS_CELL, "logits": r.logits} for r in ref],
        )[-1]
        self.assertEqual((summary["positions"], summary["argmax_agree"], summary["verdict"]), (8, 8, "PASS"))


class Recipe(unittest.TestCase):
    def test_run_run_compare_is_fidelity_and_band_pass(self):
        sets = ["--set", "mode.grid=smoke"]
        with tempfile.TemporaryDirectory() as td:
            for name, extra in (
                ("ours", ["--set", "mode.return_outputs=logits"]),
                ("ref", ["--set", "engine.name=reference"]),
                ("truncated", ["--set", "engine.name=reference", "--set", "engine.layers=2"]),
            ):
                run = run_grid_cli(
                    grid_argv("llama", *sets, *extra, "--set", f"run_name={name}", "--results-dir", td), {}
                )
                self.assertIsNone(run["exit"])
            from quettabench import compare

            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                compare.main(
                    [
                        str(Path(td, "ours")),
                        str(Path(td, "ref")),
                        "--floor",
                        str(Path(td, "truncated")),
                        "--json",
                        str(Path(td, "x.json")),
                    ]
                )
        self.assertIn("SUMMARY positions=99 argmax_agree=99 (100.00%)", out.getvalue())
        self.assertIn("FIDELITY verdict=PASS hard_disagreements=0", out.getvalue())
        self.assertIn("BAND verdict=PASS flips=0 floor=0 floor_runs=0 positions=99", out.getvalue())


RUNS = REPO / "configs" / "runs"
PINNED = ["allow_unverified=true", "hardware.backend=cpu", "hardware.link_profile=nvlink"]


def config(name, *sets):
    (cfg,) = loader.load(RUNS / f"{name}.yaml", [*PINNED, *sets], env={})
    return cfg


class VllmDriver(unittest.TestCase):
    def test_grid_run_with_returns_is_in_process(self):
        cfg = config("llama", "engine.model=/w", "mode.return_outputs=text")
        self.assertFalse(vllm_adapter.ADAPTER.needs_server(cfg))
        spec = vllm_adapter.ADAPTER.translate(cfg)
        driver = vllm_adapter.ADAPTER.driver(spec, launcher.NoServer(Path("/r")))
        self.assertIsInstance(driver, InProcessDriver)

    def test_serve_run_talks_to_the_launched_server(self):
        cfg = config("llama", "engine.model=/w", "mode.grid=null", "mode.timeout=7")
        self.assertTrue(vllm_adapter.ADAPTER.needs_server(cfg))
        spec = vllm_adapter.ADAPTER.translate(cfg)
        handle = launcher.NoServer(Path("/r"), endpoint="http://127.0.0.1:8000")
        driver = vllm_adapter.ADAPTER.driver(spec, handle)
        self.assertIsInstance(driver, HttpDriver)
        self.assertEqual((driver.url, driver.model, driver.timeout), ("http://127.0.0.1:8000/v1/completions", "/w", 7))


class QuettaServeBenchRun(unittest.TestCase):
    def test_grid_spec_lists_every_cell(self):
        cfg = config("qwen3", "engine.name=quettaserve", "engine.model=/ckpt/qw")
        self.assertFalse(quettaserve.ADAPTER.needs_server(cfg))
        spec = quettaserve.ADAPTER.translate(cfg)
        self.assertEqual([(c, b) for c, b, _ in spec.cells], list(cfg.grid.cells))
        for ctx, bs, cell in spec.cells:
            self.assertEqual(cell, quettaserve.ADAPTER.translate(cfg, cell=(ctx, bs)))

    def test_serve_run_writes_measured_rows(self):
        cfg = config("llama", "engine.model=/w", "mode.grid=smoke")
        cells = tuple(
            (
                ctx,
                bs,
                LaunchSpec(
                    adapter="quettaserve",
                    bench_argv=(sys.executable, str(FAKE)),
                    cwd=".",
                    env={"QW_BENCH_PROMPT": str(ctx), "QW_BATCH": str(bs), "QW_WORLD": "1"},
                    ok_pattern=r"^LOOP ",
                    log_header=f">>> ctx={ctx} bs={bs} world=1",
                ),
            )
            for ctx, bs in ((1024, 1), (1024, 16))
        )
        with tempfile.TemporaryDirectory() as td:
            driver = BenchDriver(Path(td) / "bench.log", cells)
            serve.run(None, driver, cfg.mode, RunWriter(td, cfg))
            doc = json.loads((Path(td) / "run.json").read_text())
            rows = [json.loads(line) for line in (Path(td) / "requests.jsonl").read_text().splitlines()]
        self.assertEqual(
            doc["steps"],
            [
                {"ctx": 1024, "bs": 1, "tp": 1, "ms_per_step": 3.5, "measured": True},
                {"ctx": 1024, "bs": 16, "tp": 1, "ms_per_step": 3.5, "measured": True},
            ],
        )
        self.assertEqual([(r["cell"]["bs"], r["measured_ms_per_step"]) for r in rows], [(1, 3.5), (16, 3.5)])
        self.assertNotIn("meta", doc["summary"]["grid"])


class QuettaServeThroughCli(unittest.TestCase):
    SETS = [
        *PINNED,
        "engine.name=quettaserve",
        "engine.model=/ckpt/qw",
        "engine.tp=1",
        f"engine.extra.bench_bin={FAKE}",
        "engine.max_model_len=10000",
        "mode.max_kv_fraction=0.3",
    ]

    def qbench(self, *argv):
        args = ["run", str(RUNS / "qwen3.yaml"), *(a for s in self.SETS for a in ("--set", s)), *argv]
        out = io.StringIO()
        with (
            contextlib.redirect_stdout(out),
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(BenchDriver, "device", return_value=("H200", 141 * 2**30)),
        ):
            self.assertEqual(cli.main(args), 0)
        return out.getvalue()

    def test_dry_run_prints_each_cells_shell_line(self):
        lines = [line for line in self.qbench("--dry-run").splitlines() if line.startswith("cell ")]
        self.assertEqual(len(lines), 16)
        self.assertEqual(
            lines[0],
            (
                "cell ctx=1024 bs=1 = env XVAL_LINK_PROFILE=nvlink NCCL_IB_DISABLE=1 QS_KERNELS=rust QW_MODE=graph "
                "QW_CKPT=/ckpt/qw QW_WORLD=1 QW_TIMING_STEPS=100 QW_BATCH=1 QW_BENCH_PROMPT=1024 "
                f"QW_MAX_SEQ=1152 {FAKE} --nocapture"
            ),
        )

    def test_run_skips_then_measures_the_rest(self):
        with tempfile.TemporaryDirectory() as td:
            out = self.qbench("--results-dir", td)
            run = Path(td) / "qwen3"
            doc = json.loads((run / "run.json").read_text())
            rows = [json.loads(line) for line in (run / "requests.jsonl").read_text().splitlines()]
        skips = [line for line in out.splitlines() if line.startswith("SKIP")]
        self.assertEqual(
            skips,
            [
                "SKIP ctx=8192 bs=64 need_kv=532480 cap=1112473",
                *(f"SKIP ctx=16384 bs={bs} reason=maxlen" for bs in (1, 4, 16, 64)),
            ],
        )
        self.assertEqual(len(rows), 11)
        self.assertEqual({r["measured_ms_per_step"] for r in rows}, {3.5})
        self.assertNotIn((8192, 64), {(r["cell"]["ctx"], r["cell"]["bs"]) for r in rows})
        self.assertEqual(len(doc["steps"]), 11)
        self.assertTrue(all(s["measured"] for s in doc["steps"]))
        self.assertEqual(len(doc["summary"]["grid"]["events"]), 5)


if __name__ == "__main__":
    unittest.main()
