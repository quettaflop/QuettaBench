"""HttpDriver keeps both first-token rules, picked by endpoint and recorded."""

import asyncio
import json
import unittest
from contextlib import asynccontextmanager
from unittest import mock

import aiohttp

from quettabench.engines.drivers import GridBatchFailed, HttpDriver, PrefixCacheResetError
from quettabench.metrics.step_fit import fit_run
from quettabench.workloads.base import Request
from tests.profiling import fake_endpoint


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
    result = asyncio.run(HttpDriver(endpoint).send_request(
        session=session, url="http://stub/v1/x", model="m",
        messages=[{"role": "user", "content": "hi"}], max_tokens=4, **kw))
    return result, session.posts[0][1]


class TtftRules(unittest.TestCase):
    def test_chat_fires_on_a_reasoning_chunk(self):
        r, payload = send("chat", [{"choices": [{"delta": {"reasoning": "r"}}]},
                                   {"choices": [{"delta": {"content": "a"}}]}])
        self.assertEqual((r.ttft_rule, len(r.itl)), ("first_payload_chunk", 1))
        self.assertEqual(payload["messages"][0]["content"], "hi")

    def test_completions_skips_empty_text(self):
        r, payload = send("completions", [{"choices": [{"text": ""}]},
                                          {"choices": [{"text": "a"}]},
                                          {"choices": [{"text": "b"}]}],
                          prompt_token_ids=[1, 2])
        self.assertEqual((r.ttft_rule, len(r.itl)), ("first_nonempty_text", 1))
        self.assertEqual(payload["prompt"], [1, 2])

    def test_completions_rejects_capture_text_as_before(self):
        with self.assertRaises(TypeError):
            send("completions", [], capture_text=True)

    def test_completions_without_a_prompt_is_a_client_error(self):
        r = asyncio.run(HttpDriver("completions").send_request(
            session=None, url="u", model="m", messages=[], max_tokens=1))
        self.assertEqual((r.success, r.error_kind, r.ttft_rule),
                         (False, "client_error", "first_nonempty_text"))


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
            asyncio.run(HttpDriver("chat", "vllm", url="http://127.0.0.1:9/v1/x").reset(
                _Session([], error=aiohttp.ClientConnectionError("refused"))))


class BoundDriver(unittest.TestCase):
    """send/run/reset against the server the driver was built for."""

    def driver(self, endpoint="completions", chunks=({"choices": [{"text": "a"}]},)):
        drv = HttpDriver(endpoint, "vllm-completions", url="http://h:8000/v1/completions",
                         model="m", api_key="k", timeout=9)
        self.stub = _Session(list(chunks))

        @asynccontextmanager
        async def session(concurrency, open_loop=False):
            drv._http = self.stub
            yield self.stub

        return drv, session

    def test_send_routes_token_ids_and_sampling(self):
        drv, session = self.driver()

        async def go():
            async with session(1):
                return await drv.send(Request([5, 6], 2, output_length=3), ignore_eos=True,
                                      seed=7, min_tokens=3, request_id="r1")

        result = asyncio.run(go())
        url, body = self.stub.posts[0]
        self.assertEqual(url, "http://h:8000/v1/completions")
        self.assertEqual((body["prompt"], body["max_tokens"], body["seed"], body["min_tokens"],
                          body["ignore_eos"], body["request_id"]), ([5, 6], 3, 7, 3, True, "r1"))
        self.assertEqual(result.ttft_rule, "first_nonempty_text")

    def test_chat_send_uses_messages_and_captures_text_when_asked(self):
        drv, session = self.driver("chat", [{"choices": [{"delta": {"content": "hey"}}]}])
        msgs = [{"role": "user", "content": "hi"}]

        async def go():
            async with session(1):
                return await drv.send(Request([], 0, output_length=2, messages=msgs),
                                      capture_text=True)

        result = asyncio.run(go())
        self.assertEqual(self.stub.posts[0][1]["messages"], msgs)
        self.assertEqual(result.generated_text, "hey")

    def test_run_yields_in_request_order(self):
        drv, session = self.driver()
        with mock.patch.object(drv, "session", session):
            out = list(drv.run([Request([i], 1, output_length=1) for i in range(5)],
                               concurrency=2))
        self.assertEqual(len(out), 5)
        self.assertEqual(sorted(b["prompt"] for _, b in self.stub.posts), [[i] for i in range(5)])
        self.assertTrue(all(r.success for r in out))

    def test_reset_defaults_to_the_bound_server(self):
        drv, session = self.driver()

        async def go():
            async with session(1):
                return await drv.reset()

        self.assertEqual(asyncio.run(go()),
                         "prefix cache reset via http://h:8000/reset_prefix_cache")


class GridOverHttp(unittest.TestCase):
    """A grid batch is bs concurrent completions against the fake server."""

    def driver(self, endpoint):
        return HttpDriver("completions", url=endpoint + "/v1/completions", model="m", timeout=9)

    def cell(self, ctx, bs, step, attempt=0):
        return {"ctx": ctx, "bs": bs, "tp": 1, "step_point": step, "phase": "fit",
                "attempt": attempt}

    def batch(self, drv, ctx, bs, step):
        requests = [Request([i] * ctx, ctx, output_length=step) for i in range(bs)]
        return list(drv.run(requests, cell=self.cell(ctx, bs, step)))

    def test_rows_have_the_in_process_shape(self):
        with fake_endpoint.serve() as (endpoint, posts):
            rows = self.batch(self.driver(endpoint), 4, 3, 5)
        sent = [b for path, b in posts if path == "/v1/completions"]
        self.assertEqual(len(sent), 3)
        for body in sent:
            self.assertEqual((body["max_tokens"], body["ignore_eos"], body["temperature"],
                              len(body["prompt"])), (5, True, 0.0, 4))
        self.assertEqual(len(rows), 3)
        for r in rows:
            self.assertEqual((r.success, r.e2el, r.ttft, r.output_tokens, r.input_tokens,
                              r.ttft_basis), (True, r.batch_wall_s, None, 5, 4, "batch"))
            self.assertEqual(r.cell, self.cell(4, 3, 5))
        self.assertEqual(len({r.batch_wall_s for r in rows}), 1)

    def test_cache_resets_once_per_cell(self):
        with fake_endpoint.serve() as (endpoint, posts):
            drv = self.driver(endpoint)
            drv.warmup()
            for ctx, bs, step in ((4, 2, 3), (4, 2, 6), (8, 1, 3)):
                self.batch(drv, ctx, bs, step)
        paths = [path for path, _ in posts]
        self.assertEqual(paths, ["/v1/completions", "/reset_prefix_cache",
                                 *["/v1/completions"] * 4, "/reset_prefix_cache",
                                 "/v1/completions"])

    def test_step_fit_reads_the_rows_unchanged(self):
        with fake_endpoint.serve() as (endpoint, _):
            drv = self.driver(endpoint)
            rows = [r for step in (3, 6, 9) for r in self.batch(drv, 4, 2, step)]
        (step,) = fit_run(rows, 4, 131072, 15.0)
        self.assertEqual((step["ctx"], step["bs"], step["step_points"]), (4, 2, [3, 6, 9]))
        self.assertEqual(step["raw"], [rows[0].batch_wall_s, rows[2].batch_wall_s,
                                       rows[4].batch_wall_s])

    def test_a_short_decode_fails_the_batch(self):
        with fake_endpoint.serve() as (endpoint, _):
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
        with fake_endpoint.serve() as (endpoint, _):
            drv = self.driver(endpoint)
            ok = drv._stream

            async def no_usage(*args, **kw):
                result = await ok(*args, **kw)
                result.usage_reported = False
                return result

            drv._stream = no_usage
            with self.assertRaisesRegex(GridBatchFailed, "step=5: no usage reported"):
                self.batch(drv, 4, 2, 5)


if __name__ == "__main__":
    unittest.main()
