"""HttpDriver keeps both first-token rules, picked by endpoint and recorded."""

import asyncio
import json
import unittest
from contextlib import asynccontextmanager
from unittest import mock

from quettabench.engines.drivers import HttpDriver, PrefixCacheResetError
from quettabench.workloads.base import Request


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
    def __init__(self, chunks, status=200):
        self.lines = [b"data: " + json.dumps(c).encode() + b"\n" for c in chunks]
        self.lines.append(b"data: [DONE]\n")
        self.status, self.posts = status, []

    def post(self, url, headers=None, json=None, timeout=None):
        self.posts.append((url, json))
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


if __name__ == "__main__":
    unittest.main()
