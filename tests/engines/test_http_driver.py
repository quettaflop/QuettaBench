"""HttpDriver keeps both first-token rules, picked by endpoint and recorded."""

import asyncio
import json
import unittest

from quettabench.engines import get_backend
from quettabench.engines.drivers import HttpDriver, PrefixCacheResetError


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


class Backends(unittest.TestCase):
    def test_names_map_onto_endpoints(self):
        self.assertEqual(
            {n: get_backend(n).endpoint for n in ("openai", "vllm", "sglang", "vllm-completions")},
            {"openai": "chat", "vllm": "chat", "sglang": "chat", "vllm-completions": "completions"})
        with self.assertRaisesRegex(ValueError, "Unknown backend 'tgi'"):
            get_backend("tgi")

    def test_reset_posts_to_the_dev_mode_endpoint(self):
        session = _Session([])
        status = asyncio.run(get_backend("vllm").reset(session, "http://h:8000/v1/completions"))
        self.assertEqual(status, "prefix cache reset via http://h:8000/reset_prefix_cache")

    def test_reset_has_no_sglang_endpoint(self):
        with self.assertRaisesRegex(PrefixCacheResetError, "backend 'sglang'"):
            asyncio.run(get_backend("sglang").reset(_Session([]), "http://h:1/v1/x"))

    def test_reset_404_names_dev_mode(self):
        with self.assertRaisesRegex(PrefixCacheResetError, "VLLM_SERVER_DEV_MODE=1"):
            asyncio.run(get_backend("vllm").reset(_Session([], status=404), "http://h:1/v1/x"))


if __name__ == "__main__":
    unittest.main()
