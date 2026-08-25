"""send_request against a stubbed SSE stream, so the streaming loop needs no server."""

import asyncio
import json
import unittest

from src.engines.openai_chat import send_request


def _sse(chunks: list[dict]) -> list[bytes]:
    lines = [b"data: " + json.dumps(c).encode() + b"\n" for c in chunks]
    return lines + [b"data: [DONE]\n"]


def _delta(content=None, **extra) -> dict:
    d = dict(extra)
    d["content"] = content
    return {"choices": [{"delta": d}]}


class _Resp:
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


class _Session:
    """Just enough aiohttp.ClientSession for `send_request`."""

    def __init__(self, lines, status=200):
        self._lines, self._status = lines, status
        self.payload = None

    def post(self, url, headers=None, json=None):
        self.payload = json
        return _Resp(self._lines, self._status)


def _run(session):
    return asyncio.run(send_request(
        session=session, url="http://stub/v1/chat/completions", model="m",
        messages=[{"role": "user", "content": "hi"}], max_tokens=8,
        capture_text=True,
    ))


class PinnedOutputTests(unittest.TestCase):
    """min_tokens must reach the payload, per request."""

    def test_min_tokens_is_sent_when_given(self):
        session = _Session(_sse([_delta("a")]))
        asyncio.run(send_request(
            session=session, url="http://stub/v1/chat/completions", model="m",
            messages=[{"role": "user", "content": "hi"}], max_tokens=120,
            min_tokens=120))
        self.assertEqual(session.payload["min_tokens"], 120)
        self.assertEqual(session.payload["max_tokens"], 120)

    def test_absent_unless_asked(self):
        session = _Session(_sse([_delta("a")]))
        asyncio.run(send_request(
            session=session, url="http://stub/v1/chat/completions", model="m",
            messages=[{"role": "user", "content": "hi"}], max_tokens=120))
        self.assertNotIn("min_tokens", session.payload)

    def test_each_turn_keeps_its_own_length(self):
        """Pinning removes the early stop, not the variation between turns."""
        for planned in (37, 204, 91):
            session = _Session(_sse([_delta("a")]))
            asyncio.run(send_request(
                session=session, url="http://stub/v1/chat/completions", model="m",
                messages=[{"role": "user", "content": "hi"}], max_tokens=planned,
                min_tokens=planned))
            self.assertEqual(session.payload["min_tokens"], planned)


class CaptureTextTests(unittest.TestCase):
    def test_content_deltas_are_joined(self):
        r = _run(_Session(_sse([
            _delta("Hel"), _delta("lo"), _delta(" there"),
            {"choices": [{"delta": {}}], "usage": {"prompt_tokens": 4,
                                                   "completion_tokens": 3}},
        ])))
        self.assertTrue(r.success, msg=r.error)
        self.assertEqual(r.generated_text, "Hello there")

    def test_reasoning_and_tool_calls_are_not_transcript_content(self):
        """Reasoning/tool_calls count as token events but are not transcript content."""
        r = _run(_Session(_sse([
            _delta(None, reasoning_content="thinking"),
            _delta("answer"),
            _delta(None, tool_calls=[{"index": 0}]),
        ])))
        self.assertTrue(r.success, msg=r.error)
        self.assertEqual(r.generated_text, "answer")
        # three payload-bearing chunks: one ttft plus two inter-token gaps
        self.assertEqual(len(r.itl), 2)

    def test_capture_off_returns_none_and_still_measures(self):
        session = _Session(_sse([_delta("a"), _delta("b")]))
        r = asyncio.run(send_request(
            session=session, url="http://stub/v1/chat/completions", model="m",
            messages=[{"role": "user", "content": "hi"}], max_tokens=8,
        ))
        self.assertTrue(r.success, msg=r.error)
        self.assertIsNone(r.generated_text)
        self.assertIsNotNone(r.ttft)

    def test_empty_reply_is_not_an_error(self):
        """Empty capture is a success with empty text, not a failure."""
        r = _run(_Session(_sse([_delta(None, tool_calls=[{"index": 0}])])))
        self.assertTrue(r.success, msg=r.error)
        self.assertEqual(r.generated_text, "")

    def test_non_200_is_a_failure_with_the_status_in_it(self):
        r = _run(_Session([], status=503))
        self.assertFalse(r.success)
        self.assertIn("503", str(r.error))


if __name__ == "__main__":
    unittest.main()
