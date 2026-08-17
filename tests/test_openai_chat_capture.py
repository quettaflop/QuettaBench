"""`send_request` against a stubbed SSE stream, so the streaming loop is exercised
without a server.

This file exists because it was missing. `capture_text` shipped with its
accumulator never initialised: every request raised `NameError` inside the
streaming loop, `send_request`'s blanket `except Exception` turned that into
`success=False`, and the runner reported "Server may not be functional" — on a
server that was working. The sweep produced ten failed cells and pointed at the
wrong half of the system.

Anything in that loop is now reachable from a unit test.
"""

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
        """They count as tokens — dropping them coalesces ITL gaps — but they are
        not what a chat client sends back, so they must not enter the reply."""
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
        """A model may legitimately return nothing. The runner leaves the planned
        text in place rather than writing an empty string through the transcript,
        so this has to arrive as empty rather than as a failure."""
        r = _run(_Session(_sse([_delta(None, tool_calls=[{"index": 0}])])))
        self.assertTrue(r.success, msg=r.error)
        self.assertEqual(r.generated_text, "")

    def test_non_200_is_a_failure_with_the_status_in_it(self):
        r = _run(_Session([], status=503))
        self.assertFalse(r.success)
        self.assertIn("503", str(r.error))


if __name__ == "__main__":
    unittest.main()
