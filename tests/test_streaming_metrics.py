"""Tests for the reasoning-channel streaming fix.

gpt-oss on vLLM streams analysis-channel tokens as `reasoning_content` /
`reasoning` deltas with content=None. The client used to skip those chunks,
collapsing whole reasoning phases into single ITL gaps, and mean(ITL) then
inflated tpot by up to ~13x (QuettaSim tools/GT_QUALITY_FLAGS.md, Finding 1).

Covers both halves of the fix:
  1. engines/openai_chat.py counts reasoning and tool-call deltas as token
     events (verified against a local SSE server).
  2. metrics.RequestResult.tpot falls back to the wall-clock formula when ITL
     chunk coverage is incomplete, and is byte-identical to the old
     definition when coverage is healthy.
"""

import asyncio
import json
import unittest

# aiohttp is a runtime dependency of the client (requirements.txt) but, like
# runner.py, we import it lazily so metric-only tests run without it.
try:
    import aiohttp
    from aiohttp import web
    HAVE_AIOHTTP = True
except ModuleNotFoundError:
    HAVE_AIOHTTP = False

from src.benchmark.metrics import RequestResult


class TestTpotCoverageGuard(unittest.TestCase):
    def test_full_coverage_keeps_mean_itl(self):
        # Healthy stream: one ITL entry per token after the first.
        r = RequestResult(success=True, ttft=0.05, e2el=1.05,
                          itl=[0.01] * 99, output_tokens=100)
        self.assertAlmostEqual(r.tpot, 0.01)

    def test_sparse_coverage_falls_back_to_wall_clock(self):
        # gpt-oss-on-vllm signature: 54 chunks for 269 tokens, one giant gap.
        itl = [2.505] + [0.0104] * 53
        r = RequestResult(success=True, ttft=0.25, e2el=3.32,
                          itl=itl, output_tokens=269)
        # Old definition: mean(itl) ~ 57 ms/token. Corrected: ~11.5 ms/token.
        self.assertAlmostEqual(r.tpot, (3.32 - 0.25) / 268)
        self.assertLess(r.tpot, 0.013)

    def test_short_outputs_keep_mean_itl(self):
        # len(itl) == output_tokens - 1 exactly: correct, must not trip guard.
        r = RequestResult(success=True, ttft=0.05, e2el=0.10,
                          itl=[0.011, 0.012], output_tokens=3)
        self.assertAlmostEqual(r.tpot, (0.011 + 0.012) / 2)

    def test_no_itl_falls_back(self):
        r = RequestResult(success=True, ttft=0.4, e2el=2.4,
                          itl=[], output_tokens=201)
        self.assertAlmostEqual(r.tpot, 2.0 / 200)

    def test_no_itl_no_ttft_is_none(self):
        r = RequestResult(success=True, e2el=2.4, output_tokens=201)
        self.assertIsNone(r.tpot)

    def test_sparse_coverage_without_wall_clock_keeps_mean(self):
        # Guard cannot fire without e2el/ttft; degrade gracefully to mean.
        r = RequestResult(success=True, itl=[0.5, 0.01], output_tokens=100)
        self.assertAlmostEqual(r.tpot, 0.255)


def _sse(payload) -> bytes:
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


def _chunk(delta, finish_reason=None):
    return {"choices": [{"index": 0, "delta": delta,
                         "finish_reason": finish_reason}]}


async def _serve_and_request(chunks):
    """Stream the given SSE chunk payloads from a local server through
    send_request and return the RequestResult."""
    from src.engines.openai_chat import send_request

    async def handler(request):
        resp = web.StreamResponse(
            headers={"Content-Type": "text/event-stream"})
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
                model="m", messages=[{"role": "user", "content": "hi"}],
                max_tokens=16,
            )
    finally:
        await runner.cleanup()


@unittest.skipUnless(HAVE_AIOHTTP, "aiohttp not installed")
class TestReasoningChunksCounted(unittest.TestCase):
    def test_reasoning_content_chunks_are_token_events(self):
        # vllm 0.19 gpt-oss harmony shape: role chunk, analysis-channel
        # deltas (content absent), then final-channel content deltas.
        chunks = (
            [_chunk({"role": "assistant", "content": ""})]
            + [_chunk({"reasoning_content": "r"}) for _ in range(3)]
            + [_chunk({"content": "t"}) for _ in range(4)]
            + [_chunk({}, finish_reason="stop")]        # must NOT count
            + [{"choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 8}}]
        )
        r = asyncio.run(_serve_and_request(chunks))
        self.assertTrue(r.success)
        # 8 token events (1 role + 3 reasoning + 4 content): ttft + 7 ITLs.
        # Before the fix the 3 reasoning chunks were dropped (4 ITLs).
        self.assertIsNotNone(r.ttft)
        self.assertEqual(len(r.itl), 7)
        self.assertEqual(r.output_tokens, 8)

    def test_newer_reasoning_field_and_tool_calls_count(self):
        chunks = (
            [_chunk({"role": "assistant", "content": ""})]
            + [_chunk({"reasoning": "r"}) for _ in range(2)]
            + [_chunk({"tool_calls": [{"index": 0, "function":
                                       {"arguments": "{"}}]}) for _ in range(2)]
            + [_chunk({"content": "t"})]
        )
        r = asyncio.run(_serve_and_request(chunks))
        self.assertTrue(r.success)
        self.assertEqual(len(r.itl), 5)

    def test_plain_content_stream_unchanged(self):
        # Clean backends (Llama, Qwen, sglang) must behave exactly as before.
        chunks = (
            [_chunk({"role": "assistant", "content": ""})]
            + [_chunk({"content": "t"}) for _ in range(5)]
            + [_chunk({}, finish_reason="stop")]
        )
        r = asyncio.run(_serve_and_request(chunks))
        self.assertTrue(r.success)
        self.assertEqual(len(r.itl), 5)


if __name__ == "__main__":
    unittest.main()
