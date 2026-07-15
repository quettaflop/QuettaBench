"""
Backward-compatible re-exports.

New code should import directly from src.engines.openai_chat or src.engines.
"""

from ..benchmark.metrics import RequestResult
from ..engines.openai_chat import send_request as send_chat_request, run_warmup

__all__ = ["RequestResult", "send_chat_request", "run_warmup"]
