"""
Backward-compatible re-exports.

New code should import directly from quettabench.engines.openai_chat or quettabench.engines.
"""

from ..engines.openai_chat import send_request as send_chat_request
from ..metrics.core import RequestResult

__all__ = ["RequestResult", "send_chat_request"]
