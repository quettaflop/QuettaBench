"""
Multi-turn mode — growing conversation history with prefix caching.

Uses ShareGPT pre-recorded replies to build deterministic growing-history
request sequences (Option B design — see .claude/docs/multi_turn_design.md).

Design:
  - ShareGPTMultiTurnDataset: extracts full conversations, builds per-session
    request sequences with growing message history
  - Interleaved round-robin scheduling: [A1,B1,C1,A2,B2,C2,...] forces KV
    cache eviction between turns, testing prefix cache reuse under memory pressure
  - Per-turn metrics: TTFT by turn number shows prefix cache effectiveness
    (turn 2+ should have lower TTFT due to shared prefix)

Server requirements (same as single-turn):
  - vLLM: --enable-prefix-caching
  - SGLang: radix cache ON by default
"""

REQUIRED_CLIENT_FLAGS: list[str] = []
PREFIX_CACHING_REQUIRED = True

from ..workloads.profiles import filter_profiles
PROFILES = list(filter_profiles(turn_style="multi-turn").keys())

SERVER_NOTES = """
vLLM: pass --enable-prefix-caching
SGLang: radix cache is on by default (no flag needed)
"""
