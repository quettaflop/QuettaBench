"""
Multi-turn mode — growing conversation history with prefix caching.

Later turns send the engine's own reply back (shared assistant dicts). Trace
replay keeps recorded assistant text. Default scheduling is per-session (turns
back to back, no cross-session barrier). --turn-pacing interleaved barriers
each turn across sessions (turn-aligned herd; not production traffic).

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
