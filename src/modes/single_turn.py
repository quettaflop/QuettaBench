"""
Single-turn mode — realistic workload with prefix caching.

Server requirements:
  - Prefix caching: ON
  - vLLM: --enable-prefix-caching
  - SGLang: radix cache is ON by default (no flag needed)

Client requirements:
  - --ignore-eos: NOT recommended (real text hits EOS naturally)
  - Warmup requests are important: first N requests populate the prefix cache

Profiles:
  - chat-singleturn: canonical natural ShareGPT single-turn chat. The historical
    chat-short/chat-medium variants are retired from default sweeps.
  - coding-singleturn: real SWE-bench-style coding single-turn prompts.
  - prefill-heavy/decode-heavy/random-1k: synthetic stress profiles.
"""

REQUIRED_CLIENT_FLAGS: list[str] = []
PREFIX_CACHING_REQUIRED = True

from ..workloads.profiles import filter_profiles
PROFILES = list(filter_profiles(mode="single-turn").keys())

SERVER_NOTES = """
vLLM: pass --enable-prefix-caching
SGLang: radix cache is on by default (no flag needed)
"""
