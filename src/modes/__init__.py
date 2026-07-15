"""
Benchmark modes — defines the three execution modes and their required flags.

stress-test:
    Random tokens, prefix caching OFF, --ignore-eos required.
    Mirrors InferenceX methodology. Tests raw GPU throughput.

single-turn:
    ShareGPT real prompts, prefix caching ON.
    Tests realistic workload performance. Server must be launched
    with --enable-prefix-caching (vLLM) or radix cache (SGLang default).

multi-turn:
    Growing conversation history, prefix caching ON.
    Not yet implemented.
"""

MODES = ["stress-test", "single-turn", "multi-turn"]
