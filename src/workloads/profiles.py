"""
Workload profile definitions.

Profiles are organized into groups:
  Group 1: Real agent data (SWEBench PLLM, SWEBench trajectories, TerminalBench trajectories)
  Group 2: Chat — single-turn and multi-turn (ShareGPT)
  Group 3: Synthetic stress tests (random tokens, file-based)

Each profile defines the data source, ISL/OSL bounds, and metadata tags.
"""

from dataclasses import dataclass
from typing import Optional


# Valid tag values
AGENT_TYPES = ["chat", "coding", "terminal", "computer-use", "stress"]
TURN_STYLES = ["single-turn", "multi-turn"]
SERVING_STYLES = ["disaggregated", "not-disaggregated"]
DATA_SOURCES = [
    "sharegpt",
    "swebench",
    "terminalbench",
    "osworld",
    "distributional",
    "file",
    "random",
    "test",
]


@dataclass
class WorkloadProfile:
    name: str
    isl_tokens: int   # for random: exact target ISL; for sharegpt: max ISL filter bound
    osl_tokens: int   # for random: exact target OSL; for sharegpt: max OSL filter bound (also max_tokens)
    isl_stddev: float        # stddev as fraction of isl (for Gaussian sampling)
    description: str
    dataset: str             # "sharegpt", "file", "test", "random", "jsonl", "sharegpt-multi-turn", "swebench-multi-turn", "terminalbench-multi-turn"
    file_path: str = ""      # used when dataset="file" or "jsonl"
    system_prompt: str = "You are a helpful assistant."
    tokenizer_name: str = "" # used when dataset="random"
    mode: str = "single-turn"           # "stress-test" | "single-turn" | "multi-turn"
    prefix_caching_required: bool = False  # True = server must be launched with --enable-prefix-caching
    min_turns: int = 1                   # multi-turn: minimum turns per session
    max_turns: int = 1                   # multi-turn: maximum turns per session
    num_sessions: int = 200              # multi-turn: number of concurrent sessions
    agent_type: str = ""           # "chat" | "coding" | "terminal"
    turn_style: str = "single-turn"  # "single-turn" | "multi-turn"
    serving_style: str = "not-disaggregated"  # "disaggregated" | "not-disaggregated"
    data_source: str = ""          # "sharegpt" | "swebench" | "terminalbench" | "file" | "random" | "test"
    active: bool = True            # False = legacy/runnable, hidden from default sweeps/profile lists


# ---------------------------------------------------------------------------
# Profile registry
# ---------------------------------------------------------------------------

PROFILES: dict[str, WorkloadProfile] = {

    # ===================================================================
    # Group 1: Real Agent Data
    # ===================================================================

    # Single-turn planning-model call — real SWE-Bench prompts. The JSONL
    # drives actual token lengths; isl_tokens/osl_tokens are metadata caps.
    "coding-singleturn": WorkloadProfile(
        name="coding-singleturn",
        isl_tokens=17000,
        osl_tokens=800,
        isl_stddev=0.0,
        description="Real coding single-turn prompts captured from SWE-Bench-style planning calls",
        dataset="jsonl",
        file_path="data/coding_agent_prompts.jsonl",
        system_prompt="",  # system prompt is embedded in the JSONL
        mode="single-turn",
        prefix_caching_required=True,
        agent_type="coding",
        turn_style="single-turn",
        serving_style="not-disaggregated",
        data_source="swebench",
    ),

    # Canonical distributional multi-turn profiles.
    "swebench-multiturn": WorkloadProfile(
        name="swebench-multiturn",
        isl_tokens=131072,
        osl_tokens=2000,
        isl_stddev=0.0,
        description="Distributional synthetic SWE-bench agent sessions sampled from empirical turn/input/output traces",
        dataset="distributional-multi-turn",
        file_path="data/distributions/swebench_multiturn.json",
        system_prompt="",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=1,
    max_turns=320,
    num_sessions=1,
        agent_type="coding",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="distributional",
    ),

    "terminalbench-multiturn": WorkloadProfile(
        name="terminalbench-multiturn",
        isl_tokens=131072,
        osl_tokens=2000,
        isl_stddev=0.0,
        description="Distributional synthetic TerminalBench CLI-agent sessions sampled from empirical turn/input/output traces",
        dataset="distributional-multi-turn",
        file_path="data/distributions/terminalbench_multiturn.json",
        system_prompt="",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=1,
    max_turns=876,
    num_sessions=1,
        agent_type="terminal",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="distributional",
    ),

    "osworld-multiturn": WorkloadProfile(
        name="osworld-multiturn",
        isl_tokens=65536,
        osl_tokens=500,
        isl_stddev=0.0,
        description="Distributional synthetic OSWorld computer-use sessions sampled from empirical turn/input/output traces",
        dataset="distributional-multi-turn",
        file_path="data/distributions/osworld_multiturn.json",
        system_prompt="",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=1,
    max_turns=30,
    num_sessions=1,
        agent_type="computer-use",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="distributional",
    ),

    # Synthetic-scope profiles. These intentionally use separate profile names
    # so dashboard scope, result scope, and coverage state cannot be confused
    # with canonical fixed/current/archive data.
    "swebench-multiturn-synth": WorkloadProfile(
        name="swebench-multiturn-synth",
        isl_tokens=131072,
        osl_tokens=2000,
        isl_stddev=0.0,
        description="APC-aware synthetic SWE-bench agent sessions, validated short-trace regime",
        dataset="distributional-multi-turn",
        file_path="data/distributions/swebench_multiturn_short_tracereplay_filtered-mse.json",
        system_prompt="",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=1,
        max_turns=30,
        num_sessions=1,
        agent_type="coding",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="distributional",
    ),

    "terminalbench-multiturn-synth": WorkloadProfile(
        name="terminalbench-multiturn-synth",
        isl_tokens=131072,
        osl_tokens=2000,
        isl_stddev=0.0,
        description="APC-aware synthetic TerminalBench CLI-agent sessions, validated short-trace regime",
        dataset="distributional-multi-turn",
        file_path="data/distributions/terminalbench_multiturn_short_tracereplay_filtered-mse.json",
        system_prompt="",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=1,
        max_turns=30,
        num_sessions=1,
        agent_type="terminal",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="distributional",
    ),

    "osworld-multiturn-synth": WorkloadProfile(
        name="osworld-multiturn-synth",
        isl_tokens=65536,
        osl_tokens=500,
        isl_stddev=0.0,
        description="APC-aware synthetic OSWorld computer-use sessions sampled from empirical turn/input/output traces",
        dataset="distributional-multi-turn",
        file_path="data/distributions/osworld_multiturn.json",
        system_prompt="",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=1,
        max_turns=30,
        num_sessions=1,
        agent_type="computer-use",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="distributional",
    ),

    # Multi-turn SWEBench coding agent — real trajectories from harbor/jobs/
    # Note: "turns" here are agent steps (tool calls), not logical conversation rounds.
    # SWEBench sessions have min=13, median=85, max=320 steps.
    # Data uses compressed trajectory.json (summary messages ~100 chars each).
    # TODO: Extract full rollout JSONL for realistic ISL/OSL per step.
    "swebench-multiturn-short": WorkloadProfile(
        name="swebench-multiturn-short",
        isl_tokens=32768,
        osl_tokens=2000,
        isl_stddev=0.0,
        description="Real SWEBench coding agent: 13-30 step sessions (shortest available)",
        dataset="swebench-multi-turn",
        file_path="data/swebench_trajectories.jsonl",
        system_prompt="",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=13,
        max_turns=30,
        num_sessions=100,
        agent_type="coding",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="swebench",
        active=False,
    ),
    "swebench-multiturn-medium": WorkloadProfile(
        name="swebench-multiturn-medium",
        isl_tokens=65536,
        osl_tokens=2000,
        isl_stddev=0.0,
        description="Real SWEBench coding agent: 30-80 step sessions",
        dataset="swebench-multi-turn",
        file_path="data/swebench_trajectories.jsonl",
        system_prompt="",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=30,
        max_turns=80,
        num_sessions=100,
        agent_type="coding",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="swebench",
        active=False,
    ),
    "swebench-multiturn-long": WorkloadProfile(
        name="swebench-multiturn-long",
        isl_tokens=131072,
        osl_tokens=2000,
        isl_stddev=0.0,
        description="Real SWEBench coding agent: 80-150 step sessions",
        dataset="swebench-multi-turn",
        file_path="data/swebench_trajectories.jsonl",
        system_prompt="",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=80,
        max_turns=150,
        num_sessions=50,
        agent_type="coding",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="swebench",
        active=False,
    ),
    # Multi-turn TerminalBench CLI agent — real trajectories from harbor/jobs/
    # Note: "turns" here are agent steps (tool calls), not logical conversation rounds.
    # TerminalBench sessions have min=2, median=61, max=876 steps.
    # Data uses compressed trajectory.json (summary messages ~100 chars each).
    # TODO: Extract full rollout JSONL for realistic ISL/OSL per step.
    "terminalbench-multiturn-short": WorkloadProfile(
        name="terminalbench-multiturn-short",
        isl_tokens=32768,
        osl_tokens=2000,
        isl_stddev=0.0,
        description="Real TerminalBench CLI agent: 2-20 step sessions (shortest available)",
        dataset="terminalbench-multi-turn",
        file_path="data/terminalbench_trajectories.jsonl",
        system_prompt="",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=2,
        max_turns=20,
        num_sessions=100,
        agent_type="terminal",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="terminalbench",
        active=False,
    ),
    "terminalbench-multiturn-medium": WorkloadProfile(
        name="terminalbench-multiturn-medium",
        isl_tokens=65536,
        osl_tokens=2000,
        isl_stddev=0.0,
        description="Real TerminalBench CLI agent: 20-60 step sessions",
        dataset="terminalbench-multi-turn",
        file_path="data/terminalbench_trajectories.jsonl",
        system_prompt="",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=20,
        max_turns=60,
        num_sessions=100,
        agent_type="terminal",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="terminalbench",
        active=False,
    ),
    "terminalbench-multiturn-long": WorkloadProfile(
        name="terminalbench-multiturn-long",
        isl_tokens=131072,
        osl_tokens=2000,
        isl_stddev=0.0,
        description="Real TerminalBench CLI agent: 60-150 step sessions",
        dataset="terminalbench-multi-turn",
        file_path="data/terminalbench_trajectories.jsonl",
        system_prompt="",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=60,
        max_turns=150,
        num_sessions=50,
        agent_type="terminal",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="terminalbench",
        active=False,
    ),
    # Multi-turn OSWorld computer-use agent — real WebArena trajectories
    # Note: "turns" here are agent steps (browser actions).
    # OSWorld sessions have min=1, median=8, max=30 steps.
    # ISL/OSL ratio ~120:1 (massive DOM context, tiny action output).
    "osworld-multiturn-short": WorkloadProfile(
        name="osworld-multiturn-short",
        isl_tokens=32768,
        osl_tokens=500,
        isl_stddev=0.0,
        description="Real OSWorld computer-use agent: 2-10 step sessions (short browsing tasks)",
        dataset="osworld-multi-turn",
        file_path="data/osworld_trajectories.jsonl",
        system_prompt="",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=2,
        max_turns=10,
        num_sessions=50,
        agent_type="computer-use",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="osworld",
        active=False,
    ),
    "osworld-multiturn-medium": WorkloadProfile(
        name="osworld-multiturn-medium",
        isl_tokens=65536,
        osl_tokens=500,
        isl_stddev=0.0,
        description="Real OSWorld computer-use agent: 10-20 step sessions",
        dataset="osworld-multi-turn",
        file_path="data/osworld_trajectories.jsonl",
        system_prompt="",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=10,
        max_turns=20,
        num_sessions=30,
        agent_type="computer-use",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="osworld",
        active=False,
    ),
    "osworld-multiturn-long": WorkloadProfile(
        name="osworld-multiturn-long",
        isl_tokens=131072,
        osl_tokens=500,
        isl_stddev=0.0,
        description="Real OSWorld computer-use agent: 20-30 step sessions (longest available)",
        dataset="osworld-multi-turn",
        file_path="data/osworld_trajectories.jsonl",
        system_prompt="",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=20,
        max_turns=30,
        num_sessions=20,
        agent_type="computer-use",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="osworld",
        active=False,
    ),

    # ===================================================================
    # Group 2: Chat — ShareGPT (single-turn and multi-turn)
    # ===================================================================

    # --- Single-turn ---

    "chat-short": WorkloadProfile(
        name="chat-short",
        isl_tokens=500,
        osl_tokens=300,
        isl_stddev=0.15,
        description="Legacy ShareGPT single-turn chat, short-answer bucket (retired from default sweeps)",
        dataset="sharegpt",
        mode="single-turn",
        prefix_caching_required=True,
        agent_type="chat",
        turn_style="single-turn",
        serving_style="not-disaggregated",
        data_source="sharegpt",
        active=False,
    ),
    "chat-medium": WorkloadProfile(
        name="chat-medium",
        isl_tokens=2000,
        osl_tokens=1000,
        isl_stddev=0.15,
        description="Legacy ShareGPT single-turn chat, medium-answer bucket (retired from default sweeps)",
        dataset="sharegpt",
        mode="single-turn",
        prefix_caching_required=True,
        agent_type="chat",
        turn_style="single-turn",
        serving_style="not-disaggregated",
        data_source="sharegpt",
        active=False,
    ),
    "chat-singleturn": WorkloadProfile(
        name="chat-singleturn",
        # Runtime caps are deliberately natural-chat sized so this profile is
        # safe for 4K-context sweep cells.
        isl_tokens=2048,
        osl_tokens=1024,
        isl_stddev=0.15,
        description="Canonical natural ShareGPT single-turn chat (not a long-context prefill stress workload)",
        dataset="sharegpt",
        mode="single-turn",
        prefix_caching_required=True,
        agent_type="chat",
        turn_style="single-turn",
        serving_style="not-disaggregated",
        data_source="sharegpt",
    ),
    "chat-singleturn-synth": WorkloadProfile(
        name="chat-singleturn-synth",
        isl_tokens=2048,
        osl_tokens=1024,
        isl_stddev=0.15,
        description="Synthetic-scope natural ShareGPT single-turn chat baseline",
        dataset="sharegpt",
        mode="single-turn",
        prefix_caching_required=True,
        agent_type="chat",
        turn_style="single-turn",
        serving_style="not-disaggregated",
        data_source="sharegpt",
    ),

    # --- Multi-turn ---

    "chat-multiturn": WorkloadProfile(
        name="chat-multiturn",
        isl_tokens=32768,
        osl_tokens=2000,
        isl_stddev=0.0,
        description="Distributional synthetic ShareGPT multi-turn chat sampled from empirical turn/input/output summaries",
        dataset="distributional-multi-turn",
        file_path="data/distributions/chat_multiturn.json",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=1,
        max_turns=20,
        num_sessions=1,
        agent_type="chat",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="distributional",
    ),
    "chat-multiturn-synth": WorkloadProfile(
        name="chat-multiturn-synth",
        isl_tokens=32768,
        osl_tokens=2000,
        isl_stddev=0.0,
        description="APC-aware synthetic ShareGPT multi-turn chat sampled from empirical turn/input/output summaries",
        dataset="distributional-multi-turn",
        file_path="data/distributions/chat_multiturn.json",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=1,
        max_turns=20,
        num_sessions=1,
        agent_type="chat",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="distributional",
    ),

    "chat-multiturn-short": WorkloadProfile(
        name="chat-multiturn-short",
        isl_tokens=8192,
        osl_tokens=1000,
        isl_stddev=0.0,
        description="Natural ShareGPT multi-turn chat: 3-5 turns. Short/medium/long denote turn depth, not monotonic ISL or OSL.",
        dataset="sharegpt-multi-turn",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=3,
        max_turns=5,
        num_sessions=200,
        agent_type="chat",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="sharegpt",
        active=False,
    ),
    "chat-multiturn-medium": WorkloadProfile(
        name="chat-multiturn-medium",
        isl_tokens=16384,
        osl_tokens=1500,
        isl_stddev=0.0,
        description="Natural ShareGPT multi-turn chat: 5-10 turns. Short/medium/long denote turn depth, not monotonic ISL or OSL.",
        dataset="sharegpt-multi-turn",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=5,
        max_turns=10,
        num_sessions=100,
        agent_type="chat",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="sharegpt",
        active=False,
    ),
    "chat-multiturn-long": WorkloadProfile(
        name="chat-multiturn-long",
        isl_tokens=32768,
        osl_tokens=2000,
        isl_stddev=0.0,
        description="Natural ShareGPT multi-turn chat: 10-20 turns. Short/medium/long denote turn depth, not monotonic ISL or OSL.",
        dataset="sharegpt-multi-turn",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=10,
        max_turns=20,
        num_sessions=50,
        agent_type="chat",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="sharegpt",
        active=False,
    ),
    # ===================================================================
    # Group 3: Synthetic Stress Tests
    # ===================================================================

    "prefill-heavy": WorkloadProfile(
        name="prefill-heavy",
        isl_tokens=8192,
        osl_tokens=256,
        isl_stddev=0.0,
        description="Synthetic prefill stress: long input, short output (ISL=8192, OSL=256)",
        dataset="random",
        tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
        mode="stress-test",
        prefix_caching_required=False,
        agent_type="stress",
        turn_style="single-turn",
        serving_style="not-disaggregated",
        data_source="random",
        active=False,
    ),
    "decode-heavy": WorkloadProfile(
        name="decode-heavy",
        isl_tokens=256,
        osl_tokens=4096,
        isl_stddev=0.0,
        description="Synthetic decode stress: short input, long output (ISL=256, OSL=4096)",
        dataset="random",
        tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
        mode="stress-test",
        prefix_caching_required=False,
        agent_type="stress",
        turn_style="single-turn",
        serving_style="not-disaggregated",
        data_source="random",
        active=False,
    ),
    "random-1k": WorkloadProfile(
        name="random-1k",
        isl_tokens=1024,
        osl_tokens=1024,
        isl_stddev=0.0,
        description="InferenceX cross-validation: random tokens ISL=1024 OSL=1024",
        dataset="random",
        tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
        mode="stress-test",
        prefix_caching_required=False,
        agent_type="stress",
        turn_style="single-turn",
        serving_style="not-disaggregated",
        data_source="random",
        active=False,
    ),

    # Fixed-shape profiles for per-kernel/per-op predictor validation.
    # Matches the ncu sweep (prefill_seq128_bs1) so --vs-measured can be
    # compared apples-to-apples at a known seq. Stress-test mode keeps
    # prefix caching OFF and uses random tokens to guarantee each request
    # actually prefills seq=128 (ShareGPT-derived profiles vary per request
    # and hit prefix cache from turn 2 onward).
    "fixed-seq128": WorkloadProfile(
        name="fixed-seq128",
        isl_tokens=128,
        osl_tokens=128,
        isl_stddev=0.0,
        description="Fixed ISL=128 OSL=128 random tokens — matches ncu prefill_seq128_bs1 for predictor validation.",
        dataset="random",
        tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
        mode="stress-test",
        prefix_caching_required=False,
        agent_type="stress",
        turn_style="single-turn",
        serving_style="not-disaggregated",
        data_source="random",
        active=False,
    ),

    # ===================================================================
    # Group 4: Utility
    # ===================================================================

    "test": WorkloadProfile(
        name="test",
        isl_tokens=10,
        osl_tokens=20,
        isl_stddev=0.0,
        description="Quick smoke test",
        dataset="test",
        mode="single-turn",
        prefix_caching_required=False,
        agent_type="chat",
        turn_style="single-turn",
        serving_style="not-disaggregated",
        data_source="test",
        active=False,
    ),

    # === MSE-validation profiles (active=False) ===
    # Filtered distributions matched to legacy ISL=32768 filter semantics.
    # Use these for head-to-head comparison against legacy _short profiles.
    "swebench-multiturn-mse": WorkloadProfile(
        name="swebench-multiturn-mse",
        isl_tokens=32768, osl_tokens=2000, isl_stddev=0.0,
        description="MSE: filtered swebench distribution, ISL=32768 (matches legacy -short)",
        dataset="distributional-multi-turn",
        file_path="data/distributions/swebench_multiturn_filtered.json",
        mode="multi-turn", prefix_caching_required=True,
        min_turns=1, max_turns=30, num_sessions=100,
        agent_type="coding", turn_style="multi-turn",
        serving_style="not-disaggregated", data_source="distributional",
        active=False,
    ),
    "terminalbench-multiturn-mse": WorkloadProfile(
        name="terminalbench-multiturn-mse",
        isl_tokens=32768, osl_tokens=2000, isl_stddev=0.0,
        description="MSE: filtered terminalbench distribution, ISL=32768 (matches legacy -short)",
        dataset="distributional-multi-turn",
        file_path="data/distributions/terminalbench_multiturn_filtered.json",
        mode="multi-turn", prefix_caching_required=True,
        min_turns=1, max_turns=30, num_sessions=100,
        agent_type="terminal", turn_style="multi-turn",
        serving_style="not-disaggregated", data_source="distributional",
        active=False,
    ),
    "osworld-multiturn-mse": WorkloadProfile(
        name="osworld-multiturn-mse",
        isl_tokens=32768, osl_tokens=500, isl_stddev=0.0,
        description="MSE: filtered osworld distribution, ISL=32768 (matches legacy -short)",
        dataset="distributional-multi-turn",
        file_path="data/distributions/osworld_multiturn_filtered.json",
        mode="multi-turn", prefix_caching_required=True,
        min_turns=1, max_turns=30, num_sessions=50,
        agent_type="computer-use", turn_style="multi-turn",
        serving_style="not-disaggregated", data_source="distributional",
        active=False,
    ),

    # === MSE-validation profiles — bucketed by turn-count range (2026-05-05) ===
    # These match the REAL trace_replay profiles' turn-count ranges:
    #   -short: 13-30 turns (swebench), 2-30 (terminalbench)
    #   -medium: 50-125 turns
    # Use for head-to-head per-turn validation against archived trace_replay runs.
    "swebench-multiturn-mse-short": WorkloadProfile(
        name="swebench-multiturn-mse-short",
        isl_tokens=32768, osl_tokens=2000, isl_stddev=0.0,
        description="MSE (short): swebench distribution bucketed 13-30 turns, ISL≤32K",
        dataset="distributional-multi-turn",
        file_path="data/distributions/swebench_multiturn_short_tracereplay_filtered-mse.json",
        mode="multi-turn", prefix_caching_required=True,
        min_turns=13, max_turns=30, num_sessions=100,
        agent_type="coding", turn_style="multi-turn",
        serving_style="not-disaggregated", data_source="distributional",
        active=False,
    ),
    "swebench-multiturn-mse-medium": WorkloadProfile(
        name="swebench-multiturn-mse-medium",
        isl_tokens=32768, osl_tokens=2000, isl_stddev=0.0,
        description="MSE (medium): swebench distribution bucketed 50-125 turns, ISL≤32K",
        dataset="distributional-multi-turn",
        file_path="data/distributions/swebench_multiturn_medium_tracereplay_filtered-mse.json",
        mode="multi-turn", prefix_caching_required=True,
        min_turns=50, max_turns=125, num_sessions=100,
        agent_type="coding", turn_style="multi-turn",
        serving_style="not-disaggregated", data_source="distributional",
        active=False,
    ),
    "terminalbench-multiturn-mse-short": WorkloadProfile(
        name="terminalbench-multiturn-mse-short",
        isl_tokens=32768, osl_tokens=2000, isl_stddev=0.0,
        description="MSE (short): terminalbench distribution bucketed 2-30 turns, ISL≤32K",
        dataset="distributional-multi-turn",
        file_path="data/distributions/terminalbench_multiturn_short_tracereplay_filtered-mse.json",
        mode="multi-turn", prefix_caching_required=True,
        min_turns=2, max_turns=30, num_sessions=100,
        agent_type="terminal", turn_style="multi-turn",
        serving_style="not-disaggregated", data_source="distributional",
        active=False,
    ),
    "terminalbench-multiturn-mse-medium": WorkloadProfile(
        name="terminalbench-multiturn-mse-medium",
        isl_tokens=32768, osl_tokens=2000, isl_stddev=0.0,
        description="MSE (medium): terminalbench distribution bucketed 50-125 turns, ISL≤32K",
        dataset="distributional-multi-turn",
        file_path="data/distributions/terminalbench_multiturn_medium_tracereplay_filtered-mse.json",
        mode="multi-turn", prefix_caching_required=True,
        min_turns=50, max_turns=125, num_sessions=100,
        agent_type="terminal", turn_style="multi-turn",
        serving_style="not-disaggregated", data_source="distributional",
        active=False,
    ),
}


# ---------------------------------------------------------------------------
# Historical profile names accepted for archived result ingestion and a narrow
# CLI compatibility path. New scripts should use canonical profile names.
# ---------------------------------------------------------------------------

PROFILE_ALIASES: dict[str, str] = {
    "chat-long": "chat-singleturn",
    "coding-agent": "coding-singleturn",
    "multi-turn-short": "chat-multiturn-short",
}


# ---------------------------------------------------------------------------
# Filtering and lookup
# ---------------------------------------------------------------------------

def filter_profiles(
    agent_type: Optional[str] = None,
    turn_style: Optional[str] = None,
    serving_style: Optional[str] = None,
    data_source: Optional[str] = None,
    mode: Optional[str] = None,
    include_inactive: bool = False,
) -> dict:
    """Filter profiles by tag values. None means 'any'."""
    result = {}
    for name, p in PROFILES.items():
        if not include_inactive and not p.active:
            continue
        if agent_type is not None and p.agent_type != agent_type:
            continue
        if turn_style is not None and p.turn_style != turn_style:
            continue
        if serving_style is not None and p.serving_style != serving_style:
            continue
        if data_source is not None and p.data_source != data_source:
            continue
        if mode is not None and p.mode != mode:
            continue
        result[name] = p
    return result


# Convenience filters
STRESS_TEST_PROFILES = filter_profiles(mode="stress-test")
SINGLE_TURN_PROFILES = filter_profiles(turn_style="single-turn")
MULTI_TURN_PROFILES = filter_profiles(turn_style="multi-turn")
REAL_DATA_PROFILES = {
    k: v for k, v in PROFILES.items()
    if v.data_source in ("swebench", "terminalbench")
}


def get_profile(name: str) -> WorkloadProfile:
    """Look up a profile by name, resolving aliases for old names."""
    if name in PROFILES:
        return PROFILES[name]
    if name in PROFILE_ALIASES:
        resolved = PROFILE_ALIASES[name]
        return PROFILES[resolved]
    raise ValueError(
        f"Unknown profile '{name}'. Available: {sorted(PROFILES.keys())}"
    )


def resolve_profile_name(name: str) -> str:
    """Return the canonical profile name, resolving aliases."""
    if name in PROFILES:
        return name
    if name in PROFILE_ALIASES:
        return PROFILE_ALIASES[name]
    return name
