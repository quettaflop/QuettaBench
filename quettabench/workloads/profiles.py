"""Workload profiles: the registry over configs/workloads/<name>.yaml.

A profile is data. One yaml per profile, named after it; its keys are
WorkloadProfile's fields and an unknown key is an error at import. A profile
with a sampler names synthetic sessions the package does not generate (see
analysis/synthetic/); it keeps its tags for filtering and refuses to build.
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import yaml

WORKLOADS = Path(__file__).resolve().parents[2] / "configs" / "workloads"


@dataclass
class WorkloadProfile:
    name: str
    isl_tokens: int   # for random: exact target ISL; for sharegpt: max ISL filter bound
    osl_tokens: int   # for random: exact target OSL; for sharegpt: max OSL filter bound (also max_tokens)
    isl_stddev: float        # stddev as fraction of isl (for Gaussian sampling)
    description: str
    dataset: str             # "sharegpt", "test", "random", "jsonl", "sharegpt-multi-turn", "swebench-multi-turn", "terminalbench-multi-turn"
    file_path: str = ""      # used when dataset="jsonl" or "distributional-multi-turn"
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
    data_source: str = ""          # "sharegpt" | "swebench" | "terminalbench" | "random" | "test"
    active: bool = True            # False = legacy/runnable, hidden from default sweeps/profile lists
    sampler: str = ""              # set: synthetic sessions generated outside the package


def load_profiles(directory=WORKLOADS):
    """Every <name>.yaml under directory as a WorkloadProfile, by name."""
    keys = {f.name for f in dataclasses.fields(WorkloadProfile)} - {"name"}
    profiles = {}
    for path in sorted(Path(directory).glob("*.yaml")):
        data = yaml.safe_load(path.read_text()) or {}
        unknown = sorted(set(data) - keys)
        if unknown:
            raise ValueError(f"{path}: unknown profile key(s) {unknown}; allowed: {sorted(keys)}")
        try:
            profiles[path.stem] = WorkloadProfile(name=path.stem, **data)
        except TypeError as e:
            raise ValueError(f"{path}: {e}") from None
    return profiles


PROFILES: dict[str, WorkloadProfile] = load_profiles()

# Historical names accepted for archived result ingestion and old command lines.
PROFILE_ALIASES: dict[str, str] = {
    "chat-long": "chat-singleturn",
    "coding-agent": "coding-singleturn",
    "multi-turn-short": "chat-multiturn-short",
}


def filter_profiles(
    agent_type: str | None = None,
    turn_style: str | None = None,
    serving_style: str | None = None,
    data_source: str | None = None,
    mode: str | None = None,
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


def get_profile(name: str) -> WorkloadProfile:
    """Look up a profile by name, resolving aliases for old names."""
    if name in PROFILES:
        return PROFILES[name]
    if name in PROFILE_ALIASES:
        return PROFILES[PROFILE_ALIASES[name]]
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
