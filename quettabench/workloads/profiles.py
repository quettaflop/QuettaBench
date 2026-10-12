"""Workload profiles: the registry over configs/workloads.yaml.

A profile is data: one entry per name, its keys WorkloadProfile's fields; an unknown key is an error at import.
A profile with active: false is left out of listings (active_profiles, the unknown-name message) but still builds when named.
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import yaml

WORKLOADS = Path(__file__).resolve().parents[2] / "configs" / "workloads.yaml"


@dataclass
class WorkloadProfile:
    name: str
    isl_tokens: int   # for random: exact target ISL; for sharegpt: max ISL filter bound
    osl_tokens: int   # for random: exact target OSL; for sharegpt: max OSL filter bound (also max_tokens)
    isl_stddev: float        # stddev as fraction of isl (for Gaussian sampling)
    description: str
    dataset: str             # "sharegpt", "test", "random", "jsonl", "sharegpt-multi-turn", "swebench-multi-turn", "terminalbench-multi-turn"
    file_path: str = ""      # used when the dataset reads a file
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


def load_profiles(path=WORKLOADS):
    """Every entry of the workloads yaml as a WorkloadProfile, by name."""
    keys = {f.name for f in dataclasses.fields(WorkloadProfile)} - {"name"}
    profiles = {}
    for name, data in yaml.safe_load(Path(path).read_text()).items():
        unknown = sorted(set(data) - keys)
        if unknown:
            raise ValueError(f"{path}: {name}: unknown profile key(s) {unknown}; "
                             f"allowed: {sorted(keys)}")
        try:
            profiles[name] = WorkloadProfile(name=name, **data)
        except TypeError as e:
            raise ValueError(f"{path}: {name}: {e}") from None
    return profiles


PROFILES: dict[str, WorkloadProfile] = load_profiles()

# Historical names accepted for archived result ingestion and old command lines.
PROFILE_ALIASES: dict[str, str] = {
    "chat-long": "chat-singleturn",
    "coding-agent": "coding-singleturn",
    "multi-turn-short": "chat-multiturn-short",
}


def active_profiles() -> list[str]:
    """The names a listing shows: every profile with active true."""
    return sorted(name for name, p in PROFILES.items() if p.active)


def get_profile(name: str) -> WorkloadProfile:
    """Look up a profile by name, resolving aliases for old names."""
    if name in PROFILES:
        return PROFILES[name]
    if name in PROFILE_ALIASES:
        return PROFILES[PROFILE_ALIASES[name]]
    raise ValueError(f"Unknown profile '{name}'. Available: {active_profiles()} "
                     "(inactive profiles are not listed)")
