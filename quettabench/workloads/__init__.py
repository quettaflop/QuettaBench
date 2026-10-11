"""Every Trace a run can replay; build(cfg) picks the one a Config names."""

from .base import DoubleWrapRandomTokenTrace, LegacyRandomTokenTrace, RandomTokenTrace
from .grid import GridTrace
from .live import RealTrace
from .profile_trace import ProfileSessionTrace, ProfileTrace
from .profiles import PROFILE_ALIASES, PROFILES, get_profile
from .sessions import TurnPacing, resolve_multi_turn_num_sessions

SESSION_DATASETS = ("sharegpt-multi-turn", "swebench-multi-turn", "terminalbench-multi-turn",
                    "osworld-multi-turn")
RANDOM_TRACES = {"random": RandomTokenTrace, "random-legacy": LegacyRandomTokenTrace,
                 "random-doublewrap": DoubleWrapRandomTokenTrace}


def build(cfg, tokenizer=None):
    """The run's Trace. A grid run (mode.grid) gets the GridTrace of its
    cells; otherwise workload.source is a profile name (configs/workloads/),
    drawn as single-turn or, for kind sessions, as multi-turn sessions, or a
    replay JSONL path. tokenizer is the engine's, called lazily by corpus
    grids."""
    w = cfg.workload
    if cfg.mode.grid is not None:
        return GridTrace.from_config(cfg, tokenizer=tokenizer)
    if w.source not in PROFILES and w.source not in PROFILE_ALIASES:
        return RealTrace.from_jsonl(w.source, seed=w.seed)
    profile = get_profile(w.source)
    if profile.sampler:
        raise ValueError(
            f"profile {profile.name!r} samples synthetic sessions from a trace distribution; "
            "synthetic traces are not a quettabench workload. The sampler lives in "
            "analysis/synthetic/ (distributional.py, trace_distributions.py); replay a real "
            "trace instead (workload.kind=sessions with a trajectory profile, or a jsonl file).")
    dataset_kw = {"random_seed": w.seed, "tokenizer_name": cfg.engine.model or ""}
    if w.kind == "sessions":
        if profile.dataset not in SESSION_DATASETS:
            raise ValueError(f"Profile '{profile.name}' does not use a multi-turn dataset")
        pacing = TurnPacing(w.pacing, w.tool_wait_ms, w.human_wait_ms, w.use_recorded_waits,
                            w.use_recorded_arrivals)
        sessions, _ = resolve_multi_turn_num_sessions(profile, cfg.mode.concurrency)
        return ProfileSessionTrace(profile, warmup=cfg.mode.warmup, pacing=pacing,
                                   max_turn_index=w.max_turn_index, num_sessions=sessions,
                                   **dataset_kw)
    if profile.dataset in RANDOM_TRACES:
        trace = RANDOM_TRACES[profile.dataset](profile.tokenizer_name, profile.isl_tokens,
                                               profile.osl_tokens, seed=w.seed)
        return trace.draw(w.num_requests, cfg.mode.warmup, profile)
    return ProfileTrace(profile, w.num_requests, warmup=cfg.mode.warmup, **dataset_kw)
