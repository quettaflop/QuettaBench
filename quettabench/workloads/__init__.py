"""Every Trace a run can replay; build(cfg) picks the one a Config names."""

from .grid import GridTrace
from .live import RealTrace
from .profile_trace import ProfileSessionTrace, ProfileTrace
from .profiles import PROFILE_ALIASES, PROFILES, get_profile
from .sessions import TurnPacing, resolve_multi_turn_num_sessions


def build(cfg, tokenizer=None):
    """The run's Trace. A grid run (mode.grid) gets the GridTrace of its
    cells; otherwise workload.source is a profile name (profiles.py), drawn
    as single-turn or, for kind sessions, as multi-turn sessions, or a
    replay JSONL path. tokenizer is the engine's, called lazily by corpus
    grids."""
    w = cfg.workload
    if cfg.mode.grid is not None:
        return GridTrace.from_config(cfg, tokenizer=tokenizer)
    if w.source not in PROFILES and w.source not in PROFILE_ALIASES:
        return RealTrace.from_jsonl(w.source, seed=w.seed)
    profile = get_profile(w.source)
    dataset_kw = {"random_seed": w.seed, "tokenizer_name": cfg.engine.model or ""}
    if w.kind == "sessions":
        pacing = TurnPacing(w.pacing, w.tool_wait_ms, w.human_wait_ms, w.use_recorded_waits,
                            w.use_recorded_arrivals)
        sessions, _ = resolve_multi_turn_num_sessions(profile, cfg.mode.concurrency)
        return ProfileSessionTrace(profile, warmup=cfg.mode.warmup, pacing=pacing,
                                   max_turn_index=w.max_turn_index, num_sessions=sessions,
                                   **dataset_kw)
    return ProfileTrace(profile, w.num_requests, warmup=cfg.mode.warmup, **dataset_kw)
