"""Every Trace a run can replay; build(cfg) picks the one a Config names."""

from .base import DoubleWrapRandomTokenTrace, LegacyRandomTokenTrace, RandomTokenTrace
from .grid import GridTrace
from .live import MooncakeTrace, PromptJsonlTrace, RealTrace, ShareGPTTrace, SmokeTrace
from .profile_trace import ProfileSessionTrace
from .profiles import PROFILE_ALIASES, PROFILES, get_profile
from .sessions import TurnPacing, resolve_multi_turn_num_sessions

TRAJECTORY_DATASETS = ("swebench-multi-turn", "terminalbench-multi-turn", "osworld-multi-turn")
RANDOM_TRACES = {"random": RandomTokenTrace, "random-legacy": LegacyRandomTokenTrace,
                 "random-doublewrap": DoubleWrapRandomTokenTrace}

# Room build leaves between a trajectory turn's prompt plus output and the
# profile's isl_tokens context window.
CONTEXT_SAFETY_MARGIN = 256


def single_turn_trace(profile, seed, model):
    """The undrawn single-turn trace a profile's dataset names."""
    d = profile.dataset
    if d == "test":
        return SmokeTrace()
    if d == "mooncake-trace":
        return MooncakeTrace(profile.file_path, seed=seed)
    if d == "jsonl":
        return PromptJsonlTrace(profile.file_path, seed=seed)
    if d == "sharegpt":
        return ShareGPTTrace(num_prompts=1000, seed=seed, system_prompt=profile.system_prompt,
                             max_isl_tokens=profile.isl_tokens, max_osl_tokens=profile.osl_tokens,
                             context_safety_margin_tokens=CONTEXT_SAFETY_MARGIN,
                             tokenizer_name=model or profile.tokenizer_name)
    if d in RANDOM_TRACES:
        return RANDOM_TRACES[d](profile.tokenizer_name, profile.isl_tokens, profile.osl_tokens,
                                seed=seed)
    if d == "sharegpt-multi-turn" or d in TRAJECTORY_DATASETS:
        raise ValueError(f"Profile '{profile.name}' is multi-turn; run it with "
                         "workload.kind=sessions")
    raise ValueError(f"Unknown dataset type: {d}")


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
    if w.kind == "sessions":
        pacing = TurnPacing(w.pacing, w.tool_wait_ms, w.human_wait_ms, w.use_recorded_waits,
                            w.use_recorded_arrivals)
        sessions, _ = resolve_multi_turn_num_sessions(profile, cfg.mode.concurrency)
        if profile.dataset != "sharegpt-multi-turn" and profile.dataset not in TRAJECTORY_DATASETS:
            raise ValueError(f"Profile '{profile.name}' does not use a multi-turn dataset")
        return ProfileSessionTrace(profile, warmup=cfg.mode.warmup, pacing=pacing,
                                   max_turn_index=w.max_turn_index, num_sessions=sessions,
                                   random_seed=w.seed, tokenizer_name=cfg.engine.model or "")
    trace = single_turn_trace(profile, w.seed, cfg.engine.model or "")
    return trace.draw(w.num_requests, cfg.mode.warmup, profile)
