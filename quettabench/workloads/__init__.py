"""Every Trace a run can replay; build(cfg) picks the one a Config names."""

from quettabench.config.schema import ConfigError

from .base import DoubleWrapRandomTokenTrace, LegacyRandomTokenTrace, RandomTokenTrace
from .grid import GridTrace
from .live import MooncakeTrace, PromptJsonlTrace, RealTrace, ShareGPTTrace, SmokeTrace
from .profiles import PROFILE_ALIASES, PROFILES, get_profile
from .sessions import (
    ShareGPTSessionTrace,
    TrajectorySessionTrace,
    TurnPacing,
    resolve_multi_turn_num_sessions,
)

TRAJECTORY_DATASETS = ("swebench-multi-turn", "terminalbench-multi-turn", "osworld-multi-turn")
RANDOM_TRACES = {"random": RandomTokenTrace, "random-legacy": LegacyRandomTokenTrace,
                 "random-doublewrap": DoubleWrapRandomTokenTrace}

# Room build leaves between a trajectory turn's prompt plus output and the
# profile's isl_tokens context window.
CONTEXT_SAFETY_MARGIN = 256


def single_turn_trace(profile, seed):
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
                             context_safety_margin_tokens=CONTEXT_SAFETY_MARGIN)
    if d in RANDOM_TRACES:
        return RANDOM_TRACES[d](profile.tokenizer_name, profile.isl_tokens, profile.osl_tokens,
                                seed=seed)
    if d == "sharegpt-multi-turn" or d in TRAJECTORY_DATASETS:
        raise ValueError(f"Profile '{profile.name}' is multi-turn; run it with "
                         "workload.kind=sessions")
    raise ValueError(f"Unknown dataset type: {d}")


def session_trace(profile, num_sessions, seed, **trace_kw):
    """A multi-turn profile's sessions as a LoadedSessionTrace."""
    if profile.dataset == "sharegpt-multi-turn":
        return ShareGPTSessionTrace(
            min_turns=profile.min_turns, max_turns=profile.max_turns, num_sessions=num_sessions,
            seed=seed, system_prompt=profile.system_prompt, max_isl_tokens=profile.isl_tokens,
            max_osl_tokens=profile.osl_tokens, profile=profile, **trace_kw)
    if profile.dataset in TRAJECTORY_DATASETS:
        return TrajectorySessionTrace(
            profile.file_path, min_turns=profile.min_turns, max_turns=profile.max_turns,
            num_sessions=num_sessions, seed=seed, max_isl_tokens=profile.isl_tokens,
            max_osl_tokens=profile.osl_tokens,
            context_safety_margin_tokens=CONTEXT_SAFETY_MARGIN, profile=profile, **trace_kw)
    raise ValueError(f"Profile '{profile.name}' does not use a multi-turn dataset")


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
        return session_trace(profile, sessions, w.seed, warmup=cfg.mode.warmup, pacing=pacing,
                             max_turn_index=w.max_turn_index)
    trace = single_turn_trace(profile, w.seed)
    if (w.arrival == "trace" and isinstance(trace, MooncakeTrace) and trace.lap_ms == 0
            and w.num_requests > len(trace.records)):
        # Every lap would land on the first lap's times; a lap length is not invented.
        raise ConfigError(f"workload.arrival=trace: {profile.name}'s timestamps span 0 ms, so "
                          f"{w.num_requests} requests over its {len(trace.records)} records "
                          "wrap with no lap length; lower workload.num_requests or use "
                          "another arrival")
    return trace.draw(w.num_requests, cfg.mode.warmup, profile)
