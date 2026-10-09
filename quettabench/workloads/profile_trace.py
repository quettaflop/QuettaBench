"""Workload profiles as Traces.

A profile's dataset is a stateful iterator the old runner drew from at
dispatch time. Here it is drawn once, in the order that runner drew it, into
fixed Request lists: warmup first, then the measured requests, after
rewinding a dataset that can rewind (a trace replay, so request i is still
record i). Requests keep the dataset's own messages and metadata objects, so
a multi-turn session's later turns see the replies written into its
assistant placeholders.
"""

import dataclasses

from .base import Request, Trace
from .dataset import (
    DistributionalMultiTurnDataset,
    ShareGPTMultiTurnDataset,
    TrajectoryMultiTurnDataset,
    make_dataset,
)
from .sessions import SessionTrace, TurnPacing

MULTI_TURN_DATASETS = (DistributionalMultiTurnDataset, ShareGPTMultiTurnDataset,
                       TrajectoryMultiTurnDataset)


def profile_request(req, order, session_id=None, turn=None, arrival_ts=None):
    """A dataset's BenchmarkRequest as a Request; token ids from its metadata."""
    ids = req.metadata.get("prompt_token_ids") or []
    return Request(ids, len(ids), order=order, output_length=req.max_tokens,
                   session_id=session_id, turn=turn, arrival_ts=arrival_ts,
                   messages=req.messages, metadata=req.metadata)


def _profile_meta(profile, n):
    return {"profile": profile.name, "dataset": profile.dataset,
            "path": profile.file_path, "reqs": n}


class ProfileTrace(Trace):
    """Single-turn profile: `warmup` draws, rewind, then `num_requests` draws."""

    def __init__(self, profile, num_requests, warmup=0, workload_file="", **dataset_kw):
        if workload_file:
            profile = dataclasses.replace(profile, file_path=workload_file)
        self.profile = profile
        dataset = make_dataset(profile, **dataset_kw)
        self._warmup = [profile_request(dataset.get_next_request(), i) for i in range(warmup)]
        reset = getattr(dataset, "reset", None)
        if warmup and callable(reset):
            reset()
        self._requests = [profile_request(dataset.get_next_request(), i)
                          for i in range(num_requests)]

    def requests(self):
        return self._requests

    def warmup_requests(self, n):
        return self._warmup[:n]

    def source_meta(self):
        return _profile_meta(self.profile, len(self._requests))


class ProfileSessionTrace(SessionTrace):
    """Multi-turn profile: every session's turns up to max_turn_index, turn 0
    carrying the session's recorded arrival. max_turns is the longest
    session as loaded, before that cut. Warmup draws come from the dataset's
    flat round-robin, which repeats the sessions' own turns."""

    def __init__(self, profile, warmup=0, max_turn_index=None, pacing=None, **dataset_kw):
        dataset = make_dataset(profile, **dataset_kw)
        if not isinstance(dataset, MULTI_TURN_DATASETS):
            raise ValueError(  # noqa: TRY004 (the old runner's error, kept)
                f"Profile '{profile.name}' does not use a multi-turn dataset")
        sessions = dataset.sessions
        if not sessions:
            raise ValueError("No multi-turn sessions loaded -- check ShareGPT dataset and filter bounds")
        self.max_turns = max(len(s.turns) for s in sessions)
        print(f"Loaded {len(sessions)} sessions, max {self.max_turns} turns per session")
        reqs = []
        for s in sessions:
            for t, req in enumerate(s.turns):
                if max_turn_index is not None and t > max_turn_index:
                    break
                arrival = s.arrival_time_ms / 1000.0 if t == 0 else None
                reqs.append(profile_request(req, len(reqs), s.session_id, t, arrival))
        super().__init__(reqs, profile.file_path or profile.name, f"profile:{profile.dataset}")
        self.profile = profile
        self.pacing = pacing or TurnPacing()
        self.replies = {s.session_id: s.assistant_messages for s in sessions}
        self._warmup = [profile_request(dataset.get_next_request(), i) for i in range(warmup)]

    def warmup_requests(self, n):
        return self._warmup[:n]

    def source_meta(self):
        return {**_profile_meta(self.profile, len(self.requests())),
                "sessions": len(self.replies), "pacing": self.pacing.pacing}
