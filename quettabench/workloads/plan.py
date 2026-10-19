"""A run's plan, frozen before anything is sent, and the digest of each request as it goes out. freeze() before the
run; the run digests every request with sent_digest() just before sending it; Plan.digest() says what that digest
should have been. Keys are a request's index for single-turn traces and (session, turn) for session traces, the
session being "<session>#<play>" in a recycled replay."""

import copy
import hashlib
import json
from dataclasses import dataclass, field

from .sessions import PLAY_RULES, SessionTrace, play


def spliced_at(request, replies) -> frozenset:
    """Indexes of the messages in request that are placeholders the run fills with the engine's reply (the trace's
    replies for the session); recorded assistant turns are not among them."""
    placeholders = {id(m) for m in (replies or {}).get(request.session_id) or ()}
    return frozenset(i for i, m in enumerate(request.messages or ()) if id(m) in placeholders)


def sent_digest(request, spliced=frozenset()) -> str:
    """What the server receives: the prompt token ids or messages, the tools, max_tokens and cache_salt. The content
    of a spliced placeholder is left out, because the run writes the engine's reply there."""
    messages = [
        {k: v for k, v in m.items() if not (i in spliced and k == "content")}
        for i, m in enumerate(request.messages or ())
    ]
    body = {
        "prompt_token_ids": request.prompt_token_ids or None,
        "messages": messages or None,
        "tools": request.tools,
        "max_tokens": request.max_tokens,
        "cache_salt": request.cache_salt,
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def play_of(key) -> tuple[str, int]:
    """A recycled key's session and play number: "12#3" is session 12, play 3."""
    session, _, number = str(key[0]).rpartition("#")
    return session, int(number)


@dataclass
class Plan:
    """plan_hash is the hash runs compare on. digests holds the expected digest per key; a recycled replay keeps
    frozen copies of its sessions instead, because which plays start depends on the deadline and the server."""

    plan_hash: str
    digests: dict = field(default_factory=dict)
    frozen: dict | None = None
    seed: int = 0
    salt: str = "text"
    _plays: dict = field(default_factory=dict)

    def digest(self, key) -> str | None:
        if self.frozen is None:
            return self.digests.get(key)
        session, number = play_of(key)
        if session not in self.frozen:
            return None
        if (session, number) not in self._plays:
            turns = play(self.frozen[session], self.seed, number, self.salt)
            self._plays[session, number] = {r.turn: sent_digest(r) for r in turns}
        return self._plays[session, number].get(key[1])

    def planned_keys(self, executed) -> list:
        """Every key the run was to send: the whole plan, or for a recycled replay every turn of the plays started."""
        if self.frozen is None:
            return list(self.digests)
        started = dict.fromkeys(k[0] for k in executed)
        return [(p, r.turn) for p in started for r in self.frozen.get(play_of((p, 0))[0], ())]


def _hash(header: dict, digests) -> str:
    h = hashlib.sha256(json.dumps(header, sort_keys=True, default=str).encode())
    for key, d in digests:
        h.update(f"\n{key}|{d}".encode())
    return h.hexdigest()[:16]


def freeze(trace, recycled=False, seed=0, salt="text") -> Plan:
    """Digests every planned request now. A single-turn trace keeps its workload_hash as plan_hash. A session
    trace's plan_hash covers the file's sha256, the sample, the session order, the play rule (seed and salt policy)
    when recycled, and every request's digest; a recycled replay's does not depend on its duration or the server's
    speed."""
    if not isinstance(trace, SessionTrace):
        return Plan(trace.workload_hash(), {i: sent_digest(r) for i, r in enumerate(trace.requests())})
    sessions = trace.sessions()
    replies = getattr(trace, "replies", None)
    digests = {(r.session_id, r.turn): sent_digest(r, spliced_at(r, replies)) for s in sessions for r in s}
    header = {
        "file_sha256": getattr(trace, "file_sha256", None),
        "sample": list(getattr(trace, "sample", [])),
        "order": [str(s[0].session_id) for s in sessions],
        "play_rule": {"seed": seed, "salt": salt, "rule": PLAY_RULES[salt]} if recycled else None,
    }
    plan_hash = _hash(header, digests.items())
    if not recycled:
        return Plan(plan_hash, digests)
    frozen = {str(s[0].session_id): copy.deepcopy(s) for s in sessions}
    return Plan(plan_hash, frozen=frozen, seed=seed, salt=salt)
