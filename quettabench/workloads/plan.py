"""Freezes a run's plan before sending and digests each request as sent, so serve can check one against the other.
plan_hash is what runs compare on; a recycled replay keys each play "<session>#<play>"."""

import copy
import hashlib
import json
from dataclasses import dataclass, field

from .sessions import PLAY_RULES, SessionTrace, play


def spliced_at(request, replies) -> frozenset:
    """Positions the run fills with live replies, left out of the digest."""
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
    # The run's own id in every salt; not in plan_hash, so two runs of one plan still compare.
    run: str = ""
    _digests: dict = field(default_factory=dict)

    def digest(self, key) -> str | None:
        """The planned digest of one request, computed once per key."""
        if self.frozen is None:
            return self.digests.get(key)
        if key not in self._digests:
            session, number = play_of(key)
            turn = next((r for r in self.frozen.get(session, ()) if r.turn == key[1]), None)
            if turn is None:
                self._digests[key] = None
            else:
                self._digests[key] = sent_digest(play([turn], self.seed, number, self.salt, self.run)[0])
        return self._digests[key]

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


def freeze(trace, recycled=False, seed=0, salt="text", run="") -> Plan:
    """Digest every planned request before the run; plan_hash is what runs compare on and does not depend on
    duration or server speed."""
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
    return Plan(plan_hash, frozen=frozen, seed=seed, salt=salt, run=run)
