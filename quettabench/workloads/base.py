"""Request, the Trace base class, and workload_hash, which names a request list so two engines can prove they ran identical work."""

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, replace

VOCAB = 100_000  # safe id ceiling for the checkpoints under test


def synth_ids(n, seed):
    """Distinct pseudo-random token stream; different seed gives different routing."""
    a = 1103515245 * (seed + 1) + 12345
    return [((a + i * (2 * seed + 7)) % (VOCAB - 1)) + 1 for i in range(n)]


@dataclass
class Request:
    """One request as the replay schema writes it; None means absent from the line, which the hash treats differently from zero."""

    prompt_token_ids: list
    prompt_len: int
    order: int | None = None
    output_length: int | None = None
    output_token_ids: list | None = None
    arrival_ts: float | None = None
    session_id: object = None
    turn: int | None = None
    seed: int | None = None
    image_tokens: int | None = None
    source: str | None = None
    # Chat messages and metadata shared with the session so later turns see replies; not in the replay schema, hashed only without token ids.
    messages: list | None = None
    metadata: dict | None = None

    @property
    def max_tokens(self):
        return self.output_length

    def to_dict(self):
        """The request as a replay JSONL dict: set fields only."""
        plain = replace(self, messages=None, metadata=None)
        return {k: v for k, v in asdict(plain).items() if v is not None}


def messages_digest(messages):
    """sha256 of the messages as canonical JSON, assistant content left out because a live run writes replies there."""
    plain = [{k: v for k, v in m.items() if not (m.get("role") == "assistant" and k == "content")} for m in messages]
    return hashlib.sha256(json.dumps(plain, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def request_digest(r):
    """sha256 of one request's deterministic inputs (seed, order, arrival, prompt, requested output), never generated tokens or timings; a chat request without token ids adds its messages digest."""
    messages = r.messages if isinstance(r, Request) else r.get("messages")
    if isinstance(r, Request):
        r = r.to_dict()
    ids = r.get("prompt_token_ids") or []
    parts = [
        str(r.get("seed", "")),
        str(r.get("order", "")),
        f"{float(r.get('arrival_ts', 0.0)):.6f}",
        str(r.get("prompt_len", len(ids))),
        ",".join(map(str, ids)),
        str(r.get("output_length", "")),
        str(r.get("session_id", "")),
        str(r.get("turn", "")),
    ]
    if not ids and messages:
        parts.append(messages_digest(messages))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def workload_hash(reqs):
    """First 16 hex of a sha256 over the request count and sorted per-request digests; independent of order, changed by a dropped or altered request. Takes Requests or dicts."""
    digs = sorted(request_digest(r) for r in reqs)
    h = hashlib.sha256()
    h.update(str(len(digs)).encode())
    for d in digs:
        h.update(b"\n")
        h.update(d.encode())
    return h.hexdigest()[:16]


class Trace(ABC):
    """A fixed request list plus its source; supports_spec says whether requests carry real assistant tokens to score speculative acceptance against."""

    supports_spec: bool = False

    @abstractmethod
    def requests(self) -> list[Request]:
        """Every request, in submission order."""

    @abstractmethod
    def source_meta(self) -> dict:
        """Provenance for the results file: path, reader, sources."""

    def workload_hash(self) -> str:
        return workload_hash(self.requests())

    def warmup_requests(self, n) -> list[Request]:
        """n requests to warm the engine with: the trace's own, cycled."""
        reqs = self.requests()
        return [reqs[i % len(reqs)] for i in range(n)] if reqs else []


class DrawnTrace(Trace):
    """A trace drawn one request at a time from a source that wraps around; draw() takes `warmup` draws, rewinds if the source replays a file in order, then takes `num_requests` draws numbered from zero."""

    rewinds = False

    def __init__(self):
        self._warmup, self._requests, self.profile = [], [], None

    @abstractmethod
    def get_next_request(self) -> Request:
        """The next request; the source wraps around at its end."""

    def reset(self):
        """Back to the first request; only sources that rewind override it."""

    def draw(self, num_requests, warmup=0, profile=None):
        self.profile = profile
        self._warmup = [replace(self.get_next_request(), order=i) for i in range(warmup)]
        if warmup and self.rewinds:
            self.reset()
        self._requests = [replace(self.get_next_request(), order=i) for i in range(num_requests)]
        return self

    def requests(self):
        return self._requests

    def warmup_requests(self, n):
        return self._warmup[:n]

    def source_meta(self):
        p = self.profile
        meta = {"profile": p.name, "dataset": p.dataset, "path": p.file_path} if p else {"reader": type(self).__name__}
        return {**meta, "reqs": len(self._requests)}
