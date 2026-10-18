"""Request, the Trace base class and workload_hash."""

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, replace

VOCAB = 100_000


def synth_ids(n, seed):
    """A different seed gives different routing."""
    a = 1103515245 * (seed + 1) + 12345
    return [((a + i * (2 * seed + 7)) % (VOCAB - 1)) + 1 for i in range(n)]


@dataclass
class Request:
    """None means absent from the line, which the hash treats differently from zero."""

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
    # Not in the replay schema; hashed only without token ids.
    messages: list | None = None
    # The tool schema an agent sent with its messages; not in the replay schema, hashed like messages.
    tools: list | None = None
    metadata: dict | None = None

    @property
    def max_tokens(self):
        return self.output_length

    def to_dict(self):
        plain = replace(self, messages=None, tools=None, metadata=None)
        return {k: v for k, v in asdict(plain).items() if v is not None}


def messages_digest(messages):
    """Assistant content left out because a live run writes replies there."""
    plain = [{k: v for k, v in m.items() if not (m.get("role") == "assistant" and k == "content")} for m in messages]
    return hashlib.sha256(json.dumps(plain, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def request_digest(r):
    """Never generated tokens or timings."""
    messages = r.messages if isinstance(r, Request) else r.get("messages")
    tools = r.tools if isinstance(r, Request) else r.get("tools")
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
    if not ids and tools:
        parts.append(hashlib.sha256(json.dumps(tools, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def workload_hash(reqs):
    """Independent of order."""
    digs = sorted(request_digest(r) for r in reqs)
    h = hashlib.sha256()
    h.update(str(len(digs)).encode())
    for d in digs:
        h.update(b"\n")
        h.update(d.encode())
    return h.hexdigest()[:16]


class Trace(ABC):
    """supports_spec: requests carry real assistant tokens to score acceptance against."""

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
        """The trace's own, cycled."""
        reqs = self.requests()
        return [reqs[i % len(reqs)] for i in range(n)] if reqs else []


class DrawnTrace(Trace):
    """draw() takes `warmup` draws, then `num_requests` draws."""

    rewinds = False

    def __init__(self):
        self._warmup, self._requests, self.profile = [], [], None

    @abstractmethod
    def get_next_request(self) -> Request:
        """The next request; the source wraps around at its end."""

    def reset(self):
        """Only sources that rewind override it."""

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
