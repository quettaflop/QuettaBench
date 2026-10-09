"""Request, the Trace base class, and the workload identity hash.

A Trace is a fixed list of Requests. workload_hash() names that list so two
engines can prove they ran identical work: serve gates on it, crossval joins
on it.
"""

import hashlib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

VOCAB = 100_000  # safe id ceiling for the checkpoints under test


def synth_ids(n, seed):
    """Distinct pseudo-random token stream; different seed gives different routing."""
    a = 1103515245 * (seed + 1) + 12345
    return [((a + i * (2 * seed + 7)) % (VOCAB - 1)) + 1 for i in range(n)]


@dataclass
class Request:
    """One request as the replay schema writes it. None means the field is
    absent from the line, which is not the same as zero for the hash."""

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

    def to_dict(self):
        """The dict synth_bench.load_trace returns: set fields only."""
        return {k: v for k, v in asdict(self).items() if v is not None}


def request_digest(r):
    """Identity of one request over deterministic inputs only (seed, order, arrival,
    prompt content, requested output), never generated tokens or timings, so two
    engines doing identical work hash the same."""
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
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def workload_hash(reqs):
    """Order-independent identity of the whole workload: first 16 hex of sha256
    over the request count and the sorted per-request digests. Completion order
    does not move it; a dropped or altered request does. Takes Requests or the
    equivalent dicts."""
    digs = sorted(request_digest(r) for r in reqs)
    h = hashlib.sha256()
    h.update(str(len(digs)).encode())
    for d in digs:
        h.update(b"\n")
        h.update(d.encode())
    return h.hexdigest()[:16]


class Trace(ABC):
    """A fixed request list plus where it came from. supports_spec says whether
    requests carry real assistant tokens to score speculative acceptance against."""

    supports_spec: bool = False

    @abstractmethod
    def requests(self) -> list[Request]:
        """Every request, in submission order."""

    @abstractmethod
    def source_meta(self) -> dict:
        """Provenance for the results file: path, reader, sources."""

    def workload_hash(self) -> str:
        return workload_hash(self.requests())
