"""Recorded traces: the replay JSONL schema and the Mooncake production trace.

One reader for both. A line carries prompt_token_ids, or prompt_len, or the
Mooncake fields (input_length, output_length, timestamp in ms, hash_ids).
Mooncake hash_ids name 512-token prefix blocks; each id expands to the same
seeded block every time, so requests that share a hash id share those tokens
byte for byte and prefix sharing survives the replay. A Mooncake line with no
hash_ids falls back to synth_ids(input_length, line index).

Trace format reference: https://github.com/kvcache-ai/Mooncake
"""

import json
import random
from dataclasses import dataclass

from .base import Request, Trace, synth_ids

BLOCK_SIZE = 512

# Range avoids low special token ids and stays under small vocabs.
TOKEN_ID_LOW = 1000
TOKEN_ID_HIGH = 20000


@dataclass(frozen=True)
class MooncakeRecord:
    index: int  # line index in the file
    input_length: int
    output_length: int | None
    hash_ids: tuple[int, ...]


def _rows(path):
    """(line index, parsed dict) for every non-blank line."""
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                yield i, json.loads(line)


def mooncake_record(row, index):
    """input_length is required; output_length and hash_ids may be absent."""
    out = row.get("output_length")
    return MooncakeRecord(
        index=index,
        input_length=int(row["input_length"]),
        output_length=None if out is None else int(out),
        hash_ids=tuple(int(h) for h in row.get("hash_ids") or ()),
    )


def parse_mooncake_trace(path, limit=0):
    """Raw Mooncake records in file order, for tools that study the trace itself."""
    records = []
    for i, row in _rows(path):
        rec = mooncake_record(row, i)
        if rec.output_length is None:
            raise ValueError(f"{path}:{i + 1}: Mooncake line needs output_length")
        records.append(rec)
        if limit and len(records) >= limit:
            break
    if not records:
        raise ValueError(f"No records parsed from {path}")
    return records


def expand_block(hash_id, seed):
    """One hash id always yields the same BLOCK_SIZE token ids."""
    rng = random.Random(f"{seed}:{hash_id}")
    return [rng.randrange(TOKEN_ID_LOW, TOKEN_ID_HIGH) for _ in range(BLOCK_SIZE)]


def fill_tail(record_index, length, seed):
    """Request-unique filler for input tokens not covered by hash blocks."""
    rng = random.Random(f"{seed}:tail:{record_index}")
    return [rng.randrange(TOKEN_ID_LOW, TOKEN_ID_HIGH) for _ in range(length)]


def mooncake_prompt_ids(record, seed=42, block_cache=None):
    """Exactly input_length ids: hash blocks in order, tail filled per request,
    truncated when the blocks overrun (the common case in the published trace).
    No hash_ids: synth_ids(input_length, line index)."""
    if not record.hash_ids:
        return synth_ids(record.input_length, record.index)
    tokens = []
    for hash_id in record.hash_ids:
        if len(tokens) >= record.input_length:
            break
        if block_cache is None:
            block = expand_block(hash_id, seed)
        else:
            block = block_cache.get(hash_id)
            if block is None:
                block = block_cache[hash_id] = expand_block(hash_id, seed)
        tokens.extend(block)
    if len(tokens) < record.input_length:
        tokens.extend(fill_tail(record.index, record.input_length - len(tokens), seed))
    return tokens[: record.input_length]


def _carry_meta(req, row):
    """Scheduling and provenance fields; Mooncake timestamp is ms."""
    if row.get("arrival_ts") is not None:
        req.arrival_ts = float(row["arrival_ts"])
    elif row.get("timestamp") is not None:
        req.arrival_ts = float(row["timestamp"]) / 1e3
    for k in ("session_id", "turn", "image_tokens", "seed", "source"):
        if row.get(k) is not None:
            setattr(req, k, row[k])
    return req


def _mooncake_request(row, index, seed, block_cache):
    """A Mooncake line keeps output_length 0: it is the trace's own number."""
    rec = mooncake_record(row, index)
    ids = mooncake_prompt_ids(rec, seed, block_cache)
    req = Request(ids, rec.input_length, order=index, output_length=rec.output_length)
    return _carry_meta(req, row)


def _line_request(path, row, index, seed, block_cache):
    """One replay-schema line; order is the line index, a hash input."""
    if row.get("prompt_token_ids"):
        ids = list(row["prompt_token_ids"])
        req = Request(ids, len(ids), order=index)
        if row.get("output_length"):
            req.output_length = int(row["output_length"])
        # Reference tokens for acceptance scoring (SPECSUM); must survive load
        # or the replay has nothing to compare the engine's output against.
        if row.get("output_token_ids"):
            req.output_token_ids = list(row["output_token_ids"])
        return _carry_meta(req, row)
    if not row.get("prompt_len") and row.get("input_length"):
        return _mooncake_request(row, index, seed, block_cache)
    length = row.get("prompt_len")
    if not length:
        raise ValueError(f"{path}:{index + 1}: need prompt_token_ids, prompt_len, or input_length")
    req = Request(synth_ids(int(length), index), int(length), order=index)
    if row.get("output_length"):
        req.output_length = int(row["output_length"])
    return _carry_meta(req, row)


class RealTrace(Trace):
    """Requests read from a file. supports_spec when any request carries the
    trace's real assistant tokens (output_token_ids)."""

    def __init__(self, requests, path, reader):
        if not requests:
            raise ValueError(f"{path}: no requests")
        self._requests = requests
        self.path = str(path)
        self.reader = reader
        self.supports_spec = any(r.output_token_ids for r in requests)

    @classmethod
    def from_jsonl(cls, path, seed=42):
        """The replay schema; seed only matters for Mooncake hash blocks."""
        cache = {}
        reqs = [_line_request(path, row, i, seed, cache) for i, row in _rows(path)]
        return cls(reqs, path, "jsonl")

    @classmethod
    def from_mooncake(cls, path, seed=42, limit=None):
        """Mooncake lines only (input_length required), first `limit` of them."""
        cache, reqs = {}, []
        for i, row in _rows(path):
            reqs.append(_mooncake_request(row, i, seed, cache))
            if limit and len(reqs) >= limit:
                break
        return cls(reqs, path, "mooncake")

    def requests(self):
        return self._requests

    def source_meta(self):
        return {
            "path": self.path,
            "reader": self.reader,
            "reqs": len(self._requests),
            "sources": sorted({"?" if r.source is None else str(r.source) for r in self._requests}),
            "text_mode": "real" if self.supports_spec else "synthetic",
        }
