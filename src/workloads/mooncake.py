"""Mooncake production trace support.

Each trace line is one request: input and output token counts, and
hash_ids naming 512-token prefix blocks (equal ids mean
equal blocks, which encodes prefix sharing). The trace has no token
content, so blocks expand to deterministic token ids and shared
prefixes stay shared byte for byte in the replay. Note 512 is the hash
granularity, not the cache hit granularity: records usually truncate
mid block, so sharing is effectively token for token up to input_length.

Trace format reference: https://github.com/kvcache-ai/Mooncake
"""

import json
import random
import threading
from dataclasses import dataclass
from typing import Optional

from .dataset import BaseDataset, BenchmarkRequest

BLOCK_SIZE = 512

# Token ids are sampled from this range. It avoids the low ids where
# most tokenizers keep special tokens, and stays under small vocabs.
TOKEN_ID_LOW = 1000
TOKEN_ID_HIGH = 20000


@dataclass(frozen=True)
class MooncakeRecord:
    index: int
    input_length: int
    output_length: int
    hash_ids: tuple[int, ...]


def parse_mooncake_trace(filepath: str, limit: int = 0) -> list[MooncakeRecord]:
    """Read the raw Mooncake JSONL trace, preserving trace order."""
    records = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            records.append(MooncakeRecord(
                index=len(records),
                input_length=int(row["input_length"]),
                output_length=int(row["output_length"]),
                hash_ids=tuple(int(h) for h in row["hash_ids"]),
            ))
            if limit and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"No records parsed from {filepath}")
    return records


def synthesize_block(hash_id: int, seed: int) -> list[int]:
    """One hash id always yields the same BLOCK_SIZE token ids."""
    rng = random.Random(f"{seed}:{hash_id}")
    return [rng.randrange(TOKEN_ID_LOW, TOKEN_ID_HIGH) for _ in range(BLOCK_SIZE)]


def synthesize_tail(record_index: int, length: int, seed: int) -> list[int]:
    """Request-unique filler for input tokens not covered by hash blocks."""
    rng = random.Random(f"{seed}:tail:{record_index}")
    return [rng.randrange(TOKEN_ID_LOW, TOKEN_ID_HIGH) for _ in range(length)]


def build_prompt_token_ids(record: MooncakeRecord, seed: int = 42,
                           block_cache: Optional[dict] = None) -> list[int]:
    """Expand a record into exactly input_length token ids.

    Short block coverage is tail filled with request-unique tokens,
    long coverage is truncated (the common case in the published trace).
    """
    tokens: list[int] = []
    for hash_id in record.hash_ids:
        if len(tokens) >= record.input_length:
            break
        if block_cache is not None:
            block = block_cache.get(hash_id)
            if block is None:
                block = synthesize_block(hash_id, seed)
                block_cache[hash_id] = block
        else:
            block = synthesize_block(hash_id, seed)
        tokens.extend(block)
    if len(tokens) < record.input_length:
        tokens.extend(synthesize_tail(record.index,
                                      record.input_length - len(tokens), seed))
    return tokens[: record.input_length]


class MooncakeTraceDataset(BaseDataset):
    """Serves the Mooncake trace in original order with synthesized tokens.

    Requests carry prompt_token_ids in metadata. The token ids are meant
    for a completions style backend that submits them directly, so no
    chat template can disturb block boundaries.
    """

    def __init__(self, filepath: str, seed: int = 42, limit: int = 0):
        self.filepath = filepath
        self.seed = seed
        self.limit = limit
        self._records: Optional[list[MooncakeRecord]] = None
        self._cursor = 0
        self._block_cache: dict = {}
        self._lock = threading.Lock()

    def _load(self):
        if self._records is None:
            with self._lock:
                if self._records is None:
                    self._records = parse_mooncake_trace(self.filepath, self.limit)

    @property
    def records(self) -> list[MooncakeRecord]:
        self._load()
        return self._records

    def reset(self) -> None:
        """Rewind so a post-warmup run starts at record zero."""
        with self._lock:
            self._cursor = 0

    def get_next_request(self) -> BenchmarkRequest:
        self._load()
        with self._lock:
            record = self._records[self._cursor % len(self._records)]
            self._cursor += 1
        token_ids = build_prompt_token_ids(record, self.seed, self._block_cache)
        return BenchmarkRequest(
            messages=[],
            max_tokens=record.output_length,
            metadata={
                "prompt_token_ids": token_ids,
                "request_index": record.index,
                "num_hash_blocks": len(record.hash_ids),
            },
        )

