"""Mooncake trace as a runner dataset.

The reading and the token expansion live in live.py; this adapter only
serves the records in file order through the BaseDataset interface.
"""

import threading

from .dataset import BaseDataset, BenchmarkRequest
from .live import MooncakeRecord, mooncake_prompt_ids, parse_mooncake_trace


class MooncakeTraceDataset(BaseDataset):
    """Serves the Mooncake trace in original order with expanded token ids.

    Requests carry prompt_token_ids in metadata for a completions style
    backend, so no chat template can disturb block boundaries.
    """

    def __init__(self, filepath: str, seed: int = 42, limit: int = 0):
        self.filepath = filepath
        self.seed = seed
        self.limit = limit
        self._records: list[MooncakeRecord] | None = None
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
        token_ids = mooncake_prompt_ids(record, self.seed, self._block_cache)
        return BenchmarkRequest(
            messages=[],
            max_tokens=record.output_length,
            metadata={
                "prompt_token_ids": token_ids,
                "request_index": record.index,
                "num_hash_blocks": len(record.hash_ids),
            },
        )
