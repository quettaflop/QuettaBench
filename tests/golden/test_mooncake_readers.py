"""Golden: the two Mooncake readers on the same 5-line trace.

src/workloads/mooncake.py (the src runner's mooncake-trace profile) and
synth_bench.load_trace (the crossval Mooncake path) read the same file into
different requests: different token ids, different fields, and only one of
them turns timestamp into an arrival. Both outputs are pinned as they are;
REPORT.md lists the differences. Token id lists are pinned by sha256 plus
length and first ids, and by which blocks are shared with request 0.
"""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from quettabench.workloads.mooncake import MooncakeTraceDataset
from tests.golden._harness import FIXTURES, check_json, crossval_path, fresh_import

TRACE = FIXTURES / "mooncake_5.jsonl"
BLOCK = 512


def describe_ids(ids, first_ids):
    """Length, digest, head, and how many leading 512-token blocks match request 0."""
    shared = 0
    while (shared + 1) * BLOCK <= min(len(ids), len(first_ids)) and \
            ids[shared * BLOCK:(shared + 1) * BLOCK] == first_ids[shared * BLOCK:(shared + 1) * BLOCK]:
        shared += 1
    return {
        "len": len(ids),
        "sha256": hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest(),
        "head": ids[:4],
        "min": min(ids),
        "max": max(ids),
        "leading_blocks_shared_with_req0": shared,
    }


def read_src(path):
    """Six get_next_request() calls: the five lines, then the wrap-around."""
    ds = MooncakeTraceDataset(str(path))
    reqs = [ds.get_next_request() for _ in range(6)]
    first = reqs[0].metadata["prompt_token_ids"]
    return [{
        "messages": r.messages,
        "max_tokens": r.max_tokens,
        "metadata": {k: v for k, v in r.metadata.items() if k != "prompt_token_ids"},
        "prompt_token_ids": describe_ids(r.metadata["prompt_token_ids"], first),
    } for r in reqs]


def read_crossval(path):
    with crossval_path(), mock.patch.dict(os.environ, {}, clear=True):
        synth_bench = fresh_import("quettabench.workloads.synth_bench")
        reqs = synth_bench.load_trace(str(path))
        workload_hash = synth_bench.workload_hash(reqs)
    first = reqs[0]["prompt_token_ids"]
    return {
        "requests": [{**{k: v for k, v in r.items() if k != "prompt_token_ids"},
                      "prompt_token_ids": describe_ids(r["prompt_token_ids"], first)}
                     for r in reqs],
        "workload_hash": workload_hash,
    }


def outcome(fn, path):
    try:
        return fn(path)
    except Exception as e:  # a reader refusing the file is part of the pin
        return {"error": type(e).__name__, "message": str(e)}


class MooncakeReadersGolden(unittest.TestCase):
    def test_both_readers_on_one_trace(self):
        with tempfile.TemporaryDirectory() as td:
            no_hash_ids = Path(td) / "mooncake_no_hash_ids.jsonl"
            no_hash_ids.write_text("".join(
                json.dumps({k: v for k, v in json.loads(line).items() if k != "hash_ids"}) + "\n"
                for line in TRACE.read_text().splitlines()))
            record = {
                "with_hash_ids": {"src_mooncake": outcome(read_src, TRACE),
                                  "crossval_synth_bench": outcome(read_crossval, TRACE)},
                "without_hash_ids": {"src_mooncake": outcome(read_src, no_hash_ids),
                                     "crossval_synth_bench": outcome(read_crossval, no_hash_ids)},
            }
        check_json(self, "mooncake_readers", record)


if __name__ == "__main__":
    unittest.main()
