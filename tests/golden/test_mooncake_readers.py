"""Golden: the one Mooncake reader (workloads/live.py) on a 5-line trace.

RealTrace.from_jsonl, RealTrace.from_mooncake and the runner's
MooncakeTraceDataset all expand hash_ids through live.mooncake_prompt_ids, so
they must produce the same token ids. Pinned with hash_ids, without them
(synth_ids fallback), and with a zero output_length (kept, not dropped).
Token id lists are pinned by sha256 plus length and first ids, and by how
many leading 512-token blocks each request shares with request 0.
"""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


def read_all(case, path):
    """The trace through every entry point; ids must agree across all three."""
    with crossval_path(), mock.patch.dict(os.environ, {}, clear=True):
        live = fresh_import("quettabench.workloads.live")
        mooncake = fresh_import("quettabench.workloads.mooncake")
        trace = live.RealTrace.from_jsonl(str(path))
        strict = live.RealTrace.from_mooncake(str(path))
        limited = live.RealTrace.from_mooncake(str(path), limit=2)
        ds = mooncake.MooncakeTraceDataset(str(path))
        served = [ds.get_next_request() for _ in range(6)]  # five lines, then the wrap
    reqs = [r.to_dict() for r in trace.requests()]
    case.assertEqual(reqs, [r.to_dict() for r in strict.requests()])
    case.assertEqual(reqs[:2], [r.to_dict() for r in limited.requests()])
    case.assertEqual([r["prompt_token_ids"] for r in reqs] + [reqs[0]["prompt_token_ids"]],
                     [s.metadata["prompt_token_ids"] for s in served])
    first = reqs[0]["prompt_token_ids"]
    return {
        "requests": [{**{k: v for k, v in r.items() if k != "prompt_token_ids"},
                      "prompt_token_ids": describe_ids(r["prompt_token_ids"], first)}
                     for r in reqs],
        "workload_hash": trace.workload_hash(),
        "source_meta": {k: v for k, v in trace.source_meta().items() if k != "path"},
        "dataset": [{"max_tokens": s.max_tokens,
                     "metadata": {k: v for k, v in s.metadata.items() if k != "prompt_token_ids"}}
                    for s in served],
    }


class MooncakeReaderGolden(unittest.TestCase):
    def test_one_reader_on_one_trace(self):
        rows = [json.loads(line) for line in TRACE.read_text().splitlines()]
        with tempfile.TemporaryDirectory() as td:
            no_hash_ids = Path(td) / "mooncake_no_hash_ids.jsonl"
            no_hash_ids.write_text("".join(
                json.dumps({k: v for k, v in r.items() if k != "hash_ids"}) + "\n" for r in rows))
            zero_out = Path(td) / "mooncake_zero_output.jsonl"
            zero_out.write_text("".join(
                json.dumps({**r, "output_length": 0} if i == 4 else r) + "\n"
                for i, r in enumerate(rows)))
            record = {
                "with_hash_ids": read_all(self, TRACE),
                "without_hash_ids": read_all(self, no_hash_ids),
                "zero_output_length": read_all(self, zero_out),
            }
        self.assertEqual(record["zero_output_length"]["requests"][4]["output_length"], 0)
        check_json(self, "mooncake_readers", record)


if __name__ == "__main__":
    unittest.main()
