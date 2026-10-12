"""Golden: the Mooncake readers and workload_hash on fixed traces."""

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.golden._harness import FIXTURES, check_json, crossval_path, fresh_import

MOONCAKE_TRACE = FIXTURES / "mooncake_5.jsonl"
BLOCK = 512


def describe_ids(ids, first_ids):
    """Length, digest, head, and how many leading 512-token blocks match request 0."""
    shared = 0
    while (shared + 1) * BLOCK <= min(len(ids), len(first_ids)) and ids[
        shared * BLOCK : (shared + 1) * BLOCK
    ] == first_ids[shared * BLOCK : (shared + 1) * BLOCK]:
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
    """The trace through both entry points; ids must agree across them."""
    with crossval_path(), mock.patch.dict(os.environ, {}, clear=True):
        live = fresh_import("quettabench.workloads.live")
        trace = live.RealTrace.from_jsonl(str(path))
        ds = live.MooncakeTrace(str(path))
        served = [ds.get_next_request() for _ in range(6)]  # five lines, then the wrap
    reqs = [r.to_dict() for r in trace.requests()]
    case.assertEqual(
        [r["prompt_token_ids"] for r in reqs] + [reqs[0]["prompt_token_ids"]],
        [s.metadata["prompt_token_ids"] for s in served],
    )
    first = reqs[0]["prompt_token_ids"]
    return {
        "requests": [
            {
                **{k: v for k, v in r.items() if k != "prompt_token_ids"},
                "prompt_token_ids": describe_ids(r["prompt_token_ids"], first),
            }
            for r in reqs
        ],
        "workload_hash": trace.workload_hash(),
        "source_meta": {k: v for k, v in trace.source_meta().items() if k != "path"},
        "dataset": [
            {"max_tokens": s.max_tokens, "metadata": {k: v for k, v in s.metadata.items() if k != "prompt_token_ids"}}
            for s in served
        ],
    }


class MooncakeReaderGolden(unittest.TestCase):
    def test_one_reader_on_one_trace(self):
        rows = [json.loads(line) for line in MOONCAKE_TRACE.read_text().splitlines()]
        with tempfile.TemporaryDirectory() as td:
            no_hash_ids = Path(td) / "mooncake_no_hash_ids.jsonl"
            no_hash_ids.write_text(
                "".join(json.dumps({k: v for k, v in r.items() if k != "hash_ids"}) + "\n" for r in rows)
            )
            zero_out = Path(td) / "mooncake_zero_output.jsonl"
            zero_out.write_text(
                "".join(json.dumps({**r, "output_length": 0} if i == 4 else r) + "\n" for i, r in enumerate(rows))
            )
            record = {
                "with_hash_ids": read_all(self, MOONCAKE_TRACE),
                "without_hash_ids": read_all(self, no_hash_ids),
                "zero_output_length": read_all(self, zero_out),
            }
        self.assertEqual(record["zero_output_length"]["requests"][4]["output_length"], 0)
        check_json(self, "mooncake_readers", record)


HASH_TRACE = FIXTURES / "hash_trace.jsonl"


class WorkloadHashGolden(unittest.TestCase):
    def test_hash_identity_rules(self):
        with crossval_path(), mock.patch.dict(os.environ, {}, clear=True):
            base = fresh_import("quettabench.workloads.base")
            live = fresh_import("quettabench.workloads.live")
            arrival = fresh_import("quettabench.workloads.arrival")

            def load_trace(path):
                return [r.to_dict() for r in live.RealTrace.from_jsonl(path).requests()]

            reqs = load_trace(str(HASH_TRACE))
            paced = base.workload_hash(reqs)
            completion_reversed = base.workload_hash(list(reversed(reqs)))
            completion_rotated = base.workload_hash(reqs[1:] + reqs[:1])

            altered = copy.deepcopy(reqs)
            altered[1]["prompt_token_ids"][0] += 1
            altered_hash = base.workload_hash(altered)

            injected = copy.deepcopy(reqs)
            schedule = arrival.parse_arrival("poisson:2.0", len(injected))
            for r, t in zip(injected, schedule):
                r["arrival_ts"] = round(t, 6)
            injected_hash = base.workload_hash(injected)

            lines = HASH_TRACE.read_text().splitlines()
            with tempfile.TemporaryDirectory() as td:
                reordered_file = Path(td) / "reordered.jsonl"
                reordered_file.write_text("\n".join([lines[2], lines[0], lines[1]]) + "\n")
                reordered_lines = base.workload_hash(load_trace(str(reordered_file)))

            digests = [base.request_digest(r) for r in reqs]
            as_requests = base.workload_hash([base.Request(**r) for r in reqs])
            round_trip = [base.Request(**r).to_dict() for r in reqs]

        self.assertEqual(paced, completion_reversed)
        self.assertEqual(paced, completion_rotated)
        self.assertNotEqual(paced, altered_hash)
        self.assertNotEqual(paced, injected_hash)
        self.assertEqual(as_requests, paced)
        self.assertEqual(round_trip, reqs)
        check_json(
            self,
            "workload_hash",
            {
                "loaded_requests": reqs,
                "request_digests": digests,
                "paced": paced,
                "completion_order_reversed": completion_reversed,
                "completion_order_rotated": completion_rotated,
                "one_token_altered": altered_hash,
                "injected_poisson_2.0": {"arrival_ts": [r["arrival_ts"] for r in injected], "hash": injected_hash},
                "file_lines_reordered": reordered_lines,
            },
        )


if __name__ == "__main__":
    unittest.main()
