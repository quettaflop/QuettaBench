"""Golden: workload_hash on a 3-request fixture trace, read as serve reads a replay JSONL.

Pins the hash values and the identity rules: completion order does not
matter, any altered request does, and injected arrivals (stamped the way
trace_serve.main stamps them) never share a hash with the trace's own pacing.
The file line index is the request's `order`, so the same requests written
in a different line order are a different workload.
"""

import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.golden._harness import FIXTURES, check_json, crossval_path, fresh_import

TRACE = FIXTURES / "hash_trace.jsonl"


class WorkloadHashGolden(unittest.TestCase):
    def test_hash_identity_rules(self):
        with crossval_path(), mock.patch.dict(os.environ, {}, clear=True):
            base = fresh_import("quettabench.workloads.base")
            live = fresh_import("quettabench.workloads.live")
            arrival = fresh_import("quettabench.workloads.arrival")

            def load_trace(path):
                return [r.to_dict() for r in live.RealTrace.from_jsonl(path).requests()]

            reqs = load_trace(str(TRACE))
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

            lines = TRACE.read_text().splitlines()
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
        check_json(self, "workload_hash", {
            "loaded_requests": reqs,
            "request_digests": digests,
            "paced": paced,
            "completion_order_reversed": completion_reversed,
            "completion_order_rotated": completion_rotated,
            "one_token_altered": altered_hash,
            "injected_poisson_2.0": {"arrival_ts": [r["arrival_ts"] for r in injected],
                                     "hash": injected_hash},
            "file_lines_reordered": reordered_lines,
        })


if __name__ == "__main__":
    unittest.main()
