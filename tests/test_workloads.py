"""Workload package: traces, arrivals, grid prompts, sessions."""

import json
import tempfile
import unittest
from pathlib import Path

from quettabench.workloads.arrival import (
    make_arrival_times, parse_arrival, poisson_arrivals, ramp_arrivals, rebase_arrivals)
from quettabench.workloads.live import RealTrace


def write_jsonl(td, rows):
    p = Path(td) / "t.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


class RealTraceTest(unittest.TestCase):
    def test_supports_spec_only_with_real_assistant_tokens(self):
        with tempfile.TemporaryDirectory() as td:
            plain = RealTrace.from_jsonl(write_jsonl(td, [{"prompt_token_ids": [1, 2]}]))
            self.assertFalse(plain.supports_spec)
            self.assertEqual(plain.source_meta()["text_mode"], "synthetic")
            real = RealTrace.from_jsonl(write_jsonl(td, [
                {"prompt_token_ids": [1, 2]},
                {"prompt_token_ids": [3], "output_token_ids": [4, 5]}]))
            self.assertTrue(real.supports_spec)
            self.assertEqual(real.source_meta()["text_mode"], "real")

    def test_empty_file_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                RealTrace.from_jsonl(write_jsonl(td, []))


class ArrivalTest(unittest.TestCase):
    def test_serve_pair_and_spec_string_give_one_schedule(self):
        for seed in (0, 7, 42):
            self.assertEqual(make_arrival_times("poisson", 50, target_rate=2.0, seed=seed),
                             parse_arrival("poisson:2.0", 50, seed))
            self.assertEqual(make_arrival_times("poisson", 50, target_rate=2.0, seed=seed),
                             poisson_arrivals(50, 2.0, seed))
            self.assertEqual(make_arrival_times("ramp", 50, target_rate=3.0, seed=seed),
                             ramp_arrivals(50, 3.0 * 0.1, 3.0, seed))
            self.assertEqual(make_arrival_times("steady", 5, seed=seed), parse_arrival("steady", 5, seed))
        self.assertEqual(parse_arrival("steady", 3), [0.0, 0.0, 0.0])

    def test_unknown_spec_is_refused(self):
        with self.assertRaises(ValueError):
            parse_arrival("gamma:2", 4)
        with self.assertRaises(ValueError):
            make_arrival_times("gamma", 4)

    def test_rebase_shifts_by_the_earliest_and_a_missing_time_is_zero(self):
        self.assertEqual(rebase_arrivals([5.0, 6.0, 9.0], speed=2.0), [0.0, 0.5, 2.0])
        self.assertEqual(rebase_arrivals([5.0, None, 7.0], speed=1.0), [5.0, 0.0, 7.0])
        self.assertEqual(rebase_arrivals([None, 3.0], speed=1.0), [0.0, 3.0])


if __name__ == "__main__":
    unittest.main()
