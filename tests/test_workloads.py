"""Workload package: traces, arrivals, grid prompts, sessions."""

import json
import tempfile
import unittest
from pathlib import Path

from quettabench.workloads.arrival import (
    make_arrival_times, parse_arrival, poisson_arrivals, ramp_arrivals, rebase_arrivals)
from quettabench.workloads.grid import GridTrace, clone_ids
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


class GridTraceTest(unittest.TestCase):
    CELLS = [(64, 2), (128, 3), (1000, 1)]
    STEPS = [10, 20, 40]

    def test_cells_over_maxlen_carry_no_requests(self):
        grid = GridTrace(self.CELLS, self.STEPS, "distinct", maxlen=1040)
        self.assertEqual(grid.cells_over_maxlen, [(1000, 1)])  # 1000 + 40 + 2 > 1040
        reqs = grid.requests()
        self.assertEqual(len(reqs), (2 + 3) * len(self.STEPS))
        self.assertEqual(sorted({r.output_length for r in reqs}), self.STEPS)
        self.assertEqual({r.prompt_len for r in reqs}, {64, 128})

    def test_prompt_modes(self):
        distinct, _ = GridTrace(self.CELLS, self.STEPS, "distinct", 4096).cell_prompts(64, 2)
        self.assertEqual(distinct[1][:3], [11, 12, 13])
        clone, _ = GridTrace(self.CELLS, self.STEPS, "clone", 4096).cell_prompts(64, 2)
        self.assertEqual(clone, [clone_ids(64)] * 2)
        with tempfile.TemporaryDirectory() as td:
            trace = RealTrace.from_jsonl(write_jsonl(td, [{"prompt_token_ids": [7] * 80}]))
        ids, cycled = GridTrace(self.CELLS, self.STEPS, "trace", 4096, trace=trace).cell_prompts(64, 2)
        self.assertTrue(cycled)
        self.assertEqual(ids, [[7] * 64] * 2)

    def test_corpus_windows_need_a_file(self):
        with self.assertRaises(ValueError):
            GridTrace(self.CELLS, self.STEPS, "corpus", 4096).cell_prompts(64, 1)
        with tempfile.TemporaryDirectory() as td:
            corpus = Path(td) / "c.txt"
            corpus.write_text("abcdef")
            grid = GridTrace(self.CELLS, self.STEPS, "corpus", 4096, corpus_path=corpus,
                             tokenizer=lambda: (lambda text: {"input_ids": [ord(c) for c in text]}))
            ids, _ = grid.cell_prompts(8, 1)
        self.assertEqual(ids, [[97, 98, 99, 100, 101, 102, 97, 98]])

    def test_grid_hash_covers_every_request(self):
        grid = GridTrace(self.CELLS, self.STEPS, "synth", 4096)
        self.assertEqual(grid.workload_hash(), GridTrace(self.CELLS, self.STEPS, "synth", 4096).workload_hash())
        self.assertNotEqual(grid.workload_hash(),
                            GridTrace(self.CELLS, [10, 20, 41], "synth", 4096).workload_hash())


if __name__ == "__main__":
    unittest.main()
