"""A grid run's files: run.json's steps section is fit_run over the rows,
the same numbers RESULT prints, and requests.jsonl holds one row per
request of every batch, told apart by cell."""

import json
import tempfile
import unittest
from pathlib import Path

from tests.golden._harness import grid_argv, run_grid_cli


def grid_run(*sets):
    with tempfile.TemporaryDirectory() as td:
        run = run_grid_cli(grid_argv("llama", *sets, "--results-dir", td), {})
        doc = json.loads((Path(td) / "llama" / "run.json").read_text())
        rows = [json.loads(line) for line in
                (Path(td) / "llama" / "requests.jsonl").read_text().splitlines()]
    return run, doc, rows


class GridRunFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli, cls.doc, cls.rows = grid_run()

    def test_steps_are_the_result_lines(self):
        results = [line for line in self.cli["stdout"] if line.startswith("RESULT")]
        self.assertEqual(len(self.doc["steps"]), len(results))
        for s, line in zip(self.doc["steps"], results):
            self.assertIn(f"ctx={s['ctx']} bs={s['bs']} ms_per_step={s['ms_per_step']:.3f} "
                          f"ms_at_eval={s['ms_at_eval']:.3f}", line)
            self.assertIn(f"r2={s['r2']:.5f}", line)
        self.assertEqual(self.doc["run_status"], {"status": "ok", "reason": None})
        self.assertTrue(self.doc["summary"]["grid"]["done"])

    def test_rows_cover_every_batch(self):
        batches = {}
        for r in self.rows:
            c = r["cell"]
            key = (c["ctx"], c["bs"], c["step_point"], c["phase"], c["attempt"])
            batches.setdefault(key, []).append(r)
        # warmup has no cell; every other generate() is one batch of bs rows
        self.assertEqual(len(batches), self.cli["generate_calls"] - 1)
        for (ctx, bs, step, phase, _), rows in batches.items():
            self.assertEqual(len(rows), bs)
            self.assertEqual(len({r["batch_wall_s"] for r in rows}), 1)
            self.assertEqual({r["output_tokens"] for r in rows}, {step})
            self.assertEqual(all(r.get("excluded_from_summary", False) for r in rows),
                             phase != "fit")

    def test_dump_tokens_writes_text_rows(self):
        _, _, rows = grid_run("--set", "mode.grid=smoke", "--set", "mode.returns=text")
        text = [r for r in rows if r["cell"]["phase"] == "text"]
        self.assertTrue(text)
        self.assertEqual({r["cell"]["step_point"] for r in text}, {64})
        self.assertEqual(text[0]["token_ids"], list(range(64)))


if __name__ == "__main__":
    unittest.main()
