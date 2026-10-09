"""crossval/agreement.py: greedy rows from two grid runs that returned text,
logit rows and the FIDELITY verdict from rows that returned logits."""

import json
import tempfile
import unittest
from pathlib import Path

from quettabench.crossval import agreement, reader
from tests.golden._harness import run_grid_cli


def text_run(out):
    """A stub llama grid run with mode.returns=text (serve --grid --dump-tokens)."""
    run = run_grid_cli(["--grid", "--workload", "llama", "--model", "<MODEL>", "--dump-tokens",
                        "--results-dir", str(out)], {})
    assert run["exit"] is None, run["exit"]
    return reader.read_rows(out)


def logit_row(i, argmax, gap, logprob):
    return {"request_index": i, "logits": {"argmax": argmax, "top2_gap": gap, "logprob": logprob}}


REF = [logit_row(0, [5, 6, 7, 8], [2.0, 0.05, 1.0, 3.0], [-0.1, -0.2, -0.3, -0.4])]


class Greedy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as td:
            cls.rows = text_run(Path(td) / "a")

    def test_same_run_agrees_everywhere(self):
        rows = agreement.greedy(self.rows, self.rows)
        per_request = [r for r in rows if r["kind"] == "greedy"]
        self.assertEqual(len(per_request), sum(1 for r in self.rows if r.get("token_ids")))
        self.assertTrue(all(r["prefix"] == r["of"] == 64 for r in per_request))
        summary = rows[-1]
        self.assertEqual((summary["kind"], summary["agree4plus"]),
                         ("greedy_summary", summary["prompts"]))

    def test_a_flipped_token_ends_the_prefix(self):
        theirs = json.loads(json.dumps(self.rows))
        first = next(r for r in theirs if r.get("token_ids"))
        first["token_ids"][10] += 1
        rows = {r["request"]: r for r in agreement.greedy(self.rows, theirs)
                if r["kind"] == "greedy"}
        cell = first["cell"]
        hit = rows[f"{cell['ctx']}x{cell['bs']} #0"]
        self.assertEqual((hit["prefix"], hit["of"]), (10, 64))
        self.assertIn("AGREE 10/64 tokens ( 15.6%)", "\n".join(
            agreement.render(agreement.greedy(self.rows, theirs))))

    def test_runs_without_returns_give_no_rows(self):
        self.assertEqual(agreement.compare_runs([{"cell": None}], [{"cell": None}]), [])


class Logit(unittest.TestCase):
    def test_identical_logits_pass(self):
        rows = agreement.logit(REF, REF)
        self.assertEqual(rows[-1]["verdict"], "PASS")
        self.assertEqual(rows[-1]["delta_max"], 0.0)

    def test_near_tie_flip_passes_and_hard_flip_fails(self):
        near = [logit_row(0, [5, 9, 7, 8], [0, 0, 0, 0], [-0.1, -0.25, -0.3, -0.4])]
        rows = agreement.logit(near, REF)
        self.assertEqual((rows[-1]["near_tie_flips"], rows[-1]["verdict"]), (1, "PASS"))
        hard = [logit_row(0, [5, 6, 9, 8], [0, 0, 0, 0], [-0.1, -0.2, -0.3, -0.9])]
        rows = agreement.logit(hard, REF)
        self.assertEqual(rows[-1]["verdict"], "FAIL")
        self.assertAlmostEqual(rows[-1]["delta_max"], 0.5)
        text = agreement.render(rows)
        self.assertEqual(text[0], "DISAGREE req 0 pos=2 reference_gap=1.0000 reference=7 ours=9")
        self.assertEqual(text[-1], "FIDELITY verdict=FAIL hard_disagreements=1")


if __name__ == "__main__":
    unittest.main()
