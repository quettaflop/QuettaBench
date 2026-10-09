"""Workload package: traces, arrivals, grid prompts, sessions."""

import json
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
