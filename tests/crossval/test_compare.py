"""crossval/compare.py: the withhold reasons behind each golden row, and the
workload_hash gate."""

import calendar
import time
import unittest

from quettabench.crossval import compare, reader
from tests.golden._harness import FIXTURES

NOW = calendar.timegm(time.strptime("2026-09-21T14:13:20Z", "%Y-%m-%dT%H:%M:%SZ")) + 2.5 * 86400


def golden_pair(log="bench_loop.log", **kw):
    return compare.pair(reader.read_bench_log(FIXTURES / log),
                        reader.read_baseline(FIXTURES / "baseline_llama.json"), now=NOW, **kw)


def by_cell(report):
    return {(r["ctx"], r["bs"]): r for r in report.rows}


class GoldenRows(unittest.TestCase):
    def test_loop_log_reasons(self):
        report = golden_pair()
        rows = by_cell(report)
        self.assertEqual(rows[(100, 1)]["withheld"], ["kv"])         # batch_bench kv "?"
        self.assertEqual(rows[(4096, 8)]["withheld"], ["kv"])        # bf16 against fp16
        # LOW R2, RAW and EAGER are flags only: the ratio stays
        for cell, flag in (((1024, 16), "LOW R2"), ((4096, 64), "RAW"), ((100, 8), "EAGER")):
            self.assertIn(flag, rows[cell]["flags"])
            self.assertEqual(rows[cell]["withheld"], [])
            self.assertIsNotNone(rows[cell]["ratio"])
        self.assertEqual(report.ours_only, [(2048, 4, 2), (8192, 1, 1)])
        self.assertEqual(report.tps_without_theirs, [2])
        self.assertAlmostEqual(report.age_days, 2.5)

    def test_criterion_log_withholds_on_method(self):
        report = golden_pair("bench_criterion.log")
        self.assertFalse(report.comparable)
        self.assertTrue(all("method" in r["withheld"] for r in report.rows))
        forced = golden_pair("bench_criterion.log", allow_method_mismatch=True)
        self.assertTrue(all(r["ratio"] is not None for r in forced.rows))


class WorkloadGate(unittest.TestCase):
    def sides(self, ours_hash, theirs_hash):
        ours = reader.read_bench_log(FIXTURES / "bench_loop.log")
        theirs = reader.read_baseline(FIXTURES / "baseline_llama.json")
        ours.header["workload_hash"], theirs.header["workload_hash"] = ours_hash, theirs_hash
        return ours, theirs

    def test_different_hashes_withhold_every_row(self):
        report = compare.pair(*self.sides("aaaa", "bbbb"), now=NOW)
        self.assertTrue(report.workload_mismatch)
        self.assertTrue(all("workload" in r["withheld"] and r["ratio"] is None
                            for r in report.rows))
        self.assertIn("WORKLOAD MISMATCH engine=aaaa vllm=bbbb", compare.render(report))

    def test_one_missing_hash_joins_on_cells_alone(self):
        for hashes in (("aaaa", None), (None, "bbbb"), ("aaaa", "aaaa")):
            report = compare.pair(*self.sides(*hashes), now=NOW)
            self.assertFalse(report.workload_mismatch)
            self.assertEqual(compare.render(report), compare.render(golden_pair()))


if __name__ == "__main__":
    unittest.main()
