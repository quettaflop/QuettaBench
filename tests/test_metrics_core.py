"""Unit tests for quettabench.metrics.core: percentile methods.

Expected values are the table in tests/golden/REPORT.md (STOP finding,
item 4), one column per method.
"""

import unittest

from quettabench.metrics.core import pcts, percentile

SEVEN = [3.0, 9.0, 1.0, 7.0, 5.0, 11.0, 13.0]  # unsorted on purpose
TIES = [1.0, 2.0, 2.0, 2.0, 3.0]
TWO = [10.0, 20.0]
SINGLE = [42.0]
ARANGE = [float(i) for i in range(1000)]

# (array, {p: (nearest, linear)})
TABLE = [
    (SINGLE, {50: (42.0, 42.0), 90: (42.0, 42.0), 95: (42.0, 42.0), 99: (42.0, 42.0)}),
    (TWO, {50: (20.0, 15.0), 90: (20.0, 19.0), 95: (20.0, 19.5), 99: (20.0, 19.900000000000002)}),
    (TIES, {50: (2.0, 2.0), 90: (3.0, 2.6), 95: (3.0, 2.8), 99: (3.0, 2.96)}),
    (SEVEN, {50: (7.0, 7.0), 90: (13.0, 11.8), 95: (13.0, 12.399999999999999),
             99: (13.0, 12.879999999999999)}),
    (ARANGE, {50: (500.0, 499.5), 90: (900.0, 899.1), 95: (950.0, 949.05), 99: (990.0, 989.01)}),
]


class PercentileTest(unittest.TestCase):
    def test_report_table_both_methods(self):
        for values, by_p in TABLE:
            for p, (nearest, linear) in by_p.items():
                with self.subTest(n=len(values), p=p):
                    self.assertEqual(percentile(values, p, "nearest"), nearest)
                    self.assertEqual(percentile(values, p, "linear"), linear)

    def test_pcts_maps_the_four_quantiles(self):
        for values, by_p in TABLE:
            for method, col in (("nearest", 0), ("linear", 1)):
                self.assertEqual(pcts(values, method),
                                 {f"p{p}": v[col] for p, v in by_p.items()})

    def test_empty(self):
        self.assertEqual(percentile([], 99, "linear"), 0.0)
        with self.assertRaisesRegex(ValueError, "no values"):
            percentile([], 50, "nearest")
        with self.assertRaises(ValueError):
            pcts([], "nearest")

    def test_input_order_does_not_matter(self):
        self.assertEqual(percentile(SEVEN, 90, "nearest"), percentile(sorted(SEVEN), 90, "nearest"))
        self.assertEqual(percentile(SEVEN, 90, "linear"), percentile(sorted(SEVEN), 90, "linear"))

    def test_unknown_method_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown percentile method"):
            percentile([1.0], 50, "midpoint")


if __name__ == "__main__":
    unittest.main()
