"""Golden: the two percentile methods on fixed arrays.

method="nearest" (crossval SERVESUM, formerly trace_serve._pct / _pcts)
indexes the sorted list at int(q * n), clamped: a nearest-rank floor.
method="linear" (src runner summaries, formerly metrics._percentile)
interpolates linearly between ranks over n - 1. They disagree on most
inputs; both are pinned as they are and the disagreement table is in
REPORT.md. Neither is preferred here.
"""

import unittest

from quettabench.metrics.core import pcts, percentile
from tests.golden._harness import check_json

QUANTILES = (50, 90, 95, 99)
ARRAYS = {
    "empty": [],
    "single": [42.0],
    "two": [10.0, 20.0],
    "ties": [1.0, 2.0, 2.0, 2.0, 3.0],
    "seven": [3.0, 9.0, 1.0, 7.0, 5.0, 11.0, 13.0],
    "arange_1000": [float(i) for i in range(1000)],
}


def _call(fn, *args):
    try:
        return fn(*args)
    except Exception as e:  # the empty-array failure mode is part of the pin
        return {"error": type(e).__name__, "message": str(e)}


class PercentileGolden(unittest.TestCase):
    def test_both_implementations_on_fixed_arrays(self):
        record = {}
        for name, data in ARRAYS.items():
            record[name] = {
                "crossval_pct": {f"p{q}": _call(percentile, data, q, "nearest")
                                 for q in QUANTILES},
                "crossval_pcts": _call(pcts, data, "nearest"),
                "metrics_percentile": {f"p{q}": _call(percentile, data, q, "linear")
                                       for q in QUANTILES},
            }
        check_json(self, "percentiles", record)


if __name__ == "__main__":
    unittest.main()
