"""Golden: the two percentile implementations on fixed arrays.

trace_serve._pct / _pcts (crossval SERVESUM) index the sorted list at
int(q * n), clamped: a nearest-rank floor. metrics._percentile (src runner
summaries) interpolates linearly between ranks over n - 1. They disagree
on most inputs; both are pinned as they are and the disagreement table is in
REPORT.md. Neither is preferred here.
"""

import os
import unittest
from unittest import mock

from quettabench.metrics.core import _percentile
from tests.golden._harness import check_json, crossval_path, fresh_import

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
        with crossval_path(), mock.patch.dict(os.environ, {}, clear=True):
            trace_serve = fresh_import("legacy.trace_serve")
            record = {}
            for name, data in ARRAYS.items():
                ordered = sorted(data)  # _pct's callers always pass sorted values
                record[name] = {
                    "crossval_pct": {f"p{q}": _call(trace_serve._pct, ordered, q / 100)
                                     for q in QUANTILES},
                    "crossval_pcts": _call(trace_serve._pcts, ordered),
                    "metrics_percentile": {f"p{q}": _call(_percentile, data, q)
                                           for q in QUANTILES},
                }
        check_json(self, "percentiles", record)


if __name__ == "__main__":
    unittest.main()
