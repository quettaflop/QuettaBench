"""The committed HTTP-vs-in-process slope gate: every cell inside its rule."""

import json
import unittest
from pathlib import Path

GATE = json.loads((Path(__file__).with_name("slope_gate.json")).read_text())


class SlopeGate(unittest.TestCase):
    def test_every_cell_passes(self):
        rule, cells = GATE["gate"], GATE["cells"]
        self.assertEqual(len(cells), 15)
        for c in cells:
            with self.subTest(ctx=c["ctx"], bs=c["bs"]):
                self.assertGreaterEqual(min(c["r2_http"], c["r2_inproc"]), rule["r2_min"])
                self.assertAlmostEqual(c["delta_ms"], c["ms_per_step_http"] - c["ms_per_step_inproc"], places=6)
                self.assertLessEqual(abs(c["delta_ms"]), rule["abs_delta_ms_max"])


if __name__ == "__main__":
    unittest.main()
