"""Unit tests for quettabench.metrics.step_fit on hand-computed cases."""

import unittest

from quettabench.metrics.step_fit import fit, ms_at_eval


class FitTest(unittest.TestCase):
    def test_exact_line(self):
        # y = 0.5 + 0.01 x
        slope, inter, r2 = fit([64, 128, 256], [1.14, 1.78, 3.06])
        self.assertAlmostEqual(slope, 0.01)
        self.assertAlmostEqual(inter, 0.5)
        self.assertAlmostEqual(r2, 1.0)

    def test_noisy_three_points(self):
        # xs mean 2, ys mean 2; sxx = 2, sxy = 1 -> slope 0.5, intercept 1.
        # residuals (-0.5, 1, -0.5): ss_res 1.5, ss_tot 2 -> r2 0.25.
        self.assertEqual(fit([1, 2, 3], [1, 3, 2]), (0.5, 1.0, 0.25))

    def test_flat_ys_gives_r2_zero(self):
        self.assertEqual(fit([1, 2, 3], [5.0, 5.0, 5.0]), (0.0, 5.0, 0.0))


class MsAtEvalTest(unittest.TestCase):
    def test_eval_at_centroid_is_the_slope(self):
        self.assertEqual(ms_at_eval(10.0, 0.4, 128, 128, 4096), 10.0)

    def test_no_attention_share_no_shift(self):
        self.assertEqual(ms_at_eval(10.0, 0.0, 512, 128, 4096), 10.0)

    def test_shift_scales_with_attn_and_distance(self):
        # 10 * (1 + 0.5 * (1000 - 100) / (800 + 100)) = 10 * 1.5
        self.assertEqual(ms_at_eval(10.0, 0.5, 1000, 100, 800), 15.0)
        # eval before the centroid shrinks it: 10 * (1 - 0.5 * 100 / 1000) = 9.5
        self.assertEqual(ms_at_eval(10.0, 0.5, 0, 100, 900), 9.5)


if __name__ == "__main__":
    unittest.main()
