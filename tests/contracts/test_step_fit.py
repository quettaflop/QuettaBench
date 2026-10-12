"""metrics/step_fit.py on hand-computed cases."""

import unittest

from quettabench.metrics.core import RequestResult
from quettabench.metrics.step_fit import fit, fit_run, ms_at_eval

STEPS = [60, 230, 400]
GIB = 1 << 30


def batch(ctx, bs, step, wall, phase="fit", attempt=0, tp=1):
    """bs rows of one InProcessDriver batch."""
    cell = {"ctx": ctx, "bs": bs, "tp": tp, "step_point": step, "phase": phase,
            "attempt": attempt}
    return [RequestResult(success=True, e2el=wall, batch_wall_s=wall, cell=dict(cell))
            for _ in range(bs)]


def attempt_rows(ctx, bs, walls, attempt):
    return [r for step, wall in zip(STEPS, walls) for r in batch(ctx, bs, step, wall,
                                                                  attempt=attempt)]


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


class FitRunTest(unittest.TestCase):
    def test_entry_matches_the_direct_fit(self):
        walls = [0.28, 1.01, 1.73]
        rows = batch(4096, 2, 400, 9.0, phase="warm", attempt=None) + attempt_rows(4096, 2, walls, 0)
        (entry,) = fit_run(rows, eval_pos=50, kv_bytes=131072, weights_gib=15.0)
        slope, inter, r2 = fit(STEPS, walls)
        kv_gib = 4096 * 2 * 131072 / GIB
        attn = kv_gib / (kv_gib + 15.0)
        self.assertEqual(entry["ms_per_step"], slope * 1e3)
        self.assertEqual(entry["ms_at_eval"], ms_at_eval(slope * 1e3, attn, 50, 230.0, 4096))
        self.assertEqual((entry["r2"], entry["prefill_s"], entry["kv_gib"], entry["attn_share"]),
                         (r2, inter, kv_gib, attn))
        self.assertEqual((entry["raw"], entry["step_points"], entry["mean_steps"]),
                         (walls, STEPS, 230.0))
        self.assertEqual(entry["attempts"], 1)

    def test_best_r2_attempt_wins_earliest_on_tie(self):
        bent, straight = [0.35, 1.51, 2.06], [0.27, 0.96, 1.65]
        rows = (attempt_rows(1024, 1, bent, 0) + attempt_rows(1024, 1, straight, 1)
                + attempt_rows(1024, 1, [0.27, 0.96, 1.6500001], 2))
        (entry,) = fit_run(rows, 50, 131072, 15.0)
        self.assertEqual((entry["raw"], entry["attempts"]), (straight, 3))
        tie = attempt_rows(1024, 1, bent, 0) + attempt_rows(1024, 1, list(bent), 1)
        self.assertEqual(fit_run(tie, 50, 131072, 15.0)[0]["r2"], fit(STEPS, bent)[2])

    def test_cells_keep_first_seen_order(self):
        rows = attempt_rows(100, 8, [1, 2, 3.5], 0) + attempt_rows(100, 1, [1, 2, 3], 0)
        self.assertEqual([(e["ctx"], e["bs"]) for e in fit_run(rows, 50, 1, 1.0)],
                         [(100, 8), (100, 1)])

    def test_measured_rows_pass_through(self):
        row = {"cell": {"ctx": 1024, "bs": 4, "tp": 2, "step_point": None},
               "measured_ms_per_step": 3.5, "group": "decode_loop", "kv": "fp16"}
        self.assertEqual(fit_run([row], 50, 1, 1.0),
                         [{"ctx": 1024, "bs": 4, "tp": 2, "ms_per_step": 3.5, "measured": True}])

    def test_rows_without_a_cell_or_fit_phase_are_ignored(self):
        rows = [RequestResult(success=True, e2el=1.0)] + batch(100, 1, 64, 0.5, phase="text",
                                                                 attempt=None)
        self.assertEqual(fit_run(rows, 50, 1, 1.0), [])


if __name__ == "__main__":
    unittest.main()
