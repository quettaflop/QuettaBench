"""crossval.py: the readers, the pairing and withhold rules, the agreement rows and main."""

import calendar
import contextlib
import io
import json
import math
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from quettabench import crossval
from quettabench.engines.drivers import BenchDriver
from quettabench.metrics.results import RunWriter
from quettabench.metrics.step_fit import fit_run
from tests.contracts.helpers import bench_config, bench_run_dir
from tests.golden._harness import FIXTURES, grid_argv, run_grid_cli


def text_run(out):
    """A stub llama grid run with mode.returns=text."""
    run = run_grid_cli(grid_argv("llama", "--set", "mode.returns=text", "--set", f"run_name={out.name}",
                                 "--results-dir", str(out.parent)), {})
    assert run["exit"] is None, run["exit"]
    return crossval.read_rows(out)


def logit_row(i, argmax, gap, logprob):
    return {"request_index": i, "logits": {"argmax": argmax, "top2_gap": gap, "logprob": logprob}}


REF = [logit_row(0, [5, 6, 7, 8], [2.0, 0.05, 1.0, 3.0], [-0.1, -0.2, -0.3, -0.4])]


class Greedy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as td:
            cls.rows = text_run(Path(td) / "a")

    def test_same_run_agrees_everywhere(self):
        rows = crossval.greedy(self.rows, self.rows)
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
        rows = {r["request"]: r for r in crossval.greedy(self.rows, theirs)
                if r["kind"] == "greedy"}
        cell = first["cell"]
        hit = rows[f"{cell['ctx']}x{cell['bs']} #0"]
        self.assertEqual((hit["prefix"], hit["of"]), (10, 64))
        self.assertIn("AGREE 10/64 tokens ( 15.6%)", "\n".join(
            crossval.render_agreement(crossval.greedy(self.rows, theirs))))

    def test_runs_without_returns_give_no_rows(self):
        self.assertEqual(crossval.compare_runs([{"cell": None}], [{"cell": None}]), [])


class Logit(unittest.TestCase):
    def test_identical_logits_pass(self):
        rows = crossval.logit(REF, REF)
        self.assertEqual(rows[-1]["verdict"], "PASS")
        self.assertEqual(rows[-1]["delta_max"], 0.0)

    def test_near_tie_flip_passes_and_hard_flip_fails(self):
        near = [logit_row(0, [5, 9, 7, 8], [0, 0, 0, 0], [-0.1, -0.25, -0.3, -0.4])]
        rows = crossval.logit(near, REF)
        self.assertEqual((rows[-1]["near_tie_flips"], rows[-1]["verdict"]), (1, "PASS"))
        hard = [logit_row(0, [5, 6, 9, 8], [0, 0, 0, 0], [-0.1, -0.2, -0.3, -0.9])]
        rows = crossval.logit(hard, REF)
        self.assertEqual(rows[-1]["verdict"], "FAIL")
        self.assertAlmostEqual(rows[-1]["delta_max"], 0.5)
        text = crossval.render_agreement(rows)
        self.assertEqual(text[0], "DISAGREE req 0 pos=2 reference_gap=1.0000 reference=7 ours=9")
        self.assertEqual(text[-1], "FIDELITY verdict=FAIL hard_disagreements=1")

    def test_one_bf16_step_is_a_near_tie(self):
        ref = [logit_row(0, [5, 6], [0.125, 0.126], [-0.1, -0.2])]
        rows = crossval.logit([logit_row(0, [9, 6], [0, 0], [-0.1, -0.2])], ref)
        self.assertEqual((rows[-1]["near_tie_flips"], rows[-1]["verdict"]), (1, "PASS"))
        rows = crossval.logit([logit_row(0, [5, 9], [0, 0], [-0.1, -0.2])], ref)
        self.assertEqual(rows[-1]["verdict"], "FAIL")


class Band(unittest.TestCase):
    ONE_FLIP = [logit_row(0, [5, 9, 7, 8], [0, 0, 0, 0], [-0.1, -0.2, -0.3, -0.4])]
    TWO_FLIPS = [logit_row(0, [1, 9, 7, 8], [0, 0, 0, 0], [-0.1, -0.2, -0.3, -0.4])]

    def test_inside_the_worst_floor_run_passes(self):
        row = crossval.band(self.ONE_FLIP, REF, [REF, self.TWO_FLIPS])
        self.assertEqual((row["flips"], row["floor_flips"], row["verdict"]), (1, [0, 2], "PASS"))
        self.assertEqual(crossval.render_agreement([row]),
                         ["BAND verdict=PASS flips=1 floor=2 floor_runs=0,2 positions=4"])

    def test_above_the_floor_fails(self):
        row = crossval.band(self.TWO_FLIPS, REF, [self.ONE_FLIP])
        self.assertEqual(row["verdict"], "FAIL")

    def test_different_position_counts_are_withheld(self):
        short = [logit_row(0, [5, 6], [0, 0], [-0.1, -0.2])]
        self.assertEqual(crossval.band(self.ONE_FLIP, REF, [short])["verdict"], "WITHHELD")


NOW = calendar.timegm(time.strptime("2026-09-21T14:13:20Z", "%Y-%m-%dT%H:%M:%SZ")) + 2.5 * 86400


def golden_pair(log="bench_loop.log", **kw):
    return crossval.pair(crossval.read_bench_log(FIXTURES / log),
                        crossval.read_baseline(FIXTURES / "baseline_llama.json"), now=NOW, **kw)


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


class StaleBaseline(unittest.TestCase):
    def test_the_warning_names_a_justfile_recipe(self):
        report = golden_pair()
        report.age_days = crossval.STALE_DAYS + 1
        self.assertIn("re-run `just run configs/runs/<wl>.yaml`", crossval.render(report))
        justfile = (Path(__file__).resolve().parents[2] / "justfile").read_text()
        self.assertIn("\nrun cfg *args:", justfile)


class WorkloadGate(unittest.TestCase):
    def sides(self, ours_hash, theirs_hash):
        ours = crossval.read_bench_log(FIXTURES / "bench_loop.log")
        theirs = crossval.read_baseline(FIXTURES / "baseline_llama.json")
        ours.header["workload_hash"], theirs.header["workload_hash"] = ours_hash, theirs_hash
        return ours, theirs

    def test_different_hashes_withhold_every_row(self):
        report = crossval.pair(*self.sides("aaaa", "bbbb"), now=NOW)
        self.assertTrue(report.workload_mismatch)
        self.assertTrue(all("workload" in r["withheld"] and r["ratio"] is None
                            for r in report.rows))
        self.assertIn("WORKLOAD MISMATCH engine=aaaa vllm=bbbb", crossval.render(report))

    def test_one_missing_hash_joins_on_cells_alone(self):
        for hashes in (("aaaa", None), (None, "bbbb"), ("aaaa", "aaaa")):
            report = crossval.pair(*self.sides(*hashes), now=NOW)
            self.assertFalse(report.workload_mismatch)
            self.assertEqual(crossval.render(report), crossval.render(golden_pair()))


BASELINE = FIXTURES / "baseline_llama.json"


def run(argv, env=None):
    out = io.StringIO()
    with mock.patch.dict(os.environ, env or {}, clear=True), contextlib.redirect_stdout(out):
        crossval.main([str(a) for a in argv])
    return out.getvalue()


class Door(unittest.TestCase):
    def test_json_report(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "r.json"
            run([FIXTURES / "bench_loop.log", BASELINE, "--json", out])
            report = json.loads(out.read_text())
        self.assertEqual(report["theirs"]["header"]["engine"], "vllm")
        self.assertEqual(report["ours"]["header"]["method"], "loop")
        rows = {(r["ctx"], r["bs"]): r for r in report["rows"]}
        self.assertIsNone(rows[(4096, 8)]["ratio"])
        self.assertEqual(rows[(4096, 8)]["withheld"], ["kv"])
        self.assertEqual(report["ours_only"], [[2048, 4, 2], [8192, 1, 1]])

    def test_default_report_path(self):
        with tempfile.TemporaryDirectory() as td, \
                mock.patch.object(crossval, "XVAL_DIR", Path(td)):
            run([FIXTURES / "bench_loop.log", BASELINE])
            self.assertTrue((Path(td) / "bench_loop_vs_baseline_llama.json").is_file())

    def test_method_mismatch_forced_by_flag_or_env(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [FIXTURES / "bench_criterion.log", BASELINE, "--json", Path(td) / "r.json"]
            self.assertIn("ratios are withheld", run(argv))
            for forced in (run(argv + ["--allow-method-mismatch"]),
                           run(argv, {"XVAL_ALLOW_METHOD_MISMATCH": "1"})):
                self.assertNotIn("ratios are withheld", forced)
                self.assertIn("1.04x", forced)

    def test_engine_facts_from_env(self):
        with tempfile.TemporaryDirectory() as td:
            text = run([FIXTURES / "bench_loop.log", BASELINE, "--json", Path(td) / "r.json"],
                       {"NCCL_ALGO": "Ring", "XVAL_QUANT": "nvfp4"})
        self.assertIn("NCCL MISMATCH engine=Ring/None vllm=auto/auto", text)
        self.assertIn("QUANT MISMATCH engine=nvfp4 vllm=float16; ratios withheld", text)

    def test_fidelity_or_band_fail_exits_1_after_the_json(self):
        for kind in ("logit_summary", "band"):
            with self.subTest(kind), tempfile.TemporaryDirectory() as td, \
                    mock.patch.object(crossval, "read_side"), \
                    mock.patch.object(crossval, "read_rows"), \
                    mock.patch.object(crossval, "render", return_value=""), \
                    mock.patch.object(crossval, "pair",
                                      return_value=mock.Mock(rows=[], agreement=[])), \
                    mock.patch.object(crossval, "report_json", return_value={}), \
                    mock.patch.object(crossval, "compare_runs",
                                      return_value=[{"kind": kind, "verdict": "FAIL"}]):
                with self.assertRaises(SystemExit) as e:
                    run([td, td, "--json", Path(td) / "r.json"])
                self.assertEqual(e.exception.code, 1)
                self.assertTrue((Path(td) / "r.json").is_file())

    def test_missing_input(self):
        with self.assertRaises(SystemExit) as e:
            run(["/nonexistent/run", BASELINE])
        self.assertEqual(e.exception.code, "no results at /nonexistent/run")


def side_by_cell(side):
    return {(r["ctx"], r["bs"], r["tp"]): r for r in side.rows}


class BenchLog(unittest.TestCase):
    def test_loop_log(self):
        side = crossval.read_bench_log(FIXTURES / "bench_loop.log")
        self.assertEqual(side.header["method"], "loop")
        self.assertEqual(side.notes, [
            "ignored 1 truncated-model batch_bench line(s) (layers != all)",
            "engine gdn kernel(s): Tiled"])
        cells = side_by_cell(side)
        self.assertEqual(len(cells), 8)
        self.assertEqual(cells[(100, 1, 1)]["group"], "batch_bench")
        self.assertEqual(cells[(100, 1, 1)]["kv"], "?")
        self.assertEqual(cells[(100, 8, 1)]["mode"], "eager")
        self.assertEqual(cells[(2048, 4, 2)]["ms_per_step"], 4.1)
        self.assertTrue(all(r["measured"] for r in side.rows))

    def test_criterion_log(self):
        side = crossval.read_bench_log(FIXTURES / "bench_criterion.log")
        self.assertEqual(side.header["method"], "criterion")
        cells = side_by_cell(side)
        # decode_graph outranks plain decode on the same cell
        self.assertEqual(cells[(100, 1, 1)]["group"], "decode_graph")
        self.assertEqual(cells[(1024, 1, 2)]["group"], "decode_tp2")
        self.assertIsNone(cells[(1024, 1, 1)]["kv"])

    def test_engine_facts_fill_the_header(self):
        side = crossval.read_bench_log(FIXTURES / "bench_loop.log",
                                     {"quant": "nvfp4", "nccl_algo": "Ring"})
        self.assertEqual((side.header["quant"], side.header["nccl_algo"]), ("nvfp4", "Ring"))


class Baseline(unittest.TestCase):
    def test_golden_baseline(self):
        side = crossval.read_baseline(FIXTURES / "baseline_llama.json")
        self.assertEqual(len(side.rows), 15)
        h = side.header
        self.assertEqual((h["engine"], h["version"], h["method"], h["timestamp"]),
                         ("vllm", "0.0.0+stub", "slope", "2026-09-21T14:13:20Z"))
        self.assertEqual((h["quant"], h["serving_style"], h["comm_bound_bs"]),
                         ("float16", "aggregated", None))
        raw = side_by_cell(side)[(4096, 64, 1)]
        self.assertIsNone(raw["ms_at_eval"])
        self.assertFalse(raw["measured"])

    def test_age(self):
        side = crossval.read_baseline(FIXTURES / "baseline_llama.json")
        self.assertTrue(math.isclose(crossval.age_days(side, 1790000000 + 86400), 1.0))


class RunDir(unittest.TestCase):
    def test_bench_run_reads_as_its_log(self):
        for log in ("bench_loop.log", "bench_criterion.log"):
            with self.subTest(log=log), tempfile.TemporaryDirectory() as td:
                run = crossval.read_run(bench_run_dir(Path(td) / "qs", FIXTURES / log))
                side = crossval.read_bench_log(FIXTURES / log)
                self.assertEqual(run.rows, side.rows)
                self.assertEqual(run.notes, side.notes)
                self.assertEqual(run.header["method"], side.header["method"])

    def test_writer_rows_read_as_hand_written_rows(self):
        def hand_written(out, log):
            out.mkdir(parents=True)
            shutil.copy(log, out / "bench.log")
            rows = list(BenchDriver(out / "bench.log").rows())
            with (out / "requests.jsonl").open("w") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            RunWriter(out, bench_config(out.name)).finish(None, steps=fit_run(rows, 0, 1, 1.0))
            return out

        for log in ("bench_loop.log", "bench_criterion.log"):
            # Clock and commit frozen in the writer's own module (the golden
            # harness re-imports quettabench): a commit mid-test changes bench_sha.
            with self.subTest(log=log), tempfile.TemporaryDirectory() as td, \
                    mock.patch.dict(RunWriter.finish.__globals__, {
                        "_now": lambda: "2026-10-09T00:00:00+00:00",
                        "bench_sha": lambda *a, **k: "0123abc"}):
                ours = crossval.read_run(bench_run_dir(Path(td) / "qs", FIXTURES / log))
                theirs = crossval.read_run(hand_written(Path(td) / "hand" / "qs", FIXTURES / log))
                self.assertEqual(ours, theirs)
                self.assertTrue(ours.rows)


if __name__ == "__main__":
    unittest.main()
