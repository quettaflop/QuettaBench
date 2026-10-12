"""profiling/native.py sources, and the frontend and kv_transfer sources' contracts."""

import argparse
import ast
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from quettabench.metrics import results
from quettabench.profiling import native
from tests.contracts import helpers

LABELS0 = (("engine", "0"), ("model_name", "m"))


class Parse(unittest.TestCase):
    def setUp(self):
        self.m = native.parse_prometheus(helpers.METRICS)

    def test_labels_are_a_sorted_tuple_key(self):
        self.assertEqual(self.m[("vllm:gpu_prefix_cache_hits_total", LABELS0)], 800.0)
        self.assertEqual(self.m[("vllm:num_requests_running", LABELS0)], 3.0)
        self.assertEqual(self.m["process_open_fds"], 42.0)

    def test_escaped_label_and_timestamp(self):
        self.assertEqual(self.m[("vllm:label_escape", (("path", 'a\\"b'),))], 1.0)

    def test_comments_and_odd_lines_are_skipped(self):
        self.assertEqual(len(self.m), 8)
        self.assertEqual(native.parse_prometheus("# only\n\nfoo bar\n"), {})

    def test_hit_rate_is_token_weighted_over_label_sets(self):
        # (800 + 400) hit tokens / (1000 + 600) queried tokens; _created is not a count
        self.assertEqual(native.prefix_cache_hit_rate(self.m), 0.75)
        self.assertEqual(native.counter_total(self.m, native.QUERIES), 1600.0)

    def test_unsuffixed_counters_and_absent_ones(self):
        m = native.parse_prometheus("vllm:gpu_prefix_cache_hits 3\nvllm:gpu_prefix_cache_queries 4\n")
        self.assertEqual(native.prefix_cache_hit_rate(m), 0.75)
        self.assertIsNone(native.prefix_cache_hit_rate({"vllm:gpu_prefix_cache_hits": 3.0}))
        self.assertIsNone(native.prefix_cache_hit_rate(
            {"vllm:gpu_prefix_cache_hits": 0.0, "vllm:gpu_prefix_cache_queries": 0.0}))


class Scrape(unittest.TestCase):
    def test_scrape_and_section(self):
        with helpers.serve() as (endpoint, _):
            section = native.metrics_section(native.scrape_metrics(endpoint))
        self.assertEqual(section["prefix_cache_hit_rate"], 0.75)
        self.assertEqual(section["samples"][
            'vllm:gpu_prefix_cache_hits_total{engine="0",model_name="m"}'], 800.0)
        self.assertEqual(section["samples"]["process_open_fds"], 42.0)


class SpecCounters(unittest.TestCase):
    def test_summed_over_engines(self):
        text = ('vllm:spec_decode_num_drafts_total{engine="0"} 10\n'
                'vllm:spec_decode_num_drafts_total{engine="1"} 5\n'
                'vllm:spec_decode_num_drafts_created{engine="0"} 1.7e9\n'
                'vllm:spec_decode_num_draft_tokens_total{engine="0"} 45\n'
                'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 20\n')
        with helpers.serve(text) as (endpoint, _):
            self.assertEqual(native.spec_decode_counters(endpoint),
                             {"accepted_tokens": 20.0, "draft_tokens": 45.0, "drafts": 15.0})

    def test_none_without_spec_decoding_or_metrics(self):
        with helpers.serve() as (endpoint, _):
            self.assertIsNone(native.spec_decode_counters(endpoint))
            self.assertIsNone(native.spec_decode_counters(endpoint + "/nowhere"))


class Traces(unittest.TestCase):
    def test_lists_what_the_arming_left(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(native.collect_armed_traces(tmp, "none"),
                             {"instrumentation": "none", "files": [], "missing": False})
            self.assertTrue(native.collect_armed_traces(tmp, "nsys")["missing"])
            (Path(tmp) / "nsys.nsys-rep").write_bytes(b"")
            self.assertEqual(native.collect_armed_traces(tmp, "nsys"),
                             {"instrumentation": "nsys", "files": ["nsys.nsys-rep"],
                              "missing": False})
            (Path(tmp) / "torch_trace.json").write_text("{}")
            self.assertEqual(native.collect_armed_traces(tmp, "torch")["files"],
                             ["torch_trace.json"])


def fake_nsys_export(db):
    """An nsys sqlite export: two decode steps of 1000 ns each on device 0,
    a 400 ns prefill-only range, and one memcpy."""
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE StringIds (id INTEGER, value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INT, end INT, deviceId INT, shortName INT);
        CREATE TABLE NVTX_EVENTS (start INT, end INT, text TEXT, textId INT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INT, end INT, deviceId INT);
        INSERT INTO StringIds VALUES (1, 'nvjet_gemm'), (2, 'one_shot_all_reduce'),
                                     (3, 'rms_norm_kernel');
        INSERT INTO NVTX_EVENTS VALUES (0, 400, 'prefill_only', NULL),
                                       (1000, 4000, 'prefill_decode', NULL);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
            (100, 300, 0, 1),
            (1000, 1200, 0, 1), (1500, 2100, 0, 1), (1600, 1800, 0, 2), (2200, 2400, 0, 3),
            (3000, 3200, 0, 3), (5000, 5100, 1, 1);
        INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (1300, 1350, 0);
    """)
    con.commit()
    con.close()


class Kernels(unittest.TestCase):
    def test_exports_once_then_tabulates_per_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "nsys.nsys-rep").write_bytes(b"")
            with mock.patch.object(native.subprocess, "run",
                                   side_effect=lambda argv, check: fake_nsys_export(
                                       Path(argv[argv.index("-o") + 1]))) as run:
                out = native.kernel_summary(tmp, 2, window="prefill_decode")
                native.kernel_summary(tmp, 2)
            run.assert_called_once_with(["nsys", "export", "--type", "sqlite", "-o",
                                         str(Path(tmp) / "nsys.partial.sqlite"),
                                         str(Path(tmp) / "nsys.nsys-rep")], check=True)
        self.assertEqual(out["launches_per_step"], 2.5)
        self.assertAlmostEqual(out["kernel_ms_per_step"], 0.0007)
        self.assertEqual([(f["family"], f["launches_per_step"]) for f in out["families"]],
                         [("GEMM / GEMV (library)", 1.0), ("act / norm (fused)", 1.0),
                          ("collective (TP)", 0.5)])
        self.assertEqual(out["kernels"][0], {"kernel": "nvjet_gemm", "launches_per_step": 1.0,
                                             "ms_per_step": 0.0004, "us_per_call": 0.4})
        self.assertEqual(out["memops"], [{"kind": "memcpy", "per_step": 0.5,
                                          "ms_per_step": 0.000025}])
        # 1400 ns of kernels covering 1200 ns of wall over a 3000 ns window
        (dev,) = out["devices"]
        self.assertAlmostEqual(dev["sum_ms_per_step"], 0.0007)
        self.assertAlmostEqual(dev["union_ms_per_step"], 0.0006)
        self.assertAlmostEqual(dev["busy_pct"], 100.0 * 1200 / 3000)
        self.assertEqual(dev["gaps_per_step"], 0.0)

    def test_a_stale_partial_export_is_removed_first(self):
        def export(argv, check):
            part = Path(argv[argv.index("-o") + 1])
            self.assertFalse(part.exists())  # nsys refuses an existing -o file
            fake_nsys_export(part)

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "nsys.nsys-rep").write_bytes(b"")
            (Path(tmp) / "nsys.partial.sqlite").write_bytes(b"half")
            with mock.patch.object(native.subprocess, "run", side_effect=export):
                out = native.kernel_summary(tmp, 2, window="prefill_decode")
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()),
                             ["nsys.nsys-rep", "nsys.sqlite"])
        self.assertEqual(out["launches_per_step"], 2.5)

    def test_subtract_takes_the_prefill_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_nsys_export(Path(tmp) / "nsys.sqlite")
            out = native.kernel_summary(tmp, 1, window="prefill_decode", subtract="prefill_only")
        gemm = next(k for k in out["kernels"] if k["kernel"] == "nvjet_gemm")
        self.assertEqual((gemm["launches_per_step"], gemm["ms_per_step"]), (1.0, 0.0006))
        self.assertIsNone(out["devices"])

    def test_without_a_trace_or_a_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "no armed nsys trace"):
                native.kernel_summary(tmp, 1)
            fake_nsys_export(Path(tmp) / "nsys.sqlite")
            with self.assertRaisesRegex(ValueError, "'decode' occurs 0 times"):
                native.kernel_summary(tmp, 1, window="decode")


def fake_probe(result):
    """A frontend module whose two sources share one parser and one result."""
    def parser():
        ap = argparse.ArgumentParser()
        for flag in ("--base-url", "--proxy-url", "--model", "--gpu-label"):
            ap.add_argument(flag)
        return ap
    return types.SimpleNamespace(frontend_parser=parser, frontend_probe=lambda a: result(a),
                                 pd_host_parser=parser, pd_host_probe=lambda a: result(a))


class LiveProbes(unittest.TestCase):
    def test_defaults_fill_only_missing_flags(self):
        self.assertEqual(native.with_defaults(["--model", "x"], {"--model": "m", "--tp": 2,
                                                                 "--skip": None}),
                         ["--model", "x", "--tp", "2"])

    def test_frontend_points_at_the_endpoint(self):
        fake = fake_probe(lambda a: {"base_url": a.base_url, "model": a.model, "gpu": a.gpu_label})
        with mock.patch.dict(sys.modules, {native.FRONTEND: fake}):
            got = native.serving_frontend("http://h:1", "m", ["--gpu-label", "H200"])
        self.assertEqual(got, {"base_url": "http://h:1/v1", "model": "m", "gpu": "H200"})

    def test_pd_host_rows_and_argv_wins(self):
        fake = fake_probe(lambda a: [{"proxy": a.proxy_url, "model": a.model}])
        with mock.patch.dict(sys.modules, {native.FRONTEND: fake}):
            got = native.pd_host("http://h:1", "m", ["--model", "served"])
        self.assertEqual(got, {"rows": [{"proxy": "http://h:1", "model": "served"}]})

    def test_kv_transfer_reads_the_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "H200_nixl.csv"
            path.write_text("gpu,layout,total_ms\nH200,contig,1.5\n")
            self.assertEqual(native.kv_transfer(path),
                             {"csv": str(path), "rows": [{"gpu": "H200", "layout": "contig",
                                                          "total_ms": "1.5"}]})

    def test_moved_probes_keep_a_runnable_main_with_out(self):
        tree = ast.parse((Path(native.__file__).parent / "frontend.py").read_text())
        funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.assertIn("main", funcs)
        for name in ("frontend", "pd_host"):
            with self.subTest(name):
                flags = {a.value for n in ast.walk(funcs[f"{name}_parser"]) if isinstance(n, ast.Call)
                         and getattr(n.func, "attr", "") == "add_argument"
                         for a in n.args if isinstance(a, ast.Constant)}
                self.assertIn("--out", flags)
                self.assertTrue({f"{name}_probe", f"{name}_main"} <= set(funcs))

    def test_kv_transfer_records_a_probe_run(self):
        """kv_transfer takes the probes' output args and records its run
        through run_recorded, so its table lands in its run dir and only
        `qbench export` carries it into QuettaSim's kernel_data."""
        text = (Path(native.__file__).parent / "kv_transfer.py").read_text()
        calls = {(n.func.id, tuple(getattr(a, "id", getattr(a, "value", None)) for a in n.args))
                 for n in ast.walk(ast.parse(text))
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertIn(("add_output_args", ("ap",)), calls)
        self.assertIn(("out_root", ("a", "kv_transfer")), calls)
        self.assertIn(("run_recorded", ("main", "kv_transfer")), calls)
        self.assertNotIn("data/kernel_data", text)

    def test_kv_transfer_fails_its_run_on_a_hang_or_sigterm(self):
        """The gloo group times out after five minutes, and SIGTERM raises
        SystemExit(143), so run_recorded records the run as failed."""
        tree = ast.parse((Path(native.__file__).parent / "kv_transfer.py").read_text())
        (init,) = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                   and getattr(n.func, "attr", "") == "init_process_group"]
        self.assertEqual(ast.unparse(next(k.value for k in init.keywords if k.arg == "timeout")),
                         "timedelta(minutes=5)")
        calls = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and getattr(n.func, "attr", "") == "signal"]
        self.assertEqual(calls, ["signal.signal(signal.SIGTERM, _on_sigterm)"])
        (handler,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_on_sigterm"]
        namespace = {}
        exec(compile(ast.Module([handler], []), "kv_transfer", "exec"), namespace)
        with self.assertRaises(SystemExit) as stop:
            namespace["_on_sigterm"](15, None)
        self.assertEqual(stop.exception.code, 143)
        self.assertEqual(results.exit_status(stop.exception)[0], "failed")

    def test_frontend_sources_take_quettasims_device_yaml(self):
        """The frontend and pd_host sources load the GPU from --device-yaml, a
        required path into QuettaSim's device_spec/; a default under this
        repo named a file that does not exist. QuettaSim main has
        engine.sim.closed_loop and kernel_composed.types, so they import at
        the top."""
        here = Path(native.__file__).parent
        text = (here / "frontend.py").read_text()
        tree = ast.parse(text)
        funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        for name in ("frontend", "pd_host"):
            with self.subTest(name):
                (call,) = [n for n in ast.walk(funcs[f"{name}_parser"]) if isinstance(n, ast.Call)
                           and getattr(n.func, "attr", "") == "add_argument"
                           and any(getattr(x, "value", None) == "--device-yaml" for x in n.args)]
                kw = {k.arg: getattr(k.value, "value", None) for k in call.keywords}
                self.assertIs(kw.get("required"), True)
                self.assertNotIn("default", kw)
                self.assertIn("device_spec/", kw["help"])
                self.assertIn("a.device_yaml", ast.unparse(funcs[f"{name}_probe"]))
        self.assertIn("--device-yaml", ast.get_docstring(tree))
        self.assertNotIn("HERE", text)
        top = {n.module for n in tree.body if isinstance(n, ast.ImportFrom)}
        self.assertLessEqual({"engine.sim.closed_loop", "kernel_composed.types"}, top)
        justfile = (here.parents[1] / "justfile").read_text()
        self.assertIn("--device-yaml {{sim_root}}/device_spec/{{lowercase(gpu)}}.yaml", justfile)


if __name__ == "__main__":
    unittest.main()
