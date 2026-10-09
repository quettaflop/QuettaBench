"""profiling/native.py: Prometheus parsing, the token-weighted prefix-cache
hit rate, listing the traces instrumentation armed at launch, and the glue
to the moved live probes. The /metrics endpoint is a localhost fake; the
probe modules (which need QuettaSim) are stood in by fakes in sys.modules."""

import argparse
import ast
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from quettabench.profiling import native
from tests.profiling import fake_endpoint

LABELS0 = (("engine", "0"), ("model_name", "m"))


class Parse(unittest.TestCase):
    def setUp(self):
        self.m = native.parse_prometheus(fake_endpoint.METRICS)

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
        with fake_endpoint.serve() as (endpoint, _):
            section = native.metrics_section(native.scrape_metrics(endpoint))
        self.assertEqual(section["prefix_cache_hit_rate"], 0.75)
        self.assertEqual(section["samples"][
            'vllm:gpu_prefix_cache_hits_total{engine="0",model_name="m"}'], 800.0)
        self.assertEqual(section["samples"]["process_open_fds"], 42.0)


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


def fake_probe(result):
    """A module with the parser()/probe() surface the moved probes have."""
    def parser():
        ap = argparse.ArgumentParser()
        for flag in ("--base-url", "--proxy-url", "--model", "--gpu-label"):
            ap.add_argument(flag)
        return ap
    return types.SimpleNamespace(parser=parser, probe=lambda a: result(a))


class LiveProbes(unittest.TestCase):
    def test_defaults_fill_only_missing_flags(self):
        self.assertEqual(native.with_defaults(["--model", "x"], {"--model": "m", "--tp": 2,
                                                                 "--skip": None}),
                         ["--model", "x", "--tp", "2"])

    def test_frontend_points_at_the_endpoint(self):
        fake = fake_probe(lambda a: {"base_url": a.base_url, "model": a.model, "gpu": a.gpu_label})
        with mock.patch.dict(sys.modules, {native.PROBES["frontend"]: fake}):
            got = native.serving_frontend("http://h:1", "m", ["--gpu-label", "H200"])
        self.assertEqual(got, {"base_url": "http://h:1/v1", "model": "m", "gpu": "H200"})

    def test_pd_host_rows_and_argv_wins(self):
        fake = fake_probe(lambda a: [{"proxy": a.proxy_url, "model": a.model}])
        with mock.patch.dict(sys.modules, {native.PROBES["pd_host"]: fake}):
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
        here = Path(native.__file__).parent
        for name, flag in (("serving_frontend", "--out"), ("pd_host", "--out"),
                           ("kv_transfer", "--out-dir")):
            with self.subTest(name):
                tree = ast.parse((here / f"{name}.py").read_text())
                funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
                self.assertIn("main", funcs)
                flags = {a.value for n in ast.walk(tree) if isinstance(n, ast.Call)
                         and getattr(n.func, "attr", "") == "add_argument"
                         for a in n.args if isinstance(a, ast.Constant)}
                self.assertIn(flag, flags)
                if name != "kv_transfer":
                    self.assertTrue({"parser", "probe"} <= funcs)


if __name__ == "__main__":
    unittest.main()
