"""profiling/native.py: Prometheus parsing, the token-weighted prefix-cache
hit rate, and listing the traces instrumentation armed at launch. The
/metrics endpoint is a localhost fake."""

import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
