"""quettabench/profiler.py against a fake results/<run>/ and its engine.json."""

import contextlib
import functools
import hashlib
import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from quettabench import profiler
from quettabench.metrics.results import write_atomic
from tests.contracts import helpers
from tests.contracts.test_native import FAKE_SMI, Tokenizer, fake_nsys_export

RUN = {
    "schema_version": 1,
    "config": {"engine": {"model": "m", "tp": 2, "pp": 1}, "hardware": {"visible_devices": "0,1,2,3"}},
    "summary": {"total_output_tokens": 1_500_000, "duration_s": 1800.0},
    "run_status": {"status": "ok", "reason": None},
}


def write_run(d, endpoint, instrumentation="none", devices="0,2"):
    (d / "engine.json").write_text(
        json.dumps(
            {
                "adapter": "vllm",
                "endpoint": endpoint,
                "pid": 1,
                "instrumentation": instrumentation,
                "launch_spec": {"env": {"CUDA_VISIBLE_DEVICES": devices}},
            }
        )
    )
    (d / "run.json").write_text(json.dumps(RUN, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Profile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_layout_and_run_json_untouched(self):
        gsm = self.dir / "gsm8k.jsonl"
        gsm.write_text(
            "\n".join(
                json.dumps(x) for x in ({"question": "abc", "answer": "#### 3"}, {"question": "ab", "answer": "#### 9"})
            )
        )
        with helpers.serve() as (endpoint, posts):
            write_run(self.dir, endpoint, instrumentation="nsys")
            (self.dir / "nsys.nsys-rep").write_bytes(b"")
            before = digest(self.dir / "run.json")
            out = profiler.profile(
                self.dir,
                ["metrics", "cost", "gsm8k"],
                gpu_cost_hr=3.0,
                gsm8k_file=gsm,
                gsm8k_n=2,
                gsm8k_tokenizer=Tokenizer(),
            )
        self.assertEqual(digest(self.dir / "run.json"), before)
        self.assertEqual(json.loads((self.dir / "profile.json").read_text()), out)
        self.assertEqual(sorted(out), ["accuracy", "cost", "errors", "metrics", "profiled_at", "sources_run", "traces"])
        self.assertEqual(out["metrics"]["prefix_cache_hit_rate"], 0.75)
        self.assertEqual(
            out["cost"],
            {
                "usd": 3.0,
                "usd_per_m_out_tokens": 2.0,
                "gpu_cost_hr": 3.0,
                "devices": 2,
                "wall_s": 1800.0,
                "out_tokens": 1_500_000,
            },
        )
        # the fake answers "#### <prompt length>": 3 matches, 2 does not
        self.assertEqual(out["accuracy"]["exact_match"], 0.5)
        self.assertEqual([b["model"] for _, b in posts], ["m", "m"])
        self.assertEqual(out["traces"], {"instrumentation": "nsys", "files": ["nsys.nsys-rep"], "missing": False})
        self.assertEqual(out["sources_run"], ["cost", "gsm8k", "metrics"])
        self.assertEqual(out["errors"], {})

    def test_energy_waits_for_run_json(self):
        with helpers.serve() as (endpoint, _):
            write_run(self.dir, endpoint)
        run_json = self.dir / "run.json"
        body = run_json.read_text()
        run_json.unlink()

        def finish_run():
            time.sleep(0.3)
            write_atomic(run_json, body)  # as RunWriter does, so no half file is read

        writer = threading.Thread(target=finish_run)
        writer.start()
        fast = functools.partial(profiler.measure_energy, poll_s=0.01)
        with mock.patch.object(profiler, "measure_energy", fast):
            out = profiler.profile(self.dir, ["energy"], nvidia_smi=FAKE_SMI, interval_s=0.05)
        writer.join()
        e = out["energy"]
        self.assertEqual(e["window"], "run")
        self.assertEqual(e["devices"], 2)  # engine.json's 0,2, not the Config's four
        self.assertGreater(e["samples"], 0)
        self.assertEqual(e["joules"], 220.0 * e["samples"] * 0.05)
        self.assertEqual(e["joules_per_token"], e["joules"] / 1_500_000)
        self.assertEqual(e["scope"], "gpu_board")

    def test_energy_is_refused_off_cuda(self):
        write_run(self.dir, "http://127.0.0.1:9")
        run = json.loads((self.dir / "run.json").read_text())
        run["config"]["hardware"]["backend"] = "rocm"
        (self.dir / "run.json").write_text(json.dumps(run))
        with self.assertRaisesRegex(ValueError, "needs hardware.backend=cuda; this run's is rocm"):
            profiler.profile(self.dir, ["energy"], nvidia_smi=FAKE_SMI)

    def test_energy_after_the_run_needs_a_fixed_window(self):
        write_run(self.dir, "http://127.0.0.1:9")
        out = profiler.profile(self.dir, ["energy"], nvidia_smi=FAKE_SMI)
        self.assertIn("run.json already exists", out["errors"]["energy"])
        out = profiler.profile(self.dir, ["energy"], nvidia_smi=FAKE_SMI, interval_s=0.05, energy_s=0.3)
        self.assertEqual(out["energy"]["window"], "fixed")
        self.assertIsNone(out["energy"]["joules_per_token"])
        self.assertEqual(profiler.profile_tag_lines(out), [])

    def test_a_dead_endpoint_is_recorded_and_others_still_run(self):
        write_run(self.dir, "http://127.0.0.1:9")
        out = profiler.profile(self.dir, ["metrics", "cost"], gpu_cost_hr=1.0)
        self.assertIn("metrics", out["errors"])
        self.assertIn("cost", out)
        self.assertEqual(out["sources_run"], ["cost"])

    def test_later_profile_merges(self):
        write_run(self.dir, "http://127.0.0.1:9")
        profiler.profile(self.dir, ["cost"], gpu_cost_hr=1.0)
        (self.dir / "rows.jsonl").write_text(
            json.dumps(
                {
                    "probe": "kv_transfer",
                    "table": "kv_transfer/H200_nixl.csv",
                    "key": {"gpu": "H200"},
                    "value": {"total_ms": 1.5},
                    "timing": "cuda_event",
                }
            )
            + "\n"
        )
        out = profiler.profile(self.dir, ["kv_transfer"])
        self.assertEqual(out["sources_run"], ["cost", "kv_transfer"])
        self.assertIn("cost", out)
        self.assertEqual(out["kv_transfer"]["rows"], [{"gpu": "H200", "total_ms": 1.5}])

    def test_kernels_reads_the_armed_trace(self):
        write_run(self.dir, "http://127.0.0.1:9")
        fake_nsys_export(self.dir / "nsys.sqlite")
        out = profiler.profile(self.dir, ["kernels"], kernel_steps=2, kernel_window="prefill_decode")
        self.assertEqual(out["errors"], {})
        self.assertEqual(out["kernels"]["launches_per_step"], 2.5)

    def test_missing_source_args_fail_before_anything_runs(self):
        for sources, args in ((["cost"], {}), (["gsm8k"], {"gsm8k_file": "x"}), (["kernels"], {}), (["bogus"], {})):
            with self.subTest(sources), self.assertRaises(ValueError):
                profiler.profile(self.dir, sources, **args)
        self.assertFalse((self.dir / "profile.json").exists())


class ProfileTagLines(unittest.TestCase):
    def test_formats(self):
        profile = {
            "energy": {
                "joules": 900.75,
                "joules_per_token": 0.0123456,
                "scope": "gpu_board",
                "devices": 2,
                "samples": 3,
            },
            "accuracy": {"exact_match": 2 / 3, "n": 3},
            "cost": {"usd": 3.0, "usd_per_m_out_tokens": 2.0, "gpu_cost_hr": 2.5, "devices": 2, "wall_s": 1800.04},
        }
        self.assertEqual(
            profiler.profile_tag_lines(profile),
            [
                "ENERGYSUM joules=901 joules_per_token=0.012 scope=gpu_board devices=2 samples=3",
                "ACCSUM dataset=gsm8k exact_match=0.667 n=3",
                "COSTSUM usd=3.0000 usd_per_m_out_tokens=2.00 gpu_cost_hr=2.5 devices=2 wall_s=1800.0",
            ],
        )
        profile["energy"]["devices"] = "all"
        self.assertTrue(profiler.profile_tag_lines(profile)[0].endswith("devices=all samples=3"))
        self.assertEqual(profiler.profile_tag_lines({}), [])


class Main(unittest.TestCase):
    def test_cli_writes_profile_and_prints_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_run(d, "http://127.0.0.1:9")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = profiler.main(["--run", tmp, "--source", "cost", "--gpu-cost-hr", "3"])
            self.assertEqual(code, 0)
            self.assertEqual(
                stdout.getvalue(),
                "COSTSUM usd=3.0000 usd_per_m_out_tokens=2.00 gpu_cost_hr=3.0 devices=2 wall_s=1800.0\n",
            )
            self.assertTrue((d / "profile.json").exists())
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(profiler.main(["--run", tmp, "--source", "metrics"]), 1)

    def test_cli_refuses_missing_args(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            profiler.main(["--run", "x", "--source", "cost"])


if __name__ == "__main__":
    unittest.main()
