"""qbench run's sequence without a GPU: --dry-run on every run yaml prints
its Config and LaunchSpec and writes nothing, a matrix runs once per point,
and mode: probe hands the probe its flags plus --out results/<run>/."""

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from quettabench import cli
from tests.golden._harness import REPO

RUNS = sorted((REPO / "configs" / "runs").glob("*.yaml"))
# Hybrid attention without kv_bytes: the loader refuses on purpose.
REFUSED = {"qwen": "attention=hybrid needs kv_bytes"}


def qbench(*argv):
    """(stdout, SystemExit message or None) of cli.main(argv)."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out), mock.patch("shutil.which", return_value=None):
        try:
            cli.main(list(argv))
        except SystemExit as e:
            return out.getvalue(), str(e.code)
    return out.getvalue(), None


@contextlib.contextmanager
def scratch_configs(name, run):
    """configs/ copied to a scratch dir with one more run yaml."""
    with tempfile.TemporaryDirectory() as td:
        configs = Path(td, "configs")
        shutil.copytree(REPO / "configs", configs)
        path = configs / "runs" / f"{name}.yaml"
        path.write_text(yaml.safe_dump(run))
        yield path, Path(td)


class DryRun(unittest.TestCase):
    def test_every_run_yaml(self):
        for path in RUNS:
            with self.subTest(path.stem), tempfile.TemporaryDirectory() as td:
                out, exit_msg = qbench("run", str(path), "--dry-run", "--allow-unverified",
                                       "--set", "engine.model=/w", "--results-dir", td)
                self.assertEqual(list(Path(td).iterdir()), [])
                if path.stem in REFUSED:
                    self.assertIn(REFUSED[path.stem], exit_msg)
                    continue
                self.assertIsNone(exit_msg)
                lines = out.splitlines()
                self.assertEqual(lines[0], f"== {path.stem}")
                self.assertIn("engine.model = '/w'  (cli)", lines)
                self.assertIn("spec.adapter = 'vllm'", lines)

    def test_matrix_is_one_run_per_point(self):
        run = {"model": "llama31_8b", "workload": {"kind": "grid"}, "mode": {"grid": "smoke"},
               "matrix": {"engine.tp": [1, 2]}}
        with scratch_configs("sweep", run) as (path, _):
            out, exit_msg = qbench("run", str(path), "--dry-run", "--set", "engine.model=/w")
        self.assertIsNone(exit_msg)
        self.assertEqual([line for line in out.splitlines() if line.startswith("==")],
                         ["== sweep-tp1", "== sweep-tp2"])
        self.assertIn("launch = none, the driver runs it", out)  # grid runs launch nothing

    def test_refusal_is_the_exit_message(self):
        _, exit_msg = qbench("run", str(RUNS[0].with_name("llama.yaml")), "--dry-run",
                             "--set", "mode.grid=nope")
        self.assertIn("unknown grid 'nope'", exit_msg)


class ProbeMode(unittest.TestCase):
    def test_probe_gets_its_flags_and_the_run_dir(self):
        run = {"mode": {"mode": "probe", "probe": "fake", "probe_args": "--gpu-label A100 --wide"}}
        probe = mock.Mock()
        with scratch_configs("gemm-a100", run) as (path, td), \
                mock.patch.dict(sys.modules, {"quettabench.probes.fake": probe}):
            _, exit_msg = qbench("run", str(path), "--allow-unverified",
                                 "--results-dir", str(td / "results"))
            doc = json.loads((td / "results" / "gemm-a100" / "run.json").read_text())
        self.assertIsNone(exit_msg)
        argv = ["--gpu-label", "A100", "--wide", "--out", str(td / "results" / "gemm-a100")]
        probe.main.assert_called_once_with(argv)
        self.assertEqual(doc["summary"], {"probe": "fake", "argv": argv})

    def test_probe_mode_needs_a_probe(self):
        with scratch_configs("bare", {"mode": {"mode": "probe"}}) as (path, _):
            _, exit_msg = qbench("run", str(path), "--dry-run", "--allow-unverified")
        self.assertIn("mode=probe needs mode.probe", exit_msg)


if __name__ == "__main__":
    unittest.main()
