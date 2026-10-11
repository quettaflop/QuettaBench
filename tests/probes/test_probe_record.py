"""probes.base: run_recorded leaves <out>/<probe>/run.json through RunWriter
with its argv and run_status; launch_vllm starts the server through the vllm
adapter and launcher.start, and no probe starts a process itself."""

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest

from quettabench import runner
from quettabench.config.schema import Config, ConfigError, EngineCfg, HardwareCfg, ModeCfg, WorkloadCfg
from quettabench.probes.base import launch_vllm, run_recorded

PROBES = Path(__file__).resolve().parents[2] / "quettabench" / "probes"


def recorded(argv, main, name="gemm", **kw):
    """(raised exception or None, run.json dict or None, the out dir's files)."""
    with tempfile.TemporaryDirectory() as td:
        argv = [a.replace("<TD>", td) for a in argv]
        raised = None
        with mock.patch.object(sys, "argv", ["probe", *argv]):
            try:
                run_recorded(main, name, **kw)
            except BaseException as e:  # noqa: BLE001 -- the test inspects what escaped
                raised = e
        found = sorted(Path(td).rglob("run.json"))
        doc = json.loads(found[0].read_text()) if found else None
        return raised, doc, [str(p.relative_to(td)) for p in found]


class RunRecorded(unittest.TestCase):
    def test_ok_run_records_probe_and_argv(self):
        argv = ["--gpu-label", "H100", "--out", "<TD>", "--wide"]
        raised, doc, files = recorded(argv, lambda: None)
        self.assertIsNone(raised)
        self.assertEqual(files, ["gemm/run.json"])
        self.assertEqual(doc["run_status"], {"status": "ok", "reason": None})
        self.assertEqual(doc["config"]["mode"]["mode"], "probe")
        self.assertEqual(doc["config"]["mode"]["probe"], "gemm")
        self.assertEqual(doc["summary"]["argv"][-1], "--wide")
        self.assertIn("bench_sha", doc["provenance"])

    def test_argv_secrets_are_redacted(self):
        argv = ["--gpu-label", "H100", "--out", "<TD>", "--api-key", "sk-live", "--hf-token=t"]
        _, doc, _ = recorded(argv, lambda: None)
        self.assertNotIn("sk-live", json.dumps(doc))
        self.assertEqual(doc["summary"]["argv"][-3:], ["--api-key", "REDACTED", "--hf-token=REDACTED"])

    def test_failure_is_recorded_and_raised(self):
        def boom():
            raise RuntimeError("no GPU")
        raised, doc, _ = recorded(["--gpu-label", "H100", "--out", "<TD>"], boom)
        self.assertIsInstance(raised, RuntimeError)
        self.assertEqual(doc["run_status"], {"status": "failed", "reason": "RuntimeError: no GPU"})

    def test_non_zero_rank_and_no_output_write_nothing(self):
        self.assertEqual(recorded(["--gpu-label", "H100", "--out", "<TD>"], lambda: False)[2], [])
        self.assertEqual(recorded(["--gpu-label", "H100", "--out", "-"], lambda: None)[2], [])

    def test_a_probe_that_names_its_file(self):
        _, _, files = recorded(["--out-raw", "<TD>/h100/raw.jsonl.gz"], lambda: None,
                               name="serving_decode_grid", file_flag="--out-raw")
        self.assertEqual(files, ["h100/serving_decode_grid/run.json"])


def probe_modules():
    return sorted(p.stem for p in PROBES.glob("*.py") if not p.stem.startswith("_") and p.stem != "base")


def main_def(name):
    """The probe's module-level `def main` node."""
    tree = ast.parse((PROBES / f"{name}.py").read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")


def probe_cfg(name):
    return Config(EngineCfg(), WorkloadCfg(), ModeCfg(mode="probe", probe=name, probe_args="--gpu A"),
                  HardwareCfg(), run_name=f"p-{name}")


@pytest.mark.parametrize("name", probe_modules())
def test_probe_main_takes_argv_and_run_mode_decides(name):
    fn = main_def(name)
    assert [a.arg for a in fn.args.args] == ["argv"], name
    assert len(fn.args.defaults) == 1 and fn.args.defaults[0].value is None, name
    source = (PROBES / f"{name}.py").read_text()
    with tempfile.TemporaryDirectory() as td:
        fake = mock.Mock()
        with mock.patch.dict(sys.modules, {f"quettabench.probes.{name}": fake}):
            if "init_process_group" in source or "init_distributed_environment" in source:
                with pytest.raises(ConfigError, match="torchrun"):
                    runner.run_probe(probe_cfg(name), Path(td))
            elif "add_output_args(ap)" not in source:
                with pytest.raises(ConfigError, match="just probe"):
                    runner.run_probe(probe_cfg(name), Path(td))
            else:
                runner.run_probe(probe_cfg(name), Path(td))
                fake.main.assert_called_once_with(["--gpu", "A", "--out", td])
                return
        fake.main.assert_not_called()
        assert not (Path(td) / "run.json").exists()


class LaunchVllm(unittest.TestCase):
    def test_server_goes_through_the_adapter_and_launcher(self):
        # by name: other tests re-import quettabench.engines.launcher
        with mock.patch("quettabench.engines.launcher.start") as start:
            launch_vllm(Path("/r/pool_capacity"), "/w/llama", port=8794, max_model_len=32768,
                        prefix_caching=True, timeout_s=420,
                        vllm_flags={"served_model_name": "llama", "api_key": "test",
                                    "no_enable_log_requests": True})
        spec, run_dir = start.call_args.args
        self.assertEqual((run_dir, start.call_args.kwargs), (Path("/r/pool_capacity"),
                                                             {"timeout_s": 420}))
        argv = list(spec.server_argv)
        self.assertEqual((Path(argv[0]).name, argv[1:3]), ("vllm", ["serve", "/w/llama"]))
        for flag in ("--enable-prefix-caching", "--served-model-name", "--api-key",
                     "--no-enable-log-requests"):
            self.assertIn(flag, argv)
        self.assertEqual(argv[argv.index("--port") + 1], "8794")

    def test_no_probe_starts_a_process(self):
        for path in sorted(PROBES.glob("*.py")):
            calls = {f"{n.func.value.id}.{n.func.attr}" for n in ast.walk(ast.parse(path.read_text()))
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and isinstance(n.func.value, ast.Name)}
            self.assertFalse(calls & {"subprocess.Popen", "launcher.start"} - (
                {"launcher.start"} if path.name == "base.py" else set()), path.name)


if __name__ == "__main__":
    unittest.main()
