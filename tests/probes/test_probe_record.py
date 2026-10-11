"""probes.base: run_recorded leaves <out>/<probe>/<run-id>/run.json, beside the
probe's tables, through RunWriter
with its argv and run_status; launch_vllm starts the server through the vllm
adapter and launcher.start, and no probe starts a process itself."""

import argparse
import ast
import contextlib
import dataclasses
import importlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest

from quettabench import runner
from quettabench.config.schema import Config, ConfigError, EngineCfg, HardwareCfg, ModeCfg, WorkloadCfg
from quettabench.probes import base as probes_base
from quettabench.probes.base import launch_vllm, out_root, prior_table, run_id, run_recorded

PROBES = Path(__file__).resolve().parents[2] / "quettabench" / "probes"
NOW = datetime(2026, 10, 11, 12, 0, 0, 123456, tzinfo=timezone.utc)
RUN = "20261011T120000.123456Z"


def recorded(argv, main, name="gemm", **kw):
    """(raised exception or None, run.json dict or None, the out dir's files)."""
    with tempfile.TemporaryDirectory() as td:
        argv = [a.replace("<TD>", td) for a in argv]
        raised = None
        with mock.patch.object(sys, "argv", ["probe", *argv]):
            try:
                run_recorded(main, name, **{"now": NOW, **kw})
            except BaseException as e:  # noqa: BLE001 -- the test inspects what escaped
                raised = e
        found = sorted(Path(td).rglob("run.json"))
        doc = json.loads(found[0].read_text()) if found else None
        return raised, doc, [str(p.relative_to(td)) for p in found]


class RunRecorded(unittest.TestCase):
    def test_ok_run_records_probe_and_argv(self):
        argv = ["--gpu-label", "H100", "--out", "<TD>", "--wide"]
        raised, doc, files = recorded(argv, lambda argv: None)
        self.assertIsNone(raised)
        self.assertEqual(files, [f"gemm/{RUN}/run.json"])
        self.assertEqual(doc["run_status"], {"status": "ok", "reason": None})
        self.assertEqual(doc["config"]["mode"]["mode"], "probe")
        self.assertEqual(doc["config"]["mode"]["probe"], "gemm")
        self.assertEqual(doc["summary"]["argv"][:2], ["--run-id", RUN])
        self.assertEqual(doc["summary"]["argv"][-1], "--wide")
        self.assertEqual(doc["provenance"]["started"], NOW.isoformat())
        self.assertIn("bench_sha", doc["provenance"])

    def test_argv_secrets_are_redacted(self):
        argv = ["--gpu-label", "H100", "--out", "<TD>", "--api-key", "sk-live", "--hf-token=t"]
        _, doc, _ = recorded(argv, lambda argv: None)
        self.assertNotIn("sk-live", json.dumps(doc))
        self.assertEqual(doc["summary"]["argv"][-3:], ["--api-key", "REDACTED", "--hf-token=REDACTED"])

    def test_failure_is_recorded_and_raised(self):
        def boom(argv):
            raise RuntimeError("no GPU")
        raised, doc, _ = recorded(["--gpu-label", "H100", "--out", "<TD>"], boom)
        self.assertIsInstance(raised, RuntimeError)
        self.assertEqual(doc["run_status"], {"status": "failed", "reason": "RuntimeError: no GPU"})

    def test_non_zero_rank_and_no_output_write_nothing(self):
        self.assertEqual(recorded(["--gpu-label", "H100", "--out", "<TD>"], lambda argv: False)[2], [])
        self.assertEqual(recorded(["--gpu-label", "H100", "--out", "-"], lambda argv: None)[2], [])

    def test_only_rank_0_records_even_when_it_raises(self):
        argv = ["--gpu-label", "H100", "--out", "<TD>"]
        mine = [f"gemm/{RUN}/run.json"]
        for rank, files in (("1", []), ("0", mine), (None, mine)):
            env = {} if rank is None else {"RANK": rank}
            with self.subTest(rank=rank), mock.patch.dict(os.environ, env, clear=True):
                raised, _, found = recorded(argv, mock.Mock(side_effect=RuntimeError("nccl")))
                self.assertIsInstance(raised, RuntimeError)
                self.assertEqual(found, files)

    def test_a_probe_that_names_its_file(self):
        _, _, files = recorded(["--out-raw", "<TD>/h100/raw.jsonl.gz"], lambda argv: None,
                               name="serving_decode_grid", file_flag="--out-raw")
        self.assertEqual(files, ["h100/run.json"])

    def test_tables_and_run_json_share_the_run_dir(self):
        def main(argv):
            ap = argparse.ArgumentParser()
            probes_base.add_output_args(ap)
            table = out_root(ap.parse_args(argv), "gemm") / "ncu" / "gemm" / "H100.csv"
            table.parent.mkdir(parents=True)
            table.write_text("m,n,k,us\n")
            seen.append(table)
        seen = []
        _, _, files = recorded(["--gpu-label", "H100", "--out", "<TD>"], main)
        self.assertEqual(files, [f"gemm/{RUN}/run.json"])
        self.assertEqual(seen[0].parts[-5:], ("gemm", RUN, "ncu", "gemm", "H100.csv"))

    def test_each_invocation_gets_its_own_run_dir(self):
        """Two runs of one probe never share a dir, so the second's run.json
        cannot stand for the first's tables."""
        with tempfile.TemporaryDirectory() as td:
            argv = ["probe", "--gpu-label", "H100", "--out", td]
            for i, fails in enumerate((False, True)):
                with mock.patch.object(sys, "argv", argv), \
                        contextlib.suppress(RuntimeError):
                    run_recorded(mock.Mock(side_effect=RuntimeError("x") if fails else None),
                                 "gemm", now=NOW + timedelta(seconds=i))
            docs = {p.parent.name: json.loads(p.read_text())["run_status"]["status"]
                    for p in Path(td).rglob("run.json")}
        self.assertEqual(docs, {RUN: "ok", run_id(NOW + timedelta(seconds=1)): "failed"})

    def test_a_run_id_is_a_timestamp_not_a_path(self):
        for bad in ("../x", "/tmp", "a"):
            a = argparse.Namespace(out_dir="r", gpu_label="H100", run_id=bad)
            with self.subTest(bad), self.assertRaisesRegex(SystemExit, "not a run id"):
                out_root(a, "gemm")


class PriorTable(unittest.TestCase):
    """An upserting probe merges into the newest earlier ok run's table only."""

    def run_dir(self, root, when, status, table=True):
        d = root / "gemm" / run_id(when)
        (d / "ncu" / "gemm").mkdir(parents=True)
        if table:
            (d / "ncu" / "gemm" / "H100.csv").write_text(f"{when.second}\n")
        if status:
            (d / "run.json").write_text(json.dumps({"run_status": {"status": status}}))
        return d

    def test_newest_earlier_ok_run(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            at = [NOW + timedelta(seconds=s) for s in range(6)]
            self.run_dir(root, at[0], "ok")
            self.run_dir(root, at[1], "ok")
            self.run_dir(root, at[2], "failed")
            self.run_dir(root, at[3], None)        # unfinished: no run.json
            self.run_dir(root, at[4], "ok", table=False)
            self.run_dir(root, at[5], "ok")        # later than this run: not prior
            (root / "gemm" / "notes").mkdir()
            a = argparse.Namespace(out_dir=td, gpu_label="H100", run_id=run_id(NOW + timedelta(seconds=4, microseconds=1)))
            path = out_root(a, "gemm") / "ncu" / "gemm" / "H100.csv"
            self.assertEqual(prior_table(a, "gemm", path).read_text(), "1\n")
            a.run_id = run_id(at[0])
            self.assertIsNone(prior_table(a, "gemm", out_root(a, "gemm") / "ncu" / "gemm" / "H100.csv"))

    def test_without_a_run_id_the_table_itself(self):
        with tempfile.TemporaryDirectory() as td:
            a = argparse.Namespace(out_dir=td, gpu_label="H100", run_id=None)
            path = out_root(a, "gemm") / "t.csv"
            self.assertIsNone(prior_table(a, "gemm", path))
            path.parent.mkdir(parents=True)
            path.write_text("x\n")
            self.assertEqual(prior_table(a, "gemm", path), path)


@pytest.mark.parametrize("name", ["cross_attn", "flash_attn", "gemm", "moe", "reparallel"])
def test_upserting_probes_read_only_through_prior_table(name):
    """A probe that merges an earlier table reads it through prior_table, never
    its own output path, so a failed run's rows do not carry forward."""
    source = (PROBES / f"{name}.py").read_text()
    assert "prior_table(a, " in source, name
    assert ".exists()" not in source, name


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
    with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ):
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
        for flag in ("--enable-prefix-caching", "--served-model-name", "--no-enable-log-requests"):
            self.assertIn(flag, argv)
        self.assertNotIn("--api-key", argv)
        self.assertEqual(spec.env["VLLM_API_KEY"], "test")
        self.assertEqual(argv[argv.index("--port") + 1], "8794")

    def test_a_run_probe_launches_with_the_runs_hardware(self):
        cfg = dataclasses.replace(probe_cfg("pool_capacity"),
                                  hardware=HardwareCfg(provision={"py": "/venv/bin/python"}))
        with mock.patch("quettabench.engines.launcher.start") as start:
            with probes_base.run_config(cfg):
                launch_vllm(Path("/r/x"), "/w", port=1, max_model_len=8, prefix_caching=False)
            launch_vllm(Path("/r/x"), "/w", port=1, max_model_len=8, prefix_caching=False)
        ran, after = (call.args[0].server_argv[0] for call in start.call_args_list)
        self.assertEqual(ran, "/venv/bin/vllm")
        self.assertEqual(after, str(Path(sys.executable).with_name("vllm")))

    def test_each_live_probe_launch_is_pinned(self):
        """argv and env of the three live-server probes' launches; what changed
        against their own launches before d7287c5 is in tests/golden/REPORT.md."""
        from quettabench.probes import pool_capacity, serving_decode_grid, serving_stage_split

        def launched(fn):
            with mock.patch("quettabench.engines.launcher.start") as start:
                start.return_value = mock.Mock(results_dir=Path(td))
                fn()
            spec = start.call_args.args[0]
            return [Path(spec.server_argv[0]).name, *spec.server_argv[1:]], dict(spec.env)

        common = ["--tensor-parallel-size", "1", "--dtype", "bfloat16", "--max-model-len", "32768",
                  "--gpu-memory-utilization", "0.9"]
        tail = ["--enable-chunked-prefill", "--trust-remote-code", "--served-model-name", "llama",
                "--host", "127.0.0.1"]
        env = {"VLLM_SERVER_DEV_MODE": "1", "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1"}
        with tempfile.TemporaryDirectory() as td:
            Path(td, "server.log").write_text("")
            with mock.patch.object(pool_capacity, "run", mock.Mock()), \
                    mock.patch.object(pool_capacity.asyncio, "run"):
                pool = launched(lambda: pool_capacity.main(
                    ["--model", "/w", "--gpu-label", "H100", "--out", td]))
            split = launched(lambda: serving_stage_split.launch_server(
                "/w", 8794, 0.9, 32768, "k", "sha256", Path(td)))
            grid = launched(lambda: serving_decode_grid.launch_server(
                "/w", 8795, 0.9, 32768, "k", Path(td), 1, 256, 8192))
        self.assertEqual(pool, (["vllm", "serve", "/w", "--port", "8794", *common,
                                 "--enable-prefix-caching", *tail,
                                 "--prefix-caching-hash-algo", "sha256",
                                 "--no-enable-log-requests"], {**env, "VLLM_API_KEY": "test"}))
        self.assertEqual(split, (["vllm", "serve", "/w", "--port", "8794", *common,
                                  "--enable-prefix-caching", *tail,
                                  "--no-enable-log-requests", "--prefix-caching-hash-algo",
                                  "sha256"], {**env, "VLLM_API_KEY": "k"}))
        self.assertEqual(grid, (["vllm", "serve", "/w", "--port", "8795", *common,
                                 "--no-enable-prefix-caching", *tail, "--max-num-seqs", "256",
                                 "--max-num-batched-tokens", "8192",
                                 "--no-enable-log-requests"], {**env, "VLLM_API_KEY": "k"}))

    def test_no_probe_starts_a_process(self):
        for path in sorted(PROBES.glob("*.py")):
            calls = {f"{n.func.value.id}.{n.func.attr}" for n in ast.walk(ast.parse(path.read_text()))
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and isinstance(n.func.value, ast.Name)}
            self.assertFalse(calls & {"subprocess.Popen", "launcher.start"} - (
                {"launcher.start"} if path.name == "base.py" else set()), path.name)


if __name__ == "__main__":
    unittest.main()


class StageSplitKey(unittest.TestCase):
    """serving_stage_split sends its --api-key and never prints it."""

    def test_requests_carry_the_api_key(self):
        from quettabench.probes import serving_stage_split as sss

        class Stop(Exception):
            pass

        made = []

        class Session:
            def __init__(self, **kw):
                made.append(kw)

            async def __aenter__(self):
                raise Stop

            async def __aexit__(self, *exc):
                return False

        with mock.patch.object(sss.aiohttp, "ClientSession", Session), \
                self.assertRaises(Stop):
            sss.asyncio.run(sss.run_sweep("http://h:1", "llama", [8], [0], 1, "/dev/null",
                                          api_key="sk-live"))
        self.assertEqual(made[0]["headers"], {"Authorization": "Bearer sk-live"})
        self.assertNotIn("Bearer test", (PROBES / "serving_stage_split.py").read_text())

    def test_the_nsys_command_takes_the_key_from_the_env(self):
        from quettabench.probes import serving_stage_split as sss
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            sss.main(["--model", "/w", "--emit-nsys-cmd", "--api-key", "sk-live", "--out", "x.csv"])
        self.assertNotIn("sk-live", out.getvalue())
        self.assertIn("VLLM_API_KEY", out.getvalue())

    def test_out_is_required_so_the_run_is_recorded(self):
        """With no --out the probe would write a table record_dir cannot place;
        argparse refuses before anything is written."""
        from quettabench.probes import serving_stage_split as sss
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stop:
            sss.main(["--model", "/w", "--no-launch"])
        self.assertEqual(stop.exception.code, 2)
        self.assertNotIn("H100", (PROBES / "serving_stage_split.py").read_text().split("def main")[1])


def import_without_torch(name):
    """quettabench.probes.<name> imported on CPU with an empty torch module."""
    fake = types.ModuleType("torch")
    fake.nn = types.ModuleType("torch.nn")
    fake.nn.functional = types.ModuleType("torch.nn.functional")
    with mock.patch.dict(sys.modules, {"torch": fake, "torch.nn": fake.nn,
                                       "torch.nn.functional": fake.nn.functional}):
        sys.modules.pop(f"quettabench.probes.{name}", None)
        return importlib.import_module(f"quettabench.probes.{name}")


def test_gemm_ncu_reads_kernels_through_the_shared_parser():
    """gemm --ncu takes its kernel times from _ncu.ncu_kernel_durations, the one
    NCU CSV parser, and keeps the GEMM family's fastest launch."""
    gemm = import_without_torch("gemm")
    launches = [("distribution_elementwise_kernel", 1.0), ("nvjet_tst_64x8", 5.0),
                ("ampere_bf16_s16816gemm", 4.0), ("at::fill", 0.5)]
    with mock.patch.object(gemm._ncu, "ncu_kernel_durations", return_value=launches) as durs:
        assert gemm.ncu_gemm_us(1, 4096, 4096, "/opt/ncu") == 4.0
    assert durs.call_args.args[1] == "/opt/ncu"
    with mock.patch.object(gemm._ncu, "ncu_kernel_durations", return_value=[("at::fill", 1.0)]):
        assert gemm.ncu_gemm_us(1, 4096, 4096, "/opt/ncu") is None
    assert "Metric Value" not in (PROBES / "gemm.py").read_text()
