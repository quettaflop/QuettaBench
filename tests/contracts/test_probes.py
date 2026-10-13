"""probes/base.py run recording and remote runs; every probe's justfile and flag contracts."""

from __future__ import annotations

import argparse
import ast
import contextlib
import dataclasses
import importlib
import io
import json
import os
import re
import shutil
import statistics
import subprocess
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
from quettabench.metrics.results import RunWriter
from quettabench.probes import base as probes_base
from quettabench.probes.base import ProbeRow, emit, launch_vllm, out_root, run_id, run_recorded

PROBES = Path(__file__).resolve().parents[2] / "quettabench" / "probes"
NOW = datetime(2026, 10, 11, 12, 0, 0, 123456, tzinfo=timezone.utc)
RUN = "20261011T120000.123456Z"


def recorded(argv, main, name="gemm", **kw):
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
        _, _, files = recorded(
            ["--out-raw", "<TD>/h100/raw.jsonl.gz"],
            lambda argv: None,
            name="serving_decode_grid",
            file_flag="--out-raw",
        )
        self.assertEqual(files, ["h100/run.json"])

    def test_emitted_rows_and_run_json_share_the_run_dir(self):
        row = ProbeRow("gemm", "gemm/H100.csv", {"M": 1, "N": 2, "K": 3}, {"latency_us": 4.5}, "ncu")

        def main(argv):
            emit(row)
            emit(row)

        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch.object(sys, "argv", ["probe", "--gpu-label", "H100", "--out", td]),
        ):
            run_recorded(main, "gemm", now=NOW)
            run = Path(td, "gemm", RUN)
            self.assertEqual(sorted(p.name for p in run.iterdir()), ["requests.jsonl", "rows.jsonl", "run.json"])
            lines = (run / "rows.jsonl").read_text().splitlines()
        self.assertEqual([json.loads(line) for line in lines], [dataclasses.asdict(row)] * 2)
        self.assertIsNone(probes_base._WRITER)

    def test_a_row_outside_a_recorded_run_is_refused(self):
        row = ProbeRow("gemm", "gemm/H100.csv", {"M": 1}, {"latency_us": 4.5}, "eager")
        with self.assertRaisesRegex(RuntimeError, "no run records"):
            emit(row)
        with self.assertRaisesRegex(ValueError, "timing 'cuda_event'"):
            ProbeRow("gemm", "gemm/H100.csv", {}, {}, "cuda_event")

    def test_a_probe_emits_each_measured_point_and_skips_none(self):
        class Grid(probes_base.Probe):
            def grid(self):
                return [1, 2, 3]

            def run_point(self, point):
                return None if point == 2 else ProbeRow("x", "x/A.csv", {"n": point}, {"us": 1}, "eager")

        writer = mock.Mock()
        cfg = Config(EngineCfg(), WorkloadCfg(), ModeCfg(), HardwareCfg(), run_name="x")
        with probes_base.run_config(cfg, writer):
            self.assertEqual(Grid().run(), 2)
        self.assertEqual([c.args[0].key for c in writer.probe_row.call_args_list], [{"n": 1}, {"n": 3}])

    def test_a_qbench_run_probe_writes_its_rows_into_the_run_dir(self):
        def main(argv):
            emit(ProbeRow("gemm", "gemm/A.csv", {"M": 1}, {"us": 2}, "eager"))

        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch.dict(os.environ),
            mock.patch.dict(sys.modules, {"quettabench.probes.gemm": mock.Mock(main=main)}),
            mock.patch.object(runner.launcher, "idle_gpus", return_value="0"),
        ):
            runner.run_probe(probe_cfg("gemm"), Path(td))
            self.assertEqual(json.loads(Path(td, "rows.jsonl").read_text())["table"], "gemm/A.csv")
            self.assertTrue(Path(td, "run.json").is_file())

    def test_each_invocation_gets_its_own_run_dir(self):
        with tempfile.TemporaryDirectory() as td:
            argv = ["probe", "--gpu-label", "H100", "--out", td]
            for i, fails in enumerate((False, True)):
                with mock.patch.object(sys, "argv", argv), contextlib.suppress(RuntimeError):
                    run_recorded(
                        mock.Mock(side_effect=RuntimeError("x") if fails else None),
                        "gemm",
                        now=NOW + timedelta(seconds=i),
                    )
            docs = {
                p.parent.name: json.loads(p.read_text())["run_status"]["status"] for p in Path(td).rglob("run.json")
            }
        self.assertEqual(docs, {RUN: "ok", run_id(NOW + timedelta(seconds=1)): "failed"})

    def test_a_run_dir_that_already_exists_is_refused(self):
        for argv, kw, made in (
            (["--gpu-label", "H100", "--out", "<TD>"], {}, f"gemm/{RUN}"),
            (
                ["--out-raw", "<TD>/h100/raw.jsonl.gz"],
                {"name": "serving_decode_grid", "file_flag": "--out-raw"},
                "h100",
            ),
        ):
            with self.subTest(made), tempfile.TemporaryDirectory() as td:
                (Path(td) / made).mkdir(parents=True)
                (Path(td) / made / "notes.txt").write_text("mine\n")
                main = mock.Mock()
                with (
                    mock.patch.object(sys, "argv", ["probe", *(a.replace("<TD>", td) for a in argv)]),
                    self.assertRaisesRegex(SystemExit, "already exists"),
                ):
                    run_recorded(main, kw.get("name", "gemm"), now=NOW, file_flag=kw.get("file_flag"))
                main.assert_not_called()
                self.assertEqual(sorted(p.name for p in (Path(td) / made).iterdir()), ["notes.txt"])

    def test_the_run_writer_can_refuse_an_existing_dir(self):
        cfg = Config(EngineCfg(), WorkloadCfg(), ModeCfg(), HardwareCfg(), run_name="x")
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileExistsError):
                RunWriter(td, cfg, exclusive=True)
            RunWriter(Path(td, "new"), cfg, exclusive=True).discard()

    def test_a_run_id_is_a_timestamp_not_a_path(self):
        for bad in ("../x", "/tmp", "a"):
            a = argparse.Namespace(out_dir="r", gpu_label="H100", run_id=bad)
            with self.subTest(bad), self.assertRaisesRegex(SystemExit, "not a run id"):
                out_root(a, "gemm")


@pytest.mark.parametrize("name", ["chunked_prefill_attn", "flash_attn", "gemm", "moe", "reparallel"])
def test_a_probe_run_writes_only_the_rows_it_measured(name):
    source = (PROBES / f"{name}.py").read_text()
    assert "prior_table" not in source, name
    assert "prior" not in ast.dump(ast.parse(source)).lower(), name
    assert not hasattr(probes_base, "prior_table")


def probe_modules():
    return sorted(p.stem for p in PROBES.glob("*.py") if not p.stem.startswith("_") and p.stem != "base")


def main_def(name):
    tree = ast.parse((PROBES / f"{name}.py").read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")


def probe_cfg(name):
    return Config(
        EngineCfg(),
        WorkloadCfg(),
        ModeCfg(mode="probe", probe=name, probe_args="--gpu A"),
        HardwareCfg(),
        run_name=f"p-{name}",
    )


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
        # By name: other tests re-import quettabench.engines.launcher.
        with mock.patch("quettabench.engines.launcher.start") as start:
            launch_vllm(
                Path("/r/serving_stage_split"),
                "/w/llama",
                port=8794,
                max_model_len=32768,
                prefix_caching=True,
                timeout_s=420,
                vllm_flags={"served_model_name": "llama", "api_key": "test", "no_enable_log_requests": True},
            )
        spec, run_dir = start.call_args.args
        self.assertEqual((run_dir, start.call_args.kwargs), (Path("/r/serving_stage_split"), {"timeout_s": 420}))
        argv = list(spec.server_argv)
        self.assertEqual((Path(argv[0]).name, argv[1:3]), ("vllm", ["serve", "/w/llama"]))
        for flag in ("--enable-prefix-caching", "--served-model-name", "--no-enable-log-requests"):
            self.assertIn(flag, argv)
        self.assertNotIn("--api-key", argv)
        self.assertEqual(spec.env["VLLM_API_KEY"], "test")
        self.assertEqual(argv[argv.index("--port") + 1], "8794")

    def test_a_run_probe_launches_with_the_runs_hardware(self):
        cfg = dataclasses.replace(
            probe_cfg("serving_stage_split"), hardware=HardwareCfg(provision={"py": "/venv/bin/python"})
        )
        with mock.patch("quettabench.engines.launcher.start") as start:
            with probes_base.run_config(cfg):
                launch_vllm(Path("/r/x"), "/w", port=1, max_model_len=8, prefix_caching=False)
            launch_vllm(Path("/r/x"), "/w", port=1, max_model_len=8, prefix_caching=False)
        ran, after = (call.args[0].server_argv[0] for call in start.call_args_list)
        self.assertEqual(ran, "/venv/bin/vllm")
        self.assertEqual(after, str(Path(sys.executable).with_name("vllm")))

    def test_each_live_probe_launch_is_pinned(self):
        from quettabench.probes import serving_decode_grid, serving_stage_split

        def launched(fn):
            with mock.patch("quettabench.engines.launcher.start") as start:
                start.return_value = mock.Mock(results_dir=Path(td))
                fn()
            spec = start.call_args.args[0]
            return [Path(spec.server_argv[0]).name, *spec.server_argv[1:]], dict(spec.env)

        common = [
            "--tensor-parallel-size",
            "1",
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "32768",
            "--gpu-memory-utilization",
            "0.9",
        ]
        tail = [
            "--enable-chunked-prefill",
            "--trust-remote-code",
            "--served-model-name",
            "llama",
            "--host",
            "127.0.0.1",
        ]
        env = {"VLLM_SERVER_DEV_MODE": "1", "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1"}
        with tempfile.TemporaryDirectory() as td:
            Path(td, "server.log").write_text("")
            split = launched(lambda: serving_stage_split.launch_server("/w", 8794, 0.9, 32768, "k", "sha256", Path(td)))
            grid = launched(
                lambda: serving_decode_grid.launch_server("/w", 8795, 0.9, 32768, "k", Path(td), 1, 256, 8192)
            )
        self.assertEqual(
            split,
            (
                [
                    "vllm",
                    "serve",
                    "/w",
                    "--port",
                    "8794",
                    *common,
                    "--enable-prefix-caching",
                    *tail,
                    "--no-enable-log-requests",
                    "--prefix-caching-hash-algo",
                    "sha256",
                ],
                {**env, "VLLM_API_KEY": "k"},
            ),
        )
        self.assertEqual(
            grid,
            (
                [
                    "vllm",
                    "serve",
                    "/w",
                    "--port",
                    "8795",
                    *common,
                    "--no-enable-prefix-caching",
                    *tail,
                    "--max-num-seqs",
                    "256",
                    "--max-num-batched-tokens",
                    "8192",
                    "--no-enable-log-requests",
                ],
                {**env, "VLLM_API_KEY": "k"},
            ),
        )

    def test_no_probe_starts_a_process(self):
        for path in sorted(PROBES.glob("*.py")):
            calls = {
                f"{n.func.value.id}.{n.func.attr}"
                for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
            }
            self.assertFalse(
                calls
                & {"subprocess.Popen", "launcher.start"} - ({"launcher.start"} if path.name == "base.py" else set()),
                path.name,
            )


class StageSplitKey(unittest.TestCase):
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

        with mock.patch.object(sss.aiohttp, "ClientSession", Session), self.assertRaises(Stop):
            sss.asyncio.run(sss.run_sweep("http://h:1", "llama", [8], [0], 1, "/dev/null", api_key="sk-live"))
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
        from quettabench.probes import serving_stage_split as sss

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stop:
            sss.main(["--model", "/w", "--no-launch"])
        self.assertEqual(stop.exception.code, 2)
        self.assertNotIn("H100", (PROBES / "serving_stage_split.py").read_text().split("def main")[1])


def import_without_torch(name):
    fake = types.ModuleType("torch")
    fake.nn = types.ModuleType("torch.nn")
    fake.nn.functional = types.ModuleType("torch.nn.functional")
    fake.bfloat16 = "bfloat16"
    fake.cuda = types.SimpleNamespace(is_available=lambda: True)
    fake.version = types.SimpleNamespace(hip=None)
    with mock.patch.dict(sys.modules, {"torch": fake, "torch.nn": fake.nn, "torch.nn.functional": fake.nn.functional}):
        sys.modules.pop(f"quettabench.probes.{name}", None)
        return importlib.import_module(f"quettabench.probes.{name}")


def test_gemm_ncu_reads_kernels_through_the_shared_parser():
    gemm = import_without_torch("gemm")
    launches = [
        ("distribution_elementwise_kernel", 1.0),
        ("nvjet_tst_64x8", 5.0),
        ("ampere_bf16_s16816gemm", 4.0),
        ("at::fill", 0.5),
    ]
    with mock.patch.object(gemm._timing, "ncu_kernel_durations", return_value=launches) as durs:
        assert gemm.ncu_gemm_min_us(1, 4096, 4096, "/opt/ncu") == 4.0
    assert durs.call_args.args[1] == "/opt/ncu"
    with mock.patch.object(gemm._timing, "ncu_kernel_durations", return_value=[("at::fill", 1.0)]):
        assert gemm.ncu_gemm_min_us(1, 4096, 4096, "/opt/ncu") is None
    assert "Metric Value" not in (PROBES / "gemm.py").read_text()


def test_ncu_without_counter_access_raises_its_error():
    timing = import_without_torch("gemm")._timing
    denied = (
        "==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters\n"
    )
    with mock.patch.object(timing.subprocess, "run", return_value=mock.Mock(stdout=denied)):
        with pytest.raises(RuntimeError, match="ERR_NVGPUCTRPERM"):
            timing.ncu_kernel_durations("pass", "ncu")


def test_elementwise_floor_survives_a_cache_knee():
    """residual_add measured on an RTX PRO 6000: flat while L2 holds the data, then memory bound."""
    mb = [1.6, 3.1, 6.3, 12.6, 25.2, 50.3, 100.7, 201.3, 402.7, 805.3, 1610.6, 3221.2]
    us = [7.14, 7.22, 7.23, 7.07, 7.14, 10.56, 16.74, 138.99, 271.26, 542.50, 1084.26, 2168.69]
    floor, bw = import_without_torch("elementwise").fit([m * 1e6 for m in mb], us)
    assert floor == 7.22
    assert 1450 < bw < 1520


@pytest.mark.parametrize("name", ["gemm", "moe"])
def test_append_cannot_be_abbreviated(name, capsys):
    """export reads --append from probe_args as written; --app would append unseen."""
    with pytest.raises(SystemExit) as stop:
        import_without_torch(name).main(["--gpu-label", "H100", "--app"])
    assert stop.value.code == 2 and "unrecognized" in capsys.readouterr().err


@pytest.mark.parametrize(
    "name,flags,said",
    [
        ("moe", ["--geometries", "nope"], "unknown --geometries"),
        ("gemm", ["--model-shapes", "nope"], "unknown --model-shapes"),
        ("gemm", ["--dtype", "fp8", "--ncu"], "no --ncu path"),
    ],
)
def test_bad_probe_flags_fail_at_parse_time(name, flags, said, capsys):
    with pytest.raises(SystemExit) as stop:
        import_without_torch(name).main(["--gpu-label", "H100", *flags])
    assert stop.value.code == 2 and said in capsys.readouterr().err


def test_event_time_us_warms_up_then_reduces_over_reps():
    timing = import_without_torch("_timing")
    elapsed_ms = iter([1.0, 5.0, 3.0, 1.0, 5.0, 3.0])

    class Event:
        def __init__(self, enable_timing):
            pass

        def record(self):
            pass

        def elapsed_time(self, other):
            return next(elapsed_ms)

    timing.torch.cuda = types.SimpleNamespace(is_available=lambda: True, Event=Event, synchronize=lambda: None)
    calls = []
    assert timing.event_time_us(lambda: calls.append(1), 2, 3, min) == 1000.0
    assert len(calls) == 5
    assert timing.event_time_us(lambda: None, 0, 3, statistics.median) == 3000.0


def timing_on(available, hip=None):
    timing = import_without_torch("_timing")
    timing.torch.cuda = types.SimpleNamespace(is_available=lambda: available)
    timing.torch.version = types.SimpleNamespace(hip=hip)
    return timing


@pytest.mark.parametrize("available,hip,want", [(True, None, "cuda"), (True, "6.2", "rocm"), (False, None, "cpu")])
def test_backend_follows_the_torch_build(available, hip, want):
    timing = timing_on(available, hip)
    assert timing.backend() == want
    assert timing.device() == ("cuda" if available else "cpu")


def test_backend_names_a_tpu_or_neuron_plugin():
    timing = timing_on(False)
    for module, want in (("torch_xla", "tpu"), ("torch_neuronx", "neuron")):
        with mock.patch.object(timing.importlib.util, "find_spec", side_effect=lambda m, module=module: m == module):
            assert timing.backend() == want


def test_timing_without_a_gpu_is_wall_clock_and_sync_is_a_no_op():
    timing = timing_on(False)
    calls = []
    assert timing.event_time_us(lambda: calls.append(1), 2, 3, len) == 3
    assert len(calls) == 5
    assert timing.graph_time_us(lambda: calls.append(1), warmup=1, reps=2, reduce=len) == 2
    timing.sync()
    timing.empty_cache()
    timing.set_device(0)
    assert timing.device_name() == "cpu"


def test_probe_device_layer_leaves_the_launcher_visible_device_names():
    from quettabench.engines import launcher

    assert timing_on(True, "6.2").backend() == "rocm"
    assert launcher.visible_env("cuda", "0,1") == ({"CUDA_VISIBLE_DEVICES": "0,1"}, ())
    assert launcher.visible_env("rocm", "2")[0] == {"HIP_VISIBLE_DEVICES": "2", "CUDA_VISIBLE_DEVICES": "2"}


@pytest.mark.parametrize(
    "name,flags,backend,said",
    [
        ("gemm", ["--ncu"], "rocm", "rocprofv3"),
        ("gemm", ["--dtype", "fp8"], "cpu", "gemm --dtype fp8 runs on cuda, not cpu"),
        ("moe", [], "cpu", "moe runs on cuda, rocm, not cpu"),
        ("elementwise", ["--vllm"], "cpu", "elementwise --vllm runs on cuda, rocm, not cpu"),
        ("elementwise", [], "tpu", "elementwise runs on cuda, rocm, cpu, not tpu"),
        ("moe", [], "neuron", "moe runs on cuda, rocm, not neuron"),
    ],
)
def test_probe_refuses_a_backend_it_does_not_support(name, flags, backend, said):
    probe = import_without_torch(name)
    with mock.patch.object(probe._timing, "backend", return_value=backend):
        with pytest.raises(SystemExit, match=said):
            probe.main(["--gpu-label", "H100", *flags])


SUPPORTED = {
    "gemm": ("cuda", "rocm", "cpu"),
    "elementwise": ("cuda", "rocm", "cpu"),
    "moe": ("cuda", "rocm"),
    "gdn": ("cuda", "rocm"),
    "flash_attn": ("cuda", "rocm"),
    "chunked_prefill_attn": ("cuda", "rocm"),
    "collective": ("cuda", "rocm"),
    "reparallel": ("cuda", "rocm"),
    "vllm_allreduce": ("cuda",),
}


def declared_backends(path):
    """The (probe, backends...) of every plain require_backend call in the file."""
    calls = (n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.Call))
    return [
        tuple(a.value for a in c.args)
        for c in calls
        if getattr(c.func, "attr", None) == "require_backend" and all(isinstance(a, ast.Constant) for a in c.args)
    ]


@pytest.mark.parametrize("name", sorted(SUPPORTED))
def test_each_probe_declares_its_backends_once(name):
    assert (name, *SUPPORTED[name]) in declared_backends(PROBES / f"{name}.py")


def test_only_the_timing_layer_calls_torch_cuda():
    pkg = PROBES.parent
    files = [*(pkg / "probes").glob("*.py"), *(pkg / "profiling").glob("*.py")]
    assert len(files) > 10
    assert [f.name for f in files if f.name != "_timing.py" and "torch.cuda" in f.read_text()] == []


def test_kv_transfer_declares_cuda_only():
    assert ("kv_transfer", "cuda") in declared_backends(PROBES.parent / "profiling" / "kv_transfer.py")


def run_on_cpu(tmp_path, name, *flags):
    """A child with no visible GPU, so a GPU box runs the cpu path too."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""}
    cmd = [sys.executable, "-m", f"quettabench.probes.{name}", "--gpu-label", "T", "--out", str(tmp_path), *flags]
    subprocess.run(cmd, check=True, timeout=300, cwd=PROBES.parents[1], env=env, capture_output=True)
    return [json.loads(line) for f in tmp_path.rglob("rows.jsonl") for line in f.read_text().splitlines()]


def test_elementwise_and_gemm_run_on_cpu_through_the_device_layer(tmp_path):
    pytest.importorskip("torch")
    elem = run_on_cpu(tmp_path / "e", "elementwise", "--max-mem-gb", "0.01")
    assert {r["key"]["entry"] for r in elem} == {"rmsnorm", "silu_mul", "rotary_emb", "residual_add"}
    gemm = run_on_cpu(tmp_path / "g", "gemm", "--pairs", "8:8", "--m-axis", "1,4")
    assert sorted(r["key"]["M"] for r in gemm) == [1, 4]
    assert all(r["value"]["latency_us"] > 0 for r in gemm)


def test_moe_routing_flag_takes_the_routing_parser(capsys):
    with pytest.raises(SystemExit) as stop:
        import_without_torch("moe").main(["--routing", "--gpu-label", "H100"])
    assert stop.value.code == 2 and "--model" in capsys.readouterr().err


def test_probe_flags_cannot_be_abbreviated():
    ap = argparse.ArgumentParser()
    probes_base.add_output_args(ap)
    for argv in (["--gpu-l", "H100"], ["--gpu", "H100", "--ru", "x"]):
        with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
            ap.parse_args(argv)
    assert ap.parse_args(["--gpu", "H100"]).gpu_label == "H100"


@pytest.mark.parametrize(
    "name,out",
    [("serving_stage_split", ["--out", "x.csv"]), ("serving_decode_grid", ["--out-raw", "r", "--out-summary", "s"])],
)
def test_probes_without_output_args_take_no_abbreviation_either(name, out, capsys):
    probe = importlib.import_module(f"quettabench.probes.{name}")
    with pytest.raises(SystemExit) as stop:
        probe.main(["--mod", "/w", "--no-launch", *out])
    assert stop.value.code == 2 and "unrecognized" in capsys.readouterr().err


def test_record_dir_takes_no_abbreviation(tmp_path):
    argv = [
        "--run-id",
        "20260101T000000.000000Z",
        "--gpu-label",
        "H100",
        "--out",
        str(tmp_path / "a"),
        "--out-d",
        str(tmp_path / "b"),
    ]
    assert probes_base.record_dir("flash_attn", argv) == tmp_path / "a" / "flash_attn" / argv[1]


ROOT = Path(__file__).resolve().parents[2]
JUSTFILE = ROOT / "justfile"
PROBES_DIR = ROOT / "quettabench" / "probes"
PROFILING_DIR = ROOT / "quettabench" / "profiling"


def argparse_flags(probe: Path) -> set[str]:
    flags = set()
    if probe.name != "base.py" and "add_output_args" in probe.read_text():
        flags |= argparse_flags(PROBES_DIR / "base.py")
    for node in ast.walk(ast.parse(probe.read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            flags.update(
                arg.value for arg in node.args if isinstance(arg, ast.Constant) and str(arg.value).startswith("--")
            )
    return flags


def defined_names(module: Path) -> set[str]:
    names = set()
    for node in ast.parse(module.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def recipe_blocks(justfile: Path) -> list[str]:
    text = justfile.read_text().replace("\\\n", " ")
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if line and not line[0].isspace() and current:
            blocks.append(current)
            current = []
        current.append(line)
    if current:
        blocks.append(current)
    return ["\n".join(b) for b in blocks]


def probe_invocations() -> list[tuple[Path, set[str]]]:
    out = []
    for block in recipe_blocks(JUSTFILE):
        probes = [PROBES_DIR / f"{m}.py" for m in re.findall(r"-m quettabench\.probes\.([a-z0-9_]+)", block)]
        probes += [PROFILING_DIR / f"{m}.py" for m in re.findall(r"-m quettabench\.profiling\.([a-z0-9_]+)", block)]
        flags = {t for t in block.split() if t.startswith("--") and t != "--" and "=" not in t}
        out.extend((probe, flags) for probe in probes)
    return out


def test_every_probe_parses():
    probes = sorted(PROBES_DIR.glob("*.py"))
    assert probes, f"no probes found under {PROBES_DIR}"
    for probe in probes:
        ast.parse(probe.read_text(), filename=str(probe))


def test_recipe_probes_exist_and_accept_their_flags():
    invocations = probe_invocations()
    assert invocations, "no probe invocations found in the justfile"
    for probe, flags in invocations:
        assert probe.is_file(), f"justfile invokes {probe.name} but {probe} is missing"
        defined = argparse_flags(probe)
        missing = sorted(flags - defined)
        assert not missing, f"{probe.name} does not define recipe flags: {missing}"


LIVE_SERVER_PROBES = {"serving_decode_grid.py", "serving_stage_split.py"}


def test_every_probe_labels_its_gpu():
    for probe in sorted(PROBES_DIR.glob("[!_]*.py")):
        if probe.name in LIVE_SERVER_PROBES:
            continue
        assert "--gpu-label" in argparse_flags(probe), f"{probe.name} has no --gpu-label"


def test_sibling_helper_imports_resolve():
    for probe in sorted(PROBES_DIR.glob("[!_]*.py")):
        for node in ast.walk(ast.parse(probe.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("quettabench.probes."):
                module = node.module.rsplit(".", 1)[-1]
                helper = PROBES_DIR / f"{module}.py"
                assert helper.is_file(), f"{probe.name} imports {node.module}, no {helper.name}"
                exported = defined_names(helper)
                for alias in node.names:
                    assert alias.name in exported, f"{probe.name} imports {node.module}.{alias.name}, not defined there"


def test_every_moved_probe_takes_out():
    for probe in sorted(PROBES_DIR.glob("[!_]*.py")):
        if probe.name != "base.py" and probe.name not in LIVE_SERVER_PROBES:
            assert "--out" in argparse_flags(probe), f"{probe.name} has no --out"


def test_remote_recipe_is_one_line_into_base():
    block = next(b for b in recipe_blocks(JUSTFILE) if b.startswith("remote-probe "))
    body = [ln.strip() for ln in block.splitlines()[1:] if ln.strip() and not ln.strip().startswith("#")]
    assert len(body) == 1 and "-m quettabench.probes.base remote-probe" in body[0], body
    assert "rm -rf" not in block and "/tmp/qbench-probes" not in block


def test_remote_probe_ships_what_base_imports():
    from quettabench.probes.base import SHIPPED

    assert "quettabench/probes/" in SHIPPED and "quettabench/__init__.py" in SHIPPED
    with tempfile.TemporaryDirectory() as td:
        for rel in SHIPPED:
            src, dst = ROOT / rel, Path(td) / rel
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); import quettabench.probes.base as b; "
            "assert b.__file__.startswith(sys.argv[1]), b.__file__"
        )
        proc = subprocess.run([sys.executable, "-I", "-c", code, td], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


class FakeHost:
    def __init__(self, rdir="/tmp/qbench-probes.Ab12Cd", probe_rc=0, pull_rc=0):
        self.rdir, self.probe_rc, self.pull_rc, self.calls = rdir, probe_rc, pull_rc, []

    def __call__(self, cmd, check=False, **kw):
        self.calls.append(cmd)
        rc = 0
        if cmd[0] == "rsync" and cmd[-2].endswith("/out/"):
            rc = self.pull_rc
        elif cmd[0] == "ssh" and "quettabench.probes." in cmd[-1]:
            rc = self.probe_rc
        if check and rc:
            raise subprocess.CalledProcessError(rc, cmd)
        if cmd[0] == "rsync" and cmd[-2].endswith("/out/") and not rc:
            self.pulled(Path(cmd[-1]))
        return subprocess.CompletedProcess(cmd, rc, stdout="banner\n" + self.rdir + "\n" if "mktemp" in cmd[-1] else "")

    def pulled(self, local):
        """What the pull brings back: nothing, as from an empty out/."""


REMOTE = ["--host", "a100", "--py", "/v/bin/python", "--gpu-id", "3"]
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "a100"]


def remote(argv, host, tmp_path, monkeypatch):
    from quettabench.probes.base import remote_probe

    monkeypatch.chdir(tmp_path)
    return remote_probe([*REMOTE, *argv], run=host, now=NOW)


def test_remote_probe_runs_in_its_own_dir_and_always_pulls(tmp_path, monkeypatch):
    host = FakeHost(probe_rc=3)
    rc = remote(["gemm", "A100", "--probe-args=--ncu --wide"], host, tmp_path, monkeypatch)
    rdir = host.rdir
    assert rc == 3
    mktemp, ship, probe, pull, rm = host.calls
    assert mktemp == [*SSH, "d=$(mktemp -d /tmp/qbench-probes.XXXXXX) && mkdir -p $d/out && echo $d"]
    assert ship[:4] == ["rsync", "-a", "--relative", "-e"] and ship[-1] == f"a100:{rdir}/"
    assert probe[-1] == (
        f"cd {rdir} && CUDA_VISIBLE_DEVICES=3 /v/bin/python -m quettabench.probes.gemm "
        f"--gpu-label A100 --out {rdir}/out --run-id 20261011T120000.123456Z --ncu --wide"
    )
    assert pull[-2:] == [f"a100:{rdir}/out/", "results/probes/A100/"] and "--ignore-existing" in pull
    assert rm == [*SSH, f"rm -rf {rdir}"]
    assert (tmp_path / "results" / "probes" / "A100").is_dir()


def test_remote_probe_under_torchrun_and_ncu(tmp_path, monkeypatch):
    host = FakeHost()
    remote(["--nproc", "2", "--gpu-ids", "0,1", "collective", "H100"], host, tmp_path, monkeypatch)
    assert (
        "CUDA_VISIBLE_DEVICES=0,1 /v/bin/python -m torch.distributed.run --nproc_per_node=2 "
        "-m quettabench.probes.collective --gpu-label H100" in host.calls[2][-1]
    )
    host = FakeHost()
    remote(["--ncu-bin", "/opt/ncu", "gemm", "H100"], host, tmp_path, monkeypatch)
    assert "CUDA_VISIBLE_DEVICES=3 NCU_BIN=/opt/ncu /v/bin/python -m quettabench.probes.gemm" in host.calls[2][-1]


def test_a_failed_pull_keeps_the_remote_dir(tmp_path, monkeypatch):
    host = FakeHost(pull_rc=23)
    with pytest.raises(SystemExit, match="kept"):
        remote(["gemm", "A100"], host, tmp_path, monkeypatch)
    assert not any("rm -rf" in c[-1] for c in host.calls)


def test_a_pull_that_raises_still_says_the_remote_dir_is_kept(tmp_path, monkeypatch, capsys):
    class PullRaises(FakeHost):
        def __call__(self, cmd, check=False, **kw):
            if cmd[0] == "rsync" and cmd[-2].endswith("/out/"):
                raise KeyboardInterrupt
            return super().__call__(cmd, check=check, **kw)

    host = PullRaises(probe_rc=4)
    with pytest.raises(KeyboardInterrupt):
        remote(["gemm", "A100"], host, tmp_path, monkeypatch)
    assert f"a100:{host.rdir}" in capsys.readouterr().err
    assert not any("rm -rf" in c[-1] for c in host.calls)


def test_a_failed_pull_names_the_probe_exit_code(tmp_path, monkeypatch):
    host = FakeHost(probe_rc=5, pull_rc=23)
    with pytest.raises(SystemExit, match="probe exit 5"):
        remote(["gemm", "A100"], host, tmp_path, monkeypatch)


class Pulls(FakeHost):
    def __init__(self, record, **kw):
        super().__init__(**kw)
        self.record = record

    def pulled(self, local):
        run = local / "gemm" / "20261011T120000.123456Z"
        run.mkdir(parents=True)
        (run / "run.json").write_text(self.record)


def test_the_pulled_run_is_stamped_on_the_local_clock(tmp_path, monkeypatch):
    """The remote clock may be skewed: pulled gets this machine's clock."""
    remote(["gemm", "A100"], Pulls(json.dumps({"provenance": {"finished": "2099-01-01"}})), tmp_path, monkeypatch)
    doc = json.loads((tmp_path / "results/probes/A100/gemm/20261011T120000.123456Z/run.json").read_text())
    assert doc["provenance"]["finished"] == "2099-01-01" and "remote_finished" not in doc["provenance"]
    assert datetime.fromisoformat(doc["provenance"]["pulled"]) < datetime(2099, 1, 1, tzinfo=timezone.utc)


@pytest.mark.parametrize("record", ["{", '{"provenance": null}'])
def test_a_bad_pulled_run_json_warns_and_the_remote_dir_still_goes(record, tmp_path, monkeypatch, capsys):
    host = Pulls(record, probe_rc=2)
    assert remote(["gemm", "A100"], host, tmp_path, monkeypatch) == 2
    assert "20261011T120000.123456Z" in capsys.readouterr().err
    assert host.calls[-1] == [*SSH, f"rm -rf {host.rdir}"]


def test_probe_args_pass_through_intact(tmp_path, monkeypatch):
    host = FakeHost()
    remote(["--", "gemm", "A100", "--model-shapes", 'a b$c"d'], host, tmp_path, monkeypatch)
    assert host.calls[2][-1].endswith("--model-shapes 'a b$c\"d'")


def test_the_justfile_passes_remote_probe_args_as_they_are():
    text = JUSTFILE.read_text()
    assert "set positional-arguments" in text
    block = next(b for b in recipe_blocks(JUSTFILE) if b.startswith("remote-probe "))
    assert block.rstrip().endswith('-- "$@"') and "{{args}}" not in block


def test_a_failed_ship_removes_the_remote_dir(tmp_path, monkeypatch):
    class ShipFails(FakeHost):
        def __call__(self, cmd, check=False, **kw):
            if cmd[:3] == ["rsync", "-a", "--relative"]:
                self.calls.append(cmd)
                raise subprocess.CalledProcessError(12, cmd)
            return super().__call__(cmd, check=check, **kw)

    host = ShipFails()
    with pytest.raises(subprocess.CalledProcessError):
        remote(["gemm", "A100"], host, tmp_path, monkeypatch)
    assert host.calls[-1][-1] == f"rm -rf {host.rdir}"


@pytest.mark.parametrize("rdir", ["", "/tmp", "/tmp/qbench-probes", "/tmp/qbench-probes.x y", "/"])
def test_an_odd_mktemp_answer_is_never_removed(rdir, tmp_path, monkeypatch):
    host = FakeHost(rdir=rdir)
    with pytest.raises(SystemExit, match="mktemp"):
        remote(["gemm", "A100"], host, tmp_path, monkeypatch)
    assert len(host.calls) == 1


@pytest.mark.parametrize(
    "argv",
    [
        ["../x", "A100"],
        ["gemm", "A 100"],
        ["gemm", "A100", "--host", ""],
        ["gemm", "A100", "--host=-oProxyCommand=x"],
        ["gemm", "A100", "--probe-args=--out /x"],
        ["gemm", "A100", "--probe-args=--run-id=1"],
        ["gemm", "A100", "--probe-args=--gpu-label H100"],
        ["--", "gemm", "A100", "--gpu=H100"],
        ["--", "gemm", "A100", "--out-dir", "/x"],
        ["gemm", "A100", "--probe-args=--out-d /x"],
        ["gemm", "A100", "--probe-args=--gpu-l H100"],
        ["gemm", "A100", "--probe-args=--ru=1"],
        ["--", "gemm", "A100", "--o", "/x"],
    ],
)
def test_remote_probe_refuses_odd_names(argv, tmp_path, monkeypatch):
    host = FakeHost()
    with pytest.raises(SystemExit):
        remote(argv, host, tmp_path, monkeypatch)
    assert host.calls == []


def test_the_r2_mirror_has_its_own_prefix_and_never_deletes():
    text = JUSTFILE.read_text()
    assert 'r2_prefix := "profiling/qbench_probes"' in text
    assert "--delete" not in text and "mirror-up-prune" not in text
    assert "kernel_data" not in "".join(ln for ln in text.splitlines() if "r2_" in ln)


def test_probes_import_their_siblings_at_the_top():
    for probe in sorted(PROBES_DIR.glob("*.py")):
        text = probe.read_text()
        assert "noqa: PLC0415" not in text, probe.name
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                inner = [n.module for n in ast.walk(node) if isinstance(n, ast.ImportFrom)]
                assert not any(m and m.startswith("quettabench.probes._") for m in inner), f"{probe.name}:{node.name}"


def inline_imports_without_a_reason(path: Path) -> list[str]:
    text = path.read_text()
    lines = text.splitlines()
    bare = []
    for fn in ast.walk(ast.parse(text)):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        doc = (ast.get_docstring(fn) or "").lower()
        for node in ast.walk(fn):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            above = node.lineno - 2
            while above >= 0 and lines[above].strip().startswith(("from ", "import ")):
                above -= 1
            comment = [ln.lower() for ln in lines[node.lineno - 1 : node.end_lineno] if "#" in ln]
            while above >= 0 and lines[above].strip().startswith("#"):
                comment.append(lines[above].lower())
                above -= 1
            if not any("import" in c for c in comment) and "import" not in doc:
                bare.append(f"{path.name}:{node.lineno}")
    return bare


def test_every_inline_import_states_why():
    bare = []
    for path in sorted([*PROBES_DIR.glob("*.py"), *PROFILING_DIR.rglob("*.py")]):
        assert "noqa: PLC0415" not in path.read_text(), path.name
        bare += inline_imports_without_a_reason(path)
    assert bare == []


def test_moe_docstrings_name_only_real_backends():
    text = (PROBES_DIR / "moe.py").read_text()
    assert set(re.findall(r"--backend (\w+)", text)) <= {"triton", "flashinfer"}


RUNBOOKS = sorted((ROOT / "docs" / "runbooks").glob("*.md"))


def runbook_drift(path: Path) -> list[str]:
    found = []
    recipes = {b.split()[0].rstrip(":") for b in recipe_blocks(JUSTFILE)}
    text = path.read_text().replace("\\\n", " ")
    for n, line in enumerate(text.splitlines(), 1):
        where = f"{path.name}:{n}"
        if "--api-key test" in line:
            found.append(f"{where} --api-key test")
        for probe in re.findall(r"-m quettabench\.probes\.([a-z0-9_]+)", line):
            flags = {t.strip("`") for t in line.split() if t.startswith("--") and "=" not in t}
            missing = flags - argparse_flags(PROBES_DIR / f"{probe}.py")
            if missing:
                found.append(f"{where} {probe} lacks {sorted(missing)}")
        for recipe in re.findall(r"`just (?:[a-z_]+=\S+ )*([a-z][a-z0-9-]*)", line):
            if recipe not in recipes:
                found.append(f"{where} no recipe {recipe}")
        for script in re.findall(r"((?:quettabench/)?(?:profiling|scripts)/[\w./-]+\.py)", line):
            if not (ROOT / script).exists():
                found.append(f"{where} no {script}")
    return found


def test_runbook_commands_still_run_here():
    assert [d for path in RUNBOOKS for d in runbook_drift(path)] == []


def test_emitting_the_nsys_command_records_no_run(tmp_path, capsys):
    from quettabench.probes import serving_stage_split as sss

    assert sss.main(["--model", "/w", "--emit-nsys-cmd", "--out", str(tmp_path / "x.csv")]) is False
    assert list(tmp_path.iterdir()) == []


if __name__ == "__main__":
    unittest.main()
