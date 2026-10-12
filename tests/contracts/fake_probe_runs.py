#!/usr/bin/env python3
"""Every kernel probe's main on CPU, then kernel_tables: the identity proof's driver.

Usage: fake_probe_runs.py <code root> <work dir> <dest seed json>. torch, vLLM
and NIXL are mocks and each timing call returns the next value of one counter,
so two code trees that time the same points in the same order export the same
tables. Writes <work>/data (seeded, then exported over twice), <work>/fresh and <work>/stop.txt.
"""

import contextlib
import io
import json
import os
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

CODE, WORK, SEED = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(CODE))
T0 = datetime(2026, 10, 12, 9, 0, 0, tzinfo=timezone.utc)
STEP = [0]
CLOCK = [0]


def tick() -> float:
    """The next synthetic timing, in the probe's own unit."""
    STEP[0] += 1
    return 3.0 + STEP[0] * 0.7317


def now() -> str:
    CLOCK[0] += 1
    return (T0 + timedelta(minutes=CLOCK[0])).isoformat()


def fake_module(name: str) -> mock.MagicMock:
    module = mock.MagicMock(name=name)
    module.__name__, module.__path__ = name, []
    return module


torch = fake_module("torch")
torch.cuda.get_device_name.return_value = "FAKE"
torch.cuda.device_count.return_value = 8
torch.randn.return_value.numel.return_value = 4096
dist = fake_module("torch.distributed")
dist.get_rank.return_value = 0
dist.get_world_size.return_value = 2
torch.distributed = dist
KV_ROWS = [{"gpu": "H200", "kv_heads_per_rank": h, "head_dim": 128, "block_size": 16, "layers": 94,
            "layout": layout, "tokens": t, "n_descs": 94 * t // 16, "bytes": 49152 * t,
            "prep_ms": 0.01 * t, "xfer_ms": 0.5, "total_ms": round(0.01 * t + 0.5, 3), "eff_gbps": 1.25}
           for h in (2, 1) for layout in ("runs8", "contig") for t in (16, 1024)]


def broadcast_object_list(objs, src=0, **kw):
    """kv_transfer's rank 0 receives the rows rank 1 measured."""
    if src == 1:
        objs[0] = [dict(r) for r in KV_ROWS]


dist.broadcast_object_list.side_effect = broadcast_object_list
fakes = {"torch": torch, "torch.distributed": dist, "torch.nn": torch.nn,
         "torch.nn.functional": torch.nn.functional, "triton": fake_module("triton")}
for name in ("vllm", "vllm.config", "vllm.distributed", "vllm.vllm_flash_attn",
             "vllm.vllm_flash_attn.flash_attn_interface", "vllm.model_executor",
             "vllm.model_executor.layers", "vllm.model_executor.layers.fla",
             "vllm.model_executor.layers.fla.ops", "vllm.model_executor.layers.fused_moe",
             "vllm.model_executor.layers.fused_moe.moe_align_block_size", "nixl", "nixl._api"):
    fakes[name] = fake_module(name)
sys.modules.update(fakes)

from quettabench.export import quettasim  # noqa: E402
from quettabench.metrics import results  # noqa: E402
from quettabench.probes import base  # noqa: E402

SEQ = mock.Mock(side_effect=lambda *a, **k: tick())
GDN_CALLS = [0]


def gdn_time(*a, **k) -> float:
    """gdn's second and fifth points fail, as its illegal-memory decode bug does."""
    GDN_CALLS[0] += 1
    if GDN_CALLS[0] in (2, 5):
        raise RuntimeError("illegal memory access")
    return tick()


FP8 = types.SimpleNamespace(quant_fit=lambda: (1.2345678, 812.25), quant_pts=[1, 2, 3])
TIMING = {
    "gemm": {"time_matmul": SEQ, "time_matmul_fp8": SEQ, "ncu_gemm_us": SEQ,
             "fp8_linear_op": lambda: FP8},
    "moe": {"_time": SEQ, "_fused_experts": lambda: (mock.MagicMock(), lambda *a: (1, 2, 3))},
    "elementwise": {"time_op": SEQ, "graph_median_us": SEQ, "op_for": mock.Mock(),
                    "op_for_vllm": mock.Mock()},
    "flash_attn": {"time_call": SEQ, "graph_median_us": SEQ, "_ncu_us": SEQ},
    "cross_attn": {"time_call": SEQ},
    "gdn": {"time_call": gdn_time, "ncu_op_us": SEQ, "mk": lambda *a: (mock.MagicMock(),) * 5},
    "collective": {"_time_op": mock.Mock(side_effect=lambda *a: tick() * 1e-6)},
    "vllm_allreduce": {"_time": SEQ},
    "reparallel": {"_time_ms": SEQ},
    "kv_transfer": {},
}
TORCHRUN = {"collective", "vllm_allreduce", "reparallel", "kv_transfer"}
PAIRS = ["--pairs", "1024:1024", "1024:2048", "--m-axis", "1,64,3"]
# (probe, flags) per run; the second batch lands in another results dir.
BATCHES = [
    [("gemm", ["--graph", "--append", *PAIRS]),
     ("gemm", ["--graph", "--append", "--dtype", "fp8", "--pairs", "128:4096", "--m-axis", "1,2,4"]),
     ("gemm", ["--ncu", "--append"]),
     ("gemm", ["--wide", "--append", "--pairs", "1024:1024", "--m-axis", "1,64"]),
     ("elementwise", ["--graph", "--vllm"]),
     ("elementwise", ["--vllm"]),
     ("moe", ["--graph", "--append", "--geometries", "qwen3-30b-a3b", "--tp", "1", "--tokens", "1,2,4",
              "--max-mem-gb", "1000"]),
     ("flash_attn", ["--n-heads", "32", "--n-kv-heads", "4", "--tag", "tp1", "--fa-version", "3",
                     "--layers", "48", "--graph", "--kv-axis", "128,256", "--batch-axis", "1,2"]),
     ("cross_attn", ["--n-heads", "16", "--n-kv-heads", "2", "--fa-version", "3", "--q-axis", "512",
                     "--ctx-axis", "0,1024,99"]),
     ("gdn", []),
     ("collective", ["--hidden", "4096", "--layers", "32", "--ops", "all_reduce,all_gather"]),
     ("vllm_allreduce", []),
     ("reparallel", ["--switches", "2:1", "4:8", "--weights-gb", "1", "3"]),
     ("kv_transfer", [])],
    [("gemm", ["--graph", *PAIRS]),
     ("moe", ["--graph", "--geometries", "qwen3-30b-a3b", "--tp", "1", "--tokens", "1,8",
              "--max-mem-gb", "1000", "--skip-routing"]),
     ("flash_attn", ["--n-heads", "32", "--n-kv-heads", "4", "--tag", "tp1", "--fa-version", "3",
                     "--ncu", "--kv-axis", "512", "--batch-axis", "1"]),
     ("elementwise", ["--graph", "--vllm"]),
     ("collective", ["--hidden", "4096", "--layers", "32", "--ops", "all_reduce,all_gather"])],
]


def run(results_dir: Path, probe: str, flags: list[str], at: int) -> None:
    package = "profiling" if probe == "kv_transfer" else "probes"
    module = __import__(f"quettabench.{package}.{probe}", fromlist=["main"])
    env = {"RANK": "0", "WORLD_SIZE": "8" if probe == "reparallel" else "2", "LOCAL_RANK": "0"}
    argv = ["probe", "--gpu-label", "H200", "--out", str(results_dir / "probes" / "H200"), *flags]
    with contextlib.ExitStack() as stack:
        for name, fake in TIMING[probe].items():
            stack.enter_context(mock.patch.object(module, name, fake))
        stack.enter_context(mock.patch.dict(os.environ, env if probe in TORCHRUN else {}))
        if probe not in TORCHRUN:
            os.environ.pop("RANK", None)
        stack.enter_context(mock.patch.object(sys, "argv", argv))
        base.run_recorded(module.main, probe, now=T0 + timedelta(seconds=at))


def main() -> None:
    seed = json.loads(SEED.read_text())
    data = WORK / "data"
    for rel, text in seed.items():
        path = data / "kernel_data" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode())
    dirs = [WORK / "results1", WORK / "results2"]
    with mock.patch.object(results, "_now", now), contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        at = 0
        for results_dir, batch in zip(dirs, BATCHES):
            for probe, flags in batch:
                at += 1
                run(results_dir, probe, flags, at)
        try:
            quettasim.export(dirs[:1], kernel_dest=data, only=["kernel_tables"])
            stop = "no stop"
        except SystemExit as e:
            stop = str(e)
        quettasim.export(dirs[:1], kernel_dest=data, only=["kernel_tables"], replace_tables=True)
        quettasim.export(dirs, kernel_dest=data, only=["kernel_tables"], replace_tables=True)
        quettasim.export(dirs, kernel_dest=WORK / "fresh", only=["kernel_tables"])
    (WORK / "stop.txt").write_text(stop.replace(str(WORK), "<WORK>") + "\n")


if __name__ == "__main__":
    main()
