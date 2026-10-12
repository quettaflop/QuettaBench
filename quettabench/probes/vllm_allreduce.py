#!/usr/bin/env python3
"""TP all-reduce through vLLM's GroupCoordinator next to plain NCCL, under torchrun.
Rank 0 writes the vLLM path as all_reduce rows of collectives/{gpu}_tp{N}.csv and both paths to ..._nccl_vs_vllm.csv.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

from quettabench.probes.base import ProbeRow, add_output_args, emit, run_recorded

_ITERS, _WARMUP = 50, 10
# Full logical buffer bytes: decode payloads (batch x hidden x 2B) up to prefill chunks (8192 x 4096 x 2B = 64 MB).
BYTES_AXIS = [
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
    2097152,
    4194304,
    8388608,
    16777216,
    33554432,
    67108864,
]


def _time(call, iters=_ITERS, warmup=_WARMUP) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        call()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0  # us per call


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_output_args(ap)
    ap.add_argument(
        "--disable-custom-allreduce",
        action="store_true",
        help="mirror a server launched with --disable-custom-all-reduce "
        "(vLLM then routes TP all-reduce through pynccl)",
    )
    a = ap.parse_args(argv)
    rank, world, local = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    # Imported after set_device, the order a vLLM worker uses.
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import get_tp_group, init_distributed_environment, initialize_model_parallel

    # The communicator reads custom-all-reduce enablement from the current vLLM config, as a worker does.
    vcfg = VllmConfig(
        parallel_config=ParallelConfig(tensor_parallel_size=world, disable_custom_all_reduce=a.disable_custom_allreduce)
    )
    with set_current_vllm_config(vcfg):
        init_distributed_environment(world_size=world, rank=rank, distributed_init_method="env://", local_rank=local)
        initialize_model_parallel(tensor_model_parallel_size=world)
        tp = get_tp_group()
    comm = getattr(tp, "device_communicator", None)
    ca = getattr(comm, "ca_comm", None)
    if rank == 0:
        print(
            f"[vllm-ar] world={world} communicator={type(comm).__name__} "
            f"custom_allreduce={'ON' if ca is not None and not getattr(ca, 'disabled', True) else 'off'}",
            flush=True,
        )
    rows = []
    for nbytes in BYTES_AXIS:
        x = torch.ones(nbytes // 2, dtype=torch.bfloat16, device="cuda")
        us_vllm = _time(lambda: tp.all_reduce(x))
        us_nccl = _time(lambda: dist.all_reduce(x))
        rows.append((nbytes, us_vllm, us_nccl))
        if rank == 0:
            print(f"  {nbytes:>9d} B: vllm path {us_vllm:8.2f} us   plain nccl {us_nccl:8.2f} us", flush=True)
    if rank == 0:
        table = f"collectives/{a.gpu_label}_tp{world}.csv"
        for nbytes, us_v, _ in rows:
            emit(
                ProbeRow(
                    "vllm_allreduce",
                    table,
                    {"op": "all_reduce", "world": world, "bytes": nbytes},
                    {"latency_us": round(us_v, 4), "dtype_bytes": 2, "path": "vllm"},
                    "amortized_wallclock",
                )
            )
        print(f"[vllm-ar] {len(rows)} all_reduce rows (vllm path) of {table}", flush=True)
        for nbytes, us_v, us_n in rows:
            emit(
                ProbeRow(
                    "vllm_allreduce",
                    f"collectives/{a.gpu_label}_tp{world}_nccl_vs_vllm.csv",
                    {"bytes": nbytes},
                    {"vllm_us": round(us_v, 4), "nccl_us": round(us_n, 4)},
                    "amortized_wallclock",
                )
            )
    dist.barrier()
    dist.destroy_process_group()
    return rank == 0  # only rank 0 records the run


if __name__ == "__main__":
    run_recorded(main, "vllm_allreduce")
