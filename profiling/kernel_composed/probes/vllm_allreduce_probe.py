#!/usr/bin/env python3
"""TP all-reduce through vLLM's OWN communicator -> the measured collectives grid.

collective_probe.py times torch.distributed (plain NCCL). vLLM does not call that
for tensor-parallel all-reduce: its GroupCoordinator routes small payloads through
the custom all-reduce kernel (one-/two-shot over NVLink P2P) and only large ones to
NCCL. On H200 tp2 the plain-NCCL grid over-priced graphed decode comm by ~1.5 ms/step
(legacy 15.8 us/op x 96 sub-layers), which showed up as a 15-17pp TPOT MAPE gap vs a
zero-latency analytic ring (Mooncake GT, 2026-08-25). This probe measures the path
vLLM actually takes and UPSERTS the all_reduce rows of ncu/collectives/{gpu}_tp{N}.csv
(other ops keep their NCCL rows -- vLLM uses NCCL for those).

  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 vllm_allreduce_probe.py \
      --gpu-label H200 --out-dir <kernel_data>
"""
from __future__ import annotations
import argparse, csv, os
from pathlib import Path
import torch
import torch.distributed as dist

_ITERS, _WARMUP = 50, 10
# Full logical buffer sizes (bytes): decode payloads (batch x hidden x 2B, a few KB..MB)
# up to prefill chunks (8192 x 4096 x 2B = 64 MB).
BYTES_AXIS = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152,
              4194304, 8388608, 16777216, 33554432, 67108864]


def _time(call, iters=_ITERS, warmup=_WARMUP) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        call()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0   # us per call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-label", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--disable-custom-allreduce", action="store_true",
                    help="mirror a server launched with --disable-custom-all-reduce "
                         "(vLLM then routes TP all-reduce through pynccl)")
    a = ap.parse_args()
    rank, world, local = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (get_tp_group, init_distributed_environment,
                                  initialize_model_parallel)
    # The device communicator reads parallel_config (custom-all-reduce enablement)
    # from the current vLLM config, exactly as an engine worker would.
    vcfg = VllmConfig(parallel_config=ParallelConfig(
        tensor_parallel_size=world,
        disable_custom_all_reduce=a.disable_custom_allreduce))
    with set_current_vllm_config(vcfg):
        init_distributed_environment(world_size=world, rank=rank,
                                     distributed_init_method="env://", local_rank=local)
        initialize_model_parallel(tensor_model_parallel_size=world)
        tp = get_tp_group()
    comm = getattr(tp, "device_communicator", None)
    ca = getattr(comm, "ca_comm", None)
    if rank == 0:
        print(f"[vllm-ar] world={world} communicator={type(comm).__name__} "
              f"custom_allreduce={'ON' if ca is not None and not getattr(ca, 'disabled', True) else 'off'}",
              flush=True)
    rows = []
    for nbytes in BYTES_AXIS:
        x = torch.ones(nbytes // 2, dtype=torch.bfloat16, device="cuda")
        us_vllm = _time(lambda: tp.all_reduce(x))
        us_nccl = _time(lambda: dist.all_reduce(x))
        rows.append((nbytes, us_vllm, us_nccl))
        if rank == 0:
            print(f"  {nbytes:>9d} B: vllm path {us_vllm:8.2f} us   plain nccl {us_nccl:8.2f} us", flush=True)
    if rank == 0:
        root = Path(a.out_dir); out = root / "ncu" / "collectives" / f"{a.gpu_label}_tp{world}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        kept = []
        if out.exists():
            with out.open(newline="") as f:
                kept = [r for r in csv.DictReader(f) if (r.get("op") or "all_reduce") != "all_reduce"]
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["op", "world", "bytes", "latency_us", "dtype_bytes", "path"])
            w.writeheader()
            for r in kept:
                w.writerow({"op": r["op"], "world": r["world"], "bytes": r["bytes"],
                            "latency_us": r["latency_us"], "dtype_bytes": r.get("dtype_bytes", 2),
                            "path": r.get("path", "nccl")})
            for nbytes, us_v, _ in rows:
                w.writerow({"op": "all_reduce", "world": world, "bytes": nbytes,
                            "latency_us": round(us_v, 4), "dtype_bytes": 2, "path": "vllm"})
        print(f"[vllm-ar] wrote {out}: {len(rows)} all_reduce rows (vllm path) + {len(kept)} other-op rows kept", flush=True)
        ref = root / "ncu" / "collectives" / f"{a.gpu_label}_tp{world}_nccl_vs_vllm.csv"
        with ref.open("w", newline="") as f:
            w = csv.writer(f); w.writerow(["bytes", "vllm_us", "nccl_us"])
            for r in rows: w.writerow([r[0], round(r[1], 4), round(r[2], 4)])
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
