# profiling/probes/custom_allreduce_probe.py
"""vLLM custom all-reduce latency at decode payloads — the real kernel behind
tp_comm latency_us_per_op (the NCCL probe only bounded it from above). Times
vLLM's tensor_model_parallel_all_reduce and raw NCCL in the same run; if the
two match, custom AR did not engage and the result is just the NCCL bound.

  CUDA_VISIBLE_DEVICES=4,5 python custom_allreduce_probe.py --out custom_allreduce_H100_tp2.json
"""

import argparse
import json
import os
import statistics as st

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 320, 512, 1024, 8192]  # 8192 = prefill chunk (>8MB, NCCL path)
HIDDEN = 4096
REPS, WARMUP = 200, 20


def timed(fn, t) -> float:
    for _ in range(WARMUP):
        fn(t)
    torch.cuda.synchronize()
    dist.barrier()
    times = []
    for _ in range(REPS):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn(t)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)  # us
    return st.median(times)


def worker(rank: int, world: int, out_path: str):
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    from vllm.distributed import (init_distributed_environment,
                                  initialize_model_parallel,
                                  tensor_model_parallel_all_reduce)
    init_distributed_environment(world_size=world, rank=rank, local_rank=rank,
                                 distributed_init_method="tcp://127.0.0.1:29672")
    initialize_model_parallel(tensor_model_parallel_size=world)
    torch.cuda.set_device(rank)

    custom_ar = None  # best-effort introspection; version-dependent layout
    try:
        from vllm.distributed.parallel_state import get_tp_group
        g = get_tp_group()
        ca = getattr(g, "ca_comm", None)
        if ca is None:
            ca = getattr(getattr(g, "device_communicator", None), "ca_comm", None)
        custom_ar = ca is not None and not getattr(ca, "disabled", False)
    except Exception:
        pass

    vllm_us, nccl_us = {}, {}
    for b in BATCHES:
        t = torch.randn(b * HIDDEN, device="cuda", dtype=torch.bfloat16)
        vllm_us[b] = timed(tensor_model_parallel_all_reduce, t)
        nccl_us[b] = timed(dist.all_reduce, t)
        if rank == 0:
            payload = b * HIDDEN * 2
            print(f"batch={b:>5} payload={payload:>9}B: vllm {vllm_us[b]:7.1f} us  "
                  f"nccl {nccl_us[b]:7.1f} us  ratio {vllm_us[b]/nccl_us[b]:.2f}", flush=True)
        dist.barrier()

    if rank == 0:
        floor = min(vllm_us[b] for b in BATCHES[:4])
        json.dump({"schema": "custom_allreduce_probe.v1", "world": world,
                   "hidden": HIDDEN, "dtype": "bfloat16", "reps": REPS,
                   "custom_ar_handle_active": custom_ar,
                   "vllm_per_batch_us": vllm_us, "nccl_per_batch_us": nccl_us,
                   "vllm_latency_floor_us": round(floor, 2)},
                  open(out_path, "w"), indent=1)
        print(f"vllm latency_floor_us={floor:.1f}  custom_ar_handle={custom_ar} -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="custom_allreduce_H100_tp2.json")
    a = ap.parse_args()
    mp.spawn(worker, args=(2, a.out), nprocs=2, join=True)


if __name__ == "__main__":
    main()
