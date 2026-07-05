#!/usr/bin/env python3
# profiling/probes/allreduce_probe.py
"""Decode-regime all-reduce cost on an NVLink pair: CUDA-event-timed NCCL
all_reduce over decode-step payloads (batch x hidden x 2B, bf16), batch-swept.
Yields the latency floor (small-payload asymptote) and effective bandwidth for
the tp_comm decode analytic.

  CUDA_VISIBLE_DEVICES=0,4 python allreduce_probe.py --out allreduce_H100_tp2.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 320, 512, 1024]
HIDDEN = 4096
REPS, WARMUP = 200, 20


def worker(rank: int, world: int, out_path: str):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29671"
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    results = {}
    for b in BATCHES:
        t = torch.randn(b * HIDDEN, device="cuda", dtype=torch.bfloat16)
        for _ in range(WARMUP):
            dist.all_reduce(t)
        torch.cuda.synchronize()
        dist.barrier()
        times = []
        for _ in range(REPS):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            dist.all_reduce(t)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e) * 1000.0)  # us
        med = st.median(times)
        results[b] = med
        if rank == 0:
            payload = b * HIDDEN * 2
            print(f"batch={b:>5} payload={payload:>9}B: {med:7.1f} us "
                  f"({payload/med/1e3:.1f} GB/s eff)", flush=True)
        dist.barrier()
    if rank == 0:
        payloads = {b * HIDDEN * 2: us for b, us in results.items()}
        floor = min(results[b] for b in BATCHES[:4])  # small-payload asymptote
        big = BATCHES[-1] * HIDDEN * 2
        eff_bw = big / results[BATCHES[-1]] * 1e6  # bytes/s
        json.dump({"schema": "allreduce_probe.v1", "world": world, "hidden": HIDDEN,
                   "dtype": "bfloat16", "reps": REPS,
                   "per_batch_us": results, "per_payload_us": payloads,
                   "latency_floor_us": round(floor, 2),
                   "eff_bw_bytes_per_s": round(eff_bw, 0)},
                  open(out_path, "w"), indent=1)
        print(f"latency_floor_us={floor:.1f}  eff_bw={eff_bw/1e9:.0f} GB/s -> {out_path}")
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="allreduce_H100_tp2.json")
    a = ap.parse_args()
    mp.spawn(worker, args=(2, a.out), nprocs=2, join=True)


if __name__ == "__main__":
    main()
