#!/usr/bin/env python3
"""Times a prefill to decode KV hand-off between two GPUs through NIXL/UCX, laid out as vLLM's NixlConnector does; run under torchrun with two ranks.
Rank 0 emits the rows of kv_transfer/<gpu>_nixl.csv and prints the floor + per-byte fit per (kv_heads_per_rank, layout).
"""

from __future__ import annotations

import argparse
import os
import signal
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist

from quettabench.probes.base import ProbeRow, add_output_args, emit, run_recorded

KEY = ("gpu", "kv_heads_per_rank", "head_dim", "block_size", "layers", "layout", "tokens")
COLS = KEY + ("n_descs", "bytes", "prep_ms", "xfer_ms", "total_ms", "eff_gbps")


def _fit(rows):
    """Least-squares total_ms = floor + bytes / bw over one (heads, layout) series."""
    x = np.array([r["bytes"] for r in rows], dtype=float)
    y = np.array([r["total_ms"] for r in rows], dtype=float)
    A = np.stack([np.ones_like(x), x], axis=1)
    (b0, b1), *_ = np.linalg.lstsq(A, y, rcond=None)
    bw = 1.0 / (b1 / 1e3) if b1 > 0 else float("inf")  # bytes/s
    return b0, bw


def _emit(table: str, rows: list[dict]) -> None:
    """Emit this run's rows as table, sorted by heads per rank, layout and tokens."""
    for r in sorted(rows, key=lambda r: (r["kv_heads_per_rank"], r["layout"], r["tokens"])):
        emit(
            ProbeRow(
                "kv_transfer",
                table,
                {c: r.get(c) for c in KEY},
                {c: r.get(c) for c in COLS if c not in KEY},
                "host_wallclock",
            )
        )


def _on_sigterm(signum, frame):
    """A SIGTERM (torchrun stopping a rank) fails the run in run_recorded."""
    raise SystemExit(143)


def main(argv=None) -> bool:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_output_args(ap)
    ap.add_argument("--layers", type=int, default=94, help="one KV tensor and one descriptor per layer and block")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument(
        "--kv-heads-per-rank", type=int, nargs="+", default=[1, 2], help="4 kv heads / tp: tp4 -> 1, tp2 -> 2"
    )
    ap.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        help="context lengths to transfer",
        default=[16, 1024, 4096, 8192, 16384, 32768, 65536, 131072],
    )
    ap.add_argument(
        "--scattered-max-tokens",
        type=int,
        default=16384,
        help="cap the fully-scattered sweep (it runs at ~20 us/descriptor)",
    )
    ap.add_argument("--iters", type=int, default=11, help="timed transfers per cell, after one warm-up")
    ap.add_argument(
        "--layouts",
        nargs="+",
        default=["contig", "runs64", "runs8", "scattered"],
        help="block placement: contig (consecutive ids, NIXL merges descriptors), runsN (runs of N blocks "
        "with gaps), scattered (nothing merges)",
    )
    a = ap.parse_args(argv)

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    assert world == 2, "run with --nproc_per_node=2 (rank 0 = P, rank 1 = D)"
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    signal.signal(signal.SIGTERM, _on_sigterm)
    dist.init_process_group("gloo", rank=rank, world_size=world, timeout=timedelta(minutes=5))

    # Only hosts with nixl can import it.
    from nixl._api import nixl_agent, nixl_agent_config

    os.environ.setdefault("UCX_RCACHE_MAX_UNRELEASED", "1024")
    agent = nixl_agent(f"{'P' if rank == 0 else 'D'}{rank}", nixl_agent_config(backends=["UCX"]))

    # Twice the largest request, so `scattered` never degenerates into `contig`.
    max_blocks = 2 * (-(-max(a.tokens) // a.block_size))
    out_rows: list[dict] = []
    for heads in a.kv_heads_per_rank:
        # FlashAttention HND layout: a block's K and V are adjacent, one descriptor per (layer, block).
        page_bytes = 2 * a.block_size * heads * a.head_dim * 2
        pool = [torch.empty(max_blocks * page_bytes, dtype=torch.uint8, device=dev) for _ in range(a.layers)]
        if rank == 0:
            for t in pool:
                t.fill_(1)
        torch.cuda.synchronize()
        dev_id = local
        caches = [(t.data_ptr(), t.numel(), dev_id, "") for t in pool]
        reg = agent.register_memory(agent.get_reg_descs(caches, "VRAM"))

        meta = [None]
        if rank == 0:
            meta = [(agent.get_agent_metadata(), [t.data_ptr() for t in pool], dev_id)]
        dist.broadcast_object_list(meta, src=0)

        if rank == 1:
            remote_meta, remote_bases, remote_dev = meta[0]
            remote_name = agent.add_remote_agent(remote_meta)
            blocks = np.arange(max_blocks, dtype=np.uint64)

            def _descs(bases, d):
                parts = []
                for base in bases:
                    addrs = np.uint64(base) + blocks * np.uint64(page_bytes)
                    parts.append(np.stack([addrs, np.full_like(addrs, page_bytes), np.full_like(addrs, d)], axis=1))
                return np.concatenate(parts)

            local_side = agent.prep_xfer_dlist(
                "NIXL_INIT_AGENT", agent.get_xfer_descs(_descs([t.data_ptr() for t in pool], dev_id), "VRAM")
            )
            remote_side = agent.prep_xfer_dlist(
                remote_name, agent.get_xfer_descs(_descs(remote_bases, remote_dev), "VRAM")
            )
            rng = np.random.default_rng(0)
            perm = rng.permutation(max_blocks)
            region_ids = np.arange(a.layers, dtype=np.int64)[:, None]

            def _block_ids(layout: str, nb: int, side: int) -> np.ndarray:
                """Block ids of one request on one side (0 local, 1 remote)."""
                if layout == "contig":
                    return np.arange(nb) + side * nb
                if layout == "scattered":
                    return np.sort(perm[side * nb : (side + 1) * nb])
                run = int(layout[4:])
                n_runs = -(-nb // run)
                # A one-run gap keeps runs from touching.
                starts = (np.arange(n_runs) * 2 * run + side * run) % (max_blocks - run)
                ids = (starts[:, None] + np.arange(run)[None, :]).flatten()[:nb]
                return np.sort(ids)

            for layout in a.layouts:
                for tokens in a.tokens:
                    if layout == "scattered" and tokens > a.scattered_max_tokens:
                        continue
                    nb = -(-tokens // a.block_size)
                    ids = (region_ids * max_blocks + _block_ids(layout, nb, 1)[None, :]).flatten()
                    lids = (region_ids * max_blocks + _block_ids(layout, nb, 0)[None, :]).flatten()
                    prep, xfer = [], []
                    for it in range(a.iters + 1):
                        t0 = time.perf_counter()
                        h = agent.make_prepped_xfer("READ", local_side, lids, remote_side, ids)
                        t1 = time.perf_counter()
                        agent.transfer(h)
                        while agent.check_xfer_state(h) == "PROC":
                            pass
                        t2 = time.perf_counter()
                        agent.release_xfer_handle(h)
                        if it > 0:
                            prep.append((t1 - t0) * 1e3)
                            xfer.append((t2 - t1) * 1e3)
                    nbytes = int(len(ids)) * page_bytes
                    row = {
                        "gpu": a.gpu_label,
                        "kv_heads_per_rank": heads,
                        "head_dim": a.head_dim,
                        "block_size": a.block_size,
                        "layers": a.layers,
                        "layout": layout,
                        "tokens": tokens,
                        "n_descs": int(len(ids)),
                        "bytes": nbytes,
                        "prep_ms": round(float(np.median(prep)), 3),
                        "xfer_ms": round(float(np.median(xfer)), 3),
                    }
                    row["total_ms"] = round(row["prep_ms"] + row["xfer_ms"], 3)
                    row["eff_gbps"] = round(nbytes / (row["total_ms"] / 1e3) / 1e9, 2)
                    out_rows.append(row)
                    print(
                        f"heads={heads} {layout:9s} tokens={tokens:>6} descs={len(ids):>8} "
                        f"{nbytes / 1e9:7.3f} GB  prep {row['prep_ms']:8.2f} ms  xfer {row['xfer_ms']:8.2f} ms"
                        f"  -> {row['eff_gbps']:6.1f} GB/s per GPU pair",
                        flush=True,
                    )
            agent.release_dlist_handle(local_side)
            agent.release_dlist_handle(remote_side)
            agent.remove_remote_agent(remote_name)
        dist.barrier()
        agent.deregister_memory(reg)
        del pool
        torch.cuda.empty_cache()
        dist.barrier()

    sent = [out_rows]
    dist.broadcast_object_list(sent, src=1)
    out_rows = sent[0]
    if rank == 0:
        _emit(f"kv_transfer/{a.gpu_label}_nixl.csv", out_rows)
        print(f"\n{len(out_rows)} rows of kv_transfer/{a.gpu_label}_nixl.csv")
        print("\nfit total_ms = floor + bytes/bw   (per GPU pair; system YAML links.intra_node)")
        for heads in a.kv_heads_per_rank:
            for layout in a.layouts:
                rows = [
                    r
                    for r in out_rows
                    if r["kv_heads_per_rank"] == heads and r["layout"] == layout and r["tokens"] >= 1024
                ]
                if len(rows) >= 2:
                    floor, bw = _fit(rows)
                    print(
                        f"  kv_heads_per_rank={heads} ({2 * a.block_size * heads * a.head_dim * 2 // 1024} KB/desc) "
                        f"{layout:9s}: floor {floor:7.2f} ms   bw {bw / 1e9:6.1f} GB/s"
                    )
    dist.destroy_process_group()
    return rank == 0


if __name__ == "__main__":
    run_recorded(main, "kv_transfer")
