#!/usr/bin/env python3
"""PD KV hand-off through NIXL/UCX, laid out exactly as vLLM's NixlConnector does it.

A device-level PD system YAML prices the P->D KV move as ``links.intra_node`` --
bandwidth + a per-transfer floor. Writing the NVLink datasheet there (900 GB/s) is
wrong by an order of magnitude: the consumer does not memcpy one buffer, it issues ONE
NIXL READ over a descriptor list with one entry per (layer, block) -- 94 layers x
ctx/16 blocks of 8 KB each at tp4 (the per-rank K+V of a block, "K/V packed into the
content dim", base_worker._build_fa_local) -- and UCX moves those small pieces over
cuda_ipc. The effective rate is set by the descriptor size and count, not by the link.
Against the E6 GT (Qwen3-235B-FP8, tp4 <-> tp4) the client-side D-first-token time
fitted 31 GB/s of UNSHARDED bytes, i.e. ~8 GB/s per GPU pair.

This probe reproduces that transfer between two GPUs of this box and times it:

  * P (rank 0) allocates a paged KV pool -- one tensor per layer, blocks contiguous,
    ``page_bytes = 2 * block_size * kv_heads_per_rank * head_dim * 2`` per block -- and
    registers every layer with NIXL. D (rank 1) does the same, adds P as a remote agent,
    and preps both descriptor lists once (as the connector does at handshake).
  * Per context length: D picks the request's blocks, builds the desc ids
    (region * num_blocks + block, all layers), make_prepped_xfer("READ") + transfer(),
    then polls check_xfer_state until DONE -- the connector's _read_blocks +
    _pop_done_transfers path. Prep time (host, descriptor build) and xfer time are
    recorded separately.
  * Block layouts: ``contig`` (a fresh server hands out consecutive block ids, and NIXL
    merges adjacent descriptors into one), ``runsN`` (contiguous runs of N blocks with a
    gap between runs -- a partially churned pool; one merged descriptor per run per
    layer) and ``scattered`` (a fully churned pool; nothing merges). The per-run cost is
    what a block-allocator model of the paged pool needs to price fragmentation.

Run (two GPUs, same node -> NVLink):

    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 profiling/probes/kv_transfer_probe.py \
        --gpu-label H200 --out-dir data/kernel_data

Writes eager/kv_transfer/<gpu>_nixl.csv (upsert on the key columns) and prints,
per (kv_heads_per_rank, layout), the floor + per-byte fit to paste into a system YAML's
``links.intra_node`` (per GPU PAIR: the replay divides the unsharded bytes by the
reader tp, see engine/sim/disagg.py::Link.transfer_ms).
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

KEY = ("gpu", "kv_heads_per_rank", "head_dim", "block_size", "layers", "layout", "tokens")
COLS = KEY + ("n_descs", "bytes", "prep_ms", "xfer_ms", "total_ms", "eff_gbps")


def _fit(rows):
    """Least-squares total_ms = floor + bytes / bw over one (heads, layout) series."""
    x = np.array([r["bytes"] for r in rows], dtype=float)
    y = np.array([r["total_ms"] for r in rows], dtype=float)
    A = np.stack([np.ones_like(x), x], axis=1)
    (b0, b1), *_ = np.linalg.lstsq(A, y, rcond=None)
    bw = 1.0 / (b1 / 1e3) if b1 > 0 else float("inf")     # bytes/s
    return b0, bw


def _upsert(path: Path, rows: list[dict]) -> None:
    old = {}
    if path.exists():
        with path.open() as fh:
            for r in csv.DictReader(fh):
                old[tuple(r[k] for k in KEY)] = r
    for r in rows:
        old[tuple(str(r[k]) for k in KEY)] = {k: str(r[k]) for k in COLS}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COLS))
        w.writeheader()
        for k in sorted(old, key=lambda t: (t[0], int(t[1]), t[5], int(t[6]))):
            w.writerow(old[k])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu-label", required=True)
    ap.add_argument("--out-dir", required=True, help="kernel_data root")
    ap.add_argument("--layers", type=int, default=94)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--kv-heads-per-rank", type=int, nargs="+", default=[1, 2],
                    help="4 kv heads / tp: tp4 -> 1, tp2 -> 2")
    ap.add_argument("--tokens", type=int, nargs="+",
                    default=[16, 1024, 4096, 8192, 16384, 32768, 65536, 131072])
    ap.add_argument("--scattered-max-tokens", type=int, default=16384,
                    help="cap the fully-scattered sweep (it runs at ~20 us/descriptor)")
    ap.add_argument("--iters", type=int, default=11)
    ap.add_argument("--layouts", nargs="+", default=["contig", "runs64", "runs8", "scattered"])
    a = ap.parse_args()

    rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    assert world == 2, "run with --nproc_per_node=2 (rank 0 = P, rank 1 = D)"
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("gloo", rank=rank, world_size=world)

    from nixl._api import nixl_agent, nixl_agent_config
    os.environ.setdefault("UCX_RCACHE_MAX_UNRELEASED", "1024")   # as vllm/distributed/nixl_utils.py
    agent = nixl_agent(f"{'P' if rank == 0 else 'D'}{rank}",
                       nixl_agent_config(backends=["UCX"]))

    # Pool = 2x the largest request so `scattered` never degenerates into `contig`.
    max_blocks = 2 * (-(-max(a.tokens) // a.block_size))
    out_rows: list[dict] = []
    for heads in a.kv_heads_per_rank:
        # vLLM's per-layer KV tensor for a FlashAttention backend (HND layout): a block's
        # K and V sit next to each other, so one descriptor per (layer, block).
        page_bytes = 2 * a.block_size * heads * a.head_dim * 2
        pool = [torch.empty(max_blocks * page_bytes, dtype=torch.uint8, device=dev)
                for _ in range(a.layers)]
        if rank == 0:
            for t in pool:
                t.fill_(1)
        torch.cuda.synchronize()
        dev_id = local
        caches = [(t.data_ptr(), t.numel(), dev_id, "") for t in pool]
        reg = agent.register_memory(agent.get_reg_descs(caches, "VRAM"))

        # handshake: P's agent metadata + base addresses -> D  (connector: NixlAgentMetadata)
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
                    parts.append(np.stack([addrs, np.full_like(addrs, page_bytes),
                                           np.full_like(addrs, d)], axis=1))
                return np.concatenate(parts)

            local_side = agent.prep_xfer_dlist("NIXL_INIT_AGENT",
                                               agent.get_xfer_descs(_descs([t.data_ptr() for t in pool], dev_id), "VRAM"))
            remote_side = agent.prep_xfer_dlist(remote_name,
                                                agent.get_xfer_descs(_descs(remote_bases, remote_dev), "VRAM"))
            rng = np.random.default_rng(0)
            perm = rng.permutation(max_blocks)
            region_ids = np.arange(a.layers, dtype=np.int64)[:, None]

            def _block_ids(layout: str, nb: int, side: int) -> np.ndarray:
                """Block ids of one request on one side (0 local, 1 remote)."""
                if layout == "contig":
                    return np.arange(nb) + side * nb
                if layout == "scattered":
                    return np.sort(perm[side * nb:(side + 1) * nb])
                run = int(layout[4:])
                n_runs = -(-nb // run)
                # runs placed every 2*run blocks, so consecutive runs never touch
                starts = (np.arange(n_runs) * 2 * run + side * run) % (max_blocks - run)
                ids = (starts[:, None] + np.arange(run)[None, :]).flatten()[:nb]
                return np.sort(ids)

            for layout in a.layouts:
                for tokens in a.tokens:
                    if layout == "scattered" and tokens > a.scattered_max_tokens:
                        continue
                    nb = -(-tokens // a.block_size)
                    # region*num_blocks + block, all layers -- base_worker._compute_desc_ids
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
                        if it > 0:                      # first is warm-up (UCX rkey cache)
                            prep.append((t1 - t0) * 1e3); xfer.append((t2 - t1) * 1e3)
                    nbytes = int(len(ids)) * page_bytes
                    row = {"gpu": a.gpu_label, "kv_heads_per_rank": heads, "head_dim": a.head_dim,
                           "block_size": a.block_size, "layers": a.layers, "layout": layout,
                           "tokens": tokens, "n_descs": int(len(ids)), "bytes": nbytes,
                           "prep_ms": round(float(np.median(prep)), 3),
                           "xfer_ms": round(float(np.median(xfer)), 3)}
                    row["total_ms"] = round(row["prep_ms"] + row["xfer_ms"], 3)
                    row["eff_gbps"] = round(nbytes / (row["total_ms"] / 1e3) / 1e9, 2)
                    out_rows.append(row)
                    print(f"heads={heads} {layout:9s} tokens={tokens:>6} descs={len(ids):>8} "
                          f"{nbytes / 1e9:7.3f} GB  prep {row['prep_ms']:8.2f} ms  xfer {row['xfer_ms']:8.2f} ms"
                          f"  -> {row['eff_gbps']:6.1f} GB/s per GPU pair", flush=True)
            agent.release_dlist_handle(local_side); agent.release_dlist_handle(remote_side)
            agent.remove_remote_agent(remote_name)
        dist.barrier()
        agent.deregister_memory(reg)
        del pool
        torch.cuda.empty_cache()
        dist.barrier()

    if rank == 1:
        path = Path(a.out_dir) / "eager" / "kv_transfer" / f"{a.gpu_label}_nixl.csv"
        _upsert(path, out_rows)
        print(f"\nwrote {path}")
        import sys as _sys  # noqa: PLC0415
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _manifest import write_manifest  # noqa: PLC0415
        write_manifest(path, mode="host_wallclock", tool="kv_transfer_probe.py",
                       reduce="per-cell host wall-clock incl. NIXL prep + completion polling",
                       gpu_label=a.gpu_label, upsert=True, notes="PD KV hand-off as vLLM's NixlConnector issues it")
        print("\nfit total_ms = floor + bytes/bw   (per GPU pair; system YAML links.intra_node)")
        for heads in a.kv_heads_per_rank:
            for layout in a.layouts:
                rows = [r for r in out_rows if r["kv_heads_per_rank"] == heads and r["layout"] == layout
                        and r["tokens"] >= 1024]
                if len(rows) >= 2:
                    floor, bw = _fit(rows)
                    print(f"  kv_heads_per_rank={heads} ({2 * a.block_size * heads * a.head_dim * 2 // 1024} KB/desc) "
                          f"{layout:9s}: floor {floor:7.2f} ms   bw {bw / 1e9:6.1f} GB/s")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
