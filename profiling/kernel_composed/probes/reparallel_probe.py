#!/usr/bin/env python3
"""NVLink cost of switching tensor-parallel degree in place (TP reshard).

A device-level engine that changes TP on the fly (tp2 <-> tp4) does NOT reload the
weights from host/disk -- it re-shards what is already resident on the GPUs over
NVLink. This probe measures the wall-time of that reshard so a cost model can price a
``(tp_from -> tp_to)`` switch, the same way ``kv_transfer_probe`` measures the PD KV
hand-off.

WHAT MOVES (fixed GPU pool, dp compensates so ``tp * dp == world`` is constant,
optimal GPU->new-rank relabelling that minimises movement):

  * tp UP (q = k*p, dp DOWN):   each GPU's new 1/q-slice is a SUBSET of the 1/p-slice
    it already holds, so a smart relabel places every GPU for free. Weight movement is
    ZERO -- the switch just DISCARDS the unneeded weights. We still time the local
    re-pack (a narrowing device copy) as the floor.
  * tp DOWN (p = k*q, dp UP):   each GPU's new 1/q-slice is the UNION of k old
    1/p-slices; it keeps one and fetches the other k-1. Partition the old ranks into
    groups of k and ALL-GATHER within each group -- every member ends with the full
    q-slice. Bytes received per GPU = (k-1) * (W/p) = W * (p-q)/(p*q).

Unlike the paged-KV hand-off, a weight shard is a handful of large contiguous tensors,
so the reshard runs near the NVLink asymptote (700-900 GB/s intra-node on this
NVSwitch box) rather than the ~200 GB/s NIXL floor a fragmented KV pool sees. Pinning
that number by measurement is the whole point: the analytic volume model above needs a
per-byte rate and a launch floor, and those are what this probe fits.

Run (all 8 GPUs of the node, over NVLink):

    python3 -m torch.distributed.run --nproc_per_node=8 \
        profiling/probes/reparallel_probe.py --gpu-label H200 --out-dir data/kernel_data

Writes eager/reparallel/<gpu>_reshard.csv (upsert on the key columns) and prints,
per (tp_from, tp_to), the floor + per-byte fit to paste into a device YAML's
``reshard:`` block (see engine/reparallel.py::ReshardLink).
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

KEY = ("gpu", "world", "tp_from", "tp_to", "total_weight_gb")
COLS = KEY + ("k", "bytes_moved_per_gpu", "comm_ms", "eff_gbps", "pattern",
              "bytes_inter_per_gpu")   # cross-node share of the move (0 intra-node)


def _fit(rows: list[dict]) -> tuple[float, float]:
    """Least-squares comm_ms = floor + bytes/bw over one (tp_from, tp_to) series."""
    x = np.array([r["bytes_moved_per_gpu"] for r in rows], dtype=float)
    y = np.array([r["comm_ms"] for r in rows], dtype=float)
    keep = x > 0
    if keep.sum() < 2:
        return float(np.median(y)) if len(y) else 0.0, float("inf")
    x, y = x[keep], y[keep]
    A = np.stack([np.ones_like(x), x], axis=1)
    (b0, b1), *_ = np.linalg.lstsq(A, y, rcond=None)
    bw = 1.0 / (b1 / 1e3) if b1 > 0 else float("inf")     # bytes/s
    return b0, bw


def _upsert(path: Path, rows: list[dict]) -> None:
    old: dict[tuple, dict] = {}
    if path.exists():
        with path.open() as fh:
            for r in csv.DictReader(fh):
                old[tuple(r[k] for k in KEY)] = r
    for r in rows:
        old[tuple(str(r[k]) for k in KEY)] = {k: str(r.get(k, 0)) for k in COLS}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COLS), restval="0")
        w.writeheader()
        for k in sorted(old, key=lambda t: (t[0], int(t[1]), int(t[2]), int(t[3]),
                                            float(t[4]))):
            w.writerow(old[k])


def _time_ms(fn, iters: int) -> float:
    """Median ms of ``fn`` over ``iters`` timed runs (one warm-up), cuda-event timed."""
    dev = torch.cuda.current_device()
    samples = []
    for it in range(iters + 1):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        dist.barrier()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        if it > 0:
            samples.append(start.elapsed_time(end))
    return float(np.median(samples))


def _groups_of(k: int, world: int) -> list[list[int]]:
    """Partition ranks 0..world-1 into contiguous groups of size ``k``."""
    return [list(range(g, g + k)) for g in range(0, world, k)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu-label", required=True)
    ap.add_argument("--out-dir", required=True, help="kernel_data root")
    ap.add_argument("--switches", nargs="+",
                    default=["4:2", "2:1", "8:4", "4:1", "8:2", "2:4", "4:8"],
                    help="tp_from:tp_to pairs (down = reshard, up = discard floor)")
    ap.add_argument("--mode", default="tp",
                    choices=["tp", "dp", "ep", "pp", "kv", "plan", "xnode"],
                    help="tp: fixed-pool TP reshard (all-gather); dp: replica "
                         "replication (broadcast a shard to fresh GPUs); ep: expert "
                         "restage (all-to-all); pp: pipeline-stage reshard (whole-layer "
                         "all-gather); kv: in-flight paged-KV reshard (head-shard "
                         "all-gather); plan: whole PD-plan reshard (faithful per-GPU "
                         "byte-range exchange for a P/D layout change; node-aware when "
                         "launched across nodes); xnode: cross-node link physics -- "
                         "concurrent p2p pair sweep (rail sharing / replica hydration) "
                         "and node-spanning all-gathers. One dim per mode.")
    ap.add_argument("--plans", nargs="+",
                    default=["2:2,4:1>4:1,2:2", "4:1,4:1>2:3,2:1",
                             "2:2,4:1>2:2,1:4", "4:1,4:1>4:1,4:1"],
                    help="mode=plan: OLD>NEW where each side is 'tp_p:dp_p,tp_d:dp_d' "
                         "(prefill pool, decode pool); worlds must sum to --nproc")
    ap.add_argument("--weights-gb", type=float, nargs="+",
                    default=[1, 2, 4, 8, 16, 32],
                    help="total logical weight W swept; per-GPU move derived from it")
    ap.add_argument("--iters", type=int, default=11)
    ap.add_argument("--kv-tokens", type=int, nargs="+", dest="kv_tokens",
                    default=[1024, 4096, 16384, 65536, 131072, 262144],
                    help="mode=kv: per-request token counts to reshard")
    ap.add_argument("--layers", type=int, default=94, help="mode=kv: model layers")
    ap.add_argument("--head-dim", type=int, default=128, help="mode=kv: KV head dim")
    ap.add_argument("--out-name", default=None,
                    help="CSV stem (default <gpu>_reshard); use e.g. <gpu>_reshard_val "
                         "for a held-out validation sweep that must not touch calibration")
    a = ap.parse_args()

    rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", rank=rank, world_size=world)

    # Cache the process subgroups the all-gathers use (creating a group is collective).
    group_cache: dict[int, list] = {}

    def _get_groups(k: int) -> list:
        if k not in group_cache:
            group_cache[k] = [dist.new_group(g) for g in _groups_of(k, world)]
        return group_cache[k]

    out_rows: list[dict] = []
    if a.mode == "dp":
        out_rows = _measure_dp(a, rank, world)
    elif a.mode == "ep":
        out_rows = _measure_ep(a, rank, world)
    elif a.mode == "pp":
        out_rows = _measure_pp(a, rank, world, _get_groups)
    elif a.mode == "kv":
        out_rows = _measure_kv(a, rank, world, _get_groups)
    elif a.mode == "plan":
        out_rows = _measure_plan(a, rank, world)
    elif a.mode == "xnode":
        out_rows = _measure_xnode(a, rank, world)
    else:
        out_rows = _measure_tp(a, rank, world, _get_groups)

    if rank == 0:
        default_stem = {"tp": "reshard", "dp": "reshard_dp", "ep": "reshard_ep",
                        "pp": "reshard_pp", "kv": "reshard_kv",
                        "plan": "reshard_plan", "xnode": "reshard_xnode"}[a.mode]
        stem = a.out_name or f"{a.gpu_label}_{default_stem}"
        path = Path(a.out_dir) / "eager" / "reparallel" / f"{stem}.csv"
        _upsert(path, out_rows)
        print(f"\nwrote {path}")
        import sys as _sys  # noqa: PLC0415
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _manifest import write_manifest  # noqa: PLC0415
        write_manifest(path, mode="host_wallclock", tool=f"reparallel_probe.py --mode {a.mode}",
                       reduce="per-cell wall-clock (see _measure_*)", gpu_label=a.gpu_label, upsert=True,
                       notes=f"in-place parallelism switch cost, world={world}")
        print(f"\nfit comm_ms = floor + bytes/bw   (mode={a.mode}, per GPU)")
        seen = []
        pairs = sorted({(r["tp_from"], r["tp_to"]) for r in out_rows})
        for p, q in pairs:
            rows = [r for r in out_rows if r["tp_from"] == p and r["tp_to"] == q
                    and r["bytes_moved_per_gpu"] > 0]
            if len(rows) >= 2:
                floor, bw = _fit(rows)
                seen.append(bw)
                print(f"  {p}->{q}: floor {floor:7.3f} ms   bw {bw / 1e9:6.1f} GB/s")
        if seen:
            print(f"\n  bw_bytes_per_s (median): {np.median(seen):.3e}")
    dist.destroy_process_group()


def _measure_tp(a, rank: int, world: int, _get_groups) -> list[dict]:
    """Fixed-pool TP reshard: all-gather (down) / discard (up)."""
    out_rows: list[dict] = []
    for sw in a.switches:
        p, q = (int(x) for x in sw.split(":"))
        if world % p or world % q:
            if rank == 0:
                print(f"skip {sw}: world {world} not divisible by both tp", flush=True)
            continue
        my_group = None
        if q < p:                              # tp DOWN: all-gather within groups of k
            k = p // q
            groups = _get_groups(k)
            my_group = groups[rank // k]
        for wgb in a.weights_gb:
            W = wgb * 1e9
            in_bytes = int(W / p)              # bytes this GPU holds under tp_from
            in_bytes -= in_bytes % 4           # keep float32 element aligned
            if q < p:
                k = p // q
                moved = (k - 1) * in_bytes     # bytes received on the reshard
                src = torch.empty(in_bytes // 4, dtype=torch.float32, device="cuda")
                dst = [torch.empty_like(src) for _ in range(k)]

                def _do() -> None:
                    dist.all_gather(dst, src, group=my_group)

                pattern = "allgather"
            else:                              # tp UP: discard -> local re-pack floor
                k = q // p
                moved = 0
                keep = in_bytes // k
                src = torch.empty(in_bytes // 4, dtype=torch.float32, device="cuda")
                dst_buf = torch.empty(keep // 4, dtype=torch.float32, device="cuda")

                def _do() -> None:
                    dst_buf.copy_(src[: keep // 4])   # narrowing device copy (repack)

                pattern = "discard"
            comm_ms = _time_ms(_do, a.iters)
            eff = moved / (comm_ms / 1e3) / 1e9 if (moved and comm_ms > 0) else 0.0
            row = {"gpu": a.gpu_label, "world": world, "tp_from": p, "tp_to": q,
                   "total_weight_gb": wgb, "k": k, "bytes_moved_per_gpu": moved,
                   "comm_ms": round(comm_ms, 4), "eff_gbps": round(eff, 2),
                   "pattern": pattern}
            out_rows.append(row)
            if rank == 0:
                print(f"tp{p}->tp{q} {pattern:9s} W={wgb:5.0f}GB  "
                      f"move {moved / 1e9:7.3f} GB/gpu  {comm_ms:8.3f} ms  "
                      f"-> {eff:7.1f} GB/s", flush=True)
            del src
            if q < p:
                del dst
            torch.cuda.empty_cache()
    return out_rows


def _measure_dp(a, rank: int, world: int) -> list[dict]:
    """DP replication: a fresh replica GPU pulls its whole W/tp shard from the source
    replica's peer over NVLink (a P2P send). Measured pairwise (even->odd) in parallel,
    which is how independent replica pairs replicate at once. ``tp_from``/``tp_to`` carry
    the per-GPU shard's tp so the same sweep covers every shard size."""
    out_rows: list[dict] = []
    if world < 2:
        return out_rows
    peer = rank ^ 1                                # 0<->1, 2<->3, ...
    for tp in (1, 2, 4, 8):
        for wgb in a.weights_gb:
            nbytes = int(wgb * 1e9 / tp)
            nbytes -= nbytes % 4
            buf = torch.empty(nbytes // 4, dtype=torch.float32, device="cuda")

            def _do() -> None:
                if rank % 2 == 0:                 # even = source, odd = fresh replica
                    dist.send(buf, dst=peer)
                else:
                    dist.recv(buf, src=peer)

            comm_ms = _time_ms(_do, a.iters)
            moved = nbytes                        # the fresh GPU receives its full shard
            eff = moved / (comm_ms / 1e3) / 1e9 if comm_ms > 0 else 0.0
            row = {"gpu": a.gpu_label, "world": world, "tp_from": tp, "tp_to": tp,
                   "total_weight_gb": wgb, "k": 1, "bytes_moved_per_gpu": moved,
                   "comm_ms": round(comm_ms, 4), "eff_gbps": round(eff, 2),
                   "pattern": "p2p_bcast"}
            out_rows.append(row)
            if rank == 0:
                print(f"dp replicate tp{tp} W={wgb:5.0f}GB  move {moved / 1e9:7.3f} GB/gpu"
                      f"  {comm_ms:8.3f} ms  -> {eff:7.1f} GB/s", flush=True)
            del buf
            torch.cuda.empty_cache()
    return out_rows


def _measure_ep(a, rank: int, world: int) -> list[dict]:
    """EP expert restage: experts redistribute across the ep ranks via all-to-all.
    Measured with all_to_all_single over the whole world at GB scale (the collectives
    grid tops out at a few MB). Run with --nproc_per_node=<ep> to measure a given ep;
    ``ep = world`` here. moved per GPU = (ep-1)/ep of the per-GPU expert bytes."""
    out_rows: list[dict] = []
    ep = world
    if ep < 2:
        return out_rows
    for wgb in a.weights_gb:
        chunk = int(wgb * 1e9 / ep) // ep
        chunk -= chunk % 4
        per_gpu = chunk * ep
        src = torch.empty(per_gpu // 4, dtype=torch.float32, device="cuda")
        dst = torch.empty_like(src)

        def _do() -> None:
            dist.all_to_all_single(dst, src)

        comm_ms = _time_ms(_do, a.iters)
        moved = (ep - 1) * chunk                  # bytes received from other ep ranks
        eff = moved / (comm_ms / 1e3) / 1e9 if comm_ms > 0 else 0.0
        row = {"gpu": a.gpu_label, "world": world, "tp_from": ep, "tp_to": ep,
               "total_weight_gb": wgb, "k": ep, "bytes_moved_per_gpu": moved,
               "comm_ms": round(comm_ms, 4), "eff_gbps": round(eff, 2),
               "pattern": "all_to_all"}
        out_rows.append(row)
        if rank == 0:
            print(f"ep restage ep{ep} W={wgb:5.0f}GB  move {moved / 1e9:7.3f} GB/gpu"
                  f"  {comm_ms:8.3f} ms  -> {eff:7.1f} GB/s", flush=True)
        del src, dst
        torch.cuda.empty_cache()
    return out_rows


def _measure_pp(a, rank: int, world: int, _get_groups) -> list[dict]:
    """Pipeline-stage reshard: on a fixed pool, pp DOWN gathers whole layers onto the
    surviving stages (all-gather within groups of k = pp_from/pp_to); pp UP discards.
    Physically the SAME NVLink all-gather as a TP reshard, but over whole-layer tensors
    -- this mode confirms that the tp grid prices it. ``switches`` reused as pp pairs."""
    out_rows: list[dict] = []
    for sw in a.switches:
        p, q = (int(x) for x in sw.split(":"))
        if world % p or world % q or q >= p:      # pp DOWN only (up is a free discard)
            continue
        k = p // q
        my_group = _get_groups(k)[rank // k]
        for wgb in a.weights_gb:
            in_bytes = int(wgb * 1e9 / p)          # per-stage weight under pp_from
            in_bytes -= in_bytes % 4
            src = torch.empty(in_bytes // 4, dtype=torch.float32, device="cuda")
            dst = [torch.empty_like(src) for _ in range(k)]

            def _do() -> None:
                dist.all_gather(dst, src, group=my_group)

            comm_ms = _time_ms(_do, a.iters)
            moved = (k - 1) * in_bytes
            eff = moved / (comm_ms / 1e3) / 1e9 if comm_ms > 0 else 0.0
            out_rows.append({"gpu": a.gpu_label, "world": world, "tp_from": p,
                             "tp_to": q, "total_weight_gb": wgb, "k": k,
                             "bytes_moved_per_gpu": moved, "comm_ms": round(comm_ms, 4),
                             "eff_gbps": round(eff, 2), "pattern": "pp_allgather"})
            if rank == 0:
                print(f"pp{p}->pp{q} W={wgb:5.0f}GB  move {moved / 1e9:7.3f} GB/gpu"
                      f"  {comm_ms:8.3f} ms  -> {eff:7.1f} GB/s", flush=True)
            del src, dst
            torch.cuda.empty_cache()
    return out_rows


def _measure_kv(a, rank: int, world: int, _get_groups) -> list[dict]:
    """In-flight paged-KV reshard: a TP-down move re-shards the KV by head. Each
    surviving rank gathers a sibling rank's head-shard for every layer. Measured as an
    NCCL all-gather of a CONTIGUOUS per-rank KV buffer (the engine repacks the request's
    blocks before the collective) -- the fast mechanism, an independent GT for the
    cost model's contiguous KV bandwidth. A churned pool that cannot repack falls back
    to the fragmented kv_transfer path (kv_transfer_probe), which is slower.

    Buffer per rank = 2 (K+V) * tokens * heads_per_rank * head_dim * 2 (bf16) * layers."""
    out_rows: list[dict] = []
    k = 2                                          # single-step reshard (e.g. tp4->tp2)
    my_group = _get_groups(k)[rank // k]
    heads_per_rank = 1                             # tp4 with 4 kv heads
    per_tok = 2 * heads_per_rank * a.head_dim * 2 * a.layers
    for tokens in a.kv_tokens:
        nbytes = tokens * per_tok
        nbytes -= nbytes % 4
        src = torch.empty(nbytes // 4, dtype=torch.float32, device="cuda")
        dst = [torch.empty_like(src) for _ in range(k)]

        def _do() -> None:
            dist.all_gather(dst, src, group=my_group)

        comm_ms = _time_ms(_do, a.iters)
        moved = (k - 1) * nbytes
        eff = moved / (comm_ms / 1e3) / 1e9 if comm_ms > 0 else 0.0
        out_rows.append({"gpu": a.gpu_label, "world": world, "tp_from": 4, "tp_to": 2,
                         "total_weight_gb": round(tokens / 1e3, 3), "k": tokens,
                         "bytes_moved_per_gpu": moved, "comm_ms": round(comm_ms, 4),
                         "eff_gbps": round(eff, 2), "pattern": "kv_allgather"})
        if rank == 0:
            print(f"kv reshard tok={tokens:>7}  move {moved / 1e9:7.3f} GB/gpu"
                  f"  {comm_ms:8.3f} ms  -> {eff:7.1f} GB/s", flush=True)
        del src, dst
        torch.cuda.empty_cache()
    return out_rows


def _measure_xnode(a, rank: int, world: int) -> list[dict]:
    """Cross-node link physics for the reshard model, two patterns:

    * ``xpair{n}`` -- n concurrent cross-node P2P pairs (GPU i of node 0 -> GPU i of
      node 1), n in {1, 2, 4, ..., gpn}. One pair measures the single-stream fabric
      rate; the sweep measures RAIL SHARING -- a whole-node replica hydration runs gpn
      concurrent pulls over (typically fewer) NICs, and the per-pair effective rate is
      what the cost model needs.
    * ``xallgather`` -- gpn concurrent 2-way all-gathers, each spanning the two nodes
      (group {i, gpn+i}): the pp/tp reshard of an instance that spans nodes.

    Launch: torchrun --nnodes=2 --nproc_per_node=<gpn> ... --mode xnode
    (tp_from/tp_to columns carry the concurrent-pair count for the fit.)"""
    gpn = int(os.environ.get("LOCAL_WORLD_SIZE", world))
    n_nodes = world // gpn
    out_rows: list[dict] = []
    if n_nodes < 2:
        if rank == 0:
            print("xnode mode needs --nnodes >= 2; nothing to measure", flush=True)
        return out_rows
    node, local = rank // gpn, rank % gpn

    npairs_axis = [n for n in (1, 2, 4, 8) if n <= gpn]
    for npairs in npairs_axis:
        active = node < 2 and local < npairs
        for wgb in a.weights_gb:
            nbytes = int(wgb * 1e9)
            nbytes -= nbytes % 4
            buf = (torch.empty(nbytes // 4, dtype=torch.float32, device="cuda")
                   if active else None)

            def _do() -> None:
                if not active:
                    return
                if node == 0:
                    dist.send(buf, dst=gpn + local)      # node0 GPU i -> node1 GPU i
                else:
                    dist.recv(buf, src=local)

            comm_ms = _time_ms(_do, a.iters)
            moved = nbytes if active else 0
            eff = moved / (comm_ms / 1e3) / 1e9 if (moved and comm_ms > 0) else 0.0
            if rank == 0:
                print(f"xpair n={npairs}  {wgb:5.1f} GB/pair  {comm_ms:8.3f} ms  "
                      f"-> {eff:6.1f} GB/s per pair ({eff * npairs:6.1f} aggregate)",
                      flush=True)
            out_rows.append({"gpu": a.gpu_label, "world": world, "tp_from": npairs,
                             "tp_to": npairs, "total_weight_gb": wgb, "k": npairs,
                             "bytes_moved_per_gpu": nbytes, "comm_ms": round(comm_ms, 4),
                             "eff_gbps": round(eff, 2), "pattern": f"xpair{npairs}",
                             "bytes_inter_per_gpu": nbytes})
            if buf is not None:
                del buf
            torch.cuda.empty_cache()

    # node-spanning all-gathers: gpn concurrent groups {i, gpn+i}, k=2
    groups = [dist.new_group([i, gpn + i]) for i in range(gpn)]
    my_group = groups[local] if node < 2 else None
    for wgb in a.weights_gb:
        nbytes = int(wgb * 1e9)
        nbytes -= nbytes % 4
        src = torch.empty(nbytes // 4, dtype=torch.float32, device="cuda")
        dst = [torch.empty_like(src) for _ in range(2)]

        def _do() -> None:
            if my_group is not None:
                dist.all_gather(dst, src, group=my_group)

        comm_ms = _time_ms(_do, a.iters)
        eff = nbytes / (comm_ms / 1e3) / 1e9 if comm_ms > 0 else 0.0
        if rank == 0:
            print(f"xallgather k=2 x{gpn}  {wgb:5.1f} GB moved/gpu  {comm_ms:8.3f} ms  "
                  f"-> {eff:6.1f} GB/s per gpu", flush=True)
        out_rows.append({"gpu": a.gpu_label, "world": world, "tp_from": 2, "tp_to": 1,
                         "total_weight_gb": wgb, "k": 2,
                         "bytes_moved_per_gpu": nbytes, "comm_ms": round(comm_ms, 4),
                         "eff_gbps": round(eff, 2), "pattern": "xallgather",
                         "bytes_inter_per_gpu": nbytes})
        del src, dst
        torch.cuda.empty_cache()
    return out_rows


_UNITS = 24            # per axis: LCM of 1,2,3,4,6,8 -- fine enough for tp/pp up to 8


def _pool_cells(tp: int, pp: int, stage: int, t: int) -> set[int]:
    """Cells of the (layers x width) weight grid held by rank (stage, t) of a tp x pp
    layout. The grid is _UNITS x _UNITS cells, flattened row-major; a pipeline stage
    holds a LAYER-row band, a tp rank a WIDTH-column band. This 2-D model is what lets
    a tp8 <-> tp4xpp2 switch price correctly: both hold 1/8 of W but slice it along
    different axes, so most of the new shard must be fetched (the 1-D unit model
    wrongly called it free)."""
    rows = range(stage * _UNITS // pp, (stage + 1) * _UNITS // pp)
    cols = range(t * _UNITS // tp, (t + 1) * _UNITS // tp)
    return {r * _UNITS + c for r in rows for c in cols}


def _plan_owner_units(spec: str, world: int) -> dict[int, set[int]]:
    """Map each rank to the set of weight-grid cells it holds under a plan side.

    ``spec`` = comma-separated pools, each 'tp:dp' or 'tp:pp:dp'. Ranks fill the pools
    in order; within a pool a dp replica is pp*tp contiguous ranks (stage-major, tp
    within a stage -- vLLM's rank order). Weight content is role-independent, so a cell
    held by ANY rank (either pool, any replica) is a valid source for it."""
    owner: dict[int, set[int]] = {}
    r = 0
    for pool in spec.split(","):
        parts = [int(x) for x in pool.split(":")]
        if len(parts) == 2:
            tp, pp, dp = parts[0], 1, parts[1]
        else:
            tp, pp, dp = parts
        for _ in range(dp):
            for stage in range(pp):
                for t in range(tp):
                    owner[r] = _pool_cells(tp, pp, stage, t)
                    r += 1
    if r != world:
        raise ValueError(f"plan side {spec!r} uses {r} GPUs, world is {world}")
    return owner


def _plan_relabel(old_own: dict[int, set[int]], new_own: dict[int, set[int]],
                  world: int, gpus_per_node: int | None = None) -> dict[int, int]:
    """Assign each physical GPU to a NEW-plan slot to MAXIMISE retained bytes (minimise
    the reshard), a greedy max-overlap matching. Deterministic, so every rank computes
    the same map with no communication. The cost model assumes this optimal relabel; a
    runtime that skips it (naive rank->rank) pays far more -- e.g. the pd_4x4 swap costs
    W/2 without it and 0 with it.

    ``gpus_per_node`` restricts the matching to same-node (GPU, slot) pairs: slots are
    physical positions (a new-plan pool pins its node), and GPUs do not move between
    boxes -- only bytes do."""
    gpn = gpus_per_node or world
    edges = sorted(((len(old_own[p] & new_own[s]), p, s)
                    for p in range(world) for s in range(world)
                    if p // gpn == s // gpn),
                   key=lambda e: (-e[0], e[1], e[2]))
    phys_to_slot: dict[int, int] = {}
    used_slot: set[int] = set()
    for _, p, s in edges:
        if p in phys_to_slot or s in used_slot:
            continue
        phys_to_slot[p] = s
        used_slot.add(s)
    return phys_to_slot


def _coalesce_runs(unit_peer: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge a unit list (sorted by unit) into (peer, run_length) runs, breaking on a
    peer change or a gap in the unit index. Applied identically on the recv and send
    sides so the coalesced messages match one-to-one."""
    runs: list[tuple[int, int]] = []
    last_u = None
    for u, p in unit_peer:
        if runs and runs[-1][0] == p and last_u == u - 1:
            runs[-1] = (p, runs[-1][1] + 1)
        else:
            runs.append((p, 1))
        last_u = u
    return runs


def _measure_plan(a, rank: int, world: int) -> list[dict]:
    """Faithful whole-PD-plan reshard: lay the model out per the OLD plan across all
    GPUs, relabel physical GPUs to the NEW plan optimally, then move exactly the
    byte-ranges each GPU is still missing (P2P from any GPU that holds them) and time it.
    Validates the plan cost model's VOLUME -- which new shards are free vs must gather."""
    out_rows: list[dict] = []
    gpn = int(os.environ.get("LOCAL_WORLD_SIZE", world))
    n_cells = _UNITS * _UNITS
    for pid, spec in enumerate(a.plans):
        old_s, new_s = spec.split(">")
        old_own = _plan_owner_units(old_s, world)
        new_own = _plan_owner_units(new_s, world)
        phys_to_slot = _plan_relabel(old_own, new_own, world, gpus_per_node=gpn)
        my_new = new_own[phys_to_slot[rank]]
        final_own = {p: new_own[phys_to_slot[p]] for p in range(world)}
        my_fetch = sorted(my_new - old_own[rank])
        # One source rank per cell. Every rank must pick the same map, and a REAL
        # runtime pulls from a same-node holder when one exists (NVLink/PCIe beats the
        # fabric): prefer the lowest holder on the DESTINATION's node, else the lowest
        # holder anywhere. Per-destination, so precompute holders once.
        holders = {u: [r for r in range(world) if u in old_own[r]]
                   for u in range(n_cells)}

        def _src_for(u: int, dst: int) -> int:
            same = [r for r in holders[u] if r // gpn == dst // gpn]
            return min(same) if same else min(holders[u])

        src_of = {u: _src_for(u, rank) for u in range(n_cells)}
        # a real reshard moves CONTIGUOUS byte ranges, so coalesce adjacent units that
        # share a source (recv side) / a destination (send side) into one message.
        my_recv_runs = _coalesce_runs([(u, src_of[u]) for u in my_fetch])   # (peer, len)
        my_send_runs = []
        for r in range(world):
            if r == rank:
                continue
            fetched = sorted(u for u in (final_own[r] - old_own[r])
                             if _src_for(u, r) == rank)
            my_send_runs += [(r, ln) for _peer, ln in _coalesce_runs([(u, r) for u in fetched])]
        inter_units = sum(1 for u in my_fetch if src_of[u] // gpn != rank // gpn)
        for wgb in a.weights_gb:
            unit_bytes = int(wgb * 1e9 / n_cells)
            unit_bytes -= unit_bytes % 4
            recv_bufs = [(peer, torch.empty(ln * unit_bytes // 4, dtype=torch.float32,
                                            device="cuda")) for peer, ln in my_recv_runs]
            send_bufs = [(peer, torch.ones(ln * unit_bytes // 4, dtype=torch.float32,
                                           device="cuda")) for peer, ln in my_send_runs]

            def _do() -> None:
                ops = [dist.P2POp(dist.irecv, b, peer) for peer, b in recv_bufs]
                ops += [dist.P2POp(dist.isend, b, peer) for peer, b in send_bufs]
                if ops:
                    for w in dist.batch_isend_irecv(ops):
                        w.wait()

            comm_ms = _time_ms(_do, a.iters)
            local = torch.tensor([float(len(my_fetch) * unit_bytes),
                                  float(inter_units * unit_bytes)], device="cuda")
            dist.all_reduce(local, op=dist.ReduceOp.MAX)
            moved = float(local[0].item())               # bottleneck GPU's fetched bytes
            inter = float(local[1].item())               # bottleneck cross-node share
            eff = moved / (comm_ms / 1e3) / 1e9 if (moved and comm_ms > 0) else 0.0
            out_rows.append({"gpu": a.gpu_label, "world": world, "tp_from": pid,
                             "tp_to": pid, "total_weight_gb": wgb, "k": pid,
                             "bytes_moved_per_gpu": moved, "comm_ms": round(comm_ms, 4),
                             "eff_gbps": round(eff, 2), "pattern": f"plan:{spec}",
                             "bytes_inter_per_gpu": inter})
            if rank == 0:
                print(f"plan {spec:22s} W={wgb:5.0f}GB  move {moved / 1e9:7.3f} GB/gpu"
                      f" (x-node {inter / 1e9:6.3f})  {comm_ms:8.3f} ms  -> {eff:7.1f} GB/s",
                      flush=True)
            del recv_bufs, send_bufs
            torch.cuda.empty_cache()
    return out_rows


if __name__ == "__main__":
    main()
