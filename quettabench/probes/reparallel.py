#!/usr/bin/env python3
"""Cost of switching parallelism in place over NVLink, under torchrun."""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.distributed as dist

from quettabench.probes import _timing
from quettabench.probes.base import ProbeRow, add_output_args, emit, run_recorded

KEY = ("gpu", "world", "tp_from", "tp_to", "total_weight_gb")
COLS = KEY + (
    "k",
    "bytes_moved_per_gpu",
    "comm_ms",
    "eff_gbps",
    "pattern",
    "bytes_inter_per_gpu",
)


def _fit(rows: list[dict]) -> tuple[float, float]:
    x = np.array([r["bytes_moved_per_gpu"] for r in rows], dtype=float)
    y = np.array([r["comm_ms"] for r in rows], dtype=float)
    keep = x > 0
    if keep.sum() < 2:
        return float(np.median(y)) if len(y) else 0.0, float("inf")
    x, y = x[keep], y[keep]
    A = np.stack([np.ones_like(x), x], axis=1)
    (b0, b1), *_ = np.linalg.lstsq(A, y, rcond=None)
    bw = 1.0 / (b1 / 1e3) if b1 > 0 else float("inf")
    return b0, bw


def _emit(table: str, rows: list[dict]) -> None:
    """The last row on a KEY wins."""
    cells = {tuple(str(r[k]) for k in KEY): {k: str(r.get(k, 0)) for k in COLS} for r in rows}
    for k in sorted(cells, key=lambda t: (t[0], int(t[1]), int(t[2]), int(t[3]), float(t[4]))):
        row = cells[k]
        emit(
            ProbeRow(
                "reparallel",
                table,
                {c: row[c] for c in KEY},
                {c: row[c] for c in COLS if c not in KEY},
                "host_wallclock",
            )
        )


def _time_ms(fn, iters: int) -> float:
    samples = []
    for it in range(iters + 1):
        dist.barrier()
        ms = _timing.time_once_ms(fn)
        if it > 0:
            samples.append(ms)
    return float(np.median(samples))


def _groups_of(k: int, world: int) -> list[list[int]]:
    return [list(range(g, g + k)) for g in range(0, world, k)]


def main(argv=None) -> bool:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_output_args(ap)
    ap.add_argument(
        "--switches",
        nargs="+",
        default=["4:2", "2:1", "8:4", "4:1", "8:2", "2:4", "4:8"],
        help="tp_from:tp_to pairs (down = reshard, up = discard floor)",
    )
    ap.add_argument(
        "--mode",
        default="tp",
        choices=["tp", "dp", "ep", "pp", "kv", "plan", "cross_node"],
        help="tp: fixed-pool TP reshard (all-gather); dp: replica "
        "replication (broadcast a shard to fresh GPUs); ep: expert "
        "restage (all-to-all); pp: pipeline-stage reshard (whole-layer "
        "all-gather); kv: in-flight paged-KV reshard (head-shard "
        "all-gather); plan: whole PD-plan reshard (faithful per-GPU "
        "byte-range exchange for a P/D layout change; node-aware when "
        "launched across nodes); cross_node: cross-node link physics -- "
        "concurrent p2p pair sweep (rail sharing / replica hydration) "
        "and node-spanning all-gathers. One dim per mode.",
    )
    ap.add_argument(
        "--plans",
        nargs="+",
        default=["2:2,4:1>4:1,2:2", "4:1,4:1>2:3,2:1", "2:2,4:1>2:2,1:4", "4:1,4:1>4:1,4:1"],
        help="mode=plan: OLD>NEW where each side is 'tp_p:dp_p,tp_d:dp_d' "
        "(prefill pool, decode pool); worlds must sum to --nproc",
    )
    ap.add_argument(
        "--weights-gb",
        type=float,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32],
        help="total logical weight W swept; per-GPU move derived from it",
    )
    ap.add_argument("--iters", type=int, default=11)
    ap.add_argument(
        "--kv-tokens",
        type=int,
        nargs="+",
        dest="kv_tokens",
        default=[1024, 4096, 16384, 65536, 131072, 262144],
        help="mode=kv: per-request token counts to reshard",
    )
    ap.add_argument("--layers", type=int, default=94, help="mode=kv: model layers")
    ap.add_argument("--head-dim", type=int, default=128, help="mode=kv: KV head dim")
    ap.add_argument(
        "--out-name",
        default=None,
        help="CSV stem (default <gpu>_reshard); use e.g. <gpu>_reshard_val "
        "for a held-out validation sweep that must not touch calibration",
    )
    a = ap.parse_args(argv)
    _timing.require_backend("reparallel", "cuda", "rocm")

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    _timing.set_device(local)
    dist.init_process_group("nccl", rank=rank, world_size=world)

    # Creating a group is collective, so each is made once.
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
    elif a.mode == "cross_node":
        out_rows = _measure_cross_node(a, rank, world)
    else:
        out_rows = _measure_tp(a, rank, world, _get_groups)

    if rank == 0:
        default_stem = {
            "tp": "reshard",
            "dp": "reshard_dp",
            "ep": "reshard_ep",
            "pp": "reshard_pp",
            "kv": "reshard_kv",
            "plan": "reshard_plan",
            "cross_node": "reshard_xnode",
        }[a.mode]
        stem = a.out_name or f"{a.gpu_label}_{default_stem}"
        _emit(f"reparallel/{stem}.csv", out_rows)
        print(f"\n{len(out_rows)} rows of reparallel/{stem}.csv")
        print(f"\nfit comm_ms = floor + bytes/bw   (mode={a.mode}, per GPU)")
        seen = []
        pairs = sorted({(r["tp_from"], r["tp_to"]) for r in out_rows})
        for p, q in pairs:
            rows = [r for r in out_rows if r["tp_from"] == p and r["tp_to"] == q and r["bytes_moved_per_gpu"] > 0]
            if len(rows) >= 2:
                floor, bw = _fit(rows)
                seen.append(bw)
                print(f"  {p}->{q}: floor {floor:7.3f} ms   bw {bw / 1e9:6.1f} GB/s")
        if seen:
            print(f"\n  bw_bytes_per_s (median): {np.median(seen):.3e}")
    dist.destroy_process_group()
    return rank == 0


def _measure_tp(a, rank: int, world: int, _get_groups) -> list[dict]:
    """All-gather on tp down, discard on tp up."""
    out_rows: list[dict] = []
    for sw in a.switches:
        p, q = (int(x) for x in sw.split(":"))
        if world % p or world % q:
            if rank == 0:
                print(f"skip {sw}: world {world} not divisible by both tp", flush=True)
            continue
        my_group = None
        if q < p:
            k = p // q
            groups = _get_groups(k)
            my_group = groups[rank // k]
        for wgb in a.weights_gb:
            W = wgb * 1e9
            in_bytes = int(W / p)
            in_bytes -= in_bytes % 4  # float32 aligned
            if q < p:
                k = p // q
                moved = (k - 1) * in_bytes
                src = torch.empty(in_bytes // 4, dtype=torch.float32, device=_timing.device())
                dst = [torch.empty_like(src) for _ in range(k)]

                def _do() -> None:
                    dist.all_gather(dst, src, group=my_group)

                pattern = "allgather"
            else:  # tp up: only a local re-pack
                k = q // p
                moved = 0
                keep = in_bytes // k
                src = torch.empty(in_bytes // 4, dtype=torch.float32, device=_timing.device())
                dst_buf = torch.empty(keep // 4, dtype=torch.float32, device=_timing.device())

                def _do() -> None:
                    dst_buf.copy_(src[: keep // 4])

                pattern = "discard"
            comm_ms = _time_ms(_do, a.iters)
            eff = moved / (comm_ms / 1e3) / 1e9 if (moved and comm_ms > 0) else 0.0
            row = {
                "gpu": a.gpu_label,
                "world": world,
                "tp_from": p,
                "tp_to": q,
                "total_weight_gb": wgb,
                "k": k,
                "bytes_moved_per_gpu": moved,
                "comm_ms": round(comm_ms, 4),
                "eff_gbps": round(eff, 2),
                "pattern": pattern,
            }
            out_rows.append(row)
            if rank == 0:
                print(
                    f"tp{p}->tp{q} {pattern:9s} W={wgb:5.0f}GB  "
                    f"move {moved / 1e9:7.3f} GB/gpu  {comm_ms:8.3f} ms  "
                    f"-> {eff:7.1f} GB/s",
                    flush=True,
                )
            del src
            if q < p:
                del dst
            _timing.empty_cache()
    return out_rows


def _measure_dp(a, rank: int, world: int) -> list[dict]:
    """A fresh replica pulls its whole W/tp shard by P2P send, even->odd pairs at once."""
    out_rows: list[dict] = []
    if world < 2:
        return out_rows
    peer = rank ^ 1
    for tp in (1, 2, 4, 8):
        for wgb in a.weights_gb:
            nbytes = int(wgb * 1e9 / tp)
            nbytes -= nbytes % 4
            buf = torch.empty(nbytes // 4, dtype=torch.float32, device=_timing.device())

            def _do() -> None:
                if rank % 2 == 0:
                    dist.send(buf, dst=peer)
                else:
                    dist.recv(buf, src=peer)

            comm_ms = _time_ms(_do, a.iters)
            moved = nbytes
            eff = moved / (comm_ms / 1e3) / 1e9 if comm_ms > 0 else 0.0
            row = {
                "gpu": a.gpu_label,
                "world": world,
                "tp_from": tp,
                "tp_to": tp,
                "total_weight_gb": wgb,
                "k": 1,
                "bytes_moved_per_gpu": moved,
                "comm_ms": round(comm_ms, 4),
                "eff_gbps": round(eff, 2),
                "pattern": "p2p_bcast",
            }
            out_rows.append(row)
            if rank == 0:
                print(
                    f"dp replicate tp{tp} W={wgb:5.0f}GB  move {moved / 1e9:7.3f} GB/gpu"
                    f"  {comm_ms:8.3f} ms  -> {eff:7.1f} GB/s",
                    flush=True,
                )
            del buf
            _timing.empty_cache()
    return out_rows


def _measure_ep(a, rank: int, world: int) -> list[dict]:
    """all_to_all_single at GB scale; each GPU moves (ep-1)/ep of its expert bytes."""
    out_rows: list[dict] = []
    ep = world
    if ep < 2:
        return out_rows
    for wgb in a.weights_gb:
        chunk = int(wgb * 1e9 / ep) // ep
        chunk -= chunk % 4
        per_gpu = chunk * ep
        src = torch.empty(per_gpu // 4, dtype=torch.float32, device=_timing.device())
        dst = torch.empty_like(src)

        def _do() -> None:
            dist.all_to_all_single(dst, src)

        comm_ms = _time_ms(_do, a.iters)
        moved = (ep - 1) * chunk
        eff = moved / (comm_ms / 1e3) / 1e9 if comm_ms > 0 else 0.0
        row = {
            "gpu": a.gpu_label,
            "world": world,
            "tp_from": ep,
            "tp_to": ep,
            "total_weight_gb": wgb,
            "k": ep,
            "bytes_moved_per_gpu": moved,
            "comm_ms": round(comm_ms, 4),
            "eff_gbps": round(eff, 2),
            "pattern": "all_to_all",
        }
        out_rows.append(row)
        if rank == 0:
            print(
                f"ep restage ep{ep} W={wgb:5.0f}GB  move {moved / 1e9:7.3f} GB/gpu"
                f"  {comm_ms:8.3f} ms  -> {eff:7.1f} GB/s",
                flush=True,
            )
        del src, dst
        _timing.empty_cache()
    return out_rows


def _measure_pp(a, rank: int, world: int, _get_groups) -> list[dict]:
    """Same all-gather as tp but over whole layers, to confirm the tp grid prices it."""
    out_rows: list[dict] = []
    for sw in a.switches:
        p, q = (int(x) for x in sw.split(":"))
        if world % p or world % q or q >= p:
            continue
        k = p // q
        my_group = _get_groups(k)[rank // k]
        for wgb in a.weights_gb:
            in_bytes = int(wgb * 1e9 / p)
            in_bytes -= in_bytes % 4
            src = torch.empty(in_bytes // 4, dtype=torch.float32, device=_timing.device())
            dst = [torch.empty_like(src) for _ in range(k)]

            def _do() -> None:
                dist.all_gather(dst, src, group=my_group)

            comm_ms = _time_ms(_do, a.iters)
            moved = (k - 1) * in_bytes
            eff = moved / (comm_ms / 1e3) / 1e9 if comm_ms > 0 else 0.0
            out_rows.append(
                {
                    "gpu": a.gpu_label,
                    "world": world,
                    "tp_from": p,
                    "tp_to": q,
                    "total_weight_gb": wgb,
                    "k": k,
                    "bytes_moved_per_gpu": moved,
                    "comm_ms": round(comm_ms, 4),
                    "eff_gbps": round(eff, 2),
                    "pattern": "pp_allgather",
                }
            )
            if rank == 0:
                print(
                    f"pp{p}->pp{q} W={wgb:5.0f}GB  move {moved / 1e9:7.3f} GB/gpu"
                    f"  {comm_ms:8.3f} ms  -> {eff:7.1f} GB/s",
                    flush=True,
                )
            del src, dst
            _timing.empty_cache()
    return out_rows


def _measure_kv(a, rank: int, world: int, _get_groups) -> list[dict]:
    """All-gather of a contiguous per-rank KV buffer."""
    out_rows: list[dict] = []
    k = 2
    my_group = _get_groups(k)[rank // k]
    heads_per_rank = 1
    per_tok = 2 * heads_per_rank * a.head_dim * 2 * a.layers
    for tokens in a.kv_tokens:
        nbytes = tokens * per_tok
        nbytes -= nbytes % 4
        src = torch.empty(nbytes // 4, dtype=torch.float32, device=_timing.device())
        dst = [torch.empty_like(src) for _ in range(k)]

        def _do() -> None:
            dist.all_gather(dst, src, group=my_group)

        comm_ms = _time_ms(_do, a.iters)
        moved = (k - 1) * nbytes
        eff = moved / (comm_ms / 1e3) / 1e9 if comm_ms > 0 else 0.0
        out_rows.append(
            {
                "gpu": a.gpu_label,
                "world": world,
                "tp_from": 4,
                "tp_to": 2,
                "total_weight_gb": round(tokens / 1e3, 3),
                "k": tokens,
                "bytes_moved_per_gpu": moved,
                "comm_ms": round(comm_ms, 4),
                "eff_gbps": round(eff, 2),
                "pattern": "kv_allgather",
            }
        )
        if rank == 0:
            print(
                f"kv reshard tok={tokens:>7}  move {moved / 1e9:7.3f} GB/gpu  {comm_ms:8.3f} ms  -> {eff:7.1f} GB/s",
                flush=True,
            )
        del src, dst
        _timing.empty_cache()
    return out_rows


def _measure_cross_node(a, rank: int, world: int) -> list[dict]:
    """Launch with torchrun --nnodes=2."""
    gpn = int(os.environ.get("LOCAL_WORLD_SIZE", world))
    n_nodes = world // gpn
    out_rows: list[dict] = []
    if n_nodes < 2:
        if rank == 0:
            print("cross_node mode needs --nnodes >= 2; nothing to measure", flush=True)
        return out_rows
    node, local = rank // gpn, rank % gpn

    npairs_axis = [n for n in (1, 2, 4, 8) if n <= gpn]
    for npairs in npairs_axis:
        active = node < 2 and local < npairs
        for wgb in a.weights_gb:
            nbytes = int(wgb * 1e9)
            nbytes -= nbytes % 4
            buf = torch.empty(nbytes // 4, dtype=torch.float32, device=_timing.device()) if active else None

            def _do() -> None:
                if not active:
                    return
                if node == 0:
                    dist.send(buf, dst=gpn + local)
                else:
                    dist.recv(buf, src=local)

            comm_ms = _time_ms(_do, a.iters)
            moved = nbytes if active else 0
            eff = moved / (comm_ms / 1e3) / 1e9 if (moved and comm_ms > 0) else 0.0
            if rank == 0:
                print(
                    f"xpair n={npairs}  {wgb:5.1f} GB/pair  {comm_ms:8.3f} ms  "
                    f"-> {eff:6.1f} GB/s per pair ({eff * npairs:6.1f} aggregate)",
                    flush=True,
                )
            out_rows.append(
                {
                    "gpu": a.gpu_label,
                    "world": world,
                    "tp_from": npairs,
                    "tp_to": npairs,
                    "total_weight_gb": wgb,
                    "k": npairs,
                    "bytes_moved_per_gpu": nbytes,
                    "comm_ms": round(comm_ms, 4),
                    "eff_gbps": round(eff, 2),
                    "pattern": f"xpair{npairs}",
                    "bytes_inter_per_gpu": nbytes,
                }
            )
            if buf is not None:
                del buf
            _timing.empty_cache()

    groups = [dist.new_group([i, gpn + i]) for i in range(gpn)]
    my_group = groups[local] if node < 2 else None
    for wgb in a.weights_gb:
        nbytes = int(wgb * 1e9)
        nbytes -= nbytes % 4
        src = torch.empty(nbytes // 4, dtype=torch.float32, device=_timing.device())
        dst = [torch.empty_like(src) for _ in range(2)]

        def _do() -> None:
            if my_group is not None:
                dist.all_gather(dst, src, group=my_group)

        comm_ms = _time_ms(_do, a.iters)
        eff = nbytes / (comm_ms / 1e3) / 1e9 if comm_ms > 0 else 0.0
        if rank == 0:
            print(
                f"xallgather k=2 x{gpn}  {wgb:5.1f} GB moved/gpu  {comm_ms:8.3f} ms  -> {eff:6.1f} GB/s per gpu",
                flush=True,
            )
        out_rows.append(
            {
                "gpu": a.gpu_label,
                "world": world,
                "tp_from": 2,
                "tp_to": 1,
                "total_weight_gb": wgb,
                "k": 2,
                "bytes_moved_per_gpu": nbytes,
                "comm_ms": round(comm_ms, 4),
                "eff_gbps": round(eff, 2),
                "pattern": "xallgather",
                "bytes_inter_per_gpu": nbytes,
            }
        )
        del src, dst
        _timing.empty_cache()
    return out_rows


_UNITS = 24


def _pool_cells(tp: int, pp: int, stage: int, t: int) -> set[int]:
    """A stage holds a row band, a tp rank a column band."""
    rows = range(stage * _UNITS // pp, (stage + 1) * _UNITS // pp)
    cols = range(t * _UNITS // tp, (t + 1) * _UNITS // tp)
    return {r * _UNITS + c for r in rows for c in cols}


def _plan_owner_units(spec: str, world: int) -> dict[int, set[int]]:
    """spec is comma-separated pools, each 'tp:dp' or 'tp:pp:dp'."""
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


def _plan_relabel(
    old_own: dict[int, set[int]], new_own: dict[int, set[int]], world: int, gpus_per_node: int | None = None
) -> dict[int, int]:
    """Deterministic, so every rank agrees without communication."""
    gpn = gpus_per_node or world
    edges = sorted(
        ((len(old_own[p] & new_own[s]), p, s) for p in range(world) for s in range(world) if p // gpn == s // gpn),
        key=lambda e: (-e[0], e[1], e[2]),
    )
    phys_to_slot: dict[int, int] = {}
    used_slot: set[int] = set()
    for _, p, s in edges:
        if p in phys_to_slot or s in used_slot:
            continue
        phys_to_slot[p] = s
        used_slot.add(s)
    return phys_to_slot


def _coalesce_runs(unit_peer: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Recv and send sides use it alike so messages pair one-to-one."""
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
    """Relabel GPUs, then time P2P moves of the byte ranges each GPU lacks."""
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
        # Every rank must pick the same source: the lowest holder on the destination's node, else the lowest anywhere.
        holders = {u: [r for r in range(world) if u in old_own[r]] for u in range(n_cells)}

        def _src_for(u: int, dst: int) -> int:
            same = [r for r in holders[u] if r // gpn == dst // gpn]
            return min(same) if same else min(holders[u])

        src_of = {u: _src_for(u, rank) for u in range(n_cells)}
        # A real reshard moves contiguous ranges: adjacent units with one peer become one message.
        my_recv_runs = _coalesce_runs([(u, src_of[u]) for u in my_fetch])
        my_send_runs = []
        for r in range(world):
            if r == rank:
                continue
            fetched = sorted(u for u in (final_own[r] - old_own[r]) if _src_for(u, r) == rank)
            my_send_runs += [(r, ln) for _peer, ln in _coalesce_runs([(u, r) for u in fetched])]
        inter_units = sum(1 for u in my_fetch if src_of[u] // gpn != rank // gpn)
        for wgb in a.weights_gb:
            unit_bytes = int(wgb * 1e9 / n_cells)
            unit_bytes -= unit_bytes % 4
            recv_bufs = [
                (peer, torch.empty(ln * unit_bytes // 4, dtype=torch.float32, device=_timing.device()))
                for peer, ln in my_recv_runs
            ]
            send_bufs = [
                (peer, torch.ones(ln * unit_bytes // 4, dtype=torch.float32, device=_timing.device()))
                for peer, ln in my_send_runs
            ]

            def _do() -> None:
                ops = [dist.P2POp(dist.irecv, b, peer) for peer, b in recv_bufs]
                ops += [dist.P2POp(dist.isend, b, peer) for peer, b in send_bufs]
                if ops:
                    for w in dist.batch_isend_irecv(ops):
                        w.wait()

            comm_ms = _time_ms(_do, a.iters)
            local = torch.tensor(
                [float(len(my_fetch) * unit_bytes), float(inter_units * unit_bytes)], device=_timing.device()
            )
            dist.all_reduce(local, op=dist.ReduceOp.MAX)
            moved = float(local[0].item())
            inter = float(local[1].item())
            eff = moved / (comm_ms / 1e3) / 1e9 if (moved and comm_ms > 0) else 0.0
            out_rows.append(
                {
                    "gpu": a.gpu_label,
                    "world": world,
                    "tp_from": pid,
                    "tp_to": pid,
                    "total_weight_gb": wgb,
                    "k": pid,
                    "bytes_moved_per_gpu": moved,
                    "comm_ms": round(comm_ms, 4),
                    "eff_gbps": round(eff, 2),
                    "pattern": f"plan:{spec}",
                    "bytes_inter_per_gpu": inter,
                }
            )
            if rank == 0:
                print(
                    f"plan {spec:22s} W={wgb:5.0f}GB  move {moved / 1e9:7.3f} GB/gpu"
                    f" (x-node {inter / 1e9:6.3f})  {comm_ms:8.3f} ms  -> {eff:7.1f} GB/s",
                    flush=True,
                )
            del recv_bufs, send_bufs
            _timing.empty_cache()
    return out_rows


if __name__ == "__main__":
    run_recorded(main, "reparallel")
