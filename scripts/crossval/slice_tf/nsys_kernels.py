"""Per-family / per-kernel GPU-time attribution from an nsys sqlite export.

The methodology is `docs/perf/2026-08-25-deepseek-v4-ablation.md`'s: take a
steady-state window, divide by a known step count, and report BOTH the sum of
kernel durations (what the work costs) and the union of their intervals (what
the wall clock sees). The two differ by exactly the overlap, so quoting only
the sum hides serialisation and quoting only the union hides duplicated work.

Windows come from one of two places, and both mean "exactly N decode steps":

* `--window NAME` — an NVTX range pushed by the harness (vLLM side).
* no `--window` — the whole capture, which for our own bench is already
  bracketed by `cuProfilerStart/Stop` (`KIMI_BENCH_PROFILE=1`).

`--subtract NAME` subtracts a second window's totals before dividing, which is
how the vLLM side removes a prefill it cannot bracket separately.

    nsys export --type sqlite -o out.sqlite out.nsys-rep
    python nsys_kernels.py out.sqlite --steps 10
    python nsys_kernels.py vllm.sqlite --window prefill_decode \
        --subtract prefill_only --steps 200
"""

import argparse
import re
import sqlite3
import sys

# Ordered: first match wins. One list covers both engines on purpose — the point
# of the exercise is a like-for-like split, so "expert weights" must mean the
# same row whether the kernel is MARLIN or ours, and the candle op-chain that
# used to stand in for an expert GEMM has to land in a family that names it.
FAMILIES = [
    # ---- the MXFP4 expert path, both engines ----
    (r"kimi_moe_gemv", "expert MXFP4 GEMV (ours, fused)"),
    (r"Marlin", "expert MXFP4 GEMM (vLLM MARLIN)"),
    (r"moe_align|moe_sum|count_and_sort_expert|group_topk|kimi_moe_topk",
     "MoE routing / scatter"),
    # ---- ours: the rest of the crate ----
    # These sit ahead of the vLLM block because `kimi_attn_res` would otherwise
    # be swallowed by its `attn_` pattern. The op-chain fusion wave's kernels
    # are deliberately mapped onto the SAME rows as vLLM's equivalents — that is
    # the whole point of one family list for two engines. Ours had no such row
    # before 2026-09-02 because the work was spread across elementwise, cast and
    # reduce; a fused kernel belongs where the engine that already fused it puts
    # its own.
    (r"kimi_kda_decode|kimi_attn_res", "KDA / attention (fused)"),
    (r"kimi_(rmsnorm|situ|moe_combine)", "act / norm (fused)"),
    (r"kimi_(scatter_rows|copy|causal_mask)", "cache/state plumbing"),
    # ---- vLLM: its own fused kernels ----
    (r"kda_decode_fusion|MLADecode|attn_res_kernel|flash|fmha|attn_",
     "KDA / attention (fused)"),
    (r"situ_and_mul|rmsnorm_kernel|rms_norm|_fused_q_kv", "act / norm (fused)"),
    # `one_shot_all_reduce` has to be listed EXPLICITLY, and the underscored
    # spellings alongside the camel-case ones. Until 2026-09-02 neither was:
    # `allreduce` does not match `all_reduce` even case-insensitively, so
    # llama's one-shot NVLink kernel fell through to "reduce / norm" on the word
    # "reduce". That is how PART 6 of the slice bench came to report our
    # collectives at 0.013 ms/GPU/step — the single NCCL call — against vLLM's
    # 0.377 and call it a 29x win, while 1.03 ms/GPU/step of one-shot all-reduce
    # sat hidden inside the reduce/norm row it was trying to attribute to candle
    # op chains. Two rows of that table were wrong in opposite directions.
    (r"one_shot|nccl|AllReduce|Allreduce|allreduce|all_reduce|"
     r"cross_device_reduce|AllGather|all_gather",
     "collective (TP)"),
    # ---- shared: real GEMMs ----
    (r"nvjet|cutlass|cute_|(sgemm|hgemm|gemv|gemm)|nn_|xmma|sm90_|ampere_|"
     r"^gemmk1|splitKreduce|dot_kernel|reduce_1Block", "GEMM / GEMV (library)"),
    # ---- the candle op chain the fused GEMV replaced ----
    (r"index_select|is_u32|gather", "index_select / gather"),
    (r"ucopy|copy2d|ccopy|scopy|bcopy|copy_", "copy / cat"),
    (r"cast|convert", "dtype cast"),
    (r"affine|mul|add|sub|div|binary|ub[a-z]*_|bfloat|maximum|minimum|neg|"
     r"floor|exp|sigmoid|silu|sqr|sqrt|recip|where|powf|elu|relu|gelu|tanh",
     "elementwise"),
    (r"reduce|sum|max_|min_|softmax|layernorm|argmax|fast_", "reduce / norm"),
    (r"sort|topk", "sort / top-k"),
    (r"fill|memset", "fill"),
]


def family(name):
    for pat, fam in FAMILIES:
        if re.search(pat, name, re.I):
            return fam
    return "other"


def union_ms(intervals):
    """Total wall time covered by `intervals` (ns pairs), in ms."""
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    total += ce - cs
    return total / 1e6


def gaps(intervals, min_ns):
    """Idle stretches between merged intervals, in ns."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = []
    _, ce = intervals[0]
    for s, e in intervals[1:]:
        if s > ce:
            if s - ce >= min_ns:
                out.append(s - ce)
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return out


def name_column(con, table):
    cols = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
    for c in ("shortName", "demangledName", "nameId", "name"):
        if c in cols:
            return c
    return None


def load_kernels(con, window, table="CUPTI_ACTIVITY_KIND_KERNEL"):
    """Kernels fully inside `window`. The bound is pushed into SQL rather than
    filtered in Python: a vLLM capture that also traced the 197 GiB model load
    holds ~90 M events, which is a multi-GB materialisation for a window that
    keeps a few hundred thousand."""
    col = name_column(con, table)
    if col is None:
        return []
    q = (f"SELECT k.start, k.end, k.deviceId, COALESCE(s.value, 'unknown') "
         f"FROM {table} k LEFT JOIN StringIds s ON s.id = k.{col}")
    if window:
        return list(con.execute(q + " WHERE k.start >= ? AND k.end <= ?", window))
    return list(con.execute(q))


def load_memops(con, window):
    rows = []
    for table, kind in (("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"),
                        ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
        q = f"SELECT start, end, deviceId FROM {table}"
        try:
            if window:
                q += " WHERE start >= ? AND end <= ?"
                rows += [(s, e, d, kind) for s, e, d in con.execute(q, window)]
            else:
                rows += [(s, e, d, kind) for s, e, d in con.execute(q)]
        except sqlite3.OperationalError:
            pass
    return rows


def nvtx_window(con, name):
    try:
        rows = list(con.execute(
            "SELECT e.start, e.end FROM NVTX_EVENTS e "
            "LEFT JOIN StringIds s ON s.id = e.textId "
            "WHERE e.text = ? OR s.value = ?", (name, name)))
    except sqlite3.OperationalError:
        rows = []
    rows = [r for r in rows if r[1] is not None]
    if not rows:
        sys.exit(f"no NVTX range named {name!r} in the export")
    if len(rows) > 1:
        sys.exit(f"NVTX range {name!r} occurs {len(rows)} times; expected once")
    return rows[0]


def tally(kernels, memops):
    per_kernel, per_device = {}, {}
    for s, e, dev, nm in kernels:
        d = e - s
        k = per_kernel.setdefault(nm, [0, 0])
        k[0] += 1
        k[1] += d
        per_device.setdefault(dev, []).append((s, e))
    mem = {}
    for s, e, dev, kind in memops:
        m = mem.setdefault(kind, [0, 0])
        m[0] += 1
        m[1] += e - s
    return per_kernel, per_device, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--window", help="NVTX range name; default = whole capture")
    ap.add_argument("--subtract", help="NVTX range whose totals to subtract")
    ap.add_argument("--steps", type=int, required=True,
                    help="decode steps the (possibly subtracted) window holds")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--gap-us", type=float, default=20.0,
                    help="idle stretches at least this long are reported")
    ap.add_argument("--tail-ms", type=float,
                    help="restrict the per-device section to the last N ms of "
                         "the window (a vLLM window opens with a prefill)")
    args = ap.parse_args()

    con = sqlite3.connect(args.sqlite)
    win = nvtx_window(con, args.window) if args.window else None
    kernels = load_kernels(con, win)
    memops = load_memops(con, win)
    if not kernels:
        sys.exit("no kernel rows — was the export made with -t cuda?")

    pk, pd, mem = tally(kernels, memops)
    if args.subtract:
        w2 = nvtx_window(con, args.subtract)
        pk2, pd2, mem2 = tally(load_kernels(con, w2), load_memops(con, w2))
        for nm, (c, d) in pk2.items():
            if nm in pk:
                pk[nm][0] -= c
                pk[nm][1] -= d
        for kind, (c, d) in mem2.items():
            if kind in mem:
                mem[kind][0] -= c
                mem[kind][1] -= d
        pk = {nm: v for nm, v in pk.items() if v[0] > 0}

    n = args.steps
    total_ms = sum(d for _, d in pk.values()) / 1e6
    total_launches = sum(c for c, _ in pk.values())

    print(f"window: {args.window or 'whole capture'}"
          + (f" minus {args.subtract}" if args.subtract else "")
          + f", {n} decode steps")
    print(f"kernel launches: {total_launches} total, "
          f"{total_launches / n:.1f} per step")
    print(f"summed kernel time: {total_ms:.2f} ms total, "
          f"{total_ms / n:.3f} ms per step\n")

    fam = {}
    for nm, (c, d) in pk.items():
        f = fam.setdefault(family(nm), [0, 0])
        f[0] += c
        f[1] += d
    print(f"{'family':38s} {'launches/step':>13s} {'ms/step':>9s} {'share':>7s}")
    for f, (c, d) in sorted(fam.items(), key=lambda kv: -kv[1][1]):
        ms = d / 1e6 / n
        print(f"{f:38s} {c / n:13.1f} {ms:9.3f} {100 * d / 1e6 / n / (total_ms / n):6.1f}%")

    print(f"\n{'kernel':58s} {'launches/step':>13s} {'ms/step':>9s} {'us/call':>9s}")
    for nm, (c, d) in sorted(pk.items(), key=lambda kv: -kv[1][1])[:args.top]:
        print(f"{nm[:58]:58s} {c / n:13.1f} {d / 1e6 / n:9.3f} "
              f"{d / 1e3 / max(c, 1):9.2f}")

    if mem:
        print()
        for kind, (c, d) in sorted(mem.items()):
            print(f"{kind:>10s}: {c / n:8.1f} per step, {d / 1e6 / n:7.3f} ms/step")

    if args.tail_ms:
        # Steady-state tail of the window: the last `tail_ms` of wall clock,
        # which for a vLLM window that opens with a prefill is the only part
        # that is purely decode. `--steps` must name the step count in it.
        hi = win[1] if win else max(e for _, e, _, _ in kernels)
        kernels = [r for r in kernels if r[0] >= hi - int(args.tail_ms * 1e6)]
        pd = tally(kernels, [])[1]
        win = (hi - int(args.tail_ms * 1e6), hi)

    if not args.subtract or args.tail_ms:
        print("\nper-device occupancy over the window "
              "(sum = work, union = wall time it covers)")
        wlo = min(s for s, _, _, _ in kernels) if win is None else win[0]
        whi = max(e for _, e, _, _ in kernels) if win is None else win[1]
        span = (whi - wlo) / 1e6
        print(f"{'dev':>3s} {'sum ms/step':>12s} {'union ms/step':>14s} "
              f"{'busy%':>7s} {'gaps/step':>10s} {'idle ms/step':>13s}")
        for dev in sorted(pd):
            iv = pd[dev]
            s_ms = sum(e - s for s, e in iv) / 1e6
            u_ms = union_ms(iv)
            g = gaps(iv, int(args.gap_us * 1000))
            print(f"{dev:3d} {s_ms / n:12.3f} {u_ms / n:14.3f} "
                  f"{100 * u_ms / span:6.1f}% {len(g) / n:10.1f} "
                  f"{sum(g) / 1e6 / n:13.3f}")
        print(f"window span {span:.2f} ms, {span / n:.3f} ms/step")


if __name__ == "__main__":
    main()
