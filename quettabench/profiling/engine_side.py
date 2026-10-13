"""Profiler sources read from the engine side: /metrics, armed traces, nsys kernel tables, probe wrappers."""

import importlib
import json
import re
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path

from quettabench.engines import launcher
from quettabench.metrics.summary import counter_total, parse_prometheus

# vLLM renamed gpu_prefix_cache_* to prefix_cache_*; the old names are still read.
PREFIX_CACHE_COUNTERS = (
    ("vllm:prefix_cache_hits", "vllm:prefix_cache_queries"),
    ("vllm:gpu_prefix_cache_hits", "vllm:gpu_prefix_cache_queries"),
)

# Imported on demand: it needs QuettaSim, which the GPU-free profiler must not.
FRONTEND = "quettabench.profiling.client_side"

# nsys writes <out>.nsys-rep, or <out>.qdrep before 2021.4.
TRACE_FILES = {
    "nsys": ("nsys.nsys-rep", "nsys.qdrep", "nsys.rep"),
    "torch": ("torch_trace.json",),
}


def prefix_cache_hit_rate(metrics):
    for hits_name, queries_name in PREFIX_CACHE_COUNTERS:
        hits, queries = counter_total(metrics, hits_name), counter_total(metrics, queries_name)
        if hits is not None and queries:
            return hits / queries
    return None


def scrape_metrics(endpoint, timeout_s=10):
    with launcher.urlopen(f"{endpoint}/metrics", timeout=timeout_s) as resp:
        return parse_prometheus(resp.read().decode())


def metrics_section(metrics):
    samples = {}
    for key, value in metrics.items():
        if isinstance(key, tuple):
            name, labels = key
            key = name + "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}"
        samples[key] = value
    return {"samples": samples, "prefix_cache_hit_rate": prefix_cache_hit_rate(metrics)}


def collect_armed_traces(results_dir, instrumentation):
    """`missing` is true when none exist yet."""
    names = TRACE_FILES.get(instrumentation, ())
    found = [n for n in names if (Path(results_dir) / n).exists()]
    return {"instrumentation": instrumentation, "files": found, "missing": bool(names) and not found}


# First match wins; the kimi_* patterns are QuettaServe kernel symbols.
KERNEL_FAMILIES = [
    (r"kimi_moe_gemv", "expert MXFP4 GEMV (ours, fused)"),
    (r"Marlin", "expert MXFP4 GEMM (vLLM MARLIN)"),
    (r"moe_align|moe_sum|count_and_sort_expert|group_topk|kimi_moe_topk", "MoE routing / scatter"),
    (r"kimi_kda_decode|kimi_attn_res", "KDA / attention (fused)"),
    (r"kimi_(rmsnorm|situ|moe_combine)", "act / norm (fused)"),
    (r"kimi_(scatter_rows|copy|causal_mask)", "cache/state plumbing"),
    (r"kda_decode_fusion|MLADecode|attn_res_kernel|flash|fmha|attn_", "KDA / attention (fused)"),
    (r"situ_and_mul|rmsnorm_kernel|rms_norm|_fused_q_kv", "act / norm (fused)"),
    (
        r"one_shot|nccl|AllReduce|Allreduce|allreduce|all_reduce|"
        r"cross_device_reduce|AllGather|all_gather",
        "collective (TP)",
    ),
    (
        r"nvjet|cutlass|cute_|(sgemm|hgemm|gemv|gemm)|nn_|xmma|sm90_|ampere_|"
        r"^gemmk1|splitKreduce|dot_kernel|reduce_1Block",
        "GEMM / GEMV (library)",
    ),
    (r"index_select|is_u32|gather", "index_select / gather"),
    (r"ucopy|copy2d|ccopy|scopy|bcopy|copy_", "copy / cat"),
    (r"cast|convert", "dtype cast"),
    (
        r"affine|mul|add|sub|div|binary|ub[a-z]*_|bfloat|maximum|minimum|neg|"
        r"floor|exp|sigmoid|silu|sqr|sqrt|recip|where|powf|elu|relu|gelu|tanh",
        "elementwise",
    ),
    (r"reduce|sum|max_|min_|softmax|layernorm|argmax|fast_", "reduce / norm"),
    (r"sort|topk", "sort / top-k"),
    (r"fill|memset", "fill"),
]


def _merged(intervals):
    out: list[list] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def kernel_summary(results_dir, steps, window=None, subtract=None, top=25, gap_us=20.0, nsys="nsys"):
    """Exported once to <results_dir>/nsys.sqlite."""
    results_dir = Path(results_dir)
    db = results_dir / "nsys.sqlite"
    if not db.exists():
        rep = next((results_dir / n for n in TRACE_FILES["nsys"] if (results_dir / n).exists()), None)
        if rep is None:
            raise ValueError(f"no armed nsys trace in {results_dir}")
        # A temp file keeps an interrupted export from leaving a db; nsys refuses an existing -o path.
        part = db.with_name("nsys.partial.sqlite")
        part.unlink(missing_ok=True)
        subprocess.run([nsys, "export", "--type", "sqlite", "-o", str(part), str(rep)], check=True)
        part.replace(db)
    with closing(sqlite3.connect(db)) as con:

        def nvtx(name):
            rows = [
                r
                for r in con.execute(
                    "SELECT e.start, e.end FROM NVTX_EVENTS e LEFT JOIN StringIds s ON s.id = e.textId "
                    "WHERE e.text = ? OR s.value = ?",
                    (name, name),
                )
                if r[1] is not None
            ]
            if len(rows) != 1:
                raise ValueError(f"NVTX range {name!r} occurs {len(rows)} times; expected once")
            return rows[0]

        cols = {r[1] for r in con.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
        col = next(c for c in ("shortName", "demangledName", "nameId", "name") if c in cols)

        def tally(win):
            """win None means everything."""
            bound, args = (" WHERE k.start >= ? AND k.end <= ?", win) if win else ("", ())
            kernels: dict[str, list[int]] = {}
            devices: dict[int, list[tuple]] = {}
            mem: dict[str, list[int]] = {}
            for s, e, dev, name in con.execute(
                "SELECT k.start, k.end, k.deviceId, COALESCE(s.value, 'unknown') "
                f"FROM CUPTI_ACTIVITY_KIND_KERNEL k LEFT JOIN StringIds s ON s.id = k.{col}" + bound,
                args,
            ):
                k = kernels.setdefault(name, [0, 0])
                k[0], k[1] = k[0] + 1, k[1] + e - s
                devices.setdefault(dev, []).append((s, e))
            for table, kind in (("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"), ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
                try:
                    rows = con.execute(f"SELECT start, end FROM {table} k" + bound, args).fetchall()
                except sqlite3.OperationalError:
                    continue
                for s, e in rows:
                    m = mem.setdefault(kind, [0, 0])
                    m[0], m[1] = m[0] + 1, m[1] + e - s
            return kernels, devices, mem

        win = nvtx(window) if window else None
        kernels, devices, mem = tally(win)
        if not kernels:
            raise ValueError("no kernel rows in the trace; was it captured with -t cuda?")
        if subtract:
            less_kernels, _, less_mem = tally(nvtx(subtract))
            for totals, less in ((kernels, less_kernels), (mem, less_mem)):
                for name, (c, d) in less.items():
                    if name in totals:
                        totals[name] = [totals[name][0] - c, totals[name][1] - d]
            kernels = {name: v for name, v in kernels.items() if v[0] > 0}
    n = steps
    total_ns = sum(d for _, d in kernels.values())
    families: dict[str, list[int]] = {}
    for name, (c, d) in kernels.items():
        fam = next((f for pat, f in KERNEL_FAMILIES if re.search(pat, name, re.I)), "other")
        f = families.setdefault(fam, [0, 0])
        f[0], f[1] = f[0] + c, f[1] + d
    out = {
        "window": window or "whole capture",
        "subtract": subtract,
        "steps": n,
        "launches_per_step": sum(c for c, _ in kernels.values()) / n,
        "kernel_ms_per_step": total_ns / 1e6 / n,
        "families": [
            {"family": f, "launches_per_step": c / n, "ms_per_step": d / 1e6 / n, "share_pct": 100.0 * d / total_ns}
            for f, (c, d) in sorted(families.items(), key=lambda kv: -kv[1][1])
        ],
        "kernels": [
            {"kernel": name, "launches_per_step": c / n, "ms_per_step": d / 1e6 / n, "us_per_call": d / 1e3 / max(c, 1)}
            for name, (c, d) in sorted(kernels.items(), key=lambda kv: -kv[1][1])[:top]
        ],
        "memops": [{"kind": k, "per_step": c / n, "ms_per_step": d / 1e6 / n} for k, (c, d) in sorted(mem.items())],
        "devices": None,
    }
    if not subtract:
        lo = win[0] if win else min(s for iv in devices.values() for s, _ in iv)
        hi = win[1] if win else max(e for iv in devices.values() for _, e in iv)
        out["span_ms_per_step"] = (hi - lo) / 1e6 / n
        out["devices"] = []
        for dev in sorted(devices):
            merged = _merged(devices[dev])
            union = sum(e - s for s, e in merged)
            gaps = [b[0] - a[1] for a, b in zip(merged, merged[1:]) if b[0] - a[1] >= gap_us * 1000]
            out["devices"].append(
                {
                    "device": dev,
                    "sum_ms_per_step": sum(e - s for s, e in devices[dev]) / 1e6 / n,
                    "union_ms_per_step": union / 1e6 / n,
                    "busy_pct": 100.0 * union / (hi - lo),
                    "gaps_per_step": len(gaps) / n,
                    "idle_ms_per_step": sum(gaps) / 1e6 / n,
                }
            )
    return out


def with_defaults(argv, defaults):
    argv = list(argv or ())
    for flag, value in defaults.items():
        if value is not None and flag not in argv:
            argv += [flag, str(value)]
    return argv


def serving_frontend(endpoint, model, argv=()):
    probe = importlib.import_module(FRONTEND)
    a = probe.frontend_parser().parse_args(with_defaults(argv, {"--base-url": f"{endpoint}/v1", "--model": model}))
    return probe.frontend_probe(a)


def pd_proxy_overhead(endpoint, model, argv=()):
    probe = importlib.import_module(FRONTEND)
    a = probe.pd_proxy_overhead_parser().parse_args(with_defaults(argv, {"--proxy-url": endpoint, "--model": model}))
    return {"rows": probe.pd_proxy_overhead_probe(a)}


def kv_transfer(run_dir):
    """Read as JSON since the probe module needs torch."""
    path = Path(run_dir) / "rows.jsonl"
    with path.open() as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return {"rows_jsonl": str(path), "rows": [{**r["key"], **r["value"]} for r in rows if r["probe"] == "kv_transfer"]}
