"""Native profiler sources: what the engine itself exposes.

  /metrics   the Prometheus text a vLLM server serves, scraped once
  traces     nsys / torch files the launcher armed at start; listed, never run
  kernels    per-kernel GPU time from the armed nsys trace (nsys_kernels.py)
  frontend, pd_host
             the live-server probes, run against the attached endpoint
  kv_transfer
             the NIXL hand-off probe's CSV; it needs its own torchrun ranks,
             so it is read here, never started

Counters on /metrics are cumulative since the server started, so a scrape
after a run covers everything the server did, warmup included.
"""

import csv
import importlib
import re
import sqlite3
import subprocess
from pathlib import Path

from quettabench.engines import launcher

HITS = "vllm:gpu_prefix_cache_hits"
QUERIES = "vllm:gpu_prefix_cache_queries"

# vLLM's speculative-decoding counters, named as the H200 campaign's own
# /metrics scrape recorded them (v4bench_backup_20260928/sweep_results.jsonl,
# spec_metrics: vllm:spec_decode_num_drafts_total, ..._draft_tokens_total,
# ..._accepted_tokens_total). They exist only on a server with spec decoding.
SPEC_COUNTERS = {"accepted_tokens": "vllm:spec_decode_num_accepted_tokens",
                 "draft_tokens": "vllm:spec_decode_num_draft_tokens",
                 "drafts": "vllm:spec_decode_num_drafts"}

# name{labels} value [timestamp]; label values may hold escaped quotes.
SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+\S+)?$')
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')

# The live probes import QuettaSim's `engine` package at the top, so they are
# loaded by name only when their source is asked for; the GPU-free profiler
# must import without QuettaSim.
PROBES = {"frontend": "quettabench.profiling.serving_frontend",
          "pd_host": "quettabench.profiling.pd_host"}

# File names the instrumentation leaves in results/<run>/, by kind. nsys
# writes <out>.nsys-rep (<out>.qdrep before 2021.4); the adapters pass out=nsys.
TRACE_FILES = {
    "nsys": ("nsys.nsys-rep", "nsys.qdrep", "nsys.rep"),
    "torch": ("torch_trace.json",),
}


def parse_prometheus(text):
    """{name: value} for unlabeled samples, {(name, ((label, value), ...)): value}
    for labeled ones, labels sorted. Comments and blank lines are skipped;
    a line that does not parse is skipped too, since one odd exporter line
    should not cost the whole scrape."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = SAMPLE.match(line)
        if not m:
            continue
        name, labels, value = m.groups()
        try:
            value = float(value)
        except ValueError:
            continue
        key = (name, tuple(sorted(LABEL.findall(labels)))) if labels else name
        out[key] = value
    return out


def counter_total(metrics, name):
    """Sum of a counter over every label set; prometheus_client exposes
    counter X as X_total, so both spellings count. None when absent."""
    names = (name, name + "_total")
    vals = [v for k, v in metrics.items() if (k[0] if isinstance(k, tuple) else k) in names]
    return sum(vals) if vals else None


def prefix_cache_hit_rate(metrics):
    """Token-weighted prefix-cache hit rate: hit tokens / queried tokens."""
    hits, queries = counter_total(metrics, HITS), counter_total(metrics, QUERIES)
    if hits is None or not queries:
        return None
    return hits / queries


def scrape_metrics(endpoint, timeout_s=10):
    """GET <endpoint>/metrics and parse it."""
    with launcher.urlopen(f"{endpoint}/metrics", timeout=timeout_s) as resp:
        return parse_prometheus(resp.read().decode())


def spec_decode_counters(endpoint, timeout_s=10):
    """{accepted_tokens, draft_tokens, drafts} from one /metrics scrape,
    each summed over engines; None when a counter is missing (spec decoding
    off) or /metrics does not answer."""
    try:
        metrics = scrape_metrics(endpoint, timeout_s)
    except OSError:
        return None
    counts = {key: counter_total(metrics, name) for key, name in SPEC_COUNTERS.items()}
    return None if None in counts.values() else counts


def metrics_section(metrics):
    """profile.json `metrics`: every sample keyed by its exposition text
    (name or name{k="v",...}) plus the derived prefix_cache_hit_rate."""
    samples = {}
    for key, value in metrics.items():
        if isinstance(key, tuple):
            name, labels = key
            key = name + "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}"
        samples[key] = value
    return {"samples": samples, "prefix_cache_hit_rate": prefix_cache_hit_rate(metrics)}


def collect_armed_traces(results_dir, instrumentation):
    """profile.json `traces`: the files the launch-time instrumentation left
    in results_dir. Only looks; missing means the arming did not produce
    its file (or the server has not exited and flushed it yet)."""
    names = TRACE_FILES.get(instrumentation, ())
    found = [n for n in names if (Path(results_dir) / n).exists()]
    return {"instrumentation": instrumentation, "files": found,
            "missing": bool(names) and not found}


# nsys_kernels.py's family list, unchanged: first match wins, and one list
# covers both engines so a row means the same work whichever engine ran it.
# The kimi_* patterns are QuettaServe kernel symbols. one_shot and the
# underscored all_reduce are listed explicitly: before 2026-09-02 llama's
# one-shot NVLink all-reduce fell through to "reduce / norm".
KERNEL_FAMILIES = [
    (r"kimi_moe_gemv", "expert MXFP4 GEMV (ours, fused)"),
    (r"Marlin", "expert MXFP4 GEMM (vLLM MARLIN)"),
    (r"moe_align|moe_sum|count_and_sort_expert|group_topk|kimi_moe_topk", "MoE routing / scatter"),
    (r"kimi_kda_decode|kimi_attn_res", "KDA / attention (fused)"),
    (r"kimi_(rmsnorm|situ|moe_combine)", "act / norm (fused)"),
    (r"kimi_(scatter_rows|copy|causal_mask)", "cache/state plumbing"),
    (r"kda_decode_fusion|MLADecode|attn_res_kernel|flash|fmha|attn_", "KDA / attention (fused)"),
    (r"situ_and_mul|rmsnorm_kernel|rms_norm|_fused_q_kv", "act / norm (fused)"),
    (r"one_shot|nccl|AllReduce|Allreduce|allreduce|all_reduce|"
     r"cross_device_reduce|AllGather|all_gather", "collective (TP)"),
    (r"nvjet|cutlass|cute_|(sgemm|hgemm|gemv|gemm)|nn_|xmma|sm90_|ampere_|"
     r"^gemmk1|splitKreduce|dot_kernel|reduce_1Block", "GEMM / GEMV (library)"),
    (r"index_select|is_u32|gather", "index_select / gather"),
    (r"ucopy|copy2d|ccopy|scopy|bcopy|copy_", "copy / cat"),
    (r"cast|convert", "dtype cast"),
    (r"affine|mul|add|sub|div|binary|ub[a-z]*_|bfloat|maximum|minimum|neg|"
     r"floor|exp|sigmoid|silu|sqr|sqrt|recip|where|powf|elu|relu|gelu|tanh", "elementwise"),
    (r"reduce|sum|max_|min_|softmax|layernorm|argmax|fast_", "reduce / norm"),
    (r"sort|topk", "sort / top-k"),
    (r"fill|memset", "fill"),
]


def _merged(intervals):
    """(start, end) ns intervals merged where they overlap, in order."""
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def kernel_summary(results_dir, steps, window=None, subtract=None, top=25, gap_us=20.0,
                   nsys="nsys"):
    """profile.json `kernels`: nsys_kernels.py's tables from the run's armed
    nsys trace, exported once to <results_dir>/nsys.sqlite with `nsys export
    --type sqlite`. window is an NVTX range name (default the whole capture);
    subtract is a second range whose totals come off first, how a vLLM window
    drops its prefill; steps is the decode step count left. Every number is
    per step. Each kernel's sum of durations is the work; the union of their
    intervals per device is the wall time it covers; the two differ by the
    overlap, so both are reported."""
    results_dir = Path(results_dir)
    db = results_dir / "nsys.sqlite"
    if not db.exists():
        rep = next((results_dir / n for n in TRACE_FILES["nsys"]
                    if (results_dir / n).exists()), None)
        if rep is None:
            raise ValueError(f"no armed nsys trace in {results_dir}")
        subprocess.run([nsys, "export", "--type", "sqlite", "-o", str(db), str(rep)], check=True)
    con = sqlite3.connect(db)

    def nvtx(name):
        rows = [r for r in con.execute(
            "SELECT e.start, e.end FROM NVTX_EVENTS e LEFT JOIN StringIds s ON s.id = e.textId "
            "WHERE e.text = ? OR s.value = ?", (name, name)) if r[1] is not None]
        if len(rows) != 1:
            raise ValueError(f"NVTX range {name!r} occurs {len(rows)} times; expected once")
        return rows[0]

    cols = {r[1] for r in con.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    col = next(c for c in ("shortName", "demangledName", "nameId", "name") if c in cols)

    def tally(win):
        """{kernel: [launches, ns]}, {device: [(start, end)]}, {memop: [count, ns]}
        for what lies fully inside win (None: everything)."""
        bound, args = (" WHERE k.start >= ? AND k.end <= ?", win) if win else ("", ())
        kernels, devices, mem = {}, {}, {}
        for s, e, dev, name in con.execute(
                "SELECT k.start, k.end, k.deviceId, COALESCE(s.value, 'unknown') "
                f"FROM CUPTI_ACTIVITY_KIND_KERNEL k LEFT JOIN StringIds s ON s.id = k.{col}"
                + bound, args):
            k = kernels.setdefault(name, [0, 0])
            k[0], k[1] = k[0] + 1, k[1] + e - s
            devices.setdefault(dev, []).append((s, e))
        for table, kind in (("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"),
                            ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
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
    families = {}
    for name, (c, d) in kernels.items():
        fam = next((f for pat, f in KERNEL_FAMILIES if re.search(pat, name, re.I)), "other")
        f = families.setdefault(fam, [0, 0])
        f[0], f[1] = f[0] + c, f[1] + d
    out = {
        "window": window or "whole capture", "subtract": subtract, "steps": n,
        "launches_per_step": sum(c for c, _ in kernels.values()) / n,
        "kernel_ms_per_step": total_ns / 1e6 / n,
        "families": [{"family": f, "launches_per_step": c / n, "ms_per_step": d / 1e6 / n,
                      "share_pct": 100.0 * d / total_ns}
                     for f, (c, d) in sorted(families.items(), key=lambda kv: -kv[1][1])],
        "kernels": [{"kernel": name, "launches_per_step": c / n, "ms_per_step": d / 1e6 / n,
                     "us_per_call": d / 1e3 / max(c, 1)}
                    for name, (c, d) in sorted(kernels.items(), key=lambda kv: -kv[1][1])[:top]],
        "memops": [{"kind": k, "per_step": c / n, "ms_per_step": d / 1e6 / n}
                   for k, (c, d) in sorted(mem.items())],
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
            gaps = [b[0] - a[1] for a, b in zip(merged, merged[1:])
                    if b[0] - a[1] >= gap_us * 1000]
            out["devices"].append({
                "device": dev, "sum_ms_per_step": sum(e - s for s, e in devices[dev]) / 1e6 / n,
                "union_ms_per_step": union / 1e6 / n, "busy_pct": 100.0 * union / (hi - lo),
                "gaps_per_step": len(gaps) / n, "idle_ms_per_step": sum(gaps) / 1e6 / n})
    return out


def with_defaults(argv, defaults):
    """argv plus `flag value` for each default flag argv does not already give."""
    argv = list(argv or ())
    for flag, value in defaults.items():
        if value is not None and flag not in argv:
            argv += [flag, str(value)]
    return argv


def serving_frontend(endpoint, model, argv=()):
    """profile.json `frontend`: serving_frontend.probe() against endpoint/v1;
    argv carries the probe's own flags (--gpu-label, --model-yaml, ...)."""
    probe = importlib.import_module(PROBES["frontend"])
    a = probe.parser().parse_args(with_defaults(argv, {"--base-url": f"{endpoint}/v1",
                                                       "--model": model}))
    return probe.probe(a)


def pd_host(endpoint, model, argv=()):
    """profile.json `pd_host`: pd_host.probe()'s rows, endpoint as the proxy."""
    probe = importlib.import_module(PROBES["pd_host"])
    a = probe.parser().parse_args(with_defaults(argv, {"--proxy-url": endpoint,
                                                       "--model": model}))
    return {"rows": probe.probe(a)}


def kv_transfer(csv_path):
    """profile.json `kv_transfer`: the rows kv_transfer's torchrun run wrote
    (csv strings, as written). Read directly: the probe module needs torch."""
    with Path(csv_path).open() as fh:
        return {"csv": str(csv_path), "rows": list(csv.DictReader(fh))}
