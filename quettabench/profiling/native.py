"""Native profiler sources: what the engine itself exposes.

  /metrics   the Prometheus text a vLLM server serves, scraped once
  traces     nsys / torch files the launcher armed at start; listed, never run

Counters on /metrics are cumulative since the server started, so a scrape
after a run covers everything the server did, warmup included.
"""

import re
import urllib.request
from pathlib import Path

HITS = "vllm:gpu_prefix_cache_hits"
QUERIES = "vllm:gpu_prefix_cache_queries"

# name{labels} value [timestamp]; label values may hold escaped quotes.
SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+\S+)?$')
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')

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
    with urllib.request.urlopen(f"{endpoint}/metrics", timeout=timeout_s) as resp:
        return parse_prometheus(resp.read().decode())


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
