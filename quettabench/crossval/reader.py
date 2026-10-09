"""One Side per input crossval can compare: a results/<run>/ dir, an old
baselines/*.json, or an engine bench log.

A Side is rows (one per (ctx, bs, tp) cell) plus a header of the facts the
withhold rules need. Every row has the same keys:

  ctx, bs, tp, group, ms_per_step, ms_at_eval, eval_pos, r2, kv, mode, measured

group is the bench group the old table printed (decode_loop, batch_bench,
decode_batch, ...; "slope" for a fitted cell), mode is "eager" or "graph",
measured is True for a number the engine timed itself and False for a slope
fitted over step points. notes are the parser's messages, printed above the
table the way table.py printed them while parsing.
"""

import calendar
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from quettabench.config import xval_config

GROUP_PREF = ("decode_batch_paged", "decode_batch", "decode_graph", "decode")
POINT_RE = re.compile(
    r"(?P<group>decode_batch_paged|decode_batch|decode_graph_tp\d+|decode_graph|decode_tp\d+|decode)"
    r"/(?P<param>\d+(?:x\d+)?)"
    r"\s*\n?\s*time:\s*\[[\d.]+ \w+ (?P<mid>[\d.]+) (?P<unit>ms|s|us|µs|ns)"
)
MS = {"ns": 1e-6, "us": 1e-3, "µs": 1e-3, "ms": 1.0, "s": 1e3}
UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass
class Side:
    rows: list
    header: dict
    notes: list = field(default_factory=list)


def header(**known):
    """A header with every key present; unknown facts are None."""
    keys = ("gpu", "engine", "version", "model", "dtype", "grid", "graph_mode", "method",
            "kv_dtype", "quant", "serving_style", "nccl_algo", "nccl_proto", "prompt_mode",
            "eval_pos", "step_points", "timestamp", "bench_sha", "engine_sha",
            "workload_hash", "comm_bound_bs")
    unknown = set(known) - set(keys)
    if unknown:
        raise KeyError(f"not a Side header key: {sorted(unknown)}")
    return {k: known.get(k) for k in keys}


def row(ctx, bs, tp, ms_per_step, *, group, kv, measured, ms_at_eval=None, eval_pos=None,
        r2=None, mode="graph"):
    return {"ctx": ctx, "bs": bs, "tp": tp, "group": group, "ms_per_step": ms_per_step,
            "ms_at_eval": ms_at_eval, "eval_pos": eval_pos, "r2": r2, "kv": kv, "mode": mode,
            "measured": measured}


def age_days(side, now):
    """Days from the side's timestamp (UTC, Z form) to now (epoch seconds)."""
    return (now - calendar.timegm(time.strptime(side.header["timestamp"], UTC_FORMAT))) / 86400


# --- engine bench logs: table.py's parsers, moved unchanged except that
# their messages are returned as notes instead of printed.

def _loop_points(text, notes):
    """(ctx, bs, tp) -> (ms, group, kv) from the engine's LOOP lines.
    mode=eager rows get their own group so graph and eager never share a column."""
    pts = {}
    gdns = set()
    for line in text.splitlines():
        if not line.startswith("LOOP "):
            continue
        d = dict(re.findall(r"(\w+)=([\w.]+)", line))
        key = (int(d["ctx"]), int(d["bs"]), int(d.get("tp", "1")))
        group = "decode_loop_eager" if d.get("mode") == "eager" else "decode_loop"
        pts[key] = (float(d["ms_per_step"]), group, d.get("kv"))
        if d.get("gdn"):
            gdns.add(d["gdn"])
    if gdns:
        notes.append(f"engine gdn kernel(s): {', '.join(sorted(gdns))}")
    return pts


def _batch_bench_points(text, notes):
    """(ctx, bs, tp) -> (ms, "batch_bench", kv) from the deepseek engine's
    stock bench summary: the median of the graph-replay decode steps, the
    same marginal quantity as the baseline slope. kv is "?" (unstated)."""
    pts = {}
    truncated = unparsed = 0
    for line in text.splitlines():
        if not line.startswith("[batch_bench] "):
            continue
        d = dict(re.findall(r"(\w+)=([\w.]+)", line))
        ms = re.search(r"median ([\d.]+) ms/step", line)
        if not ms or "layers" not in d:
            unparsed += 1
            continue
        if d["layers"] != "all":
            truncated += 1
            continue
        key = (int(d["prompt"]), int(d["bs"]), int(d.get("world", "1")))
        pts[key] = (float(ms.group(1)), "batch_bench", d.get("kv", "?"))
    if truncated:
        notes.append(f"ignored {truncated} truncated-model batch_bench line(s) (layers != all)")
    if unparsed:
        notes.append(f"ignored {unparsed} unparsed batch_bench line(s)")
    return pts


def bench_log_points(path):
    """((ctx, bs, tp) -> (ms, group, kv), method, notes) from a bench log.

    LOOP lines and batch_bench summaries both report steady-state marginal
    decode (method "loop"), and a LOOP line supersedes the summary on the
    same cell. The criterion decode groups synchronize inside every step, a
    different quantity from the slope (method "criterion"), so ratios
    against those are withheld. Tensor-parallel criterion groups are named
    decode_tp{N} / decode_graph_tp{N}; bare names are tp 1.
    """
    text = Path(path).read_text()
    notes = []
    bench = _batch_bench_points(text, notes)
    loop = _loop_points(text, notes)
    if loop or bench:
        return {**bench, **loop}, "loop", notes
    pref = {name: i for i, name in enumerate(GROUP_PREF)}
    by_key = {}
    for m in POINT_RE.finditer(text):
        p = m.group("param")
        ctx, bs = (int(v) for v in p.split("x")) if "x" in p else (int(p), 1)
        group = m.group("group")
        base, _, tp_suffix = group.partition("_tp")
        tp = int(tp_suffix) if tp_suffix else 1
        ms = float(m.group("mid")) * MS[m.group("unit")]
        prev = by_key.get((ctx, bs, tp))
        # Ties go to the later entry so an appended log supersedes stale runs.
        if prev is None or pref[base] <= pref.get(prev[1].partition("_tp")[0], 99):
            by_key[(ctx, bs, tp)] = (ms, group, None)
    return by_key, "criterion", notes


def measured_row(key, ms, group, kv):
    """A row the engine timed itself; eager rows carry their own group."""
    ctx, bs, tp = key
    return row(ctx, bs, tp, ms, group=group, kv=kv, measured=True,
               mode="eager" if group == "decode_loop_eager" else "graph")


# Config path -> header key for the engine facts a bench log does not state.
ENGINE_FACTS = {"engine.quant": "quant", "engine.serving_style": "serving_style",
                "hardware.collective.NCCL_ALGO": "nccl_algo",
                "hardware.collective.NCCL_PROTO": "nccl_proto"}


def engine_facts(values):
    """Header facts from flat {Config path: value} (env_map.read_env's shape)."""
    return {key: values[path] for path, key in ENGINE_FACTS.items() if path in values}


def read_bench_log(path, engine_facts=None):
    """The engine side from a bench log. The log states no NCCL, quant or
    serving topology, so those come from engine_facts (the Config values the
    old table.py read from the environment), None when not given."""
    points, method, notes = bench_log_points(path)
    rows = [measured_row(key, *point) for key, point in points.items()]
    facts = engine_facts or {}
    return Side(rows, header(engine="quettaserve", method=method, **facts), notes)


# --- old baselines/*.json, as crossval/cache.py wrote them.

def comm_bound_bs(model):
    """comm_bound_bs of the workload whose crate key or record name is model,
    from the merged workloads. Absent means never comm-bound (tp1 has no TP
    allreduce); there is no default."""
    for name, wl in xval_config.workloads().items():
        if (name == model or wl.get("name") == model) and "comm_bound_bs" in wl:
            return int(wl["comm_bound_bs"])
    return None


def read_baseline(path):
    """The vLLM side from a cache.py baseline json. A point without
    ms_at_eval is a pre-normalization raw slope and keeps ms_at_eval None."""
    b = json.loads(Path(path).read_text())
    rows = [row(p["ctx"], p["bs"], p.get("tp", 1), p["ms_per_step"], group="slope",
                kv=p.get("kv", b.get("dtype", "?")), measured=False,
                ms_at_eval=p.get("ms_at_eval"), eval_pos=p.get("eval_pos"), r2=p.get("r2"),
                mode="eager" if b["mode"] == "NOGRAPH" else "graph")
            for p in b["points"]]
    return Side(rows, header(
        gpu=b["gpu"], engine="vllm", version=b["vllm"], model=b.get("model", "?"),
        dtype=b.get("dtype", "?"), grid=b.get("grid", "?"), graph_mode=b["mode"],
        method=b.get("method", "slope"), kv_dtype=b.get("kv_dtype"),
        quant=b.get("quant", b.get("dtype", "?")),
        serving_style=b.get("serving_style", "aggregated"),
        nccl_algo=b.get("nccl_algo"), nccl_proto=b.get("nccl_proto"),
        prompt_mode=b.get("prompt_mode"), eval_pos=b.get("eval_pos"),
        step_points=b.get("step_points"), timestamp=b["recorded_utc"],
        bench_sha=b.get("bench_sha"), engine_sha=b.get("engine_sha"),
        comm_bound_bs=comm_bound_bs(b.get("model", ""))))


# --- results/<run>/ dirs, as metrics/results.py writes them.

def _utc(iso):
    """run.json's isoformat time as the baseline's second-resolution Z form."""
    return datetime.fromisoformat(iso).strftime(UTC_FORMAT) if iso else None


def _fitted_row(step, kv, mode):
    """A step-section cell, rounded as the grid RESULT line printed it, so a
    run dir and the baseline cache.py packed from its log read the same."""
    ms_eval = step.get("ms_at_eval")
    return row(step["ctx"], step["bs"], step["tp"], round(step["ms_per_step"], 3),
               group="slope", kv=kv, measured=False,
               ms_at_eval=None if ms_eval is None else round(ms_eval, 3),
               eval_pos=step.get("eval_pos"), r2=round(step["r2"], 5), mode=mode)


def read_rows(results_dir):
    """Every requests.jsonl row of a run dir, in file order."""
    path = Path(results_dir) / "requests.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_run(results_dir):
    """A Side from run.json + requests.jsonl.

    Fitted cells come from run.json's steps section; measured cells (the
    step section passes a bench number through as measured) take their
    group, kv and method from the BenchDriver rows in requests.jsonl, and
    their notes from the run's bench.log when it is there. A grid run's
    header facts are its META record (summary.grid.meta), the same values
    the old baseline recorded.
    """
    d = Path(results_dir)
    run = json.loads((d / "run.json").read_text())
    prov, cfg = run["provenance"], run["config"]
    summary = run.get("summary") or {}
    grid = summary.get("grid") or {}
    meta = grid.get("meta") or {}
    engine = cfg["engine"]
    mode = meta.get("mode") or ("FULL" if engine.get("cudagraph", True) else "NOGRAPH")
    row_mode = "eager" if mode == "NOGRAPH" else "graph"
    kv = grid.get("kv") or prov.get("kv_dtype")

    bench = {}
    for r in read_rows(d):
        if r.get("measured_ms_per_step") is not None:
            c = r["cell"]
            bench[(c["ctx"], c["bs"], c["tp"])] = r
    rows, methods = [], set()
    for step in run.get("steps") or []:
        if step.get("measured"):
            r = bench[(step["ctx"], step["bs"], step["tp"])]
            methods.add(r.get("method", "loop"))
            rows.append(measured_row((step["ctx"], step["bs"], step["tp"]),
                                     step["ms_per_step"], r.get("group"), r.get("kv")))
        else:
            methods.add("slope")
            rows.append(_fitted_row(step, kv, row_mode))
    notes = bench_log_points(d / "bench.log")[2] if (d / "bench.log").exists() else []

    workload = (grid.get("trace") or {}).get("workload_hash") \
        or (summary.get("workload") or {}).get("expected")
    return Side(rows, header(
        gpu=meta.get("gpu") or prov.get("gpu"), engine=prov.get("engine"),
        version=meta.get("vllm") or prov.get("engine_version"),
        model=meta.get("model") or engine.get("model_name") or "?",
        dtype=meta.get("dtype") or engine.get("dtype"), grid=meta.get("grid") or "?",
        graph_mode=mode, method="+".join(sorted(methods)) or None, kv_dtype=kv,
        quant=meta.get("quant") or prov.get("quant"),
        serving_style=prov.get("serving_style"),
        nccl_algo=meta.get("nccl_algo") or prov.get("nccl_algo"),
        nccl_proto=meta.get("nccl_proto") or prov.get("nccl_proto"),
        prompt_mode=meta.get("prompt_mode"), eval_pos=meta.get("eval_pos"),
        step_points=meta.get("step_points"), timestamp=_utc(prov.get("finished")),
        bench_sha=prov.get("bench_sha"), engine_sha=prov.get("engine_sha"),
        workload_hash=workload, comm_bound_bs=(cfg.get("crossval") or {}).get("comm_bound_bs")),
        notes)
