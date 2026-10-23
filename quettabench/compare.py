"""Pair two measured runs cell by cell: a ratio or a withhold per row."""

import argparse
import calendar
import dataclasses
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from quettabench.config import env_map
from quettabench.config.schema import serving_style_compatible

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
COMPARE_DIR = REPO / "results" / "compare"
ENV = ("QBENCH_QUANT", "QBENCH_SERVING_STYLE", "NCCL_ALGO", "NCCL_PROTO", "QBENCH_ALLOW_METHOD_MISMATCH")


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
    keys = (
        "gpu",
        "engine",
        "version",
        "model",
        "dtype",
        "grid",
        "graph_mode",
        "method",
        "kv_cache_dtype",
        "quant",
        "serving_style",
        "nccl_algo",
        "nccl_proto",
        "prompt_mode",
        "eval_pos",
        "step_points",
        "timestamp",
        "bench_sha",
        "engine_sha",
        "workload_hash",
        "comm_bound_bs",
        "not_comparable",
        "workload_mix",
    )
    unknown = set(known) - set(keys)
    if unknown:
        raise KeyError(f"not a Side header key: {sorted(unknown)}")
    return {k: known.get(k) for k in keys}


def row(ctx, bs, tp, ms_per_step, *, group, kv, measured, ms_at_eval=None, eval_pos=None, r2=None, mode="graph"):
    return {
        "ctx": ctx,
        "bs": bs,
        "tp": tp,
        "group": group,
        "ms_per_step": ms_per_step,
        "ms_at_eval": ms_at_eval,
        "eval_pos": eval_pos,
        "r2": r2,
        "kv": kv,
        "mode": mode,
        "measured": measured,
    }


def age_days(side, now):
    return (now - calendar.timegm(time.strptime(side.header["timestamp"], UTC_FORMAT))) / 86400


def _loop_points(text, notes):
    """From LOOP lines; eager rows get their own group."""
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
    """kv is "?"; ms is the median graph-replay step."""
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
    """Criterion groups sync every step, so their ratios are withheld."""
    text = Path(path).read_text()
    notes: list[str] = []
    bench = _batch_bench_points(text, notes)
    loop = _loop_points(text, notes)
    if loop or bench:
        return {**bench, **loop}, "loop", notes
    pref = {name: i for i, name in enumerate(GROUP_PREF)}
    by_key: dict[tuple[int, int, int], tuple] = {}
    for m in POINT_RE.finditer(text):
        p = m.group("param")
        ctx, bs = (int(v) for v in p.split("x")) if "x" in p else (int(p), 1)
        group = m.group("group")
        base, _, tp_suffix = group.partition("_tp")
        tp = int(tp_suffix) if tp_suffix else 1
        ms = float(m.group("mid")) * MS[m.group("unit")]
        prev = by_key.get((ctx, bs, tp))
        # Later entry wins: an appended log supersedes stale runs.
        if prev is None or pref[base] <= pref.get(prev[1].partition("_tp")[0], 99):
            by_key[(ctx, bs, tp)] = (ms, group, None)
    return by_key, "criterion", notes


def measured_row(key, ms, group, kv):
    ctx, bs, tp = key
    return row(
        ctx, bs, tp, ms, group=group, kv=kv, measured=True, mode="eager" if group == "decode_loop_eager" else "graph"
    )


ENGINE_FACTS = {
    "engine.quant": "quant",
    "engine.serving_style": "serving_style",
    "hardware.collective.NCCL_ALGO": "nccl_algo",
    "hardware.collective.NCCL_PROTO": "nccl_proto",
}


def engine_facts(values):
    return {key: values[path] for path, key in ENGINE_FACTS.items() if path in values}


def read_bench_log(path, engine_facts=None):
    """NCCL, quant and style come from engine_facts, else None."""
    points, method, notes = bench_log_points(path)
    rows = [measured_row(key, *point) for key, point in points.items()]
    facts = engine_facts or {}
    return Side(rows, header(engine="quettaserve", method=method, **facts), notes)


def comm_bound_bs(model):
    """None means never comm-bound."""
    for run in sorted((CONFIGS / "runs").glob("*.yaml")):
        raw = yaml.safe_load(run.read_text())
        spec = yaml.safe_load((CONFIGS / "models" / f"{raw['model']}.yaml").read_text())
        bs = (spec.get("compare") or {}).get("comm_bound_bs")
        if (run.stem == model or spec["name"] == model) and bs is not None:
            return int(bs)
    return None


def read_baseline(path):
    """ms stays None without ms_at_eval."""
    b = json.loads(Path(path).read_text())
    rows = [
        row(
            p["ctx"],
            p["bs"],
            p.get("tp", 1),
            p["ms_per_step"],
            group="slope",
            kv=p.get("kv", b.get("dtype", "?")),
            measured=False,
            ms_at_eval=p.get("ms_at_eval"),
            eval_pos=p.get("eval_pos"),
            r2=p.get("r2"),
            mode="eager" if b["mode"] == "NOGRAPH" else "graph",
        )
        for p in b["points"]
    ]
    return Side(
        rows,
        header(
            gpu=b["gpu"],
            engine="vllm",
            version=b["vllm"],
            model=b.get("model", "?"),
            dtype=b.get("dtype", "?"),
            grid=b.get("grid", "?"),
            graph_mode=b["mode"],
            method=b.get("method", "slope"),
            kv_cache_dtype=b.get("kv_cache_dtype"),
            quant=b.get("quant", b.get("dtype", "?")),
            serving_style=b.get("serving_style", "aggregated"),
            nccl_algo=b.get("nccl_algo"),
            nccl_proto=b.get("nccl_proto"),
            prompt_mode=b.get("prompt_mode"),
            eval_pos=b.get("eval_pos"),
            step_points=b.get("step_points"),
            timestamp=b["recorded_utc"],
            bench_sha=b.get("bench_sha"),
            engine_sha=b.get("engine_sha"),
            comm_bound_bs=comm_bound_bs(b.get("model", "")),
        ),
    )


def _utc(iso):
    return datetime.fromisoformat(iso).strftime(UTC_FORMAT) if iso else None


def _fitted_row(step, kv, mode):
    """Rounded as the grid RESULT line prints it."""
    ms_eval = step.get("ms_at_eval")
    return row(
        step["ctx"],
        step["bs"],
        step["tp"],
        round(step["ms_per_step"], 3),
        group="slope",
        kv=kv,
        measured=False,
        ms_at_eval=None if ms_eval is None else round(ms_eval, 3),
        eval_pos=step.get("eval_pos"),
        r2=round(step["r2"], 5),
        mode=mode,
    )


def read_rows(results_dir):
    path = Path(results_dir) / "requests.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def not_comparable(run) -> str | None:
    """Keep a run out of comparisons: a status other than ok, or a file the manifest does not list."""
    validity = (run.get("summary") or {}).get("validity") or {}
    if validity.get("latency_valid") is False:
        return f"latency not valid: {'; '.join(validity.get('reasons') or [])}"
    status = run.get("run_status") or {}
    if status.get("status", "ok") != "ok":
        return f"run status {status['status']}: {status.get('reason')}"
    manifest = ((run.get("summary") or {}).get("trace") or {}).get("manifest") or {}
    if manifest.get("listed") is False:
        return f"{manifest.get('file')} is not listed in data/MANIFEST.json"
    if manifest.get("comparable") is False:
        return f"{manifest.get('file')} is not the file data/MANIFEST.json lists"
    return None


# Environment names that change what a server computes or how fast: the launch's own and the libraries under it.
SERVER_ENV = (*env_map.SERVER_ENV_PREFIXES, "NVSHMEM_", "UCX_", "TRITON_", "FLASHINFER_", "PYTHONPATH", "LD_PRELOAD")
# What observe.log_settings reads from a startup log.
SERVER_LOG_KEYS = (
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "data_parallel_size",
    "expert_parallel",
    "chunked_prefill",
    "prefix_caching",
    "speculative_config",
    "cudagraph_capture_sizes",
    "kv_cache_tokens",
    "max_concurrency",
)


def server_facts(run) -> dict:
    """The server facts server_parity compares; --port is dropped so runs on different ports still match."""
    prov = run.get("provenance") or {}
    seen = prov.get("server_observed") or {}
    proc, log = seen.get("process") or {}, seen.get("log") or {}
    argv = list(proc.get("argv") or [])
    if "--port" in argv:
        del argv[argv.index("--port") : argv.index("--port") + 2]
    env = proc.get("env")
    return {
        "argv": argv or None,
        **{f"env {k}": v for k, v in (env or {}).items() if k.startswith(SERVER_ENV)},
        "env": "observed" if env is not None else None,
        "version": seen.get("version"),
        "weights_revision": prov.get("weights_revision"),
        **{k: log.get(k) for k in SERVER_LOG_KEYS},
    }


def server_parity(a, b) -> list[str]:
    """Name each server difference that blocks ranking two runs; a fact one run did not observe counts as a difference."""
    unseen = [
        side for side, run in (("ours", a), ("theirs", b)) if not (run.get("provenance") or {}).get("server_observed")
    ]
    if unseen:
        return [f"server not observed in {' and '.join(unseen)} (a run made before server_observed)"]
    fa, fb = server_facts(a), server_facts(b)
    out = []
    for key in sorted(set(fa) | set(fb)):
        x, y = fa.get(key), fb.get(key)
        if x is None or y is None:
            out.append(f"{key}: not observed in {'both runs' if x is None and y is None else 'one run'}")
        elif x != y:
            out.append(f"{key}: {x!r} vs {y!r}")
    return out


# Two windows asked the same of their servers when every p50 and mean here is within MIX_TOLERANCE.
MIX_FIELDS = ("input_tokens", "output_tokens", "uncached_prefill_tokens", "tool_wait_s", "session_turns")
# Run-to-run mix spread seen on repeated runs stays under this.
MIX_TOLERANCE = 0.05
# A difference below a field's floor is noise however large in proportion: with recorded waits off, the realized wait
# is the client's sub-millisecond turnaround, and 1.6 against 2.0 ms is 21% of nothing.
MIX_FLOORS = {"tool_wait_s": 0.05}


def mix_diff(a, b) -> list[str]:
    """Name the mix fields where two windows differ past MIX_TOLERANCE, so every ratio is withheld."""
    if not a and not b:
        return []
    out = []
    for name in MIX_FIELDS:
        x, y = (a or {}).get(name), (b or {}).get(name)
        if x is None or y is None:
            if x is not None or y is not None:
                out.append(f"{name}: in one run only")
            continue
        for stat in ("p50", "mean"):
            bar = max(MIX_TOLERANCE * max(abs(x[stat]), abs(y[stat])), MIX_FLOORS.get(name, 0.0))
            if abs(x[stat] - y[stat]) > bar:
                out.append(f"{name} {stat}: {x[stat]:g} vs {y[stat]:g}")
    return out


def read_run(results_dir):
    """Cells take group, kv and method from the BenchDriver rows."""
    d = Path(results_dir)
    run = json.loads((d / "run.json").read_text())
    prov, cfg = run["provenance"], run["config"]
    summary = run.get("summary") or {}
    grid = summary.get("grid") or {}
    meta = grid.get("meta") or {}
    engine = cfg["engine"]
    mode = meta.get("mode") or ("FULL" if engine.get("cudagraph", True) else "NOGRAPH")
    row_mode = "eager" if mode == "NOGRAPH" else "graph"
    kv = grid.get("kv") or prov.get("kv_cache_dtype")

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
            rows.append(
                measured_row((step["ctx"], step["bs"], step["tp"]), step["ms_per_step"], r.get("group"), r.get("kv"))
            )
        else:
            methods.add("slope")
            rows.append(_fitted_row(step, kv, row_mode))
    notes = bench_log_points(d / "bench.log")[2] if (d / "bench.log").exists() else []

    # A serve run's plan_hash is the one runs share; a run.json without one falls back to expected.
    served = summary.get("workload") or {}
    workload = (grid.get("trace") or {}).get("workload_hash") or served.get("plan_hash") or served.get("expected")
    return Side(
        rows,
        header(
            gpu=meta.get("gpu") or prov.get("gpu"),
            engine=prov.get("engine"),
            version=meta.get("vllm") or prov.get("engine_version"),
            model=meta.get("model") or engine.get("model_name") or "?",
            dtype=meta.get("dtype") or engine.get("dtype"),
            grid=meta.get("grid") or "?",
            graph_mode=mode,
            method="+".join(sorted(methods)) or None,
            kv_cache_dtype=kv,
            quant=meta.get("quant") or prov.get("quant"),
            serving_style=prov.get("serving_style"),
            nccl_algo=meta.get("nccl_algo") or prov.get("nccl_algo"),
            nccl_proto=meta.get("nccl_proto") or prov.get("nccl_proto"),
            prompt_mode=meta.get("prompt_mode"),
            eval_pos=meta.get("eval_pos"),
            step_points=meta.get("step_points"),
            timestamp=_utc(prov.get("finished")),
            bench_sha=prov.get("bench_sha"),
            engine_sha=prov.get("engine_sha"),
            workload_hash=workload,
            comm_bound_bs=(cfg.get("compare") or {}).get("comm_bound_bs"),
            not_comparable=not_comparable(run),
            workload_mix=(summary.get("steady") or {}).get("workload_mix"),
        ),
        notes,
    )


# Near-tie gap: one bf16 step for logits in [16, 32).
NEAR_TIE_GAP = 0.125
AGREE_MIN = 4  # leading tokens that count as agreeing


def returned(rows, name):
    out: dict = {}
    seen: dict[tuple, int] = {}
    for i, r in enumerate(rows):
        if r.get(name) is None:
            continue
        cell = r.get("cell")
        if cell is not None:
            group = (cell["ctx"], cell["bs"], cell["tp"])
            n = seen[group] = seen.get(group, -1) + 1
            key = f"{group[0]}x{group[1]}" + (f"@tp{group[2]}" if group[2] != 1 else "") + f" #{n}"
        else:
            key = f"req {r.get('request_index', i)}"
        out[key] = r[name]
    return out


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def greedy(ours_rows, theirs_rows):
    a, b = returned(ours_rows, "token_ids"), returned(theirs_rows, "token_ids")
    rows = []
    for key in sorted(set(a) & set(b)):
        prefix, of = common_prefix(a[key], b[key]), min(len(a[key]), len(b[key]))
        rows.append(
            {
                "kind": "greedy",
                "request": key,
                "prefix": prefix,
                "of": of,
                "match_pct": 100.0 * prefix / of if of else 0.0,
            }
        )
    if rows:
        prefixes = [r["prefix"] for r in rows]
        rows.append(
            {
                "kind": "greedy_summary",
                "prompts": len(prefixes),
                "mean_prefix": statistics.mean(prefixes),
                "median_prefix": statistics.median(prefixes),
                "agree4plus": sum(1 for p in prefixes if p >= AGREE_MIN),
            }
        )
    return rows


def logit(ours_rows, theirs_rows):
    a, b = returned(ours_rows, "logits"), returned(theirs_rows, "logits")
    keys = sorted(set(a) & set(b))
    if not keys:
        return []
    rows, total, agree, near_tie, deltas = [], 0, 0, 0, []
    for key in keys:
        ours, ref = a[key], b[key]
        for pos in range(min(len(ours["argmax"]), len(ref["argmax"]))):
            total += 1
            if ours["argmax"][pos] == ref["argmax"][pos]:
                agree += 1
            elif ref["top2_gap"][pos] <= NEAR_TIE_GAP:
                near_tie += 1
            else:
                rows.append(
                    {
                        "kind": "logit_disagree",
                        "request": key,
                        "pos": pos,
                        "reference_gap": ref["top2_gap"][pos],
                        "reference": ref["argmax"][pos],
                        "ours": ours["argmax"][pos],
                    }
                )
            if pos < len(ours["logprob"]) and pos < len(ref["logprob"]):
                deltas.append(abs(ours["logprob"][pos] - ref["logprob"][pos]))
    hard = len(rows)
    deltas.sort()
    rows.append(
        {
            "kind": "logit_summary",
            "positions": total,
            "argmax_agree": agree,
            "near_tie_flips": near_tie,
            "hard_disagreements": hard,
            "delta_mean": sum(deltas) / len(deltas) if deltas else None,
            "delta_p95": deltas[int(0.95 * len(deltas))] if deltas else None,
            "delta_max": deltas[-1] if deltas else None,
            "verdict": "SKIP" if not total else "PASS" if hard == 0 else "FAIL",
        }
    )
    return rows


def flips(rows, anchor_rows):
    """(positions, argmax flips) against anchor_rows."""
    summary = logit(rows, anchor_rows)
    if not summary:
        return 0, 0
    return summary[-1]["positions"], summary[-1]["positions"] - summary[-1]["argmax_agree"]


def band(ours_rows, anchor_rows, floor_runs):
    """Ours' flips at most the worst floor run's; withheld if positions differ."""
    positions, ours = flips(ours_rows, anchor_rows)
    floors = [flips(rows, anchor_rows) for rows in floor_runs]
    comparable = positions > 0 and all(p == positions for p, _ in floors)
    worst = max(f for _, f in floors)
    return {
        "kind": "band",
        "positions": positions,
        "flips": ours,
        "floor_flips": [f for _, f in floors],
        "floor": worst,
        "verdict": "WITHHELD" if not comparable else "PASS" if ours <= worst else "FAIL",
    }


def compare_runs(ours_rows, theirs_rows):
    return greedy(ours_rows, theirs_rows) + logit(ours_rows, theirs_rows)


def render_agreement(rows):
    lines = []
    for r in rows:
        kind = r["kind"]
        if kind == "greedy":
            lines.append(f"AGREE {r['prefix']:2d}/{r['of']} tokens ({r['match_pct']:5.1f}%) | {r['request']}")
        elif kind == "greedy_summary":
            lines.append(
                f"SUMMARY prompts={r['prompts']} mean_prefix={r['mean_prefix']:.1f} "
                f"median_prefix={r['median_prefix']} "
                f"agree4plus={r['agree4plus']}/{r['prompts']}"
            )
        elif kind == "logit_disagree":
            lines.append(
                f"DISAGREE {r['request']} pos={r['pos']} "
                f"reference_gap={r['reference_gap']:.4f} "
                f"reference={r['reference']} ours={r['ours']}"
            )
        elif kind == "logit_summary":
            if r["positions"]:
                lines.append(
                    f"SUMMARY positions={r['positions']} "
                    f"argmax_agree={r['argmax_agree']} "
                    f"({100.0 * r['argmax_agree'] / r['positions']:.2f}%) "
                    f"near_tie_flips={r['near_tie_flips']} "
                    f"hard_disagreements={r['hard_disagreements']}"
                )
            if r["delta_max"] is not None:
                lines.append(
                    f"SUMMARY true_token_logprob_delta mean={r['delta_mean']:.5f} "
                    f"p95={r['delta_p95']:.5f} max={r['delta_max']:.5f}"
                )
            lines.append(f"FIDELITY verdict={r['verdict']} hard_disagreements={r['hard_disagreements']}")
        elif kind == "band":
            lines.append(
                f"BAND verdict={r['verdict']} flips={r['flips']} floor={r['floor']} "
                f"floor_runs={','.join(map(str, r['floor_flips']))} "
                f"positions={r['positions']}"
            )
    return lines


STALE_DAYS = 14  # older vLLM captures draw a warning
LOW_R2 = 0.999  # a step fit below this r2 is flagged LOW R2
# Methods that time steady-state marginal decode; criterion groups do not.
MARGINAL = ("loop", "slope")
KV_SHORT = {"float16": "fp16", "bfloat16": "bf16", "half": "fp16"}


@dataclass
class Report:
    ours: Side
    theirs: Side
    age_days: float
    nccl_mismatch: bool
    serving: tuple  # (ours, theirs)
    quant: tuple  # (ours, theirs)
    serving_mismatch: bool
    quant_mismatch: bool
    method_matched: bool  # after allow_method_mismatch
    workload_mismatch: bool
    comm_bound_bs: int | None
    tps_without_theirs: list
    # (side, reason) for each side whose run cannot be compared at all: every ratio is withheld.
    not_comparable: list
    # How the two windows' workload mixes differ (mix_diff): every ratio is withheld when they do.
    mix_diff: list
    rows: list  # ratio None when withheld
    ours_only: list
    theirs_only: list
    agreement: list = field(default_factory=list)
    # Non-empty withholds the ranking.
    server_diff: list = field(default_factory=list)

    @property
    def comparable(self):
        return self.method_matched and not (
            self.serving_mismatch or self.quant_mismatch or self.not_comparable or self.mix_diff
        )


def _short(kv):
    return KV_SHORT.get(kv, kv)


def _nccl_mismatch(ours, theirs):
    """theirs pins an algo (else proto) that ours pins differently."""
    o, t = ours.header, theirs.header
    if t["nccl_algo"] and o["nccl_algo"] and t["nccl_algo"] != o["nccl_algo"]:
        return True
    return bool(t["nccl_proto"] and o["nccl_proto"] and t["nccl_proto"] != o["nccl_proto"])


def pair_row(a, b, report):
    bs = a["bs"]
    ms_a = a["ms_at_eval"] or a["ms_per_step"]
    # Normalized to KV position: both columns at the same resident KV.
    ms_b = b["ms_at_eval"] or b["ms_per_step"]
    r2 = b["r2"] if b["r2"] is not None else float("nan")
    flags, withheld = [], []
    if r2 < LOW_R2:
        flags.append("LOW R2")
    # Older baselines: ms is the raw slope at the fit centroid, not eval_pos.
    if not b["ms_at_eval"]:
        flags.append("RAW")
    if a["kv"] and _short(a["kv"]) != _short(b["kv"]):
        flags.append(f"KV {_short(a['kv'])}/{_short(b['kv'])}")
        withheld.append("kv")
    if report.quant_mismatch:
        flags.append(f"QUANT {report.quant[1]}/{report.quant[0]}")
        withheld.append("quant")
    if report.comm_bound_bs is not None and bs >= report.comm_bound_bs:
        flags.append("COMM")
        withheld.append("comm")
    if a["mode"] == "eager":
        flags.append("EAGER")
    if not report.method_matched:
        withheld.append("method")
    if report.serving_mismatch:
        withheld.append("serving")
    if report.workload_mismatch:
        withheld.append("workload")
    if report.not_comparable:
        withheld.append("run")
    if report.mix_diff:
        withheld.append("mix")
    tps_a, tps_b = bs * 1000.0 / ms_a, bs * 1000.0 / ms_b
    return {
        "group": a["group"],
        "ctx": a["ctx"],
        "bs": bs,
        "tp": a["tp"],
        "ours_ms": ms_a,
        "theirs_ms": ms_b,
        "ours_tok_s": tps_a,
        "theirs_tok_s": tps_b,
        "ratio": None if withheld else tps_a / tps_b,
        "r2": r2,
        "flags": flags,
        "withheld": withheld,
    }


def pair(ours, theirs, allow_method_mismatch=False, now=0.0):
    o, t = ours.header, theirs.header
    serving = (o["serving_style"] or "aggregated", t["serving_style"] or "aggregated")
    quant = (o["quant"] or t["quant"], t["quant"])
    hashes = (o["workload_hash"], t["workload_hash"])
    a = {(r["ctx"], r["bs"], r["tp"]): r for r in ours.rows}
    b = {(r["ctx"], r["bs"], r["tp"]): r for r in theirs.rows}
    comm = t["comm_bound_bs"] if t["comm_bound_bs"] is not None else o["comm_bound_bs"]
    report = Report(
        ours=ours,
        theirs=theirs,
        age_days=age_days(theirs, now),
        nccl_mismatch=_nccl_mismatch(ours, theirs),
        serving=serving,
        quant=quant,
        serving_mismatch=not serving_style_compatible(serving[1], serving[0]),
        quant_mismatch=quant[0] != quant[1],
        method_matched=(o["method"] in MARGINAL and t["method"] in MARGINAL) or bool(allow_method_mismatch),
        workload_mismatch=all(hashes) and hashes[0] != hashes[1],
        comm_bound_bs=None if comm is None else int(comm),
        tps_without_theirs=sorted({k[2] for k in a} - {k[2] for k in b}),
        mix_diff=mix_diff(o.get("workload_mix"), t.get("workload_mix")),
        not_comparable=[
            (side, h["not_comparable"]) for side, h in (("engine", o), ("vllm", t)) if h.get("not_comparable")
        ],
        rows=[],
        ours_only=sorted(set(a) - set(b)),
        theirs_only=sorted(set(b) - set(a)),
    )
    report.rows = [pair_row(a[k], b[k], report) for k in sorted(set(a) & set(b))]
    return report


def _cells(keys):
    return ", ".join(f"{c}x{b}" + (f"@tp{t}" if t != 1 else "") for c, b, t in keys)


def render_preamble(report):
    t, o = report.theirs.header, report.ours.header
    lines = list(report.ours.notes) + list(report.theirs.notes)
    nccl = t["nccl_algo"] or t["nccl_proto"]
    lines.append(
        f"vLLM {t['timestamp']} ({report.age_days:.1f}d)  "
        f"vllm {t['version']}  {t['gpu']}  {t['model']}  "
        f"{t['dtype']}  grid {t['grid']}  {t['graph_mode']}"
        + (f"  bench {t['bench_sha'][:9]}" if t["bench_sha"] else "")
        + (f"  engine {t['engine_sha'][:9]}" if t["engine_sha"] else "")
        + (f"  prompts {t['prompt_mode']}" if t["prompt_mode"] else "")
        + (f"  eval_pos {t['eval_pos']}" if t["eval_pos"] else "")
        + (f"  nccl={t['nccl_algo']}/{t['nccl_proto']}" if nccl else "")
    )
    if report.age_days > STALE_DAYS:
        lines.append(f"WARNING: baseline older than {STALE_DAYS}d; re-run `just run configs/runs/<wl>.yaml`")
    if report.nccl_mismatch:
        lines.append(f"NCCL MISMATCH engine={o['nccl_algo']}/{o['nccl_proto']} vllm={t['nccl_algo']}/{t['nccl_proto']}")
    if report.serving_mismatch:
        lines.append(f"SERVING MISMATCH engine={report.serving[0]} vllm={report.serving[1]}; ratios withheld")
    if report.quant_mismatch:
        lines.append(f"QUANT MISMATCH engine={report.quant[0]} vllm={report.quant[1]}; ratios withheld")
    if report.workload_mismatch:
        lines.append(f"WORKLOAD MISMATCH engine={o['workload_hash']} vllm={t['workload_hash']}; ratios withheld")
    for side, why in report.not_comparable:
        lines.append(f"NOT COMPARABLE {side}: {why}; ratios withheld")
    lines += [f"MIX DIFF {d}; ratios withheld" for d in report.mix_diff]
    lines += [f"SERVER DIFF {d}" for d in report.server_diff]
    if not report.method_matched or report.serving_mismatch or report.quant_mismatch:
        lines.append(
            "engine log times one synchronized step per iteration; the baseline "
            "is a pipelined slope. Not the same quantity, so ratios are withheld "
            "(QBENCH_ALLOW_METHOD_MISMATCH=1 to force)."
        )
    lines.append("")
    if report.tps_without_theirs:
        base_tps = sorted({r["tp"] for r in report.theirs.rows})
        lines.append(
            f"tp mismatch: engine rows at tp {report.tps_without_theirs} have no "
            f"baseline (baseline holds tp {base_tps}); capture one per tp"
        )
    return lines


def render_table(report):
    lines = [
        (
            f"{'group':<20} {'ctx':>6} {'bs':>4} {'tp':>4} {'ours ms':>10} {'vllm ms':>9} "
            f"{'ours tok/s':>12} {'vllm tok/s':>11} {'ratio':>7} {'r2':>7}"
        )
    ]
    for r in report.rows:
        ratio = f"{'--':>7}" if r["ratio"] is None else f"{r['ratio']:>6.2f}x"
        lines.append(
            f"{r['group']:<20} {r['ctx']:>6} {r['bs']:>4} {r['tp']:>4} {r['ours_ms']:>10.3f} "
            f"{r['theirs_ms']:>9.3f} {r['ours_tok_s']:>12.0f} {r['theirs_tok_s']:>11.0f} "
            f"{ratio} {r['r2']:>7.5f}" + "".join(f"  {f}" for f in r["flags"])
        )
    if any("comm" in r["withheld"] for r in report.rows):
        lines.append(
            f"\nCOMM = collective-bound (bs>={report.comm_bound_bs}): absolute ms is "
            f"dominated by the shared TP allreduce and is node/NCCL-specific; "
            f"cross-node reference anchoring is not meaningful for these cells."
        )
    if report.ours_only:
        lines.append(f"\nours-only: {_cells(report.ours_only)}")
    if report.theirs_only:
        lines.append(f"vLLM-only: {_cells(report.theirs_only)}")
    return lines


def render(report):
    """No table when no cell pairs."""
    lines = render_preamble(report)
    if report.rows:
        lines += render_table(report)
    if report.agreement:
        lines += [""] + render_agreement(report.agreement)
    return "".join(line + "\n" for line in lines)


def read_side(path, values):
    path = Path(path)
    if path.is_dir():
        return read_run(path)
    if path.read_text().lstrip().startswith("{"):
        return read_baseline(path)
    return read_bench_log(path, engine_facts(values))


def report_json(report):
    return json.loads(json.dumps(dataclasses.asdict(report), default=str))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="qbench compare", description=__doc__.splitlines()[0])
    ap.add_argument("ours", help="results/<run>/ dir, baselines/*.json, or bench log")
    ap.add_argument("theirs", help="results/<run>/ dir, baselines/*.json, or bench log")
    ap.add_argument(
        "--allow-method-mismatch", action="store_true", help="ratios even when the two sides time different quantities"
    )
    ap.add_argument(
        "--floor",
        action="append",
        default=[],
        metavar="RUN",
        help="a reference run at the engine's dtype; theirs is then the anchor "
        "and ours passes BAND within the worst floor run's argmax flips",
    )
    ap.add_argument(
        "--json", type=Path, default=None, help="report path (default results/compare/<ours>_vs_<theirs>.json)"
    )
    ap.add_argument(
        "--servers", action="store_true", help="only check that two run dirs measured the same server; exit 1 if not"
    )
    args = ap.parse_args(argv)
    for path in (args.ours, args.theirs, *args.floor):
        if not Path(path).exists():
            sys.exit(f"no results at {path}")
    if args.servers:
        if args.floor or args.json:
            ap.error("--servers takes no --floor or --json: it only compares the two servers")
        if not (Path(args.ours).is_dir() and Path(args.theirs).is_dir()):
            ap.error("--servers compares two run dirs")
        diff = server_parity(*(json.loads((Path(p) / "run.json").read_text()) for p in (args.ours, args.theirs)))
        print("".join(f"SERVER DIFF {d}\n" for d in diff) or "SERVER PARITY same server\n", end="")
        sys.exit(1 if diff else 0)
    if args.floor and not all(Path(p).is_dir() for p in (args.ours, args.theirs, *args.floor)):
        ap.error("--floor compares logits runs: ours, theirs and every floor must be run dirs")

    values = env_map.read_env(only=ENV)
    ours = read_side(args.ours, values)
    if not ours.rows and not Path(args.ours).is_dir():
        print("".join(n + "\n" for n in ours.notes), end="")
        sys.exit(f"no engine rows parsed from {args.ours}")
    theirs = read_side(args.theirs, values)
    allow = args.allow_method_mismatch or values.get("compare.allow_method_mismatch", False)
    report = pair(ours, theirs, allow, now=time.time())
    if Path(args.ours).is_dir() and Path(args.theirs).is_dir():
        # Serve runs only: a grid's bench is its own engine, with no server to observe.
        files = [Path(p) / "run.json" for p in (args.ours, args.theirs)]
        runs = [json.loads(f.read_text()) for f in files if f.is_file()]
        if len(runs) == 2 and not any((r.get("summary") or {}).get("grid") for r in runs):
            report.server_diff = server_parity(*runs)
        ours_rows, theirs_rows = read_rows(args.ours), read_rows(args.theirs)
        report.agreement = compare_runs(ours_rows, theirs_rows)
        if args.floor:
            report.agreement.append(band(ours_rows, theirs_rows, [read_rows(f) for f in args.floor]))
    print(render(report), end="")

    a, b = (Path(p).name if Path(p).is_dir() else Path(p).stem for p in (args.ours, args.theirs))
    out = args.json or COMPARE_DIR / f"{a}_vs_{b}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report_json(report), indent=2) + "\n")
    if not report.rows and not report.agreement:
        sys.exit("no overlapping (ctx, bs, tp) points")
    if any(r["kind"] in ("logit_summary", "band") and r["verdict"] == "FAIL" for r in report.agreement):
        sys.exit(1)
