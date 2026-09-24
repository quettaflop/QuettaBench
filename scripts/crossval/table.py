import calendar, json, os, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import xval_config as _xval

# Age past which the baseline draws a warning: drivers and kernels move, and
# ratios against a stale vLLM capture mislead.
STALE_DAYS = 14
GROUP_PREF = ("decode_batch_paged", "decode_batch", "decode_graph", "decode")
POINT_RE = re.compile(
    r"(?P<group>decode_batch_paged|decode_batch|decode_graph_tp\d+|decode_graph|decode_tp\d+|decode)"
    r"/(?P<param>\d+(?:x\d+)?)"
    r"\s*\n?\s*time:\s*\[[\d.]+ \w+ (?P<mid>[\d.]+) (?P<unit>ms|s|us|µs|ns)"
)
MS = {"ns": 1e-6, "us": 1e-3, "µs": 1e-3, "ms": 1.0, "s": 1e3}


def _loop_points(text):
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
        print(f"engine gdn kernel(s): {', '.join(sorted(gdns))}")
    return pts


def _batch_bench_points(text):
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
        print(f"ignored {truncated} truncated-model batch_bench line(s) (layers != all)")
    if unparsed:
        print(f"ignored {unparsed} unparsed batch_bench line(s)")
    return pts


def ours_points(path):
    """(ctx, bs, tp) -> (ms, group, kv), plus "loop" or "step" for how the
    engine timed them. LOOP lines and batch_bench summaries both report
    steady-state marginal decode, so either earns ratios, and a LOOP line
    supersedes the summary on the same cell. The criterion decode groups
    synchronize inside every step, a different quantity from the slope, so
    the caller withholds ratios for those. Tensor-parallel criterion groups
    are named decode_tp{N} / decode_graph_tp{N}; bare names are tp 1."""
    text = open(path).read()
    bench = _batch_bench_points(text)
    loop = _loop_points(text)
    if loop or bench:
        return {**bench, **loop}, "loop"
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
    return by_key, "step"


def main():
    bench_path, cache_path = sys.argv[1], sys.argv[2]
    if not os.path.exists(bench_path):
        sys.exit(f"no bench log at {bench_path}; run `just bench llama` first")
    if not os.path.exists(cache_path):
        sys.exit(f"no vLLM baseline at {cache_path}; run `just vllm llama`")

    ours, method = ours_points(bench_path)
    if not ours:
        sys.exit(f"no engine rows parsed from {bench_path}")
    cache = json.load(open(cache_path))
    theirs = {(p["ctx"], p["bs"], p.get("tp", 1)): p for p in cache["points"]}
    age_d = (time.time() - calendar.timegm(time.strptime(cache["recorded_utc"], "%Y-%m-%dT%H:%M:%SZ"))) / 86400
    nccl_algo = cache.get("nccl_algo")
    nccl_proto = cache.get("nccl_proto")
    print(
        f"vLLM {cache['recorded_utc']} ({age_d:.1f}d)  "
        f"vllm {cache['vllm']}  {cache['gpu']}  {cache.get('model', '?')}  "
        f"{cache.get('dtype', '?')}  grid {cache.get('grid', '?')}  {cache['mode']}"
        + (f"  bench {cache['bench_sha'][:9]}" if cache.get("bench_sha") else "")
        + (f"  engine {cache['engine_sha'][:9]}" if cache.get("engine_sha") else "")
        + (f"  prompts {cache['prompt_mode']}" if cache.get("prompt_mode") else "")
        + (f"  nccl={nccl_algo}/{nccl_proto}" if nccl_algo or nccl_proto else "")
    )
    if age_d > STALE_DAYS:
        print(f"WARNING: baseline older than {STALE_DAYS}d; re-run `just vllm`")
    eng_nccl_algo = os.environ.get("NCCL_ALGO")
    eng_nccl_proto = os.environ.get("NCCL_PROTO")
    if nccl_algo and eng_nccl_algo and nccl_algo != eng_nccl_algo:
        print(f"NCCL MISMATCH engine={eng_nccl_algo}/{eng_nccl_proto} vllm={nccl_algo}/{nccl_proto}")
    elif nccl_proto and eng_nccl_proto and nccl_proto != eng_nccl_proto:
        print(f"NCCL MISMATCH engine={eng_nccl_algo}/{eng_nccl_proto} vllm={nccl_algo}/{nccl_proto}")

    # Serving topology must match on both sides; a disaggregated engine row
    # against an aggregated baseline is a category error, never a ratio.
    base_style = cache.get("serving_style", "aggregated")
    eng_style = os.environ.get("XVAL_SERVING_STYLE", "aggregated")
    serving_mismatch = not _xval.serving_style_compatible(base_style, eng_style)
    if serving_mismatch:
        print(f"SERVING MISMATCH engine={eng_style} vllm={base_style}; ratios withheld")

    # Quantization must match; an fp4 engine row against a bf16 baseline is not a
    # fair ratio, so it is flagged per row and withheld, like a KV mismatch.
    base_quant = cache.get("quant", cache.get("dtype", "?"))
    eng_quant = os.environ.get("XVAL_QUANT", base_quant)
    quant_mismatch = base_quant != eng_quant
    if quant_mismatch:
        print(f"QUANT MISMATCH engine={eng_quant} vllm={base_quant}; ratios withheld")

    # comm_bound_bs from the merged workloads, matched on crate key or record
    # name. Absent means never comm-bound (tp1 has no TP allreduce); no default.
    crate = cache.get("model", "")
    comm_bound_bs = None
    for _wname, _wl in _xval.workloads().items():
        if (_wname == crate or _wl.get("name") == crate) and "comm_bound_bs" in _wl:
            comm_bound_bs = int(_wl["comm_bound_bs"])
            break

    matched = method == "loop" and cache.get("method", "slope") == "slope"
    if not matched and os.environ.get("XVAL_ALLOW_METHOD_MISMATCH"):
        matched = True
    if serving_mismatch or quant_mismatch:
        matched = False  # topology or quant differences are never comparable
    if not matched:
        print("engine log times one synchronized step per iteration; the baseline "
              "is a pipelined slope. Not the same quantity, so ratios are withheld "
              "(XVAL_ALLOW_METHOD_MISMATCH=1 to force).")
    print()

    shared = sorted(set(ours) & set(theirs))
    ours_tps = {t for _, _, t in ours}
    base_tps = {t for _, _, t in theirs}
    if ours_tps - base_tps:
        print(f"tp mismatch: engine rows at tp {sorted(ours_tps - base_tps)} have no "
              f"baseline (baseline holds tp {sorted(base_tps)}); capture one per tp")
    if not shared:
        sys.exit("no overlapping (ctx, bs, tp) points")

    print(f"{'group':<20} {'ctx':>6} {'bs':>4} {'tp':>4} {'ours ms':>10} {'vllm ms':>9} "
          f"{'ours tok/s':>12} {'vllm tok/s':>11} {'ratio':>7} {'r2':>7}")
    kv_short = {"float16": "fp16", "bfloat16": "bf16", "half": "fp16"}
    comm_footnote_printed = False
    for ctx, bs, tp in shared:
        ms_a, group, kv_a = ours[(ctx, bs, tp)]
        p = theirs[(ctx, bs, tp)]
        ms_b = p["ms_per_step"]
        r2 = p.get("r2", float("nan"))
        flag = "  LOW R2" if r2 < 0.999 else ""
        kv_b = p.get("kv", cache.get("dtype", "?"))
        kv_flagged = False
        if kv_a and kv_short.get(kv_a, kv_a) != kv_short.get(kv_b, kv_b):
            flag += f"  KV {kv_short.get(kv_a, kv_a)}/{kv_short.get(kv_b, kv_b)}"
            kv_flagged = True
        if quant_mismatch:
            flag += f"  QUANT {base_quant}/{eng_quant}"
        comm_flagged = comm_bound_bs is not None and bs >= comm_bound_bs
        if comm_flagged:
            flag += "  COMM"
        # Eager rows include per-step host enqueue; the flag keeps that visible.
        if group == "decode_loop_eager":
            flag += "  EAGER"
        if matched and not kv_flagged and not comm_flagged:
            ratio = f"{(bs * 1000.0 / ms_a) / p['tok_s']:>6.2f}x"
        else:
            ratio = f"{'--':>7}"
        print(
            f"{group:<20} {ctx:>6} {bs:>4} {tp:>4} {ms_a:>10.3f} {ms_b:>9.3f} "
            f"{bs * 1000.0 / ms_a:>12.0f} {p['tok_s']:>11.0f} "
            f"{ratio} {r2:>7.5f}{flag}"
        )
        if comm_flagged and not comm_footnote_printed:
            comm_footnote_printed = True

    if comm_footnote_printed:
        print(f"\nCOMM = collective-bound (bs>={comm_bound_bs}): absolute ms is dominated by "
              f"the shared TP allreduce and is node/NCCL-specific; "
              f"cross-node reference anchoring is not meaningful for these cells.")

    missing = sorted(set(ours) - set(theirs))
    unused = sorted(set(theirs) - set(ours))
    if missing:
        print(f"\nours-only: {', '.join(f'{c}x{b}' + (f'@tp{t}' if t != 1 else '') for c, b, t in missing)}")
    if unused:
        print(f"vLLM-only: {', '.join(f'{c}x{b}' + (f'@tp{t}' if t != 1 else '') for c, b, t in unused)}")


if __name__ == "__main__":
    main()
