import calendar, json, os, re, sys, time

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
    """(ctx, bs, tp) -> (ms, "decode_loop", kv) from the engine's LOOP lines:
    K pipelined decode steps timed under one sync, the same quantity as the
    baseline's slope."""
    pts = {}
    for line in text.splitlines():
        if not line.startswith("LOOP "):
            continue
        d = dict(re.findall(r"(\w+)=([\w.]+)", line))
        key = (int(d["ctx"]), int(d["bs"]), int(d.get("tp", "1")))
        pts[key] = (float(d["ms_per_step"]), "decode_loop", d.get("kv"))
    return pts


def _batch_bench_points(text):
    """(ctx, bs, tp) -> (ms, "batch_bench", kv) from the deepseek engine's
    stock bench summary. The median of the graph-replay decode steps is the
    steady-state marginal step, the same quantity as the baseline slope.
    Truncated-model runs (layers != all) time a different model and are
    ignored. The line does not state its KV format, so kv is "?" and the
    table flags it against the baseline's."""
    pts = {}
    truncated = 0
    for line in text.splitlines():
        if not line.startswith("[batch_bench] "):
            continue
        d = dict(re.findall(r"(\w+)=([\w.]+)", line))
        ms = re.search(r"median ([\d.]+) ms/step", line)
        if not ms:
            continue
        if d.get("layers") != "all":
            truncated += 1
            continue
        key = (int(d["prompt"]), int(d["bs"]), int(d.get("world", "1")))
        pts[key] = (float(ms.group(1)), "batch_bench", d.get("kv", "?"))
    if truncated:
        print(f"ignored {truncated} truncated-model batch_bench line(s) (layers != all)")
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
        sys.exit(f"no bench log at {bench_path} — run `just bench llama` first")
    if not os.path.exists(cache_path):
        sys.exit(f"no vLLM baseline at {cache_path} — run `just vllm llama`")

    ours, method = ours_points(bench_path)
    cache = json.load(open(cache_path))
    theirs = {(p["ctx"], p["bs"], p.get("tp", 1)): p for p in cache["points"]}
    age_d = (time.time() - calendar.timegm(time.strptime(cache["recorded_utc"], "%Y-%m-%dT%H:%M:%SZ"))) / 86400
    print(
        f"vLLM {cache['recorded_utc']} ({age_d:.1f}d)  "
        f"vllm {cache['vllm']}  {cache['gpu']}  {cache.get('model', '?')}  "
        f"{cache.get('dtype', '?')}  grid {cache.get('grid', '?')}  {cache['mode']}"
        + (f"  HEAD {cache['llmsrv_sha_at_capture']}" if cache.get("llmsrv_sha_at_capture") else "")
    )
    if age_d > STALE_DAYS:
        print(f"WARNING: baseline older than {STALE_DAYS}d — re-run `just vllm`")
    matched = method == "loop" and cache.get("method", "slope") == "slope"
    if not matched and os.environ.get("XVAL_ALLOW_METHOD_MISMATCH"):
        matched = True
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
    for ctx, bs, tp in shared:
        ms_a, group, kv_a = ours[(ctx, bs, tp)]
        p = theirs[(ctx, bs, tp)]
        ms_b = p["ms_per_step"]
        r2 = p.get("r2", float("nan"))
        flag = "  LOW R2" if r2 < 0.999 else ""
        kv_b = p.get("kv", cache.get("dtype", "?"))
        if kv_a and kv_short.get(kv_a, kv_a) != kv_short.get(kv_b, kv_b):
            flag += f"  KV {kv_short.get(kv_a, kv_a)}/{kv_short.get(kv_b, kv_b)}"
        ratio = f"{(bs * 1000.0 / ms_a) / p['tok_s']:>6.2f}x" if matched else f"{'--':>7}"
        print(
            f"{group:<20} {ctx:>6} {bs:>4} {tp:>4} {ms_a:>10.3f} {ms_b:>9.3f} "
            f"{bs * 1000.0 / ms_a:>12.0f} {p['tok_s']:>11.0f} "
            f"{ratio} {r2:>7.5f}{flag}"
        )

    missing = sorted(set(ours) - set(theirs))
    unused = sorted(set(theirs) - set(ours))
    if missing:
        print(f"\nours-only: {', '.join(f'{c}x{b}' + (f'@tp{t}' if t != 1 else '') for c, b, t in missing)}")
    if unused:
        print(f"vLLM-only: {', '.join(f'{c}x{b}' + (f'@tp{t}' if t != 1 else '') for c, b, t in unused)}")


if __name__ == "__main__":
    main()
