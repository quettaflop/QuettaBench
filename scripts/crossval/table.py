import json, os, re, sys, time

STALE_DAYS = 14
GROUP_PREF = ("decode_batch_paged", "decode_batch", "decode_graph", "decode")
POINT_RE = re.compile(
    r"(?P<group>decode_batch_paged|decode_batch|decode_graph|decode)"
    r"/(?P<param>\d+(?:x\d+)?)"
    r"\s*\n?\s*time:\s*\[[\d.]+ \w+ (?P<mid>[\d.]+) (?P<unit>ms|s|us|µs|ns)"
)
MS = {"ns": 1e-6, "us": 1e-3, "µs": 1e-3, "ms": 1.0, "s": 1e3}


def ours_points(path):
    pref = {name: i for i, name in enumerate(GROUP_PREF)}
    by_key = {}
    for m in POINT_RE.finditer(open(path).read()):
        p = m.group("param")
        ctx, bs = (int(v) for v in p.split("x")) if "x" in p else (int(p), 1)
        group = m.group("group")
        ms = float(m.group("mid")) * MS[m.group("unit")]
        prev = by_key.get((ctx, bs))
        if prev is None or pref[group] < pref.get(prev[1], 99):
            by_key[(ctx, bs)] = (ms, group)
    return by_key


def main():
    bench_path, cache_path = sys.argv[1], sys.argv[2]
    if not os.path.exists(bench_path):
        sys.exit(f"no bench log at {bench_path} — run `just bench llama` first")
    if not os.path.exists(cache_path):
        sys.exit(f"no vLLM baseline at {cache_path} — run `just vllm llama`")

    ours = ours_points(bench_path)
    cache = json.load(open(cache_path))
    theirs = {(p["ctx"], p["bs"]): p for p in cache["points"]}
    age_d = (time.time() - time.mktime(time.strptime(cache["recorded_utc"], "%Y-%m-%dT%H:%M:%SZ"))) / 86400
    print(
        f"vLLM {cache['recorded_utc']} ({age_d:.1f}d)  "
        f"vllm {cache['vllm']}  {cache['gpu']}  {cache.get('model', '?')}  "
        f"{cache.get('dtype', '?')}  grid {cache.get('grid', '?')}  {cache['mode']}"
        + (f"  HEAD {cache['llmsrv_sha_at_capture']}" if cache.get("llmsrv_sha_at_capture") else "")
    )
    if age_d > STALE_DAYS:
        print(f"WARNING: baseline older than {STALE_DAYS}d — re-run `just vllm`")
    print()

    shared = sorted(set(ours) & set(theirs))
    if not shared:
        sys.exit("no overlapping (ctx, bs) points")

    print(f"{'group':<20} {'ctx':>6} {'bs':>4} {'ours ms':>10} {'vllm ms':>9} "
          f"{'ours tok/s':>12} {'vllm tok/s':>11} {'ratio':>7} {'r2':>7}")
    for ctx, bs in shared:
        ms_a, group = ours[(ctx, bs)]
        p = theirs[(ctx, bs)]
        ms_b = p["ms_per_step"]
        r2 = p.get("r2", float("nan"))
        flag = "  LOW R2" if r2 < 0.999 else ""
        print(
            f"{group:<20} {ctx:>6} {bs:>4} {ms_a:>10.3f} {ms_b:>9.3f} "
            f"{bs * 1000.0 / ms_a:>12.0f} {p['tok_s']:>11.0f} "
            f"{(bs * 1000.0 / ms_a) / p['tok_s']:>6.2f}x {r2:>7.5f}{flag}"
        )

    missing = sorted(set(ours) - set(theirs))
    unused = sorted(set(theirs) - set(ours))
    if missing:
        print(f"\nours-only: {', '.join(f'{c}x{b}' for c, b in missing)}")
    if unused:
        print(f"vLLM-only: {', '.join(f'{c}x{b}' for c, b in unused)}")


if __name__ == "__main__":
    main()
