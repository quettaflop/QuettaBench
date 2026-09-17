import json, re, subprocess, sys, time


def main():
    raw, out = sys.argv[1], sys.argv[2]
    text = open(raw).read()
    meta = dict(re.findall(r"^META (\w+)=(.+)$", text, re.M))
    points = []
    for line in text.splitlines():
        if not line.startswith("RESULT"):
            continue
        d = dict(re.findall(r"(\w+)=([-\d.A-Za-z_]+)", line.split("raw=")[0]))
        ctx, bs, ms = int(d["ctx"]), int(d["bs"]), float(d["ms_per_step"])
        points.append({
            "ctx": ctx, "bs": bs, "tp": int(d.get("tp", "1")), "ms_per_step": ms,
            "tok_s": bs * 1000.0 / ms, "r2": float(d["r2"]),
            "kv": d.get("kv", meta.get("dtype", "?")),
        })
    if not points:
        sys.exit(f"no RESULT lines in {raw}")
    # bench_sha: QuettaBench commit at capture time (cwd is the bench repo).
    bench_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    # engine_sha: QuettaServe/DS engine commit; must be set by caller via
    # QS_SHA or DS_ENGINE_SHA env var. Not inferred from cwd (wrong repo).
    import os as _os
    engine_sha = _os.environ.get("QS_SHA") or _os.environ.get("DS_ENGINE_SHA") or None
    json.dump({
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": meta.get("gpu", "?"),
        "vllm": meta.get("vllm", "?"),
        "mode": meta.get("mode", "FULL"),
        "model": meta.get("model", "?"),
        "dtype": meta.get("dtype", "?"),
        "grid": meta.get("grid", "?"),
        "method": "slope",
        "kv_dtype": points[0]["kv"],
        "expert_parallel": meta.get("expert_parallel") == "1",
        "nccl_algo": meta.get("nccl_algo"),
        "nccl_proto": meta.get("nccl_proto"),
        "bench_sha": bench_sha,
        "engine_sha": engine_sha,
        "points": points,
    }, open(out, "w"), indent=2)
    print(f"wrote {out}: {len(points)} points, vllm {meta.get('vllm', '?')} on {meta.get('gpu', '?')}")


if __name__ == "__main__":
    main()
