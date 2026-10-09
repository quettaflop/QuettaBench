import json, os, re, subprocess, sys, time


def main():
    raw, out = sys.argv[1], sys.argv[2]
    text = open(raw).read()
    meta = dict(re.findall(r"^META (\w+)=(.+)$", text, re.M))
    points = []
    for line in text.splitlines():
        if not line.startswith("RESULT"):
            continue
        head, _, rawpart = line.partition("raw=")
        d = dict(re.findall(r"(\w+)=([-\d.A-Za-z_]+)", head))
        ctx, bs, ms = int(d["ctx"]), int(d["bs"]), float(d["ms_per_step"])
        # ms_at_eval is the slope normalized to the engine's KV window (eval_pos);
        # None on pre-normalization logs (kept None, not ms, so table.py flags the
        # row RAW instead of mistaking a raw slope for a normalized one). raw ys are
        # kept (with step_points in the header) so a later re-reference needs no re-run.
        ms_eval = float(d["ms_at_eval"]) if "ms_at_eval" in d else None
        try:
            raw = json.loads(rawpart) if rawpart else None
        except ValueError:
            raw = None
        points.append({
            "ctx": ctx, "bs": bs, "tp": int(d.get("tp", "1")),
            "ms_per_step": ms, "ms_at_eval": ms_eval,
            "eval_pos": int(d.get("eval_pos", "0")),
            "attn_share": float(d.get("attn_share", "0")),
            "prefill_s": float(d.get("prefill_s", "0")),
            "tok_s": bs * 1000.0 / ms, "r2": float(d["r2"]),
            "kv": d.get("kv", meta.get("dtype", "?")), "raw": raw,
        })
    if not points:
        sys.exit(f"no RESULT lines in {raw}")
    # bench_sha: QuettaBench commit at capture time; empty when the capture box
    # has no git or the tree is an rsync copy, never a crash.
    try:
        bench_sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip()
    except OSError:
        bench_sha = ""
    # engine_sha: QuettaServe/DS engine commit; must be set by caller via
    # QS_SHA or DS_ENGINE_SHA env var. Not inferred from cwd (wrong repo).
    engine_sha = os.environ.get("QS_SHA") or os.environ.get("DS_ENGINE_SHA") or None
    json.dump({
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": meta.get("gpu", "?"),
        "vllm": meta.get("vllm", "?"),
        "mode": meta.get("mode", "FULL"),
        "pp": int(meta.get("pp", "1")),
        "serving_style": meta.get("serving_style", "aggregated"),
        "quant": meta.get("quant", meta.get("dtype", "?")),
        "model": meta.get("model", "?"),
        "dtype": meta.get("dtype", "?"),
        "grid": meta.get("grid", "?"),
        "method": "slope",
        "eval_pos": int(meta.get("eval_pos", "0")),
        "step_points": json.loads(meta["step_points"]) if "step_points" in meta else None,
        "kv_dtype": points[0]["kv"],
        "expert_parallel": meta.get("expert_parallel") == "1",
        "nccl_algo": meta.get("nccl_algo"),
        "nccl_proto": meta.get("nccl_proto"),
        "prompt_mode": meta.get("prompt_mode", "distinct"),
        "bench_sha": bench_sha,
        "engine_sha": engine_sha,
        "points": points,
    }, open(out, "w"), indent=2)
    print(f"wrote {out}: {len(points)} points, vllm {meta.get('vllm', '?')} on {meta.get('gpu', '?')}")


if __name__ == "__main__":
    main()
