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
            "ctx": ctx, "bs": bs, "ms_per_step": ms,
            "tok_s": bs * 1000.0 / ms, "r2": float(d["r2"]),
        })
    if not points:
        sys.exit(f"no RESULT lines in {raw}")
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    json.dump({
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": meta.get("gpu", "?"),
        "vllm": meta.get("vllm", "?"),
        "mode": meta.get("mode", "FULL"),
        "model": meta.get("model", "?"),
        "dtype": meta.get("dtype", "?"),
        "grid": meta.get("grid", "?"),
        "llmsrv_sha_at_capture": sha,
        "points": points,
    }, open(out, "w"), indent=2)
    print(f"wrote {out}: {len(points)} points, vllm {meta.get('vllm', '?')} on {meta.get('gpu', '?')}")


if __name__ == "__main__":
    main()
