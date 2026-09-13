# vmin_fit.py MODE CRATE [GRID] — decode ms/step = slope of time vs generated tokens
import json, os, sys, time
from pathlib import Path
from vllm import LLM, SamplingParams

CFG = json.load(open(Path(__file__).with_name("workloads.json")))


def toks(n, seed=10):
    return {"prompt_token_ids": [seed + (i % 1000) for i in range(n)]}


def fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    inter = my - slope * mx
    ss_res = sum((y - (inter + slope * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    return slope, inter, r2


def main():
    mode, crate = sys.argv[1], sys.argv[2].lower()
    wl = CFG["workloads"][crate]
    grids = {k.lower(): v for k, v in CFG["grids"].items()}
    gridname = (sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else wl["grid"]).lower()
    points = grids[gridname]
    steps = CFG["step_points"]

    model = os.environ["MODELPATH"]
    weights_gib = float(os.environ.get("WEIGHTS_GIB", wl["weights_gib"]))
    maxlen = int(os.environ.get("MAXLEN", wl["maxlen"]))
    dtype = os.environ.get("DTYPE", wl["dtype"])
    gpu_gib = float(os.environ.get("GPU_GIB", "80"))
    tp = int(os.environ.get("TP", wl.get("tp", 1)))

    cc = {}
    if mode == "NOGRAPH":
        cc["cudagraph_mode"] = "NONE"
    if os.environ.get("COMPILE_MODE"):
        cc["mode"] = int(os.environ["COMPILE_MODE"])
    kw = {"compilation_config": cc} if cc else {}
    kv_dtype = os.environ.get("KV_DTYPE", wl.get("kv_dtype"))
    if kv_dtype:
        kw["kv_cache_dtype"] = kv_dtype
    if os.environ.get("EXPERT_PARALLEL") or wl.get("expert_parallel"):
        kw["enable_expert_parallel"] = True

    llm = LLM(
        model=model,
        dtype=dtype,
        tensor_parallel_size=tp,
        gpu_memory_utilization=float(os.environ.get("GPU_UTIL", wl.get("gpu_util", 0.90))),
        max_model_len=maxlen,
        enable_prefix_caching=False,
        disable_log_stats=True,
        trust_remote_code=True,
        **kw,
    )

    cfg = json.load(open(os.path.join(model, "config.json")))
    cfg = cfg.get("text_config", cfg)
    kv_bytes = os.environ.get("KV_BYTES", wl.get("kv_bytes"))
    if kv_bytes:
        kv_b = int(kv_bytes)
    else:
        hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
        hd += cfg.get("qk_rope_head_dim") or 0
        kv_b = 2 * cfg["num_hidden_layers"] * cfg["num_key_value_heads"] * hd * 2
    util = float(os.environ.get("GPU_UTIL", wl.get("gpu_util", 0.90)))
    free_gib = tp * (gpu_gib * util) - weights_gib - tp * 5.0
    cap = int(free_gib * (1 << 30) / kv_b) if free_gib > 0 else 0
    ep = 1 if (os.environ.get("EXPERT_PARALLEL") or wl.get("expert_parallel")) else 0
    print(
        f"CAPACITY mode={mode} tp={tp} ep={ep} dtype={dtype} kv_bytes_per_token={kv_b} "
        f"free_gib={free_gib:.0f} kv_tokens={cap}",
        flush=True,
    )

    def gen(prompts, n):
        sp = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
        t0 = time.perf_counter()
        llm.generate(prompts, sp, use_tqdm=False)
        return time.perf_counter() - t0

    gen([toks(256)], 20)
    print("WARM_OK", flush=True)

    for ctx, bs in points:
        top = max(steps)
        if ctx + top + 2 > maxlen:
            print(f"SKIP ctx={ctx} bs={bs} reason=maxlen", flush=True)
            continue
        need = (ctx + top) * bs
        if cap and need > float(os.environ.get("CAP_FRAC", "0.80")) * cap:
            print(f"SKIP ctx={ctx} bs={bs} need_kv={need} cap={cap}", flush=True)
            continue
        prompts = [toks(ctx, seed=10 + j) for j in range(bs)]
        gen(prompts, 20)
        ys = [gen(prompts, n) for n in steps]
        slope, inter, r2 = fit(steps, ys)
        kv_gib = ctx * bs * kv_b / (1 << 30)
        print(
            f"RESULT mode={mode} tp={tp} ep={ep} ctx={ctx} bs={bs} "
            f"ms_per_step={slope * 1e3:.3f} r2={r2:.5f} prefill_s={inter:.2f} "
            f"kv_gib={kv_gib:.1f} attn_share={kv_gib / (kv_gib + weights_gib):.3f} "
            f"raw={[round(y, 2) for y in ys]}",
            flush=True,
        )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
