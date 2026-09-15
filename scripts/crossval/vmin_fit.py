# vmin_fit.py <crate> [grid] --model <dir> [--nograph]
# decode ms/step = slope of total generate time vs generated tokens
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import vllm
from vllm import LLM, SamplingParams

CFG = json.load(open(Path(__file__).with_name("workloads.json")))

# Fraction of the KV budget a cell may claim before it is skipped; the rest
# covers fragmentation and scheduler headroom.
CAP_FRAC = 0.80


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
    ap = argparse.ArgumentParser()
    ap.add_argument("crate", help="workload name from workloads.json")
    ap.add_argument("grid", nargs="?", default="", help="grid override")
    ap.add_argument("--model", required=True, help="model weights dir")
    ap.add_argument("--nograph", action="store_true", help="disable CUDA graphs")
    ap.add_argument(
        "--allow-unverified",
        action="store_true",
        help="run a workload whose verified flag is false (bring-up only)",
    )
    args = ap.parse_args()

    wl = CFG["workloads"].get(args.crate)
    if wl is None:
        sys.exit(f"unknown crate {args.crate!r}; workloads: {', '.join(CFG['workloads'])}")
    if not wl.get("verified") and not args.allow_unverified:
        sys.exit(
            f"{args.crate} is not validated yet; pass --allow-unverified to bring it up"
        )
    grids = {k.lower(): v for k, v in CFG["grids"].items()}
    gridname = (args.grid or wl["grid"]).lower()
    if gridname not in grids:
        sys.exit(f"unknown grid {gridname!r}; grids: {', '.join(grids)}")
    points = grids[gridname]
    steps = CFG["step_points"]

    mode = "NOGRAPH" if args.nograph else "FULL"
    tp = int(wl.get("tp", 1))
    dtype = wl["dtype"]
    kv_dtype = wl.get("kv_dtype")
    weights_gib = float(wl["weights_gib"])
    maxlen = int(wl["maxlen"])
    util = float(os.environ.get("GPU_UTIL", wl.get("gpu_util", 0.90)))

    print(f"META gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(f"META vllm={vllm.__version__}", flush=True)
    print(f"META model={wl['name']}", flush=True)
    print(f"META dtype={dtype}", flush=True)
    print(f"META grid={gridname}", flush=True)
    print(f"META mode={mode}", flush=True)

    kw = {}
    if args.nograph:
        kw["compilation_config"] = {"cudagraph_mode": "NONE"}
    if kv_dtype:
        kw["kv_cache_dtype"] = kv_dtype
    # MoE workloads shard their experts across the tp ranks.
    if wl.get("expert_parallel"):
        kw["enable_expert_parallel"] = True
        print("META expert_parallel=1", flush=True)

    llm = LLM(
        model=args.model,
        dtype=dtype,
        tensor_parallel_size=tp,
        gpu_memory_utilization=util,
        max_model_len=maxlen,
        enable_prefix_caching=False,
        disable_log_stats=True,
        trust_remote_code=True,
        **kw,
    )

    cfg = json.load(open(os.path.join(args.model, "config.json")))
    cfg = cfg.get("text_config", cfg)
    kv_bytes = wl.get("kv_bytes")
    if kv_bytes:
        kv_b = int(kv_bytes)
    else:
        hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
        hd += cfg.get("qk_rope_head_dim") or 0
        kv_b = 2 * cfg["num_hidden_layers"] * cfg["num_key_value_heads"] * hd * 2
    gpu_gib = torch.cuda.get_device_properties(0).total_memory / (1 << 30)
    free_gib = tp * (gpu_gib * util) - weights_gib - tp * 5.0
    cap = int(free_gib * (1 << 30) / kv_b) if free_gib > 0 else 0
    print(
        f"CAPACITY mode={mode} tp={tp} dtype={dtype} kv_bytes_per_token={kv_b} "
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
        if cap and need > CAP_FRAC * cap:
            print(f"SKIP ctx={ctx} bs={bs} need_kv={need} cap={cap}", flush=True)
            continue
        prompts = [toks(ctx, seed=10 + j) for j in range(bs)]
        # Full-length warm: every decode graph for this batch size must be
        # captured before timing, or the capture cost lands in the first point.
        gen(prompts, max(steps))
        # Keep the cleanest of a few full slopes, chosen by r2 (not the fastest
        # sample): a lone noisy fit at low-work cells is not trusted, and
        # selecting on r2 rather than min avoids biasing the step time down.
        attempts = int(os.environ.get("VMIN_ATTEMPTS", "4"))
        best = None  # (r2, slope, inter, ys)
        for _ in range(attempts):
            ys = [gen(prompts, n) for n in steps]
            s, i, rr = fit(steps, ys)
            if best is None or rr > best[0]:
                best = (rr, s, i, ys)
            if rr >= 0.999:
                break
        r2, slope, inter, ys = best
        kv_gib = ctx * bs * kv_b / (1 << 30)
        print(
            f"RESULT mode={mode} tp={tp} kv={kv_dtype or dtype} ctx={ctx} bs={bs} "
            f"ms_per_step={slope * 1e3:.3f} r2={r2:.5f} prefill_s={inter:.2f} "
            f"kv_gib={kv_gib:.1f} attn_share={kv_gib / (kv_gib + weights_gib):.3f} "
            f"raw={[round(y, 2) for y in ys]}",
            flush=True,
        )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
