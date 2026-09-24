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

# CFG comes from xval_config: xval.yaml when present, workloads.json fallback.
sys.path.insert(0, str(Path(__file__).parent))
from xval_config import collective as _xval_collective, workloads as _xval_workloads
from xval_config import grids as _xval_grids, step_points as _xval_step_points
from xval_config import link_profile as _xval_link_profile
CFG = {
    "workloads": _xval_workloads(),
    "grids": _xval_grids(),
    "step_points": _xval_step_points(),
}
# Same link-profile rule as bench.sh (one detection, in xval_config); empty
# collective values are skipped, not exported.
os.environ.setdefault("XVAL_LINK_PROFILE", _xval_link_profile())
for _k, _v in _xval_collective().items():
    if _v:
        os.environ.setdefault(_k, _v)

# Fraction of the KV budget a cell may claim before it is skipped; the rest
# covers fragmentation and scheduler headroom.
CAP_FRAC = 0.80


def toks(n, seed=10):
    return {"prompt_token_ids": [seed + (i % 1000) for i in range(n)]}


def engine_prompt_ids(n):
    """Same synthetic prompt the engine benches feed. clone mode gives every
    slot these ids so both sides route identically under greedy."""
    return [(i * 137 + 11) % 100_000 for i in range(n)]


_CORPUS_IDS = None


def corpus_prompt_ids(tokenizer, n, slot):
    """Sliding windows over the XVAL_CORPUS text; slots start at distinct
    offsets and the text repeats when a window outruns it."""
    global _CORPUS_IDS
    if _CORPUS_IDS is None:
        path = os.environ.get("XVAL_CORPUS")
        if not path:
            sys.exit("prompt_mode=corpus needs XVAL_CORPUS pointing at a text file")
        _CORPUS_IDS = tokenizer(open(path).read())["input_ids"]
    ids = _CORPUS_IDS
    start = (slot * 997) % max(1, len(ids))
    out = []
    while len(out) < n:
        out.extend(ids[start:start + (n - len(out))])
        start = 0
    return out


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
    ap.add_argument(
        "--dump-tokens",
        metavar="FILE",
        help="append per-cell greedy token ids (JSONL)",
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
    pp = int(os.environ.get("XVAL_PP", wl.get("pp", 1)))
    dtype = wl["dtype"]
    kv_dtype = wl.get("kv_dtype")
    weights_gib = float(wl["weights_gib"])
    maxlen = int(wl["maxlen"])
    util = float(os.environ.get("GPU_UTIL", wl.get("gpu_util", 0.90)))
    # Routing regimes (matters for MoE): distinct = per-slot seeds; clone =
    # engine-identical ids so ratios compare identical routing; corpus = real-text windows.
    prompt_mode = os.environ.get("PROMPT_MODE", wl.get("prompt_mode", "distinct"))
    if prompt_mode not in ("distinct", "clone", "corpus", "synth", "trace"):
        sys.exit(f"unknown prompt_mode {prompt_mode!r}; use distinct, clone, corpus, synth or trace")
    # synth = distinct streams (synth_bench); trace = replay XVAL_TRACE (e.g. Mooncake).
    trace_reqs = None
    if prompt_mode == "trace":
        import synth_bench
        trace_reqs = synth_bench.load_trace(os.environ["XVAL_TRACE"])
        print(f"META trace={os.environ['XVAL_TRACE']} reqs={len(trace_reqs)} "
              f"workload_hash={synth_bench.workload_hash(trace_reqs)}", flush=True)

    print(f"META gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(f"META vllm={vllm.__version__}", flush=True)
    print(f"META model={wl['name']}", flush=True)
    print(f"META dtype={dtype}", flush=True)
    print(f"META grid={gridname}", flush=True)
    print(f"META mode={mode}", flush=True)
    print(f"META nccl_algo={os.environ.get('NCCL_ALGO','auto')}", flush=True)
    print(f"META nccl_proto={os.environ.get('NCCL_PROTO','auto')}", flush=True)
    print(f"META link_profile={os.environ['XVAL_LINK_PROFILE']}", flush=True)
    print(f"META prompt_mode={prompt_mode}", flush=True)
    # cache.py parses META one key per line; never combine keys.
    print(f"META tp={tp}", flush=True)
    print(f"META pp={pp}", flush=True)

    kw = {}
    if pp > 1:
        kw["pipeline_parallel_size"] = pp
    if args.nograph:
        kw["compilation_config"] = {"cudagraph_mode": "NONE"}
    if kv_dtype:
        kw["kv_cache_dtype"] = kv_dtype
    # Quantized weights (nvfp4 / fp8): pass through to vLLM and record it so the
    # table never pairs an fp4 row against a bf16 baseline as if equal.
    quant = wl.get("quant")
    if quant:
        kw["quantization"] = quant
    print(f"META quant={quant or dtype}", flush=True)
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
        # Dense formula counts every layer; hybrid stacks hold KV only in the
        # attention layers and must set kv_bytes or the capacity gate overestimates.
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

    def gen(prompts, n, capture=False):
        sp = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        return (dt, outs) if capture else dt

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
        if prompt_mode == "clone":
            base = engine_prompt_ids(ctx)
            prompts = [{"prompt_token_ids": list(base)} for _ in range(bs)]
        elif prompt_mode == "corpus":
            prompts = [
                {"prompt_token_ids": corpus_prompt_ids(llm.get_tokenizer(), ctx, j)}
                for j in range(bs)
            ]
        elif prompt_mode == "synth":
            import synth_bench
            prompts = [{"prompt_token_ids": ids}
                       for ids in synth_bench.take(
                           synth_bench.synth_requests(bs, ctx, seed=ctx), bs, ctx)[0]]
        elif prompt_mode == "trace":
            ids_list, cycled = synth_bench.take(trace_reqs, bs, ctx)
            if cycled:
                print(f"NOTE ctx={ctx} bs={bs} trace cycled (fewer reqs than bs)", flush=True)
            prompts = [{"prompt_token_ids": ids} for ids in ids_list]
        else:
            prompts = [toks(ctx, seed=10 + j) for j in range(bs)]
        if args.dump_tokens:
            # Routing-parity audit: greedy ids must match the engine's DS_TOKEN_TRACE
            # if both sides routed alike; divergence rate is the residual.
            _, outs = gen(prompts, 64, capture=True)
            with open(args.dump_tokens, "a") as fh:
                fh.write(json.dumps({
                    "ctx": ctx, "bs": bs, "prompt_mode": prompt_mode,
                    "tokens": [list(o.outputs[0].token_ids)[:64] for o in outs],
                }) + "\n")
        gen(prompts, max(steps))
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
