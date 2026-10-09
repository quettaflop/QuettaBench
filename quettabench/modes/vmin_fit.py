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
from vllm import SamplingParams

# CFG comes from xval_config: xval.yaml when present, workloads.json fallback.
sys.path.insert(0, str(Path(__file__).parents[2]))
from quettabench.config.xval_config import collective as _xval_collective, workloads as _xval_workloads
from quettabench.config.xval_config import grids as _xval_grids, step_points as _xval_step_points
from quettabench.config.xval_config import link_profile as _xval_link_profile
from quettabench.config.xval_config import run_params as _xval_run_params
from quettabench.config.schema import Config, EngineCfg, HardwareCfg, ModeCfg, WorkloadCfg
from quettabench.engines import vllm as vllm_adapter
from quettabench.engines.drivers import InProcessDriver
from quettabench.metrics.step_fit import fit, ms_at_eval
from quettabench.workloads.grid import GridTrace, distinct_ids
from quettabench.workloads.live import RealTrace
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
    # The raw slope is the marginal decode cost at the centroid of the fit
    # points (about ctx + mean(steps)); the engine times its loop around
    # ctx + timing_steps/2. eval_pos names that engine window so the vLLM number
    # normalizes to the same resident-KV length and both columns compare decode
    # cost at one KV position. XVAL_EVAL_POS overrides the default.
    mean_steps = sum(steps) / len(steps)
    eval_pos = int(os.environ.get("XVAL_EVAL_POS", _xval_run_params()["timing_steps"] // 2))

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
    trace = None
    if prompt_mode == "trace":
        trace = RealTrace.from_jsonl(os.environ["XVAL_TRACE"])
        print(f"META trace={os.environ['XVAL_TRACE']} reqs={len(trace.requests())} "
              f"workload_hash={trace.workload_hash()}", flush=True)

    print(f"META gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(f"META vllm={vllm.__version__}", flush=True)
    print(f"META model={wl['name']}", flush=True)
    print(f"META dtype={dtype}", flush=True)
    print(f"META grid={gridname}", flush=True)
    print(f"META mode={mode}", flush=True)
    print(f"META eval_pos={eval_pos}", flush=True)
    print(f"META step_points={steps}", flush=True)
    print(f"META nccl_algo={os.environ.get('NCCL_ALGO','auto')}", flush=True)
    print(f"META nccl_proto={os.environ.get('NCCL_PROTO','auto')}", flush=True)
    print(f"META link_profile={os.environ['XVAL_LINK_PROFILE']}", flush=True)
    print(f"META prompt_mode={prompt_mode}", flush=True)
    # cache.py parses META one key per line; never combine keys.
    print(f"META tp={tp}", flush=True)
    print(f"META pp={pp}", flush=True)

    # Quantized weights (nvfp4 / fp8): pass through to vLLM and record it so the
    # table never pairs an fp4 row against a bf16 baseline as if equal.
    quant = wl.get("quant")
    print(f"META quant={quant or dtype}", flush=True)
    # MoE workloads shard their experts across the tp ranks.
    if wl.get("expert_parallel"):
        print("META expert_parallel=1", flush=True)

    engine = EngineCfg(model=args.model, tp=tp, pp=pp, ep=bool(wl.get("expert_parallel")),
                       dtype=dtype, kv_dtype=kv_dtype or "auto", quant=quant or None,
                       max_model_len=maxlen, gpu_util=util, cudagraph=not args.nograph)
    run = Config(engine=engine, workload=WorkloadCfg(), mode=ModeCfg(), hardware=HardwareCfg())
    llm = InProcessDriver(vllm_adapter.ADAPTER.translate(run)).llm()

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

    grid = GridTrace(points, steps, prompt_mode, maxlen,
                     corpus_path=os.environ.get("XVAL_CORPUS"),
                     tokenizer=llm.get_tokenizer, trace=trace)
    over_maxlen = grid.cells_over_maxlen

    gen([{"prompt_token_ids": distinct_ids(256)}], 20)
    print("WARM_OK", flush=True)

    for ctx, bs in points:
        top = max(steps)
        if (ctx, bs) in over_maxlen:
            print(f"SKIP ctx={ctx} bs={bs} reason=maxlen", flush=True)
            continue
        need = (ctx + top) * bs
        if cap and need > CAP_FRAC * cap:
            print(f"SKIP ctx={ctx} bs={bs} need_kv={need} cap={cap}", flush=True)
            continue
        # Checked at the first cell that needs the corpus, so a run whose
        # cells are all skipped never asks for one.
        if prompt_mode == "corpus" and not grid.corpus_path:
            sys.exit("prompt_mode=corpus needs XVAL_CORPUS pointing at a text file")
        ids_list, cycled = grid.cell_prompts(ctx, bs)
        if cycled:
            print(f"NOTE ctx={ctx} bs={bs} trace cycled (fewer reqs than bs)", flush=True)
        prompts = [{"prompt_token_ids": ids} for ids in ids_list]
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
        attn = kv_gib / (kv_gib + weights_gib)
        ms_slope = slope * 1e3
        # ms_per_step stays the raw slope for back-compat; ms_at_eval is the
        # matched-KV number table.py pairs (see step_fit.ms_at_eval).
        ms_eval = ms_at_eval(ms_slope, attn, eval_pos, mean_steps, ctx)
        print(
            f"RESULT mode={mode} tp={tp} kv={kv_dtype or dtype} ctx={ctx} bs={bs} "
            f"ms_per_step={ms_slope:.3f} ms_at_eval={ms_eval:.3f} eval_pos={eval_pos} "
            f"mean_steps={mean_steps:.0f} r2={r2:.5f} prefill_s={inter:.2f} "
            f"kv_gib={kv_gib:.1f} attn_share={attn:.3f} "
            f"raw={[round(y, 2) for y in ys]}",
            flush=True,
        )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
