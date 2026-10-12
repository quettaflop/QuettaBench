#!/usr/bin/env python3
"""Grouped-MoE expert-kernel grid (vLLM fused_experts over tokens x (E, top_k, intermediate, hidden)) -> moe/{gpu}.csv rows, plus the routing kernels' affine fits -> moe/{gpu}_routing.json entries.
The roofline has no launch cost, so it under-prices a grouped kernel exactly when each expert gets ~1 token; this grid replaces it. Use --graph for graphed decode (graph/), the eager default (eager/) includes dispatch.
--routing instead dumps one MoE block's input, output and routing from a transformers model to results/probes/<gpu>/moe_routing/<run-id>/<model>_layer<L>.safetensors, for an engine-side diff.
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

import torch

from quettabench.probes._timing import graph_time_us
from quettabench.probes.base import ProbeRow, add_output_args, emit, out_root, run_recorded

# Per-expert geometry of every MoE model in engine/device_spec/models/. (n_experts, top_k, intermediate_per_expert, hidden). `intermediate` is sharded by tp at sweep time.
#
# gpt-oss is swept for its SHAPE coverage, but the composition deliberately will NOT price MXFP4-expert models off this bf16 grid: only the weight-read term scales with expert dtype, and a measured latency cannot be decomposed after the fact. Those models keep the analytic roofline until an MXFP4 grid is swept.
GEOMETRIES: dict[str, tuple[int, int, int, int]] = {
    "qwen3-30b-a3b": (128, 8, 768, 2048),
    "qwen3-235b-a22b": (128, 8, 1536, 4096),
    "mixtral-8x7b": (8, 2, 14336, 4096),
    "gpt-oss-20b": (32, 4, 2880, 2880),
    "gpt-oss-120b": (128, 4, 2880, 2880),
}
# Dense at the small end: that is the decode regime where the roofline fails and where every agentic / low-concurrency workload lives.
TOKENS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 2048]
REPS, WARMUP = 30, 10
DT = torch.bfloat16
DT_BYTES = 2
# --dtype fp8: measure vLLM's FP8 grouped-expert path instead of the bf16 one.
# Block-scaled (128x128 weight blocks), matching a `quant_method: fp8` checkpoint's expert tensors (Qwen3-235B-A22B-FP8). Set by main().
_FP8 = False
FP8_BLOCK = [128, 128]
# Element counts for the routing affine fits (bytes = elements * 2 * (r + w)).
ROUTE_ELEMS = [2**e for e in range(12, 26)]
# The columns a moe/{gpu}.csv row is keyed on.
KEY = ("tokens", "n_experts", "top_k", "intermediate", "hidden")


def _fused_experts():
    """vLLM's grouped MoE entry point, across the versions that moved it.

    Returns (fused_experts, fused_topk). Raising here is correct: a torch re-implementation would measure a different kernel than the one vLLM runs, which is the entire thing this probe exists to capture.
    """
    paths = [
        "vllm.model_executor.layers.fused_moe.fused_moe",
        "vllm.model_executor.layers.fused_moe",
    ]
    last = None
    for mod in paths:
        try:
            m = __import__(mod, fromlist=["fused_experts", "fused_topk"])
            fe = getattr(m, "fused_experts", None)
            ft = getattr(m, "fused_topk", None)
            if fe is not None and ft is not None:
                return fe, ft
        except Exception as ex:  # noqa: BLE001 - probe over many vLLM versions
            last = ex
    raise RuntimeError(
        "could not import vllm fused_experts/fused_topk "
        f"(tried {paths}; last error: {last}). This probe must measure vLLM's own "
        "grouped kernel -- run it in the vLLM env (PYTHON_BIN)."
    )


_GRAPH = False  # set by --graph: CUDA-graph-replay timing (graphed-decode-faithful)
# --backend flashinfer: measure FlashInfer's CUTLASS grouped-expert kernel instead of the Triton fused_experts. vLLM >= 0.28 serves unquantized bf16 MoE through it ("Using FlashInfer CUTLASS Unquantized MoE backend" in the server log), and the two differ a lot: on RTXPRO6000/Qwen3-235B CUTLASS is ~2x SLOWER at batch 1 and several times FASTER at prefill chunk sizes, so a Triton grid misprices both ends.
_BACKEND = "triton"


def _flashinfer_moe():
    """flashinfer is imported here: only --backend flashinfer needs it."""
    from flashinfer.fused_moe import cutlass_fused_moe
    return cutlass_fused_moe


def _time(fn) -> float:
    """min-of-REPS CUDA-event time (us). MIN, not mean: it matches GemmTable's min-on-load reduce and the warm, back-to-back kernels of a real decode step."""
    if _GRAPH:
        return graph_time_us(fn)
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return min(ts)


def _fp8_quant_config(e, inter, hidden, dev):
    """vLLM's FusedMoEQuantConfig for block-scaled FP8 experts.

    Scales are positive and O(1e-2) -- the range a real per-block weight scale takes. Magnitude cannot change kernel time, but a zero/NaN scale can trip a different code path, so they are not left as raw randn. vllm is imported here: only --dtype fp8 needs it.
    """
    from vllm.model_executor.layers.fused_moe.config import (
        fp8_w8a8_moe_quant_config,
    )
    bn, bk = FP8_BLOCK
    nb = lambda x, b: (x + b - 1) // b  # noqa: E731
    s1 = torch.rand(e, nb(2 * inter, bn), nb(hidden, bk),
                    device=dev, dtype=torch.float32).mul_(0.01).add_(1e-4)
    s2 = torch.rand(e, nb(hidden, bn), nb(inter, bk),
                    device=dev, dtype=torch.float32).mul_(0.01).add_(1e-4)
    return fp8_w8a8_moe_quant_config(w1_scale=s1, w2_scale=s2, block_shape=list(FP8_BLOCK))


def sweep_expert_kernel(name, e, k, inter, hidden, tokens_axis, dev, max_mem_gb):
    """One geometry's (tokens -> us) curve for the grouped expert kernel."""
    fused_experts, fused_topk = _fused_experts()
    w_bytes = 1 if _FP8 else DT_BYTES
    w_gb = (e * 2 * inter * hidden + e * hidden * inter) * w_bytes / 1e9
    if w_gb > max_mem_gb:
        print(f"  skip {name} E={e} inter={inter}: weights {w_gb:.1f} GB > {max_mem_gb}",
              flush=True)
        return []
    w1 = torch.randn(e, 2 * inter, hidden, device=dev, dtype=DT) * 0.02
    w2 = torch.randn(e, hidden, inter, device=dev, dtype=DT) * 0.02
    qc = None
    if _FP8:
        w1 = w1.to(torch.float8_e4m3fn)
        w2 = w2.to(torch.float8_e4m3fn)
        qc = _fp8_quant_config(e, inter, hidden, dev)
    rows = []
    for m in tokens_axis:
        try:
            x = torch.randn(m, hidden, device=dev, dtype=DT)
            # Real router logits -> real top-k -> realistic (imbalanced) expert occupancy, so moe_align_block_size padding is representative.
            gate = torch.randn(m, e, device=dev, dtype=DT)
            # fused_topk returns (tw, tid) through vllm ~0.10 and (tw, tid, token_expert_indices) from ~0.11 on.
            tw, tid, *_rest = fused_topk(x, gate, k, True)
            if _BACKEND == "flashinfer":
                if _FP8:
                    raise RuntimeError("--backend flashinfer supports bf16 only here "
                                       "(the served CUTLASS Unquantized path)")
                cutlass_fused_moe = _flashinfer_moe()
                tid_i = tid.to(torch.int)
                tw_f = tw.to(torch.float32)
                out = torch.empty_like(x)
                us = _time(lambda: cutlass_fused_moe(
                    input=x, token_selected_experts=tid_i, token_final_scales=tw_f,
                    fc1_expert_weights=w1, fc2_expert_weights=w2,
                    output_dtype=DT, quant_scales=[], output=out))
            elif qc is not None:
                us = _time(lambda: fused_experts(x, w1, w2, tw, tid, quant_config=qc))
            else:
                try:
                    us = _time(lambda: fused_experts(x, w1, w2, tw, tid, inplace=False))
                except TypeError:  # vllm >= 0.27 dropped the inplace kwarg
                    us = _time(lambda: fused_experts(x, w1, w2, tw, tid))
        except Exception as ex:  # noqa: BLE001
            print(f"  skip {name} tokens={m}: {type(ex).__name__}: {ex}", flush=True)
            continue
        rows.append({"tokens": m, "n_experts": e, "top_k": k, "intermediate": inter,
                     "hidden": hidden, "dtype_bytes": (1 if _FP8 else DT_BYTES),
                     "latency_us": round(us, 3)})
        print(f"  {name:16s} E={e:4d} k={k} inter={inter:6d} h={hidden:5d} "
              f"tok={m:5d}: {us:9.2f} us", flush=True)
    del w1, w2
    torch.cuda.empty_cache()
    return rows


def fit_affine(points) -> tuple[float, float] | None:
    """Least-squares (floor_us, eff_bw_gb_s) for latency = floor + bytes/eff_bw.
    Same model as kernel_composed.ElementwiseTable, so the output drops in."""
    pts = [(b, t) for b, t in points if b > 0 and t > 0]
    if len(pts) < 3:
        return None
    n = len(pts)
    sx = sum(b for b, _ in pts)
    sy = sum(t for _, t in pts)
    sxx = sum(b * b for b, _ in pts)
    sxy = sum(b * t for b, t in pts)
    den = n * sxx - sx * sx
    if den == 0:
        return None
    slope = (n * sxy - sx * sy) / den          # us per byte
    floor = (sy - slope * sx) / n
    if slope <= 0:
        return None
    return max(0.0, floor), 1e3 / slope        # us/byte -> GB/s


def sweep_routing(dev, e_default=128, k_default=8):
    """Affine fits for moe_topk / moe_permute + the moe_align launch floor.

    Byte counts MUST match kernel_composed/elementwise.py::_IO:
      moe_topk    (1, 0) -> elements * dtype * 1
      moe_permute (1, 1) -> elements * dtype * 2
    """
    out: dict[str, dict] = {}

    topk_pts, perm_pts = [], []
    for n in ROUTE_ELEMS:
        rows = max(1, n // e_default)
        try:
            logits = torch.randn(rows, e_default, device=dev, dtype=DT)
            t = _time(lambda: torch.topk(logits.float(), k_default, dim=-1))
            topk_pts.append((rows * e_default * DT_BYTES * 1, t))
        except Exception:  # noqa: BLE001
            pass
        try:
            hidden = 2048
            src = torch.randn(max(1, n // hidden), hidden, device=dev, dtype=DT)
            idx = torch.randint(0, src.shape[0], (src.shape[0],), device=dev)
            t = _time(lambda: torch.index_select(src, 0, idx))
            perm_pts.append((src.numel() * DT_BYTES * 2, t))
        except Exception:  # noqa: BLE001
            pass

    for key, pts in (("moe_topk", topk_pts), ("moe_permute", perm_pts)):
        fit = fit_affine(pts)
        if fit:
            out[key] = {"floor_us": round(fit[0], 4), "eff_bw_gb_s": round(fit[1], 1),
                        "n_points": len(pts)}

    # moe_align_block_size: launch-bound, no meaningful byte model -> report the median as a flat per-MoE-layer constant for the device YAML.
    aligns = []
    try:
        # Imported here: the module moves between vLLM versions, and a miss only skips this fit.
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size,
        )
        for m in (8, 64, 512):
            tid = torch.randint(0, e_default, (m, k_default), device=dev, dtype=torch.int32)
            aligns.append(_time(lambda: moe_align_block_size(tid, 128, e_default)))
    except Exception as ex:  # noqa: BLE001
        print(f"  moe_align_block_size unavailable ({type(ex).__name__}); "
              f"routing_launch_us not measured", flush=True)
    if aligns:
        out["_moe_align_launch_us"] = round(st.median(aligns), 3)
    return out


# --routing: 16 fixed prompt ids (below Kimi-K3's 163840 vocab; no tokenizer).
PROMPT_IDS = [1, 4321, 100, 65535, 2048, 777, 31415, 9,
              128000, 42, 5, 99991, 1234, 60000, 8, 163839]
# (submodule, "in" or "out", dump name): the stages the routing dump hooks.
STAGES = [
    ("routed_expert_down_proj", "out", "latent"),
    ("routed_expert_norm", "in", "routed"),
    ("routed_expert_norm", "out", "normed"),
    ("routed_expert_up_proj", "out", "up"),
    ("shared_experts", "out", "shared"),
]


def routing_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="One MoE block's input, output and routing decision.")
    add_output_args(ap)
    ap.add_argument("--routing", action="store_true", required=True)
    ap.add_argument("--model", required=True, help="checkpoint dir")
    ap.add_argument("--layer", type=int, default=1, help="decoder layer whose MoE block is dumped")
    ap.add_argument("--layers", type=int, default=None,
                    help="keep only the first N decoder layers (engine.layers)")
    ap.add_argument("--dtype", default="float32", help="torch dtype the model runs in")
    ap.add_argument("--kda-mode", default=None,
                    help="set self_attn.mode on every layer that has one (KDA prefill kernel)")
    ap.add_argument("--prompt-ids", default=",".join(map(str, PROMPT_IDS)),
                    help="comma-separated token ids fed in one prefill")
    return ap


def routing_model(args):
    """The model on the visible GPUs in args.dtype, cut to args.layers; MXFP4 experts are not kept packed, so a checkpoint must fit decompressed to bf16."""
    from transformers import AutoConfig, AutoModelForCausalLM  # only --routing needs transformers
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if args.layers:
        config.num_hidden_layers = args.layers
        if isinstance(getattr(config, "layer_types", None), list):
            config.layer_types = config.layer_types[:args.layers]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=config, torch_dtype=getattr(torch, args.dtype),
        device_map="auto", trust_remote_code=True)
    model.eval()
    if args.kda_mode:
        for layer in model.model.layers:
            if hasattr(layer.self_attn, "mode"):
                layer.self_attn.mode = args.kda_mode
    return model, config


def routing_tensors(args) -> dict:
    """{name: tensor} of the block's input, output, routing and stages. The block is the layer's block_sparse_moe, else its mlp; a router returning (topk_idx, topk_weight) is dumped as those two, one returning logits as router_logits."""
    torch.manual_seed(33377335)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model, config = routing_model(args)
    ids = [int(t) for t in args.prompt_ids.split(",")]
    if max(ids) >= config.vocab_size:
        raise SystemExit(f"prompt id {max(ids)} >= vocab_size {config.vocab_size}; "
                         "pass --prompt-ids")
    layer = model.model.layers[args.layer]
    moe = getattr(layer, "block_sparse_moe", None) or layer.mlp
    if not hasattr(moe, "gate"):
        raise SystemExit(f"layer {args.layer} has no MoE block with a gate")
    cap = {}

    def keep(name, side):
        def hook(_m, inputs, output=None):
            t = inputs[0] if side == "in" else output
            if isinstance(t, tuple):
                t = t[0]
            cap[name] = t.detach().float().cpu()
        return hook

    def routing(_m, _inputs, output):
        if isinstance(output, tuple):
            cap["topk_idx"] = output[0].detach().cpu()
            cap["topk_weight"] = output[1].detach().float().cpu()
        else:
            cap["router_logits"] = output.detach().float().cpu()

    handles = [moe.register_forward_pre_hook(keep("in", "in")),
               moe.register_forward_hook(keep("out", "out")),
               moe.gate.register_forward_hook(routing)]
    for attr, side, name in STAGES:
        sub = getattr(moe, attr, None)
        if sub is not None:
            register = sub.register_forward_pre_hook if side == "in" else sub.register_forward_hook
            handles.append(register(keep(name, side)))
    device = model.get_input_embeddings().weight.device
    with torch.no_grad():
        model(input_ids=torch.tensor([ids], device=device))
    for h in handles:
        h.remove()
    return {f"moe{args.layer}_{k}": (v[0] if v.dim() == 3 else v).contiguous()
            for k, v in cap.items()}


def routing_main(argv) -> None:
    args = routing_parser().parse_args(argv)
    from safetensors.torch import save_file  # only --routing needs safetensors
    tensors = routing_tensors(args)
    out = out_root(args, "moe_routing") / f"{Path(args.model).name}_layer{args.layer}.safetensors"
    out.parent.mkdir(parents=True, exist_ok=True)
    for k, v in tensors.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")
    save_file(tensors, str(out))
    print(f"wrote {out}")


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if "--routing" in argv:
        return routing_main(argv)
    # No abbreviations: export reads --append from the argv as written.
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_output_args(ap)
    ap.add_argument("--geometries", default=",".join(GEOMETRIES),
                    help="comma-separated model keys to sweep")
    ap.add_argument("--geometry-spec", default=None,
                    help="explicit geometries E:top_k:intermediate:hidden (comma-separated), "
                         "for shapes that are not a model's tp shard -- notably an "
                         "EXPERT-PARALLEL rank, which owns n_experts/ep WHOLE experts at "
                         "the FULL d_ff (e.g. 16:8:1536:4096 = Qwen3-235B at ep8). Swept "
                         "verbatim; --tp is ignored for these.")
    ap.add_argument("--tp", default="1,2,4",
                    help="tp shards to sweep (each divides `intermediate`)")
    ap.add_argument("--tokens", default=None, help="comma-separated token axis override")
    ap.add_argument("--max-mem-gb", type=float, default=40.0)
    ap.add_argument("--skip-routing", action="store_true")
    ap.add_argument("--dtype", choices=("bf16", "fp8"), default="bf16",
                    help="bf16: vLLM's unquantized fused_experts. fp8: the SAME entry "
                         "point with a block-scaled FP8 quant_config (128x128 weight "
                         "blocks), i.e. what a `quant_method: fp8` checkpoint runs. "
                         "Writes {gpu}_fp8.csv with dtype_bytes=1.")
    ap.add_argument("--append", action="store_true",
                    help="at export, add this run's rows over earlier ok runs' tables; "
                         "without it the export drops the rows of runs that finished before it")
    ap.add_argument("--graph", action="store_true",
                    help="CUDA-graph-replay timing -> graph/moe/. Faithful for vLLM's "
                         "CUDA-graphed MoE DECODE (the eager rationale below only "
                         "holds for eager execution); slight underprice of small "
                         "eager prefill chunks, negligible at real chunk sizes.")
    ap.add_argument("--backend", choices=("triton", "flashinfer"), default="triton",
                    help="triton: vLLM's fused_experts (Triton grouped GEMM). "
                         "flashinfer: FlashInfer's CUTLASS grouped kernel -- what "
                         "vLLM >= 0.28 actually serves for unquantized bf16 MoE.")
    a = ap.parse_args(argv)
    global _GRAPH, _FP8, _BACKEND
    _GRAPH = a.graph
    _FP8 = (a.dtype == "fp8")
    _BACKEND = a.backend
    if _FP8:
        # fused_experts' FP8 path builds a CustomOp internally, which needs a live vLLM config (same reason as gemm's Fp8Op); only --dtype fp8 imports it.
        from vllm.config import VllmConfig, set_current_vllm_config
        set_current_vllm_config(VllmConfig()).__enter__()

    dev = torch.device("cuda")
    tokens_axis = ([int(x) for x in a.tokens.split(",")] if a.tokens else TOKENS)
    tps = [int(x) for x in a.tp.split(",")]
    print(f"[moe] {torch.cuda.get_device_name(0)} | dtype={a.dtype} "
          f"| geometries={a.geometries} | tp={tps} | {len(tokens_axis)} token points",
          flush=True)

    rows = []
    seen: set[tuple[int, int, int, int]] = set()
    for spec in (a.geometry_spec or "").split(","):
        if not spec.strip():
            continue
        e, k, inter, hidden = (int(x) for x in spec.strip().split(":"))
        seen.add((e, k, inter, hidden))
        rows += sweep_expert_kernel(f"ep/{spec.strip()}", e, k, inter, hidden,
                                    tokens_axis, dev, a.max_mem_gb)
    for key in ("" if a.geometry_spec else a.geometries).split(","):
        key = key.strip()
        if not key:
            continue
        if key not in GEOMETRIES:
            print(f"  unknown geometry {key!r}, skipping", flush=True)
            continue
        e, k, inter, hidden = GEOMETRIES[key]
        for tp in tps:
            if inter % tp:
                continue
            shard = inter // tp
            if (e, k, shard, hidden) in seen:      # tp shards can collide across models
                continue
            seen.add((e, k, shard, hidden))
            rows += sweep_expert_kernel(f"{key}/tp{tp}", e, k, shard, hidden,
                                        tokens_axis, dev, a.max_mem_gb)

    # Default eager: the grouped kernel's LAUNCH cost is the term the roofline misses -- but that is only faithful for EAGER execution. vLLM CUDA-graphs MoE decode, so --graph (dispatch-free, graph/) is the decode-faithful method;
    # H200 GT showed the eager grid overprices graphed decode 2.3x.
    timing = "graph_replay" if _GRAPH else "eager"
    table = f"moe/{a.gpu_label}{'' if a.dtype == 'bf16' else '_' + a.dtype}.csv"
    for r in rows:
        emit(ProbeRow("moe", table, {c: r[c] for c in KEY},
                      {"dtype_bytes": r["dtype_bytes"], "latency_us": r["latency_us"]}, timing))
    print(f"[moe] {len(rows)} rows of {table}{', --append' if a.append else ''}", flush=True)

    if not a.skip_routing and a.dtype == "bf16":
        # Routing (topk / permute / align) is dtype-agnostic -- it moves indices and bf16 hidden states either way -- so an fp8 run must not rewrite the fit.
        routing = sweep_routing(dev)
        for name, value in routing.items():
            emit(ProbeRow("moe", f"moe/{a.gpu_label}_routing.json", {"entry": name}, {"value": value}, timing))
        print(f"[moe] {len(routing)} entries of moe/{a.gpu_label}_routing.json", flush=True)
        if "_moe_align_launch_us" in routing:
            print(f"[moe] set `moe.routing_launch_us: "
                  f"{routing['_moe_align_launch_us']}` in the {a.gpu_label} device YAML",
                  flush=True)


if __name__ == "__main__":
    run_recorded(main, "moe_routing" if "--routing" in sys.argv else "moe")
