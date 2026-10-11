#!/usr/bin/env python3
"""One MoE block's exact input, output and routing decision, dumped so an
engine can run its own block on the identical input and diff routing,
dispatch and combine independently.

Ported from the slice_tf harness's moe_routing_probe.py (QuettaServe
docs/scripts/kimi_moe_probe.py), which hooked layer 1 of the Kimi-K3
13-layer slice. The model is built by transformers here, cut to --layers
the way engine.layers cuts the reference (slice_truncate.py's config edit),
in --dtype with TF32 off. The block is the layer's `block_sparse_moe`, else
its `mlp`; its router is `gate`. A router returning (topk_idx, topk_weight)
is dumped as those two; one returning logits is dumped as router_logits,
since the top-k and its weights are the block's own arithmetic. The stage
hooks pin a mismatch to the latent down-projection, the routed norm, the
up-projection or the shared expert, on blocks that have them.

ref_slice.py's loader is not ported: it kept the Kimi slice's MXFP4 experts
packed and spread the layers over four GPUs with Kimi-only shims, because
transformers decompresses those experts to bf16 at load (about 709 GiB for
13 layers). A checkpoint that does not fit decompressed needs that loader.

  CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m quettabench.probes.moe_routing \\
      --gpu-label H100 --model <dir> --layers 13 --layer 1 --dtype float32 \\
      --kda-mode fused_recurrent

writes results/probes/<gpu>/moe_routing/<model>_layer<L>.safetensors.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM

from quettabench.probes.base import add_output_args, out_root, run_recorded

# ref_full.py's 16 fixed prompt ids (below Kimi-K3's 163840 vocab; no tokenizer).
PROMPT_IDS = [1, 4321, 100, 65535, 2048, 777, 31415, 9,
              128000, 42, 5, 99991, 1234, 60000, 8, 163839]
# (submodule, "in" or "out", dump name): the stages moe_routing_probe.py hooked.
STAGES = [
    ("routed_expert_down_proj", "out", "latent"),
    ("routed_expert_norm", "in", "routed"),
    ("routed_expert_norm", "out", "normed"),
    ("routed_expert_up_proj", "out", "up"),
    ("shared_experts", "out", "shared"),
]


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_output_args(ap)
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


def build(args):
    """The model on the visible GPUs in args.dtype, cut to args.layers."""
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


def probe(args) -> dict:
    """{name: tensor} of the block's input, output, routing and stages."""
    torch.manual_seed(33377335)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model, config = build(args)
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


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    tensors = probe(args)
    out = out_root(args) / "moe_routing" / f"{Path(args.model).name}_layer{args.layer}.safetensors"
    out.parent.mkdir(parents=True, exist_ok=True)
    for k, v in tensors.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")
    save_file(tensors, str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    run_recorded(main, "moe_routing")
