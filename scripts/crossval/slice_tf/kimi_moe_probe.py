"""Isolate layer 1's MoE block on the real 13-layer slice, for bug hunting.

Dumps the reference's EXACT block input, block output and routing decision
(topk_idx / topk_weight) for the first MoE layer, so the Rust side can run its
own `KimiSparseMoeBlock` on the identical input and diff the three stages
independently: routing, dispatch, combine.

    KIMI_SLICE_MODEL=/data35/kevinlau/kimi-slice/truncated \
    KIMI_REF_DTYPE=f32 KIMI_REF_KDA_MODE=fused_recurrent \
    PYTHONPATH=/data35/kevinlau/pylibs/fla CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python kimi_moe_probe.py <out.safetensors>

Model construction, weight loading and pipeline hooks are kimi_ref_slice.py's
own (imported, not copied), so the probe cannot drift from the dump harness.
"""

import os
import sys

import torch
from safetensors.torch import save_file

import kimi_ref
import kimi_ref_slice as krs


def main() -> None:
    out_path = sys.argv[1]
    torch.manual_seed(33377335)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    model, config, mdl_mod = krs.build_model(
        os.environ["KIMI_SLICE_MODEL"], os.environ.get("TMPDIR", "/tmp")
    )
    kda_mode = os.environ.get("KIMI_REF_KDA_MODE")
    if kda_mode:
        for layer in model.model.layers:
            if hasattr(layer.self_attn, "mode"):
                layer.self_attn.mode = kda_mode
        print(f"KDA prefill mode forced to {kda_mode!r}")

    cap = {}
    moe = model.model.layers[1].block_sparse_moe

    def pre(_m, args):
        cap["moe1_in"] = args[0].detach().float().cpu()

    def post(_m, _a, out):
        cap["moe1_out"] = out.detach().float().cpu()

    def gate_post(_m, _a, out):
        idx, w = out
        cap["moe1_topk_idx"] = idx.detach().cpu()
        cap["moe1_topk_weight"] = w.detach().float().cpu()

    def keep_out(name):
        def hook(_m, _a, out):
            cap[name] = out.detach().float().cpu()
        return hook

    def keep_in(name):
        def hook(_m, args):
            cap[name] = args[0].detach().float().cpu()
        return hook

    handles = [
        moe.register_forward_pre_hook(pre),
        moe.register_forward_hook(post),
        moe.gate.register_forward_hook(gate_post),
        # Stage-by-stage internals, so a mismatch can be pinned to routing,
        # dispatch/combine, norm, up-projection or the shared expert.
        moe.routed_expert_down_proj.register_forward_hook(keep_out("moe1_latent")),
        moe.routed_expert_norm.register_forward_pre_hook(keep_in("moe1_routed")),
        moe.routed_expert_norm.register_forward_hook(keep_out("moe1_normed")),
        moe.routed_expert_up_proj.register_forward_hook(keep_out("moe1_up")),
        moe.shared_experts.register_forward_hook(keep_out("moe1_shared")),
    ]

    ids = torch.tensor([kimi_ref.PROMPT_IDS], dtype=torch.long, device="cuda:0")
    cache = mdl_mod.KimiDynamicCache(config=config)
    with torch.no_grad():
        model(input_ids=ids, past_key_values=cache, use_cache=True)
    for h in handles:
        h.remove()

    tensors = {k: (v[0] if v.dim() == 3 else v).contiguous() for k, v in cap.items()}
    for k, v in tensors.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")
    save_file(tensors, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
