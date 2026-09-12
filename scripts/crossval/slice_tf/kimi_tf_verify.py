"""Teacher-forced HF bf16 envelope on the 13-layer real slice, for the vLLM
statistical verification (PART 1 of the slice bench).

A free-running greedy chain cannot answer a numerics question on this slice:
the untrained truncation amplifies bf16 rounding into router flips, so every
bf16 kernel — including torch's own two KDA prefill kernels — produces a
different chain after a few steps. The criterion that IS attainable is
teacher-forced: feed every engine the SAME 56-token prompt (the 16 fixed
prompt ids + the 40 tokens of the fp32-recurrent reference chain, which ARE
the fp32 argmaxes at those positions) and compare per-position argmaxes.

This script measures the NOISE FLOOR of that comparison: the HF reference
against itself at bf16, chunk-mode vs fused_recurrent-mode KDA prefill, on the
identical concatenated prompt. vLLM is bf16, so it cannot beat this floor;
the acceptance criterion is that it sits inside it (disagreement rate with the
fp32 chain <= the worse HF-bf16 kernel's rate).

    KIMI_SLICE_MODEL=/data35/kevinlau/kimi-slice/truncated \
    KIMI_REF_DTYPE=bf16 \
    PYTHONPATH=/data35/kevinlau/pylibs/fla \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python kimi_tf_verify.py /data35/kevinlau/kimi-ref-slice-f32-long.safetensors out.json

One model load, both KDA modes, each mode run twice (in-process determinism
check). Output JSON carries, for each mode, the per-position argmax and the
top-2 logit gap at teacher-forced positions 16..55 (0-indexed; predictions of
forced tokens t0..t39), plus agreement counts vs the fp32 chain and
cross-kernel.
"""

import json
import os
import sys

import torch
from safetensors import safe_open

import kimi_ref  # PROMPT_IDS — shared with every other harness
import kimi_ref_slice

N_FORCED = 40
MODES = ("fused_recurrent", "chunk")


def chain_from_dump(path: str, n: int) -> list[int]:
    """decode{s}_token is the INPUT to step s == the fp32 argmax at teacher-
    forced position 16+s of the concatenated prompt."""
    out = []
    with safe_open(path, framework="pt") as f:
        for s in range(n):
            out.append(int(f.get_tensor(f"decode{s}_token").reshape(-1)[0]))
    return out


def fp32_gaps(path: str, n: int) -> list[float]:
    """Top-2 logit gap of the fp32 reference at each teacher-forced position.
    Prediction of t0 comes from prefill_logits[-1]; of ts (s>=1) from
    decode{s-1}_logits."""
    gaps = []
    with safe_open(path, framework="pt") as f:
        row = f.get_tensor("prefill_logits")[-1].float()
        for s in range(n):
            v = row.topk(2).values
            gaps.append(float(v[0] - v[1]))
            if s + 1 < n:
                row = f.get_tensor(f"decode{s}_logits").float()
    return gaps


def run_mode(model, config, mdl_mod, ids_list, mode: str):
    for layer in model.model.layers:
        if hasattr(layer.self_attn, "mode"):
            layer.self_attn.mode = mode
    ids = torch.tensor([ids_list], dtype=torch.long, device="cuda:0")
    cache = mdl_mod.KimiDynamicCache(config=config)
    with torch.no_grad():
        out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    logits = out.logits[0].detach().float().cpu()  # [56, vocab]
    # Row p predicts token p+1: predictions of t0..t39 are rows 15..54.
    rows = logits[15 : 15 + N_FORCED]
    top2 = rows.topk(2, dim=-1)
    preds = [int(t) for t in top2.indices[:, 0]]
    gaps = [float(g) for g in (top2.values[:, 0] - top2.values[:, 1])]
    return preds, gaps


def main() -> None:
    ref_path, out_path = sys.argv[1], sys.argv[2]
    # bf16 is the envelope proper (vLLM's dtype). f32 is run as the ANCHOR
    # SANITY CHECK: it separates "bf16 rounding" from "teacher-forced prefill
    # is a different code path than recurrent decode". The fp32 chain was
    # produced by prefill-16 + 40 recurrent decode steps; re-deriving it from a
    # single 56-token prefill exercises the chunk/recurrent prefill kernels
    # instead. If fp32 teacher-forced does NOT reproduce the chain, the
    # disagreements at those positions are path artefacts, not engine defects,
    # and every engine (vLLM included) inherits them.
    dtype = os.environ.get("KIMI_REF_DTYPE", "bf16")
    assert dtype in ("bf16", "f32", "fp32"), f"unsupported KIMI_REF_DTYPE {dtype!r}"
    print(f"teacher-forced envelope at dtype={dtype}", flush=True)
    torch.manual_seed(33377335)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    forced = chain_from_dump(ref_path, N_FORCED)
    gaps32 = fp32_gaps(ref_path, N_FORCED)
    ids_list = list(kimi_ref.PROMPT_IDS) + forced
    print(f"concatenated prompt: {len(ids_list)} tokens "
          f"({len(kimi_ref.PROMPT_IDS)} prompt + {N_FORCED} forced)", flush=True)

    model, config, mdl_mod = kimi_ref_slice.build_model(
        os.environ["KIMI_SLICE_MODEL"], os.environ.get("TMPDIR", "/tmp")
    )

    result = {
        "dtype": dtype,
        "prompt_ids": list(kimi_ref.PROMPT_IDS),
        "forced_tokens": forced,
        "fp32_top2_gap": gaps32,
        "modes": {},
    }
    for mode in MODES:
        preds1, gaps1 = run_mode(model, config, mdl_mod, ids_list, mode)
        preds2, _ = run_mode(model, config, mdl_mod, ids_list, mode)
        deterministic = preds1 == preds2
        agree = sum(1 for p, t in zip(preds1, forced) if p == t)
        dis = [s for s, (p, t) in enumerate(zip(preds1, forced)) if p != t]
        print(f"[{mode}] agree with fp32 chain: {agree}/{N_FORCED}, "
              f"deterministic: {deterministic}", flush=True)
        for s in dis:
            print(f"  step {s}: {dtype} {preds1[s]} vs fp32-chain {forced[s]} "
                  f"(fp32-chain top-2 gap {gaps32[s]:.4f}, "
                  f"{dtype} top-2 gap {gaps1[s]:.4f})")
        result["modes"][mode] = {
            "pred": preds1,
            "top2_gap": gaps1,
            "agree_with_fp32": agree,
            "disagree_steps": dis,
            "deterministic_rerun": deterministic,
        }

    a = result["modes"][MODES[0]]["pred"]
    b = result["modes"][MODES[1]]["pred"]
    cross = sum(1 for x, y in zip(a, b) if x == y)
    result["cross_kernel_agree"] = cross
    print(f"[cross] recurrent-vs-chunk {dtype} agree: {cross}/{N_FORCED}")

    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
