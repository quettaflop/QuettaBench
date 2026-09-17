"""Teacher-forced logit agreement between vLLM and the transformers reference.

Both score the same fixed token sequences in the workload's dtype from
workloads.json, the numeric path the baseline measures, so every position is
an independent trial: argmax match per position plus the log probability
each assigns the true next token, with the reference top-2 gap printed on
any disagreement so near-ties separate from real defects. Validates the
baseline; the engine binary emits text only. Texts live in texts.txt, one
per line.

    python3 logit_agreement.py --model /path/to/Llama-3.1-8B-Instruct
"""

import argparse
import gc
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent))
import xval_config as _xval_config

TEXTS = [
    line.strip()
    for line in Path(__file__).with_name("texts.txt").read_text().splitlines()
    if line.strip()
]


def transformers_reference(model_dir, token_ids_per_text, dtype):
    """Per text: argmax prediction, top-2 gap and true-token logprob at each
    position, from a single teacher-forced forward pass."""
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=getattr(torch, dtype)
    ).to("cuda:0")
    model.eval()
    out = []
    with torch.no_grad():
        for ids in token_ids_per_text:
            t = torch.tensor([ids], device="cuda:0")
            logits = model(t).logits[0].float()
            logprobs = torch.log_softmax(logits, dim=-1)
            top2 = logits.topk(2, dim=-1)
            preds = top2.indices[:, 0].tolist()
            gaps = (top2.values[:, 0] - top2.values[:, 1]).tolist()
            true_lp = [float(logprobs[i, ids[i + 1]]) for i in range(len(ids) - 1)]
            out.append((preds, gaps, true_lp))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


def vllm_scores(model_dir, token_ids_per_text, gpu_util, dtype):
    """Per text: vLLM's argmax and true-token logprob at each prompt position."""
    llm = LLM(model=model_dir, dtype=dtype, gpu_memory_utilization=gpu_util,
              max_model_len=4096, enforce_eager=True, enable_prefix_caching=False)
    params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=2)
    outputs = llm.generate(
        [{"prompt_token_ids": ids} for ids in token_ids_per_text], params, use_tqdm=False
    )
    out = []
    for ids, o in zip(token_ids_per_text, outputs):
        argmax, true_lp = {}, {}
        for pos, entry in enumerate(o.prompt_logprobs or []):
            if not entry:
                continue
            for token_id, lp in entry.items():
                if lp.rank == 1:
                    argmax[pos] = token_id
                if token_id == ids[pos]:
                    true_lp[pos] = lp.logprob
        out.append((argmax, true_lp))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model weights dir")
    ap.add_argument("--crate", default="llama", help="workload whose dtype to validate")
    ap.add_argument("--gpu-util", type=float, default=0.5)
    args = ap.parse_args()
    dtype = _xval_config.workloads()[args.crate]["dtype"]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids_per_text = [tokenizer(t)["input_ids"] for t in TEXTS]

    reference = transformers_reference(args.model, ids_per_text, dtype)
    vllm_result = vllm_scores(args.model, ids_per_text, args.gpu_util, dtype)

    total = agree = 0
    near_tie = far = 0
    deltas = []
    for ids, (preds, gaps, true_lp), (v_argmax, v_true_lp) in zip(ids_per_text, reference, vllm_result):
        for i in range(len(ids) - 1):
            pos = i + 1
            if pos not in v_argmax:
                continue
            total += 1
            if v_argmax[pos] == preds[i]:
                agree += 1
            elif gaps[i] < 0.1:
                near_tie += 1
                print(f"NEAR_TIE pos={pos} reference_gap={gaps[i]:.4f}")
            else:
                far += 1
                print(f"DISAGREE pos={pos} reference_gap={gaps[i]:.4f} "
                      f"reference={preds[i]} vllm={v_argmax[pos]}")
            if pos in v_true_lp:
                deltas.append(abs(true_lp[i] - v_true_lp[pos]))

    if not total or not deltas:
        raise SystemExit(
            "SUMMARY empty: vLLM returned no scored positions (no prompt logprobs)"
        )
    deltas.sort()
    print(f"SUMMARY positions={total} argmax_agree={agree} "
          f"({100.0 * agree / total:.2f}%) near_tie_flips={near_tie} "
          f"hard_disagreements={far}")
    print(f"SUMMARY true_token_logprob_delta mean={sum(deltas)/len(deltas):.5f} "
          f"p95={deltas[int(0.95 * len(deltas))]:.5f} max={deltas[-1]:.5f}")


if __name__ == "__main__":
    main()
