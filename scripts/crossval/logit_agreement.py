"""Logit-level agreement between vLLM and the HF reference, teacher-forced.

greedy_agreement.py compares generated text, which cannot separate a numerics
bug from a legitimate bf16 near-tie flip once the chains diverge. This check
is teacher-forced instead: both implementations score the SAME fixed token
sequence, so every position is an independent trial and nothing cascades. For
each position it compares the argmax prediction of vLLM (prompt_logprobs)
against the HF transformers forward pass, and the log-probability the two
assign to the true next token.

What a failure means: a low argmax agreement, or disagreements at positions
where the HF top-2 logit gap is large, points at a kernel or weight-loading
defect in the serving engine; disagreements confined to near-ties (small
top-2 gap) are bf16 rounding, not defects. Both sides run bfloat16 so the
comparison has one dtype.

This validates the BASELINE lane of the crossval (vLLM against the canonical
implementation). The QuettaServe engine emits text only, so its logit-level
check needs an engine-side dump and stays with the engine team.

    python3 logit_agreement.py --model /path/to/Llama-3.1-8B-Instruct
"""

import argparse
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

TEXTS = [
    "The industrial revolution began in Britain in the late eighteenth "
    "century and transformed manufacturing from hand production to machines. "
    "New chemical and iron production processes appeared, water power gave "
    "way to steam, and the factory system concentrated labour in cities. "
    "Textiles were the dominant industry in terms of employment, value of "
    "output and capital invested.",
    "Photosynthesis is the process by which green plants convert sunlight "
    "into chemical energy. Light is absorbed by chlorophyll in the "
    "chloroplasts, water is split to release oxygen, and carbon dioxide is "
    "reduced to sugars in the Calvin cycle. The overall reaction consumes "
    "six molecules of carbon dioxide and six of water to produce one glucose "
    "molecule and six of oxygen.",
    "A binary search algorithm finds a target value in a sorted array by "
    "repeatedly halving the search interval. It compares the target with "
    "the middle element: if they are unequal, the half in which the target "
    "cannot lie is eliminated and the search continues on the remaining "
    "half until the target is found or the interval is empty. Its running "
    "time is logarithmic in the length of the array.",
]


def hf_reference(model_dir, token_ids_per_text):
    """Per text: argmax prediction, top-2 gap and true-token logprob at each
    position, from a single teacher-forced forward pass."""
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16
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


def vllm_scores(model_dir, token_ids_per_text, gpu_util):
    """Per text: vLLM's argmax and true-token logprob at each prompt position."""
    llm = LLM(model=model_dir, dtype="bfloat16", gpu_memory_utilization=gpu_util,
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
    ap.add_argument("--model", required=True, help="HF weights dir")
    ap.add_argument("--gpu-util", type=float, default=0.5)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids_per_text = [tokenizer(t)["input_ids"] for t in TEXTS]

    hf = hf_reference(args.model, ids_per_text)
    vl = vllm_scores(args.model, ids_per_text, args.gpu_util)

    total = agree = 0
    near_tie = far = 0
    deltas = []
    for ids, (preds, gaps, true_lp), (v_argmax, v_true_lp) in zip(ids_per_text, hf, vl):
        for i in range(len(ids) - 1):
            pos = i + 1
            if pos not in v_argmax:
                continue
            total += 1
            if v_argmax[pos] == preds[i]:
                agree += 1
            elif gaps[i] < 0.1:
                near_tie += 1
                print(f"NEAR_TIE pos={pos} hf_gap={gaps[i]:.4f}")
            else:
                far += 1
                print(f"DISAGREE pos={pos} hf_gap={gaps[i]:.4f} "
                      f"hf={preds[i]} vllm={v_argmax[pos]}")
            if pos in v_true_lp:
                deltas.append(abs(true_lp[i] - v_true_lp[pos]))

    deltas.sort()
    print(f"SUMMARY positions={total} argmax_agree={agree} "
          f"({100.0 * agree / total:.2f}%) near_tie_flips={near_tie} "
          f"hard_disagreements={far}")
    print(f"SUMMARY true_token_logprob_delta mean={sum(deltas)/len(deltas):.5f} "
          f"p95={deltas[int(0.95 * len(deltas))]:.5f} max={deltas[-1]:.5f}")


if __name__ == "__main__":
    main()
