"""Teacher-forced logit agreement between vLLM and the transformers reference.

Both score the same fixed token sequences in bfloat16, so every position is
an independent trial: argmax match per position plus the log probability
each assigns the true next token, with the reference top-2 gap printed on
any disagreement so near-ties separate from real defects. Validates the
baseline; the engine binary emits text only.

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
    "The transmission control protocol provides reliable, ordered delivery "
    "of a byte stream between applications. A connection is established "
    "with a three way handshake, data segments carry sequence numbers so "
    "the receiver can reassemble them in order, and acknowledgements with "
    "retransmission timers recover anything the network drops. Congestion "
    "control adjusts the sending rate to what the path can carry.",
    "To make a simple bread dough, combine flour, water, salt and yeast, "
    "then knead until the surface turns smooth and elastic. Let it rise "
    "until doubled, fold it once to redistribute the gas, shape the loaf "
    "and let it rise again. Bake in a hot oven until the crust browns and "
    "the inside reaches temperature, then cool it on a rack before "
    "slicing.",
    "The planets of the solar system divide into two groups. The inner "
    "four are small rocky bodies with thin atmospheres or none at all, "
    "while the outer four are giants composed mostly of hydrogen, helium "
    "and ices. Between the two groups lies the asteroid belt, and beyond "
    "the giants a scattered disc of icy objects marks the boundary of the "
    "planetary region.",
    "Inflation measures how fast the general level of prices rises over "
    "time. Central banks respond by adjusting interest rates: higher rates "
    "make borrowing dearer, which cools spending and investment, while "
    "lower rates do the opposite. The lag between a rate change and its "
    "effect on prices is long and variable, which is what makes the job "
    "difficult.",
    "The referee blew the whistle and the match restarted with a short "
    "corner. The defender cleared the first ball, but the winger returned "
    "it low across the face of goal and the striker arrived a step ahead "
    "of his marker to turn it in at the near post. The stadium erupted; "
    "the away end went silent.",
    "A hash table stores key value pairs in an array indexed by a hash "
    "function applied to the key. Collisions, where two keys map to the "
    "same slot, are handled either by chaining entries in a list or by "
    "probing for the next open slot. With a good hash function and a load "
    "factor kept below one, insertion and lookup take constant time on "
    "average.",
    "Cells store their genetic instructions in DNA, which is transcribed "
    "into messenger RNA in the nucleus. The message travels to a ribosome, "
    "where transfer RNA molecules deliver amino acids matching each three "
    "letter codon, and the growing chain folds into a working protein. "
    "Errors in copying are rare because polymerases proofread as they go.",
]


def transformers_reference(model_dir, token_ids_per_text):
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
    ap.add_argument("--model", required=True, help="model weights dir")
    ap.add_argument("--gpu-util", type=float, default=0.5)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids_per_text = [tokenizer(t)["input_ids"] for t in TEXTS]

    reference = transformers_reference(args.model, ids_per_text)
    vllm_result = vllm_scores(args.model, ids_per_text, args.gpu_util)

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

    deltas.sort()
    print(f"SUMMARY positions={total} argmax_agree={agree} "
          f"({100.0 * agree / total:.2f}%) near_tie_flips={near_tie} "
          f"hard_disagreements={far}")
    print(f"SUMMARY true_token_logprob_delta mean={sum(deltas)/len(deltas):.5f} "
          f"p95={deltas[int(0.95 * len(deltas))]:.5f} max={deltas[-1]:.5f}")


if __name__ == "__main__":
    main()
