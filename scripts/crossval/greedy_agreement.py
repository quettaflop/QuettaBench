"""Greedy-agreement check: a QuettaServe engine binary vs vLLM on shared prompts.

Both engines get the same prompt strings; vLLM decodes greedy (temperature 0)
and the engine binary decodes at temperature 0.01, a near-greedy proxy because
its temperature=0 path divides logits by zero and emits token 0 forever. For
each prompt the two continuations are printed side by side and re-tokenized,
and the length of the common leading token run is reported with a summary.

Known caveat that caps what the token metric means today: the demo binary
hardcodes BOS id 1 (llama/src/main.rs), the Llama 2 convention. On Llama 3
that id is a literal double quote, so the engine's model input carries a
spurious leading quote while vLLM's carries the true BOS. Content-level
agreement is still meaningful; exact token divergence is dominated by that
input difference until the BOS is fixed. The binary also emits text only, so
logit distributions cannot be compared without an engine change.

    python3 greedy_agreement.py --qs-bin /path/to/llama --model /path/to/weights
"""

import argparse
import os
import statistics
import subprocess

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

PROMPTS = [
    "The capital of France is",
    "The first three prime numbers are",
    "The chemical symbol for gold is",
    "The opposite of hot is",
    "The largest planet in the solar system is",
    "Two plus two equals",
    "The author of Romeo and Juliet is",
    "The square root of 64 is",
    "The currency of Japan is the",
    "The freezing point of water in Celsius is",
    "The past tense of run is",
    "The number of days in a week is",
]


def engine_continuation(bin_path, model, prompt, n, temperature, gpu):
    """Run the engine binary once; return the text it generated after the prompt."""
    env = dict(os.environ)
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    result = subprocess.run(
        [bin_path, "--temperature", str(temperature), "--sample-len", str(n),
         "--kind", "bfloat16", "--tokenizer", f"{model}/tokenizer.json",
         "--model", model, "--prompt", prompt],
        capture_output=True, text=True, timeout=420, env=env,
    )
    out = result.stdout.strip()
    at = out.find(prompt)
    return out[at + len(prompt):] if at >= 0 else out


def common_prefix_tokens(tokenizer, a, b):
    """Leading token agreement between two continuations, re-tokenized symmetrically."""
    a_ids = tokenizer.encode(a, add_special_tokens=False)
    b_ids = tokenizer.encode(b, add_special_tokens=False)
    n = 0
    for x, y in zip(a_ids, b_ids):
        if x != y:
            break
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qs-bin", required=True, help="QuettaServe llama binary")
    ap.add_argument("--model", required=True, help="HF weights dir")
    ap.add_argument("--n", type=int, default=20, help="tokens to generate")
    ap.add_argument("--qs-temperature", type=float, default=0.01)
    ap.add_argument("--qs-gpu", default="", help="CUDA device for the binary")
    ap.add_argument("--gpu-util", type=float, default=0.5)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_util, enforce_eager=True)
    greedy = SamplingParams(temperature=0, max_tokens=args.n)
    vllm_texts = [o.outputs[0].text
                  for o in llm.generate(PROMPTS, greedy, use_tqdm=False)]

    prefixes = []
    for prompt, vllm_text in zip(PROMPTS, vllm_texts):
        ours = engine_continuation(args.qs_bin, args.model, prompt, args.n,
                                   args.qs_temperature, args.qs_gpu)
        agree = common_prefix_tokens(tokenizer, vllm_text, ours)
        prefixes.append(agree)
        print(f"AGREE {agree:2d} tokens | {prompt}")
        print(f"  vllm: {vllm_text!r}")
        print(f"  ours: {ours!r}")

    print(f"SUMMARY prompts={len(prefixes)} "
          f"mean_prefix={statistics.mean(prefixes):.1f} "
          f"median_prefix={statistics.median(prefixes)} "
          f"agree4plus={sum(1 for p in prefixes if p >= 4)}/{len(prefixes)}")


if __name__ == "__main__":
    main()
