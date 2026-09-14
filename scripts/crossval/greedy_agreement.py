"""Greedy agreement between an engine binary and vLLM on shared prompts.

Prints both continuations and the leading-token agreement per prompt. The
engine side runs at temperature 0.01 (its temperature 0 path divides by
zero) and its hardcoded Llama 2 start token skews exact token divergence on
Llama 3, so content-level agreement is the meaningful signal. Prompts live
in prompts.txt, one per line.

    python3 greedy_agreement.py --qs-bin /path/to/llama --model /path/to/weights
"""

import argparse
import gc
import os
import statistics
import subprocess
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

PROMPTS = [
    line.strip()
    for line in Path(__file__).with_name("prompts.txt").read_text().splitlines()
    if line.strip()
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
    if result.returncode != 0:
        raise RuntimeError(
            f"engine exited {result.returncode} on {prompt!r}: {result.stderr[-400:]}"
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
    ap.add_argument("--model", required=True, help="model weights dir")
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

    # Free vLLM before the engine loop; the binary needs the memory.
    del llm
    gc.collect()

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
