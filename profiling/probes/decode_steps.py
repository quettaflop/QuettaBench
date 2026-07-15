# profiling/probes/decode_steps.py
"""Profile production-like vLLM decode-step timing.

This measures full-model generation with CUDA events and reports a TPOT-style
per-step decode interval for each (batch_size, context_len). CUDA graphs remain
enabled by default, so this is meant to be closer to serving behavior than an
isolated NCU kernel replay.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
DEFAULT_CONTEXT_LENGTHS = (512, 1024, 2048, 4096, 8192, 16384)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Output CSV path, e.g. profile_data/results/decode_profile_H100_YYYY-MM-DD.csv")
    parser.add_argument("--gpu-label", default="H100")
    parser.add_argument("--batch-sizes", nargs="*", type=int, default=DEFAULT_BATCH_SIZES)
    parser.add_argument(
        "--context-lengths",
        nargs="*",
        type=int,
        default=DEFAULT_CONTEXT_LENGTHS,
    )
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="TP degree (set 2 for 2xH100; requires CUDA_VISIBLE_DEVICES with that many GPUs).")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Disable CUDA graphs. REQUIRED for hybrid linear-attention models "
                             "(qwen3_5 / gated-delta-net) whose engine init fails under graph capture; "
                             "match the cell's GT launch flags.")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Pass trust_remote_code to the engine (needed for some custom architectures).")
    parser.add_argument("--gdn-prefill-backend", default=None,
                        help="vLLM additional_config gdn_prefill_backend, e.g. 'triton'. On Hopper (sm90) "
                             "the default 'auto' selects the flashinfer GDN kernel which JIT-compiles and "
                             "needs cuda/ptx headers (fails on some toolchains); 'triton' avoids it. Match "
                             "the cell's GT launcher (the GT sweep set 'triton').")
    parser.add_argument(
        "--max-total-kv-tokens",
        type=int,
        default=500_000,
        help="Skip shapes above this batch * (context + decode) KV footprint.",
    )
    return parser.parse_args()


def make_prompt(tokenizer: Any, target_tokens: int) -> str:
    """Build a prompt that retokenizes close to target_tokens.

    Exact prompt length is recorded from vLLM output when available. This helper
    keeps the requested grid stable without requiring a fixture dataset.
    """
    target_tokens = max(1, int(target_tokens))
    text = (" the" * target_tokens).strip()
    encode = getattr(tokenizer, "encode", None)
    decode = getattr(tokenizer, "decode", None)
    if encode is None or decode is None:
        return text

    token_ids = encode(text, add_special_tokens=False)
    while len(token_ids) < target_tokens:
        text = f"{text} {text}"
        token_ids = encode(text, add_special_tokens=False)
    return decode(token_ids[:target_tokens])


def observed_prompt_tokens(output: Any, fallback: int) -> int:
    token_ids = getattr(output, "prompt_token_ids", None)
    if token_ids is None:
        return fallback
    try:
        return len(token_ids)
    except TypeError:
        return fallback


def main() -> None:
    args = parse_args()

    import torch
    from vllm import LLM, SamplingParams

    print(f"Loading {args.model}")
    llm_kwargs: dict[str, Any] = dict(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        trust_remote_code=args.trust_remote_code,
    )
    if args.gdn_prefill_backend:
        llm_kwargs["additional_config"] = {"gdn_prefill_backend": args.gdn_prefill_backend}
    llm = LLM(**llm_kwargs)
    # Capture the actual allocated GPU KV-block pool (the available_kv_blocks the predictor
    # needs for this hardware/TP config). Path differs across vLLM versions; best-effort.
    num_gpu_blocks = None
    for path in ("llm_engine.cache_config", "llm_engine.model_config"):
        try:
            obj = llm
            for attr in path.split("."):
                obj = getattr(obj, attr)
            nb = getattr(obj, "num_gpu_blocks", None)
            if nb:
                num_gpu_blocks = nb
                break
        except Exception:
            pass
    print(f"KV_POOL tensor_parallel_size={args.tensor_parallel_size} "
          f"gpu_mem={args.gpu_memory_utilization} num_gpu_blocks={num_gpu_blocks}", flush=True)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.decode_tokens,
        ignore_eos=True,
    )

    rows: list[dict[str, object]] = []
    for batch_size in args.batch_sizes:
        for context_len in args.context_lengths:
            total_kv = batch_size * (context_len + args.decode_tokens)
            if total_kv > args.max_total_kv_tokens:
                print(
                    f"SKIP B={batch_size} T={context_len} "
                    f"(KV={total_kv} > {args.max_total_kv_tokens})"
                )
                continue

            prompt = make_prompt(tokenizer, context_len)
            prompts = [prompt] * batch_size
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            print(f"Profiling B={batch_size} T={context_len}...", end=" ", flush=True)
            torch.cuda.synchronize()
            start_event.record()
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
            end_event.record()
            torch.cuda.synchronize()

            gpu_ms = start_event.elapsed_time(end_event)
            generated_tokens = sum(len(item.outputs[0].token_ids) for item in outputs)
            decode_intervals = max(1, generated_tokens - batch_size)
            decode_step_ms = gpu_ms / decode_intervals * batch_size
            observed_context = max(
                observed_prompt_tokens(item, context_len) for item in outputs
            )

            rows.append({
                "gpu": args.gpu_label,
                "batch_size": batch_size,
                "context_len": context_len,
                "observed_context_len": observed_context,
                "total_kv_tokens": total_kv,
                "decode_step_ms": round(decode_step_ms, 4),
                "generated_tokens": generated_tokens,
                "decode_intervals": decode_intervals,
                "gpu_ms": round(gpu_ms, 4),
            })
            print(f"{decode_step_ms:.2f} ms/step")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "gpu",
                "batch_size",
                "context_len",
                "observed_context_len",
                "total_kv_tokens",
                "decode_step_ms",
                "generated_tokens",
                "decode_intervals",
                "gpu_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
