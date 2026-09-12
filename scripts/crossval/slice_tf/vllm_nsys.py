"""vLLM decode profile on the 13-layer real slice, windowed for nsys.

Companion to `vllm_tf.py`'s bench: that one fits a slope through *host*
latencies, this one hands nsys a pair of NVTX-delimited windows from which the
marginal per-step **device** work can be recovered by subtraction:

    window `prefill_only`  = one prefill of `--prompt` tokens + 1 decode step
    window `prefill_decode` = the same prefill + `--steps + 1` decode steps

so (prefill_decode - prefill_only) / steps is one decode step's kernel time,
per family, with no boundary heuristics and no guess about which kernel
delimits a step. The same subtraction is what makes the two engines
comparable: our own profile brackets exactly N decode steps with
`cuProfilerStart/Stop` (`KIMI_BENCH_PROFILE=1`), which is the same quantity
measured a different way.

NVTX ranges are pushed in the *driver* process; the TP workers are separate
processes, so the ranges do not nest around their kernels in the GUI. That is
fine and deliberate — the analysis windows GPU kernels by wall-clock timestamp
against the range boundaries, which is process-independent.

    KIMI_MODEL=/data35/kevinlau/kimi-slice/truncated CUDA_VISIBLE_DEVICES=0,1,2,3 \
    nsys profile -t cuda,nvtx --cuda-graph-trace=node -o vllm_tp4 \
      /data48/kevinlau/envs/vllm-k3/bin/python vllm_nsys.py --tp 4 --steps 200
"""

import argparse
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=512)
    ap.add_argument("--prompt", type=int, default=100)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--eager", action="store_true")
    args = ap.parse_args()

    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=os.environ["KIMI_MODEL"],
        trust_remote_code=True,
        dtype=args.dtype,
        skip_tokenizer_init=True,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tp,
        enforce_eager=args.eager,
        gpu_memory_utilization=0.85,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=1,
        enable_prefix_caching=False,
    )

    ids = [((i * 137 + 11) % 100_000) for i in range(args.prompt)]
    prompt = TokensPrompt(prompt_token_ids=ids)

    def run(n):
        llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=n,
                                              detokenize=False))

    # Warm: JIT, graph capture, allocator. Nothing in the windows below may be
    # a first touch of anything.
    for _ in range(3):
        run(32)

    windows = []
    for name, n in (("prefill_only", 1), ("prefill_decode", args.steps + 1)):
        # A quiet gap on both sides so the two windows cannot bleed into each
        # other through kernels still draining at the boundary.
        torch.cuda.synchronize()
        time.sleep(0.5)
        torch.cuda.nvtx.range_push(name)
        t0 = time.perf_counter()
        run(n)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        torch.cuda.nvtx.range_pop()
        windows.append((name, n, dt))
        print(f"[nsys] {name}: {n} generated tokens, {dt:.2f} ms host", flush=True)
        time.sleep(0.5)

    (_, _, t1), (_, _, t2) = windows
    print(f"[nsys] host marginal: {(t2 - t1) / args.steps:.3f} ms/step "
          f"over {args.steps} steps")


if __name__ == "__main__":
    main()
