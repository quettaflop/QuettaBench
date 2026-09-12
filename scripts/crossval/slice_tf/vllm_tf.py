"""vLLM teacher-forced verification + decode bench on the 13-layer real slice.

PART 1 (verify): a free-running greedy chain cannot answer a numerics question
on this slice (bf16 rounding flips 16-of-896 router choices; even torch
disagrees with itself across its two KDA kernels). So verify TEACHER-FORCED:
feed vLLM the 16-token prompt concatenated with the 40 fp32-recurrent
reference tokens as ONE prompt and request prompt_logprobs — vLLM returns
per-position rank/logprob information for prompt tokens, so the rank-1 token
at position 16+s is vLLM's argmax prediction of forced token t_s given the
identical prefix the fp32 reference saw. The acceptance criterion (computed
outside, against the HF bf16 chunk-vs-recurrent envelope from
tf_verify.py) is that vLLM's disagreement rate with the fp32 chain is
<= the worse HF-bf16 kernel's rate: vLLM is bf16 and cannot beat the bf16
noise floor, but a faithful implementation must sit inside it.

PART 2 (bench, only with --bench and only if --max-disagree holds): same
slope-fit protocol as vllm_baseline.py, but on a ~100-token prompt.

    KIMI_MODEL=/data35/kevinlau/kimi-slice/truncated \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python vllm_tf.py --ref /data35/kevinlau/kimi-ref-slice-f32-long.safetensors \
        --tp 4 --out vllm_tf.json [--bench --max-disagree N]
"""

import argparse
import json
import os
import time

from vllm_baseline import PROMPT_IDS, slope_fit

N_FORCED = 40


def chain_from_dump(ref_path, n):
    from safetensors import safe_open
    out = []
    with safe_open(ref_path, framework="pt") as f:
        for s in range(n):
            out.append(int(f.get_tensor(f"decode{s}_token").reshape(-1)[0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="fp32-recurrent 40-step dump")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=512)
    ap.add_argument("--moe-backend", default="auto")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--out", required=True, help="verdict JSON path")
    ap.add_argument("--bench", action="store_true",
                    help="run the decode slope-fit after verification")
    ap.add_argument("--max-disagree", type=int, default=-1,
                    help="refuse to bench if teacher-forced disagreements with "
                         "the fp32 chain exceed this (the worse HF-bf16 "
                         "kernel's count); -1 = never bench")
    ap.add_argument("--bench-prompt", type=int, default=100)
    ap.add_argument("--bench-repeats", type=int, default=3)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    forced = chain_from_dump(args.ref, N_FORCED)
    concat = list(PROMPT_IDS) + forced
    print(f"teacher-forced prompt: {len(concat)} tokens", flush=True)

    llm = LLM(
        model=os.environ["KIMI_MODEL"],
        trust_remote_code=True,
        dtype=args.dtype,
        skip_tokenizer_init=True,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tp,
        enforce_eager=args.eager,
        gpu_memory_utilization=0.85,
        moe_backend=args.moe_backend,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=1,
        # vLLM V1 does not serve prompt_logprobs out of a prefix-cache hit, and
        # the teacher-forced prompt shares its 16-token head with every other
        # request this process makes. Off, so every position is recomputed.
        enable_prefix_caching=False,
    )

    # --- teacher-forced verification -----------------------------------------
    out = llm.generate(
        [TokensPrompt(prompt_token_ids=concat)],
        SamplingParams(temperature=0.0, max_tokens=1, detokenize=False,
                       prompt_logprobs=5),
    )
    plps = out[0].prompt_logprobs
    assert plps is not None and len(plps) == len(concat), \
        f"prompt_logprobs length {plps and len(plps)} != {len(concat)}"

    preds, gaps, ranks_of_forced = [], [], []
    for s in range(N_FORCED):
        entry = plps[16 + s]  # distribution over token at position 16+s
        by_rank = sorted(
            ((tid, lp) for tid, lp in entry.items()),
            key=lambda kv: kv[1].rank,
        )
        top1_id, top1 = by_rank[0]
        assert top1.rank == 1, f"pos {16+s}: no rank-1 entry"
        preds.append(int(top1_id))
        gap = None
        if len(by_rank) > 1 and by_rank[1][1].rank == 2:
            gap = float(top1.logprob - by_rank[1][1].logprob)
        gaps.append(gap)
        ranks_of_forced.append(int(entry[concat[16 + s]].rank))

    agree = sum(1 for p, t in zip(preds, forced) if p == t)
    dis = [s for s, (p, t) in enumerate(zip(preds, forced)) if p != t]
    print(f"\n=== vLLM teacher-forced vs fp32 chain (tp={args.tp}, "
          f"{args.dtype}, moe={args.moe_backend}) ===")
    print(f"agree: {agree}/{N_FORCED}  disagree steps: {dis}")
    for s in dis:
        print(f"  step {s}: vllm {preds[s]} vs fp32 {forced[s]} "
              f"(vllm top2 logprob gap {gaps[s]}, forced-token rank {ranks_of_forced[s]})")

    verdict = {
        "tp": args.tp, "dtype": args.dtype, "moe_backend": args.moe_backend,
        "eager": args.eager,
        "forced_tokens": forced, "pred": preds,
        "vllm_top2_logprob_gap": gaps, "rank_of_forced": ranks_of_forced,
        "agree_with_fp32": agree, "disagree_steps": dis,
    }

    # --- bench ----------------------------------------------------------------
    n_dis = N_FORCED - agree
    if args.bench and args.max_disagree >= 0 and n_dis <= args.max_disagree:
        prompt_ids = [((i * 137 + 11) % 100_000) for i in range(args.bench_prompt)]
        prompt = TokensPrompt(prompt_token_ids=prompt_ids)
        lengths = [16, 64, 128, 256]
        print(f"\n=== decode timing (warm, prompt {args.bench_prompt}) ===")
        for _ in range(2):
            llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=32,
                                                  detokenize=False))
        xs, ys, allpts = [], [], {}
        for n in lengths:
            times = []
            for _ in range(args.bench_repeats):
                t0 = time.perf_counter()
                llm.generate([prompt], SamplingParams(temperature=0.0,
                                                      max_tokens=n,
                                                      detokenize=False))
                times.append((time.perf_counter() - t0) * 1e3)
            xs.append(float(n))
            ys.append(min(times))
            allpts[n] = times
            print(f"  {n:4d} tokens: best {min(times):8.2f} ms  all {['%.1f' % t for t in times]}")
        a, b, r2 = slope_fit(xs, ys)
        print(f"\nslope fit: {b:.3f} ms/step, intercept {a:.2f} ms, r2 {r2:.5f}")
        verdict.update({"ms_per_step": b, "intercept_ms": a, "r2": r2,
                        "points": allpts, "graphs": not args.eager,
                        "bench_prompt": args.bench_prompt})
    elif args.bench:
        print(f"\nbench REFUSED: {n_dis} disagreements > max {args.max_disagree}")
        verdict["bench_refused"] = n_dis

    with open(args.out, "w") as f:
        json.dump(verdict, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
