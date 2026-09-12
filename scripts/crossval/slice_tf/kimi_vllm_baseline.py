"""vLLM baseline for Kimi-K3-0.40B, with a correctness check in front of it.

vLLM 0.28 registers `KimiK3ForConditionalGeneration` natively, so it does *not* run
the checkpoint's own modeling code — it runs vLLM's reimplementation. A ms/step
number from a different architecture is worse than no number, so this script
refuses to time anything until vLLM's greedy continuation matches the HF reference
token for token.

    KIMI_MODEL=/data35/kevinlau/models/Kimi-K3-0.40B \
    python kimi_vllm_baseline.py --ref <hf-dump>.safetensors [--dtype bfloat16]

The reference dump supplies both the prompt and the expected continuation:
`decode{s}_token` is the input to step s, i.e. the greedy token the previous step
produced, so `decode0..decodeN` *is* the greedy continuation.

Also serves the 13-layer real-slice dir (--tp 4 on 4x80GB; MXFP4 experts load
natively via compressed-tensors). The slice dumps kimi-ref-slice.safetensors /
kimi-ref-slice-long.safetensors use the same PROMPT_IDS and decode{s}_token
keys, so no other changes are needed:

    KIMI_MODEL=/data35/kevinlau/kimi-slice/truncated \
    CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python kimi_vllm_baseline.py --ref /data35/kevinlau/kimi-ref-slice-long.safetensors \
        --tp 4 --verify-tokens 40

Known env quirk (h100, vllm-k3 venv): flashinfer 0.6.16.post3 fd_exchange.py
uses `array.array[int]` in an annotation, a TypeError at import on py3.11 that
escapes vLLM's ImportError-only guard and kills every TP worker. Fixed in the
venv by prepending `from __future__ import annotations` to that file
(site-packages/flashinfer/comm/fd_exchange.py; .orig kept alongside).

## Measured outcome on the 13-layer slice (2026-09-01, h100 GPUs 4-7): BLOCKED

The gate cannot pass on this slice at any dtype vLLM can serve, and the
reason is the slice, not vLLM (kimi/tests/real_slice.rs documents the same
wall for our port): the untrained 13-layer truncation amplifies bf16
rounding into router flips within a few steps, and torch even disagrees
with ITSELF there (chunk vs recurrent bf16 prefill: 6/16 argmax positions).
The only token-exact anchor is fp32 — which vLLM refuses for MXFP4
("torch.float32 is not supported for quantization method mxfp4").

Every bf16 configuration tracks a correct chain briefly, then flips —
different kernels, different flip points, all consistent with an honest
implementation under bf16 noise:
  TP=4 MARLIN            vs f32-recurrent chain:  5/40 (flip at 5)
  TP=4 MARLIN            vs bf16-recurrent dump:  7/8  (flip at 7)
  TP=4 emulation --eager vs bf16-recurrent dump:  6/8  (flip at 6)
  PP=4 MARLIN            vs bf16-chunk chain:     2/40 (flip at 2)
  (emulation needs --eager + the batched-token cap below: its on-the-fly
   dequant transient OOMs beside graph pools; PP=4 needs
   VLLM_PP_LAYER_PARTITION=4,3,3,3 — the auto [3,3,4,3] puts four MoE
   layers on one stage and OOMs during marlin repack — and
   --no-prefix-caching to dodge an assert in mamba_hybrid PP grouping.)

So no same-weights vLLM timing exists for the truncated slice; the number
must come from a slice whose composition is trained (more layers) or a
dtype both engines share end to end.
"""

import argparse
import json
import os
import time

# 16 fixed ids, identical to docs/scripts/kimi_ref.py.
PROMPT_IDS = [1, 4321, 100, 65535, 2048, 777, 31415, 9,
              128000, 42, 5, 99991, 1234, 60000, 8, 163839]


def expected_tokens(ref_path, n):
    from safetensors.torch import load_file
    d = load_file(ref_path)
    out = []
    for s in range(n):
        k = f"decode{s}_token"
        if k not in d:
            break
        out.append(int(d[k].reshape(-1)[0]))
    return out


def slope_fit(xs, ys):
    n = float(len(xs))
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, (1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="HF reference dump from kimi_ref.py")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--verify-tokens", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=512)
    ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    ap.add_argument("--pp", type=int, default=1,
                    help="pipeline_parallel_size. PP shards by layer with no "
                         "cross-rank reductions, so unlike TP it does not "
                         "change bf16 summation order vs a single-GPU run — "
                         "the same topology the HF reference dumps used.")
    ap.add_argument("--moe-backend", default="auto",
                    help="vLLM MoE kernel backend (e.g. marlin, emulation). "
                         "'emulation' dequantizes MXFP4 to bf16 on the fly — "
                         "numerically closest to the HF reference, which also "
                         "runs dequantized bf16 GEMMs.")
    ap.add_argument("--eager", action="store_true", help="disable vLLM's CUDA graphs")
    ap.add_argument("--no-prefix-caching", action="store_true",
                    help="disable prefix caching (also switches the hybrid "
                         "mamba cache mode away from 'align')")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    model = os.environ["KIMI_MODEL"]
    llm = LLM(
        model=model,
        trust_remote_code=True,
        dtype=args.dtype,
        skip_tokenizer_init=True,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=args.pp,
        enforce_eager=args.eager,
        gpu_memory_utilization=0.85,
        moe_backend=args.moe_backend,
        # This harness only ever runs one 16-token prompt at a time. The
        # default profiling batch (chunked prefill, 16384 tokens) is what a
        # server must survive, not what this gate runs — and under
        # --moe-backend emulation its on-the-fly dequant transient alone
        # overflows the 25 GiB left beside the resident weights (measured:
        # "Available KV cache memory: -4.36 GiB"). Capping the batch to the
        # model length bounds the transient without touching the math.
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=1,
        enable_prefix_caching=not args.no_prefix_caching,
    )

    prompt = TokensPrompt(prompt_token_ids=PROMPT_IDS)

    # --- correctness gate ---------------------------------------------------
    want = expected_tokens(args.ref, args.verify_tokens)
    out = llm.generate(
        [prompt],
        SamplingParams(temperature=0.0, max_tokens=len(want), detokenize=False),
    )
    got = list(out[0].outputs[0].token_ids)
    match = sum(1 for a, b in zip(got, want) if a == b)
    print(f"\n=== vLLM vs HF reference ({args.dtype}) ===")
    print(f"reference : {want}")
    print(f"vllm      : {got}")
    print(f"match     : {match}/{len(want)} tokens")
    verdict = {
        "dtype": args.dtype,
        "tokens_expected": want,
        "tokens_vllm": got,
        "match": match,
        "total": len(want),
    }
    if match != len(want):
        first = next(i for i, (a, b) in enumerate(zip(got, want)) if a != b)
        print(f"\nFIRST DIVERGENCE at generated token {first}: "
              f"vllm {got[first]} vs reference {want[first]}")
        print("vLLM is NOT reproducing this checkpoint's semantics. "
              "No timing is reported: a ms/step from a different architecture "
              "is not a baseline.")
        print(json.dumps(verdict))
        return

    # --- timing -------------------------------------------------------------
    # Slope fit over several generation lengths. The marginal cost of a decode
    # step is the slope of total latency against tokens generated; the intercept
    # absorbs prefill and the fixed per-request overhead, which is exactly what a
    # naive total/tokens division would smear into the per-step number.
    lengths = [16, 64, 128, 256]
    print("\n=== decode timing (warm) ===")
    for _ in range(2):  # warm: JIT, graph capture, allocator
        llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=32,
                                              detokenize=False))
    xs, ys = [], []
    for n in lengths:
        best = None
        for _ in range(3):
            t0 = time.perf_counter()
            llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=n,
                                                  detokenize=False))
            dt = (time.perf_counter() - t0) * 1e3
            best = dt if best is None else min(best, dt)
        xs.append(float(n))
        ys.append(best)
        print(f"  {n:4d} tokens: {best:8.2f} ms")
    a, b, r2 = slope_fit(xs, ys)
    print(f"\nslope fit: {b:.3f} ms/step, intercept {a:.2f} ms, r2 {r2:.5f}")
    if r2 < 0.999:
        print("r2 < 0.999 -- the fit is not clean; do not quote this number.")
    verdict.update({"ms_per_step": b, "intercept_ms": a, "r2": r2,
                    "points": dict(zip(lengths, ys)),
                    "graphs": not args.eager})
    print(json.dumps(verdict))


if __name__ == "__main__":
    main()
