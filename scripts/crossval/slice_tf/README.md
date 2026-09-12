# slice_tf -- teacher-forced slice verification, ported from the deepseek work

Ported byte identical from QuettaServe branch `kev/deepseek-batch`,
`docs/scripts/` (all nine files). Refactoring waits for the first green runs
here, so every file still matches its origin exactly and provenance stays
checkable with a diff against that branch. The harness was written for the
Kimi-K3 13-layer slice; what deepseek and qwen inherit is the method, not the
model wiring.

## Method

A free-running greedy chain cannot verify numerics on a truncated slice:
bf16 rounding flips the router, so every kernel produces a different chain
after a few steps. The attainable criterion is teacher-forced. Feed every
engine the same 56-token prompt (16 fixed ids plus the 40-token fp32
reference chain) and compare per-position argmaxes. `kimi_tf_verify.py`
measures the noise floor of that comparison (the HF reference against itself
at bf16, two kernel modes, each run twice for determinism); an engine passes
when its disagreement rate against the fp32 chain sits inside that floor.
`kimi_moe_probe.py` dumps one MoE block's exact input, output and routing
decision (topk indices and weights) so routing, dispatch and combine can be
diffed independently.

## Files

| file | role |
|---|---|
| `kimi_ref.py` | full-model HF reference activations dump; owns the shared PROMPT_IDS |
| `kimi_ref_slice.py` | HF reference on the real 13-layer slice; the model builder the others import |
| `kimi_slice_truncate.py` | builds a valid standalone 13-layer HF model dir from the downloaded slice |
| `kimi_tf_verify.py` | teacher-forced bf16 noise-floor envelope; sets the acceptance band |
| `kimi_moe_probe.py` | layer-1 MoE block dump: input, output, topk_idx, topk_weight, stage internals |
| `kimi_vllm_tf.py` | vLLM teacher-forced verification plus decode bench on the slice |
| `kimi_vllm_baseline.py` | vLLM latency baseline with a correctness check in front |
| `kimi_vllm_nsys.py` | vLLM decode profile on the slice, windowed for nsys |
| `nsys_kernels.py` | per-kernel GPU-time attribution from an nsys sqlite export |

## Prerequisites

The scripts expect the Kimi slice artifacts: the truncated 13-layer model dir
(`KIMI_SLICE_MODEL`, built by `kimi_slice_truncate.py` from the downloaded
shards), the fp32 reference dump (safetensors, produced by
`kimi_ref_slice.py`), the fla library on `PYTHONPATH`, and vLLM 0.28 for the
native Kimi architecture. Each module docstring carries its exact invocation.
Run stages and pass criteria: `../VERIFICATION.md`.
