# slice_tf -- teacher-forced slice verification, ported from the deepseek work

Ported from QuettaServe branch `kev/deepseek-batch`, `docs/scripts/` (all nine
files). The harness was written against the Kimi-K3 13-layer slice, so the
files arrived with kimi names; they are renamed here by role, because the
method is model-agnostic even though the current wiring targets that slice.
Content is otherwise identical to origin: the only edits are the renamed
module references, so provenance stays checkable -- apply the rename map below
to the origin file and diff. Model-specific wiring (`KIMI_SLICE_MODEL` and
friends) keeps its origin names until the harness is generalized after the
first green runs.

## Rename map

| origin (docs/scripts/) | here |
|---|---|
| `kimi_ref.py` | `ref_full.py` |
| `kimi_ref_slice.py` | `ref_slice.py` |
| `kimi_slice_truncate.py` | `slice_truncate.py` |
| `kimi_tf_verify.py` | `tf_verify.py` |
| `kimi_moe_probe.py` | `moe_routing_probe.py` |
| `kimi_vllm_tf.py` | `vllm_tf.py` |
| `kimi_vllm_baseline.py` | `vllm_baseline.py` |
| `kimi_vllm_nsys.py` | `vllm_nsys.py` |
| `nsys_kernels.py` | `nsys_kernels.py` |

`moe_routing_probe.py` rather than `moe_probe.py`: profiling/kernel_composed
already carries a `moe_probe.py` that measures MoE FFN latency; this one dumps
routing decisions. The `kimi_moe_*` strings inside `nsys_kernels.py` are
regexes over the engine's real kernel symbol names and are deliberately
untouched.

## Method

A free-running greedy chain cannot verify numerics on a truncated slice:
bf16 rounding flips the router, so every kernel produces a different chain
after a few steps. The attainable criterion is teacher-forced. Feed every
engine the same 56-token prompt (16 fixed ids plus the 40-token fp32
reference chain) and compare per-position argmaxes. `tf_verify.py` measures
the noise floor of that comparison (the HF reference against itself at bf16,
two kernel modes, each run twice for determinism); an engine passes when its
disagreement rate against the fp32 chain sits inside that floor.
`moe_routing_probe.py` dumps one MoE block's exact input, output and routing
decision (topk indices and weights) so routing, dispatch and combine can be
diffed independently.

## Files

| file | role |
|---|---|
| `ref_full.py` | full-model HF reference activations dump; owns the shared PROMPT_IDS |
| `ref_slice.py` | HF reference on the real 13-layer slice; the model builder the others import |
| `slice_truncate.py` | builds a valid standalone 13-layer HF model dir from the downloaded slice |
| `tf_verify.py` | teacher-forced bf16 noise-floor envelope; sets the acceptance band |
| `moe_routing_probe.py` | one MoE block dump: input, output, topk_idx, topk_weight, stage internals |
| `vllm_tf.py` | vLLM teacher-forced verification plus decode bench on the slice |
| `vllm_baseline.py` | vLLM latency baseline with a correctness check in front |
| `vllm_nsys.py` | vLLM decode profile on the slice, windowed for nsys |
| `nsys_kernels.py` | per-kernel GPU-time attribution from an nsys sqlite export |

## Prerequisites

The scripts expect the Kimi slice artifacts: the truncated 13-layer model dir
(`KIMI_SLICE_MODEL`, built by `slice_truncate.py` from the downloaded shards),
the fp32 reference dump (safetensors, produced by `ref_slice.py`), the fla
library on `PYTHONPATH`, and vLLM 0.28 for the native Kimi architecture. Each
module docstring carries its exact invocation (docstrings still show the
origin filenames where they predate the rename). Run stages and pass
criteria: `../VERIFICATION.md`.
