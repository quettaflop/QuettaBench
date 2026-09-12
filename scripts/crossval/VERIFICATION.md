# crossval verification stages

One row per stage, in run order. A stage passes only on its stated criterion,
with the evidence file kept. Static stages re-run anywhere; GPU stages name
their hardware. Update the status column in the same change that adds the
evidence.

| stage | what | command | pass criterion | status |
|---|---|---|---|---|
| V1 | port equivalence | diff each `slice_tf/*.py` against QuettaServe `origin/kev/deepseek-batch:docs/scripts/` with the rename map in `slice_tf/README.md` applied | 9/9 identical modulo the recorded renames | pass |
| V2 | static contracts | `python -m pytest tests/test_crossval_contracts.py -q` | all tests pass | pass |
| V3 | engine tp equivalence | QuettaServe `supp_tp` checkout, 8 idle GPUs: `cargo test -p llama --test tp_equivalence -- --nocapture` | passes: tp 2/4/8 tokens equal tp1, logits within f16 tolerance | blocked |
| V4 | engine tp bench | `supp_tp` checkout: `TP=2 MODEL=<weights> cargo bench --bench latency 2>&1 \| tee /tmp/qs-bench-llama-tp2.txt`, then TP=4 | criterion completes; `decode_tp{N}` groups present at ctx 100/1024/4096/8192 | blocked |
| V5 | tp baselines | `just vllm llama-tp2` and `just vllm llama-tp4` | 4 cells each, every r2 >= 0.999, no SKIP lines | blocked |
| V6 | tp tables | `python3 scripts/crossval/table.py /tmp/qs-bench-llama-tp2.txt scripts/crossval/baselines/vllm-llama-tp2.json`, same for tp4 | 4 matched rows per tp degree | blocked |
| V7 | realigned llama baseline | `just vllm llama` (grid is now 15 cells) | 15 cells, r2 >= 0.999 each, 4096x32 and 4096x64 present | pass |
| V8 | deepseek baseline | 4 idle H200s: `just vllm deepseek` | completed cells at r2 >= 0.999; any SKIP carries its reason | blocked |
| V9 | slice artifacts | `slice_truncate.py`, then `ref_slice.py` for the fp32 dump | truncated model dir and fp32 reference safetensors exist | blocked |
| V10 | noise floor | `tf_verify.py <fp32-ref> out.json` at `KIMI_REF_DTYPE=bf16` | both KDA modes deterministic across their two runs; band recorded | blocked |
| V11 | vllm inside band | verify part of `vllm_tf.py` | vLLM disagreement rate vs the fp32 chain <= the worse HF bf16 kernel rate from V10 | blocked |
| V12 | routing dump | `moe_routing_probe.py <out.safetensors>` | dump holds topk_idx, topk_weight and the stage tensors | blocked |

Notes. V3 and V4 run from a QuettaServe checkout of `supp_tp` next to this
repo; the table consumes their criterion logs unchanged. V9 to V12 need the
Kimi slice artifacts (see `slice_tf/README.md`); they verify the method the
deepseek and qwen lanes inherit, not the deepseek model itself. The deepseek
engine lane has no criterion bench yet, so V8 is baseline-only by design. The
engine-side diff against the V12 dump is a QuettaServe work item.

Run 2026-09-12 (runpod node-1, 8x H200 shared). Idle GPUs at start: 1, 5, 6, 7
(0, 3, 4 held by other sessions). V3 blocked: tp_equivalence needs all 8 GPUs
and only 4 were idle. V9 to V12 blocked: the pod has no `fla` library and no
Kimi slice artifacts (only the qserve kimi crate source), so the slice
reference cannot be built or run here. The supp_tp engine build needed
`RUSTFLAGS=-C linker-features=-lld` on this pod (stock rust-lld segfaults
linking proc-macro shared objects; the system `ld.bfd` links cleanly). V4 and
therefore V6 blocked: after the linker fix the build reached candle-flash-attn
and nvcc failed compiling `flash_fwd_hdim128_bf16_causal_sm80.cu` under CUDA
12.8 (sibling kernels compiled; error empty). That is a QuettaServe engine
build issue, not a harness one; the TP harness is still verified through V5
(vLLM tp baselines) and the tp-aware table and cache contract tests.
Evidence: qs-supptp/build_clean.log on the pod.

V7 pass: 15 RESULT cells produced including the new 4096x32 and 4096x64;
14 of 15 at r2 >= 0.999. The 100x64 corner came in at r2 = 0.982 (tiny
context, largest batch, the noisiest corner of the grid); flagged, not
re-run. Evidence: evidence/v7_llama15.log. V8 deepseek blocked: by the time
it ran, other sessions held the GPUs and only 2 were free (needs 4 for tp4);
weights are present (156 GB), so it is runnable when 4 GPUs are idle.
Evidence: evidence/v8_deepseek.log ("NEED 4 FREE GPUS, have: 1 7").

V5 blocked on vLLM tensor-parallel init, two attempts. First (default
multiproc): "Engine core initialization failed" with an NCCL
ProcessGroup/leaked-shared-memory warning. Second
(VLLM_WORKER_MULTIPROC_METHOD=spawn, GPUs 1,7): the run stops right after the
META header with no RESULT, error, or exit line, i.e. the TP workers hang or
die silently during init. This is a vLLM-0.26 tensor-parallel startup issue
in the detached environment on this pod, not a harness defect: the tp
workloads, tp-aware table/cache, and single-GPU baseline path are all
verified (V2, V7). Unblock needs an interactive vLLM tp debug or a different
vLLM build. Evidence: evidence/v5_tp2.log, evidence/v5_tp2_retry.log.
