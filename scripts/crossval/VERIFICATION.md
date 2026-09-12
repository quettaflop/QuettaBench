# crossval verification stages

One row per stage, in run order. A stage passes only on its stated criterion,
with the evidence file kept. Static stages re-run anywhere; GPU stages name
their hardware. Update the status column in the same change that adds the
evidence.

| stage | what | command | pass criterion | status |
|---|---|---|---|---|
| V1 | port equivalence | diff each `slice_tf/*.py` against QuettaServe `origin/kev/deepseek-batch:docs/scripts/` with the rename map in `slice_tf/README.md` applied | 9/9 identical modulo the recorded renames | pass |
| V2 | static contracts | `python -m pytest tests/test_crossval_contracts.py -q` | all tests pass | pass |
| V3 | engine tp equivalence | QuettaServe `supp_tp` checkout, 8 idle GPUs: `cargo test -p llama --test tp_equivalence -- --nocapture` | passes: tp 2/4/8 tokens equal tp1, logits within f16 tolerance | pending |
| V4 | engine tp bench | `supp_tp` checkout: `TP=2 MODEL=<weights> cargo bench --bench latency 2>&1 \| tee /tmp/qs-bench-llama-tp2.txt`, then TP=4 | criterion completes; `decode_tp{N}` groups present at ctx 100/1024/4096/8192 | pending |
| V5 | tp baselines | `just vllm llama-tp2` and `just vllm llama-tp4` | 4 cells each, every r2 >= 0.999, no SKIP lines | pending |
| V6 | tp tables | `python3 scripts/crossval/table.py /tmp/qs-bench-llama-tp2.txt scripts/crossval/baselines/vllm-llama-tp2.json`, same for tp4 | 4 matched rows per tp degree | pending |
| V7 | realigned llama baseline | `just vllm llama` (grid is now 15 cells) | 15 cells, r2 >= 0.999 each, 4096x32 and 4096x64 present | pending |
| V8 | deepseek baseline | 4 idle H200s: `just vllm deepseek` | completed cells at r2 >= 0.999; any SKIP carries its reason | pending |
| V9 | slice artifacts | `slice_truncate.py`, then `ref_slice.py` for the fp32 dump | truncated model dir and fp32 reference safetensors exist | pending |
| V10 | noise floor | `tf_verify.py <fp32-ref> out.json` at `KIMI_REF_DTYPE=bf16` | both KDA modes deterministic across their two runs; band recorded | pending |
| V11 | vllm inside band | verify part of `vllm_tf.py` | vLLM disagreement rate vs the fp32 chain <= the worse HF bf16 kernel rate from V10 | pending |
| V12 | routing dump | `moe_routing_probe.py <out.safetensors>` | dump holds topk_idx, topk_weight and the stage tensors | pending |

Notes. V3 and V4 run from a QuettaServe checkout of `supp_tp` next to this
repo; the table consumes their criterion logs unchanged. V9 to V12 need the
Kimi slice artifacts (see `slice_tf/README.md`); they verify the method the
deepseek and qwen lanes inherit, not the deepseek model itself. The deepseek
engine lane has no criterion bench yet, so V8 is baseline-only by design. The
engine-side diff against the V12 dump is a QuettaServe work item.
