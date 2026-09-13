# crossval -- QuettaServe vs vLLM

Cross-validates a QuettaServe engine against vLLM at matched context, batch
and tensor-parallel degree. Ported from QuettaServe `kev/xval`; QuettaServe
drives it through this repo as a submodule. Only llama is wired: the
workloads file keeps the qwen and deepseek entries it arrived with, and the
recipes refuse them until validated.

## Run

One command drives the whole thing:

```bash
PY=<python-with-vllm> QS_BIN=<engine-binary> \
  ./xval.sh llama <weights-dir> [bench-log]
```

It captures the vLLM baseline if not cached, logit-checks the baseline
against the Hugging Face transformers reference (`LOGIT=0` skips), runs the
greedy engine-vs-vLLM check when `QS_BIN` is set, and prints the comparison
table when a criterion bench log is given. From a QuettaServe checkout the same pipeline is
`just crossval llama`.

## Pipeline stages

1. `vllm.sh <crate> [grid] <weights-dir> [python]` captures the vLLM
   baseline: `vmin_fit.py` slope-fits ms per decode step for every
   (ctx, batch) cell with r2 recorded, `cache.py` writes
   `baselines/vllm-<crate>.json`.
2. The engine side produces a criterion log (`just bench llama` in
   QuettaServe).
3. `table.py <bench-log> <baseline.json>` is the cross-validation: it pairs
   the two at matched (ctx, batch, tp) and prints ours vs vLLM per cell.

Correctness runs alongside the latency: `greedy_agreement.py` (engine vs
vLLM text) and `logit_agreement.py` (vLLM vs the transformers reference,
teacher forced).

## Measurement contract

Both columns time the same quantity: steady-state decode ms per token at a
resident batch, pipelined, synchronized only at the ends, at matched
(context, batch, tp). The vLLM side is the slope of generate time over the
step points; the engine side is the `decode_loop` bench (K pipelined steps,
one sync). Criterion's decode groups synchronize inside every iteration --
a different quantity -- so `table.py` prints those cells with ratios
withheld (`XVAL_ALLOW_METHOD_MISMATCH=1` forces them). Baselines record the
KV cache dtype per point; rows with mismatched KV formats are flagged `KV`.

The H200 table is workload `llama-h200` (fp16, the 12-cell grid; 64x16384
SKIPs on fp16 capacity) plus `llama-h200-fp8kv` (bf16 model + fp8 KV, the
one cell that only fits quantized -- vLLM requires bf16 activations with
fp8 KV).

`PROF=1 xval.sh ...` nsys-captures the `prof` grid cell (8192x64, where the
decode gap is widest) for the vLLM baseline, and for the engine bench when
`QS_BENCH_BIN` points at the latency binary. Reports land in `profiles/`
with a plain-text kernel summary beside each. Kernel time tells you engine
math; the idle share of the trace tells you launch and plan overhead.

## Files

| file | role |
|---|---|
| `xval.sh` | the orchestrator: baseline if missing, logit check, greedy check, table, PROF=1 profiles |
| `prof.sh` | nsys capture of one command into profiles/, with the kernel summary dumped as text |
| `vllm.sh` | baseline capture: picks gpus, runs vmin_fit, caches the json |
| `vmin_fit.py` | vLLM decode slope fit, ms per step and r2 per cell (needs vllm) |
| `cache.py` | packs a raw run into the cached baseline json |
| `table.py` | the comparison: engine criterion log vs baseline at matched (ctx, bs, tp) |
| `greedy_agreement.py` | engine binary vs vLLM on shared prompts, leading-token agreement (needs vllm) |
| `logit_agreement.py` | vLLM vs the transformers reference, per-position argmax and true-token logprob (needs vllm, transformers) |
| `with_gpu.sh` | run a command on an idle gpu, which `free_gpu.sh` picks |
| `workloads.json` | per-crate model, dtype, maxlen, tp and the (ctx, batch) grids |

## Tests

`tests/test_crossval_contracts.py` runs without a gpu: scripts parse, shell
helpers keep their exec bit, workloads stay consistent, and the grids cover
the engine bench cells. `tests/test_crossval_smoke_gpu.py` runs the baseline
on the one-cell grid where a gpu and MODEL exist. Baselines are generated
and stay untracked.
