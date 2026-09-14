# crossval -- QuettaServe vs vLLM

Cross-validates a QuettaServe engine against vLLM at matched context, batch
and tensor-parallel degree. QuettaServe drives it through this repo as a
submodule. llama is certified; deepseek is wired for bring-up (see the
DeepSeek section); qwen exists in the workloads file but the recipes refuse it
until validated.

## Run

One command drives the whole thing:

```bash
PY=<python-with-vllm> QS_BIN=<engine-binary> \
  ./xval.sh llama <weights-dir> [bench-log]
```

It picks idle GPUs once and runs every stage on them, captures the vLLM
baseline if not cached, logit-checks the baseline against the Hugging Face
transformers reference (result cached beside the baseline; `LOGIT=1` reruns,
`LOGIT=0` skips), runs the greedy engine-vs-vLLM check when `QS_BIN` is set,
and prints the comparison table when a criterion bench log is given. From a
QuettaServe checkout the same pipeline is `just crossval llama`.

## Pipeline stages

1. `vllm.sh <crate> [grid] <weights-dir> [python]` captures the vLLM
   baseline: `vmin_fit.py` slope-fits ms per decode step for every
   (ctx, batch) cell with r2 recorded, `cache.py` writes
   `baselines/vllm-<crate>.json`. A named grid writes
   `vllm-<crate>-<grid>.json`, so scratch grids never clobber the real
   baseline.
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

## DeepSeek (bring-up)

DeepSeek-V4-Flash is a mixture-of-experts model with multi-head latent
attention, served at tensor-parallel 8. It is wired but not yet certified
(`verified: false`), so the recipes require an explicit opt-in until the
numbers are validated. The measurement contract is unchanged: both sides time
steady-state marginal decode ms per step at matched (ctx, bs, tp).

- vLLM baseline:
  `ALLOW_UNVERIFIED=1 vllm.sh deepseek deepseek <weights> <py>`. The workload
  sets `expert_parallel` (vmin_fit passes `enable_expert_parallel` to vLLM) and
  `tp: 8`. `<py>` must be the DeepSeek vLLM fork (ununnilium/vllm-ds4-sm120);
  upstream vLLM does not serve this model on sm_120.
- engine: `DS_CKPT=... DS_CFG=... ds_bench.sh` run from the QuettaServe
  checkout root (QS_DIR overrides). It sweeps the grid through the engine's
  stock batch_bench test and keeps the full cargo output as the log; the
  engine needs no patch, table.py reads the `[batch_bench]` summary lines
  directly. The bench reports the median of the graph-replay decode steps,
  the steady-state marginal step -- the same quantity as the baseline slope,
  so the rows pair. DS_WORLD defaults to the workload's tp so the cells match
  the baseline's. A host without cargo (the air-gapped GPU nodes) sets
  DS_BENCH_BIN to a cross-built copy of the test binary
  (`cargo test -p deepseek --test batch_bench --release --no-run` emits it).
- table: `table.py deepseek-bench.log baselines/vllm-deepseek.json`.

KV format differs by design: the engine caches the MLA latent in bf16, the
fork in fp8, and the comparison is each engine in its shipped configuration.
The bench line does not state its KV format, so the table prints `KV ?/fp8`
rather than hiding the difference -- read `?` as the engine's bf16. Lines
from truncated-model runs (layers != all) are ignored, not compared. The
grid is ctx {1024, 8192, 16384} x bs {1, 4, 16, 64}.

Correctness for deepseek does not use logit_agreement (the transformers
reference has no DeepSeek-V4 model). It comes from the engine's own gates:
batch_gate / tp_batch_gate prove the lockstep batched decode is bit-exact
against B independent single-sequence decodes (logits within 1e-5, argmax
exact bar head-GEMM ties), and greedy_agreement compares engine vs fork text.

Assumptions to confirm on the first run, each a one-field change if wrong: the
fork accepts `kv_cache_dtype=fp8` and `enable_expert_parallel=True`; DS_WORLD
equals the workload tp (8); all 12 cells produce a RESULT (no capacity SKIP).
Flip `verified` to true only after the numbers are checked.

## Files

| file | role |
|---|---|
| `xval.sh` | the orchestrator: baseline if missing, logit check, greedy check, table, PROF=1 profiles |
| `prof.sh` | nsys capture of one command into profiles/, with the kernel summary dumped as text |
| `vllm.sh` | baseline capture: picks gpus, runs vmin_fit, caches the json |
| `ds_bench.sh` | deepseek engine sweep: runs the stock batch_bench per grid cell from a QuettaServe checkout; its log feeds table.py |
| `vmin_fit.py` | vLLM decode slope fit, ms per step and r2 per cell; standalone as `vmin_fit.py <crate> [grid] --model <dir>` (needs vllm) |
| `cache.py` | packs a raw run into the cached baseline json |
| `table.py` | the comparison: engine bench log vs baseline at matched (ctx, bs, tp) |
| `greedy_agreement.py` | engine binary vs vLLM on shared prompts, leading-token agreement (needs vllm) |
| `logit_agreement.py` | vLLM vs the transformers reference, per-position argmax and true-token logprob (needs vllm, transformers) |
| `with_gpu.sh` | run a command on an idle gpu, which `free_gpu.sh` picks; the engine bench recipes call it |
| `workloads.json` | per-crate model, dtype, maxlen, tp and the (ctx, batch) grids |

## Tests

`tests/test_crossval_contracts.py` runs without a gpu: scripts parse, shell
helpers keep their exec bit, workloads stay consistent, and the grids cover
the engine bench cells. `tests/test_crossval_smoke_gpu.py` runs the baseline
on the one-cell grid where a gpu and MODEL exist. Baselines are generated
and stay untracked.
