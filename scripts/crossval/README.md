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
  `ALLOW_UNVERIFIED=1 vllm.sh deepseek "" <weights> <py>` -- the empty grid
  argument keeps the output at `baselines/vllm-deepseek.json`; a named grid
  writes `vllm-deepseek-<grid>.json`, which the table command below would not
  find. The workload sets `expert_parallel` (vmin_fit passes
  `enable_expert_parallel` to vLLM) and `tp: 8`. `<py>` must be the DeepSeek
  vLLM fork (ununnilium/vllm-ds4-sm120); upstream vLLM does not serve this
  model on sm_120.
- engine: `DS_CKPT=... DS_CFG=... quettabench/scripts/crossval/ds_bench.sh`
  from the QuettaServe checkout root (QS_DIR overrides). It sweeps the grid
  through the engine's stock batch_bench test; table.py reads the
  `[batch_bench]` summary lines directly, and the median of the graph-replay
  steps is the same marginal quantity as the baseline slope, so the rows
  pair. DS_WORLD defaults to the workload's tp. A host without cargo sets
  DS_BENCH_BIN to a prebuilt copy of the test binary
  (`cargo test -p deepseek --test batch_bench --release --no-run` emits it).
  DS_CKPT must point at the TP-sliced fp4 checkpoint (e.g. ds-0731-tp8-fp4).
  The expert-parallel fp8 layout (ds-*-mp8) carries per-rank expert skew and
  is not what the reference numbers used.
- table: `table.py deepseek-bench.log baselines/vllm-deepseek.json`.

KV format differs by design: the engine caches the MLA latent in bf16, the
fork in `fp8_ds_mla` (vLLM's DeepSeek MLA fp8 layout), and the comparison is
each engine in its shipped configuration. The bench line does not state its
KV format, so the table prints `KV ?/fp8_ds_mla` -- read `?` as the engine's
bf16. Lines from truncated-model runs (layers != all) are ignored, not
compared. The grid is ctx {1024, 8192, 16384} x bs {1, 4, 16, 64}. The
workload's `kv_bytes` 24768 = (512 latent + 64 rope) x 43 layers at one byte,
the baseline layout's per-token size; it only gates the capacity SKIP.

Two pairing caveats. The slope is evaluated around ctx+230 tokens while the
engine's median sits near ctx+50, so the 1024 row carries up to ~17% more KV
on the vLLM side (a wash for MLA sparse attention in practice, ~1% at 16384).
And the two sides build prompts from different token-id patterns, so a MoE
batch need not route to the same experts -- the rows pair on shape, not on
identical inputs.

Correctness for deepseek does not use logit_agreement (the transformers
reference has no DeepSeek-V4 model); through the orchestrator that means
`ALLOW_UNVERIFIED=1 LOGIT=0 xval.sh deepseek ...`, skipping the logit stage.
It comes from the engine's own gates:
batch_gate / tp_batch_gate prove the lockstep batched decode is bit-exact
against B independent single-sequence decodes (logits within 1e-5, argmax
exact bar head-GEMM ties). greedy_agreement is llama-only; the orchestrator
never runs it for deepseek.

Assumptions to confirm on the first run, each a one-field change if wrong: the
fork accepts `kv_cache_dtype=fp8_ds_mla` (plain `fp8` is confirmed accepted;
fall back to it and note that here if the MLA layout is rejected) and
`enable_expert_parallel=True` (confirmed); DS_WORLD equals the workload tp
(8); all 12 cells produce a RESULT (no capacity SKIP). Flip `verified` to
true only after the numbers are checked.

## NCCL collective pinning (link profiles)

The collective env depends on the node's GPU interconnect, selected by
`XVAL_LINK_PROFILE` (auto-detected from `nvidia-smi topo -m`; any NV* link
means nvlink, else pcie).

pcie profile: `ds_bench.sh` and `vllm.sh` both export
`NCCL_ALGO=allreduce:tree;allgather:ring NCCL_PROTO=Simple`, so both sides
use the same NCCL path and the comparison is impartial. The per-collective
form is required: plain `NCCL_ALGO=Tree` is rejected for the padded-logits
all-gather ("NCCL has no Tree all-gather"). Node env defaults on this
profile (all overridable by the caller):
- `NCCL_P2P_LEVEL=SYS` — without SYS, cross-socket tree links fall back to TCP (~4x slower)
- `LLMSRV_NO_NVLS=1`
- `LLMSRV_TWOSHOT_MIN_BYTES=0`
- `NCCL_IB_DISABLE=1`
- `NCCL_SOCKET_IFNAME` — unset by default (empty values are never exported);
  set the node's fast NIC in `xval.yaml` or the environment when NCCL picks
  the wrong interface

nvlink profile: none of the pins are exported (only `NCCL_IB_DISABLE=1`).
NCCL then picks NVLS/LL128 and the engine keeps its peer-allreduce. Pinning
Tree/Simple on an NVLink mesh is not impartial: vLLM's TP allreduce bypasses
NCCL_ALGO, so the pin slows only the engine. `vmin_fit.py` records the
active profile as `META link_profile=` and `ds_bench.sh` in its `>>>`
headers; a `collective_nvlink` block in `xval.yaml` can pin values there.

Cells with `bs >= comm_bound_bs` (default 64, set per-workload in
`workloads.json`) are collective-bound: at that batch size the TP allreduce
dominates decode latency and its cost is node- and NCCL-specific. `table.py`
flags these rows `COMM` and prints a footnote; cross-node reference anchoring
is not meaningful for them.

Runtime note: on Blackwell (sm_120 / RTX PRO 6000) the engine requires NCCL >=
2.28. The system-shipped 2.25.1 fails with "invalid argument" at enqueue. Point
`LD_LIBRARY_PATH` at a newer `libnccl.so` before running `ds_bench.sh`.

## max_seq basis

`ds_bench.sh` sets `max_seq = ctx + STEPS + 28` (ctx+128 at STEPS=100).
Declared KV capacity is scored every step for the graph-replay indexer, so
the headroom changes the engine number. The vLLM baseline does not vary with
`max_model_len`. A fair table must fix and state the basis used.

## Config (xval.yaml)

All user-editable knobs live in `scripts/crossval/xval.yaml`. Edit that file to
change timing steps, NCCL settings, headroom, or workload fields. Env vars and
CLI arguments still override the YAML values — the precedence chain is:

```
caller env > xval.yaml > hardcoded defaults in xval_config.py
```

`xval.example.yaml` documents every field with its default and a one-line
explanation of what it controls. It is valid YAML and kept in sync with the
active config structure.

`xval_config.py` is the single loader: it reads `xval.yaml` (via PyYAML),
merges over the defaults, and exposes `load()`, `collective()`, `run_params()`,
`workloads()`, `grids()`, and `step_points()`. All scripts import it; none
duplicate the default values.

## Files

| file | role |
|---|---|
| `xval.yaml` | active user-editable config: run params, collective env, workload fields |
| `xval.example.yaml` | fully-commented reference showing every field and its default |
| `xval_config.py` | single config loader: reads xval.yaml, merges defaults, exposes helpers |
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

## Trace workloads (static grid vs serving replay)

Two ways to feed request traces through the cross-validator; they answer
different questions and their numbers never share a column:

- Static grid: `PROMPT_MODE=trace XVAL_TRACE=t.jsonl ... vmin_fit.py` fills the
  usual (ctx, bs) cells from trace prompts (arrivals ignored). Works for BOTH
  engines: the vLLM side directly, the engine side by prefilling slots from the
  same JSONL. This is the apples-to-apples decode cost.
- Serving replay: `trace_serve.py t.jsonl --model <dir>` submits requests at
  their arrival times, open loop, and reports per-request TTFT/TPOT plus a
  SERVESUM aggregate. vLLM only today: QuettaServe has no continuous batching,
  prefix caching, or query routing, so it cannot take an open-loop stream; its
  comparable number stays the static grid until those land, at which point an
  engine adapter emitting the same SERVE/SERVESUM lines drops in.

Trace sources, one JSONL schema (`synth_bench.py` docstring has the fields):

- `synth_bench.py gen --profile uniform --n 64 --ctx 8192` distinct
  routing-diverse streams for a clean grid cell.
- `synth_bench.py gen --profile swebench --n 200 --groups 8 --rate 2` a
  SWE-bench-agent-shaped stream: long repo-context prompts (4k-24k), short
  outputs (128-1024), sessions sharing a prompt prefix (what a prefix cache
  would hit), bursty arrivals. Deterministic from the seed.
- Mooncake captures: `input_length`/`output_length`/`timestamp` rows load
  directly (timestamp ms becomes arrival_ts seconds).

`--prefix-caching` is off by default in trace_serve so both engines pay full
prefill; turn it on to measure the cache. That asymmetry is the point: the
serving report shows what the missing features cost, the static grid shows
what the kernels cost.

| `synth_bench.py` | trace generator/loader: uniform + swebench profiles, Mooncake JSONL |
| `trace_serve.py` | open-loop trace replay, TTFT/TPOT per request + SERVESUM (needs vllm) |

## Tests

`tests/test_crossval_contracts.py` runs without a gpu: scripts parse, shell
helpers keep their exec bit, workloads stay consistent, and the grids cover
the engine bench cells. `tests/test_crossval_smoke_gpu.py` runs the baseline
on the one-cell grid where a gpu and MODEL exist. Baselines are generated
and stay untracked.
