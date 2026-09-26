# crossval internals

Working notes for agents and contributors. The human quickstart is README.md.

## Provisioning

`oneshot.sh <workload...>` is the empty-container entry: it runs `bootstrap.sh`
then, per workload, `bench.sh` + `xval.sh`. `bootstrap.sh` is idempotent and
`XVAL_DRYRUN=1` prints the plan without acting. It resolves and writes
`.xval_env` (gitignored, `export VAR=value`, sourced by oneshot.sh and xval.sh):

1. cache dirs off the small container root (VLLM_CACHE_ROOT, TORCHINDUCTOR_CACHE_DIR,
   HF_HOME, TMPDIR under `provision.cache_root`)
2. a vLLM python: `provision.py`/`XVAL_PY` if `pip show vllm` passes, else
   `pip install provision.vllm_spec`
3. `provision.nccl_lib` prepended to LD_LIBRARY_PATH (Blackwell needs NCCL >= 2.28)
4. per workload: the engine bench binary (`<ENV>_BENCH_BIN`, or built from a
   QuettaServe checkout / `provision.qs_repo`), the engine checkpoint
   `XVAL_CKPT_<wl>` (+ `XVAL_CFG_<wl>` for moe), and the vLLM weights
   `XVAL_WEIGHTS_<wl>` (a dir, or `XVAL_HF_<wl>` to download)

Anything it cannot resolve is printed as `MISSING:` and the run stops; nothing
is guessed. `provision` config keys: cache_root, py, vllm_spec, qs_repo,
qs_commit, nccl_lib (xval.yaml or `xval_config.py provision`).

## Pipeline

1. `vllm.sh <workload> [grid] <weights-dir> [python]` captures the vLLM
   baseline: `vmin_fit.py` slope-fits ms per decode step for every (ctx, batch)
   cell with r2 recorded, `cache.py` writes `baselines/vllm-<workload>.json`.
   A named grid writes `vllm-<workload>-<grid>.json` so scratch grids never
   clobber the real baseline.
2. `bench.sh <workload> [out.log]` sweeps the engine side. The per-model pieces
   come from the workload's `bench` section in `workloads.json`:
   `crate` and `test` name the cargo test, `env` is the engine's variable
   prefix (`DS_`, `QW_`), `ok` is the per-cell pass pattern, `family` selects
   the model-specific setup:
   - `moe` (deepseek): requires `<env>_CFG`, unsets `<env>_LAYERS` (layer
     truncation would invalidate the run).
   - `gdn` (qwen3 and other linear-attention hybrids): kernel-suite env.
     `QS_KERNELS=exact` is selected only when `QS_VLLM_GDN_DIR` holds exported
     cubins (the loader panics without them), else the tiled rust suite;
     `QS_CUSTOM_AR=1` at tp >= 2; the exact suite has no CUDA-graph wiring so
     `<env>_MODE` defaults to `eager` for it, `graph` otherwise.
   - dense models need no extra block.
   A host without cargo sets `<env>_BENCH_BIN` to a prebuilt test binary
   (`cargo test -p <crate> --test <test> --release --no-run` emits it).
3. `table.py <bench-log> <baseline.json>` pairs the two at matched
   (ctx, batch, tp) and prints ours vs vLLM per cell.

Serving replay is an optional last stage: set `XVAL_SERVE_TRACE=<jsonl>` (and
`XVAL_SERVE_ARGS` for extra trace_serve flags) and xval.sh replays it on the same
GPUs, printing the workload_hash, `WORKLOADSUM` (identity gate over completed
requests) and `SERVESUM`. The hash also stamps the static grid path
(`vmin_fit` META workload_hash), so every run records which exact workload it saw.

Correctness runs alongside: `greedy_agreement.py` (engine vs vLLM text, llama
only) and `logit_agreement.py` (vLLM vs the transformers reference; skipped for
deepseek, which has no transformers model; its correctness comes from the
engine's own batch_gate / tp_batch_gate).

## Measurement contract

Both columns time the same quantity: steady-state decode ms per token at a
resident batch, pipelined, synchronized only at the ends, at matched
(context, batch, tp). The vLLM side is the slope of generate time over the
step points; the engine side is the bench's pipelined loop (K steps, one
sync). Criterion decode groups synchronize inside every iteration, a different
quantity, so table.py prints those cells with ratios withheld
(`XVAL_ALLOW_METHOD_MISMATCH=1` forces them). Baselines record KV dtype per
point; mismatched KV rows are flagged `KV`. Rows at
`bs >= comm_bound_bs` are flagged `COMM`: the TP allreduce dominates there and
the cost is node- and NCCL-specific, so cross-node anchoring is not meaningful.

`max_seq = ctx + timing_steps + max_seq_headroom` changes only the engine
number (declared KV capacity is scored every step); the vLLM side uses the
workload's `maxlen`. Fix and state the basis for every table.

## prompt modes (vmin_fit)

`PROMPT_MODE` env or per-workload `prompt_mode`; matters for MoE routing:

- `distinct` (default): per-slot synthetic seeds, spread routing
- `clone`: the engine benches' exact ids on every slot, identical routing on
  both sides
- `corpus`: sliding windows over the text at `XVAL_CORPUS` (required; no
  corpus ships in the repo)
- `synth`: distinct routing-diverse streams from synth_bench.py
- `trace`: replay the JSONL at `XVAL_TRACE`

cache.py records the mode; table.py prints it in the baseline header.

## Trace workloads

Agentic profiles (`synth_bench.py` `AGENTIC`): terminal_bench, deep_research,
deep_search, osworld (carries image_tokens), rl_cf. Each emits multi-turn
sessions with a carried prefix and context growing per turn (session_id + turn
fields). The token/turn spans are calibratable synthetic defaults, not measured
captures; real captures replace them via load_trace.

One JSONL schema (see synth_bench.py): `prompt_token_ids`, or `prompt_len`, or
Mooncake `input_length`/`output_length`/`timestamp` (ms -> `arrival_ts`
seconds); optional `session_id` and `output_token_ids` (the real assistant
tokens, kept for SPECSUM acceptance). `trace_tokenize.py <agent-bench.jsonl>
--model <dir>` turns a real agent-bench capture (per-turn prompt_text plus
reasoning_text/content_text) into this shape with the model's tokenizer. Static grid (`PROMPT_MODE=trace`) ignores
arrivals and fills the usual cells: the apples-to-apples decode cost, works
for both engines. Serving replay (`trace_serve.py`) submits at arrival times,
open loop, reports TTFT/TPOT p50/p90/p95/p99 per SERVESUM; vLLM only until
QuettaServe gains continuous batching, prefix caching, and query routing --
an engine adapter emitting the same SERVE/SERVESUM lines drops in then.
`--prefix-caching` is off by default so both engines pay full prefill.

## Cross-engine (TRT-LLM)

The benchmark client is engine-agnostic: `src/engines/openai_*` drive any
OpenAI-compatible server, so a `trtllm-serve` endpoint is measured through the
same TTFT/TPOT path as vLLM, and the workload-identity gate holds across engines.
`xval_config.py engine trtllm` gives the serve command, health path, and the
engine-dir convention; `src/engines/trtllm.py` builds the launch and health URL.

The one-time TensorRT engine build is multi-hour and GPU-specific, so it is not
automated. Build it by hand first (`src/engines/trtllm.py build_command` prints
the exact `trtllm-build` invocation), then serve. `amd`/`rocm` is a stub that
fails with "no ROCm host provisioned" until hardware exists; backend detection
stays vendor-neutral.

## Cross-accelerator portability

`xval_config.py backend` detects the accelerator (`XVAL_BACKEND` override, else
nvidia-smi -> cuda, rocm-smi -> rocm, neuron-ls -> neuron, libtpu -> tpu, else
cpu). Only idle-GPU picking (`free_gpu.sh`) and link-profile detection are
cuda-specific, and both are guarded on the backend: non-cuda honors a caller-set
CUDA_VISIBLE_DEVICES (or 0..N-1) and takes the no-pins profile. Everything else
is backend-agnostic Python.

What each backend needs to actually run, and current status:

| backend | vLLM baseline | QuettaServe engine | harness/orchestrator | to run |
|---|---|---|---|---|
| cuda (H200 / H100 / RTX PRO 6000) | native | CUDA sm_90a / sm_120a | works, validated | nothing |
| rocm (MI300X) | vLLM ROCm build | none (CUDA only) | works (backend=rocm) | ROCm vLLM image; RCCL |
| tpu (v5e / v6e / Ironwood) | vLLM-TPU / JAX, limited | none | works (backend=tpu) | vLLM-TPU backend; no NCCL |
| neuron (Trainium) | vLLM-Neuron, limited | none | works (backend=neuron) | Neuron SDK; no NCCL |

Not yet run on a real non-cuda device (no TPU/AMD instance available here); the
cuda path and the nvidia-smi-absent path are both tested. The engine stays
NVIDIA-only until a HIP/XLA/Neuron port exists, so on other backends the near-term
deliverable is the vLLM-baseline column plus the full workload/identity machinery.

## NCCL link profiles

`XVAL_LINK_PROFILE` selects the collective env; one detection
(`xval_config.py link-profile`: any NV* link in `nvidia-smi topo -m` = nvlink,
else pcie, non-cuda = nvlink) feeds bench.sh, vllm.sh and vmin_fit identically.

- pcie: `NCCL_ALGO=allreduce:tree;allgather:ring NCCL_PROTO=Simple`
  (per-collective form required; NCCL has no Tree all-gather),
  `NCCL_P2P_LEVEL=SYS` (without it cross-socket links fall back to TCP, ~4x
  slower), `LLMSRV_NO_NVLS=1`, `LLMSRV_TWOSHOT_MIN_BYTES=0`, `NCCL_IB_DISABLE=1`.
- nvlink: nothing exported except `NCCL_IB_DISABLE=1`. NCCL picks NVLS/LL128
  and the engine keeps its peer-allreduce; pinning Tree/Simple there slows only
  the engine because vLLM's TP path bypasses NCCL_ALGO.

Empty yaml values are never exported (an exported empty NCCL var is not the
same as an absent one). Blackwell (sm_120) needs NCCL >= 2.28 on
LD_LIBRARY_PATH; the shipped 2.25.1 fails at enqueue.

## Config reference (xval.yaml)

Precedence: caller env > xval.yaml > defaults in xval_config.py.
`xval_config.py` is the single loader (`load()`, `collective()`, `run_params()`,
`workloads()`, `grids()`, `step_points()`); workloads merge over workloads.json.

- `run.world`: fallback world size when a workload has no `tp`
- `run.timing_steps`: decode steps timed per cell (`<env>_TIMING_STEPS` overrides)
- `run.max_seq_headroom`: slack over ctx + steps; the bench asserts
  prompt + steps + 8 <= max_seq
- `collective.*`: the pcie-profile pins listed above; `collective_nvlink`
  optionally pins values on the nvlink profile
- `step_points`: three increasing token counts for the slope fit; every cell
  needs ctx + max(step_points) + 2 <= maxlen or vmin_fit SKIPs it
- `workloads.<name>`: `tp`, `grid`, `dtype`, `maxlen`, `kv_dtype`, `kv_bytes`,
  `expert_parallel`, `comm_bound_bs`, `prompt_mode`, `verified`, `bench`
  (crate/test/env/family/ok as above), plus bookkeeping fields
  (`engine_commit`, `checkpoint`)
- `grids.<name>`: [ctx, bs] cell lists

## GLM-5.3 bring-up notes

zai-org/GLM-5.3: MLA + DSA-indexed MoE (78 layers plus one MTP layer, 256
routed experts with 8 active, first 3 layers dense), pre-quantized fp8
checkpoint of 704 GiB, so tp 8 on H200. kv_bytes is the MLA latent:
78 x (kv_lora_rank 512 + qk_rope_head_dim 64) x 2 bytes = 89856; the dense
per-head formula would overestimate KV several-fold. The DSA indexer keeps its
own small per-token cache on top of this, so treat capacity estimates as
slightly optimistic until measured. quant: fp8 flows into vLLM as quantization
and into the table's QUANT pairing rule. A transformers reference exists
(glm_moe_dsa), so the FIDELITY gate applies, unlike deepseek. The engine side
lives on QuettaServe branch kev/glm (GLM-5.3 crate skeleton: dsa/mla/moe/mtp
plus a two-node PP split); it has no decode bench test yet, so this workload
carries no bench section until that lands and is vLLM-baseline plus serving
replay only. Do not confuse kev/glm with kev/glm53flash: the latter is
GLM-5.3-Flash (Glm5Next, hybrid KDA plus sparse MLA), a different model.
comm_bound_bs lands after the first measured grid, never
copied from another model. verified flips true only after a clean golden run
(ALLOW_UNVERIFIED=1 until then).

## DeepSeek bring-up notes

vLLM baseline: `ALLOW_UNVERIFIED=1 vllm.sh deepseek "" <weights> <py>`; `<py>`
must be the DeepSeek vLLM fork; upstream does not serve this model on
sm_120. The workload sets `expert_parallel` and `tp: 8`. Engine:
`DS_CKPT=<tp-sliced-fp4> DS_CFG=<inference/config.json> bench.sh deepseek`.
DS_CKPT must be the TP-sliced fp4 layout (ds-0731-tp8-fp4), not the mp8
expert-parallel layout (per-rank expert skew, does not match the reference).
KV differs by design (engine bf16 MLA latent vs fork `fp8_ds_mla`): the table
prints `KV ?/fp8_ds_mla` and withholds the ratio. The slope evaluates near
ctx+230 vs the engine median near ctx+50 (~17% more KV at the 1024 row, ~1% at
16384). `verified` flips true only after a clean golden run.

## Files

| file | role |
|---|---|
| `xval.yaml` | active config (all fields optional; defaults in `xval_config.py`) |
| `xval_config.py` | single config loader |
| `oneshot.sh` | empty container to full run: bootstrap then sweep+table per workload |
| `bootstrap.sh` | provision the box, write `.xval_env` |
| `xval.sh` | orchestrator: baseline if missing, logit/greedy checks, table, PROF=1 |
| `bench.sh` | engine sweep for any workload with a `bench` section |
| `vllm.sh` | baseline capture into baselines/ |
| `vmin_fit.py` | vLLM decode slope fit per cell |
| `cache.py` | packs a raw run into the baseline json |
| `table.py` | the comparison at matched (ctx, bs, tp) |
| `synth_bench.py` | trace generator/loader (uniform, swebench, Mooncake) |
| `trace_tokenize.py` | agent-bench capture -> tokenized replay trace |
| `trace_serve.py` | open-loop serving replay (vLLM) |
| `greedy_agreement.py` / `logit_agreement.py` | correctness lanes |
| `prof.sh` / `free_gpu.sh` / `with_gpu.sh` | nsys capture, GPU picking |
| `workloads.json` | canonical per-workload fields and grids |
