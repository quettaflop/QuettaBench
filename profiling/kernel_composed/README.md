# kernel_composed probes — tables for QuettaSim

Writes the `graph/` vs `eager/` layout that QuettaSim `kernel_composed`
interpolates. Not the live-server / serving-wall probes in `profiling/probes/`.

The `kernel_composed` cost backend prices each serving step from **measured**
kernel tables in QuettaSim's `data/kernel_data/`. To make the simulator support a
new GPU, you profile that GPU's kernels once and drop the tables into that tree.
This harness automates the collection.

All probes are **CUDA-event timed** against the production kernels (vLLM's
`flash_attn_varlen_func`, `torch.matmul`, vLLM's fused MoE / Gated-DeltaNet ops), in
one of two modes (`probes/_timing.py`): **graph** (the op captured into a CUDA graph
and its replay timed -- dispatch-free, how vLLM runs decode; tables go to `graph/`)
or **eager** (each launch timed as issued, dispatch included -- how vLLM runs
prefill; tables go to `eager/`). No NCU required, so a user/client can run them with
just a torch+vLLM environment; `*-ncu` recipes exist only as a cross-check. Every
table is written with a `<table>.meta.json` sidecar (`probes/_manifest.py`) naming
the mode, tool, reduce, host and software versions.

```bash
# from a QuettaSim checkout (QuettaBench as submodule):
cd QuettaBench/profiling/kernel_composed
just gpu=A100 profile-gpu
just gpu=A100 flash tag=tp1 nh=32 nkv=8 hd=128

# standalone Bench: point at the Sim tree
QUETTASIM=/path/to/QuettaSim just gpu=A100 profile-gpu
```

Tables land in `$QUETTASIM/data/kernel_data/` (or `KERNEL_DATA`). Schema
contract: filenames `{gpu}[_fp8].csv`, method split `graph/` (dispatch-free, decode)
vs `eager/` (dispatch included, prefill), columns as the current tables, one
`.meta.json` beside each table.

## Requirements
- `just` (recipe runner), and on the GPU box: python with `torch` + `vllm`.
- The target GPU. Recipes run locally, or over ssh on a fleet host (below).
- These probes import only `torch` (+ `vllm` for `moe_probe.py`) and write CSV/JSON
  with the stdlib — **no pandas**. Worth keeping that way: LLMServingSim's profiler
  cannot run in its own documented image (`vllm/vllm-openai:v0.19.0` ships no pandas,
  which its `skew.py` and `meta.yaml` writer both import). Check before booking GPU
  time on a stock vLLM container.

## Quick start — add GPU `X`

1. **Datasheet peaks** — create `device_spec/X.yaml` (in QuettaSim) with `name: X`,
   `compute.peak_flops_per_s`, `memory.peak_bw_bytes_per_s`, `memory.total_bytes`
   (util_flops/util_bw are set in step 3). The YAML `name` MUST match the
   `--gpu-label` you profile with and the GT dir prefix.

2. **Collect tables** (from this folder):
   ```
   just gpu=X profile-gpu                         # gemm + elementwise (GPU-generic)
   just gpu=X flash tag=tp1 nh=<Hq> nkv=<Hkv> hd=<head_dim>   # per model head config
   just gpu=X flash tag=tp2 nh=<Hq/2> nkv=<Hkv/2> hd=<head_dim>   # sharded, if you run tp2
   just gpu=X gdn geom=9b                         # ONLY for Qwen3.5-style hybrids
   just gpu=X moe backend=flashinfer              # ONLY if you serve MoE models
   ```
   Every recipe is graph-replay timed by default; `*-eager` writes the eager
   companion. Tables land under `$QUETTASIM/data/kernel_data/`:
   `graph/gemm/X.csv`, `graph/elementwise/X.json`,
   `graph/flash_attn/X.csv` (decode) + `eager/fa3_prefill/X.csv` (prefill),
   `graph/gdn/X_<geom>_decode.csv` + `eager/gdn/X_<geom>_prefill.csv`,
   `graph/moe/X.csv` + `graph/moe/X_routing.json`, each with a `.meta.json`.
   Then `just -f $QUETTASIM/justfile kernel-manifest` to check the manifests.

3. **Derive `util_flops` / `util_bw`** for `X.yaml` — these anchor the analytic
   roofline *fallback* (used for off-grid cells). Read them off the collected
   grids: `just -f $QUETTASIM/justfile derive-device gpu=X` prints both (does not
   edit the YAML).

4. **Serving-dynamics blocks** for `X.yaml` (`frontend:` / `tp_comm:` / `serving:`) — the
   API-server host cost, tp>1 all-reduce, and per-step host overhead the kernel tables don't
   capture. Without them a GPU prices serving as a bare compute roofline (no frontend/comm/
   overhead) — the reason A100 serving lagged H100. If `X` already has GT:
   ```
   just -f $QUETTASIM/justfile serving-fit gpu=X model=<Model>   # fits frontend:+serving: FROM GT (no GPU)
   just gpu=X tp-comm tp=2 hidden=<H> layers=<L>                 # measures tp_comm: (needs 2 GPUs, torchrun)
   ```
   If `X` has no GT yet, `just gpu=X serving-profile …` measures frontend:+serving: against a
   live `vllm serve`. Each prints a YAML block to paste into `X.yaml`.

5. **Validate** against GT: `just -f $QUETTASIM/justfile xval serving` (or the
   suite that matches your GT) and check the cell-MAPE. See QuettaSim's
   `engine/README.md`.

## Running on a remote fleet host (ssh)

Set three env vars; tables sync back into your local `kernel_data` automatically.
Pick a **free** GPU (`just gpu-info`) on shared hosts.
```
REMOTE_HOST=a100 GPU_ID=0 \
PYTHON_BIN=/home/<user>/miniconda3/envs/vllm/bin/python \
  just gpu=A100 profile-gpu
```
`run_probe.sh` scps the probe to the host, runs it on `GPU_ID`, and rsyncs the
tables back. On shared machines profile a free GPU and clean up `/tmp/qs-profiling`
when done.

## Probes (`probes/`)
| probe | table | axis |
|---|---|---|
| `gemm_probe.py` | `{graph if --graph/--ncu else eager}/gemm/{gpu}[_fp8].csv` (`--wide` → `gemm_wide/`) | M × (N,K) |
| `flash_attn_probe.py` | decode → `{graph if --graph/--ncu else eager}/flash_attn/{gpu}.csv`; prefill → `eager/fa3_prefill/{gpu}.csv` (always) | decode (kv×batch), prefill (seq) |
| `cross_attn_probe.py` | `eager/fa3_cross/{gpu}.csv` | chunked-prefill attention PER LAYER over (q_len × resident_tokens) — the rectangular shape neither the decode (q=1) nor the full-causal prefill grid covers; on long-ISL traces it dominates prefill attention FLOPs. One run per head config (`--n-heads/--n-kv-heads/--head-dim`), upserts |
| `collective_probe.py` | `graph/collectives/{gpu}_tp{N}.csv` | plain-NCCL collectives over decode-step payloads (torchrun); also prints a legacy `tp_comm:` YAML block |
| `vllm_allreduce_probe.py` | `graph/collectives/{gpu}_tp{N}.csv` (all_reduce rows, `path=vllm`) | torchrun; times TP all-reduce through vLLM's OWN `GroupCoordinator` (custom allreduce on) next to plain NCCL. NOTE: vLLM ≥0.27 fuses allreduce+rmsnorm under torch.compile (`fuse_allreduce_rms`), so this standalone grid over-prices graphed decode — see h200.yaml `tp_comm.source` |
| `elementwise_probe.py` | `{graph if --graph else eager}/elementwise/{gpu}.json` | affine `floor_us + bytes/eff_bw` per kernel. **Prefer `--vllm --graph`** (needs a vLLM env): times the REAL fused kernels (`rms_norm`, `fused_add_rms_norm`, `silu_and_mul`, `rotary_embedding`, `reshape_and_cache_flash`) instead of multi-kernel torch-chain proxies, and adds the `fused_add_rmsnorm` + `kv_cache_write` entries the composition prefers when present (each fused entry replaces a two-kernel proxy pair). The torch-chain default overpriced vLLM's fused ops ~2-3 ms/step at 48 layers |
| `gdn_probe.py` | decode → `{graph if --graph/--ncu else eager}/gdn/{gpu}_{geom}_decode.csv`; prefill → `eager/gdn/…_prefill.csv` (always) | Gated-DeltaNet; prefill (seq), decode (batch) |
| `moe_probe.py` | `{graph if --graph else eager}/moe/{gpu}[_fp8].csv` + `…/{gpu}_routing.json` (`--backend triton|flashinfer`) | tokens × (n_experts, top_k, intermediate, hidden) |
| `reparallel_probe.py` | `eager/reparallel/{gpu}_reshard[_dp\|_ep].csv` | torchrun; NVLink cost of switching parallelism in place (tp reshard, dp replica broadcast, ep expert restage; `--mode xnode/plan` for cross-node, see the probe header) |
| `kv_transfer_probe.py` | `eager/kv_transfer/{gpu}.csv` | PD KV hand-off through NIXL/UCX, laid out as vLLM's NixlConnector issues it (descriptor list per (layer, block)) |
| `pd_host_probe.py` | `eager/pd_host/{gpu}.csv` | per-request HOST cost of a PD pair by prompt length (parse/template/tokenize/hash, paid on P and again on D) |
| `decode_batch_probe.py` | mixed-step profile CSV | controlled decode-step-vs-batch curve, pins batched-decode pricing at concurrency |
| `fused_block_probe.py` | mixed-step profile CSV | ONE real transformer-block forward, to check the additivity assumption in `fused_step_ms` without a checkpoint |

`serving_frontend_probe.py` is the live-server exception in this folder: it fits
the device-YAML `frontend:` + `serving:` blocks against a running `vllm serve`
(recipe: `serving-profile`), for GPUs with no GT yet. With GT, prefer the cheaper
`just -f $QUETTASIM/justfile serving-fit`. It and `pd_host_probe.py` are
consumer-coupled like the two emitters: they import the sim's `engine` package,
so run them from a QuettaSim checkout (or with QuettaSim on `PYTHONPATH`).

### Why `moe_probe.py` exists
Every other kernel family here has a measured table; the grouped MoE FFN did not.
`sum_kernels._moe_ffn_us` priced it with an analytic roofline over touched
bytes/FLOPs, scaled by `util_bw`/`util_flops` — scalars fitted on **dense Llama**
GEMMs. A roofline has no launch cost and no tile quantization, so it is optimistic
exactly where a grouped kernel is launch-bound (128 experts × ~1 token each).
Measured against the lss-valid H100 numbers it is **−51%** at a 1-token decode step,
−14.5% at 8, converging to −4% by 64. That batch-dependent deficit is what the flat
`_decode_host_floor_us` constant patches today — which is why it cannot be right at
more than one operating point. Measure the kernel and the constant can go.

Timing method: the launch-cost rationale above only holds for EAGER execution —
vLLM **CUDA-graphs MoE decode**, so the decode-faithful grid is dispatch-free.
The `moe` recipe passes `--graph` (CUDA-graph-replay timing -> `graph/moe/`, what
the loader prefers); the eager companion (`moe-eager` -> `eager/moe/`) overpriced
H200 graphed decode 2.3x (188us vs 80.8us/layer at 8 tokens). Loading a measured
MoE grid also retires the `_decode_host_floor_us` patch automatically. The
RTXPRO6000 grid was swept with a v2 harness (real routing replayed, weight sets
cycled) that is not yet in this tree — see PROVENANCE.md "MoE grid sweep
methodology" for what the plain probe's uniform-routing sweep gets wrong at decode.

**MXFP4-expert models (gpt-oss) are swept for shape coverage but deliberately are not
priced off this bf16 grid** — only the weight-read term scales with expert dtype, and
a measured latency can't be decomposed after the fact. They keep the roofline until an
MXFP4 grid exists.

## Measurement method (important): decode → graph, prefill → eager

Which timing mode is *faithful* depends on how the kernel runs in serving, so
`kernel_data` is split by mode (`graph/` vs `eager/`) and the loader reads each
kernel from the right one (`kernel_composed/provenance.py` in QuettaSim):

- **Decode is CUDA-graphed** — the graph replays all decode kernels with no
  per-launch CPU dispatch, so dispatch-free time equals the achieved time. Decode
  grids + gemm + elementwise + MoE come from `graph/`, measured by capturing the op
  into a `torch.cuda.CUDAGraph` and timing `replay()` (`probes/_graph.py`; MIN over
  replays). This is what every default recipe does (`gemm`, `elementwise`, `flash`,
  `moe`, `gdn` all pass `--graph`). It needs no profiler counters, so it runs on
  boxes with `RmProfilingAdminOnly=1`, and it runs at the clocks the GPU serves at.
- **Prefill is eager** (dynamic shapes, not graphed) — every kernel pays its
  dispatch cost, which eager timing captures (MEDIAN over calls) and dispatch-free
  timing misses. Prefill grids come from `eager/`. The `flash`/`gdn` probes always
  emit prefill eager (even under `--graph`/`--ncu`, which only switch the *decode*
  grid).

The dispatch overhead is real and host-specific: GDN prefill's 8 tiny Triton
sub-kernels floor at ~500µs on A100 (~62µs/launch) vs far less on H100 — which is
exactly why kernel time alone can't reconstruct prefill.

**NCU is a cross-check, not a source.** `*-ncu` recipes time the same ops with
Nsight Compute `gpu__time_duration` (mode `ncu`, also under `graph/`). The legacy
H100/A100/RTX3090 tables were taken that way at NCU's default base-clock lock and
cache flush, which reads slower than graph replay; `_ncu.py` now runs with
clock/cache control off so a cross-check is comparable. Needs a real `ncu`
(`NCU_BIN=/opt/nvidia/nsight-compute/2024.3.2/ncu`, NOT stock `/usr/bin/ncu`).

Every table's mode is in its `.meta.json` (and the history in
`$QUETTASIM/data/kernel_data/PROVENANCE.md`); keep a GPU/kernel/phase on ONE mode
and re-collect rather than mix -- the loader flags a cross-dir fallback when it
has to.

## Notes / TODO
- `flash` and `gdn` grids are **head-config specific** — run them per model head
  config (and per tp shard), not once per GPU.
- `gdn_probe.py` is first-cut: decode uses the generic recurrent op (the
  serving-faithful decode is `fused_sigmoid_gating_delta_rule_update`); the batch
  grid can OOM past B≈80 on the state path. 27B geometry is `geom=27b hv=48`.
- Serving-dynamics blocks have recipes (see step 4): `serving-fit` / `serving-profile`
  (frontend: + serving:, GT-fit or live-server) and `tp-comm` (tp_comm:, all-reduce microbench,
  `probes/collective_probe.py`). CAVEAT (found applying them to A100 2026-07-21): `serving-fit`'s
  conc1 floor improves TTFT *in isolation* (43.8 -> 31.8) but is NOT a drop-in win -- applied, the
  big floor (~8x H100) with H100's PORTED mult/lanes curves gives pathological high-concurrency
  arrival delays that worsen tp>1 TPOT. The floor and curves must be fit JOINTLY against BOTH
  TTFT and TPOT (`serving-fit` fits only the conc1 floor today). `tp-comm` measures the
  real all-reduce cost fine, but on A100 it over-corrects low-conc TPOT (the real tp2 gap is
  large-batch decode under-pricing). Treat the current output as a starting point, not final.
