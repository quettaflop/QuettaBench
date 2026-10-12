# Kernel probes: tables for QuettaSim

Each run emits `ProbeRow`s into its `rows.jsonl` under `results/probes/<gpu>/`. `just export --only kernel_tables` builds QuettaSim's `kernel_data/graph/` and `kernel_data/eager/` tables from them, the layout `kernel_composed` interpolates. The live-server probes that sit beside them are listed under "Live-server probes" below.

The `kernel_composed` cost backend prices each serving step from measured kernel tables in QuettaSim's `data/kernel_data/`. Supporting a new GPU means profiling its kernels once and exporting the tables into that tree.

All probes time the production kernels (vLLM's `flash_attn_varlen_func`, `torch.matmul`, vLLM's vendored Gated-DeltaNet ops) with CUDA events, so no NCU is required and a torch plus vLLM environment is enough.

```bash
# from the QuettaBench repo root:
just gpu=<gpu> profile-gpu
just gpu=<gpu> flash tag=tp1 nh=32 nkv=8 hd=128

# any probe, flags in one quoted argument
just probe gemm "--gpu-label <gpu> --ncu"
```

Each probe run writes its own `results/probes/<gpu>/<probe>/<run-id>/` (`--out` changes the `results/probes/<gpu>/` part; the run id is the run's UTC start, e.g. `20261011T120000.000000Z`): its `rows.jsonl` and its `run.json` sit in that one dir, so `just export` drops the rows of a run whose `run.json` is not ok, and a later run never overwrites an earlier one. A run writes only the rows it measured, never an earlier run's. The export unions them (`kernel_tables`): a table with a row key is the union of the rows of every ok run of its probe, keyed as the probe's old upsert was (flash_attn and cross_attn on the head config `n_heads, n_kv_heads, head_dim`; gemm on `M, N, K`; moe on `tokens, n_experts, top_k, intermediate, hidden`; reparallel on `gpu, world, tp_from, tp_to, total_weight_gb`; gemm's `fp8_act_quant` on its JSON key), and on a key conflict the run with the latest `provenance.pulled` (else `provenance.finished`) wins. A gemm or moe run without `--append` drops the rows of runs that finished before it. A table with no row key (one table per run: collective, vllm_allreduce, elementwise, gdn, moe routing) comes from the run with the latest `provenance.pulled` (else `provenance.finished`); collective and vllm_allreduce then merge as below. A probe that names its output file (`serving_stage_split --out`, `serving_decode_grid --out-raw`) writes its `run.json` beside that file and refuses a dir that already exists, so give each run a file in a new dir. Every probe run opens its `requests.jsonl` before it starts, so a run killed before it finishes leaves no `run.json` and export skips its rows. A `ProbeRow` names its table as `<kind>/<file>`; its timing (`graph_replay`, `ncu`, `amortized_wallclock`, `eager`, `host_wallclock`, QuettaSim's mode words) picks `graph/` or `eager/`.

The one route to QuettaSim is the `kernel_tables` export builder: `just export --only kernel_tables --results results/probes --kernel-dest <QuettaSim data root>` (`$QUETTASIM_DATA`, or `../QuettaSim/data` for the in-repo tree) builds every `<kind>/<file>` table from the rows of the ok runs of its probe by the rule above into `<dest>/kernel_data/<dir>/<kind>/<file>` (`graph_replay`, `ncu` and `amortized_wallclock` rows to `graph/`, `eager` and `host_wallclock` rows to `eager/`), the paths `kernel_artifact_paths` in QuettaSim's `engine/backends/kernel_composed_cost.py` reads, each with the `<file>.meta.json` sidecar QuettaSim's `tools/kernel_manifest.py check` requires (mode, tool, reduce, gpu, measured). Two merges: the collective probe's and vllm_allreduce's `collectives/{gpu}_tp{N}.csv` rows become one table: the collective probe's rows (its `all_reduce` rows only when no vllm_allreduce run is there), then vllm_allreduce's `all_reduce` rows, which replace every `all_reduce` row of their world; gemm's `fp8_act_quant` joins the elementwise probe's `elementwise/{gpu}.json`. A table QuettaSim already holds is the oldest source: its rows stay unless a run holds their key (collectives on `op, world, bytes`, an empty `op` read as `all_reduce`), its repeated samples on a key included (but reparallel and kv_transfer keep the last), a JSON table's entries likewise, and the merge keeps its line ending, JSON indent and final newline. Its sidecar keeps its other fields, joins qbench's notes to its own and, when its rows stayed, spans `measured` over both and names both tools. Where its rows or columns would be dropped (a table with no row key, a gemm or moe run without `--append`, a run's table without a dest column) the export stops naming the file; `--replace-tables` replaces just those tables and their sidecars, and every other table still merges. `kernel_data/qbench_manifest.json` names the runs with rows in each table (when dest rows stayed, the runs its old entry named first, else `adopted from dest`), with the bench commit and export time, keeps the entries of tables this export did not write and its other keys, and stops the export when it is not an object with a `kernel_tables` object; with no table found it is not written. The R2 mirror (`just mirror-up`, `mirror-down`) is a backup of `results/probes/` under `profiling/qbench_probes/`, in the same `<gpu>/<probe>/<run-id>/` layout; QuettaSim never reads it. Schema contract: filenames `{gpu}[_tpN].csv`, timing split `graph/` (graphed decode) vs `eager/` (eager prefill), columns as QuettaSim's existing tables.

## Requirements
- `just` (recipe runner), and on the GPU box: python with `torch` + `vllm`.
- The target GPU. Recipes run locally, or over ssh on a fleet host (below).
- These probes import only `torch` (plus `vllm` for `moe`) and write JSON lines with the stdlib, no pandas. Keep it that way: LLMServingSim's profiler cannot run in its own documented image (`vllm/vllm-openai:v0.19.0` ships no pandas, which its `skew.py` and `meta.yaml` writer both import). Check before booking GPU time on a stock vLLM container.

## Quick start: add GPU `X`

1. Datasheet peaks: create `device_spec/X.yaml` (in QuettaSim) with `name: X`, `compute.peak_flops_per_s`, `memory.peak_bw_bytes_per_s`, `memory.total_bytes` (util_flops/util_bw are set in step 3). The YAML `name` must match the `--gpu-label` you profile with and the GT dir prefix.

2. Collect tables, from the repo root:
   ```
   just gpu=X profile-gpu                         # gemm + elementwise (GPU-generic)
   just gpu=X flash tag=tp1 nh=<Hq> nkv=<Hkv> hd=<head_dim>   # per model head config
   just gpu=X flash tag=tp2 nh=<Hq/2> nkv=<Hkv/2> hd=<head_dim>   # sharded, if you run tp2
   just gpu=X gdn geom=9b                         # only for Qwen3.5-style hybrids
   just gpu=X moe                                 # only if you serve MoE models
   ```
   Rows land in `results/probes/X/<probe>/<run-id>/rows.jsonl`; the export builds QuettaSim's layout from them (`graph/` = graphed-decode-faithful, `eager/` = eager-prefill-faithful): `graph/gemm/X.csv`, `eager/elementwise/X.json`, `{graph,eager}/flash_attn/X[_tpN].csv` (decode; graph with `flash-ncu`), `eager/fa3_prefill/X[_tpN].csv` (prefill), `{graph,eager}/gdn/X_<geom>_{decode,prefill}.csv`, `eager/moe/X.csv` + `eager/moe/X_routing.json`.

3. Derive `util_flops` / `util_bw` for `X.yaml`: they anchor the analytic roofline fallback used for off-grid cells. Read them off the collected grids: `just -f $QUETTASIM/justfile derive-device gpu=X` prints both (does not edit the YAML).

4. Serving-dynamics blocks for `X.yaml` (`frontend:` / `tp_comm:` / `serving:`): the API-server host cost, tp>1 all-reduce, and per-step host overhead the kernel tables do not capture. Without them a GPU prices serving as a bare compute roofline with no frontend or comm overhead. If `X` already has GT:
   ```
   just -f $QUETTASIM/justfile serving-fit gpu=X model=<Model>   # fits frontend:+serving: FROM GT (no GPU)
   just gpu=X collectives world=2 hidden=<H> layers=<L>          # measures tp_comm: (needs 2 GPUs, torchrun)
   ```
   If `X` has no GT yet, `just gpu=X serving-profile results/<run>` measures frontend:+serving: against that run's live `vllm serve` into its `profile.json` (`frontend` section); the others print a YAML block to paste into `X.yaml`.

5. Validate against GT: `just -f $QUETTASIM/justfile xval serving` (or the suite that matches your GT) and check the cell-MAPE. See QuettaSim's `engine/README.md`.

## Running on a remote fleet host (ssh)

Set `REMOTE_HOST`, `GPU_ID` and `PYTHON_BIN`; `remote-probe` (`probes/base.py` `remote_probe`) makes its own remote dir with `mktemp -d /tmp/qbench-probes.XXXXXX`, rsyncs the probes package into it, runs the probe on `GPU_ID` (`NPROC=2 GPU_IDS=0,1` for the torchrun probes, `NCU_BIN` for `--ncu`) with `--run-id` from the local clock, and always rsyncs the run back into `results/probes/<gpu>/<probe>/<run-id>/`, a failed run's too, never overwriting a local file. The pulled `run.json` keeps the remote `provenance.finished` and gets `provenance.pulled` from the local clock, which export orders runs by; a `run.json` that cannot be stamped is warned about and left. It exits with the remote run's exit code and removes that remote dir alone once the pull succeeds; a failed pull keeps it and prints where. `--gpu-label`, `--gpu`, `--out`, `--out-dir` and `--run-id` are refused among the probe args, since remote-probe sets them. No earlier run goes up, since a run writes only its own rows. Pick a free GPU (`just gpu-info`) on shared hosts.
```
REMOTE_HOST=<host> GPU_ID=0 \
PYTHON_BIN=/home/<user>/miniconda3/envs/vllm/bin/python \
  just remote-probe gemm <gpu> --ncu --wide
```

## Probes
| probe | table (exported dir) | axis |
|---|---|---|
| `gemm` | `gemm/{gpu}.csv` (`--wide` → `gemm_wide/`; `--ncu` or `--graph` → `graph/`, else `eager/`) | M × (N,K) |
| `flash_attn` | decode → `flash_attn/{gpu}[_tpN].csv` (`graph/` with `--ncu` or `--graph`, else `eager/`); prefill → `fa3_prefill/{gpu}[_tpN].csv` (`eager/` always) | decode (kv×batch), prefill (seq) |
| `cross_attn` | `fa3_cross/{gpu}.csv` (`eager/`) | chunked-prefill attention per layer over (q_len × resident_tokens), the rectangular shape neither the decode (q=1) nor the full-causal prefill grid covers; on long-ISL traces it dominates prefill attention FLOPs. One run per head config (`--n-heads/--n-kv-heads/--head-dim`); the export unions runs on the head config |
| `collective` | `collectives/{gpu}_tp{N}.csv` (`graph/`) | plain-NCCL collectives over decode-step payloads (torchrun); also prints a legacy `tp_comm:` YAML block |
| `vllm_allreduce` | `collectives/{gpu}_tp{N}.csv` (all_reduce rows, `path=vllm`; `kernel_tables` merges them over the collective probe's) and `collectives/{gpu}_tp{N}_nccl_vs_vllm.csv` (`graph/`) | torchrun; times TP all-reduce through vLLM's own `GroupCoordinator` (custom allreduce on) next to plain NCCL. vLLM ≥0.27 fuses allreduce+rmsnorm under torch.compile (`fuse_allreduce_rms`), so this standalone grid over-prices graphed decode; see the hardware yaml's `tp_comm.source` in QuettaSim |
| `elementwise` | `elementwise/{gpu}.json` (`graph/` with `--graph`, else `eager/`) | affine `floor_us + bytes/eff_bw` per kernel. Prefer `--vllm --graph` (needs a vLLM env): it times the real fused kernels (`rms_norm`, `fused_add_rms_norm`, `silu_and_mul`, `rotary_embedding`, `reshape_and_cache_flash`) instead of multi-kernel torch-chain proxies, and adds the `fused_add_rmsnorm` + `kv_cache_write` entries the composition prefers when present (each fused entry replaces a two-kernel proxy pair). The torch-chain default overpriced vLLM's fused ops ~2-3 ms/step at 48 layers |
| `gdn` | decode → `gdn/{gpu}_{geom}_decode.csv` (`graph/` with `--ncu`, else `eager/`); prefill → `gdn/…_prefill.csv` (`eager/` always) | Gated-DeltaNet; prefill (seq), decode (batch) |
| `moe` | `moe/{gpu}.csv` + `moe/{gpu}_routing.json` (`graph/` with `--graph`, else `eager/`) | tokens × (n_experts, top_k, intermediate, hidden) |
| `moe --routing` (`just moe-routing`) | `moe_routing/{model}_layer{L}.safetensors` | one MoE block's input, output, routing (topk_idx/topk_weight or router_logits) and stage tensors from one prefill, for an engine-side diff of routing, dispatch and combine; `--layers N` cuts the model like `engine.layers` (slice verification, see README.md "Slice verification") |
| `reparallel` | `reparallel/{gpu}_reshard[_dp\|_ep].csv` (`eager/`) | torchrun; NVLink cost of switching parallelism in place (tp reshard, dp replica broadcast, ep expert restage; `--mode xnode/plan` for cross-node, see the probe header) |
| `kv_transfer` (`quettabench/profiling/kv_transfer.py`, run under torchrun and recorded like a probe; the export unions its runs on the old upsert key; the profiler source reads one run's `rows.jsonl`) | `kv_transfer/{gpu}_nixl.csv` (`eager/`) | PD KV hand-off through NIXL/UCX, laid out as vLLM's NixlConnector issues it (descriptor list per (layer, block)) |
| `pd_host` (`quettabench/profiling/frontend.py` `pd_host_probe`, a profiler source; profile only, not exported) | the CSV at `--out` (e.g. `results/pd_host/{gpu}_pd_host.csv`), and profile.json `pd_host` | per-request host cost of a PD pair by prompt length (parse/template/tokenize/hash, paid on P and again on D) |

The decode-step-vs-batch curve is a grid run: `qbench run configs/runs/decode_batch_llama.yaml` (grid `decode_batch`: ctx 20000, bs 1 2 4 8 11 16 24) writes the per-cell ms/step to `results/decode_batch_llama/`.

`quettabench/profiling/frontend.py` `frontend_probe` (a profiler source) is the live-server probe that fits the device-YAML `frontend:` + `serving:` blocks against a running `vllm serve` (recipe: `serving-profile results/<run>`, which goes through `qbench profile --source frontend`), for GPUs with no GT yet. With GT, prefer the cheaper `just -f $QUETTASIM/justfile serving-fit`. It and `pd_host_probe` are consumer-coupled like the two emitters: they import the sim's `engine` package, so run them from a QuettaSim checkout (or with QuettaSim on `PYTHONPATH`), and both take the GPU's QuettaSim `device_spec/<gpu>.yaml` as the required `--device-yaml`.

### Why `moe` exists
Every other kernel family here has a measured table; the grouped MoE FFN did not. `sum_kernels._moe_ffn_us` priced it with an analytic roofline over touched bytes/FLOPs, scaled by `util_bw`/`util_flops`, scalars fitted on dense Llama GEMMs. A roofline has no launch cost and no tile quantization, so it is optimistic exactly where a grouped kernel is launch-bound (128 experts × ~1 token each). Measured against the lss-valid H100 numbers it is −51% at a 1-token decode step, −14.5% at 8, converging to −4% by 64. That batch-dependent deficit is what the flat `_decode_host_floor_us` constant patches today, which is why it cannot be right at more than one operating point. Measure the kernel and the constant can go.

Timing method: the launch-cost rationale above only holds for eager execution. vLLM CUDA-graphs MoE decode, so the decode-faithful grid is dispatch-free. Use `--graph` (CUDA-graph-replay timing -> `graph/moe/`, preferred by the loader); the eager default (`eager/moe/`) overpriced H200 graphed decode 2.3x (188us vs 80.8us/layer at 8 tokens). `--graph` works on hosts where NCU counters are locked (`RmProfilingAdminOnly: 1`); see `_timing.py`. Loading a measured MoE grid also retires the `_decode_host_floor_us` patch automatically.

MXFP4-expert models (gpt-oss) are swept for shape coverage but not priced off this bf16 grid: only the weight-read term scales with expert dtype, and a measured latency can't be decomposed after the fact. They keep the roofline until an MXFP4 grid exists.

## Measurement method: decode → NCU, prefill → cuda_event

Which timing method is *faithful* depends on how the kernel runs in serving, so `kernel_data` is split by timing (`graph/` vs `eager/`) and the loader reads each kernel from the right one:

- Decode is CUDA-graphed: the graph replays all decode kernels with no per-launch CPU dispatch, so NCU (pure kernel time) equals the achieved time. Decode grids + gemm + elementwise come from `graph/`. Use `gemm-ncu`, `flash-ncu`, `gdn-ncu` (need a real `ncu`: `NCU_BIN=/opt/nvidia/nsight-compute/2024.3.2/ncu`, not stock `/usr/bin/ncu`).
- Prefill is eager (dynamic shapes, not graphed): every kernel pays its dispatch cost, which cuda_event captures and NCU misses. Prefill grids come from `eager/`. The `flash`/`gdn` probes always emit prefill via cuda_event (even under `--ncu`, which only switches the *decode* grid).

The dispatch overhead is real and host-specific: GDN prefill's 8 tiny Triton sub-kernels floor at ~500µs on A100 (~62µs/launch) vs far less on H100, which is why NCU alone can't reconstruct prefill. Every table's method is recorded in QuettaSim's `data/kernel_data/PROVENANCE.md`; keep a GPU/kernel/phase on one method and re-collect rather than mix. `gemm` without `--ncu` times raw `torch.matmul` cuBLAS, not vLLM's GEMM; prefer `--ncu`.

## Notes / TODO
- `flash` and `gdn` grids are head-config specific: run them per model head config (and per tp shard), not once per GPU.
- `gdn` is first-cut: decode uses the generic recurrent op (the serving-faithful decode is `fused_sigmoid_gating_delta_rule_update`); the batch grid can OOM past B≈80 on the state path. 27B geometry is `geom=27b hv=48`.
- Serving-dynamics blocks have recipes (see step 4): `serving-fit` / `serving-profile` (frontend: + serving:, GT-fit or live-server) and `collectives` (tp_comm:, all-reduce microbench, `quettabench/probes/collective.py`). Caveat (found applying them to A100 2026-07-21): `serving-fit`'s conc1 floor improves TTFT *in isolation* (43.8 -> 31.8) but is not a drop-in win: applied, the big floor (~8x H100) with H100's PORTED mult/lanes curves gives pathological high-concurrency arrival delays that worsen tp>1 TPOT. The floor and curves must be fit jointly against both TTFT and TPOT (`serving-fit` fits only the conc1 floor today). `collectives` measures the real all-reduce cost fine, but on A100 it over-corrects low-conc TPOT (the real tp2 gap is large-batch decode under-pricing). Treat the current output as a starting point, not final.

## Live-server probes

`serving_decode_grid` and `serving_stage_split` drive a vLLM OpenAI server instead of a kernel; the exporters read their results.

## Install

The runner needs no GPU (`pip install -r requirements.txt`). On a GPU host add the probe deps: `pip install -r requirements.txt -r requirements-probe.txt` (torch and vllm). Nsight Systems and the CUDA toolkit are system prerequisites. Pin torch and vllm to the versions your serving stack runs, or the kernel numbers will not match production.

## Data flow

Probe runs stay in `results/probes/` and are backed up to R2 under `profiling/qbench_probes/` (`just mirror-up`); `just export` turns results dirs into the tables QuettaSim reads. QuettaBench holds the tools, not the curated tables.

## Tests

`python3 -m pytest tests/contracts/test_probes.py tests/contracts/test_export.py` runs the probe contracts and export tests on fixtures, no GPU. One skips when its optional input is absent (the decode-grid `load_grid` check needs the consumer's `simulator` package), and `tests/contracts/test_probes_gpu.py` skips without CUDA.
