# A100 roofline-utils measurement runbook (L6, audit-v2 G7)

Goal: replace the H100 PLACEHOLDERS in `configs/gpus/A100.json`
(`util_flops=0.65 / util_bw=0.93 / scheduler_overhead_ms_per_step=5.7`) with values
measured ON the A100, using the SAME pre-registered recipe that re-derived the H100
values (`profiling/docs/defit_log_entries/L6-utils.md`,
`profiling/process/build_roofline_utils.py`). The recipe is pinned — do NOT tune it on
the A100 numbers; if the A100 data violates a recipe assumption, record the deviation in
the de-fit log and stop.

## Status: DEFERRED (decided at lane launch, 2026-06-10)

The A100 host has 7/8 GPUs running someone else's benchmark campaign. The serving-wall
capture is latency-sensitive (engine step walls in the 6-20 ms range; host/HBM/PCIe
contention from sibling GPUs moves them), so running now would contaminate BOTH the
campaign's numbers and ours. Run this when `preflight_a100_roofline_utils.sh` passes.

## Host facts (recorded at launch — verify with preflight, do not assume)

- ssh alias: `a100` (preflight-verified 2026-06-10: resolves to hostname `gpu-4` — the
  campaign table's `gpu-4` and the launch note's `a100` are the SAME machine; use the
  `a100` alias).
- Model weights: under `/data/models` (expect `Llama-3.1-8B-Instruct`; **Rule #1: never
  download** — if missing, STOP and report).
- Python env: `/home/kevinlau/miniconda3/envs/vllm` (use this env's `vllm`/`python`).
- GPU: A100-SXM4-40GB (1.555 TB/s HBM2e, 312 TFLOPS BF16 dense) — matches
  `profile_data/kernels/roofline_params_A100_llama31_8b.json` peaks, which the builder
  reads for `--gpu A100`.
- Fresh run dir per attempt (mirror the h100_setup.md convention): e.g.
  `/data/kevinlau/l6_roofline_run_<date>/` — never reuse, never `rm` inside an ssh
  command (deny rule).

## Capture harness: the honest gap

The H100 `*_benchmark_serving_wall.jsonl` traces were produced by a live-server
instrumentation patch that was **never committed** (audit-v2 already flags the sched
anchor's slice as non-regenerable). Before the A100 run, recover the harness from the
H100 host (`h100_setup.md` env) or re-implement it. The builder's input contract is
small — any harness emitting one JSON object per scheduler step with these fields works:

| field | meaning |
|---|---|
| `step_id` | 1-based step counter (warmup rule keys off it) |
| `engine_step_wall_ms` | full engine-step wall time |
| `model_submit_wall_ms` | model forward submit wall |
| `decode_batch`, `prefill_tokens` | step classification (decode-only / pure-prefill / mixed) |
| `model_executed` | `"true"` when the step ran the model |
| `decode_request_ids` | space-separated ids decoding this step (context reconstruction) |
| `engine_cache_truth` | JSON string; `requests[*].{request_id,prompt_tokens}` (prompt sizes) |

`trace_scope` must be `benchmark-serving` (walls measured around the live OpenAI-server
engine step, not a replay).

## Procedure

1. `bash profiling/process/preflight_a100_roofline_utils.sh` (from this repo, local —
   it ssh-es to `a100`). All checks must pass; the GPU-quiet check has no override flag
   on purpose.
2. Launch the instrumented vLLM OpenAI server on the ONE free GPU
   (`CUDA_VISIBLE_DEVICES=<idx>`), Llama-3.1-8B-Instruct from `/data/models`, with
   engine flags matching the deployment config (`configs/deployments/` A100 tp1 entry —
   note A100 resolves `max_num_batched_tokens=2048`, unlike H100's 8192).
3. Run the same two workloads that dominate the H100 derivation's step counts
   (`swe c40 t12` and `terminal c80 t16` profiles via inference-benchmark), capturing
   `vllm_engine_step_trace_{swe_c40_t12,terminal_c80_t16}_benchmark_serving_wall.jsonl`.
   Two traces suffice for util_bw + sched medians; add `swe_c80_t12` if time allows.
   NOTE (known from H100): pure-prefill steps are too rare in these traces for a solid
   `util_flops`; plan a separate prefill-util sweep (R1-style,
   `build_prefill_gemm_util.py` pattern) for the A100 prefill side.
4. `scp` traces back (sequential, Rule #1 etiquette), drop them in a local dir, then:
   `python3 -m profiling.process.build_roofline_utils --gpu A100 --archive <dir>`
   → `profile_data/kernels/roofline_utils_A100.json`.
5. Wire `configs/gpus/A100.json` (+ `roofline_params_A100_llama31_8b.json`) from the
   artifact, drop the PLACEHOLDER label, and gate: A100 rows are BINDING
   (`gate_scoped_rows`, ≤ baseline + 0.3). H100 rows must not move (A100-only wiring).
6. Leave the GPU clean (`nvidia-smi --query-compute-apps` empty for your GPU) and
   record the run in `profiling/docs/defit_log_entries/L6-utils.md`.
