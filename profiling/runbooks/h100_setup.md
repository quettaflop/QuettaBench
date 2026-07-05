# H100 Profiling Setup

SSH alias: `h100`

## Environment

```bash
# GPU
export CUDA_VISIBLE_DEVICES=6

# Temp / cache (data48 mount)
export TMPDIR=/data48/kevinlau/tmp
export XDG_CACHE_HOME=/data48/kevinlau/tmp/.cache

# Python
PYTHON=~/miniconda3/envs/vllm/bin/python

# Model
MODEL=/data48/kevinlau/models/Llama-3.1-8B-Instruct
```

## Run a profiling script

```bash
ssh h100
cd /root/agentic-serve   # or wherever the repo is checked out

CUDA_VISIBLE_DEVICES=6 \
TMPDIR=/data48/kevinlau/tmp \
XDG_CACHE_HOME=/data48/kevinlau/tmp/.cache \
~/miniconda3/envs/vllm/bin/python <script-path> <args>
```

## Prefill decomposition (experiment.md method)

```bash
ssh h100
cd /root/agentic-serve

CUDA_VISIBLE_DEVICES=6 \
TMPDIR=/data48/kevinlau/tmp \
XDG_CACHE_HOME=/data48/kevinlau/tmp/.cache \
~/miniconda3/envs/vllm/bin/python \
  profiling/profile/vllm/cuda_events/prefill_decomposition.py \
  --model /data48/kevinlau/models/Llama-3.1-8B-Instruct \
  --output profile_data/results/prefill_decomposition_H100.csv
```

Output columns:
- `attention_ms` — attention kernel time (same in eager and compiled mode)
- `gemm_ms` — GEMM kernel time (eager mode)
- `elementwise_ms` — triton pointwise/reduction kernel time (eager mode)
- `overhead_ms` — wall time not attributed to any bucket
- Compiled prefill times from `prefill_profile_H100.csv` are shown for comparison

Interpretation: `attention_ms` is the cost of flash attention custom ops. In compiled (CUDA graph) mode these run as-is. The non-attention reduction (`compiled_ms - attention_ms` vs `eager_ms - attention_ms`) is the torch.compile + CUDA graph fusion benefit.

## Existing profiling scripts

| Script | What it does |
|--------|-------------|
| `profiling/profile/vllm/cuda_events/decode_steps.py` | Decode step wall time via CUDA events |
| `profiling/profile/vllm/cuda_events/flash_attn.py` | Isolated flash attention sweep |
| `profiling/profile/vllm/cuda_events/decode_kernel_trace.py` | Decode step kernel bucketing via torch.profiler |
| `profiling/profile/vllm/cuda_events/prefill_kernel_trace.py` | Prefill step kernel bucketing via torch.profiler (delta method) |
| `profiling/profile/vllm/cuda_events/prefill_decomposition.py` | Prefill attention vs compiled non-attention decomposition |
| `profiling/profile/vllm/sweep/prefill_steps.py` | vLLM prefill step time lattice |
| `profiling/profile/vllm/engine_trace/serving_engine_steps.py` | V1 scheduler step tracing |

## Orchestration

Use `profiling/profile/scripts/run_vllm_profile.py` for orchestration:

```bash
CUDA_VISIBLE_DEVICES=6 \
TMPDIR=/data48/kevinlau/tmp \
XDG_CACHE_HOME=/data48/kevinlau/tmp/.cache \
~/miniconda3/envs/vllm/bin/python profiling/profile/scripts/run_vllm_profile.py \
  --source cuda-events \
  --target <target> \
  -- \
  --model /data48/kevinlau/models/Llama-3.1-8B-Instruct \
  <target-specific-args>
```
