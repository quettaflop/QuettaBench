# H100 Profiling Setup

SSH alias: `h100`

## Environment

```bash
# GPU
export CUDA_VISIBLE_DEVICES=6

# Temp / cache (data48 mount)
export TMPDIR=<tmp-dir>
export XDG_CACHE_HOME=<tmp-dir>/.cache

# Python
PYTHON=~/miniconda3/envs/vllm/bin/python

# Model
MODEL=<model-dir>
```

## Run a probe

```bash
ssh h100
cd /root/agentic-serve   # or wherever this repo is checked out

TMPDIR=<tmp-dir> XDG_CACHE_HOME=<tmp-dir>/.cache \
GPU_ID=6 PYTHON_BIN=~/miniconda3/envs/vllm/bin/python \
  just probe <probe> "--gpu-label H100 <args>"
```

Each run writes `results/probes/H100/<probe>/<run-id>/`; from another box,
`REMOTE_HOST=h100 GPU_ID=6 PYTHON_BIN=... just remote-probe <probe> H100 "<args>"`
runs it there and pulls the run back (`quettabench/probes/README.md`).

## The old profiling scripts (historical)

The scripts below lived under `profiling/profile/` before the refactor and
are retired; each row gives the current command (historical paths, kept for
the record of what the H100 numbers came from).

| old script (historical, retired) | now |
|--------|-------------|
| `profiling/profile/vllm/cuda_events/decode_steps.py` | a grid run: `just run configs/runs/<model>.yaml --set engine.model=<weights>` (ms per step in `results/<run>/run.json` `steps`) |
| `profiling/profile/vllm/cuda_events/flash_attn.py` | `just gpu=H100 flash tag=tp1 nh=<Hq> nkv=<Hkv> hd=<head_dim>` |
| `profiling/profile/vllm/cuda_events/decode_kernel_trace.py` | `--set engine.instrumentation=nsys` on the run, then `just profile --run results/<run> --source kernels --kernel-steps N` |
| `profiling/profile/vllm/cuda_events/prefill_kernel_trace.py` | as decode_kernel_trace, on a run whose grid holds the prefill cells |
| `profiling/profile/vllm/cuda_events/prefill_decomposition.py` | as decode_kernel_trace (the kernels source buckets attention vs the rest) |
| `profiling/profile/vllm/sweep/prefill_steps.py` | a grid run as decode_steps, over the prefill cells in `configs/grids.yaml` |
| `profiling/profile/vllm/engine_trace/serving_engine_steps.py` | `just profile --run results/<run> --source metrics` on a live run |
| `profiling/profile/scripts/run_vllm_profile.py` (orchestration) | `just run` and `just profile` (README.md "Old command -> new command") |

The historical prefill decomposition wrote `attention_ms`, `gemm_ms`,
`elementwise_ms` and `overhead_ms` per prompt length (eager mode), next to
the compiled prefill times; the non-attention reduction (`compiled_ms -
attention_ms` vs `eager_ms - attention_ms`) was read as the torch.compile +
CUDA graph fusion benefit.
