# profiling/README.md

Measurement code for QuettaBench: GPU kernel probes, live-server probes, and the
emitters that turn raw probe output into curated tables. Peer of the runner
(`quettabench/`). The runner stays GPU-free; `profiling/probes/` needs torch and vLLM
(opt-in via requirements-probe.txt).

## Layout

```
quettabench/probes/            kernel probes (need a GPU): gemm, moe, flash_attn, cross_attn, gdn,
                               elementwise, collective, vllm_allreduce, reparallel
quettabench/export/quettasim/  emitters: results files -> QuettaSim tables (no GPU)
profiling/probes/              live-server + serving-wall probes (need a GPU)
profiling/kernel_composed/probes/  serving-shaped and live-server probes: fused_block, decode_batch,
                               serving_frontend, pd_host, kv_transfer
docs/runbooks/                 how-to-measure docs and preflight scripts
tests/probes/                  probe contracts, GPU smoke, emitter tests
justfile                       one recipe per probe, remote-probe, export
```

The serving-shaped probes (`fused_block`, `decode_batch`) and the live-server probes
(`serving_frontend`, `pd_host`, `kv_transfer`) fold into the grid trace and the profiling
sources in a later commit; until then they stay under `profiling/kernel_composed/probes/`.

### probes/

Kernel probes (isolated, CUDA-event timed):

- `fa_grid_probe.py` — flash-attention grids at tp-sharded head configs, on
  vLLM's own `flash_attn_varlen_func` (fa_version 3).
- `gemm_slice_probe.py` — row-parallel GEMM slices for tp2/tp4 shapes.
- `allreduce_probe.py` — NCCL all-reduce cost over decode-step payloads.
- `custom_allreduce_probe.py` — vLLM's custom all-reduce, the real decode kernel.
- `decode_steps.py`, `cached_prefill_steps_v3.py` — CUDA-event decode / cached
  prefill step timing.

Live-server probes (drive a running vLLM OpenAI server over SSE):

- `serving_herd_scaling.py` — how c1 prefill TTFT grows under a synchronized herd.
- `pool_capacity_probe.py` — effective KV-pool capacity under prefix caching.
- `serving_decode_grid.py` — steady-state mid-stream ITL over a (batch, context)
  grid. Also the single source of truth for the decode-grid post-processing that
  `quettabench/export/quettasim/serving_decode_grid.py` imports.
- `serving_stage_split.py` — per-stage breakdown of c1 prefill TTFT vs
  (new, cached) tokens.
- `live_split_probe.py`, `live_ttft_probe.py` — cached-host split and c1 TTFT on
  the real serving stack.

### Kernel probes and emitters

The kernel probes moved to `quettabench/probes/` (run `just --list`, or read
`quettabench/probes/README.md`). They write the `ncu/` vs `cuda_event/` layout under
`results/probes/<gpu>/`, never into the QuettaSim checkout. The emitters in
`quettabench/export/quettasim/` read results dirs and write QuettaSim's tables plus a
`manifest.json`; `just export --dest <dir>` runs them all.

Three emitters are **consumer-coupled**: `prefill_floor`, `saturated_ceiling` and
`stage_split_rates` import QuettaSim's `configs`, `simulator` (and, for the ceiling, the
retired `build_simulator_rows`), so they need QuettaSim on `PYTHONPATH`. Pick builders with
`just export --only tp_comm roofline_utils ...`.

## Data flow

```
probe (GPU host)  ->  raw CSV / JSONL.gz  ->  R2  agent-bench/profiling/raw/
                                              |
export (just export)  ->  curated table  ->  committed into the CONSUMER repos
                                               (QuettaSim, agentic-serve)
```

Raw probe output is bulky and stays on R2 under `profiling/raw/`. The curated
tables the emitters produce are small and get committed into whichever repo
consumes them (the simulator or the benchmark repo). QuettaBench itself does not
store the curated tables; it holds the tools that make them.

## Install

Base runner only (no GPU):

```bash
pip install -r requirements.txt
```

Add the probe deps on a GPU host:

```bash
pip install -r requirements.txt -r requirements-probe.txt
```

`requirements-probe.txt` pulls torch and vllm. Nsight Systems (`nsys`) and the
CUDA toolkit are system prerequisites, not pip packages. Pin torch and vllm to
the versions your serving stack runs, or the kernel numbers will not match
production.

## Run a probe

Kernel probe, one GPU:

```bash
just gpu=H100 gemm-ncu
just probe gemm "--gpu-label H100 --ncu"
```

Live-server probe: launch a vLLM OpenAI server first, then point the probe at it
(see each probe's `--help` and the runbooks). `serving_stage_split_plan.md` and
`h100_setup.md` in `docs/runbooks/` describe the full measurement setup.

Scheduler that waits for free GPUs before firing:

```bash
nohup bash profiling/probes/slice_probe_watch.sh > slice_watch.log 2>&1 &
```

## Tests

Probe contracts and emitter tests run against fixtures and need no GPU:

```bash
python3 -m pytest tests/probes/
```

Two tests skip when their optional inputs are absent: the stage-split artifact
regeneration (needs the gitignored source CSVs) and the decode-grid
`load_grid` compatibility check (needs the consumer's `simulator` package). The GPU smoke
test skips without CUDA.
