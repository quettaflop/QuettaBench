# profiling/README.md

Measurement code for QuettaBench: GPU kernel probes, live-server probes, and the
emitters that turn raw probe output into curated tables. This was proposed as a
separate QuettaProbe repo; it lives here instead, as a peer of the runner
(`src/`), so QuettaBench holds both the benchmark client and the measurement
tools.

The runner (`src/`) stays light and GPU-free. Everything under `profiling/probes/`
needs a GPU, torch, and vLLM. Those deps are opt-in (see "Install" below), so a
plain `pip install -r requirements.txt` still runs the benchmark client on a box
with no GPU.

## Layout

```
profiling/probes/     kernel + live-server probes (need a GPU)
profiling/emit/       emitters: raw CSV/JSONL.gz -> curated tables (no GPU)
profiling/watch/      schedulers that wait for free GPUs, then fire probes
profiling/runbooks/   how-to-measure docs and preflight scripts
profiling/tests/      emitter tests (run against fixtures, no GPU)
```

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
  `emit/build_serving_decode_grid.py` imports.
- `serving_stage_split.py` — per-stage breakdown of c1 prefill TTFT vs
  (new, cached) tokens.
- `live_split_probe.py`, `live_ttft_probe.py` — cached-host split and c1 TTFT on
  the real serving stack.

### emit/

Emitters read raw probe output and write curated tables. They need no GPU and run
anywhere:

- `build_tp_comm.py` — tensor-parallel prefill comm term from the tp1/tpN pair.
- `build_prefill_floor.py` — measured per-config prefill floor (ms).
- `build_saturated_ceiling.py` — measured saturated-ITL ceiling anchors.
- `build_roofline_utils.py` — roofline-utils artifact from serving-wall step traces.
- `build_stage_split_rates.py` — per-config prefill host rate + FA3 coefficient.
- `build_serving_decode_grid.py` — decode-grid CSV from raw JSONL.gz runs.
- `extract_benchmark_per_request.py` — per-request OSL/prefill/cache distributions
  from benchmark JSONs.

Two emitters are **consumer-coupled**: `build_prefill_floor.py` and
`build_saturated_ceiling.py` import from the consumer repo's `configs` and
`simulator` packages (and `build_saturated_ceiling` also imports the retired
`build_simulator_rows`, which was not migrated). They are committed here as the
canonical source, but they only run from a checkout where those consumer modules
are importable (agentic-serve / QuettaSim on `PYTHONPATH`). The other five
emitters are self-contained.

## Data flow

```
probe (GPU host)  ->  raw CSV / JSONL.gz  ->  R2  agent-bench/profiling/raw/
                                              |
emit/*            ->  curated table  ------->  committed into the CONSUMER repos
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
CUDA_VISIBLE_DEVICES=0 python3 profiling/probes/gemm_slice_probe.py \
  --out /path/to/gemm_slices_H100.csv
```

Live-server probe: launch a vLLM OpenAI server first, then point the probe at it
(see each probe's `--help` and the runbooks). `serving_stage_split_plan.md` and
`h100_setup.md` in `runbooks/` describe the full measurement setup.

Scheduler that waits for free GPUs before firing:

```bash
nohup bash profiling/watch/slice_probe_watch.sh > slice_watch.log 2>&1 &
```

## Emit a curated table

Emitters run as modules from the repo root:

```bash
python3 -m profiling.emit.build_serving_decode_grid \
  --inputs raw_run.jsonl.gz --out grid.csv
```

## Tests

The emitter tests run against fixture data and need no GPU:

```bash
python3 -m pytest profiling/tests/
```

Two tests skip when their optional inputs are absent: the stage-split artifact
regeneration (needs the gitignored source CSVs) and the decode-grid
`load_grid` compatibility check (needs the consumer's `simulator` package).
