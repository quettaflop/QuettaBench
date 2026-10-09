# profiling/README.md

Measurement code for QuettaBench: GPU kernel probes, live-server probes, and the
emitters that turn raw probe output into curated tables. Peer of the runner
(`quettabench/`). The runner stays GPU-free; `profiling/probes/` needs torch and vLLM
(opt-in via requirements-probe.txt).

## Layout

```
quettabench/probes/            kernel probes (need a GPU): gemm, moe, flash_attn, cross_attn, gdn,
                               elementwise, collective, vllm_allreduce, reparallel; and the two
                               live-server probes the exporters read: serving_decode_grid,
                               serving_stage_split
quettabench/export/quettasim/  emitters: results files -> QuettaSim tables (no GPU)
profiling/probes/              live_split_probe, pool_capacity_probe (no consumer found yet)
profiling/kernel_composed/probes/  serving-shaped probes: fused_block, decode_batch
quettabench/profiling/         profiler sources: /metrics, energy, cost, gsm8k, and the
                               live-server probes serving_frontend, pd_host, kv_transfer
docs/runbooks/                 how-to-measure docs and preflight scripts
tests/probes/                  probe contracts, GPU smoke, emitter tests
justfile                       one recipe per probe, remote-probe, export
```

The live-server probes (`serving_frontend`, `pd_host`, `kv_transfer`) moved to
`quettabench/profiling/` and run through `qbench profile --source ...`.

- `fused_block_probe.py` -- stays here; folds into a GridTrace variant in a later commit.
- `decode_batch_probe.py` -- stays here; folds into a GridTrace variant in a later commit.

### Live-server probes left here

Both drive a running vLLM OpenAI server and have no consumer in the exporters or in the
QuettaSim kernel_data / device-yaml fields, but their results feed numbers the simulator
carries as constants, so they stay until someone confirms:

- `live_split_probe.py` -- cached-host shared vs per-request split (the 50/50 in
  `docs/runbooks/serving_stage_split_plan.md`; `stage_split_rates` keeps that fraction).
- `pool_capacity_probe.py` -- usable KV-pool capacity under prefix caching (the
  `available_kv_blocks` the saturated-ceiling emitter reads).

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

## Tests

Probe contracts and emitter tests run against fixtures and need no GPU:

```bash
python3 -m pytest tests/probes/
```

Two tests skip when their optional inputs are absent: the stage-split artifact
regeneration (needs the gitignored source CSVs) and the decode-grid
`load_grid` compatibility check (needs the consumer's `simulator` package). The GPU smoke
test skips without CUDA.
