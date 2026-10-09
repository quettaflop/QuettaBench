# QuettaBench

Measures LLM serving engines on production-shaped workloads, and
cross-validates the QuettaServe engine against vLLM per decode step. One
command, `qbench`, with four doors: `run | profile | crossval | export`.

## Quickstart

```bash
pip install -r requirements.txt        # or: pip install -e .  (gives the qbench command)
python3 data/fetch.py                  # trajectories + distributions from R2 (gitignored)

just provision configs/runs/llama.yaml --dry-run \
    --set hardware.provision.py=/venv/bin/python --set llama:hardware.provision.weights=/w/llama
just dry configs/runs/llama.yaml --set engine.model=/w/llama   # Config + LaunchSpec, starts nothing
just run configs/runs/llama.yaml --set engine.model=/w/llama   # -> results/llama/
just profile --run results/llama --source metrics --source energy
just crossval results/<ours> results/<theirs>   # compares two results dirs
just export --results results/llama --dest ../QuettaSim/engine/data
```

Without `just`, every recipe is one line: `python3 -m quettabench.cli run ...`,
`python3 -m quettabench.provision ...`. `just test` and `just lint` run pytest
and ruff.

## Config

A run is one yaml in `configs/runs/` naming a model (`configs/models/`), a
workload, a mode and optionally a `matrix:` (a sweep: one run per point,
`just sweep` is `just run`). Grids and step points live in
`configs/grids.yaml`, link-profile facts in `configs/hardware/<profile>.yaml`.

Precedence, highest first: `--set` > env (through `config/env_map.py`, the
only reader of `os.environ`) > run yaml > model yaml > hardware yaml >
schema defaults. Unknown keys are refused. `just dry` prints every value
with its origin.

A grid run (`mode.grid` set) times decode ms per step per (ctx, bs) cell:
vLLM in-process, QuettaServe through its cargo bench
(`--set engine.name=quettaserve`). A serve run (`mode.grid` unset)
launches the server and replays the workload against it. `mode: probe`
runs a kernel probe (`mode.probe`, `mode.probe_args`).

## Provisioning

`quettabench/provision.py <run.yaml>...` checks, and sets up when missing:
cache dirs under `hardware.provision.cache_root`, a python with vLLM
(`provision.py`, else pip install `provision.vllm_spec`), `provision.nccl_lib`
(Blackwell needs NCCL >= 2.28), a QuettaServe checkout (`qs_dir`, else cloned
from `qs_repo` at `qs_commit`), and per run the bench binary (`bench_bin`, else
built), the engine checkpoint (`qs_ckpt`), the moe config (`qs_cfg`) and the
vLLM weights (`weights`, else `hf_id` downloaded). Per-run values take a
`<run>:` prefix: `--set qwen3:hardware.provision.weights=/w/qwen3`.

It is idempotent. Anything it cannot resolve prints `MISSING: <what>` and it
exits 1 after listing them all; nothing is guessed. `--dry-run` prints the
plan and writes nothing. A complete pass writes `.env` (caches, PYTHON_BIN,
LD_LIBRARY_PATH; `just` loads it) and prints the `qbench run` lines for both
engines.

## results/<run>/

| file | written by | holds |
|---|---|---|
| `run.json` | run | schema_version, provenance (secrets redacted), config, origins, summary, steps, run_status |
| `requests.jsonl` | run | one row per request (grid: per request of every batch, `cell` tells them apart), appended as each finishes |
| `engine.json`, `server.log` | launcher | endpoint, pid, launch spec of a launched server |
| `bench.log` | QuettaServe runs | the bench's stdout per cell |
| `profile.json` | `qbench profile` | metrics, energy, cost, accuracy; never changes run.json |
| `nsys.*` | `engine.instrumentation=nsys` | the capture armed at launch |

Grid runs no longer write `results/baselines/*.json`; `qbench crossval`
reads the run dir, and still reads old baselines files. Probe recipes write
`results/probes/<gpu>/`; `just export` is the only route to QuettaSim.

## Cross-validation

### Measurement contract

Both columns time the same quantity: steady-state decode ms per token at a
resident batch, pipelined, synchronized only at the ends, at matched
(context, batch, tp). The vLLM side is the slope of batch wall time over the
step points (`configs/grids.yaml`), refit up to `mode.vmin_attempts` times
until r2 >= 0.999; the engine side is the bench's pipelined loop (K steps,
one sync). `max_seq = ctx + timing_steps + max_seq_headroom` changes only the
engine number; the vLLM side uses the model's `max_model_len`. A cell needing
more than 80% of the KV tokens left after weights and a 5 GiB per-GPU reserve
is skipped, as is one with ctx + max(step_points) + 2 > max_model_len.

### Withhold rules

The table prints a ratio only for like with like. It is withheld when the
method differs (criterion groups synchronize every iteration; force with
`XVAL_ALLOW_METHOD_MISMATCH=1`), when quantization differs (`QUANT
MISMATCH`), and when serving topology differs (`SERVING MISMATCH`). KV dtype
mismatches are flagged `KV`, rows at `bs >= comm_bound_bs` `COMM` (the TP
allreduce dominates and is node-specific), fits with r2 < 0.999 `LOW R2`,
eager engine loops `EAGER`. Baselines older than 14 days warn, and NCCL
settings that differ between the sides print `NCCL MISMATCH`.

### Prompt modes (grid runs)

`workload.prompt_mode` (env `PROMPT_MODE`); it matters for MoE routing:

- `distinct` (default): per-slot synthetic seeds, spread routing
- `clone`: the engine benches' exact ids on every slot, identical routing
- `corpus`: sliding windows over the text at `workload.source` (`XVAL_CORPUS`)
- `synth`: routing-diverse synthetic streams
- `trace`: prompts from the replay JSONL at `workload.source` (`XVAL_TRACE`)

The baseline records the mode and the table prints it.

### QuettaServe families

The model yaml's `quettaserve:` block names the cargo `crate` and `test`, the
env prefix (`DS_`, `QW_`) and the family:

- `moe` (deepseek): needs `engine.extra.qs_cfg`; `<P>_LAYERS` is removed, since
  layer truncation would invalidate the run.
- `gdn` (qwen3 and other linear-attention hybrids): exact kernels only when
  `engine.extra.gdn_dir` holds the vLLM GDN cubins (the loader panics
  without them), else the tiled rust suite; custom all-reduce at tp >= 2;
  the exact suite has no CUDA-graph wiring, so it runs eager, rust runs graph.
- `dense`: nothing extra.

A host without cargo sets `engine.extra.bench_bin` to a prebuilt test binary.

### NCCL link profiles

`hardware.link_profile` is detected once (an NV* link in `nvidia-smi topo -m`
is nvlink, else pcie; non-cuda backends take nvlink) and feeds both engines.
pcie pins `NCCL_ALGO=allreduce:tree;allgather:ring`, `NCCL_PROTO=Simple`,
`NCCL_P2P_LEVEL=SYS` (without it cross-socket links fall back to TCP),
`LLMSRV_NO_NVLS=1`, `LLMSRV_TWOSHOT_MIN_BYTES=0`, `NCCL_IB_DISABLE=1`. nvlink
exports only `NCCL_IB_DISABLE=1`: NCCL picks NVLS/LL128 and the engine keeps
its peer all-reduce. Empty values are never exported. The caller's
`NCCL_ALGO`/`NCCL_PROTO` win.

### Correctness

Agreement is two runs compared by `qbench crossval`. Runs with
`mode.returns=text` give greedy-agreement rows (engine vs vLLM text); runs
with `mode.returns=logits` give the FIDELITY gate, vLLM against the
transformers reference on the same grid prompts:

```bash
qbench run configs/runs/llama.yaml --set mode.grid=smoke --set mode.returns=logits \
    --set 'engine.extra.vllm={enforce_eager: true}' --set run_name=llama-logits
qbench run configs/runs/llama.yaml --set mode.grid=smoke --set engine.name=reference \
    --set run_name=llama-reference
qbench crossval results/llama-logits results/llama-reference
```

FIDELITY passes only with zero hard disagreements (an argmax flip where the
reference's top-2 gap is not a near tie). deepseek has no transformers
reference; its correctness comes from the engine's own batch gates.

### Model notes

llama is verified. deepseek, qwen, qwen3, glm53 and glm53flash are bring-up
(`verified: false`, run with `--allow-unverified`) until a clean golden run.
MLA and hybrid stacks must set `kv_bytes`; the dense per-head formula
overestimates their KV (the loader refuses them without it). So
`configs/runs/qwen.yaml` does not load until `kv_bytes` is set in
`configs/models/qwen35_9b.yaml`: qwen is hybrid attention, and the loader
refuses rather than fall back to the dense formula. deepseek needs
the DeepSeek vLLM fork, the tp-sliced fp4 checkpoint (ds-0731-tp8-fp4), and
its KV differs by design (engine bf16 latent vs `fp8_ds_mla`), so its ratios
are withheld. glm53 and glm53flash have no engine bench yet: vLLM side only.
`comm_bound_bs` is set only after a measured grid, never copied across models.

## Old command -> new command

| old | new |
|---|---|
| `scripts/crossval/oneshot.sh <wl>` | `just provision configs/runs/<wl>.yaml ...`, then the two `qbench run` lines it prints, then `just crossval` |
| `scripts/crossval/bootstrap.sh <wl>` (`XVAL_DRYRUN=1`) | `just provision configs/runs/<wl>.yaml [--dry-run]`; `XVAL_CKPT_<wl>`, `XVAL_WEIGHTS_<wl>`, `XVAL_HF_<wl>`, `XVAL_CFG_<wl>`, `<P>_BENCH_BIN` become `--set <wl>:hardware.provision.{qs_ckpt,weights,hf_id,qs_cfg,bench_bin}=...` |
| `scripts/crossval/xval.sh <wl> <weights> <log>` | `just run configs/runs/<wl>.yaml --set engine.model=<weights>` (vLLM) and `... --set engine.name=quettaserve ...` (engine), then `just crossval` |
| `scripts/crossval/vllm.sh <wl> [grid] <weights>` | `just run configs/runs/<wl>.yaml --set engine.model=<weights> [--set mode.grid=<grid> --set run_name=<wl>-<grid>]` |
| `scripts/crossval/bench.sh <wl> [log]` (`<P>_CKPT`, `<P>_CFG`) | `just run configs/runs/<wl>.yaml --set engine.name=quettaserve --set engine.model=<ckpt> [--set engine.extra.qs_cfg=<cfg>]`; log in `results/<wl>/bench.log` |
| `scripts/crossval/prof.sh <name> <cmd>` (`PROF=1`) | `--set engine.instrumentation=nsys` on the run |
| `scripts/crossval/free_gpu.sh`, `with_gpu.sh` | automatic: `qbench run` takes `CUDA_VISIBLE_DEVICES`, else the first idle tp * pp GPUs |
| `benchmark.sh`, `scripts/bench.sh`, `scripts/launch_server.sh` + `python -m src.benchmark.runner --profile P --concurrency N` | `just run configs/runs/<model>.yaml --set mode.grid=null --set workload.kind=live --set workload.source=P --set mode.concurrency=N --set engine.model=<weights>` (the runner launches the server) |
| `scripts/sweep_*_profiles[_sglang].sh` | a `matrix:` block in the run yaml, `just sweep <yaml>` |
| `python -m quettabench.modes.serve ...` (and its `--output` file) | `just run ...`; results in `results/<run>/` |
| `vmin_fit.py <wl>` / `serve --grid --workload <wl>` | `just run configs/runs/<wl>.yaml --set engine.model=<weights>` (`--nograph`: `--set engine.cudagraph=false`, `--dump-tokens`: `--set mode.returns=text`) |
| `trace_serve.py <trace> --model <w>` | `just run <yaml> --set mode.grid=null --set workload.kind=live --set workload.source=<trace> --set mode.load=open --set workload.arrival=trace`; energy, cost, gsm8k: `just profile --source energy|cost|gsm8k` |
| `cache.py <log> <out>` | nothing: grid runs no longer write `results/baselines/*.json`; old files are still readable by `qbench crossval` |
| `python3 table.py <bench-log> <baseline>` | `qbench crossval <ours> <theirs>` (`just crossval`); each side a results dir, an old baselines json or a bench log |
| `xval_config.py tp|devices|cells|bench <wl>` | `just dry configs/runs/<wl>.yaml` (`engine.tp`, `engine.pp`, `grid.cells`, `engine.extra.quettaserve`) |
| `xval_config.py collective|link-profile|backend|run-params` | `just dry ...` (`hardware.*` lines) |
| `xval_config.py provision [key]` | `just provision ... --dry-run`, or the `hardware.provision.*` lines of `just dry` |
| `xval.yaml`, `workloads.json` | `configs/models/`, `configs/runs/`, `configs/grids.yaml`, `configs/hardware/` |

## Layout

```
quettabench/cli.py        qbench: run | profile | crossval | export
quettabench/runner.py     run: load, translate, launch, drive, serve, write
quettabench/provision.py  box setup, --dry-run plan, MISSING: and stop
quettabench/config/       schema, loader (yaml + --set + env), env_map, validate
quettabench/workloads/    traces: live, sessions, grid; arrival schedules; profiles
quettabench/engines/      adapters (vllm, quettaserve), drivers, launcher
quettabench/modes/serve.py  the one measurement procedure
quettabench/metrics/      percentiles, step fit, the results writer
quettabench/profiling/    sources the profiler attaches with
quettabench/probes/       kernel probes; quettabench/export/ routes to QuettaSim
configs/                  models, runs, grids, hardware
analysis/, data/          one-off studies and data fetch; not imported by the package
```

## Data

Canonical data lives on the `agent-bench` R2 bucket:

| prefix | contents |
|---|---|
| `data/` | raw trajectory datasets and `data/distributions/` histograms (fetched by `data/fetch.py`) |
| `results/` | benchmark ground truth (synthetic_distributional, trace_replay, archived) |
| `json/current/` | dashboard runtime JSONs |
| `archive/` | pre-2026-05 result snapshots |
