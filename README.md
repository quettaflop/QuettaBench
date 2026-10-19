# QuettaBench

Measures LLM serving engines on production-shaped workloads, and compares the QuettaServe engine against vLLM per decode step. One command, `qbench`, with four subcommands: `run | profile | compare | export`, and `just provision` for box setup.

## Quickstart

```bash
pip install -e .                       # gives the qbench command; add .[probe] on a GPU host
just fetch                             # trajectories from R2 (gitignored)

just provision configs/runs/llama.yaml --dry-run \
    --set hardware.provision.py=/venv/bin/python --set llama:hardware.provision.weights=/w/llama
just dry-run configs/runs/llama.yaml --set engine.model=/w/llama   # Config + LaunchSpec, starts nothing
just run configs/runs/llama.yaml --set engine.model=/w/llama   # -> results/llama/
just profile --run results/llama --source metrics --source energy
just compare results/<ours> results/<theirs>       # compares two results dirs
just export --results results/llama --dest <export dir>   # the builders but kernel_tables
just export --only kernel_tables --results results/probes --kernel-dest ../QuettaSim/data   # -> kernel_data/{graph,eager}/
```

Without `just`, every recipe is one line: `python3 -m quettabench.cli run ...`, `python3 -m quettabench.provision ...` for `just provision`. `just test` and `just lint` run pytest and ruff.

## Config

A run is one yaml in `configs/runs/` naming a model (`configs/models/`), a workload, a mode and optionally a `matrix:` (a sweep: one run per point of the axes' product, `just sweep` is `just run`; paths joined by `+` are one axis whose choices list one value per path, as in `mode.concurrency+engine.extra.vllm.max_num_seqs: [[8, 64], [32, 256]]`; a `--set` naming an axis replaces its choices, so `--set 'mode.concurrency+engine.extra.vllm.max_num_seqs=[[32, 256]]'` runs that point alone). Grids and step points live in `configs/grids.yaml`, link-profile facts in `configs/hardware/<profile>.yaml`.

Precedence, highest first: `--set` > env (through `config/env_map.py`, the only reader of `os.environ`) > run yaml > model yaml > hardware yaml > schema defaults. Unknown keys are refused. `engine.extra.<engine>.<flag>` sets one flag in that engine's extra block, keeping the rest; a whole `engine.extra.<engine>` from a higher layer replaces the block. `just dry-run` prints every value with its origin.

A grid run (`mode.grid` set) times decode ms per step per (ctx, bs) cell: vLLM over HTTP against a launched server (bs concurrent completions per batch, prefix cache reset per cell), QuettaServe through its cargo bench (`--set engine.name=quettaserve`). A vLLM grid run with `mode.return_outputs` text or logits builds vLLM in-process instead, the only path that returns token ids, text or logits; only such a run takes `prompt_mode=corpus`, which a timing grid refuses. `engine.pp` and `engine.cudagraph=false` reach the server as `--pipeline-parallel-size` and `--enforce-eager`. `tests/parity/slope_gate.json` holds the HTTP-vs-in-process gate both paths passed (8x H200, vLLM 0.30.1: 15 cells, worst |delta| 0.228 ms). A serve run (`mode.grid` unset) launches the server and replays the workload against it. `mode: probe` runs a kernel probe (`mode.probe`, `mode.probe_args`) with `--out results/<run>/`. It refuses the torchrun probes (`collective`, `reparallel`, `vllm_allreduce`) and the probes that name output files (`serving_decode_grid`, `serving_stage_split`); run those with `just probe`.

## Measuring a running server

`hardware.launcher: attach` measures the server already answering on `engine.port`: qbench checks it is healthy and serves the model, then starts and stops nothing, so the same server can be measured by qbench and another harness (vllm bench serve, AIPerf, AgentX) back to back. `engine.json` records what the server serves, its `/version`, and the launch spec qbench would have used; the server's real argv and environment are the operator's to record.

## Session replay

An agent profile (`mode: multi-turn` in `configs/workloads.yaml`) replays recorded agent sessions: each turn's prompt is the session's whole context so far, with the `prompt_tokens` and `osl_tokens` the serving engine reported when the session was recorded. The file never carries an estimate; a turn without the counts refuses the file. `just fetch` pulls the files named in `data/MANIFEST.json` (sha256, source run, agent, model, date, token quantiles) and refuses one whose digest differs; `data/build_trajectories.py` rebuilds them from the recorded runs.

`mode.concurrency` is the number of agents. An agent holds its slot from its session's first turn to its last, waits included, and the next session starts when one finishes, so the server never has more than `concurrency` sessions open and no request waits in a client queue. With `workload.use_recorded_waits` a session the builder marked `recorded_waits: false` is dropped before sampling, counted in `sessions_dropped_no_waits`, rather than replayed back to back. This is TensorRT-LLM's trace replay and GenAI-Perf's `--session-concurrency`; a request-level cap with every session open at once is not comparable with either.

The sample is fixed before a sweep: every eligible session, or `workload.num_sessions` of a seeded shuffle, the same set at every concurrency (`sample_digest` in the summary). `workload.stratified` orders it round-robin across strata of turn count and prompt size (quartiles of the sample), and with `num_sessions` takes each stratum's share, so the sessions open at any moment mix like the sample; `strata` in `summary.trace` gives the counts. A file with fewer sessions than asked, or fewer than `mode.concurrency`, is refused. A session is cut at the first turn whose prompt does not fit `isl_tokens` beside its output or whose output passes `osl_tokens`, never clipped; the last turn kept says `truncated_by_context_limit`, and `sessions_cut` and `sessions_dropped` are in the trace metadata.

A recycled replay (`mode.duration_s`) puts more agents on a sample than it has sessions: each agent draws the next session from a cycle over the sample, plays it to the end, and draws again until the deadline. Every play is salted so a replay of a session does not hit the prefix cache its earlier play left: `workload.salt: text` (the default) puts a tag the server has not seen before its first user message, as AgentX's `first_turn_prefix` does; `workload.salt: cache_salt` leaves the prompt as recorded and sends vLLM's `cache_salt` request field, one value per play (refused on engines that do not take it); the identity gate checks every turn of the plays started, `sections.plays` counts them, and the steady window runs to the deadline. Its `plan_hash` covers the sample, the seed and the play rule (`sections.plays.play_rule`), so it does not move with how many plays the duration and the server's speed let start. Results compare only at the same sample, seed, duration and agent count. A trace that fills its replies from the engine (ShareGPT) cannot be recycled.

The whole-run throughput in the summary includes the ramp while the first agents fill and the drain while the last finish. `sections.steady` is the window in which every agent is busy: from the moment every agent has finished its first session (the synchronized first sessions on a cold cache stay out) to the first session finish after the last admission, when the backlog is empty. The cold start is reported beside it, never dropped: `first_turn_ttft_ms` over the first turn of each agent's first session on the cold cache, and `ramp.ttft_ms` over every request sent before the window opened. Output throughput counts the tokens that arrived inside it: each chunk carries the server's own count when every chunk has one (`mode.chunk_usage` asks vLLM for cumulative usage on every chunk; off by default, so request bodies match vllm bench serve's and AIPerf's), else the request's output tokens in proportion to the chunk's delta characters (content, reasoning, tool-call arguments), the finish chunk carrying everything when there are none; `requests.jsonl` keeps `chunk_chars` and `chunk_tokens` per request. So a request straddling an edge counts only what arrived inside, and a silent tail's tokens land when they arrived. `timeline` bins output tok/s and requests in flight (every request, failed ones too, from its send to its completion) so the window can be seen to be steady (10 s bins, a last bin under 5 s folded into the one before, mean and cv weighted by bin width), TTFT/TPOT/ITL percentiles take the requests sent at or after its start and completed by its end, `sessions_per_hour` the sessions that finished inside it with every turn successful, and `interactivity_p90_tok_s` is 1000 / p90 `tpot_ms`, AgentX's figure: its inter-token latency is per request, (E2E - TTFT) / (output tokens - 1); `chunk_gap_ms` is the gap between streamed chunks, AIPerf's inter-chunk latency. A per-session run with 2C sessions or fewer (the agents never get past their synchronized first sessions), a recycled run whose window opens after the deadline or lasts under 300 s, or a run whose recorded arrivals left an agent idle inside the window, has `valid: false`, the failed condition in `reason`, and no steady number; a whole-run number never stands in. A recycled window also reports `plays_completed` inside it, valid or too short. Every timestamp is read off the client's event loop, so `sections.client` samples how late that loop runs over the pass and `steady.client_loop_lag_ms` inside the window, percentiles weighted by the wall time each sample spans; above 5 ms at p99 in the window the steady TTFT is withheld, with the reason in `steady.withheld`. `sessions_completed` counts sessions whose every planned turn succeeded. A failed turn ends its session: the turns after it are not sent, `workload.turns_not_sent` lists them, the success-rate gate counts them as failed and the run is `partial`; the identity gate checks what was sent against the plan less those turns, and `workload.plan_hash` stays the whole plan's. Only a run with no failed request is `ok`. In a `partial` run every latency mean and percentile is withheld (`withheld.latency_percentiles` says why; throughput and counts stay), because the survivors leave out the slowest requests, and an SLA verdict or goodput probe with a failed request FAILs before any p99 is read; a sweep reports its best passing probe's requests.

The workload identity has two sides that never share an object. `workload.plan_hash`, the hash runs compare on, is frozen before anything is sent: the file's sha256, the sample, the session order, the play rule when recycled, and a digest of every request (messages, tools, max_tokens) taken then. Each request is digested again at dispatch, just before it goes out, and compared with the frozen one per request; a request that differs, was not planned or was never sent fails the run (exit 3), and `workload.first_mismatch` names the first. Only the placeholders the run fills with the engine's replies are left out of both digests; a recorded assistant turn is hashed. `server_cache_hit_rate` is the engine's reported cached-token share (vLLM: `--enable-prompt-tokens-details`); the per-turn `cache_hit_rate` is the ceiling an ideal cache could reach, not a hit. Nearest-rank percentiles (serve, SLA, burst) take the ceil(p/100 * n)-th value. A chat request's TTFT is its first chunk with choices and its E2EL ends at its last, the finish chunk included, vllm bench serve's rule: decode the server buffered (a tool call its parser emits whole) is in the request's time. AIPerf starts at the first chunk with content and ends at the last, so it skips an empty role chunk and reads a lower ITL on a reply that ends in a buffered tool call. The usage chunk, `[DONE]` and the close come after, in `stream_closed_s`. Every row of `requests.jsonl` carries its raw events on the run clock, so a reader can rebuild any timeline: `sent_at_ms`, the driver's clock start just before the POST, from which TTFT, E2EL and `stream_closed_s` count; the chunks at `sent_at_ms + ttft_ms` plus the running sum of `itl_ms`; `stream_closed_ms`; and `completed_at_ms`.

## Provisioning

`quettabench/provision.py <run.yaml>...` checks, and sets up when missing: cache dirs under `hardware.provision.cache_root`, a python with vLLM (`provision.py`, else pip install `provision.vllm_spec`), `provision.nccl_lib` (Blackwell needs NCCL >= 2.28), a QuettaServe checkout (`quettaserve_dir`, else cloned from `quettaserve_repo` at `quettaserve_commit`), and per run the bench binary (`bench_bin`, else built), the engine checkpoint (`quettaserve_checkpoint`), the moe config (`quettaserve_config`) and the vLLM weights (`weights`, else `hf_id` downloaded). Per-run values take a `<run>:` prefix: `--set qwen3:hardware.provision.weights=/w/qwen3`.

It is idempotent. Anything it cannot resolve prints `MISSING: <what>`, and it exits 1 after listing them all; nothing is guessed. `--dry-run` prints the plan and writes nothing. A complete pass writes `.env` (caches, PYTHON_BIN, LD_LIBRARY_PATH; `just` loads it) and prints the `qbench run` lines for both engines, or the vLLM line alone for a model yaml without a `quettaserve:` block.

## One image on a fresh box

`just image [<base>]` builds `docker/Dockerfile`, qbench installed editable on the platform's vLLM image (default `vllm/vllm-openai:v0.26.0`), tagged `qbench:<short sha>` with the sha in `QBENCH_SHA`. On a non-cuda base add `--build-arg NIXL=` (the NIXL wheel is cuda only). Bases whose tags were checked on Docker Hub or in the vendor docs:

| `hardware.backend` | base image | source |
|---|---|---|
| cuda | `vllm/vllm-openai:v0.26.0` (tested on H200 and RTX 6000 Pro) | Docker Hub |
| rocm | `vllm/vllm-openai-rocm:v0.30.0` | Docker Hub; vLLM `docs/getting_started/installation/gpu.rocm.inc.md` |
| tpu | `vllm/vllm-tpu:v0.30.0` | Docker Hub; tpu-inference docs |
| neuron | `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.24.0.1.1.0-neuronx-py313-sdk2.32.0-ubuntu24.04` (vLLM 0.24 with vllm-neuron) | AWS Neuron DLC quickstart (SDK 2.32.0) |

`hardware.backend` picks the devices a launched server sees (`CUDA_VISIBLE_DEVICES`; `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` on rocm; `TPU_VISIBLE_CHIPS` with its process bounds on tpu; `NEURON_VISIBLE_DEVICES` on neuron, with `NEURON_RT_VISIBLE_CORES` unset) and, with `hardware.launcher: docker`, the device flags. On tpu `engine.dtype=float16` is refused; nsys, `hardware.provision.nccl_lib`, the transformers reference and `mode.return_outputs` grids (in-process vLLM) are cuda only. The kernel probes run on cuda and rocm through `probes/_timing.py` (rocm is mock-tested only, and `--ncu` is cuda only), `gemm` and `elementwise` also on cpu; tpu and neuron refuse. Only cuda checks which GPUs are busy; rocm, tpu and neuron runs must set `hardware.visible_devices`, else they take devices 0..n-1. amd-smi, neuron-ls and tpu report no memory size, so grids there run without the KV SKIP.

Mount the HF cache, so `engine.model` can be a repo id, the vLLM compile cache and `results/`:

```bash
docker run --rm --gpus all --network host --ipc host -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/.cache/vllm:/root/.cache/vllm -v $PWD/results:/opt/qbench/results \
    qbench:<sha> run configs/runs/llama.yaml --set engine.model=meta-llama/Llama-3.1-8B-Instruct
```

On 2026-10-12 this command (Qwen3-4B, GPU 0, HTTP grid) agreed with the same qbench source run outside docker on vLLM 0.26.0 (GPU 1) within 1.51% on 15/15 cells, mean +0.52%:

```bash
docker run --rm --gpus '"device=0"' --network host --ipc host \
    -v ~/hf:/root/.cache/huggingface -v ~/qbt-run2/docker:/opt/qbench/results \
    -e HF_HUB_OFFLINE=1 -e HF_HOME=/root/.cache/huggingface \
    qbench:81ac0e4 run configs/runs/llama.yaml --results-dir results \
    --set hardware.link_profile=pcie --allow-unverified \
    --set engine.model=/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
    --set engine.port=8040 --set run_name=docker_qwen4b
```

## Running inside docker

`hardware.launcher: docker` runs the server in `engine.image` (`docker run -d`, the GPUs, the port, and the repo, `hardware.hf_home` and an absolute `engine.model` dir mounted read-only at their host paths, and `results/<run>/` read-write so nsys output lands there); `configs/runs/llama-tp1-docker.yaml` is the example. Provision never pulls; pull the image yourself. The docker backend is validated on one cuda host (see Validation); its rocm, tpu and neuron device flags are tested against mocks only.

```bash
docker pull vllm/vllm-openai:v0.26.0
just dry-run configs/runs/llama-tp1-docker.yaml --set engine.model=/w/llama   # prints the docker command
just run configs/runs/llama-tp1-docker.yaml --set engine.model=/w/llama
```

## results/<run>/

| file | written by | holds |
|---|---|---|
| `run.json` | run | schema_version, provenance (secrets redacted), config, origins, summary, steps, run_status |
| `rows.jsonl` | probe run | one `ProbeRow` per measured row (probe, table, key, value, timing); `just export` builds the kernel tables from them |
| `requests.jsonl` | run | one row per request (grid: per request of every batch, `cell` tells them apart), appended as each finishes |
| `engine.json`, `server.log` | launcher | endpoint, pid, launch spec of a launched server |
| `bench.log` | QuettaServe runs | the bench's stdout per cell |
| `profile.json` | `qbench profile` | metrics, energy, cost, accuracy; never changes run.json |
| `nsys.*`, `nsys-<ctx>x<bs>.*` | `engine.instrumentation=nsys` | the capture armed at launch (QuettaServe: one per bench cell) |

`qbench run` refuses a `results/<run>/` that already holds files; `--overwrite` empties it first: every file and subdirectory there belongs to the old run, so `profile.json` and the nsys export go with it. A run whose `run.json` says failed is replaced without `--overwrite`. Only a real dir directly under the results dir that holds `run.json`, `requests.jsonl` or `engine.json` is ever emptied; a symlink or a dir qbench did not make is refused. `run_name` (the yaml stem or `--set run_name=`) matches `[A-Za-z0-9][A-Za-z0-9._-]*` and is none of `probes`, `compare`, `baselines`; matrix suffixes are cut to that charset. `--reuse-server` runs consecutive matrix points that share an engine and hardware config against one server, started by the first point and stopped after the last; each point keeps its own run dir with a copy of `engine.json`, and every point must set `mode.reset_prefix_cache` so it measures from an empty prefix cache.

Grid runs write no `results/baselines/*.json`; `qbench compare` reads the run dir, and also reads baselines json files. Each probe run writes its own `results/probes/<gpu>/<probe>/<run-id>/` (the run id is its UTC start, e.g. `20261011T120000.000000Z`): `run.json` (argv, provenance, run_status) and `rows.jsonl`, one `ProbeRow` (probe, table, key, value, timing) per measured row. `just export` builds every kernel table from those rows, unioning every ok run's rows per table on the probe's row key, the run that finished last winning a key, and takes a table with no row key from the run that finished last (`docs/runbooks/probes.md`). It merges into the tables QuettaSim already holds, keeping their rows unless a run supersedes them, and stops where it would drop one unless given `--replace-tables`, which replaces just the tables that would stop. It is the only route to QuettaSim.

## Compare

### Measurement contract

Both columns time the same quantity: steady-state decode ms per token at a resident batch, pipelined, synchronized only at the ends, at matched (context, batch, tp). The vLLM side is the slope of batch wall time over the step points (`configs/grids.yaml`), refit up to `mode.fit_attempts` times until r2 >= 0.999; the engine side is the bench's pipelined loop (K steps, one sync). `max_seq = ctx + timing_steps + max_seq_headroom` changes only the engine number; the vLLM side uses the model's `max_model_len`. A cell needing more than 80% of the KV tokens left after weights and a 5 GiB per-GPU reserve is skipped, as is one with ctx + max(step_points) + 2 > max_model_len.

### Withhold rules

The table prints a ratio only for like with like. It is withheld when the method differs (criterion groups synchronize every iteration; force with `--allow-method-mismatch`), when quantization differs (`QUANT MISMATCH`), and when serving topology differs (`SERVING MISMATCH`). KV dtype mismatches are flagged `KV`, rows at `bs >= comm_bound_bs` `COMM` (the TP allreduce dominates and is node-specific), fits with r2 < 0.999 `LOW R2`, eager engine loops `EAGER`. Baselines older than 14 days warn, and NCCL settings that differ between the sides print `NCCL MISMATCH`.

### Prompt modes (grid runs)

`workload.prompt_mode` (env `PROMPT_MODE`); it matters for MoE routing:

- `distinct` (default): per-slot synthetic seeds, spread routing
- `clone`: the engine benches' exact ids on every slot, identical routing
- `corpus`: sliding windows over the text at `workload.source` (`QBENCH_CORPUS`); in-process, so `mode.return_outputs` runs only
- `trace`: prompts from the replay JSONL at `workload.source` (`QBENCH_TRACE`)

The baseline records the mode and the table prints it.

### QuettaServe families

The model yaml's `quettaserve:` block names the cargo `crate` and `test`, the env prefix (`DS_`, `QW_`) and the family:

- `moe` (deepseek): needs `engine.extra.quettaserve_config`; an inherited `<P>_LAYERS` is removed, since layer truncation would invalidate the run. `engine.layers=N` sets it on purpose for a layer-truncated run; compare ignores the `layers=N` lines.
- `gdn` (qwen3 and other linear-attention hybrids): exact kernels only when `engine.extra.gdn_dir` holds the vLLM GDN cubins (the loader panics without them), else the tiled rust suite; custom all-reduce at tp >= 2; the exact suite has no CUDA-graph wiring, so it runs eager, rust runs graph.
- `dense`: nothing extra.

A host without cargo sets `engine.extra.bench_bin` to a prebuilt test binary.

### NCCL link profiles

`hardware.link_profile` is detected once (an NV* link in `nvidia-smi topo -m` is nvlink, else pcie; non-cuda backends take nvlink) and feeds both engines. pcie pins `NCCL_ALGO=allreduce:tree;allgather:ring`, `NCCL_PROTO=Simple`, `NCCL_P2P_LEVEL=SYS` (without it cross-socket links fall back to TCP), `LLMSRV_NO_NVLS=1`, `LLMSRV_TWOSHOT_MIN_BYTES=0`, `NCCL_IB_DISABLE=1`. nvlink exports only `NCCL_IB_DISABLE=1`: NCCL picks NVLS/LL128 and the engine keeps its peer all-reduce. Empty values are never exported. The caller's `NCCL_ALGO`/`NCCL_PROTO` win.

### Correctness

Agreement is two runs compared by `qbench compare`. Runs with `mode.return_outputs=text` give greedy-agreement rows (engine vs vLLM text); runs with `mode.return_outputs=logits` give the FIDELITY gate, vLLM against the transformers reference on the same grid prompts:

```bash
qbench run configs/runs/llama.yaml --set mode.grid=smoke --set mode.return_outputs=logits \
    --set 'engine.extra.vllm={enforce_eager: true}' --set run_name=llama-logits
qbench run configs/runs/llama.yaml --set mode.grid=smoke --set engine.name=reference \
    --set run_name=llama-reference
qbench compare results/llama-logits results/llama-reference
```

FIDELITY passes only with zero hard disagreements (an argmax flip where the reference's top-2 gap is not a near tie). deepseek has no transformers reference; its correctness comes from the engine's own batch gates.

### Truncated-model verification

A model cut to its first N layers cannot be checked on a free-running greedy chain: bf16 rounding flips its router within a few steps. The check is teacher-forced logits against a float32 reference anchor, passing when the engine's argmax flips are no more than the reference's own at the engine's dtype (the noise floor). `engine.layers=N` cuts the reference (and sets the QuettaServe batch_bench's `<P>_LAYERS`); vLLM loads a checkpoint dir cut the same way as `engine.model`. `qbench compare <ours> <anchor> --floor <run>` prints the BAND line:

```
qbench run <run>.yaml --set mode.return_outputs=logits --set engine.model=<cut dir> --set run_name=truncated-vllm
qbench run <run>.yaml --set engine.name=reference --set engine.layers=N --set engine.dtype=float32 --set run_name=truncated-f32
qbench run <run>.yaml --set engine.name=reference --set engine.layers=N --set engine.dtype=bfloat16 --set run_name=truncated-bf16
qbench compare results/truncated-vllm results/truncated-f32 --floor results/truncated-bf16
```

A floor run is trusted once it is deterministic: the same run compared twice gives argmax_agree equal to positions and delta max 0. `qbench profile --source kernels` gives per-kernel GPU time per decode step from an nsys-armed run, and `just moe-routing <model>` (`probes/moe.py --routing`) dumps one MoE block for an engine-side diff.

### Model notes

llama is verified. deepseek, qwen, qwen3, glm53 and glm53flash are bring-up (`verified: false`, run with `--allow-unverified`) until a clean golden run. MLA and hybrid stacks must set `kv_bytes_per_token`; the dense per-head formula overestimates their KV (the loader refuses them without it). So `configs/runs/qwen.yaml` does not load until `kv_bytes_per_token` is set in `configs/models/qwen35_9b.yaml`: qwen is hybrid attention, and the loader refuses rather than fall back to the dense formula. deepseek needs the DeepSeek vLLM fork, the tp-sliced fp4 checkpoint (ds-0731-tp8-fp4), and its KV differs by design (engine bf16 latent vs `fp8_ds_mla`), so its ratios are withheld. glm53 and glm53flash have no engine bench yet: vLLM side only. deepseek_v41_flash, qwen38_flash_next and qwen38_max_nvfp4 are bring-up for vLLM serving runs, sized from their safetensors index. `comm_bound_bs` is set only after a measured grid, never copied across models.

## Layout

```
quettabench/cli.py        qbench: run | profile | compare | export
quettabench/runner.py     run: load, build the Trace, translate, launch, drive, serve, write
quettabench/provision.py  box setup, --dry-run plan, MISSING: and stop
quettabench/config/       schema (with topology rules), loader (yaml + --set + env), env_map
quettabench/workloads/    traces: live (with the profiles registry), sessions, grid; arrival schedules
quettabench/engines/      adapters (vllm, quettaserve), drivers (with the reference adapter), launcher
quettabench/compare.py   pairs two runs' cells; withhold rules; agreement rows
quettabench/modes/serve.py  the one measurement procedure: replay, sessions, grid cells
quettabench/metrics/      percentiles, step fit, the results writer
quettabench/profiling/    sources the profiler attaches with
quettabench/probes/       kernel probes
quettabench/export/quettasim.py  the one route to QuettaSim: tp_comm, roofline_utils, serving_decode_grid,
                          kernel_tables
configs/                  models, runs, grids, hardware
docker/Dockerfile         qbench on the vLLM image (just image)
docs/runbooks/            the probe guide (probes.md)
tests/golden/             characterization fixtures
tests/contracts/          one test module per package area, plus the compare contracts and GPU smoke
tests/parity/             the committed HTTP-vs-in-process slope gate and its test
data/                     data fetch scripts; not imported by the package
```

## Data

Canonical data lives on the `agent-bench` R2 bucket:

| prefix | contents |
|---|---|
| `data/` | raw trajectory datasets (fetched by `just fetch`) |
| `results/` | benchmark ground truth (synthetic_distributional, trace_replay, archived) |
| `archive/` | earlier result snapshots |

## Known open items

- `InProcessDriver` stays as the only path that returns token ids, text or logits.
