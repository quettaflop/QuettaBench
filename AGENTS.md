# AGENTS.md

What the repo is, where each thing lives, and the rules every commit keeps. Read it before changing anything.

## What this repo is

QuettaBench is a measurement harness. It has one command, `qbench`, with four subcommands: `run | profile | compare | export`; `just provision` sets the box up. It has one measurement procedure, `quettabench/modes/serve.py`. Three structs flow one way, `Config -> LaunchSpec -> RequestResult`, and nothing flows back. Every measurement lands in one results layer, `results/<run>/`, and leaves the repo by `quettabench/export/quettasim.py`; the probe runs also leave by `just mirror-up` (below).

## Layout

- `quettabench/cli.py`, `runner.py`, `provision.py`: the command, the run sequence, and box setup. `runner.py` is the only code that starts a run.
- `quettabench/config/`: schema (with the serving-topology rules), loader, env_map. The only place `os.environ` is read is `config/env_map.py`, enforced by `tests/contracts/test_env_map.py`; its allowlist only shrinks.
- `quettabench/workloads/`: every request source is a `Trace` with a `workload_hash`. A source without a hash does not belong here.
- `quettabench/engines/`: adapters translate a Config to a LaunchSpec from a field table; a field with no flag raises `UnsupportedField`, never a silent drop. Drivers are the only code in a run that talks to an engine. `launcher.py` is the only code that starts an engine server, for a run or a probe (`probes/base.py` `launch_vllm`); `BenchDriver` runs the QuettaServe bench, which is its own engine, once per grid cell. Other subprocesses are tools (nvidia-smi, amd-smi, neuron-ls, git, nsys export, docker, cargo in provision), never an engine.
- `quettabench/modes/`: `serve.py`, nothing else: the request replay, the session dispatch and the grid cells. Nothing in `modes/` branches on the engine name; an engine difference is a driver.
- `quettabench/metrics/`: one percentile function, and the caller states the method. `step_fit.py` turns grid rows into ms per step. `results.py` writes every `run.json`, `requests.jsonl` and a probe run's `rows.jsonl` (one `ProbeRow` per line, append-only); besides it only the profiler (`profile.json` and the nsys export in the run dir) and compare (`results/compare/`) write under `results/`.
- `quettabench/profiling/` and `profiler.py`: attach only. They never launch an engine. `engine_side.py` (what the engine reports) and `energy_cost_accuracy.py` (board energy, dollar cost, gsm8k accuracy) are the sources; `client_side.py` (what a client sees) holds the two that import QuettaSim, loaded only when asked for; `kv_transfer.py` is a torchrun target whose `rows.jsonl` the `kv_transfer` source reads.
- `quettabench/probes/`: kernel microbenchmarks and live-server probes. Each run records itself in `run.json` through `metrics/results.py` (`probes/base.py` `run_recorded`). A kernel probe emits `ProbeRow`s (`probes/base.py`: probe, table, key, value, timing) into its run's `rows.jsonl`; it writes no table, and only `kernel_tables` builds them. `Probe` (`grid()`, `run_point() -> ProbeRow`) is the ABC the grid probes use. `probes/_timing.py` holds the shared device layer, CUDA-graph and NCU timing. Their guide and runbooks are in `docs/runbooks/`.
- Probes and profiler sources may send requests to an engine that launcher.py started, to measure engine-level behaviour; they never start an engine or a server and never write outside results/ (`serving_decode_grid`'s request shards are `multiprocessing` children of the probe, listed under "Who writes" below). Today that is the probes `serving_decode_grid` and `serving_stage_split`, and the profiling sources `frontend` and `pd_proxy_overhead` (`profiling/client_side.py`) and gsm8k (`energy_cost_accuracy.py`). Their `--out` paths are the caller's; nothing checks that they stay under results/. Those that take an endpoint instead (`serving_stage_split --no-launch`) may drive an operator-provided server that launcher.py did not start; the operator owns that server.
- The docker launch backend is validated on one cuda host (the qbench image, 15-cell grid within 1.5% of a local install); its rocm, tpu and neuron device flags are tested against mocks only.
- `quettabench/compare.py`: reads run files; its only write is the report json (default `results/compare/`).
- `quettabench/export/quettasim.py`: the only measurement code that writes outside the repo; its builders are `tp_comm`, `roofline_utils`, `serving_decode_grid` and `kernel_tables`; `provision.py` sets the box up outside it (below).
- `configs/`: `models/` holds per-model facts, once. `runs/` reference a model and never restate its facts. `grids.yaml` holds the grids and step points, `hardware/` the link profiles, `workloads.yaml` every workload profile by name.
- `data/`: fetch scripts. Never imported by `quettabench/`, enforced by `tests/contracts/test_imports.py`.
- `docs/runbooks/`: the probe guide; with README.md and AGENTS.md the only docs.
- `tests/golden/`: characterization fixtures of the measured behaviour. Change a fixture only with a dated line in `tests/golden/REPORT.md` saying why.
- `tests/contracts/`: one test module per package area, not per module: `test_workloads.py` covers `workloads/` (sessions included), `test_probes.py` `probes/`, `test_adapters.py` the adapters, `test_export.py` `export/`, `test_profiling.py` `profiling/`, `test_metrics_summary.py` `metrics/summary.py` and `metrics/step_fit.py`, and `config/schema.py` is tested through `test_loader.py`; `provision.py` is tested in `tests/golden/test_provision.py` only. Plus `test_compare_contracts.py` (with `compare.py`'s tests), `test_imports.py` and the GPU smoke module `test_probes_gpu.py`; `helpers.py` and the `fake_*.py` modules are their plumbing.
- `tests/parity/`: `slope_gate.json`, the committed HTTP-vs-in-process slope gate, and the test that holds every cell to its rule.

## Deviations from QBENCH_STRUCTURE.md

One line each; everything else follows section 2 "Target layout".

- The xval door is named `compare`: the subcommand is `qbench compare`, its module `quettabench/compare.py`, its results dir `results/compare/` and its config section `compare`; the structure doc calls the concept `xval`. The structure doc also calls the subcommands "doors"; the code and docs say subcommands.
- The justfile also carries the kernel probe recipes, `probe`, `remote-probe`, `serving-profile`, `profile-gpu`, `gpu-info`, `hooks`, `image`, `fetch` and the `mirror-*` R2 recipes, beyond section 2's nine.
- The probe recipes (`just probe`, the kernel recipes, `remote-probe`) run `python -m quettabench.probes.<name>`, a second way in beside `qbench run` `mode: probe`: torchrun needs a module path per rank, a remote host gets only the probe package, and a live-server probe names its output file; both ways in record through `run_recorded` or `run_probe` into the same `run.json` and `rows.jsonl`.
- `runner.py` is about 370 lines, not under 100: it also holds `build_trace(cfg)`, which picks the Trace a Config names, and runs probe-mode runs, matrix sweeps, the in-process env pins and the run dir rules.
- `configs/hardware/` holds the NCCL link profiles and `configs/workloads.yaml` the named workload profiles, beside section 2's models, grids and runs.
- `InProcessDriver` stays though the parity gate passed (`tests/parity/slope_gate.json`): grid timing runs over HTTP, and in-process is kept only for `mode.return_outputs` text or logits runs, the only path that returns token ids, text or logits.
- `engines/drivers.py` also holds the transformers FIDELITY reference adapter (`engine.name: reference`) beside `ReferenceDriver`; `engines/__init__.py` imports the adapters so they register.
- `modes/serve.py` `model_dir` resolves engine.model through the local HF cache for the grid's KV formula: config resolution inside the procedure, kept there because only a grid run with no `engine.kv_bytes_per_token` needs it.
- `EngineAdapter` has no `profiler_hook`: the profiler attaches through `engine.json` and nothing would call one.
- `modes/serve.py` is about 880 lines, not under 400: the replay, the session dispatch and the grid cells all live in serve, as section 2 assigns them.
- `probes/_timing.py` holds the device layer (backend, sync, device name, event timing), the CUDA-graph and NCU timing, not `probes/base.py`: it imports torch, and the runner imports `base.py` on CPU. It is the only probe or profiling module that calls `torch.cuda`; each probe names the backends it supports there with `require_backend`. Probes run on cuda and rocm (`gemm` and `elementwise` also on cpu); rocm is mock-tested only, `--ncu` is cuda only (rocprofv3 is not wired), and tpu and neuron refuse.
- `serving_decode_grid.py` and `serving_stage_split.py` (read by the export builders) are probes beyond section 2's list.
- `profiling/client_side.py` holds the `frontend` and `pd_proxy_overhead` sources, not `engine_side.py`: both import QuettaSim at the top, and the profiler imports `engine_side.py` for every source.
- `moe`, `collective`, `vllm_allreduce`, `reparallel` and `profiling/kv_transfer.py` emit `ProbeRow`s without the `Probe` ABC: moe allocates each geometry's weights once for all its token points, and the others run under torchrun, where rank 0 emits rows the ranks measured together.
- A run that measured no rows for a table exports no table for it; the per-probe CSVs wrote a header-only table, which replaced (or, without `--append`, reset) the earlier runs' rows.
- `profiling/kv_transfer.py` stays its own module: it is a torchrun target (`-m quettabench.profiling.kv_transfer`) and the tool name in the kernel manifest.
- `tests/` has `golden/`, `contracts/`, `parity/` and `_stubs/` (the vllm, torch and transformers stand-ins the golden CPU runs import): the probe contracts and GPU smoke tests live in `contracts/`.
- `data/` keeps `download_agent_trajectories.py` and `download_coding_prompts.py` apart: section 2's `fetch.py` merge is not done.
- `tools/pre_commit/` (the commit-message and em-dash hooks) and `docker/` (the qbench plus vLLM image `just image` builds) sit outside section 2's tree.
- The root also holds plumbing outside section 2's tree: `.github/`, `.pre-commit-config.yaml`, `_typos.toml`, `.dockerignore` and `LICENSE`.
- `data/truncate_checkpoint.py` builds the layer-truncated vLLM checkpoint for truncated-model runs; it is not a fetch script.
- rocm, tpu and neuron launch paths are tested against mocks only; no hardware run is recorded yet. On them the loader refuses `engine.name=quettaserve` and `qbench profile` refuses the `energy` source, which reads nvidia-smi.
- qbench runs real traces only: a workload source named `random*` or `synthetic*` that is not a file is refused.
- `probes/base.py` imports `quettabench.engines` inside `launch_vllm`, so a remote kernel-probe host without the engine stack can still run a probe.
- `probes/gemm.py`, `flash_attn.py`, `gdn.py`, `vllm_allreduce.py`, `reparallel.py` and `profiling/kv_transfer.py` read `os.environ` themselves: they run as `python -m` modules outside the Config path (under torchrun or on a remote host that gets only the probe package), so torchrun's `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `LOCAL_WORLD_SIZE` and the `--ncu-bin` default `NCU_BIN` reach them only through the environment.
- `serving_decode_grid.py`, `serving_stage_split.py` and `profiling/client_side.py` write their own CSV/JSON at `--out`, not `ProbeRow`s through `metrics/results.py`: `export/quettasim.py` reads the serving probes' files by name and in their own schema, and frontend's `--out` is a standalone dump beside the result `engine_side.py` puts in `profile.json`; `run_recorded` still writes the serving probes' `run.json`.
- `probes/gemm.py` and `probes/moe.py` hold the GEMM shapes and MoE geometries they time: probe grids, not model configs, so they are not read from `configs/models/`.
- `data/` scripts are not under the env rule: `fetch.py` reads `AWS_PROFILE`, `R2_ENDPOINT` and `R2_BUCKET`, the two download scripts take `--credentials` (default `$GOOGLE_APPLICATION_CREDENTIALS`) and need `google-cloud-storage`, and `truncate_checkpoint.py` reads `TMPDIR`. They are one-off data tools outside the package.
- `probes/moe.py --routing` writes a safetensors dump of one MoE block, not `ProbeRow`s: it is an engine-side diff input, not a table.
- `config/loader.py` runs `nvidia-smi topo -m` on cuda to pick the NCCL link profile when `hardware.link_profile` is `auto`; it only reads the topology.
- `just mirror-up` is a second route out of the repo beside `export/quettasim.py`: it uploads `results/probes/` to R2 as a backup of the probe runs.
- `profiling/client_side.py` imports QuettaSim, the one package module that needs a QuettaSim checkout; `engine_side.py` loads it by name only for the `frontend` and `pd_proxy_overhead` sources.

## Who writes, who starts a process

Every writer and every process starter, exactly:

- `engines/launcher.py`: `engine.json` and `server.log` in the run dir; starts the engine server process (local, or `docker run`), for a run or a probe.
- `BenchDriver` (`engines/drivers.py`): `bench.log` in the run dir; starts the QuettaServe bench process, once per grid cell.
- `InProcessDriver` (`engines/drivers.py`): builds the vLLM engine inside this process; starts no process.
- `metrics/results.py` `RunWriter`: `run.json`, `requests.jsonl` and a probe run's `rows.jsonl`.
- probes: their `run.json` and `rows.jsonl` (through `RunWriter`) in one dir under `results/` (default `results/probes/<gpu>/<probe>/<run-id>/`); the live-server probes their CSV and JSONL outputs there; `serving_decode_grid` also starts its request shards as `multiprocessing` processes, which write their results to a `shards-*` dir in that run dir, and `probes/_timing.py` (gemm, flash_attn, gdn `--ncu`) starts `ncu` with a `python -c` child per timed point.
- `justfile` `remote-probe` (`probes/base.py` `remote_probe`): starts `ssh` and `rsync`; on `REMOTE_HOST`, makes its own `mktemp -d /tmp/qbench-probes.XXXXXX`, writes the shipped package and `out/` there, runs the probe, and removes that dir once its run is pulled; locally, adds the pulled run under `results/probes/<gpu>/` and stamps its `run.json` `provenance.pulled` through `metrics/results.py` `stamp_pulled`.
- `profiler.py`: `profile.json` (and the nsys export) in the run dir.
- `profiling/kv_transfer.py`: its `rows.jsonl` and `run.json` in its probe run dir (`results/probes/<gpu>/kv_transfer/<run-id>/`), run under its own torchrun ranks.
- `profiling/client_side.py`: the frontend source's JSON at `--out`, when given, and the pd_proxy_overhead source's CSV at `--out`.
- `workloads/tokenize.py`: the tokenized request jsonl at `--out`.
- `compare.py`: its report json (default `results/compare/`).
- `export/quettasim.py`: the QuettaSim tables at `--dest`, and the kernel tables, built from the ok probe runs' `rows.jsonl`, under `--kernel-dest` (QuettaSim's data root), outside the repo.
- `provision.py`: outside the repo, the `cache_root` dirs, the QuettaServe `git clone`, `pip install`, the weights download and `cargo` builds; in the repo, `.env`.
- `justfile` `image`: starts `docker build` of `docker/Dockerfile`, which writes the image `qbench:<short sha>` to the local docker store and nothing in the repo.
- `justfile` `mirror-up`: uploads `results/probes/` to R2 under `results/probes/`, a backup of the probe runs and the second route out of the repo; it never deletes there and never writes QuettaSim's kernel tables. `mirror-down` writes that prefix back into `results/probes/`.

## Laws

1. The runner launches (probes start a server only through the launcher), the profiler attaches, compare and export read.
2. Every measured run's requests go through `serve.py`; the probe and profiler exception above is the only other code that sends requests.
3. One hash: workloads compute it, serve gates on it, compare joins on it.
4. A number that is not comparable is withheld, not footnoted.
5. A setting is added in exactly one place: a `schema.py` field, an adapter table row, and an `env_map` row only if an env var carries it.
6. No module outside `config/` reads the environment, except the allowlist in `tests/contracts/test_env_map.py`: the torchrun probes read `RANK`, `WORLD_SIZE`, `LOCAL_RANK` and `LOCAL_WORLD_SIZE` (the launcher's protocol; reparallel reads GPUs per node from the last), the `--ncu` probes read `NCU_BIN`, and `profiling/kv_transfer.py` reads its torchrun ranks and sets `UCX_RCACHE_MAX_UNRELEASED` as vLLM does.
7. Rows are append-only jsonl, so a crash keeps partial data.

## How to change things

- One mechanism per commit, about 300 changed lines (deletions not counted). The message states the mechanism.
- Pure moves (`git mv`) are their own commits and carry nothing else.
- Goldens pass unchanged. A fixture that has to change is a behaviour change and gets its line in `tests/golden/REPORT.md`.
- `just test` and `just lint` are green before every commit. The lint gate is ruff, mypy and the pre-commit hooks; mypy checks `quettabench/` as configured in `pyproject.toml`.
- No attribution trailers in code, commits or PRs: the author field is the only authorship.
- No em-dash in prose; write `--`.
- Imports at the top of the file, except torch, vLLM, GPU-only and optional packages, imported where used so the CPU paths and tests load without them, and transformers and datasets, imported where used because they load slowly and only tokenizing paths need them; a new one goes in `OPTIONAL_PACKAGES` in `tests/contracts/test_imports.py`. No inline python in shell or justfile recipes. A recipe calls into the package where it can; `gpu-info`, `image`, `hooks`, `test`, `lint`, `fetch` and the `mirror-*` recipes call host tools (ssh, the backend's device tool (nvidia-smi, amd-smi, neuron-ls), docker, pre-commit, pytest, ruff, aws) or `data/fetch.py` and `data/build_trajectories.py` instead, and `gpu-info` and `profile-gpu` run more than one line.
- Measured before kept: an optimization lands with the measurement that justifies it, and a feature lands with its measurement.
- Report every number, including the ones that weaken a claim.

## Where to look

- `README.md`: usage and the results layout.
- `tests/golden/REPORT.md`: every fixture change.
- `QBENCH_STRUCTURE.md`: the design. The owner keeps it outside the repo. When QBENCH_STRUCTURE.md and the code disagree, QBENCH_STRUCTURE.md is the source of truth.
