# AGENTS.md

Read this before you change anything here. It states what the repo is, where
each thing lives, and the rules every commit keeps.

## What this repo is

QuettaBench is a measurement harness. It has one command, `qbench`, with four
doors: `run | profile | crossval | export`, plus `provision` for box setup.
It has one measurement procedure, `quettabench/modes/serve.py`. Three structs flow one way, `Config ->
LaunchSpec -> RequestResult`, and nothing flows back. Every measurement lands
in one results layer, `results/<run>/`, and leaves the repo by
`quettabench/export/quettasim.py`; the probes' tables also leave by `just mirror-up`
(below).

## Layout

- `quettabench/cli.py`, `runner.py`, `provision.py`: the command, the run
  sequence, and box setup. `runner.py` is the only code that starts a run.
- `quettabench/config/`: schema (with the serving-topology rules), loader,
  env_map. The only place `os.environ` is read is `config/env_map.py`,
  enforced by `tests/contracts/test_env_map.py`; its allowlist only shrinks.
- `quettabench/workloads/`: every request source is a `Trace` with a
  `workload_hash`. A source without a hash does not belong here.
- `quettabench/engines/`: adapters translate a Config to a LaunchSpec from a
  field table; a field with no flag raises `UnsupportedField`, never a silent
  drop. Drivers are the only code in a run that talks to an engine.
  `launcher.py` is the only code that starts an engine server, for a run or
  a probe (`probes/base.py` `launch_vllm`); `BenchDriver` runs the
  QuettaServe bench, which is its own engine, once per grid cell. Other
  subprocesses are tools (nvidia-smi, git, nsys export, docker, cargo in
  provision), never an engine.
- `quettabench/modes/`: `serve.py`, nothing else: the request replay, the
  session dispatch and the grid cells. Nothing in `modes/` branches on the
  engine name; an engine difference is a driver.
- `quettabench/metrics/`: one percentile function, and the caller states the
  method. `step_fit.py` turns grid rows into ms per step. `results.py`
  writes every `run.json` and `requests.jsonl`; besides it only the probes
  (their tables under `results/probes/`), the profiler (`profile.json` and
  the nsys export in the run dir) and crossval (`results/xval/`) write
  under `results/`.
- `quettabench/profiling/` and `profiler.py`: attach only. They never launch
  an engine. `native.py` and `wrapped.py` are the sources; `frontend.py` holds
  the two that import QuettaSim, loaded only when asked for; `kv_transfer.py`
  is a torchrun target whose CSV the `kv_transfer` source reads.
- `quettabench/probes/`: kernel microbenchmarks and live-server probes. Each
  run records itself in `run.json` through `metrics/results.py`
  (`probes/base.py` `run_recorded`). `probes/_timing.py` holds the shared
  CUDA-graph and NCU timing. Their guide and runbooks are in `docs/runbooks/`.
- Owner decision (2026-10-11): probes and profiler sources may send requests
  to an engine that launcher.py started, to measure engine-level behaviour;
  they never start an engine or a server and never write outside results/
  (`serving_decode_grid`'s request shards are `multiprocessing` children of
  the probe, listed under "Who writes" below). Today that
  is the probes `serving_decode_grid` and `serving_stage_split`, and the
  profiling sources `frontend` and `pd_host` (`profiling/frontend.py`) and
  gsm8k (`wrapped.py`). Their `--out` paths are the caller's; nothing checks
  that they stay under results/. Those that take an endpoint instead
  (`serving_stage_split --no-launch`) may drive
  an operator-provided server that launcher.py did not start; the operator
  owns that server.
- Docker launch backend ships tested against mocks only, by owner decision; validate on a docker host before relying on it.
- `quettabench/crossval.py`: reads run files; its only write is the report json
  (default `results/xval/`).
- `quettabench/export/quettasim.py`: the only measurement code that writes
  outside the repo; its builders are `tp_comm`, `stage_split_rates`,
  `roofline_utils`, `serving_decode_grid` and `kernel_tables`; `provision.py` sets the box up outside it (below).
- `configs/`: `models/` holds per-model facts, once. `runs/` reference a model
  and never restate its facts. `grids.yaml` holds the grids and step points,
  `hardware/` the link profiles, `workloads.yaml` every workload profile by name.
- `data/` and `analysis/`: fetch scripts and one-off studies. Never imported by
  `quettabench/`, enforced by `tests/contracts/test_imports.py`.
- `docs/runbooks/`: the probe guide and runbooks; with README.md and AGENTS.md
  the only docs.
- `tests/golden/`: characterization fixtures captured from the pre-refactor
  code. Change a fixture only with a dated line in `tests/golden/REPORT.md`
  saying why.
- `tests/contracts/`: one test module per package module (`test_<module>.py`),
  plus `test_crossval_contracts.py`, `test_imports.py` and the two GPU smoke
  modules; `helpers.py` and the `fake_*.py` scripts are their plumbing.

## Deviations from QBENCH_STRUCTURE.md

One line each; everything else follows section 2 "Target layout".

- The doors are `run | profile | crossval | export` plus `provision`; section 2 names the third `xval`.
- The justfile also carries the kernel probe recipes, `probe`, `remote-probe`, `serving-profile`, `profile-gpu`, `gpu-info`, `hooks` and the `mirror-*` R2 recipes, beyond section 2's nine.
- `runner.py` is about 370 lines, not under 100: it also holds `build_trace(cfg)`, which picks the Trace a Config names, and runs probe-mode runs, matrix sweeps, the in-process env pins and the run dir rules.
- `configs/hardware/` holds the NCCL link profiles and `configs/workloads.yaml` the named workload profiles, beside section 2's models, grids and runs.
- `engines/drivers.py` also holds the transformers FIDELITY reference adapter (`engine.name: reference`) beside `ReferenceDriver`; `engines/__init__.py` imports the adapters so they register.
- `modes/serve.py` is about 820 lines, not under 400: the replay, the session dispatch and the grid cells all live in serve, as section 2 assigns them.
- `probes/_timing.py` holds the CUDA-graph and NCU timing, not `probes/base.py`: it imports torch, and the runner imports `base.py` on CPU.
- `serving_decode_grid.py` and `serving_stage_split.py` (read by the export builders) are probes beyond section 2's list.
- `profiling/frontend.py` holds the `frontend` and `pd_host` sources, not `native.py`: both import QuettaSim at the top, and the profiler imports `native.py` for every source.
- `profiling/kv_transfer.py` stays its own module: it is a torchrun target (`-m quettabench.profiling.kv_transfer`) and the tool name in the kernel manifest.
- `tests/` has `golden/` and `contracts/` only: the probe contracts and GPU smoke tests live in `contracts/` (owner decision 2026-10-12), and `tests/parity/` waits for the GPU parity gate.
- `analysis/` has no synthetic-sampler studies or `build_trace_distributions`: `analysis/synthetic/` was deleted by owner decision (2026-10-12, "scrap synthetic").
- `data/` keeps `download_agent_trajectories.py`, `download_coding_prompts.py`, `build_chat_trajectories.py` and `build_claweval_trajectories.py` apart: section 2's `fetch.py` and `build_trajectories.py` merges are not done.
- `legacy/trace_serve.py` is gone; `modes/serve.py` replaced it.
- `scripts/crossval/baselines/` (the compat baselines dir) and `tools/pre_commit/` (the commit-message and em-dash hooks) sit outside section 2's tree.

## Who writes, who starts a process

Every writer and every process starter, exactly:

- `engines/launcher.py`: `engine.json` and `server.log` in the run dir; starts
  the engine server process (local, or `docker run`), for a run or a probe.
- `BenchDriver` (`engines/drivers.py`): `bench.log` in the run dir; starts the
  QuettaServe bench process, once per grid cell.
- `InProcessDriver` (`engines/drivers.py`): builds the vLLM engine inside this
  process; starts no process.
- `metrics/results.py` `RunWriter`: `run.json` and `requests.jsonl`.
- probes: their tables and their `run.json` in one dir under `results/`
  (default `results/probes/<gpu>/<probe>/<run-id>/`); `serving_decode_grid` also starts its request
  shards as `multiprocessing` processes, and `probes/_timing.py` (gemm, flash_attn,
  gdn `--ncu`) starts `ncu` with a `python -c` child per timed point.
- `justfile` `remote-probe` (`probes/base.py` `remote_probe`): starts `ssh`
  and `rsync`; on `REMOTE_HOST`, makes its own `mktemp -d
  /tmp/qbench-probes.XXXXXX`, writes the shipped package and `out/` there,
  runs the probe, and removes that dir once its run is pulled; locally, adds
  the pulled run under `results/probes/<gpu>/` and stamps its `run.json`
  `provenance.pulled` through `metrics/results.py` `stamp_pulled`.
- `profiler.py`: `profile.json` (and the nsys export) in the run dir.
- `profiling/kv_transfer.py`: its table and `run.json` in its probe run dir
  (`results/probes/<gpu>/kv_transfer/<run-id>/`), run under its own torchrun
  ranks.
- `profiling/frontend.py`: the frontend source's JSON at `--out`, when given,
  and the pd_host source's CSV at `--out`.
- `workloads/tokenize.py`: the tokenized request jsonl at `--out`.
- `crossval.py`: its report json (default `results/xval/`).
- `export/quettasim.py`: the QuettaSim tables at `--dest`, and the kernel tables under
  `--kernel-dest` (QuettaSim's data root), outside the repo.
- `provision.py`: outside the repo, the `cache_root` dirs, the QuettaServe
  `git clone`, `pip install`, the weights download and `cargo` builds; in the
  repo, `.env`.
- `justfile` `mirror-up`: uploads `results/probes/` to R2 under
  `profiling/qbench_probes/`, a backup of the probe runs and the second
  route out of the repo; it never deletes there and never writes
  QuettaSim's `profiling/kernel_data/`. `mirror-down` writes that prefix
  back into `results/probes/`.

## Laws

1. The runner launches (probes start a server only through the launcher),
   the profiler attaches, crossval and export read.
2. Every measured run's requests go through `serve.py`; the owner-decision
   exception above is the only other code that sends requests.
3. One hash: workloads compute it, serve gates on it, crossval joins on it.
4. A number that is not comparable is withheld, not footnoted.
5. A setting is added in exactly one place: a `schema.py` field, an adapter
   table row, and an `env_map` row only if an env var used to carry it.
6. No module outside `config/` reads the environment, except the allowlist
   in `tests/contracts/test_env_map.py`: the torchrun probes read `RANK`,
   `WORLD_SIZE`, `LOCAL_RANK` and `LOCAL_WORLD_SIZE` (the launcher's
   protocol; reparallel reads GPUs per node from the last), the `--ncu`
   probes read `NCU_BIN`, and `profiling/kv_transfer.py` reads its torchrun
   ranks and sets `UCX_RCACHE_MAX_UNRELEASED` as vLLM does.
7. Rows are append-only jsonl, so a crash keeps partial data.

## How to change things

- One mechanism per commit, about 300 changed lines (deletions not counted).
  The message states the mechanism.
- Pure moves (`git mv`) are their own commits and carry nothing else.
- Goldens pass unchanged. A fixture that has to change is a behaviour change
  and gets its line in `tests/golden/REPORT.md`.
- `just test` and `just lint` are green before every commit. The lint gate
  is ruff and the pre-commit hooks; mypy is not part of it.
- No attribution trailers in code, commits or PRs: the author field is the only authorship.
- No em-dash in prose; write `--`.
- Imports at the top of the file. No inline python in shell or justfile
  recipes; a recipe is one command line into the package. `gpu-info` is the
  exception: host-shell glue (ssh, nvidia-smi) that runs where the package
  may not be installed.
- Measured before kept: an optimization lands with the measurement that
  justifies it, and a feature lands with its measurement.
- Report every number, including the ones that weaken a claim.

## Where to look

- `README.md`: usage, the results layout, and the old-command to new-command
  map.
- `tests/golden/REPORT.md`: every known behaviour difference from the
  pre-refactor code.
- `QBENCH_STRUCTURE.md`: the design. The owner keeps it outside the repo. When
  QBENCH_STRUCTURE.md and the code disagree, QBENCH_STRUCTURE.md is the source
  of truth.
