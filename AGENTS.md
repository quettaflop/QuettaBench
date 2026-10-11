# AGENTS.md

Read this before you change anything here. It states what the repo is, where
each thing lives, and the rules every commit keeps.

## What this repo is

QuettaBench is a measurement harness. It has one command, `qbench`, with four
doors: `run | profile | crossval | export`, plus `provision` for box setup.
It has one measurement procedure, `quettabench/modes/serve.py`. Three structs flow one way, `Config ->
LaunchSpec -> RequestResult`, and nothing flows back. Every measurement lands
in one results layer, `results/<run>/`, and leaves the repo by one route,
`quettabench/export/`.

## Layout

- `quettabench/cli.py`, `runner.py`, `provision.py`: the command, the run
  sequence, and box setup. `runner.py` is the only code that starts a run.
- `quettabench/config/`: schema, loader, validate, env_map. The only place
  `os.environ` is read is `config/env_map.py`, enforced by
  `tests/config/test_no_environ.py`; its allowlist only shrinks.
- `quettabench/workloads/`: every request source is a `Trace` with a
  `workload_hash`. A source without a hash does not belong here.
- `quettabench/engines/`: adapters translate a Config to a LaunchSpec from a
  field table; a field with no flag raises `UnsupportedField`, never a silent
  drop. Drivers are the only code that talks to an engine. `launcher.py` is
  the only code that starts a process.
- `quettabench/modes/`: `serve.py` and its two private siblings (`_grid.py`,
  `_sessions_dispatch.py`), nothing else. Nothing in `modes/` branches on the
  engine name; an engine difference is a driver.
- `quettabench/metrics/`: one percentile function, and the caller states the
  method. `step_fit.py` turns grid rows into ms per step. `results.py` is the
  only writer of `results/`.
- `quettabench/profiling/` and `profiler.py`: attach only. They never launch
  an engine and never send load.
- `quettabench/probes/`: kernel microbenchmarks; they write through
  `metrics/results.py`. Their runbooks are in `probes/runbooks/`.
- `quettabench/crossval/`: reads files only.
- `quettabench/export/`: the only code that writes outside the repo.
- `configs/`: `models/` holds per-model facts, once. `runs/` reference a model
  and never restate its facts. `grids.yaml` holds the grids and step points,
  `hardware/` the link profiles, `workloads/` one yaml per workload profile.
- `data/` and `analysis/`: fetch scripts and one-off studies. Never imported by
  `quettabench/`, enforced by `tests/test_analysis_is_leaf.py`.
- `tests/golden/`: characterization fixtures captured from the pre-refactor
  code. Change a fixture only with a dated line in `tests/golden/REPORT.md`
  saying why.

## Laws

1. The runner launches, the profiler attaches, crossval and export read.
2. Everything that sends requests or steps an engine is `serve.py`.
3. One hash: workloads compute it, serve gates on it, crossval joins on it.
4. A number that is not comparable is withheld, not footnoted.
5. A setting is added in exactly one place: a `schema.py` field, an adapter
   table row, and an `env_map` row only if an env var used to carry it.
6. No module outside `config/` reads the environment.
7. Rows are append-only jsonl, so a crash keeps partial data.

## How to change things

- One mechanism per commit, about 300 changed lines (deletions not counted).
  The message states the mechanism.
- Pure moves (`git mv`) are their own commits and carry nothing else.
- Goldens pass unchanged. A fixture that has to change is a behaviour change
  and gets its line in `tests/golden/REPORT.md`.
- `just test` and `just lint` are green before every commit.
- No attribution trailers in code, commits or PRs: the author field is the only authorship.
- No em-dash in prose; write `--`.
- Imports at the top of the file. No inline python in shell or justfile
  recipes; a recipe is one command line into the package.
- Measured before kept: an optimization lands with the measurement that
  justifies it, and a feature lands with its measurement.
- Report every number, including the ones that weaken a claim.

## Where to look

- `README.md`: usage, the results layout, and the old-command to new-command
  map.
- `tests/golden/REPORT.md`: every known behaviour difference from the
  pre-refactor code.
- `QBENCH_STRUCTURE.md`: the design. The owner keeps it outside the repo. When
  this file and the code disagree, it is the source of truth.
