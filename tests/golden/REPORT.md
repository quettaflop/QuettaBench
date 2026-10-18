# Golden fixtures

Run `python3 -m pytest tests/golden -q`. `QBENCH_GOLDEN_REGEN=1` rewrites the fixtures from the code, for an intended behaviour change only; each such change gets a dated line below.

## Fixture changes

2026-10-18: `aggregate_summary.json` gains `server_cached_tokens` and `server_cache_hit_rate`, null here because no request in the fixture carries an engine-reported cached count. These are the cache-hit metric; `cache_hit_rate` and the per-turn `median_cache_hit_rate` are the prompt share an ideal cache could reuse, an upper bound. Every other value is as before.

2026-10-18: `percentiles.json` nearest-rank values moved one rank down where `p/100 * n` is a whole number: `arange_1000` p50/p90/p95/p99 are 499/899/949/989 (were 500/900/950/990) and `two` p50 is 10 (was 20). `percentile(..., "nearest")` now takes the ceil(p/100 * n)-th smallest value, the nearest-rank definition; it took the (floor(p/100 * n) + 1)-th, one rank high whenever the product was whole. Serve, SLA and burst percentiles in recorded runs are from the old rank; linear percentiles did not move.

2026-10-18: `aggregate_summary.json` gains `sessions_failed`, 0 here. `sessions_completed` now counts sessions whose every planned turn succeeded instead of sessions that sent anything; the fixture's sessions all succeed, so its value and every other value are as before.

2026-10-12: `aggregate_summary.json` gains `itl_coalesced_frac`, 0.0 here. Every other value is as before.

2026-10-12: `aggregate_summary.json` gains `steady_median_ttft_ms` and `steady_p99_ttft_ms`, null here because the fixture is not a closed-loop run. Every other value is as before.

2026-10-11: `config_failures.json` and `config_trace_serve_style_error.json` hold `use aggregated, disagg:<P>p<D>d, afd:<A>a<F>f, or ep<N>dp<M>` as the serving-style refusal, because `parse_serving_style` accepts `afd:<A>a<F>f` and its message lists it. Every spec is refused as before; only the text changed.

2026-10-11: no fixture changed. A serve run whose workload profile sets `prefix_caching_required: true` gets `engine.prefix_caching = true` (origin `profile`), and an explicit `engine.prefix_caching=false` on such a profile is refused at load.

2026-10-11: no fixture changed. The serve identity gate hashes every dispatched request, failed or not (`serve.served`), in a single run and in each goodput-sweep probe, so one failed request leaves the run `partial`, and below `mode.min_success_rate` it aborts with exit 1. WORKLOAD_MISMATCH means the requests sent differ from the trace: a request changed after hashing, or a multi-turn session whose failed turn left its later turns unsent. Chat requests without token ids hash their messages.

2026-10-11: no fixture changed; the workload_hash pins in `tests/contracts/test_workloads.py` did. A chat request without token ids hashes a canonical-JSON digest of its messages (assistant content left out) together with seed, order, arrival, lengths, session and turn, so different chat profiles no longer share a hash (`chat-singleturn` held `2c2c330fe937283f`, as `chat-medium` and `chat-singleturn-synth` did). Token-id requests hash as before and the content digests did not move. A run recorded before 4318bf0 does not join on workload_hash with one recorded after it for these profiles:

| profile | before 4318bf0 | after |
|---|---|---|
| `chat-singleturn` | `2c2c330fe937283f` | `1b84966cb9c4facc` |
| `chat-multiturn-short` | `57c82245406bcdf1` | `c51f134d3946f53d` |
| `coding-singleturn` | `03481570266bb2d4` | `15384d696a4d7a1b` |
| `swebench-multiturn-short` | `d4dca931e18de175` | `9f0aa9d6f98e5bd3` |
| `test` | `c19e2bb3db6d4434` | `7ae706bdf8eeae1d` |

2026-10-11: no fixture changed; the `mooncake-trace` pin in `tests/contracts/test_workloads.py` holds `52bae3bbfe646fd8` / `a1384dbf00844c6c` (workload_hash / content digest), from `0e6f083e59f1f9dd` / `d483cd1411833245`, because `MooncakeTrace` keeps each record's timestamp in seconds as `arrival_ts`, so `workload.arrival=trace` replays the trace's timing. A run recorded before 969c7f7 does not join on workload_hash with one recorded after it.

2026-10-11: no fixture changed; the `mooncake-trace` pin holds `a03be7b0ed268d81` / `b5ed59cd7a0d7aff`, from `52bae3bbfe646fd8` / `a1384dbf00844c6c`, because a Mooncake draw that wraps past the file's end shifts each lap k by k * (span + mean gap) of the file's timestamps, so wrapped requests no longer fire together with the first lap's.

2026-10-11: no fixture changed. The live-server probes start vLLM through `probes.base.launch_vllm` (the vllm adapter, then `launcher.start`), pinned in `tests/contracts/test_probes.py`:

- all three: the server is `<python's dir>/vllm serve <model>`; its env holds `VLLM_SERVER_DEV_MODE=1` (mounts /reset_prefix_cache) and `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` (KV is sized after graph capture, so the reported KV pool can differ from a launch without it); the argv holds `--trust-remote-code`; flags come in the adapter's order.
- `pool_capacity` and `serving_stage_split` pass `--enable-chunked-prefill`.
- `serving_decode_grid` also passes `--enable-chunked-prefill`. Whether that changes its scheduling depends on the vLLM build's default for chunked prefill; no run has measured it.

2026-10-11: no fixture changed. A run with an SLA whose requests produced no TPOT sample (none decoded 2+ tokens) records `sla.tpot_p99` as null and prints `SLA_SUMMARY ... tpot_p99=-/<sla>`, not a number nothing measured.

2026-10-11: no fixture changed. `profiling/serving_frontend.py` takes its frontend-off reference from QuettaSim's `engine.sim.closed_loop.predict_cell` (the vLLM-scheduler replay). `mult_curve` is the measured TTFT residual over that reference, so a `mult_curve` exported to a device yaml before 4383be8 is not comparable with one exported after it; no run has measured how far they differ.

2026-10-11: no fixture changed. The vllm adapter passes `engine.extra.vllm.api_key` to the server as the `VLLM_API_KEY` env var, so vLLM's startup args log does not print it; the three live-server probes' launches pinned in `tests/contracts/test_probes.py` hold that env var and no `--api-key`. A run's HttpDriver sends that key.

2026-10-11: no fixture changed. `run_name` matches `[A-Za-z0-9][A-Za-z0-9._-]*` and is none of `probes`, `compare`, `baselines`; load refuses any other name, since it names `results/<run_name>/`, which `--overwrite` empties. A matrix suffix has every other character replaced by `_` (`engine.model: /w/llama` gives `-model_w_llama`). `--overwrite` empties only a real dir directly under the results dir that holds `run.json`, `requests.jsonl` or `engine.json`, and a run dir whose `run.json` says failed is replaced without it.

2026-10-11: no fixture changed. Each probe run started from its CLI (`run_recorded`) writes its tables and `run.json` into its own `results/probes/<gpu>/<probe>/<run-id>/`, the run id being its UTC start (`20261011T120000.000000Z`), so each `run.json` stands for one run's tables. A `qbench run` probe (`mode: probe`) writes `results/<run>/<probe>/`.

2026-10-11: no fixture changed. A probe run from its CLI opens `requests.jsonl` in its run dir before the probe starts, so a run killed before it finishes leaves no `run.json` and export skips its tables. A probe that names its output file (`serving_stage_split --out`, `serving_decode_grid --out-raw`) refuses a dir that already records a run.

2026-10-11: no fixture changed. The `vllm_allreduce` probe writes its `all_reduce` rows to its own table and the collective probe writes its rows to another, so no probe-run table holds both; the `kernel_tables` export builder merges them into one `kernel_data/graph/collectives/{gpu}_tp{N}.csv`: the collective probe's rows other than `all_reduce` (`dtype_bytes` 2 and `path` nccl when absent), then `vllm_allreduce`'s `all_reduce` rows. The merge does not depend on which probe ran last. Likewise gemm's `fp8_act_quant` entry and the elementwise probe's entries merge into one `elementwise/{gpu}.json`.

2026-10-11: no fixture changed. `just mirror-up`, `mirror-down` and `mirror-preview` sync `results/probes/` with the R2 prefix `profiling/qbench_probes/`, fixed in the justfile, which keeps the mirror apart from QuettaSim's kernel data.

2026-10-11: no fixture changed. `engine.name` accepts only the engines with an adapter (`vllm`, `quettaserve`, `reference`); any other name is refused at load with that list.

2026-10-11: no fixture changed. A probe run writes only the rows it measured, and `remote-probe` sends no earlier run up. The `kernel_tables` export unions the rows of every ok run of a probe per table, keyed on the probe's row key (flash_attn and chunked_prefill_attn on the head config, gemm on M, N, K, moe on its five geometry columns, reparallel on its five key columns, gemm's `fp8_act_quant` on its JSON key), and on a key conflict the run with the latest `provenance.finished` wins; a gemm or moe run without `--append` drops the rows of runs that finished before it. A table with no row key comes from the run with the latest `provenance.finished`. Results dirs given to export are pooled by that rule.

2026-10-11: no fixture changed. `just remote-probe` runs `probes/base.py` `remote_probe`: each invocation makes its own remote `mktemp -d /tmp/qbench-probes.XXXXXX`, passes `--run-id` from the local clock, always pulls the run back (a failed run's too) and exits with the remote run's exit code, then removes only that dir; a failed pull keeps it. Concurrent invocations do not touch each other's runs.

2026-10-11: no fixture changed. `qbench export` takes the kernel tables' destination as its own `--kernel-dest` (QuettaSim's data root), and the `kernel_tables` builder runs only when it is given; the other builders take `--dest` and run only when that is given.

2026-10-11: no fixture changed. `redact` treats every string value that holds a flag (a `-` at its start or after whitespace) as a command line, and every argv item holding whitespace too, so `--api-key X` at the start of probe_args, `-H 'Authorization: Bearer X'`, a newline before the flag and `sh -c "... --api-key X"` are written REDACTED. `passphrase` is a secret word and ssh, encryption and signing name a secret key.

2026-10-11: no fixture changed. `kernel_tables` builds every table in a temp dir under `--kernel-dest` and checks each row first: a row with an extra or missing cell, an empty key cell or a key that does not parse (gemm `M` of `1.0` included), JSON that does not parse, or a union JSON that is no object stops the export with `<rel>: <run dir> row N malformed` (or `malformed JSON`) and the dest files untouched; then each file replaces its dest file with `os.replace`. Key columns compare as numbers (`total_weight_gb` 16 and 16.0 are one key), and within one run's table the last row on a key wins (geometry-keyed tables keep every row). Only gemm's elementwise `.json` unions across runs; a `.csv` there is the latest run's.

2026-10-11: no fixture changed. Export compares `provenance.finished` as UTC instants (`datetime.fromisoformat(...).astimezone(timezone.utc)`); a missing or unparsable time sorts earliest, and `01:00+01:00` sorts before `00:30+00:00`.

2026-10-11: no fixture changed. The kernel tables' manifest is `<kernel-dest>/kernel_data/qbench_manifest.json`.

2026-10-11: no fixture changed. A unioned CSV's columns come in the order the runs first name them, earliest run first.

2026-10-11: no fixture changed. `clear_run_dir` unlinks `run.json`, `requests.jsonl` and `engine.json` in the `.clearing-*` dir before its rmtree, and export's run scan skips any `.clearing-*` dir, so an interrupted rmtree is never read as an ok run.

2026-10-11: no fixture changed. `remote_probe` makes the remote dir and its `out/` in one ssh (`mktemp -d ... && mkdir -p $d/out && echo $d`), taking the last output line that is such a dir, so a probe that wrote nothing pulls nothing and its exit code is returned; ssh gets `-o ConnectTimeout=30`; a `--host` starting with `-` and `--out`, `--out-dir` or `--run-id` among the probe args are refused; a pull that fails or raises says the remote dir is kept, and a failed pull's exit names the probe's exit code. `just remote-probe` passes its args through `"$@"` (`set positional-arguments`) after `--`, so an arg holding a space, `$` or `"` reaches the probe whole.

2026-10-11: no fixture changed. The gemm and moe probes take no abbreviated flags, so `--app` is refused instead of appending while export reads it as a reset. `serving_stage_split --emit-nsys-cmd` records no run.json.

2026-10-11: no fixture changed. The `frontend` and `pd_proxy_overhead` profiler sources take the GPU's QuettaSim `device_spec/<gpu>.yaml` as a required `--device-yaml`; `just serving-profile` passes `$QUETTASIM/device_spec/<gpu>.yaml`. `serving_frontend` imports `engine.sim.closed_loop` and `kernel_composed.types` at the top, as QuettaSim main (d8503b8) has both.

2026-10-12: no fixture changed. `kernel_tables` writes QuettaSim main's (d8503b8) `kernel_data/graph/` and `kernel_data/eager/`, a probe run's `ncu/` going to `graph/` and `cuda_event/` to `eager/`, and writes beside each table the `<table>.meta.json` sidecar QuettaSim's `tools/kernel_manifest.py check` requires: `mode` from the probe, its flags and the kind (`graph_replay`, `ncu` with `--ncu`, `amortized_wallclock` for collectives, `reference` for `*_nccl_vs_vllm.csv`, `host_wallclock` for host-timed kinds, else `eager`; `unknown` when the runs kept were timed two ways), `tool`, `reduce`, `gpu` and `measured`. A stopped export removes the `--kernel-dest` it made; no `qbench_manifest.json` is written when no table was found; an empty CSV and unparsable `probe_args` stop the export naming the run instead of raising a traceback; `_inputs.files` and `find` skip paths under a `.clearing-*` dir, as `run_dirs` does.

2026-10-12: no fixture changed. A pulled remote run's `provenance.finished` is not rewritten; `remote_probe` stamps `provenance.pulled` from the local clock through `metrics/results.py` `stamp_pulled`, and export orders a run by `pulled` when present, else `finished`. A pulled `run.json` that does not parse or whose provenance is not an object is warned about, naming the run, and the remote dir is still removed. Every probe parser takes no abbreviated flag (`add_output_args`, and `serving_stage_split` and `serving_decode_grid` themselves), and `remote_probe` refuses `--gpu-label` and `--gpu` among the probe args.

2026-10-12: no fixture changed. In a command line held as one string, only a flag item hides the next word, so free text such as `Please - pass the salt` or `You are a pass-through bot` is kept whole. A mapping value with no flag but a secret `NAME=value` (`HF_TOKEN=sk vllm serve m`, `export HF_TOKEN=sk; vllm serve`) is redacted as a command line, rejoined by `shlex.join`, so the `;` stuck to the value goes with it. An argv item that is a JSON object or list (`--hf-overrides '{"api_key": "sk"}'`) is parsed, redacted and dumped again, and kept as written when it holds no secret.

2026-10-12: no fixture changed. `profiling/kv_transfer.py` records its run like every probe: it takes `add_output_args` and runs through `run_recorded`, rank 1 sends the rows it measured to rank 0, and rank 0 writes them to `results/probes/<gpu>/kv_transfer/<run-id>/cuda_event/kv_transfer/<gpu>_nixl.csv`. `qbench export --only kernel_tables` carries the runs to `kernel_data/eager/kv_transfer/<gpu>_nixl.csv` with a `host_wallclock` sidecar whose tool is `quettabench.profiling.kv_transfer`, unioned on the row key (`layout` compared as text) and sorted on it, the run that finished last winning a key.

2026-10-12: no fixture changed. `kernel_tables` merges into the tables `--kernel-dest` holds instead of replacing them, so an export over QuettaSim's data root keeps `graph/gemm/H200.csv` (1949 rows), `eager/kv_transfer/H200_nixl.csv`, `graph/flash_attn/H200.csv` (ten head configs) and `graph/collectives/H200_tp2.csv` (69 rows) whole and keeps the sidecars' notes. A dest table is the earliest source on the same row key (collectives on `op, world, bytes`), its columns join the union and a JSON table's entries join too; where its rows would be dropped (no row key, or a gemm or moe run without `--append`) the export stops naming the file and its row counts unless `--replace-tables` is given. A dest sidecar keeps `notes` and the fields qbench does not write, and kept rows timed in another mode make `mode` unknown. `qbench_manifest.json` merges per table, each entry with its own `runs` (`adopted from dest` first when dest rows stayed), `bench_sha` and `exported_at`, and its runs are read before any table moves. `HOST_KINDS` holds no `decode_batch` or `frontend`, which no probe exports; `pd_proxy_overhead`'s example writes under `results/` and is profile only. kv_transfer's union sorts on `gpu, kv_heads_per_rank, layout, tokens`, then the other key columns.

2026-10-12: no fixture changed. `record_dir`'s parser takes no abbreviation, and `remote_probe` refuses a probe arg that abbreviates one of its own flags (`--out-d`, `--gpu-l`, `--ru`, `--o`). In a command line `bearer` and `authorization` hide the next word, so the argv item "Bearer sk" is redacted. An argv item holding whitespace is redacted as a command line before its name is checked, unless it is a secret `--flag=value`, so `export HF_TOKEN=sk; vllm serve m` and `OPENAI_API_KEY=sk python -m x` keep their tails. A command line is tokenized with `;`, `&`, `|` and the other shell punctuation as their own tokens and written back unquoted, so `--api-key sk;echo hi` becomes `--api-key REDACTED ; echo hi`. Redacted JSON keeps unicode (`ensure_ascii=False`); a number such as `1e5` in a JSON value that held a secret is written back as `100000.0`. A mapping value that is one secret HTTP header, or JSON in one string (object or list), is redacted, and the password in `scheme://user:password@host` is written as REDACTED in any string. `profiling/kv_transfer.py` gives its gloo group a five-minute timeout and turns SIGTERM into SystemExit(143), so `run_recorded` records the run as failed. Not redacted: a short flag (`-k sk`) is not a secret name, and free text such as `my password is hunter2` or `aws_secret_access_key = abc` is kept.

2026-10-12: no fixture changed. A dest table's repeated rows on a key all stay through a kernel_tables merge (QuettaSim's `graph/gemm/H100.csv` holds 63 rows over 38 (M, N, K) keys, which `load_gemm_table` reduces by min); a run that holds the key supersedes every one of them. A run's own table keeps its last row on a key, and a dest reparallel or kv_transfer table keeps its last row on a key too.

2026-10-12: no fixture changed. Every kernel_tables merge over a dest CSV is checked after it is built: each dest row must be in the output cell for cell or have its key re-measured by a run, and each dest column must be in the output, else the export stops naming the file. `--replace-tables` replaces only the tables that stop this way; every other table, a gemm or moe `--append` merge included, still merges.

2026-10-12: no fixture changed. A collective run's `all_reduce` rows are dropped only when a vllm_allreduce run writes the same table, so a collective re-run over a dest tp4 or tp8 table updates its `all_reduce` rows. vllm_allreduce's `all_reduce` rows replace every `all_reduce` row of their world, dest's included. An empty `op` keys as `all_reduce`, as QuettaSim's collectives loader reads it. The merged header is dest's and the runs' columns, so no `path` column is added to a table that had none, and a fresh export and one over the same tables give the same rows.

2026-10-12: no fixture changed. A merged kernel table keeps its source's format: a CSV the line ending of dest (or of the earliest run), a JSON table its indent and final newline (or none). Re-exporting the same runs, or exporting runs that hold what QuettaSim's tables hold, leaves the tables byte-identical.

2026-10-12: no fixture changed. A dest sidecar keeps a field qbench would write as null (the collectives `reduce`), and qbench's note is joined to its notes with `; `. When dest rows stay, `measured` spans both (`<first> .. <last>`) and `tool` names both, joined with ` + `.

2026-10-12: no fixture changed. `qbench_manifest.json` must be an object whose `kernel_tables` is an object, else the export stops with `malformed manifest` before any table moves; its other top-level keys stay, and it is written atomically. A table that kept dest rows lists the runs the manifest recorded for it first, then the new runs; `adopted from dest` only when there were none.

2026-10-12: no fixture changed. Redaction replaces a non-empty list or mapping under a secret name with REDACTED whole; a secret NAME=value after a comma (`A=1,HF_TOKEN=sk`) is redacted, and in a command line a quoted value (`export HF_TOKEN='sk x'`) is replaced whole. The stale-baseline warning points at `just run configs/runs/<wl>.yaml`.

2026-10-12: no fixture changed; `configs/workloads.yaml` holds every workload profile, and `tests/contracts/test_workloads.py` pins only the profiles it still names: `chat-singleturn`, `chat-multiturn-short`, `coding-singleturn`, `mooncake-trace`, `swebench-multiturn-short` and `test`. The `cut` pin truncates `swebench-multiturn-short`: `0e8b8a8de4442ecb` / `c8f75cbcdd1e78dc` (workload_hash / content digest).

2026-10-12: `bootstrap_plan.json` holds llama's weights gap, not a quettaserve block gap, because a model yaml without a `quettaserve:` block does not count as MISSING: provision logs `llama: no quettaserve: block in its model yaml; vLLM side only` and resolves the vLLM side. Exit code and every other line are unchanged.

2026-10-12: no fixture changed. `test_workloads.py` reads the Mooncake fixtures through `RealTrace.from_jsonl` and `MooncakeTrace`.

2026-10-12: no fixture changed for the docker tag. `configs/runs/llama-tp1-docker.yaml`, the README docker section and the launcher, runner and provision tests name `vllm/vllm-openai:v0.26.0`, the base `docker/Dockerfile` builds on and the tag the docker run was tested with.

2026-10-12: `config_failures.json` holds `distinct, clone, corpus or trace` in `grid_fit_unknown_prompt_mode`, because the grid's prompt modes are those four. Every other entry is unchanged.

2026-10-12: `config_failures.json` holds no `disagg_oversubscribed` or `kv_role` keys, because no adapter or run calls `schema.disagg_gpu_sets` or `schema.disagg_kv_config`. Every other key is unchanged.

2026-10-12: grid timing runs over HTTP; the golden harness serves it with `StandInServer`, which runs each posted batch on the stub engine, so every grid fixture (`grid_fit_llama.log`, `config_workloads.json`, `engine_args.json`, the compare tables) holds the same bytes as a direct run. The eleven `override` `grid_fit` records in `config_workloads.json` are present because vLLM serves `engine.pp` as `--pipeline-parallel-size` and `engine.cudagraph=false` as `--enforce-eager`, so the `QBENCH_PP=2` timing grid launches the server. `engine_args.json`'s kwargs are taken from a `mode.return_outputs=text` run, the path that builds `vllm.LLM`.

2026-10-12: `config_global.json`, `config_workloads.json`, `engine_args.json` and `config_failures.json` name the runs `llama-big` and `llama-big-fp8kv` and the grids `table_big` and `table_16k_bs64` (its one cell is ctx 16384 at batch 64); no value changed. `table_16k_bs64` sorts ahead of `table_big` in `config_global.json`'s keys.

2026-10-12: no fixture changed; qbench runs real traces only. `RandomTokenTrace` and the `random-1k` profile are gone, so their pin is too, and a workload source named `random*` or `synthetic*` that is not a file is refused.

2026-10-12: `config_global.json` changed by rename only: the provision `cache_root` is `/workspace/.qbench-cache`, not `/workspace/.xval-cache`; no measured value changed.

2026-10-12: `config_workloads.json` changed by rename only: the captured per-workload config key `xval_config` is `captured_config`; no value changed.

2026-10-12: `percentiles.json` changed by rename only: the keys `crossval_pct` and `crossval_pcts` are `compare_pct` and `compare_pcts`; no value changed.

2026-10-12: `table_criterion.txt` changed by rename only: the force hint names `QBENCH_ALLOW_METHOD_MISMATCH=1`, not `XVAL_ALLOW_METHOD_MISMATCH=1`; no value changed.

2026-10-12: `config_global.json` changed by rename only: the provision keys `qs_repo` and `qs_commit` are `quettaserve_repo` and `quettaserve_commit`; no value changed.

2026-10-12: `bootstrap_plan.json` changed by rename only: the missing-setting hints name `quettaserve_checkpoint` and `quettaserve_config`, not `qs_ckpt` and `qs_cfg`; no value changed.

2026-10-12: `baseline_llama.json` changed by rename only: the key `kv_dtype` is `kv_cache_dtype`; no value changed.

2026-10-12: `config_workloads.json` changed by rename only: the captured engine key `kv_dtype` is `kv_cache_dtype`; no value changed.

2026-10-12: `config_workloads.json` changed by rename only: the captured engine key `kv_bytes` is `kv_bytes_per_token`; no value changed.

2026-10-12: `config_failures.json` changed by rename only: the failure case `mla_without_kv_bytes` is `mla_without_kv_bytes_per_token` and its message says `kv_bytes_per_token`; no value changed.

2026-10-12: `engine_args.json` changed by rename only: the key `kv_bytes` is `kv_bytes_per_token`; no value changed.

2026-10-12: `grid_fit_llama.log` is `vmin_fit_llama.log` renamed, content unchanged.

2026-10-12: `config_failures.json`, `config_workloads.json` and `engine_args.json` changed by rename only: the `vmin_fit` keys (`vmin_fit_unknown_grid`, `vmin_fit_env_override`, `diff_vmin_fit_vs_trace_serve` and the rest) start `grid_fit`; no value changed.

2026-10-12: no fixture changed. `RunWriter` writes `requests.jsonl` rows on one background thread in append order, so a slow disk never blocks the event loop; a hard kill keeps every row already written, which is every row older than a few milliseconds, and a failed row write marks the run failed.
