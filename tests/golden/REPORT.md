# Golden fixtures: what is pinned, and what the current code says

Stage 1 of the structural refactor. Every file here was captured from the code
at the parent of the commit that adds this directory; no production file was
edited. Run `python3 -m pytest tests/golden -q`. `QBENCH_GOLDEN_REGEN=1`
rewrites the fixtures from the code (only for an intended behaviour change).

Three STOP findings below (items 2, 4, 9). Each is recorded, not reconciled:
the fixtures pin the current divergent behaviour on both sides.

## How the GPU-only code runs on CPU

- `tests/_stubs/vllm/` records every `LLM(...)` and `AsyncEngineArgs(...)`
  kwargs dict. `LLM.generate` advances a fake clock instead of running;
  `AsyncLLMEngine.from_engine_args` raises `StubHalt` so `trace_serve.main()`
  stops right after building its engine args.
- `tests/_stubs/torch/` exists because the grid run reads the GPU name and
  memory from torch (`InProcessDriver.device`) and torch is not installed. It
  reports one 48 GiB GPU named `STUB-GPU-48GiB`.
- A grid run (`serve.main --grid`, which replaced `vmin_fit.py`; the vmin_fit
  fixtures below were captured from that script and are compared unchanged)
  runs end to end on the stubs with `drivers.time` swapped for the fake
  clock. The cost model (`tests/golden/_harness.py`) is an exact
  line in max_tokens, so r2 = 1 and the fit stops after one attempt, except
  cell (1024, 16), whose middle step point is bent so every attempt fits
  r2 = 0.9605 and `VMIN_ATTEMPTS` is used up. The generate() call count is
  pinned, which is how `VMIN_ATTEMPTS` is observed.
- `shutil.which` returns None during resolution so backend and link profile
  never depend on the host (no nvidia-smi: backend `cpu`, link profile
  `nvlink`). `os.environ` is replaced, not merged, for every capture.
- Stubs are on `sys.path` only inside the capture context and are dropped
  from `sys.modules` afterwards, so no other test sees a fake vllm or torch.

### vLLM version the stub targets

None is pinned for the crossval path. `provision.vllm_spec` is empty in the
`xval_config.py` defaults and in `xval.yaml`; `requirements-probe.txt` says
`vllm>=0.6` (profiling probes, not crossval); `scripts/crossval/AGENTS.md`
says `vLLM >= 0.29` serves GLM-5.3-Flash; profiling notes mention 0.19.0,
0.19.1 and 0.27.1. vmin_fit writes the runtime version into `META vllm=`.
The stub reports `0.0.0+stub` and implements only the names the scripts
import: `vllm.LLM`, `LLM.generate(prompts, sp, use_tqdm=)`,
`LLM.get_tokenizer`, `vllm.SamplingParams`, `vllm.__version__`,
`vllm.engine.arg_utils.AsyncEngineArgs`,
`vllm.engine.async_llm_engine.AsyncLLMEngine.from_engine_args`. Whether these
import paths and kwargs exist in a given vLLM release was not checked (no
vllm installed, no pin to check against).

## What is pinned, per item

| # | test | fixtures | pins |
|---|---|---|---|
| 1 | `golden/test_loader.py` | `config_global.json`, `config_workloads.json`, `config_failures.json`, `config_trace_serve_style_error.json` | for all 11 workloads, env cleared and env overridden (`XVAL_LINK_PROFILE=pcie XVAL_PP=2 XVAL_BACKEND=cuda GPU_UTIL=0.85 PROMPT_MODE=clone VMIN_ATTEMPTS=2 XVAL_EVAL_POS=25`): merged workload record, grid cells, serving_style parse, `xval_config.py` cli output (`tp devices cells bench`), and vmin_fit's full stdout (META, CAPACITY, SKIP, RESULT, DONE) plus generate() count. Globally: grids, step_points, run_params, provision, backend, link profile, collective env per profile, cli output. Failures: type + message |
| 2 | `golden/test_engines.py` | `engine_args.json` | vmin_fit `LLM` kwargs per workload (cleared env, override env, `--nograph`), trace_serve `AsyncEngineArgs` kwargs per workload with xval.sh's argv, and the per-key diff |
| 3 | `golden/test_workloads.py` | `hash_trace.jsonl`, `workload_hash.json` | loaded requests, per-request digests, hash; equal under completion reorder; differs on one altered token, on injected `poisson:2.0` arrivals, and on reordered file lines |
| 4 | `golden/test_metrics_core.py` | `percentiles.json` | `trace_serve._pct`, `_pcts` and `metrics._percentile` on empty, 1, 2, ties, 7, arange(1000) at p50/p90/p95/p99 |
| 5 | `golden/test_metrics_core.py` | `aggregate_summary.json` | `aggregate()` summary dict and per-request tpot over normal, single-token, wall-clock-tpot, failed-with-partial-ttft, timeout and held-out requests |
| 6 | `golden/test_crossval.py` | `vmin_fit_llama.log`, `baseline_llama.json` | the log is byte-equal to what vmin_fit prints (vllm.sh filter); the baseline json cache.py writes, byte for byte (clock and git sha frozen) |
| 7 | `golden/test_crossval.py` | `bench_loop.log`, `bench_criterion.log`, `table_loop.txt`, `table_criterion.txt` | exact table.py stdout for both logs against item 6's baseline |
| 8 | `golden/test_provision.py` | `bootstrap_plan.json` | `XVAL_DRYRUN=1` plan text, exit code, and that no file is left behind, for a run with gaps and a complete run |
| 9 | `golden/test_workloads.py` | `mooncake_5.jsonl`, `mooncake_readers.json` | the one reader (`workloads/live.py`) through `RealTrace.from_jsonl` and `MooncakeTrace`, with and without `hash_ids` and with a zero `output_length` (resolved, see item 9 below) |
| 10 | this file | | SPECSUM/SERVESUM/WORKLOADSUM formats, below |

## Where the logs came from

`<backup-dir>/v4bench_backup_20260928` was searched read-only. It holds only
`vllm bench serve` result JSONs (`*_bench.json`), `sweep_results.jsonl` and
the sweep harness: no `META`/`RESULT` vmin_fit log and no `LOOP` or
`[batch_bench]` bench log. Nothing was copied from it.

- `vmin_fit_llama.log`: produced by the real `vmin_fit.py` print statements
  for workload `llama` on the stubs (48 GiB stub GPU, fake clock), filtered
  with vllm.sh's `^META|^RESULT|^SKIP|^CAPACITY`. 14 RESULT lines and one
  SKIP come from vmin_fit; the 15th RESULT line, for the SKIPped cell
  (4096, 64), is hand-written in the pre-normalization format (no
  `ms_at_eval`, `eval_pos`, `attn_share`) that cache.py still accepts, so the
  baseline holds one RAW point. Timings are synthetic, not measurements.
- `bench_loop.log`, `bench_criterion.log`: synthesized in the line formats
  table.py parses (`LOOP ...`, `[batch_bench] ...`, criterion
  `group/param time: [lo mid hi]`), with bench.sh's `>>>` header lines.

## STOP finding, item 2: vmin_fit and trace_serve build different engine args

For every workload the two kwargs dicts differ. Cells are
`vmin_fit` / `trace_serve` (as xval.sh calls it: `<trace> --model <dir> --tp <tp>`).

| workload | disable_log_stats | dtype | enable_expert_parallel | kv_cache_dtype | max_model_len | pipeline_parallel_size | quantization |
|---|---|---|---|---|---|---|---|
| deepseek | `true` / absent | same | `true` / `false` | `"fp8_ds_mla"` / absent | `33280` / `40960` | absent / `1` | same |
| glm53 | `true` / absent | same | `true` / `false` | same | `33280` / `40960` | absent / `1` | `"fp8"` / absent |
| glm53flash | `true` / absent | same | `true` / `false` | same | `33280` / `40960` | absent / `1` | `"fp8"` / absent |
| llama | `true` / absent | `"float16"` / `"bfloat16"` | absent / `false` | same | `9216` / `40960` | absent / `1` | same |
| llama-h200 | `true` / absent | `"float16"` / `"bfloat16"` | absent / `false` | same | `17408` / `40960` | absent / `1` | same |
| llama-h200-fp8kv | `true` / absent | same | absent / `false` | `"fp8"` / absent | `17408` / `40960` | absent / `1` | same |
| llama-tp1 | `true` / absent | `"float16"` / `"bfloat16"` | absent / `false` | same | `9216` / `40960` | absent / `1` | same |
| llama-tp2 | `true` / absent | `"float16"` / `"bfloat16"` | absent / `false` | same | `9216` / `40960` | absent / `1` | same |
| llama-tp4 | `true` / absent | `"float16"` / `"bfloat16"` | absent / `false` | same | `9216` / `40960` | absent / `1` | same |
| qwen | `true` / absent | same | absent / `false` | same | `33280` / `40960` | absent / `1` | same |
| qwen3 | `true` / absent | same | absent / `false` | same | same | absent / `1` | same |

Same on both sides for every workload: `model`, `tensor_parallel_size`,
`gpu_memory_utilization` (0.9), `enable_prefix_caching` (False),
`trust_remote_code` (True).

Root of the split: vmin_fit reads the workload record (dtype, maxlen,
kv_dtype, quant, expert_parallel; `XVAL_PP`/`GPU_UTIL` from env), trace_serve
reads only its own argparse defaults plus `--tp`. These are semantic
differences, not spelling: llama replays in bf16 against an fp16 baseline;
deepseek, glm53 and glm53flash replay without expert parallel; deepseek and
llama-h200-fp8kv replay with the default KV dtype instead of the workload's;
glm53/glm53flash replay without `quantization="fp8"`. `XVAL_SERVE_ARGS` can
add `--ep`, `--dtype`, `--max-model-len`, `--gpu-util`, but nothing derives
them from the workload. `pipeline_parallel_size=1` vs absent is only a
spelling difference.

Resolved by the owner: one builder, workload record wins.

- Both scripts now take their kwargs from `quettabench/engines/vllm.py`
  `translate(cfg).engine_kwargs` and build the engine through
  `drivers.InProcessDriver`. vmin_fit fills an `EngineCfg` from the workload
  record it already reads; trace_serve takes `--workload <name>` (xval.sh
  passes it) and loads `configs/runs/<name>.yaml`, with explicit engine
  flags overriding it.
- `engine_args.json` regenerated. trace_serve side: equal to vmin_fit's for
  10 workloads, so every `diff_vmin_fit_vs_trace_serve` is `{}`. qwen's
  trace_serve side is the loader's refusal (`attention=hybrid needs
  kv_bytes`), recorded as an exit message; its diff is `null`.
- vmin_fit side: one key added, `pipeline_parallel_size: 1` wherever pp is
  1 (the owner picked "always present" for this spelling difference). No
  other vmin_fit value changed.

## STOP finding, item 4: the two percentile implementations disagree

`trace_serve._pct(sorted_vals, q)` returns `sorted_vals[min(n - 1, int(q * n))]`
(nearest rank, floor; the caller must pass sorted values; `_pcts` maps it over
p50/p90/p95/p99). `metrics._percentile(data, p)` sorts, then interpolates
linearly at index `p / 100 * (n - 1)`. Arrays: `seven` is
[1, 3, 5, 7, 9, 11, 13] (fed unsorted to `_percentile`), `ties` is
[1, 2, 2, 2, 3], `two` is [10, 20], `single` is [42], `arange_1000` is
0.0 .. 999.0.

| array | q | crossval `_pct` | metrics `_percentile` | |
|---|---|---|---|---|
| empty | p50, p90, p95, p99 | `IndexError: list index out of range` | 0.0 | DIFFER |
| single | p50, p90, p95, p99 | 42.0 | 42.0 | agree |
| two | p50 | 20.0 | 15.0 | DIFFER |
| two | p90 | 20.0 | 19.0 | DIFFER |
| two | p95 | 20.0 | 19.5 | DIFFER |
| two | p99 | 20.0 | 19.900000000000002 | DIFFER |
| ties | p50 | 2.0 | 2.0 | agree |
| ties | p90 | 3.0 | 2.6 | DIFFER |
| ties | p95 | 3.0 | 2.8 | DIFFER |
| ties | p99 | 3.0 | 2.96 | DIFFER |
| seven | p50 | 7.0 | 7.0 | agree |
| seven | p90 | 13.0 | 11.8 | DIFFER |
| seven | p95 | 13.0 | 12.399999999999999 | DIFFER |
| seven | p99 | 13.0 | 12.879999999999999 | DIFFER |
| arange_1000 | p50 | 500.0 | 499.5 | DIFFER |
| arange_1000 | p90 | 900.0 | 899.1 | DIFFER |
| arange_1000 | p95 | 950.0 | 949.05 | DIFFER |
| arange_1000 | p99 | 990.0 | 989.01 | DIFFER |

Resolved by the owner: one `metrics.core.percentile(values, p, method)`,
method `"nearest"` or `"linear"` exactly as in the two columns above, named
by every caller. The one change is the empty row for `"nearest"`: it now
raises `ValueError("nearest-rank percentile of no values")` instead of
`IndexError`, and the fixture records that; every number is unchanged.

Also: `aggregate()` reports `median_*` with `statistics.median`, not
`_percentile(.., 50)`; the two coincide on every input here. trace_serve's
`summarize` raises on zero records before `_pct` can see an empty list, and
an empty tpot list is skipped by its `if tpot` guard.

## STOP finding, item 9: the two Mooncake readers differ

Same file, `mooncake_5.jsonl` (5 lines: `timestamp` in ms, `input_length`,
`output_length`, `hash_ids`).

| | `src/workloads/mooncake.py` | `synth_bench.load_trace` (crossval) |
|---|---|---|
| output shape | `BenchmarkRequest(messages=[], max_tokens, metadata={prompt_token_ids, request_index, num_hash_blocks})` | dict `{prompt_token_ids, prompt_len, output_length, order, arrival_ts}` |
| token ids | each `hash_id` expands to 512 ids from `random.Random(f"{seed}:{hash_id}")` in [1000, 20000), tail filled per request, truncated to `input_length`; seed defaults to 42 | `hash_ids` ignored; `synth_ids(input_length, line_index)`, an arithmetic stream in [1, 99999] |
| prefix sharing | kept: requests 1 and 3 share request 0's first 512-id block (bounded by request 0's 700 tokens) | lost: no request shares a block with request 0 |
| arrival | `timestamp` ignored; pacing comes from the runner's own load mode | `arrival_ts = timestamp / 1e3` seconds, absolute (trace_serve rebases to the first arrival and divides by `--speed`) |
| ordering | file order; `get_next_request` wraps around after the last line (the 6th call returns request 0) | file order; line index stored as `order`, a hash input |
| missing `hash_ids` | `KeyError: 'hash_ids'` | accepted, same output as with them |
| zero / falsy `output_length` | kept as `int` | dropped (`if d.get("output_length")`) |
| `limit` | supported | none (trace_serve `--limit` slices after loading) |
| workload hash | none | `6336408d62c729aa` |

Lengths and `max_tokens` / `output_length` agree on every line.

Resolved by the owner: one reader, `quettabench/workloads/live.py`.
`RealTrace.from_jsonl` and the runner's `MooncakeTrace` both go through it. Token ids: `hash_ids` expand into
shared 512-id blocks exactly as `src/workloads/mooncake.py` did (seed 42,
tail fill, truncation; the ids are byte-identical to that column above), so
prefix sharing is preserved; a line without `hash_ids` falls back to
`synth_ids(input_length, line index)`, the crossval column above.
`arrival_ts = timestamp / 1e3` always. A zero `output_length` is kept.
`MooncakeTrace` takes `limit`. `request_index` and the tail-fill index are
the file line index (the old src reader counted non-blank lines; the two
differ only on a file with blank lines).

Consequences, all intended: the workload hash of a crossval Mooncake replay
that carries `hash_ids` changes (`mooncake_5.jsonl`: `6336408d62c729aa` ->
`c009ec57c5c23d36`), and so does one with a zero `output_length`
(`5215d22affa91623` for the zero-output variant). Without `hash_ids` the
hash is unchanged (`6336408d62c729aa`). The runner's dataset no longer
raises `KeyError` on a line without `hash_ids`. Only
`mooncake_readers.json` was regenerated; `workload_hash.json` has no
Mooncake entry and is unchanged.

## Item 10: trace_serve summary line formats (reference for the port)

From `scripts/crossval/trace_serve.py`, format strings as written there:

```
META trace={args.trace} reqs={len(reqs)} engine={args.engine} tp={args.tp} pp={args.pp} ep={int(args.ep)} speed={args.speed} arrival={args.arrival} serving_style={args.serving_style} closed_loop={args.closed_loop or 0} ttft_basis={'slot' if args.closed_loop else 'arrival'} prefix_caching={int(args.prefix_caching)} source={sources} text_mode={text_mode}
SERVE id={i} session={req.get('session_id', '-')} in={req['prompt_len']} out={n_tok} ttft_ms={ttft_ms:.1f} tpot_ms={tp}        (tp = f"{tpot_ms:.3f}" or "-")
WORKLOADSUM expected={expected} replayed={replayed} match={int(match)} n={len(done)}/{len(reqs)}
WORKLOADVOID replayed workload differs from the trace; comparison is void
WORKLOADVOID replayed workload differs from the trace; sweep is void        (goodput sweep; its WORKLOADSUM prints match=0)
SERVESUM n={n} ttft_ms_p50={:.1f} ttft_ms_p90={:.1f} ttft_ms_p95={:.1f} ttft_ms_p99={:.1f} tpot_ms_p50={:.3f} tpot_ms_p90={:.3f} tpot_ms_p95={:.3f} tpot_ms_p99={:.3f} req_s={:.3f} tok_s={:.1f}
SERVESUM n={n} ttft_ms_p50=... ttft_ms_p99=... tpot_ms_p50=- req_s={:.3f} tok_s={:.1f}        (no request produced 2+ tokens)
SLASUM verdict={PASS|FAIL} bound={ttft_p99,tpot_p99|none} ttft_p99={:.1f}/{sla_ttft_ms or '-'} tpot_p99={:.3f}/{sla_tpot_ms or '-'} rate_req_s={:.3f}
BURSTSUM peak_ttft_p99={peak:.1f} recovery_s={na|:.1f} target_ms={args.sla_ttft_ms}
SPECSUM spec=measured accept_len={:.2f} accept_rate={:.3f} n={len(acc)}
SPECSUM spec=off note=needs-real-text-trace
ENERGYSUM joules={joules:.0f} joules_per_token={:.3f} scope=gpu_board devices={n|all} samples={len(samples)}
ACCSUM dataset=gsm8k exact_match={:.3f} n={n}
COSTSUM usd={usd:.4f} usd_per_m_out_tokens={per_m:.2f} gpu_cost_hr={args.gpu_cost_hr} devices={devices} wall_s={wall:.1f}
GOODPROBE speed={speed:g} verdict={PASS|FAIL} bound={...|none} req_s={:.3f}
GOODPUTSUM max_req_s={:.3f} speed={:g} bound={fail_bound or 'none-within-range'} iterations={iters}
GOODPUTSUM max_req_s=none bound={fail_bound or 'na'} iterations={iters} note=no-passing-speed-within-budget
NOTE all requests submit at t=0 (no arrival_ts in trace): ttft is queue-dominated; use --closed-loop N or --arrival poisson:<rate> for a comparable first-token number
```

The SERVESUM percentiles are `_pcts` (nearest rank, item 4). Exit codes:
3 on a workload mismatch, 1 when goodput finds no passing speed.

2026-10-09: `legacy/trace_serve.py` is deleted. Where each line above prints
from now:

- WORKLOADSUM, WORKLOADVOID, SERVESUM, SLASUM, BURSTSUM, SPECSUM, GOODPUTSUM:
  `quettabench/metrics/results.py` `legacy_lines` / `print_legacy_lines`,
  called by `modes/serve.py` at the end of a serve run.
- GOODPROBE: `metrics/results.py` `goodprobe_line`, printed per probe by
  `modes/serve.py`'s goodput sweep.
- ENERGYSUM, ACCSUM, COSTSUM: `quettabench/profiler.py` (`qbench profile`),
  same formats and order.
- META (serve run), SERVE per request, NOTE: no printer. The run's settings
  are run.json's config and each request is a requests.jsonl row.
  Exit codes are unchanged (`modes/serve.py`: 3 on a workload mismatch, 1
  when goodput finds no passing speed).

## Item 8: bootstrap.sh dry run

Runs on CPU. It needs bash, python3 with PyYAML (for `xval_config.py`), and a
`$PY -m pip show vllm` that succeeds, which the test supplies as a two-line
`/bin/sh` stand-in. Found while capturing: on a dry run bootstrap.sh still
truncates `.xval_env.tmp` next to itself and removes it at exit, so the test
runs it from a scratch copy of `scripts/crossval`; nothing is left behind
(pinned as `files_left_behind: []`). A workload with no `bench` section
(llama) prints `workload llama has no bench config` from `xval_config.py`
on stderr and then `MISSING:`.

## What the current code contradicts in QBENCH_STRUCTURE.md

Section 5 and the 2.2/2.3 contracts it relies on:

1. Section 5 STOP list: three of the four conditions hold today
   (percentiles differ, Mooncake readers differ, vmin_fit and trace_serve
   build different engine args). The fourth (BenchDriver vs bench.sh rows)
   has no BenchDriver yet.
2. "Every engine flag verified against the pinned engine version": there is
   no pinned vLLM version for crossval (see above).
3. Sections 2.2/2.3 call `workload_hash` a "sha256 hex" over "sorted
   per-request digests". Today it is the first 16 hex chars of sha256 over
   the request count plus the sorted per-request sha256 digests. The digest
   covers `seed`, `order` (the file line index), `arrival_ts` to 6 places,
   `prompt_len`, the ids, `output_length`, `session_id`, `turn`. So the same
   requests written in a different line order hash differently
   (`file_lines_reordered` in `workload_hash.json`); only completion order is
   free.
4. The brief expects these to fail at parse time; today they do not:
   - illegal EP degree: `xval_config.check_ep_legal` raises correctly but has
     no production caller, and no workload carries `num_experts` or an EP
     degree.
   - missing `kv_bytes` on an MLA or hybrid stack: no error. vmin_fit falls
     back to the dense per-head formula after loading the model. On the
     DeepSeek-V3 shaped probe config that gives 3,747,840 B/token
     (`config_failures.json` `mla_without_kv_bytes`), against an MLA latent
     of roughly 61 x (512 + 64) x 2 = 70,272.
   - `verified: false`: enforced only by vmin_fit (`sys.exit` before the
     model load, after its import has already set collective env vars).
     trace_serve has no verified gate.
   - malformed `serving_style`: raises `ValueError` at parse, as expected,
     but `serving_style` is not a workloads.json key and vmin_fit never
     prints `META serving_style`, so every cache.py baseline records
     `aggregated`.
5. Section 2.2 crossval rules: today a baseline older than 14 days only
   prints `WARNING`, it does not withhold; an NCCL mismatch prints a line
   but does not withhold; RAW, LOW R2 and EAGER rows still get a ratio
   (flag only). `[batch_bench]` rows always carry kv `?`, so against any
   baseline they print `KV ?/<kv>` and the ratio is withheld
   (`table_loop.txt`).

Other current behaviour the fixtures now pin, worth knowing before the move:

- When vmin_fit's free-memory estimate goes negative, `kv_tokens=0` and the
  capacity gate is off for every cell (`if cap and ...`), so no cell is
  SKIPped: glm53 and glm53flash on the 48 GiB stub GPU.
- `XVAL_PP` changes `pipeline_parallel_size` and `META pp` only; the
  capacity estimate uses tp alone.
- glm53 and glm53flash have no `kv_dtype`, so RESULT lines say
  `kv=bfloat16` while `META quant=fp8`.
- cache.py records `nccl_algo`/`nccl_proto` as `auto` when the capture ran
  without NCCL pins (from vmin_fit's `META nccl_algo=auto`).

## Not captured on CPU

- Real timings: vmin_fit's fit loop runs on a fake clock; RESULT numbers are
  synthetic. vmin_fit's `corpus` prompt mode needs a tokenizer and was not run.
- trace_serve's serving path (`_serve`, goodput sweep, energy, gsm8k): needs
  a live `AsyncLLMEngine`. Only its engine args, its serving_style parse
  failure, and its line formats (item 10) are captured.
- bench.sh, vllm.sh, xval.sh, oneshot.sh, prof.sh: need a GPU, cargo or the
  engine binaries. Their python inputs (`xval_config.py` cli) are pinned.
- greedy_agreement.py, logit_agreement.py: need vLLM and transformer models.

## Fixture changes

2026-10-11: `config_failures.json` and `config_trace_serve_style_error.json`
now read `use aggregated, disagg:<P>p<D>d, afd:<A>a<F>f, or ep<N>dp<M>`.
`parse_serving_style` accepted `afd:<A>a<F>f` but its refusal did not list
it; only the message text changed, every spec is refused as before.

2026-10-11: no fixture changed. A serve run whose workload profile sets
`prefix_caching_required: true` now gets `engine.prefix_caching = true`
(origin `profile`), as the retired sweeps passed `--enable-prefix-caching`;
before, the profile key was never read and such runs launched with prefix
caching off. An explicit `engine.prefix_caching=false` on such a profile is
refused at load.

2026-10-11: no fixture changed. The serve identity gate hashes every
dispatched request, failed or not (`serve.served`), in a single run and in
each goodput-sweep probe. 4318bf0 had hashed only successful single-turn
requests, so one failed request failed the run with exit 3 and
WORKLOADVOID; such a run is `partial` again, and below
`mode.min_success_rate` it aborts with exit 1. WORKLOADVOID now means the
requests sent differ from the trace: a request changed after hashing, or a
multi-turn session whose failed turn left its later turns unsent. Chat
requests without token ids still hash their messages (4318bf0).

2026-10-11: no fixture changed; the workload_hash pins in
`tests/test_profile_hashes.py` did. Since 4318bf0 a chat request without
token ids hashes a canonical-JSON digest of its messages (assistant content
left out); before, it hashed only seed, order, arrival, lengths, session and
turn, so different chat profiles could share a hash (`chat-medium`,
`chat-singleturn` and `chat-singleturn-synth` all hashed
`2c2c330fe937283f`). Token-id requests hash as before (the `mooncake-trace`
pin did not move), and the content digests did not move. A run recorded
before 4318bf0 does not join on workload_hash with one recorded after it
for these profiles (`cut` is the test's own truncated trajectory):

| profile | before 4318bf0 | after |
|---|---|---|
| `chat-medium` | `2c2c330fe937283f` | `1b84966cb9c4facc` |
| `chat-multiturn-long` | `274162becf17c102` | `3101d70c80d8606c` |
| `chat-multiturn-medium` | `489bcc4b486495ac` | `26a991f2b7529daf` |
| `chat-multiturn-short` | `57c82245406bcdf1` | `c51f134d3946f53d` |
| `chat-short` | `e0b677fed285920c` | `3d6dd927203af7fa` |
| `chat-singleturn` | `2c2c330fe937283f` | `1b84966cb9c4facc` |
| `chat-singleturn-synth` | `2c2c330fe937283f` | `1b84966cb9c4facc` |
| `coding-singleturn` | `03481570266bb2d4` | `15384d696a4d7a1b` |
| `decode-heavy` | `f0719d3492f4fd06` | `adc88f3eae2adc96` |
| `fixed-seq128` | `05a41401be097be3` | `3439077f38252544` |
| `osworld-multiturn-long` | `f741501038aeb62c` | `762312f6b602cb88` |
| `osworld-multiturn-medium` | `5af8ab8b94a75466` | `29482f55adf89f20` |
| `osworld-multiturn-short` | `5fa7da0de14a64fb` | `ab09ce8e07b841c8` |
| `prefill-heavy` | `8c28ae1fb55fecea` | `0964edcd447c2121` |
| `random-1k` | `fdaa0aa7451df053` | `00740b698acbf0f7` |
| `random-1k-random-doublewrap` | `fdaa0aa7451df053` | `633909c6f1f06765` |
| `random-1k-random-legacy` | `fdaa0aa7451df053` | `1635e8cfbd8dcba8` |
| `swebench-multiturn-long` | `b94b837574393d16` | `60b6ad3b16fac67e` |
| `swebench-multiturn-medium` | `8cb100a908e8d018` | `f02e03a9e1282fc2` |
| `swebench-multiturn-short` | `d4dca931e18de175` | `9f0aa9d6f98e5bd3` |
| `terminalbench-multiturn-long` | `0c2fa9aec3717f8c` | `716641f80b9ffb4e` |
| `terminalbench-multiturn-medium` | `80bf8305e3891345` | `6bc7cd20245181c3` |
| `terminalbench-multiturn-short` | `a28027e756a8a273` | `2ab1e354437af5f0` |
| `test` | `c19e2bb3db6d4434` | `7ae706bdf8eeae1d` |
| `cut` | `c769b8ce9e3a8fe2` | `a7ffbc60a718ff0d` |

2026-10-11: no fixture changed; the `mooncake-trace` pin in
`tests/test_profile_hashes.py` moved in 969c7f7 from `0e6f083e59f1f9dd` /
`d483cd1411833245` (workload_hash / content digest) to `52bae3bbfe646fd8` /
`a1384dbf00844c6c`. `MooncakeTrace` dropped each record's timestamp, so
its requests carried no `arrival_ts` and `workload.arrival=trace`
submitted them all at once; its requests now carry `arrival_ts`, the
record's timestamp in seconds. A mooncake-trace run recorded before
969c7f7 does not join on workload_hash with one recorded after it.

2026-10-11: no fixture changed; the `mooncake-trace` pin in
`tests/test_profile_hashes.py` moved from `52bae3bbfe646fd8` /
`a1384dbf00844c6c` (workload_hash / content digest) to `a03be7b0ed268d81`
/ `b5ed59cd7a0d7aff`. A Mooncake draw that wraps past the file's end
shifted each lap k by k * (span + mean gap) of the file's timestamps;
before, every lap repeated the first lap's arrival times, so
`workload.arrival=trace` fired the wrapped requests together with the
first lap's.

2026-10-11: no fixture changed. Since d7287c5 the live-server probes start
vLLM through `probes.base.launch_vllm` (the vllm adapter, then
`launcher.start`), so each launch differs from the probe's own launch
before it, pinned in `tests/probes/test_probe_record.py`:

- all three: the server is `<python's dir>/vllm serve <model>` instead of
  `<python> -m vllm.entrypoints.openai.api_server --model <model>`; its env
  gains `VLLM_SERVER_DEV_MODE=1` (mounts /reset_prefix_cache) and
  `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` (KV is sized after graph
  capture, so the KV pool the server reports can differ from before); the
  argv gains `--trust-remote-code`; flags come in the adapter's order.
- `pool_capacity`: no other flag changed (it already passed
  `--enable-chunked-prefill`).
- `serving_stage_split`: no other flag changed (it already passed
  `--enable-chunked-prefill`).
- `serving_decode_grid`: also gains `--enable-chunked-prefill`, which it
  did not pass before. Whether that changes its scheduling depends on the
  vLLM build's default for chunked prefill; no run has measured it.

2026-10-11: no fixture changed. A run with an SLA whose requests produced
no TPOT sample (none decoded 2+ tokens) records `sla.tpot_p99` as null and
prints `SLASUM ... tpot_p99=-/<sla>`; before, it recorded 0 and printed
`tpot_p99=0.000`, a number nothing measured.

2026-10-11: no fixture changed. Since 4383be8 `profiling/serving_frontend.py`
takes its frontend-off reference from QuettaSim's
`engine.sim.closed_loop.predict_cell` (the vLLM-scheduler replay) instead of
`engine.sim.queue_sim.predict_cell` (the queue sim), which QuettaSim main
removed. `mult_curve` is the measured TTFT residual over that reference,
so a `mult_curve` exported to a device yaml before 4383be8 and one exported
after it are not comparable; no run has measured how far they differ.

2026-10-11: no fixture changed. The vllm adapter passes `engine.extra.vllm.api_key`
to the server as the `VLLM_API_KEY` env var instead of `--api-key`, so vLLM's
startup args log no longer prints it; the three live-server probes' launches
pinned in `tests/probes/test_probe_record.py` lose `--api-key <key>` and gain
that env var. A run's HttpDriver now sends that key; before, it always sent
`test`, which a server started with another key refused.

2026-10-11: no fixture changed. `run_name` must match
`[A-Za-z0-9][A-Za-z0-9._-]*` and be none of `probes`, `xval`, `baselines`;
load refuses any other name, since it names `results/<run_name>/`, which
`--overwrite` empties. A matrix suffix has every other character replaced
by `_` (`engine.model: /w/llama` gives `-model_w_llama`, before
`-model/w/llama`). `--overwrite` now empties only a real dir directly under
the results dir that holds `run.json`, `requests.jsonl` or `engine.json`,
and a run dir whose `run.json` says failed is replaced without it.

2026-10-11: no fixture changed. Each probe run started from its CLI
(`run_recorded`) writes its tables and `run.json` into its own
`results/probes/<gpu>/<probe>/<run-id>/`, the run id being its UTC start
(`20261011T120000.000000Z`); before, every run of a probe shared
`results/probes/<gpu>/<probe>/`, so one `run.json` stood for tables of
several runs. A probe that upserts (gemm and moe `--append`, flash_attn,
cross_attn, reparallel, and gemm's `fp8_act_quant` entry) now starts from
the same table of its newest earlier run whose `run.json` is ok, where
before it started from whatever the last run, ok or not, left. A
`qbench run` probe (`mode: probe`) still writes `results/<run>/<probe>/`.
`remote-probe` empties the remote out dir, sends the probe's local runs up,
and pulls back without overwriting local files.

2026-10-11: no fixture changed. A probe run from its CLI opens
`requests.jsonl` in its run dir before the probe starts, so a run killed
before it finishes leaves no `run.json` and export skips its tables; before,
a killed run left tables with no record at all, and export kept them. A
probe that names its output file (`serving_stage_split --out`,
`serving_decode_grid --out-raw`) refuses a dir that already records a run;
before, the second run's `run.json` replaced the first's.

2026-10-11: no fixture changed. Since the probe move, `vllm_allreduce` wrote
its `all_reduce` rows to its own `vllm_allreduce/.../ncu/collectives/{gpu}_tp{N}.csv`
and the collective probe its rows to `collective/.../ncu/collectives/`, so
no table held both, where before the move `vllm_allreduce` upserted its
rows into the collective probe's table in QuettaSim's `kernel_data/`. The
new `kernel_tables` export builder merges them back into one
`kernel_data/ncu/collectives/{gpu}_tp{N}.csv` by the old rule: the
collective probe's rows other than `all_reduce` (`dtype_bytes` 2 and
`path` nccl when absent), then `vllm_allreduce`'s `all_reduce` rows. Unlike
the old upsert, the merge does not depend on which probe ran last: a
collective run after a `vllm_allreduce` run used to replace the vLLM rows
with NCCL ones. Likewise gemm's `fp8_act_quant` entry and the elementwise
probe's entries merge into one `elementwise/{gpu}.json`, where an
elementwise run after a gemm fp8 run used to drop `fp8_act_quant`.

2026-10-11: no fixture changed. `just mirror-up`, `mirror-down` and
`mirror-preview` sync `results/probes/` with the R2 prefix
`profiling/qbench_probes/`, fixed in the justfile, instead of
`profiling/kernel_data/` (overridable by `R2_KERNEL_PREFIX`), which is
QuettaSim's kernel data. `mirror-up-prune`, an `s3 sync --delete` onto that
prefix, is gone.

2026-10-11: no fixture changed. `engine.name` accepts only the engines with
an adapter (`vllm`, `quettaserve`, `reference`); `sglang` and `trtllm`,
which the schema listed but no adapter served, are refused at load with
the list of those three, where before they failed later in the runner.

2026-10-11: no fixture changed. A probe run writes only the rows it
measured; `probes/base.py` `prior_table`, which seeded an upserting probe's
table from its newest earlier ok run, is gone, and `remote-probe` no longer
sends earlier runs up. Two runs that overlapped in time lost the rows of
the one that had not seeded from the other. The `kernel_tables` export now
unions the rows of every ok run of a probe per table, keyed as the old
upsert was (flash_attn and cross_attn on the head config, gemm on M, N, K,
moe on its five geometry columns, reparallel on its five key columns,
gemm's `fp8_act_quant` on its JSON key), and on a key conflict the run with
the latest `provenance.finished` wins; a gemm or moe run without `--append`
drops the rows of runs that finished before it, as its whole-file write
did. A table with no row key comes from the run with the latest
`provenance.finished`, where before it came from the newest run by start.
Results dirs given to export are pooled by that rule; before, an earlier
dir won over a later one.

2026-10-11: no fixture changed. `just remote-probe` runs
`probes/base.py` `remote_probe`: each invocation makes its own remote
`mktemp -d /tmp/qbench-probes.XXXXXX`, passes `--run-id` from the local
clock, always pulls the run back (a failed run's too) and exits with the
remote run's exit code, then removes only that dir; a failed pull keeps
it. Before, every invocation emptied and reused `/tmp/qbench-probes/`, so
two concurrent invocations deleted each other's runs, a failed remote run
was never pulled, and the run id came from the remote clock.

2026-10-11: no fixture changed. `qbench export` takes the kernel tables'
destination as its own `--kernel-dest` (QuettaSim's data root), and the
`kernel_tables` builder runs only when it is given; the other builders keep
`--dest` and run only when that is given. Before, every builder wrote under
the one `--dest`, so a default export put `kernel_data/` beside `kernels/`
and `results/`, and the examples named `../QuettaSim/engine/data`, a dir
QuettaSim does not have.

2026-10-11: no fixture changed. QuettaSim's justfile recipes `collectives`,
`cross-attn`, `vllm-allreduce`, `reparallel-probe` and `reparallel-probe-val`
call `QuettaBench/profiling/kernel_composed/...` paths this refactor deleted
and fail as they stand. README.md "QuettaSim recipes that call the old probe
paths" gives each one's replacement here plus the `kernel_tables` export;
QuettaSim itself is unchanged.

2026-10-11: no fixture changed. `redact` treats every string value that
holds a flag (a `-` at its start or after whitespace) as a command line,
and every argv item holding whitespace too, so `--api-key X` at the start
of probe_args, `-H 'Authorization: Bearer X'`, a newline before the flag
and `sh -c "... --api-key X"` are written REDACTED. `passphrase` is a
secret word and ssh, encryption and signing name a secret key. Before,
only a value holding ` --` was redacted as a command line.

2026-10-11: no fixture changed. `kernel_tables` builds every table in a
temp dir under `--kernel-dest` and checks each row first: a row with an
extra or missing cell, an empty key cell or a key that does not parse
(gemm `M` of `1.0` included, as the old upsert's `int()` read it), JSON
that does not parse, or a union JSON that is no object stops the export
with `<rel>: <run dir> row N malformed` (or `malformed JSON`) and the
dest files as they were; then each file replaces its dest file with
`os.replace`. Before, a bad row raised halfway and left dest part
written. Key columns compare as numbers (`total_weight_gb` 16 and 16.0
are one key), and within one run's table the last row on a key wins
(geometry-keyed tables keep every row). Only gemm's elementwise `.json`
unions across runs; a `.csv` there is the latest run's.

2026-10-11: no fixture changed. Export compares `provenance.finished` as
UTC instants (`datetime.fromisoformat(...).astimezone(timezone.utc)`); a
missing or unparsable time sorts oldest. Before, the strings were
compared, so `01:00+01:00` sorted after `00:30+00:00`.

2026-10-11: no fixture changed. The kernel tables' manifest is
`<kernel-dest>/kernel_data/qbench_manifest.json`; before, it was
`manifest.json` at QuettaSim's data root.

2026-10-11: no fixture changed. A unioned CSV's columns come in the order
the runs first name them, oldest run first. The old flash_attn upsert put
the newest run's columns first, then the older runs' extra ones; a table
whose runs name different columns comes out in a different column order.

2026-10-11: no fixture changed. `clear_run_dir` unlinks `run.json`,
`requests.jsonl` and `engine.json` in the `.clearing-*` dir before its
rmtree, and export's run scan skips any `.clearing-*` dir. Before, an
interrupted rmtree left a `.clearing-*` dir export read as an ok run.

2026-10-11: no fixture changed. `remote_probe` makes the remote dir and
its `out/` in one ssh (`mktemp -d ... && mkdir -p $d/out && echo $d`),
taking the last output line that is such a dir, so a probe that wrote
nothing pulls nothing and its exit code is returned; ssh gets
`-o ConnectTimeout=30`; a `--host` starting with `-` and `--out`,
`--out-dir` or `--run-id` among the probe args are refused; a pull that
fails or raises says the remote dir is kept, and a failed pull's exit
names the probe's exit code. After the pull, the run's
`provenance.finished` is set from the local clock (the remote one kept
as `provenance.remote_finished`), so export orders remote and local
runs on one clock. `just remote-probe` passes its args through `"$@"`
(`set positional-arguments`) after `--`, so an arg holding a space, `$`
or `"` reaches the probe whole; before, the args were joined into one
quoted string and split again.

2026-10-11: no fixture changed. The gemm and moe probes take no
abbreviated flags, so `--app` is refused instead of appending while
export reads it as a reset. `serving_stage_split --emit-nsys-cmd`
records no run.json; before, it wrote an ok run with no table.

2026-10-11: no fixture changed. The `frontend` and `pd_host` profiler
sources (`serving_frontend.py`, `pd_host.py`) take the GPU's QuettaSim
`device_spec/<gpu>.yaml` as a required `--device-yaml`; `just
serving-profile` passes `$QUETTASIM/device_spec/<gpu>.yaml`. Before,
both loaded `device_spec/<gpu>.yaml` beside this repo, a file that does
not exist, so neither could run. serving_frontend imports
`engine.sim.closed_loop` and `kernel_composed.types` at the top, as
QuettaSim main (d8503b8) has both.

2026-10-12: no fixture changed. `kernel_tables` writes QuettaSim main's
(d8503b8) `kernel_data/graph/` and `kernel_data/eager/` instead of
`ncu/` and `cuda_event/`, a probe run's `ncu/` going to `graph/` and
`cuda_event/` to `eager/`, and writes beside each table the
`<table>.meta.json` sidecar QuettaSim's `tools/kernel_manifest.py check`
requires: `mode` from the probe, its flags and the kind (`graph_replay`,
`ncu` with `--ncu`, `amortized_wallclock` for collectives, `reference`
for `*_nccl_vs_vllm.csv`, `host_wallclock` for host-timed kinds, else
`eager`; `unknown` when the runs kept were timed two ways), `tool`,
`reduce`, `gpu` and `measured`. Before, QuettaSim read neither dir. A
stopped export removes the `--kernel-dest` it made; no
`qbench_manifest.json` is written when no table was found; an empty CSV
and unparsable `probe_args` stop the export naming the run instead of
raising a traceback; `_inputs.files` and `find` skip paths under a
`.clearing-*` dir, as `run_dirs` does.

2026-10-12: no fixture changed. This supersedes the 2026-10-11 entry's
last clause: a pulled remote run's `provenance.finished` is no longer
rewritten (and no `remote_finished` is added); `remote_probe` stamps
`provenance.pulled` from the local clock through `metrics/results.py`
`stamp_pulled`, and export orders a run by `pulled` when present, else
`finished`. A pulled `run.json` that does not parse or whose provenance
is not an object is warned about, naming the run, and the remote dir is
still removed; before, it raised and the remote dir was kept. Every
probe parser takes no abbreviated flag (`add_output_args`, and
`serving_stage_split` and `serving_decode_grid` themselves), and
`remote_probe` refuses `--gpu-label` and `--gpu` among the probe args.

2026-10-12: no fixture changed. In a command line held as one string,
only a flag item hides the next word, so free text such as `Please -
pass the salt` or `You are a pass-through bot` is kept whole; before, a
bare secret word there (`pass`) replaced the next word with REDACTED. A
mapping value with no flag but a secret `NAME=value` (`HF_TOKEN=sk vllm
serve m`, `export HF_TOKEN=sk; vllm serve`) is redacted as a command
line, rejoined by `shlex.join`, so the `;` stuck to the value goes with
it. An argv item that is a JSON object or list (`--hf-overrides
'{"api_key": "sk"}'`) is parsed, redacted and dumped again, and kept as
written when it holds no secret.

2026-10-12: no fixture changed. `profiling/kv_transfer.py` records its
run like every probe: it takes `add_output_args` and runs through
`run_recorded`, rank 1 sends the rows it measured to rank 0, and rank 0
writes them to
`results/probes/<gpu>/kv_transfer/<run-id>/cuda_event/kv_transfer/<gpu>_nixl.csv`;
before, rank 1 upserted them into `<out-dir>/cuda_event/kv_transfer/`,
and the docstring pointed `--out-dir` at QuettaSim's `data/kernel_data`.
`qbench export --only kernel_tables` carries the runs to
`kernel_data/eager/kv_transfer/<gpu>_nixl.csv` with a `host_wallclock`
sidecar whose tool is `quettabench.profiling.kv_transfer`, unioned on the
old upsert key (`layout` compared as text) and sorted on it, the run that
finished last winning a key.

2026-10-12: no fixture changed. `kernel_tables` merges into the tables
`--kernel-dest` holds instead of replacing them: before, an export over
QuettaSim's data root left `graph/gemm/H200.csv` (1949 rows),
`eager/kv_transfer/H200_nixl.csv`, `graph/flash_attn/H200.csv` (ten head
configs) and `graph/collectives/H200_tp2.csv` (69 rows) with only the rows
the runs held, and dropped the sidecars' notes. A dest table is now the
oldest source on the same row key (collectives on `op, world, bytes`), its
columns join the union and a JSON table's entries join too; where its rows
would be dropped (no row key, or a gemm or moe run without `--append`) the
export stops naming the file and its row counts unless `--replace-tables`
is given. A dest sidecar keeps `notes` and the fields qbench does not
write, and kept rows timed in another mode make `mode` unknown.
`qbench_manifest.json` merges per table, each entry with its own `runs`
(`adopted from dest` first when dest rows stayed), `bench_sha` and
`exported_at`, and its runs are read before any table moves. `HOST_KINDS`
dropped `decode_batch` and `frontend`, which no probe exports; `pd_host`'s
example writes under `results/` and is profile only. kv_transfer's union
sorts on `gpu, kv_heads_per_rank, layout, tokens`, then the other key
columns.

2026-10-12: no fixture changed. Redaction and probe failure paths
tightened. `record_dir`'s parser takes no abbreviation, and `remote_probe`
refuses a probe arg that abbreviates one of its own flags (`--out-d`,
`--gpu-l`, `--ru`, `--o`). In a command line `bearer` and `authorization`
hide the next word again, so the argv item "Bearer sk" is redacted. An
argv item holding whitespace is redacted as a command line before its name
is checked, unless it is a secret `--flag=value`, so
`export HF_TOKEN=sk; vllm serve m` and `OPENAI_API_KEY=sk python -m x`
keep their tails. A command line is tokenized with `;`, `&`, `|` and the
other shell punctuation as their own tokens and written back unquoted, so
`--api-key sk;echo hi` becomes `--api-key REDACTED ; echo hi`. Redacted
JSON keeps unicode (`ensure_ascii=False`); a number such as `1e5` in a
JSON value that held a secret is written back as `100000.0`. A mapping
value that is one secret HTTP header, or JSON in one string (object or
list), is redacted, and the password in `scheme://user:password@host` is
written as REDACTED in any string. `profiling/kv_transfer.py` gives its
gloo group a five-minute timeout and turns SIGTERM into SystemExit(143), so
`run_recorded` records the run as failed. Known and left: a short flag
(`-k sk`) is not a secret name, and free text such as
`my password is hunter2` or `aws_secret_access_key = abc` is kept.

2026-10-12: no fixture changed. A dest table's repeated rows on a key all
stay through a kernel_tables merge (QuettaSim's `graph/gemm/H100.csv` holds
63 rows over 38 (M, N, K) keys, which `load_gemm_table` reduces by min); a
run that holds the key supersedes every one of them. A run's own table
still keeps its last row on a key, and a dest reparallel or kv_transfer
table, whose old upserts held a dict, keeps its last row on a key too.

2026-10-12: every kernel_tables merge over a dest CSV is checked after it
is built: each dest row must be in the output cell for cell or have its
key re-measured by a run, and each dest column must be in the output,
else the export stops naming the file. `--replace-tables` now replaces
only the tables that stop this way; every other table, a gemm or moe
`--append` merge included, still merges.

2026-10-12: a collective run's `all_reduce` rows are dropped only when a
vllm_allreduce run writes the same table, so a collective re-run over a
dest tp4 or tp8 table updates its `all_reduce` rows, as the old
collective_probe did by rewriting the file. vllm_allreduce's `all_reduce`
rows replace every `all_reduce` row of their world, dest's included, as its
old upsert did. An empty `op` keys as `all_reduce`, as QuettaSim's
collectives loader reads it. The merged header is dest's and the runs'
columns, so no `path` column is added to a table that had none, and a
fresh export and one over the same tables give the same rows.

2026-10-12: a merged kernel table keeps its source's format: a CSV the
line ending of dest (or of the oldest run), a JSON table its indent and
final newline (or none). Re-exporting the same runs, or exporting runs
that hold what QuettaSim's tables hold, leaves the tables byte-identical.

2026-10-12: a dest sidecar keeps a field qbench would write as null (the
collectives `reduce`), and qbench's note is joined to its notes with `; `.
When dest rows stay, `measured` spans both (`<old> .. <new>`) and `tool`
names both, joined with ` + `.

2026-10-12: `qbench_manifest.json` must be an object whose `kernel_tables`
is an object, else the export stops with `malformed manifest` before any
table moves; its other top-level keys stay, and it is written atomically.
A table that kept dest rows lists the runs the old manifest recorded for
it first, then the new runs; `adopted from dest` only when there were none.

2026-10-12: redaction replaces a non-empty list or mapping under a secret
name with REDACTED whole; a secret NAME=value after a comma
(`A=1,HF_TOKEN=sk`) is redacted, and in a command line a quoted value
(`export HF_TOKEN='sk x'`) is replaced whole. The stale-baseline warning
points at `just run configs/runs/<wl>.yaml` instead of the removed
`just vllm`.

2026-10-12: no fixture changed; the workload profiles moved into one `configs/workloads.yaml` and these unused profiles were deleted with their `tests/test_profile_hashes.py` pins: `chat-medium`, `chat-multiturn`, `chat-multiturn-long`, `chat-multiturn-medium`, `chat-multiturn-synth`, `chat-short`, `chat-singleturn-synth`, `claweval-multiturn-synth`, `decode-heavy`, `fixed-seq128`, `osworld-multiturn`, `osworld-multiturn-long`, `osworld-multiturn-medium`, `osworld-multiturn-mse`, `osworld-multiturn-short`, `osworld-multiturn-synth`, `prefill-heavy`, `swebench-multiturn`, `swebench-multiturn-long`, `swebench-multiturn-medium`, `swebench-multiturn-mse`, `swebench-multiturn-mse-medium`, `swebench-multiturn-mse-short`, `swebench-multiturn-synth`, `terminalbench-multiturn`, `terminalbench-multiturn-long`, `terminalbench-multiturn-medium`, `terminalbench-multiturn-mse`, `terminalbench-multiturn-mse-medium`, `terminalbench-multiturn-mse-short`, `terminalbench-multiturn-short`, `terminalbench-multiturn-synth`. The `cut` pin now truncates `swebench-multiturn-short` instead of the deleted `swebench-multiturn-medium`: `0e8b8a8de4442ecb` / `c8f75cbcdd1e78dc`, the same values that profile gives at 51b18a1.

2026-10-12: `bootstrap_plan.json` changed. A model yaml without a `quettaserve:` block no longer counts as MISSING: provision logs `llama: no quettaserve: block in its model yaml; vLLM side only` and resolves the vLLM side, so the gaps record shows llama's weights gap in place of the old block gap. Exit code and every other line are unchanged.

2026-10-12: no fixture changed. `RealTrace.from_mooncake` is deleted (only tests called it); `test_workloads.py` reads the Mooncake fixtures through `RealTrace.from_jsonl` and `MooncakeTrace`, which give the same requests.

2026-10-12: no fixture changed. `LegacyRandomTokenTrace` and `DoubleWrapRandomTokenTrace` are deleted with the runner's `random-legacy` and `random-doublewrap` datasets and their two pins above: no profile named them.
