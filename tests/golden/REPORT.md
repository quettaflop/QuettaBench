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
- `tests/_stubs/torch/` exists because `vmin_fit.py` imports torch at module
  top and torch is not installed. It reports one 48 GiB GPU named
  `STUB-GPU-48GiB`.
- `vmin_fit.main()` runs end to end on the stubs with `vmin_fit.time` swapped
  for the fake clock. The cost model (`tests/golden/_harness.py`) is an exact
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
| 1 | `test_config_resolution.py` | `config_global.json`, `config_workloads.json`, `config_failures.json`, `config_trace_serve_style_error.json` | for all 11 workloads, env cleared and env overridden (`XVAL_LINK_PROFILE=pcie XVAL_PP=2 XVAL_BACKEND=cuda GPU_UTIL=0.85 PROMPT_MODE=clone VMIN_ATTEMPTS=2 XVAL_EVAL_POS=25`): merged workload record, grid cells, serving_style parse, `xval_config.py` cli output (`tp devices cells bench`), and vmin_fit's full stdout (META, CAPACITY, SKIP, RESULT, DONE) plus generate() count. Globally: grids, step_points, run_params, provision, backend, link profile, collective env per profile, cli output. Failures: type + message |
| 2 | `test_engine_args.py` | `engine_args.json` | vmin_fit `LLM` kwargs per workload (cleared env, override env, `--nograph`), trace_serve `AsyncEngineArgs` kwargs per workload with xval.sh's argv, and the per-key diff |
| 3 | `test_workload_hash.py` | `hash_trace.jsonl`, `workload_hash.json` | loaded requests, per-request digests, hash; equal under completion reorder; differs on one altered token, on injected `poisson:2.0` arrivals, and on reordered file lines |
| 4 | `test_percentiles.py` | `percentiles.json` | `trace_serve._pct`, `_pcts` and `metrics._percentile` on empty, 1, 2, ties, 7, arange(1000) at p50/p90/p95/p99 |
| 5 | `test_aggregate.py` | `aggregate_summary.json` | `aggregate()` summary dict and per-request tpot over normal, single-token, wall-clock-tpot, failed-with-partial-ttft, timeout and held-out requests |
| 6 | `test_cache_packing.py` | `vmin_fit_llama.log`, `baseline_llama.json` | the log is byte-equal to what vmin_fit prints (vllm.sh filter); the baseline json cache.py writes, byte for byte (clock and git sha frozen) |
| 7 | `test_table_text.py` | `bench_loop.log`, `bench_criterion.log`, `table_loop.txt`, `table_criterion.txt` | exact table.py stdout for both logs against item 6's baseline |
| 8 | `test_bootstrap_plan.py` | `bootstrap_plan.json` | `XVAL_DRYRUN=1` plan text, exit code, and that no file is left behind, for a run with gaps and a complete run |
| 9 | `test_mooncake_readers.py` | `mooncake_5.jsonl`, `mooncake_readers.json` | the one reader (`workloads/live.py`) through `RealTrace.from_jsonl`, `RealTrace.from_mooncake` and `MooncakeTraceDataset`, with and without `hash_ids` and with a zero `output_length` (resolved, see item 9 below) |
| 10 | this file | | SPECSUM/SERVESUM/WORKLOADSUM formats, below |

## Where the logs came from

`/home/kw/v4bench_backup_20260928` was searched read-only. It holds only
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
`RealTrace.from_jsonl`, `RealTrace.from_mooncake` and the runner's
`MooncakeTraceDataset` all go through it. Token ids: `hash_ids` expand into
shared 512-id blocks exactly as `src/workloads/mooncake.py` did (seed 42,
tail fill, truncation; the ids are byte-identical to that column above), so
prefix sharing is preserved; a line without `hash_ids` falls back to
`synth_ids(input_length, line index)`, the crossval column above.
`arrival_ts = timestamp / 1e3` always. A zero `output_length` is kept.
`from_mooncake` takes `limit`. `request_index` and the tail-fill index are
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
