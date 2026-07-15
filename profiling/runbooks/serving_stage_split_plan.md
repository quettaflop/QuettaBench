# Serving-level prefill stage-split — design + H100 run recipe

**Goal.** A serving-level per-stage cost breakdown of c1 prefill **TTFT** as a function of
`(new, cached)` prompt tokens, with a **reliable device-vs-host split** — the thing the offline
attempt (`prefill_stage_split.py`) could not do. Partition the wall TTFT into:

```
HTTP-recv/parse | chat-template | tokenize | enqueue/ZMQ-IPC | scheduler-admit |
model-forward (DEVICE) | framework-dispatch | sample | detokenize | response-stream
```

Artifacts:
- `profiling/gpu_profiling/vllm/serving_stage_split.py` — the instrumented live bench.
- `profiling/gpu_profiling/vllm/analyze_serving_stage_split.py` — regress each stage on `(new,cached)`.
- this doc — rationale, device-timing method, exact H100 recipe, VERIFY-ON-H100 checklist, feed-back.

---

## 1. What is already solved (do NOT re-measure)

From `prefill_stage_split_results.md` + `prefill_law_defit_trace.md` (all H100, vLLM 0.19.0,
Llama-3.1-8B, c1):

| quantity | value | source |
|---|---|---|
| client wall TTFT(new,cached) | **new 29.4 / cached 5.89 ms/1k** | `prefill_live_ttft_H100.csv` |
| tokenize | **1.33 ms/1k** | `prefill_stage_split_*` `tokenize_ms` |
| NEW GEMM roofline (device) | **0.02498 ms/tok ≈ 25 ms/1k** | `roofline_params_H100_llama31_8b.json` |
| FLOOR | **~26 ms** | `prefill_floor_llama31_8b.json` |
| GPU paged-attn cached slope (device) | **~1.5 ms/1k** | `cached_prefill_v3_H100.csv` (R²=0.9992) |
| cached top-level stack | model 2.4 / IPC 0.7 / HTTP 2.8 ms/1k | results doc §1 (live) |
| shared/perreq split | **50/50** | `live_split_probe.py` B-sweep |

The new bench **reuses these as constants** and only resolves the two open residuals:
- the **NEW dispatch residual** (~6 ms/1k = serving 31 − offline GEMM 25) → framework-dispatch stage,
- the **CACHED host residual** (~3.7 ms/1k = live 5.89 − tokenize 1.33 − paged-attn ~1.5 + IPC) → HTTP/template/IPC.

---

## 2. Why the offline host/device split failed, and the fix

`prefill_stage_split.py` computed `device_ms = Σ_k self_device_time_total(k)` and
`host = max(0, wall − device)`. Three independent failures (all banked in the results doc):

1. **Eager over-counts wall.** Σ kernel self-time > the serialized wall when launches overlap →
   `host` goes negative (new=2048: device 82 ms > wall 57 ms). The split is undefined.
2. **Hidden under CUDA graphs.** A replayed graph emits ONE graph-exec CUPTI activity, not the
   ~1000 constituent kernels → `key_averages()` self-time collapses to ~0 → "all host".
3. **EngineCore subprocess.** In v1 (multiprocessing ON) the forward runs in a separate process,
   so a main-process `torch.profiler` sees **zero** CUDA kernels. The offline workaround
   `VLLM_ENABLE_V1_MULTIPROCESSING=0` is NOT the serving topology and breaks tp>1.

**Fix — three layered lanes:**

- **LANE 0 (no patch).** Client `perf_counter` around the aiohttp SSE stream → un-perturbed wall
  TTFT. This is the trustworthy ground truth (== `live_ttft_probe.py`) and the partition target.
- **LANE A (no patch).** Per-request **Prometheus `/metrics` `_sum` delta-scrape** at c1. The
  engine owns the monotonic `queued_ts / scheduled_ts / first_token_ts`; vLLM v1 exposes them ONLY
  as histograms. At c1, scraping `_sum` before send and after the SSE first token isolates THAT
  request's value (one request advances each `_sum` once). This yields a reliable **3-way split**
  with zero patching: `wall = frontend_residual + queue_span + prefill_span`.
- **LANE B (device split, nsys-wrapped run).** `nsys -t cuda,nvtx --cuda-graph-trace=node` attaches
  to the CUDA-owning **subprocess** (fixes failure 3) and **re-expands graph nodes** (fixes failure
  2). Device time = `Σ(end−start)` over CUPTI kernel rows inside a per-step **NVTX** range emitted
  from a worker monkeypatch. `device = Σ busy` (never `> wall` pathologically; fixes failure 1).
  `framework_dispatch = prefill_span − device_kernel`. Cross-checked by `torch.cuda.Event` brackets
  (the pattern proven in `cached_prefill_steps_v3.py` — Events time the graph **replay** span, so
  they are immune to all three failures).

Why this beats the offline attempt: neither nsys nor the in-process Event hook needs
`VLLM_ENABLE_V1_MULTIPROCESSING=0`, so we characterize the **real serving topology**.

---

## 3. The 11 stages and their instrument (c1, f(new,cached))

`[CLIENT]` = client perf_counter (no patch); `[USAGE]` = OpenAI `usage` block (no patch);
`[PROM]` = `/metrics` `_sum` delta-scrape (no patch); `[HOOK]` = needs a worker/frontend monkeypatch.

| # | stage | instrument | patch? |
|---|---|---|---|
| 1 | HTTP recv/connect | `[CLIENT]` send→status; loopback ~const | no |
| 2 | body/JSON parse | folded into frontend_residual; `[HOOK]` to isolate | (yes) |
| 3 | chat-template render | folded into frontend_residual; `[HOOK]`/offline standalone to isolate | (yes) |
| 4 | tokenize | **reuse 1.33 ms/1k** (banked); inside frontend_residual | no |
| 5 | enqueue / ZMQ-IPC | **reuse 0.7 ms/1k** (banked); inside frontend_residual | (yes) |
| 6 | scheduler-admit / queue | `[PROM]` `request_queue_time` `_sum` delta; ~0 at c1 | no |
| 7 | model-forward DEVICE | `[PROM]` `request_prefill_time` WALL; device-only = `[HOOK]` LANE B nsys/Event | yes (split) |
| 8 | framework-dispatch | `prefill_span − device_kernel` (needs LANE B) | yes |
| 9 | sample (1 token, greedy) | ~const, into FLOOR | no |
| 10 | detokenize (1 token) | ~const, into frontend_residual | no |
| 11 | response-stream / SSE | `[CLIENT]`; loopback sub-ms | no |

**6 of 11 stages are measurable with zero patching** (1,4,6,7-wall,9,11 + the lumped residual).
Splitting frontend_residual into 2/3/5 and prefill_span into 7-device/8-dispatch needs the hooks
(LANE B for the device split; an optional frontend hook for the parse/template split).

---

## 4. Exact H100 run recipe

Env (per `h100_setup.md`, but **GPU 7** per the task, not 6):

```bash
ssh h100
cd /root/agentic-serve            # CLEAN cwd (no stray flash_attn.py)

export CUDA_VISIBLE_DEVICES=7
export TMPDIR=/data48/kevinlau/tmp
export XDG_CACHE_HOME=/data48/kevinlau/tmp/.cache
PYTHON=~/miniconda3/envs/vllm/bin/python
MODEL=/data48/kevinlau/models/Llama-3.1-8B-Instruct
```

### 4a. LANE 0 + LANE A (no patch) — the main run

The script self-launches the OpenAI server (same bench config: prefix-cache + chunked-prefill,
gpu-mem 0.9, stats ON), health-polls `/health`, then sweeps `(new × cached)` at c1 with a
fresh tail per trial and a `/metrics` scrape around each request.

```bash
$PYTHON profiling/gpu_profiling/vllm/serving_stage_split.py \
  --model "$MODEL" --port 8771 \
  --news 8,128,512,1024,2048 --cacheds 0,2000,8000,16000 --trials 5 \
  --out profile_data/results/serving_stage_split_H100.csv
```

Writes one row per `(new,cached)` with columns:
`ttft_ms, client_connect_ms, queue_span_ms, prefill_span_ms, frontend_residual_ms, e2e_span_ms,
ttft_prom_ms, prompt_tokens_actual`.

**Block-hash attribution (optional second run):** re-run with `--hash builtin`; the
`frontend_residual.cached` delta vs `--hash sha256` is the SHA-256 prefix-cache block-hash cost
(the external-lit lead). Write to a `_builtin` CSV and diff.

### 4b. LANE B (device split) — separate nsys-wrapped run

Print the exact command + the worker hook, then run it by hand (kept separate so the profiled wall
does not contaminate the Lane-0/A walls):

```bash
$PYTHON profiling/gpu_profiling/vllm/serving_stage_split.py --emit-nsys-cmd --port 8771
```

That prints (paths are **VERIFY-ON-H100**):

```bash
nsys profile -t cuda,nvtx,cublas --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o serving_stage_split_nsys -f true -- \
  $PYTHON -m vllm.entrypoints.openai.api_server \
    --model $MODEL --served-model-name llama --host 127.0.0.1 --port 8771 \
    --dtype bfloat16 --gpu-memory-utilization 0.90 --max-model-len 32768 \
    --enable-prefix-caching --enable-chunked-prefill --api-key test \
    --no-enable-log-requests --worker-extension-cls my_ext.NvtxForwardTimer
```

Worker hook (`my_ext.NvtxForwardTimer`, dropped into a module on `PYTHONPATH`) brackets the model
forward INSIDE the EngineCore subprocess with an NVTX range + a `torch.cuda.Event` pair (read off
the hot path). Skeleton printed by `--emit-nsys-cmd`. Drive the nsys server with the SAME bench
client (`--no-launch`), export `.nsys-rep` → `.sqlite`, then compute per-step device time:

```
device_kernel_ms(window) = SUM(k.end - k.start)
  over CUPTI_ACTIVITY_KIND_KERNEL rows whose [start,end] fall inside the 'vllm_prefill_step'
  NVTX range (NVTX_EVENTS table),
  bucketed GEMM/attn/kv_write/elementwise/sampling via the existing classifier
  (profiling/process/_legacy/extract_nsys_prefill_breakdown.py -- reuse classify() + the
   CUPTI_ACTIVITY_KIND_KERNEL ⋈ StringIds query; ADD the NVTX-range timestamp join).
```

Emit a small `serving_stage_device_H100.csv` (`new,cached,device_kernel_ms`) for the analyzer.

### 4c. Analyze

```bash
$PYTHON profiling/gpu_profiling/vllm/analyze_serving_stage_split.py \
  --csv profile_data/results/serving_stage_split_H100.csv \
  --device-csv profile_data/results/serving_stage_device_H100.csv   # optional (LANE B)
```

Prints the SERVING-LEVEL COST BREAKDOWN table (FLOOR + ms/1k-new + ms/1k-cached per stage), the
de-fit reconciliation gates, and (if the device CSV is present) the `device vs framework_dispatch`
split of the NEW prefill slope.

---

## 5. VERIFY-ON-H100 checklist (vLLM internals I cannot introspect here; vary across versions)

1. **`RequestOutput.metrics is None`** on the v1 OpenAI-server path in 0.19.0 (v0 `RequestMetrics`
   removed; PR #24947 re-attaches `RequestStateStats` but is likely post-0.19.0 and offline-AsyncLLM
   only). **Do NOT build any instrument on `output.metrics`.** The bench uses Prometheus instead.
2. **Prometheus metric names** exist and populate in 0.19.0:
   `vllm:request_queue_time_seconds`, `vllm:request_prefill_time_seconds`,
   `vllm:e2e_request_latency_seconds`, `vllm:time_to_first_token_seconds`,
   `vllm:request_prompt_tokens`. Confirm `prefill_time` / `queue_time` specifically (newer than
   TTFT). If a name differs, the matching column is left blank and `frontend_residual` still works
   (it only needs queue + prefill; if those are missing, fall back to the LANE 0 wall + LANE B).
3. **Each histogram has a `*_sum`** cumulative counter and at c1 it advances by exactly one request
   between two scrapes (single model label series — one model loaded).
4. **Stats ON by default** (server launched WITHOUT `--disable-log-stats`). The launch in this
   script does not pass it; confirm `/metrics` returns the histograms at startup (the script prints
   which are present).
5. **`model_forward_time` / `model_execute_time` are OTel-only** (need `--otlp-traces-endpoint`),
   NOT on the default `/metrics`. So the device span is hook-only (LANE B), not a free metric.
6. **Frontend vs EngineCore are separate processes** in v1 0.19.0; the monotonic timestamps live in
   EngineCore; an in-frontend profiler cannot see GPU kernels.
7. **LANE B hook paths** (only if running the device split): the worker forward entrypoint
   (`vllm.v1.worker.gpu_worker.Worker.execute_model` OR
   `vllm.v1.worker.gpu_model_runner.GPUModelRunner.execute_model`), that `--worker-extension-cls` is
   the supported in-subprocess injection on 0.19.0, that `torch.cuda.nvtx` ranges land in
   `NVTX_EVENTS` joinable to `CUPTI_ACTIVITY_KIND_KERNEL` by timestamp, that
   `--cuda-graph-trace=node` actually expands the replayed prefill graph into per-kernel rows, and
   that `vllm.v1.core.sched.scheduler.Scheduler.schedule` still exposes `num_scheduled_tokens`
   (reused from `cached_prefill_steps_v3.py`) for per-step `(new,cached)` tagging.
8. **`--prefix-caching-hash-algo`** is still a settable server flag (the offline script guards it
   with try/except). If rejected, drop the block-hash toggle.
9. **`X-Request-Id`** header is honored by the chat endpoint (propagation key for a future per-request
   Lane-B join). If unsupported it is ignored — the c1 1:1 step↔request map still holds.
10. **Reproduce the banked rates** new 29.4 / cached 5.89 ms/1k on `ttft_ms` before trusting any new
    per-stage split (the analyzer prints this gate).

---

## 6. How the outputs feed back into the de-fit

- **`PREFILL_NEW_DISPATCH_RESIDUAL`** ← `framework_dispatch.new` (LANE B device split). Without
  LANE B, `prefill_span.new − 25 (roofline)` is an upper bound. Expect ~6 ms/1k.
- **`PREFILL_HOST_SHARED` / `PREFILL_HOST_PERREQ`** ← `frontend_residual.cached` is the c1 **SUM**
  (~3.7 ms/1k of HTTP-parse + chat-template + IPC + detok). c1 cannot identify the shared/perreq
  split (the two regressors are the identical `cached` column at 1 req/step) — the split stays at
  the **50/50** from the live concurrency B-sweep (`live_split_probe.py`). This bench measures the
  SUM and attributes it to the frontend host stack (not model physics), confirming the 2.8 HTTP +
  0.7 IPC decomposition.
- **Already-de-fitted constants** (tokenize 1.33, IPC 0.7, GEMM 25, paged-attn 1.5) are reused: they
  live INSIDE `frontend_residual` (tokenize, IPC) and `prefill_span` (GEMM, paged-attn). The bench's
  job is the residual attribution, not re-deriving them.
- **Validation gate:** `ttft_ms` slopes must reproduce 29.4 / 5.89; the three LANE A spans must sum
  back to the wall (exact by construction for `frontend_residual`); `queue_span ≈ 0` at c1.
