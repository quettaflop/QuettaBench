# Session replay: the full contract

The README has the short version. This file is the reference: what each field means and when a number is withheld.

## Measuring a running server

`hardware.launcher: attach` measures the server already answering on `engine.port`. qbench checks it is healthy and serves the model, sending the server's own key (`engine.extra.vllm.api_key`) on those calls as it does on every request, then starts and stops nothing, so qbench and another harness can measure the same server back to back. `engine.json` records what the server serves, its `/version`, and the launch spec qbench would have used; the server's real argv and environment are the operator's to record.

## Trace files

- Each turn's prompt is the session's whole context so far, with the `prompt_tokens` and `osl_tokens` the serving engine reported when the session was recorded. A turn without the counts refuses the file; nothing is estimated.
- `just fetch` pulls the files in `data/MANIFEST.json` (sha256, source run, agent, model, date, token quantiles) and refuses one whose digest differs. `data/build_trajectories.py` rebuilds them from the recorded runs. A run on a file whose digest is not the manifest's needs `--allow-unverified` and is marked not comparable; a file the manifest does not list runs without the flag but is not comparable either, and `compare` withholds every ratio for both. The manifest is the `MANIFEST.json` beside the file, so a file in another directory (the Mooncake trace under `data/mooncake/`, or any path given directly) is not comparable until a manifest beside it lists it; run.json names the manifest the check read.

## Agents and sessions

- `mode.concurrency` is the number of agents. An agent holds its slot from its session's first turn to its last, waits included, so the server never has more than `concurrency` sessions open and no request waits in a client queue. This is TensorRT-LLM's trace replay and GenAI-Perf's `--session-concurrency`.
- `workload.use_recorded_waits` sleeps the recorded tool and human time after each turn. A session the builder marked `recorded_waits: false` is dropped before sampling and counted in `sessions_dropped_no_waits`.
- First starts are 10 ms apart, or less so that every agent starts within 2 s (`sections.client.first_start_stagger_s`).
- A failed turn ends its session. The turns after it are not sent, are listed in `workload.turns_not_sent`, and count as failed (kind `not_sent`).

## The sample

- Fixed before a sweep: every eligible session, or `workload.num_sessions` of a seeded shuffle, the same set at every concurrency (`sample_digest`).
- `workload.stratified` orders it round-robin across strata of turn count and prompt size, so the sessions open at any moment mix like the sample (`summary.trace.strata`).
- A session is cut, never clipped, at the first turn that does not fit `isl_tokens` beside its output or whose output passes `osl_tokens`. The last turn kept says `truncated_by_context_limit`.
- A file with fewer sessions than asked, or fewer than `mode.concurrency`, is refused.

## Recycled replay (`mode.duration_s`)

- Each agent draws the next session from a cycle over the sample, plays it to the end, and draws again until the deadline. Every play started before the deadline runs to its end.
- A failed turn ends its play, and the run goes on to the deadline with the failure counted. The run ends aborted (exit 1, the reason in `run.json`) once 8 requests, or as many as there are agents, failed in a row with no success between them, whatever the failure kind: no request succeeds any more, and since a failed play returns at once, drawing on would only pile up failed plays until the deadline.
- Every play is salted so that, from its first user message on, it cannot hit the prefix cache an earlier play left; the system prompt and tools before it stay shared, as they are between real sessions. `workload.salt: text` (default) puts `[rid:<12 hex>]` before the first user message; `workload.salt: cache_salt` sends vLLM's `cache_salt` field instead, which keeps the whole prefix apart. The tag is `sha256(seed:run:session:play)`, where `run` is a fresh id per run (`sections.plays.salt_run`), so a second run on the same server does not reuse the first run's prefixes.
- `plan_hash` covers the sample, the seed and the play rule (`sections.plays.play_rule`), not how many plays started.
- Results are comparable only at the same sample, seed, duration and agent count. `compare` checks the sample and seed through `plan_hash` and the window's workload mix; the duration and agent count it does not check, so the run names carry them. A trace that fills its replies from the engine (ShareGPT) cannot be recycled.

## The steady window (`sections.steady`)

- Opens when every agent has finished its own first session (the dispatcher records which agent played each session; the first C admissions are not that, since a fast agent can play again before the last agent has started), so the synchronized cold-cache first sessions stay out. Closes at the deadline for a recycled run, else at the first session finish after the last admission. A completion at the opening itself was a cold first play and is not counted.
- Not valid, with no steady number and the cause in `reason`, when: it opens after the deadline; a recycled window lasts under 300 s; a per-session run has 2C sessions or fewer; an agent is idle inside it; any request failed or was not sent; or the server's cache hit share differs between the window's two halves (`steady.cache`), which means the cache was still filling or had begun evicting, so a GPU with more KV memory cannot bank an unsettled cache as steady. The halves are compared at the window's own mix of turn depths (a deeper turn caches more of its prompt by construction); the limit is 0.05 or twice the drift's own noise from resampling plays, whichever is larger (a data-parallel server hits or misses almost per request); a half with fewer than 30 plays is reported but not judged; a server that reports no cached counts is not checked (`steady.cache` is null). A whole-run number never stands in.
- The cold start is reported beside it, never dropped: `first_turn_ttft_ms` (each agent's first turn) and `ramp.ttft_ms` (every request before the window).
- `workload_mix.input_tokens_served_over_recorded` is each request's served prompt tokens over the count the recording reported; a served model with another tokenizer shows here, and no text is lost either way.
- The window's TTFT is withheld when the client's own event loop read first chunks late: p99 lag over 5 ms, or mean lag over 1% of TTFT p50.
- Output throughput counts tokens that arrived inside the window: the server's count per chunk when every chunk has one (`mode.chunk_usage`, off by default so request bodies match vllm bench serve's and AIPerf's), else one token per chunk with characters when those number the output tokens, else output tokens in proportion to each chunk's characters.
- TTFT, TPOT and ITL percentiles take requests sent at or after the start and completed by the end. `interactivity_p90_tok_s` is 1000 / p90 `tpot_ms`, AgentX's figure. `chunk_gap_ms` is AIPerf's inter-chunk latency.
- `timeline` bins output tok/s and requests in flight in 10 s bins, so the window can be seen to be steady.

## Validity (`summary.validity`)

- `throughput_valid` is false when any request failed.
- `latency_valid` is false when any request failed, when client stalls could have moved a published percentile by more than 1%, or when the client loop's mean lag is over 1% of TTFT p50. Each row records `client_stall_ms` (stall over send to last chunk) and `client_stall_ttft_ms` (send to first chunk). The run computes TTFT, TPOT and E2EL at p50, p90 and p99 as measured and with each row's stall taken off; if any pair differs by more than 1%, every latency figure is withheld and the reason names the worst percentile with both values.
- Loop lag and gc pauses are reported in `sections.client`; the stall bound and the mean-lag rule above are what withhold, a p99 lag over 5 ms is a note.
- Only a run with no failed request is `ok`. Failure kinds: `connection_error` (the server or network dropped the connection), `http_error`, `server_error`, `timeout`, `client_error` and `not_sent`; the reason names each kind with its count.

## Workload identity

- `workload.plan_hash` is frozen before anything is sent: the file's sha256, the sample, the session order, the play rule, and a digest of every planned request (messages, tools, max_tokens).
- Each request is digested again at dispatch and compared with the plan. A request that differs, was not planned or was never sent fails the run (exit 3); `workload.first_mismatch` names the first.
- Only the placeholders filled with the engine's replies are left out of both digests.

## Timing rules

- TTFT ends at the first chunk with choices; E2EL ends at the last chunk, the finish chunk included (vllm bench serve's rule). AIPerf starts at the first chunk with content, so it skips an empty role chunk.
- The usage chunk, `[DONE]` and the close come after, in `stream_closed_s`.
- `server_cache_hit_rate` is the engine's reported cached-token share (vLLM: `--enable-prompt-tokens-details`). The per-turn `cache_hit_rate` is the ceiling an ideal cache could reach, not a hit.
- Percentiles are nearest-rank: the ceil(p/100 * n)-th value.
- Every row of `requests.jsonl` carries its raw events on the run clock (`sent_at_ms`, `ttft_ms`, `itl_ms`, `stream_closed_ms`, `completed_at_ms`), so any timeline can be rebuilt.
