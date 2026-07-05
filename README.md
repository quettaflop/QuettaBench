# QuettaBench

LLM inference benchmarking for production-representative workloads, across
vLLM, SGLang, and TensorRT-LLM. Real prompts, prefix caching on, and proper
multi-turn KV cache reuse measurement — ported from `agentic-serve`'s
`inference-benchmark`.

## Modes

| mode | server | client | measures |
|---|---|---|---|
| stress-test | no prefix cache | `--ignore-eos` | raw kernel throughput (InferenceX-comparable TPOT) |
| single-turn | `--enable-prefix-caching` | — | realistic single-shot latency |
| multi-turn | `--enable-prefix-caching` | — | KV reuse across growing conversations (the main mode) |

Multi-turn profiles replay measured agent-workload distributions (chat, osworld,
swebench, terminalbench) at fixed concurrency; per-turn TTFT/TPOT/E2EL come out
as JSON under `results/`.

## Quickstart

```bash
pip install -r requirements.txt
bash scripts/fetch_data.sh        # large trajectory datasets from R2 (gitignored)

# server (on the GPU box)
CUDA_VISIBLE_DEVICES=0 ./scripts/launch_server.sh single-turn

# client
./scripts/bench.sh --profile chat-multiturn-synth --concurrency 40
```

`benchmark.sh` holds the canonical run configurations (edit its CONFIG section
per server). `scripts/build_trace_distributions.py` rebuilds the profile
distribution JSONs in `data/distributions/` from the raw trajectory datasets.

## Layout

```
src/benchmark/    async runner, metrics (p50/p90/p99 TTFT/TPOT/E2EL), SSE client
src/engines/      vLLM/SGLang OpenAI-compatible + TRT-LLM endpoints
src/modes/        stress_test / single_turn / multi_turn
src/workloads/    profiles, datasets, arrival patterns, distributional replay
data/distributions/  measured per-profile workload distributions (committed)
configs/          server launch baselines
tests/            runner/workload unit tests
profiling/        GPU kernel + live-server probes and their emitters (opt-in, needs a GPU)
```

## Profiling

`profiling/` holds the measurement tools: GPU kernel probes, live-server probes,
and the emitters that turn raw probe output into curated tables. It is a peer of
the runner, not part of it. The base install above stays GPU-free; the probes
need torch and vLLM, kept out of the base install as opt-in extras:

```bash
pip install -r requirements.txt -r requirements-probe.txt   # on a GPU host
```

See `profiling/README.md` for the probe list, the raw-to-curated data flow, and
how to run a probe. The emitter tests run without a GPU: `pytest profiling/tests/`.

## Data

Canonical data lives on the `agent-bench` R2 bucket:

| prefix | contents |
|---|---|
| `data/` | raw trajectory datasets (fetched by `scripts/fetch_data.sh`) |
| `results/` | benchmark ground truth (synthetic_distributional, trace_replay, archived) |
| `json/current/` | dashboard runtime JSONs |
| `archive/` | pre-2026-05 result snapshots |

Results sync up with `aws s3 sync` against the same bucket (see
`agentic-serve` for the orchestrator this was trimmed from).
