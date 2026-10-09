# Benchmark ground-truth location

**Multi-turn benchmark ground truth lives in the CENTRAL result store, keyed by GPU — NOT in any
compute host's local git checkout.**

```
/mnt/100g/agent-bench/results/synthetic_distributional/<gpu>_<model>_tp<N>_<engine>/<profile>_conc<C>.json
```

Examples: `h100_Llama-3.1-8B_tp1_vllm/`, `a100_Llama-3.1-8B_tp1_vllm/`, plus `tp2`/`tp4`, `sglang`/`vllm`,
and many models. Each `<profile>_conc<C>.json` carries a `config` block + a `per_request[]` list
(`success, ttft_ms, tpot_ms, itl_ms, e2el_ms, input_tokens, output_tokens, …`). The orchestrator
dispatches sweeps and writes results here **regardless of any compute host's local repo state**.
`build_simulator_rows.py` reads it via `BENCH_BASE`.

## Before concluding "no data exists" for a GPU/model/TP — check the central store

```bash
ls /mnt/100g/agent-bench/results/synthetic_distributional/ | grep <gpu>
ls /mnt/100g/agent-bench/results/synthetic_distributional/<gpu>_<model>_tp<N>_<engine>/
```

**Wrong-place error to avoid (made 2026-06-02):** concluding "no A100 synth ground truth exists" because
the A100 *host's* local `~/agentic-serve` / `~/inference-benchmark` checkouts were stale (April,
pre-synth-profiles, `grep multiturn-synth` = 0). A host checkout's staleness is **irrelevant** to whether
runs happened — the data was in `/mnt/100g/.../a100_*` the whole time (31 A100 dirs, ~2,785 synth JSONs,
full conc sweep 1–500 for `a100_Llama-3.1-8B_tp1_vllm`). Don't conflate "this checkout is old" with "this
hardware has no data." Check the central store keyed by GPU, not a host checkout.

## A100×1 outcome

The A100×1 simulator config (`build_simulator_rows.py` CONFIGS) uses this existing data
(`a100_Llama-3.1-8B_tp1_vllm`, `gpu_memory_utilization=0.85` — same as the decode profiling, so the
measured KV pool 8,458 blocks is consistent) with `ground_truth=True`:

| metric | A100 real-GT MAPE |
|---|--:|
| TPOT | 25.4% |
| TTFT | 46.4% |
| E2EL | 25.1% |

First-cut: the A100 **decode grid is real** (`profile_data/results/decode_profile_A100_2026-06-02.csv`),
but the TTFT **cached-prefill grid** and the **saturated-ITL ceiling** are still H100-anchored → re-anchor
those on A100 next to bring TTFT/plateau down.

## Known gap

A100 **sglang** high-concurrency multiturn `tp2`/`tp4` jobs partially FAILED (`low_success_rate`, ~58–75%
on swebench conc 160–320) — that narrow slice is genuinely incomplete. The A100×1 **vLLM** config (above)
is unaffected.
