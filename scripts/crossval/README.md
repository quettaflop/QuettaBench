# crossval -- QuettaServe vs vLLM

Compares a QuettaServe engine's decode latency against a cached vLLM slope-fit
baseline, at matched context and batch. Moved here from QuettaServe so the
comparison layer is shared; QuettaServe drives it through this repo as a
submodule. The "ours" number is a `cargo bench` inside QuettaServe -- only the
vLLM baseline and the comparison live here.

Only **llama** is verified end to end. `workloads.json` carries qwen and
deepseek entries for later, but the QuettaServe recipes hard-error for any crate
other than llama until each one is validated.

## Files

| file | role |
|---|---|
| `vllm.sh` | refresh the cached vLLM baseline into `baselines/vllm-<crate>.json` |
| `vmin_fit.py` | vLLM decode slope-fit, milliseconds per step vs generated tokens (needs vllm) |
| `table.py` | print ours (a criterion bench log) against the cached baseline |
| `cache.py` | parse a raw bench log into the cached point set |
| `with_gpu.sh` | run a command on an idle GPU, which `free_gpu.sh` picks |
| `workloads.json` | per-crate model, dtype, and the (context, batch) grid |
| `greedy_agreement.py` | same prompts through the engine binary and vLLM, prints both continuations and the leading-token agreement (needs vllm) |
| `logit_agreement.py` | teacher-forced logit comparison of vLLM against the HF reference: per-position argmax agreement and true-token logprob deltas (needs vllm, transformers) |
| `slice_tf/` | teacher-forced slice verification harness ported from the deepseek branch; see `slice_tf/README.md` |

## Run

On a GPU box, from a QuettaServe checkout with this repo as a submodule. The
QuettaServe justfile wraps these; the whole comparison is one recipe:

```bash
just crossval llama
```

That refreshes the vLLM baseline if missing, runs the llama serving bench, and
prints the table. The scripts are invoked at their submodule path, for example
`QuettaBench/scripts/crossval/table.py`. `baselines/vllm-*.json` are generated
and stay untracked.

## Tests

`tests/test_crossval_contracts.py` runs without a GPU: the scripts parse, the
shell helpers keep their executable bit, every workload names a grid that
exists and fits its maxlen, and tensor-parallel workload names match their tp
field. Tensor-parallel baselines use the `llama-tp2` / `llama-tp4` workloads
(grid `llama_single`, the engine bench's tp cells); the table pairs tp rows
only with tp baselines. Run stages and pass criteria: `VERIFICATION.md`.
