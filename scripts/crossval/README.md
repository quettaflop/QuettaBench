# crossval -- QuettaServe vs vLLM

Cross-validates a QuettaServe engine against vLLM at matched context, batch and
tensor-parallel degree. llama is certified; deepseek and qwen3 are bring-up
(`verified: false`, recipes need `ALLOW_UNVERIFIED=1`).

## Run

```bash
PY=<python-with-vllm> QS_BIN=<engine-binary> \
  ./xval.sh llama <weights-dir> [bench-log]
```

Stages, runnable on their own:

```bash
vllm.sh <workload> [grid] <weights-dir> [python]  # vLLM baseline -> baselines/vllm-<workload>.json
DS_CKPT=... DS_CFG=... bench.sh deepseek [out.log] # engine sweep (env prefix per workload)
QW_CKPT=... bench.sh qwen3 [out.log]
table.py <bench-log> <baseline.json>              # the comparison
```

Trace workloads:

```bash
synth_bench.py gen --profile swebench --n 200 --out t.jsonl
PROMPT_MODE=trace XVAL_TRACE=t.jsonl vllm.sh <workload> ...  # static grid from a trace
trace_serve.py t.jsonl --model <dir>                         # open-loop serving replay (vLLM only)
```

## What the numbers mean

Both columns time steady-state decode ms per token at a resident batch,
pipelined, synchronized only at the ends, at matched (context, batch, tp).
`table.py` withholds the ratio whenever method, KV dtype, or TP differ.

Config lives in `xval.yaml` (precedence: caller env > xval.yaml > defaults in
`xval_config.py`); `xval.example.yaml` shows every field. Field meanings,
per-model bring-up notes, NCCL link profiles, and the trace schema are in
[AGENTS.md](AGENTS.md).

## Tests

```bash
python3 -m pytest tests/test_crossval_contracts.py   # no gpu needed
```
