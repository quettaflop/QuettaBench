# Config: matrix sweeps and overrides

How a run yaml's `matrix:` expands into runs, and how `--set` and the other layers override a value.

## Matrix

A `matrix:` makes one run per point of its axes' product; `just sweep` is `just run`.
An axis is a dotted path with a list of choices, as in `mode.concurrency: [8, 32]`.
Paths joined by `+` are one axis that moves together: each choice lists one value per path, as in `mode.concurrency+engine.extra.vllm.max_num_seqs: [[8, 64], [32, 256]]`.
Each point gets its own `results/<run>/`, named by the run name and the point's values.

## Overrides

Precedence, highest first: `--set` > env > run yaml > model yaml > hardware yaml > schema defaults.
Env reaches the config only through `config/env_map.py`, the only reader of `os.environ`.
A `--set` naming a matrix axis replaces its choices, so `--set 'mode.concurrency+engine.extra.vllm.max_num_seqs=[[32, 256]]'` runs that point alone.
`engine.extra.<engine>.<flag>` sets one flag in that engine's extra block and keeps the rest.
A whole `engine.extra.<engine>` from a higher layer replaces the block.
Unknown keys are refused.
`just dry-run` prints every value with its origin.
