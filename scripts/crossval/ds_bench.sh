#!/usr/bin/env bash
# Sweep the deepseek cross-validation grid through the engine's stock
# batch_bench test and collect its output. table.py reads the [batch_bench]
# summary lines directly, so the engine needs no patch. Run from the
# QuettaServe checkout root (QS_DIR overrides):
#
#   DS_CKPT=/data/ds-0731-mp8 DS_CFG=<orig>/inference/config.json \
#     quettabench/scripts/crossval/ds_bench.sh [out.log]
#
# The full cargo output is kept in the log as evidence; one cell per
# invocation, so a failed cell costs only that cell.
#
# Hosts without a toolchain (the air-gapped GPU nodes) set DS_BENCH_BIN to a
# cross-built copy of the test binary instead: `cargo test -p deepseek --test
# batch_bench --release --no-run` emits target/release/deps/batch_bench-<hash>;
# ship that and pass its absolute path.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${DS_CKPT:?set DS_CKPT (the mp<world> checkpoint)}"
: "${DS_CFG:?set DS_CFG (the original inference config.json)}"
QS="${QS_DIR:-.}"
if [ -z "${DS_BENCH_BIN:-}" ] && [ ! -f "$QS/deepseek/Cargo.toml" ]; then
  echo "no deepseek crate under '$QS'; run from the QuettaServe root, set QS_DIR, or set DS_BENCH_BIN" >&2
  exit 1
fi
STEPS="${DS_TIMING_STEPS:-100}"
OUT="${1:-deepseek-bench.log}"

# The cells and the tensor-parallel degree come from workloads.json so the
# sweep cannot drift from the baseline's grid.
CELLS="$(python3 -c '
import json, sys
cfg = json.load(open(sys.argv[1]))
wl = cfg["workloads"]["deepseek"]
for ctx, bs in cfg["grids"][wl["grid"]]:
    print(ctx, bs)
' "$HERE/workloads.json")"
WORLD="${DS_WORLD:-$(python3 -c '
import json, sys
print(json.load(open(sys.argv[1]))["workloads"]["deepseek"]["tp"])
' "$HERE/workloads.json")}"

: > "$OUT"
while read -r ctx bs; do
  # prompt + timed steps + headroom for the eager warm-up; the bench asserts
  # prompt + steps + 8 <= max_seq.
  max_seq=$((ctx + STEPS + 512))
  echo ">>> ctx=$ctx bs=$bs world=$WORLD max_seq=$max_seq" | tee -a "$OUT" >&2
  before=$(wc -l < "$OUT")
  (
    cd "$QS" || exit 1
    export DS_CKPT DS_CFG DS_WORLD="$WORLD" DS_BATCH="$bs" \
      DS_BENCH_PROMPT="$ctx" DS_MAX_SEQ="$max_seq" DS_TIMING_STEPS="$STEPS"
    if [ -n "${DS_BENCH_BIN:-}" ]; then
      exec "$DS_BENCH_BIN" --nocapture
    fi
    exec cargo test -p deepseek --test batch_bench --release -- --nocapture
  ) 2>&1 | tee -a "$OUT" || true
  if ! tail -n +"$((before + 1))" "$OUT" | grep -q "^\[batch_bench\] "; then
    echo "FAIL ctx=$ctx bs=$bs (no batch_bench line; see $OUT)" | tee -a "$OUT" >&2
  fi
done <<< "$CELLS"
echo "wrote $OUT" >&2
