#!/usr/bin/env bash
# Sweep the deepseek grid through the engine's batch_bench test into one log.
# Run from a QuettaServe checkout (QS_DIR overrides); DS_BENCH_BIN points at a
# prebuilt binary when cargo is absent.
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
# DS_LAYERS would truncate the model.
unset DS_LAYERS

# Cells and tp come from workloads.json.
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
FAILED=0
while read -r ctx bs; do
  # The bench asserts prompt + steps + 8 <= max_seq.
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
  if ! tail -n +"$((before + 1))" "$OUT" | grep -q "^\[batch_bench\] .*layers=all"; then
    echo "FAIL ctx=$ctx bs=$bs (no full-model batch_bench line; see $OUT)" | tee -a "$OUT" >&2
    FAILED=$((FAILED + 1))
  fi
done <<< "$CELLS"
echo "wrote $OUT" >&2
[ "$FAILED" -eq 0 ] || { echo "$FAILED cell(s) failed" >&2; exit 1; }
