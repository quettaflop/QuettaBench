#!/usr/bin/env bash
# Capture a vLLM slope-fit baseline into scripts/crossval/baselines/.
# Usage: scripts/crossval/vllm.sh <crate> [grid] <model-dir> [python]
# The workload's own grid writes vllm-<crate>.json; a named grid writes
# vllm-<crate>-<grid>.json so scratch grids never clobber the real baseline.
# Respects a caller-set CUDA_VISIBLE_DEVICES, else picks idle GPUs itself.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE="${1:?crate}"
GRID="${2:-}"
MDIR="${3:?model dir}"
PY="${4:-python3}"
BASELINE_DIR="$HERE/baselines"

TP="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["workloads"][sys.argv[2]].get("tp", 1))' \
  "$HERE/workloads.json" "$CRATE")"
G="${CUDA_VISIBLE_DEVICES:-$("$HERE/free_gpu.sh" "$TP")}"
echo "GPU=$G"

OUT="$BASELINE_DIR/vllm-$CRATE.json"
[ -n "$GRID" ] && OUT="$BASELINE_DIR/vllm-$CRATE-$GRID.json"

mkdir -p "$BASELINE_DIR"
RAW="$(mktemp /tmp/vllm-raw-XXXXXX.txt)"
cleanup() { rm -f "$RAW"; }
trap cleanup EXIT

# Bring-up only: ALLOW_UNVERIFIED=1 runs a workload whose verified flag is
# false, so a new engine (deepseek) can be measured before it is certified.
EXTRA=()
[ "${ALLOW_UNVERIFIED:-0}" = "1" ] && EXTRA+=(--allow-unverified)

CUDA_VISIBLE_DEVICES="$G" "$PY" "$HERE/vmin_fit.py" "$CRATE" "$GRID" --model "$MDIR" \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  | grep -E "^META|^RESULT|^SKIP|^CAPACITY" | tee "$RAW"

python3 "$HERE/cache.py" "$RAW" "$OUT"
