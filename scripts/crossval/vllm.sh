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

N_DEV="$(python3 "$HERE/xval_config.py" devices "$CRATE")"  # tp * pp
G="${CUDA_VISIBLE_DEVICES:-$("$HERE/free_gpu.sh" "$N_DEV")}"
echo "GPU=$G"

OUT="$BASELINE_DIR/vllm-$CRATE.json"
[ -n "$GRID" ] && OUT="$BASELINE_DIR/vllm-$CRATE-$GRID.json"

mkdir -p "$BASELINE_DIR"
RAW="$(mktemp /tmp/vllm-raw-XXXXXX.txt)"
cleanup() { rm -f "$RAW"; }
trap cleanup EXIT

# ALLOW_UNVERIFIED=1 runs an uncertified workload (deepseek bring-up).
EXTRA=()
[ "${ALLOW_UNVERIFIED:-0}" = "1" ] && EXTRA+=(--allow-unverified)

# Same link profile and collective env as bench.sh: caller env wins over yaml;
# empty values are never exported.
export XVAL_LINK_PROFILE="${XVAL_LINK_PROFILE:-$(python3 "$HERE/xval_config.py" link-profile)}"
while IFS='=' read -r _k _v; do
  if [ -n "${!_k:-}" ]; then _v="${!_k}"; fi
  if [ -n "$_v" ]; then export "$_k=$_v"; fi
done < <(python3 "$HERE/xval_config.py" collective)

CUDA_VISIBLE_DEVICES="$G" "$PY" "$HERE/vmin_fit.py" "$CRATE" "$GRID" --model "$MDIR" \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  | grep -E "^META|^RESULT|^SKIP|^CAPACITY" | tee "$RAW"

python3 "$HERE/cache.py" "$RAW" "$OUT"
