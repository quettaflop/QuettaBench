#!/usr/bin/env bash
# Capture a vLLM slope-fit baseline into scripts/crossval/baselines/.
# Usage: scripts/crossval/vllm.sh <crate> [grid] <model-dir> [python]
# The workload's own grid writes vllm-<crate>.json; a named grid writes
# vllm-<crate>-<grid>.json so scratch grids never clobber the real baseline.
# The grid run (serve --grid) writes that file under results/baselines/
# itself, in cache.py's layout; this script copies it here.
# Respects a caller-set CUDA_VISIBLE_DEVICES, else picks idle GPUs itself.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
CRATE="${1:?crate}"
GRID="${2:-}"
MDIR="${3:?model dir}"
PY="${4:-python3}"
BASELINE_DIR="$HERE/baselines"

N_DEV="$(python3 "$HERE/../../quettabench/config/xval_config.py" devices "$CRATE")"  # tp * pp
G="${CUDA_VISIBLE_DEVICES:-$("$HERE/free_gpu.sh" "$N_DEV")}"
echo "GPU=$G"

NAME="$CRATE"
GRID_ARG=(--grid)
if [ -n "$GRID" ]; then NAME="$CRATE-$GRID"; GRID_ARG+=("$GRID"); fi
OUT="$BASELINE_DIR/vllm-$NAME.json"
mkdir -p "$BASELINE_DIR"

# ALLOW_UNVERIFIED=1 runs an uncertified workload (deepseek bring-up).
EXTRA=()
[ "${ALLOW_UNVERIFIED:-0}" = "1" ] && EXTRA+=(--allow-unverified)

# Same link profile and collective env as bench.sh: caller env wins over yaml;
# empty values are never exported.
export XVAL_LINK_PROFILE="${XVAL_LINK_PROFILE:-$(python3 "$HERE/../../quettabench/config/xval_config.py" link-profile)}"
while IFS='=' read -r _k _v; do
  if [ -n "${!_k:-}" ]; then _v="${!_k}"; fi
  if [ -n "$_v" ]; then export "$_k=$_v"; fi
done < <(python3 "$HERE/../../quettabench/config/xval_config.py" collective)

CUDA_VISIBLE_DEVICES="$G" PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" -m quettabench.modes.serve "${GRID_ARG[@]}" --workload "$CRATE" --model "$MDIR" \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  | grep -E "^META|^RESULT|^SKIP|^CAPACITY"

cp "$REPO/results/baselines/vllm-$NAME.json" "$OUT"
echo "wrote $OUT"
