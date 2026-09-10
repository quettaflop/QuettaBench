#!/usr/bin/env bash
# Capture a vLLM slope-fit baseline into scripts/crossval/baselines/vllm-<crate>.json.
# Usage: scripts/crossval/vllm.sh <crate> [grid] <model-dir> [python]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CFG="$REPO/scripts/crossval/workloads.json"
CRATE="${1:?crate}"
GRID="${2:-}"
MDIR="${3:?model dir}"
PY="${4:-python3}"
BASELINE_DIR="$REPO/scripts/crossval/baselines"

wl() {
  python3 -c 'import json,sys; w=json.load(open(sys.argv[1]))["workloads"][sys.argv[2]]; print(w.get(sys.argv[3], sys.argv[4]))' \
    "$CFG" "$CRATE" "$1" "${2:-}"
}

TP="$(wl tp 1)"
DTYPE="$(wl dtype)"
GIB="$(wl weights_gib)"
ML="$(wl maxlen)"
MNAME="$(wl name)"
GRID="${GRID:-$(wl grid)}"

G="$("$REPO/scripts/crossval/free_gpu.sh" "$TP")"
echo "GPU=$G"

mkdir -p "$BASELINE_DIR"
RAW="$(mktemp /tmp/vllm-raw-XXXXXX.txt)"
cleanup() { rm -f "$RAW"; }
trap cleanup EXIT

{
  echo "META gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${G%%,*}")"
  echo "META vllm=$($PY -c 'import vllm;print(vllm.__version__)')"
  echo "META model=$MNAME"
  echo "META dtype=$DTYPE"
  echo "META grid=$GRID"
  echo "META mode=FULL"
  export MODELPATH="$MDIR" WEIGHTS_GIB="$GIB" MAXLEN="$ML" GPU_UTIL=0.90 DTYPE="$DTYPE"
  KV_DTYPE="$(wl kv_dtype)"; [ -n "$KV_DTYPE" ] && export KV_DTYPE
  KV_BYTES="$(wl kv_bytes)"; [ -n "$KV_BYTES" ] && export KV_BYTES
  CUDA_VISIBLE_DEVICES="$G" "$PY" "$REPO/scripts/crossval/vmin_fit.py" FULL "$CRATE" "$GRID" \
    | grep -E "^RESULT|^SKIP|^CAPACITY"
} | tee "$RAW"

python3 "$REPO/scripts/crossval/cache.py" "$RAW" "$BASELINE_DIR/vllm-$CRATE.json"
