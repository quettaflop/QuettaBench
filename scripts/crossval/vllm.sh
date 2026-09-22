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

TP="$(python3 "$HERE/xval_config.py" tp "$CRATE")"
G="${CUDA_VISIBLE_DEVICES:-$("$HERE/free_gpu.sh" "$TP")}"
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

# Link profile picks the collective env (same rule as bench.sh). Override with XVAL_LINK_PROFILE.
if [ -z "${XVAL_LINK_PROFILE:-}" ]; then
  # Capture then match; piping to grep -q would SIGPIPE nvidia-smi and pipefail misreads it.
  _topo="$(nvidia-smi topo -m 2>/dev/null || true)"
  case "$_topo" in
    *NV[0-9]*) XVAL_LINK_PROFILE=nvlink ;;
    *)         XVAL_LINK_PROFILE=pcie ;;
  esac
fi
export XVAL_LINK_PROFILE

# Same collective env as bench.sh; empty yaml values stay unset.
while IFS='=' read -r _k _v; do
  case "$_k" in
    NCCL_ALGO)          _YAML_NCCL_ALGO="$_v" ;;
    NCCL_PROTO)         _YAML_NCCL_PROTO="$_v" ;;
    NCCL_P2P_LEVEL)     _YAML_NCCL_P2P_LEVEL="$_v" ;;
    NCCL_IB_DISABLE)    _YAML_NCCL_IB_DISABLE="$_v" ;;
    NCCL_SOCKET_IFNAME) _YAML_NCCL_SOCKET_IFNAME="$_v" ;;
  esac
done < <(python3 "$HERE/xval_config.py" collective)
NCCL_ALGO="${NCCL_ALGO:-$_YAML_NCCL_ALGO}"
NCCL_PROTO="${NCCL_PROTO:-$_YAML_NCCL_PROTO}"
NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-$_YAML_NCCL_P2P_LEVEL}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-$_YAML_NCCL_IB_DISABLE}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$_YAML_NCCL_SOCKET_IFNAME}"
for _var in NCCL_ALGO NCCL_PROTO NCCL_P2P_LEVEL NCCL_IB_DISABLE NCCL_SOCKET_IFNAME; do
  if [ -n "${!_var}" ]; then export "$_var"; fi
done

CUDA_VISIBLE_DEVICES="$G" "$PY" "$HERE/vmin_fit.py" "$CRATE" "$GRID" --model "$MDIR" \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  | grep -E "^META|^RESULT|^SKIP|^CAPACITY" | tee "$RAW"

python3 "$HERE/cache.py" "$RAW" "$OUT"
