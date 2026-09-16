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

TP="$(python3 -c "
import sys; sys.path.insert(0,'$HERE')
from xval_config import workloads
print(workloads()['$CRATE'].get('tp', 1))
")"
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

# Link profile picks the collective env, same rule as ds_bench.sh: PCIe boxes
# get the tuned pins, NVLink boxes get none. Override with XVAL_LINK_PROFILE.
if [ -z "${XVAL_LINK_PROFILE:-}" ]; then
  # Capture then glob-match. Piping into `grep -q` makes grep close the pipe on
  # first match; nvidia-smi then takes SIGPIPE (141) and pipefail reads that as
  # a failure, so every box misdetects as pcie. No pipe, no SIGPIPE.
  _topo="$(nvidia-smi topo -m 2>/dev/null || true)"
  case "$_topo" in
    *NV[0-9]*) XVAL_LINK_PROFILE=nvlink ;;
    *)         XVAL_LINK_PROFILE=pcie ;;
  esac
fi
export XVAL_LINK_PROFILE

# Engine and vLLM must resolve the same collective profile; high-bs latency is
# allreduce-bound on PCIe. Read defaults from xval.yaml; caller env still
# overrides via ${VAR:-yaml_value}. Empty means "leave unset": an exported
# empty NCCL var is not the same as an absent one, so only non-empty values
# are exported.
while IFS='=' read -r _k _v; do
  case "$_k" in
    NCCL_ALGO)          _YAML_NCCL_ALGO="$_v" ;;
    NCCL_PROTO)         _YAML_NCCL_PROTO="$_v" ;;
    NCCL_P2P_LEVEL)     _YAML_NCCL_P2P_LEVEL="$_v" ;;
    NCCL_IB_DISABLE)    _YAML_NCCL_IB_DISABLE="$_v" ;;
    NCCL_SOCKET_IFNAME) _YAML_NCCL_SOCKET_IFNAME="$_v" ;;
  esac
done < <(python3 -c "
import sys; sys.path.insert(0,'$HERE')
from xval_config import collective
for k, v in collective().items():
    print(k + '=' + v)
")
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
