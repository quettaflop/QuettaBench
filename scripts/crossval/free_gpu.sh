#!/usr/bin/env bash
# Print N comma-separated idle device indices. NVIDIA polls idle GPUs via
# nvidia-smi; other backends honor a caller-set CUDA_VISIBLE_DEVICES or default to
# 0..N-1, since per-device idle polling is nvidia-specific.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
n="${1:-1}"
if [ "$(python3 "$HERE/xval_config.py" backend)" = cuda ]; then
  mapfile -t ids < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
    | awk -F', ' '$2+0<1000 { print $1 }')
  if [ "${#ids[@]}" -lt "$n" ]; then
    echo "NEED $n FREE GPUS, have: ${ids[*]:-none}" >&2
    exit 1
  fi
  out=""
  for ((i = 0; i < n; i++)); do
    [ -n "$out" ] && out+=","
    out+="${ids[i]}"
  done
  echo "$out"
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "$CUDA_VISIBLE_DEVICES"
else
  seq -s, 0 "$((n - 1))"
fi
