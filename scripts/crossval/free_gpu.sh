#!/usr/bin/env bash
# Print N comma-separated idle GPU indices (memory.used < 1 GiB).
set -euo pipefail
n="${1:-1}"
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
