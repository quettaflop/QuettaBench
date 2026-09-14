#!/usr/bin/env bash
# Run a command on an idle GPU. Usage: with_gpu.sh [-n N] cmd [args...]
set -euo pipefail
n=1
if [ "${1:-}" = "-n" ]; then
  n="$2"
  shift 2
fi
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
G="$("$here/free_gpu.sh" "$n")"
echo "GPU=$G" >&2
exec env CUDA_VISIBLE_DEVICES="$G" "$@"
