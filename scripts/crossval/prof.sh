#!/usr/bin/env bash
# prof.sh <name> <cmd...>: nsys-capture <cmd> into profiles/<name>.nsys-rep.
# Dumps the kernel summary next to the report so the numbers read without a
# GUI. Runs the command bare when nsys is not installed.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="${1:?profile name}"; shift
OUT="$HERE/profiles/$NAME"
mkdir -p "$HERE/profiles"

if ! command -v nsys >/dev/null 2>&1; then
    echo "prof.sh: nsys not installed; running unprofiled" >&2
    exec "$@"
fi

nsys profile -t cuda,nvtx -o "$OUT" --force-overwrite=true "$@"
nsys stats --report cuda_gpu_kern_sum "$OUT.nsys-rep" > "$OUT.kern.txt" 2>&1 || true
echo "profile: $OUT.nsys-rep  kernels: $OUT.kern.txt"
