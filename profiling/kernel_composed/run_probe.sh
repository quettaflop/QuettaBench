#!/usr/bin/env bash
# run_probe.sh <probe.py> [probe args...] — run a kernel probe locally or on a
# remote GPU host, writing tables into the kernel_data tree.
#
# Env:
#   REMOTE_HOST   ssh host/alias (empty = run on the local GPU)
#   GPU_ID        CUDA device index (default 0 — pick a FREE gpu on shared hosts)
#   PYTHON_BIN    python with torch + vllm (local default python3; set for remote,
#                 e.g. /home/<user>/miniconda3/envs/vllm/bin/python)
#   KDATA         output root = the kernel_data dir (required)
set -euo pipefail
PROBE="$1"; shift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="${GPU_ID:-0}"
KDATA="${KDATA:?set KDATA to the kernel_data root}"

if [[ -z "${REMOTE_HOST:-}" ]]; then
    PYTHON_BIN="${PYTHON_BIN:-python3}"
    mkdir -p "$KDATA"
    if [[ "${NPROC:-1}" -gt 1 ]]; then
        # multi-GPU probe (e.g. allreduce_probe.py): one rank per GPU via torchrun.
        # GPU_IDS is a comma-separated device list (default = GPU_ID..GPU_ID+NPROC-1).
        CUDA_VISIBLE_DEVICES="${GPU_IDS:-$GPU_ID}" "$PYTHON_BIN" -m torch.distributed.run \
            --nproc_per_node="$NPROC" "$HERE/probes/$PROBE" --out-dir "$KDATA" "$@"
    else
        CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" "$HERE/probes/$PROBE" --out-dir "$KDATA" "$@"
    fi
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
    RDIR="/tmp/qs-profiling"
    echo "[remote $REMOTE_HOST gpu$GPU_ID] $PROBE $*" >&2
    ssh -o BatchMode=yes "$REMOTE_HOST" "mkdir -p $RDIR/probes $RDIR/out"
    scp -q -o BatchMode=yes "$HERE/probes/$PROBE" "$REMOTE_HOST:$RDIR/probes/"
    # shared helpers imported by the probes (e.g. _ncu.py) must travel too
    for _h in "$HERE"/probes/_*.py; do
        [ -e "$_h" ] && scp -q -o BatchMode=yes "$_h" "$REMOTE_HOST:$RDIR/probes/"
    done
    if [[ "${NPROC:-1}" -gt 1 ]]; then
        RLAUNCH="CUDA_VISIBLE_DEVICES=${GPU_IDS:-$GPU_ID} $PYTHON_BIN -m torch.distributed.run --nproc_per_node=$NPROC probes/$PROBE"
    else
        RLAUNCH="CUDA_VISIBLE_DEVICES=$GPU_ID ${NCU_BIN:+NCU_BIN=$NCU_BIN} $PYTHON_BIN probes/$PROBE"
    fi
    ssh -o BatchMode=yes "$REMOTE_HOST" "cd $RDIR && $RLAUNCH --out-dir $RDIR/out $*"
    mkdir -p "$KDATA"
    # pull produced tables back into the local kernel_data tree (subdirs preserved)
    rsync -a -e 'ssh -o BatchMode=yes' "$REMOTE_HOST:$RDIR/out/" "$KDATA/" \
        || scp -rq -o BatchMode=yes "$REMOTE_HOST:$RDIR/out/." "$KDATA/"
    echo "[remote] tables synced into $KDATA" >&2
fi
