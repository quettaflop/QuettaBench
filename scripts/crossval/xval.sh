#!/usr/bin/env bash
# xval.sh <crate> <weights-dir> [bench-log]: run the whole cross-validation.
# Picks idle GPUs once so every stage lands on the same devices: vLLM baseline
# (cached), logit check (LOGIT=1 reruns, 0 skips), greedy check when QS_BIN is
# set, comparison table, PROF=1 nsys capture, XVAL_SERVE_TRACE serving replay.
# PY selects the python with vllm.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Pick up bootstrap.sh's resolved env (PY, caches, engine bins) when present.
# shellcheck source=/dev/null
[ -f "$HERE/.xval_env" ] && . "$HERE/.xval_env"
CRATE="${1:?crate}"
MDIR="${2:?weights dir}"
BENCH_LOG="${3:-}"
PY="${PY:-${XVAL_PY:-python3}}"

TP="$(python3 "$HERE/xval_config.py" tp "$CRATE")"
N_DEV="$(python3 "$HERE/xval_config.py" devices "$CRATE")"  # tp * pp
GPUS="${CUDA_VISIBLE_DEVICES:-$("$HERE/free_gpu.sh" "$N_DEV")}"
export CUDA_VISIBLE_DEVICES="$GPUS"
ONE="${GPUS%%,*}"
echo "GPU=$GPUS"

BASELINE="$HERE/baselines/vllm-$CRATE.json"
[ -f "$BASELINE" ] || "$HERE/vllm.sh" "$CRATE" "" "$MDIR" "$PY"

# logit_agreement uses transformers reference; skip for crates without one (deepseek).
_HAS_REFERENCE=1
[ "$CRATE" = "deepseek" ] && _HAS_REFERENCE=0

LOGIT_LOG="$HERE/baselines/logit-$CRATE.log"
if [ "$_HAS_REFERENCE" = 1 ] && [ "${LOGIT:-}" != 0 ] && \
   { [ "${LOGIT:-}" = 1 ] || [ ! -f "$LOGIT_LOG" ]; }; then
    CUDA_VISIBLE_DEVICES="$ONE" "$PY" "$HERE/logit_agreement.py" \
        --model "$MDIR" --crate "$CRATE" > "$LOGIT_LOG.tmp"
    mv "$LOGIT_LOG.tmp" "$LOGIT_LOG"
fi
if [ "$_HAS_REFERENCE" = 1 ] && [ -f "$LOGIT_LOG" ]; then
    grep -E "SUMMARY|DISAGREE|NEAR_TIE" "$LOGIT_LOG"
    # Fidelity verdict beside the perf numbers: a fast-but-wrong run must be
    # flagged. Near-ties are the precision floor; only hard disagreements fail.
    _hard=$(grep -oE 'hard_disagreements=[0-9]+' "$LOGIT_LOG" | grep -oE '[0-9]+' | head -1)
    echo "FIDELITY verdict=$([ "${_hard:-1}" = 0 ] && echo PASS || echo FAIL) hard_disagreements=${_hard:-?} crate=$CRATE"
elif [ "$_HAS_REFERENCE" = 0 ]; then
    echo "FIDELITY verdict=SKIP reason=no-transformers-reference crate=$CRATE"
fi

# greedy_agreement uses the llama/transformers reference binary; skip for deepseek.
if [ "$_HAS_REFERENCE" = 1 ] && [ -n "${QS_BIN:-}" ]; then
    CUDA_VISIBLE_DEVICES="$ONE" "$PY" "$HERE/greedy_agreement.py" \
        --qs-bin "$QS_BIN" --model "$MDIR" | grep -E "^AGREE|^SUMMARY"
fi

if [ -n "$BENCH_LOG" ]; then
    "$PY" "$HERE/table.py" "$BENCH_LOG" "$BASELINE"
else
    echo "no bench log given; baseline and checks only (table: xval.sh $CRATE <weights> <bench-log>)"
fi

if [ "${PROF:-0}" = 1 ]; then
    # Use the workload's tp for device selection; prof grid comes from merged grids().
    CUDA_VISIBLE_DEVICES="$GPUS" "$HERE/prof.sh" "vllm-$CRATE" \
        "$PY" "$HERE/vmin_fit.py" "$CRATE" prof --model "$MDIR"
    if [ -n "${QS_BENCH_BIN:-}" ]; then
        MODEL="$MDIR" CUDA_VISIBLE_DEVICES="$GPUS" "$HERE/prof.sh" "qserve-$CRATE" \
            "$QS_BENCH_BIN" decode_loop --bench
    fi
fi

# Serving replay with the workload-identity gate; XVAL_SERVE_ARGS passes extra
# trace_serve flags (--prefix-caching, --speed, --ep, ...).
if [ -n "${XVAL_SERVE_TRACE:-}" ]; then
    echo "workload_hash=$("$PY" "$HERE/synth_bench.py" hash "$XVAL_SERVE_TRACE")"
    CUDA_VISIBLE_DEVICES="$GPUS" "$PY" "$HERE/trace_serve.py" "$XVAL_SERVE_TRACE" \
        --model "$MDIR" --tp "$TP" ${XVAL_SERVE_ARGS:-} \
        | grep -E "^META|^WORKLOADSUM|^WORKLOADVOID|^SERVESUM|^SLASUM"
fi
