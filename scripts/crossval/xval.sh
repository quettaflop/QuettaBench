#!/usr/bin/env bash
# xval.sh <crate> <weights-dir> [bench-log] -- run the whole cross-validation.
# Captures the vLLM baseline if it is not cached, logit-checks the baseline
# against the transformers reference (LOGIT=0 skips), runs the greedy
# check when QS_BIN points at an engine binary, and prints the comparison
# table when a criterion bench log is given. PY selects the python with vllm.
# PROF=1 nsys-captures the profiling cell on both sides afterwards: the vLLM
# baseline, and the engine decode_loop when QS_BENCH_BIN names the bench.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE="${1:?crate}"
MDIR="${2:?weights dir}"
BENCH_LOG="${3:-}"
PY="${PY:-python3}"

BASELINE="$HERE/baselines/vllm-$CRATE.json"
[ -f "$BASELINE" ] || "$HERE/vllm.sh" "$CRATE" "" "$MDIR" "$PY"

if [ "${LOGIT:-1}" = 1 ]; then
    "$PY" "$HERE/logit_agreement.py" --model "$MDIR" | grep -E "SUMMARY|DISAGREE|NEAR_TIE"
fi

if [ -n "${QS_BIN:-}" ]; then
    "$PY" "$HERE/greedy_agreement.py" --qs-bin "$QS_BIN" --model "$MDIR" | grep -E "^AGREE|^SUMMARY"
fi

if [ -n "$BENCH_LOG" ]; then
    "$PY" "$HERE/table.py" "$BENCH_LOG" "$BASELINE"
else
    echo "no bench log given; baseline and checks only (table: xval.sh $CRATE <weights> <bench-log>)"
fi

if [ "${PROF:-0}" = 1 ]; then
    MODELPATH="$MDIR" "$HERE/prof.sh" "vllm-$CRATE" "$PY" "$HERE/vmin_fit.py" FULL "$CRATE" prof
    if [ -n "${QS_BENCH_BIN:-}" ]; then
        MODEL="$MDIR" "$HERE/prof.sh" "qserve-$CRATE" "$QS_BENCH_BIN" decode_loop --bench
    fi
fi
