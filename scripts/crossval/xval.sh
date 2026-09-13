#!/usr/bin/env bash
# xval.sh <crate> <weights-dir> [bench-log] -- run the whole cross-validation.
# Captures the vLLM baseline if it is not cached, logit-checks the baseline
# against the transformers reference (LOGIT=0 skips), runs the greedy
# check when QS_BIN points at an engine binary, and prints the comparison
# table when a criterion bench log is given. PY selects the python with vllm.
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
