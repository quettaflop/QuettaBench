#!/usr/bin/env bash
# Refresh only the private GPU/orchestrator state JSON served by the dashboard.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# TODO(phase-1): the dashboard now lives in the separate QuettaBoard repo.
DASHBOARD_DIR="${DASHBOARD_DIR:-/root/QuettaBoard}"

STATE_ROOT="${BENCH_STATE_ROOT:-/mnt/100g/agent-bench/state}"
LIVE_DIST="${DASHBOARD_LIVE_DIST:-$DASHBOARD_DIR/dist}"
GPU_STATE_OUT="${GPU_STATE_OUT:-$LIVE_DIST/gpu-state.json}"
GPU_STATE_REPORT="${GPU_STATE_REPORT:-/tmp/agentic-serve-gpu-state-latest.md}"
GPU_STATE_SSH_TIMEOUT="${GPU_STATE_SSH_TIMEOUT:-8}"
GPU_STATE_HOSTS="${GPU_STATE_HOSTS:-}"
LOCK_FILE="${DASHBOARD_LOCK_FILE:-/tmp/agentic-serve-dashboard-artifacts.lock}"

exec 9>"$LOCK_FILE"
flock 9

mkdir -p "$(dirname "$GPU_STATE_OUT")"
tmp_json="$(mktemp "$(dirname "$GPU_STATE_OUT")/.gpu-state.json.XXXXXX")"
cleanup() {
    rm -f "$tmp_json"
}
trap cleanup EXIT

gpu_state_args=(
    --jobs-config "$SCRIPT_DIR/sweep.yaml"
    --scope "${BENCH_JOBS_SCOPE:-synthetic_distributional}"
    --state-dir "$STATE_ROOT"
    --ssh-timeout "$GPU_STATE_SSH_TIMEOUT"
    --out "$GPU_STATE_REPORT"
    --json-out "$tmp_json"
    --once
)
if [[ -n "$GPU_STATE_HOSTS" ]]; then
    IFS=', ' read -r -a gpu_state_hosts <<< "$GPU_STATE_HOSTS"
    gpu_state_args+=(--hosts "${gpu_state_hosts[@]}")
fi

python3 "$SCRIPT_DIR/sweep_progress_report.py" "${gpu_state_args[@]}"
chmod 0644 "$tmp_json"
mv "$tmp_json" "$GPU_STATE_OUT"
trap - EXIT
