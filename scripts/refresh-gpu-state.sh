#!/usr/bin/env bash
# Refresh only the private GPU/orchestrator state JSON served by the dashboard.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# TODO(phase-1): the dashboard now lives in the separate QuettaBoard repo.
DASHBOARD_DIR="${DASHBOARD_DIR:-/root/QuettaBoard}"
# gpu-state.json is a dashboard-JSON artifact; it lands in the neutral artifact dir.
BENCH_ARTIFACT_DIR="${BENCH_ARTIFACT_DIR:-/mnt/100g/agent-bench/artifacts}"

STATE_ROOT="${BENCH_STATE_ROOT:-/mnt/100g/agent-bench/state}"
LIVE_DIST="${DASHBOARD_LIVE_DIST:-$DASHBOARD_DIR/dist}"
GPU_STATE_OUT="${GPU_STATE_OUT:-$BENCH_ARTIFACT_DIR/gpu-state.json}"
GPU_STATE_REPORT="${GPU_STATE_REPORT:-/tmp/agentic-serve-gpu-state-latest.md}"
# Per-host probe budget. Was 8 s, which h100 could not meet: its nvidia-smi takes
# 10-14 s PER CALL (vs <1.2 s on a100/3090, all GPUs idle, persistence disabled on all
# three -- so it is an h100 host issue, not load), and the snapshot script makes three
# nvidia-smi calls. h100 therefore recorded a TimeoutExpired on the GPU page while the
# other hosts passed. SSH multiplexing (~/.ssh/config ControlMaster) removed the ~4 s
# handshake, but the nvidia-smi cost dominates, so the budget has to cover ~3x14 s.
GPU_STATE_SSH_TIMEOUT="${GPU_STATE_SSH_TIMEOUT:-45}"
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
