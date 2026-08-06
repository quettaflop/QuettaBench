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
# Per-host probe budget. History, because this was tuned in the wrong direction twice:
#   8 s  -> too tight; recurring TimeoutExpired on the GPU page. The cost was nvidia-smi
#           taking 10-14 s PER CALL on a host whose driver had gone cold (per-GPU timing
#           decayed 9.5 s -> 0.13 s as it warmed, so a cold start, not a failing card).
#   45 s -> wrong fix. It merely accommodated the stall, and with the remote script's two
#           `timeout 20` caps the worst case (~40 s of nvidia-smi + ~5 s SSH) straddled a
#           timer that fires every 60 s, so one bad host ate most of a tick.
# The cause is now REMOVED rather than accommodated: nvidia-persistenced is active on
# 3090/a100/h100, so the driver stays initialised and queries hold ~0.14-0.21 s. The
# budget therefore fails FAST -- a pathological host is reported inside one tick instead
# of blocking it. If TimeoutExpired reappears, check persistence mode is still on
# (`nvidia-smi --query-gpu=persistence_mode`) before raising this number again.
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
