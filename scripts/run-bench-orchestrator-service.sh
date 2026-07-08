#!/usr/bin/env bash
# Systemd entrypoint for one GPU benchmark orchestration tick.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

truthy() {
    [[ "${1:-}" == "1" || "${1:-}" == "true" || "${1:-}" == "yes" ]]
}

SYNC_GPU_CODE="${BENCH_SYNC_GPU_CODE:-1}"

if truthy "$SYNC_GPU_CODE"; then
    SYNC_ARGS=()
    [[ -n "${BENCH_JOBS_FILE:-}" ]] && SYNC_ARGS+=(--jobs-file "$BENCH_JOBS_FILE")
    [[ -n "${BENCH_JOBS_CONFIG:-}" ]] && SYNC_ARGS+=(--jobs-config "$BENCH_JOBS_CONFIG")
    [[ -n "${BENCH_JOBS_SCOPE:-}" ]] && SYNC_ARGS+=(--scope "$BENCH_JOBS_SCOPE")
    [[ -n "${BENCH_HOSTS:-}" ]] && SYNC_ARGS+=(--hosts "$BENCH_HOSTS")
    [[ -n "${BENCH_REMOTE_ROOT:-}" ]] && SYNC_ARGS+=(--remote-root "$BENCH_REMOTE_ROOT")
    truthy "${BENCH_SYNC_DRY_RUN:-0}" && SYNC_ARGS+=(--dry-run)
    truthy "${BENCH_SYNC_REQUIRED:-0}" && SYNC_ARGS+=(--required)
    "$SCRIPT_DIR/sync-gpu-code.sh" "${SYNC_ARGS[@]}"
fi

exec "$SCRIPT_DIR/bench_orchestrator.sh"
