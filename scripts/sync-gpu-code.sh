#!/usr/bin/env bash
# Sync the benchmark runner tree to GPU hosts before orchestration.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
JOBS_FILE="${BENCH_JOBS_FILE:-}"
JOBS_CONFIG="${BENCH_JOBS_CONFIG:-$SCRIPT_DIR/sweep.yaml}"
JOBS_SCOPE="${BENCH_JOBS_SCOPE:-all}"
REMOTE_ROOT="${BENCH_REMOTE_ROOT:-/tmp/inference-benchmark}"
H100_REMOTE_TMP="${BENCH_H100_REMOTE_TMP:-/data48/tmp}"
H100_REMOTE_ROOT="${BENCH_H100_REMOTE_ROOT:-$H100_REMOTE_TMP/inference-benchmark}"
DRY_RUN="${BENCH_SYNC_DRY_RUN:-0}"
REQUIRED="${BENCH_SYNC_REQUIRED:-0}"
HOSTS=()

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --jobs-file PATH    Read hosts from a legacy bench_jobs row file
  --jobs-config PATH  Read hosts from sweep.yaml when --jobs-file is not set
  --scope SCOPE       Scope used with --jobs-config (default: all)
  --hosts LIST        Space- or comma-separated host list
  --remote-root PATH  Remote benchmark root (default: /tmp/inference-benchmark)
  --dry-run           Print rsync plan without writing remote files
  --required          Exit nonzero if any host sync fails
EOF
}

truthy() {
    [[ "${1:-}" == "1" || "${1:-}" == "true" || "${1:-}" == "yes" ]]
}

remote_root_for_host() {
    case "$1" in
        h100|h100-2) echo "$H100_REMOTE_ROOT" ;;
        *) echo "$REMOTE_ROOT" ;;
    esac
}

split_hosts() {
    local raw="$1"
    raw="${raw//,/ }"
    for host in $raw; do
        [[ -n "$host" ]] && HOSTS+=("$host")
    done
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --jobs-file)
            if [[ $# -lt 2 ]]; then
                echo "missing value for --jobs-file" >&2
                exit 1
            fi
            JOBS_FILE="$2"
            shift 2
            ;;
        --jobs-config)
            if [[ $# -lt 2 ]]; then
                echo "missing value for --jobs-config" >&2
                exit 1
            fi
            JOBS_CONFIG="$2"
            shift 2
            ;;
        --scope)
            if [[ $# -lt 2 ]]; then
                echo "missing value for --scope" >&2
                exit 1
            fi
            JOBS_SCOPE="$2"
            shift 2
            ;;
        --hosts)
            if [[ $# -lt 2 ]]; then
                echo "missing value for --hosts" >&2
                exit 1
            fi
            split_hosts "$2"
            shift 2
            ;;
        --remote-root)
            if [[ $# -lt 2 ]]; then
                echo "missing value for --remote-root" >&2
                exit 1
            fi
            REMOTE_ROOT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --required)
            REQUIRED=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ ${#HOSTS[@]} -eq 0 ]]; then
    if [[ -n "${BENCH_HOSTS:-}" ]]; then
        split_hosts "$BENCH_HOSTS"
    else
        if [[ -n "$JOBS_FILE" ]]; then
            if [[ ! -f "$JOBS_FILE" ]]; then
                echo "missing jobs file: $JOBS_FILE" >&2
                exit 1
            fi
            mapfile -t HOSTS < <(
                awk -F'|' '!/^#/ && NF >= 10 {gsub(/[[:space:]]/, "", $1); if ($1 != "") print $1}' "$JOBS_FILE" \
                    | sort -u
            )
        else
            mapfile -t HOSTS < <(
                python3 "$SCRIPT_DIR/compile_sweep.py" --yaml "$JOBS_CONFIG" --scope "$JOBS_SCOPE" --list-hosts
            )
        fi
    fi
fi

if [[ ${#HOSTS[@]} -eq 0 ]]; then
    echo "no GPU hosts found to sync" >&2
    exit 1
fi

RSYNC_ARGS=(
    -az
    --exclude dashboard/node_modules/
    --exclude dashboard/dist/
    --exclude dashboard/.omc/
    --exclude dashboard/public/data.json
    --exclude dashboard/public/sweep-state.json
    --exclude results/
    --exclude .venv/
    --exclude __pycache__/
)
truthy "$DRY_RUN" && RSYNC_ARGS+=(--dry-run)

failures=0
for host in "${HOSTS[@]}"; do
    host_remote_root="$(remote_root_for_host "$host")"
    remote_root_q="$(printf "%q" "$host_remote_root")"
    echo "syncing benchmark code to ${host}:${host_remote_root}"
    if { truthy "$DRY_RUN" || ssh -o ConnectTimeout=10 -o BatchMode=yes "$host" "mkdir -p -- $remote_root_q"; } \
        && rsync "${RSYNC_ARGS[@]}" "$BENCH_ROOT/" "$host:$host_remote_root/"; then
        echo "synced ${host}:${host_remote_root}"
    else
        echo "warning: failed to sync ${host}:${host_remote_root}" >&2
        failures=$((failures + 1))
    fi
done

if [[ "$failures" -gt 0 ]] && truthy "$REQUIRED"; then
    exit 1
fi

exit 0
