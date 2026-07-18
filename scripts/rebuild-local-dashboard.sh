#!/usr/bin/env bash
# Rebuild the private dashboard from the durable local benchmark store.
#
# Defaults:
#   raw results: /mnt/100g/agent-bench/results
#   state root:  /mnt/100g/agent-bench/state
#   JSON base:   /quettaboard
#
# This is the freshness path for the Tailscale dashboard. R2 mirroring is
# optional and happens only after local artifacts validate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# TODO(phase-1): the dashboard and its data build (npm run build:data) now live
# in the separate QuettaBoard repo; DASHBOARD_DIR points at that checkout.
DASHBOARD_DIR="${DASHBOARD_DIR:-/root/QuettaBoard}"
# Neutral dir for all dashboard-JSON artifacts (reads AND writes). The dashboard
# tree lives in QuettaBoard now; nothing lands in a repo-relative public/dist path.
BENCH_ARTIFACT_DIR="${BENCH_ARTIFACT_DIR:-/mnt/100g/agent-bench/artifacts}"

RESULTS_DIR="${BENCHMARK_RESULTS_DIR:-/mnt/100g/agent-bench/results}"
STATE_ROOT="${BENCH_STATE_ROOT:-/mnt/100g/agent-bench/state}"
JSON_BASE="${DASHBOARD_JSON_BASE:-/quettaboard}"
LIVE_DIST="$DASHBOARD_DIR/dist"
NEXT_DIST="${DASHBOARD_NEXT_DIST:-$DASHBOARD_DIR/dist.next}"
PREV_DIST="${DASHBOARD_PREV_DIST:-$DASHBOARD_DIR/dist.prev}"
GPU_STATE_OUT="${GPU_STATE_OUT:-$BENCH_ARTIFACT_DIR/gpu-state.json}"
GPU_STATE_REPORT="${GPU_STATE_REPORT:-/tmp/agentic-serve-gpu-state-latest.md}"
GPU_STATE_SSH_TIMEOUT="${GPU_STATE_SSH_TIMEOUT:-12}"
GPU_STATE_HOSTS="${GPU_STATE_HOSTS:-}"

ENDPOINT="${R2_ENDPOINT:-https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com}"
BUCKET="${R2_BUCKET:-agent-bench}"
PROFILE="${AWS_PROFILE:-r2}"
MIRROR_R2="${MIRROR_R2:-0}"

usage() {
    sed -n '1,13p' "$0"
    cat <<'EOF'

Options:
  --results-dir PATH   Raw benchmark results root
  --state-root PATH    Orchestrator state root
  --json-base PATH     Dashboard JSON base URL (default: /quettaboard)
  --mirror-r2          Best-effort upload of validated JSON artifacts to R2
  --no-mirror-r2       Disable R2 JSON artifact upload
EOF
}

require_option_value() {
    if [[ $# -lt 2 ]]; then
        echo "missing value for $1" >&2
        usage >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-dir)
            require_option_value "$@"
            RESULTS_DIR="$2"
            shift 2
            ;;
        --state-root)
            require_option_value "$@"
            STATE_ROOT="$2"
            shift 2
            ;;
        --json-base)
            require_option_value "$@"
            JSON_BASE="$2"
            shift 2
            ;;
        --mirror-r2)
            MIRROR_R2=1
            shift
            ;;
        --no-mirror-r2)
            MIRROR_R2=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

LOCK_FILE="${DASHBOARD_LOCK_FILE:-/tmp/agentic-serve-dashboard-artifacts.lock}"
exec 9>"$LOCK_FILE"
flock 9

if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "missing results dir: $RESULTS_DIR" >&2
    exit 1
fi

mkdir -p "$STATE_ROOT" "$BENCH_ARTIFACT_DIR"

echo "Building sweep-state.json from $STATE_ROOT"
python3 "$SCRIPT_DIR/publish_sweep_state.py" \
    --state-dir "$STATE_ROOT" \
    --out "$BENCH_ARTIFACT_DIR/sweep-state.json" \
    --no-upload

echo "Building data.json from $RESULTS_DIR into $BENCH_ARTIFACT_DIR"
# QuettaBoard's build:data (scripts/build-data.ts) honours DASHBOARD_DATA_OUTPUT for
# data.json AND the scoped data.<scope>.json sidecars (same OUTPUT_DIR), so pointing
# it at BENCH_ARTIFACT_DIR keeps every produced JSON out of the QuettaBoard checkout.
(
    cd "$DASHBOARD_DIR"
    BENCHMARK_RESULTS_DIR="$RESULTS_DIR" DASHBOARD_DATA_OUTPUT="$BENCH_ARTIFACT_DIR/data.json" npm run build:data
)

echo "Validating data.json"
(
    cd "$DASHBOARD_DIR"
    SWEEP_STATE_PATH="$BENCH_ARTIFACT_DIR/sweep-state.json" npm run validate:data -- "$BENCH_ARTIFACT_DIR/data.json"
)

# The production frontend bundle is owned by QuettaBoard via its own deploy:tailscale
# target; do not build it from this producer. Off by default; opt in with
# BENCH_BUILD_BUNDLE=1 only for standalone/debug rebuilds.
if [ "${BENCH_BUILD_BUNDLE:-0}" = "1" ]; then
    echo "Building local dashboard bundle with JSON base $JSON_BASE into $NEXT_DIST"
    rm -rf "$NEXT_DIST"
    (
        cd "$DASHBOARD_DIR"
        VITE_R2_JSON_BASE="$JSON_BASE" npm run build -- --outDir "$NEXT_DIST" --emptyOutDir
    )
fi

echo "Building coverage-blockers.synthetic_distributional.json from $STATE_ROOT"
python3 "$SCRIPT_DIR/reconcile_sweep_coverage.py" \
    --scope synthetic_distributional \
    --data "$BENCH_ARTIFACT_DIR/data.synthetic_distributional.json" \
    --sweep-yaml "$SCRIPT_DIR/sweep.yaml" \
    --bench-jobs "$SCRIPT_DIR/bench_jobs.txt" \
    --state-dir "$STATE_ROOT" \
    --write-blockers-json "$BENCH_ARTIFACT_DIR/coverage-blockers.synthetic_distributional.json" \
    --no-report \
    >/dev/null

echo "Building private gpu-state.json from $STATE_ROOT"
gpu_state_args=(
    --jobs-config "$SCRIPT_DIR/sweep.yaml"
    --scope "${BENCH_JOBS_SCOPE:-synthetic_distributional}"
    --state-dir "$STATE_ROOT"
    --ssh-timeout "$GPU_STATE_SSH_TIMEOUT"
    --out "$GPU_STATE_REPORT"
    --json-out "$GPU_STATE_OUT"
    --once
)
if [[ -n "$GPU_STATE_HOSTS" ]]; then
    IFS=', ' read -r -a gpu_state_hosts <<< "$GPU_STATE_HOSTS"
    gpu_state_args+=(--hosts "${gpu_state_hosts[@]}")
fi
python3 "$SCRIPT_DIR/sweep_progress_report.py" "${gpu_state_args[@]}"

# Bundle promotion only applies when the (opt-in) local bundle was built above;
# QuettaBoard owns the live bundle via deploy:tailscale.
if [ "${BENCH_BUILD_BUNDLE:-0}" = "1" ]; then
    echo "Promoting rebuilt dashboard bundle to $LIVE_DIST without removing the live tree"
    mkdir -p "$LIVE_DIST"
    rsync -a --delay-updates "$NEXT_DIST"/ "$LIVE_DIST"/
    rm -rf "$NEXT_DIST"
fi

if [[ "$MIRROR_R2" == "1" ]]; then
    if command -v aws >/dev/null 2>&1; then
        echo "Mirroring validated dashboard JSON artifacts to R2"
        # gpu-state.json intentionally stays local/private: it includes host,
        # user, port, and process occupancy details for the Tailscale dashboard.
        for artifact in \
            data.json \
            data.trace_replay.json \
            data.synthetic_distributional.json \
            data.archived.json \
            sweep-state.json \
            gemm-eval.json \
            serving-predictions.json \
            profiling-state.json \
            predictor-coverage.json \
            roofline-data.json \
            roofline-quadrant.json \
            gemm-extrapolation.json
        do
            path="$BENCH_ARTIFACT_DIR/$artifact"
            if [[ -f "$path" ]]; then
                # Upload gzipped with content-encoding metadata: browsers
                # decompress transparently and the wire size drops ~5-10x.
                # Uploading raw here would undo that for every consumer.
                gz_tmp="$(mktemp)"
                gzip -9 -c "$path" > "$gz_tmp"
                aws --profile "$PROFILE" --endpoint-url "$ENDPOINT" s3 cp \
                    "$gz_tmp" "s3://$BUCKET/json/current/$artifact" \
                    --content-encoding gzip --content-type application/json \
                    --cache-control "public, max-age=86400" \
                    --only-show-errors || echo "warning: failed to mirror $artifact" >&2
                rm -f "$gz_tmp"
            fi
        done
    else
        echo "warning: aws cli not found; skipping R2 mirror" >&2
    fi
fi

echo "Local dashboard rebuild complete"
