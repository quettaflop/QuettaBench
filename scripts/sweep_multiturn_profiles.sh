#!/usr/bin/env bash
# Multi-turn sweep variant of sweep_all_profiles.sh — adds --mode multi-turn.
# Profiles default to the canonical distributional multi-turn paper suite.
#
# Usage:
#   bash sweep_multiturn_profiles.sh \
#       MODEL_PATH TP SHORT_NAME BACKEND OUT_DIR \
#       [PY] [GPU_MEM] [MAX_LEN] [CONC_LIST] [PROFILE_LIST] [WARMUP]
set -euo pipefail

# Include CUDA graph memory in vLLM's pre-flight memory profiler.
# Without this, vLLM sizes the KV cache greedily and OOMs during cudagraph
# capture on tight configs (e.g. 70B/72B at TP=4 on 40GB A100). Slightly
# reduces KV cache headroom in exchange for guaranteed startup; will be
# vLLM's default in v0.19+.
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1

truthy() {
    [[ "${1:-}" == "1" || "${1:-}" == "true" || "${1:-}" == "yes" || "${1:-}" == "on" ]]
}

MODEL_PATH="${1:?model path}"
TP="${2:?tp}"
SHORT="${3:?short}"
BACKEND="${4:?backend}"
OUT_DIR="${5:?out dir}"
PY="${6:-python}"
GPU_MEM="${7:-0.85}"
MAX_LEN="${8:-32768}"
CONCS="${9:-5 20 40 80 160}"
PROFILES="${10:-chat-multiturn swebench-multiturn terminalbench-multiturn osworld-multiturn}"
WARMUP="${11:-3}"
CONTEXT_SAFETY_MARGIN_TOKENS="${CONTEXT_SAFETY_MARGIN_TOKENS:-256}"

PORT="${PORT:-8089}"
API_KEY="${API_KEY:-test}"
DASHBOARD_SCOPE="${DASHBOARD_SCOPE:-archived}"
BENCH_REMOTE_TMP="${BENCH_REMOTE_TMP:-/tmp}"
BENCH_REMOTE_ROOT="${BENCH_REMOTE_ROOT:-/tmp/inference-benchmark}"
SERVER_LOG="$BENCH_REMOTE_TMP/vllm_${PORT}.log"

mkdir -p "$BENCH_REMOTE_TMP"
export TMPDIR="$BENCH_REMOTE_TMP"
export TMP="$BENCH_REMOTE_TMP"
export TEMP="$BENCH_REMOTE_TMP"

result_scope_matches_expected() {
    local file="$1"
    "$PY" - "$file" "$DASHBOARD_SCOPE" <<'PY'
import json
import sys

try:
    with open(sys.argv[1]) as f:
        raw = json.load(f)
except Exception:
    raise SystemExit(1)

scope = (raw.get("config") or {}).get("dashboard_scope")
if scope != sys.argv[2]:
    raise SystemExit(1)

# Refuse to skip a file that has no successful requests — it is garbage
# from a previous failed run and would block a valid re-run forever.
success_count = sum(1 for r in raw.get("per_request", []) if r.get("success"))
raise SystemExit(0 if success_count > 0 else 1)
PY
}

mkdir -p "$OUT_DIR" "$BENCH_REMOTE_TMP"
echo "[mt-sweep] MODEL=$MODEL_PATH TP=$TP OUT=$OUT_DIR"
echo "[mt-sweep] concurrencies: $CONCS"
echo "[mt-sweep] profiles: $PROFILES"
echo "[mt-sweep] dashboard scope: $DASHBOARD_SCOPE"

VLLM_EXTRA_ARGS=()
if "$PY" -m vllm.entrypoints.openai.api_server --help 2>&1 | grep -q -- '--gdn-prefill-backend'; then
    GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-triton}"
    VLLM_EXTRA_ARGS+=(--gdn-prefill-backend "$GDN_PREFILL_BACKEND")
    echo "[mt-sweep] gdn prefill backend: $GDN_PREFILL_BACKEND"
fi
if truthy "${VLLM_ENFORCE_EAGER:-0}"; then
    VLLM_EXTRA_ARGS+=(--enforce-eager)
    echo "[mt-sweep] enforce eager: enabled"
fi
# Expert parallelism (MoE): ENABLE_EP=1 is injected by compile_sweep for the
# moe_ep scope.
if truthy "${ENABLE_EP:-0}"; then
    if "$PY" -m vllm.entrypoints.openai.api_server --help 2>&1 | grep -q -- '--enable-expert-parallel'; then
        VLLM_EXTRA_ARGS+=(--enable-expert-parallel)
        echo "[mt-sweep] expert parallelism: enabled"
    else
        echo "[mt-sweep] WARNING: ENABLE_EP set but vllm lacks --enable-expert-parallel; running dense"
    fi
fi

"$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --api-key "$API_KEY" \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$GPU_MEM" \
    --max-model-len "$MAX_LEN" \
    --trust-remote-code \
    "${VLLM_EXTRA_ARGS[@]}" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[mt-sweep] vllm PID=$SERVER_PID (port $PORT)"

cleanup_server() {
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        return 0
    fi

    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            return 0
        fi
        sleep 1
    done

    echo "[mt-sweep] server did not exit after SIGTERM; sending SIGKILL"
    kill -KILL "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup_server EXIT

for i in $(seq 1 180); do
    if curl -sf "http://localhost:$PORT/v1/models" -H "Authorization: Bearer $API_KEY" > /dev/null 2>&1; then
        echo "[mt-sweep] server ready after ${i}×5s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[mt-sweep] server died; tail log:"
        tail -30 "$SERVER_LOG"
        exit 1
    fi
    sleep 5
done

# Verify the API is actually functional with a real chat completion.
# Port binding alone is not enough — vLLM can bind the port before model
# loading completes, and a silent startup failure leaves a dead-but-bound port.
HEALTH_JSON='{"model":"'"$MODEL_PATH"'","messages":[{"role":"user","content":"Hi"}],"max_tokens":1}'
if ! curl -sf -m 120 "http://localhost:$PORT/v1/chat/completions" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d "$HEALTH_JSON" > /dev/null 2>&1; then
    echo "[mt-sweep] API health check failed — server is reachable but chat completions are not functional"
    echo "[mt-sweep] tail of server log:"
    tail -30 "$SERVER_LOG"
    exit 1
fi
echo "[mt-sweep] API health check passed"

cd "$BENCH_REMOTE_ROOT"

# Capture engine version alongside results so the dashboard can attribute
# each sweep to a specific vllm build. Written once per sweep; applies to
# every result file in the output dir.
VLLM_VERSION=$("$PY" -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
echo "backend=vllm version=$VLLM_VERSION" > "$OUT_DIR/_engine_version.txt"
echo "[sweep] captured engine version: vllm $VLLM_VERSION"

for PROFILE in $PROFILES; do
    for CONC in $CONCS; do
        OUT_FILE="$OUT_DIR/${PROFILE}_conc${CONC}.json"
        if [ -f "$OUT_FILE" ] && [ -s "$OUT_FILE" ]; then
            if result_scope_matches_expected "$OUT_FILE"; then
                echo "[skip] $OUT_FILE exists with dashboard_scope=$DASHBOARD_SCOPE"
                continue
            fi
            echo "[rerun] $OUT_FILE exists with stale/missing dashboard_scope; overwriting for $DASHBOARD_SCOPE"
        fi
        echo ""
        echo "=== profile=$PROFILE conc=$CONC (multi-turn) ==="
        OPENAI_API_KEY="$API_KEY" "$PY" -m src.benchmark.runner \
            --url        "http://localhost:$PORT/v1/chat/completions" \
            --model      "$MODEL_PATH" \
            --backend    "$BACKEND" \
            --profile    "$PROFILE" \
            --concurrency "$CONC" \
            --mode       multi-turn \
            --max-context-tokens "$MAX_LEN" \
            --context-safety-margin-tokens "$CONTEXT_SAFETY_MARGIN_TOKENS" \
            --prefix-caching-state on \
            --chunked-prefill on \
            --max-model-len "$MAX_LEN" \
            --gpu-memory-utilization "$GPU_MEM" \
            --tensor-parallel-size "$TP" \
            --warmup     "$WARMUP" \
            --timeout    300 \
            --api-key    "$API_KEY" \
            --scope      "$DASHBOARD_SCOPE" \
            --output     "$OUT_FILE" || echo "[warn] mt-bench failed for $PROFILE conc=$CONC (continuing)"
    done
done

echo "[mt-sweep] done; results in $OUT_DIR"
ls -la "$OUT_DIR" | tail -15
