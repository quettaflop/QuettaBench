#!/usr/bin/env bash
# Launch one vLLM server, sweep ALL profiles × concurrencies against it, teardown.
# Matches the H100 baseline sweep structure from run_all_benchmarks.sh.
#
# Usage:
#   bash sweep_all_profiles.sh \
#       MODEL_PATH TP SHORT_NAME BACKEND OUT_DIR \
#       [PY] [GPU_MEM] [MAX_LEN] [CONC_LIST] [PROFILE_LIST] [MIN_NREQ_PER_CONC]
#
# Defaults match the H100 canonical sweep (run_all_benchmarks.sh CONC_SWEEP +
# standard production profiles).
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
CONCS="${9:-1 10 20 40 80 160 256 320}"
PROFILES="${10:-chat-singleturn coding-singleturn}"
MIN_NREQ="${11:-100}"
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
echo "[sweep] MODEL=$MODEL_PATH TP=$TP OUT=$OUT_DIR"
echo "[sweep] concurrencies: $CONCS"
echo "[sweep] profiles: $PROFILES"
echo "[sweep] min requests per run: $MIN_NREQ (single-turn uses at least 2x concurrency)"
echo "[sweep] dashboard scope: $DASHBOARD_SCOPE"
echo "[sweep] context cap: $MAX_LEN tokens, safety margin: $CONTEXT_SAFETY_MARGIN_TOKENS"

VLLM_EXTRA_ARGS=()
if "$PY" -m vllm.entrypoints.openai.api_server --help 2>&1 | grep -q -- '--gdn-prefill-backend'; then
    GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-triton}"
    VLLM_EXTRA_ARGS+=(--gdn-prefill-backend "$GDN_PREFILL_BACKEND")
    echo "[sweep] gdn prefill backend: $GDN_PREFILL_BACKEND"
fi
if truthy "${VLLM_ENFORCE_EAGER:-0}"; then
    VLLM_EXTRA_ARGS+=(--enforce-eager)
    echo "[sweep] enforce eager: enabled"
fi
# Expert parallelism (MoE): ENABLE_EP=1 is injected by compile_sweep for the
# moe_ep scope.
if truthy "${ENABLE_EP:-0}"; then
    if "$PY" -m vllm.entrypoints.openai.api_server --help 2>&1 | grep -q -- '--enable-expert-parallel'; then
        VLLM_EXTRA_ARGS+=(--enable-expert-parallel)
        echo "[sweep] expert parallelism: enabled"
    else
        echo "[sweep] WARNING: ENABLE_EP set but vllm lacks --enable-expert-parallel; running dense"
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
echo "[sweep] vllm PID=$SERVER_PID (port $PORT)"

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

    echo "[sweep] server did not exit after SIGTERM; sending SIGKILL"
    kill -KILL "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup_server EXIT

for i in $(seq 1 180); do
    if curl -sf "http://localhost:$PORT/v1/models" -H "Authorization: Bearer $API_KEY" > /dev/null 2>&1; then
        echo "[sweep] server ready after ${i}×5s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[sweep] server died; tail log:"
        tail -30 "$SERVER_LOG"
        exit 1
    fi
    sleep 5
done

# Verify the API is actually functional with a real chat completion.
HEALTH_JSON='{"model":"'"$MODEL_PATH"'","messages":[{"role":"user","content":"Hi"}],"max_tokens":1}'
if ! curl -sf -m 120 "http://localhost:$PORT/v1/chat/completions" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d "$HEALTH_JSON" > /dev/null 2>&1; then
    echo "[sweep] API health check failed — server is reachable but chat completions are not functional"
    echo "[sweep] tail of server log:"
    tail -30 "$SERVER_LOG"
    exit 1
fi
echo "[sweep] API health check passed"

cd "$BENCH_REMOTE_ROOT"

# Capture engine version alongside results so the dashboard can attribute
# each sweep to a specific vllm build. Written once per sweep; applies to
# every result file in the output dir.
VLLM_VERSION=$("$PY" -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
echo "backend=vllm version=$VLLM_VERSION" > "$OUT_DIR/_engine_version.txt"
echo "[sweep] captured engine version: vllm $VLLM_VERSION"

for PROFILE in $PROFILES; do
    for CONC in $CONCS; do
        OUT_FILE="$OUT_DIR/${SHORT}_tp${TP}_${BACKEND}_${PROFILE}_conc${CONC}.json"
        if [ -f "$OUT_FILE" ] && [ -s "$OUT_FILE" ]; then
            if result_scope_matches_expected "$OUT_FILE"; then
                echo "[skip] $OUT_FILE exists with dashboard_scope=$DASHBOARD_SCOPE"
                continue
            fi
            echo "[rerun] $OUT_FILE exists with stale/missing dashboard_scope; overwriting for $DASHBOARD_SCOPE"
        fi
        local_nreq="$MIN_NREQ"
        [[ "$CONC" -eq 1 ]] && local_nreq=30
        if [[ "$PROFILE" != *multiturn* && "$CONC" -gt 1 ]]; then
            min_loaded_nreq=$(( CONC * 2 ))
            [[ "$local_nreq" -lt "$min_loaded_nreq" ]] && local_nreq="$min_loaded_nreq"
        fi
        echo ""
        echo "=== profile=$PROFILE conc=$CONC nreq=$local_nreq ==="
        OPENAI_API_KEY="$API_KEY" "$PY" -m src.benchmark.runner \
            --url        "http://localhost:$PORT/v1/chat/completions" \
            --model      "$MODEL_PATH" \
            --backend    "$BACKEND" \
            --profile    "$PROFILE" \
            --concurrency "$CONC" \
            --num-requests "$local_nreq" \
            --prefix-caching-state on \
            --chunked-prefill on \
            --max-model-len "$MAX_LEN" \
            --max-context-tokens "$MAX_LEN" \
            --context-safety-margin-tokens "$CONTEXT_SAFETY_MARGIN_TOKENS" \
            --gpu-memory-utilization "$GPU_MEM" \
            --tensor-parallel-size "$TP" \
            --warmup     2 \
            --timeout    300 \
            --api-key    "$API_KEY" \
            --scope      "$DASHBOARD_SCOPE" \
            --output     "$OUT_FILE" || echo "[warn] bench failed for $PROFILE conc=$CONC (continuing)"
    done
done

echo "[sweep] done; results in $OUT_DIR"
ls -la "$OUT_DIR" | tail -15
