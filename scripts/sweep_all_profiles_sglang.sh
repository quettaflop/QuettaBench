#!/usr/bin/env bash
# sglang variant of sweep_all_profiles.sh.
#
# Launches `python -m sglang.launch_server` (OpenAI-compat on :8089),
# sweeps profiles x concurrencies, teardown. Same positional-arg shape
# as the vllm script so bench_orchestrator.sh can swap launchers by
# backend without reshaping the CMD line.
#
# Usage (matches sweep_all_profiles.sh):
#   bash sweep_all_profiles_sglang.sh \
#       MODEL_PATH TP SHORT_NAME BACKEND OUT_DIR \
#       [PY] [GPU_MEM] [MAX_LEN] [CONC_LIST] [PROFILE_LIST] [MIN_NREQ_PER_CONC]
#
# BACKEND is expected to be "sglang" - the 4th arg is preserved for CMD
# symmetry with the vllm script; the output filenames still use it.
set -uo pipefail

# sglang 0.5.9 has a startup check that refuses to run on torch 2.9.1 +
# cudnn < 9.15 due to a nn.Conv3d bug (pytorch#168167). LLM inference
# doesn't touch Conv3d, so the check is a false positive for us.
export SGLANG_DISABLE_CUDNN_CHECK=1

# NCCL init workarounds — sglang 0.5.9 hits "unhandled system error" on
# 3090 tp>1 otherwise. Disabling direct P2P and shared-memory fast paths
# falls back to CUDA IPC which works universally (slightly slower but
# fine for benchmarks where the primary metric is forward-pass latency,
# not NCCL bandwidth).
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
# Surface the real NCCL error (not just "unhandled system error") on crash.
export NCCL_DEBUG=WARN

# sglang's CUDA graph runner JIT-compiles flashinfer kernels with nvcc
# then links with libcudart. Point CUDA_HOME at the conda-installed nvcc
# (cuda-nvcc=12.8 from nvidia channel) and make sure the linker finds
# libcudart.so (from nvidia-cuda-runtime-cu12 pip wheel or the env lib).
SGLANG_ENV_DIR="$(dirname "$(dirname "${6:-python}")")"
if [[ -x "$SGLANG_ENV_DIR/bin/nvcc" ]]; then
    export CUDA_HOME="$SGLANG_ENV_DIR"
    export PATH="$SGLANG_ENV_DIR/bin:$PATH"
    # Compile-time lib search (ld -lcudart) and runtime dynamic linker.
    export LIBRARY_PATH="$SGLANG_ENV_DIR/lib:$SGLANG_ENV_DIR/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="$SGLANG_ENV_DIR/lib:$SGLANG_ENV_DIR/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
fi

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
echo "[sweep-sglang] MODEL=$MODEL_PATH TP=$TP OUT=$OUT_DIR"
echo "[sweep-sglang] concurrencies: $CONCS"
echo "[sweep-sglang] profiles: $PROFILES"
echo "[sweep-sglang] min requests per run: $MIN_NREQ (single-turn uses at least 2x concurrency)"
echo "[sweep-sglang] dashboard scope: $DASHBOARD_SCOPE"

# Detect GPU arch - 2080Ti (sm75) has no CUDA graph kernel images.
# sm_75 kernels are not included in sglang 0.5.9 prebuilt wheels.
SGLANG_CUDA_GRAPH_ARGS=""
GPU_ARCH=$(python3 -c "import torch; print(torch.cuda.get_device_capability()[0])" 2>/dev/null || echo 0)
if [ "$GPU_ARCH" = "7" ]; then
    SGLANG_CUDA_GRAPH_ARGS="--disable-cuda-graph"
    echo "[sweep-sglang] sm75 detected; disabling CUDA graphs"
fi

# sglang.launch_server flags:
#   --model-path         path to HF model dir
#   --host / --port      bind address
#   --api-key            bearer token (matches OpenAI-compat /v1/*)
#   --tp                 tensor-parallel size
#   --mem-fraction-static  analogous to vllm --gpu-memory-utilization
#   --context-length     analogous to vllm --max-model-len
#   --trust-remote-code  HF trust-remote-code
# Expert parallelism (MoE): ENABLE_EP=1 is injected by compile_sweep for the
# moe_ep scope. Newer sglang uses --ep-size; older builds used --enable-ep-moe.
SGLANG_EP_ARGS=""
if [[ "${ENABLE_EP:-0}" == "1" || "${ENABLE_EP:-}" == "true" ]]; then
    if "$PY" -m sglang.launch_server --help 2>&1 | grep -q -- '--ep-size'; then
        SGLANG_EP_ARGS="--ep-size $TP"
    elif "$PY" -m sglang.launch_server --help 2>&1 | grep -q -- '--enable-ep-moe'; then
        SGLANG_EP_ARGS="--enable-ep-moe"
    fi
    echo "[sweep-sglang] expert parallelism enabled: ${SGLANG_EP_ARGS:-<flag not found; running dense>}"
fi
"$PY" -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --api-key "$API_KEY" \
    --tp "$TP" \
    --mem-fraction-static "$GPU_MEM" \
    --context-length "$MAX_LEN" \
    --trust-remote-code \
    $SGLANG_CUDA_GRAPH_ARGS \
    $SGLANG_EP_ARGS \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[sweep-sglang] sglang PID=$SERVER_PID (port $PORT)"

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

    echo "[sweep-sglang] server did not exit after SIGTERM; sending SIGKILL"
    kill -KILL "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup_server EXIT

# sglang takes a bit longer to warm up than vllm - same 15-min budget.
for i in $(seq 1 180); do
    if curl -sf "http://localhost:$PORT/v1/models" -H "Authorization: Bearer $API_KEY" > /dev/null 2>&1; then
        echo "[sweep-sglang] server ready after ${i}x5s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[sweep-sglang] server died; tail log:"
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
    echo "[sweep-sglang] API health check failed — server is reachable but chat completions are not functional"
    echo "[sweep-sglang] tail of server log:"
    tail -30 "$SERVER_LOG"
    exit 1
fi
echo "[sweep-sglang] API health check passed"

cd "$BENCH_REMOTE_ROOT"

# Capture engine version for dashboard attribution.
SGLANG_VERSION=$("$PY" -c "import sglang; print(sglang.__version__)" 2>/dev/null || echo "unknown")
echo "backend=sglang version=$SGLANG_VERSION" > "$OUT_DIR/_engine_version.txt"
echo "[sweep-sglang] captured engine version: sglang $SGLANG_VERSION"

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
            --chunked-prefill unknown \
            --max-model-len "$MAX_LEN" \
            --gpu-memory-utilization "$GPU_MEM" \
            --tensor-parallel-size "$TP" \
            --scope      "$DASHBOARD_SCOPE" \
            --output     "$OUT_FILE" \
            --mode       single-turn \
            2>&1 | tail -8
    done
done

echo "[sweep-sglang] done; results in $OUT_DIR"
ls -la "$OUT_DIR" | tail -20
