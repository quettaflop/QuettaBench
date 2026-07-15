#!/usr/bin/env bash
# =============================================================================
# Launch vLLM server with mode-appropriate flags.
#
# Usage:
#   ./scripts/launch_server.sh [MODE] [OPTIONS]
#
# Modes:
#   stress-test    Random token workload. Prefix caching OFF.
#   single-turn    Real prompt workload. Prefix caching ON.
#
# Examples:
#   ./scripts/launch_server.sh stress-test
#   ./scripts/launch_server.sh single-turn
#   ./scripts/launch_server.sh single-turn --model meta-llama/Llama-3.1-70B-Instruct-FP8
#   ./scripts/launch_server.sh single-turn --gpu-memory-utilization 0.90
#
# =============================================================================
set -euo pipefail

PYTHON="${PYTHON:-$(which python)}"

# Defaults
MODEL="meta-llama/Llama-3.1-8B-Instruct"
PORT=8000
GPU_MEM=0.75
DTYPE="bfloat16"
MAX_MODEL_LEN=32768
TP=1
API_KEY="test"
EXTRA_FLAGS=""

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
    echo "Usage: $0 [stress-test|single-turn] [--model MODEL] [--port PORT] [--gpu-memory-utilization N]"
    exit 1
fi
shift

# Parse remaining args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)                  MODEL="$2"; shift 2 ;;
        --port)                   PORT="$2"; shift 2 ;;
        --gpu-memory-utilization) GPU_MEM="$2"; shift 2 ;;
        --dtype)                  DTYPE="$2"; shift 2 ;;
        --max-model-len)          MAX_MODEL_LEN="$2"; shift 2 ;;
        --tensor-parallel-size)   TP="$2"; shift 2 ;;
        --api-key)                API_KEY="$2"; shift 2 ;;
        *) EXTRA_FLAGS="$EXTRA_FLAGS $1"; shift ;;
    esac
done

# Mode-specific flags
case "$MODE" in
    stress-test)
        PREFIX_FLAG=""
        echo "=== stress-test mode ==="
        echo "Prefix caching: OFF (mirrors InferenceX)"
        echo "Use --ignore-eos when running benchmark (auto-set by --mode stress-test)"
        ;;
    single-turn)
        PREFIX_FLAG="--enable-prefix-caching"
        # Use full context window — do NOT constrain to ISL+OSL+margin (InferenceX anti-pattern).
        # 32768 is the practical limit on A6000 (48GB VRAM); increase for H100/B200.
        echo "=== single-turn mode ==="
        echo "Prefix caching: ON (--enable-prefix-caching)"
        echo "Max model len:  $MAX_MODEL_LEN (full context window, not ISL+OSL+margin)"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Available: stress-test | single-turn"
        exit 1
        ;;
esac

echo "Model:   $MODEL"
echo "Port:    $PORT"
echo "GPU mem: $GPU_MEM"
echo "TP:      $TP"
echo ""
echo "Log: /tmp/vllm_${MODE}.log"
echo ""

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
"$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model                    "$MODEL" \
    --port                     "$PORT" \
    --max-model-len            "$MAX_MODEL_LEN" \
    --gpu-memory-utilization   "$GPU_MEM" \
    --dtype                    "$DTYPE" \
    --tensor-parallel-size     "$TP" \
    --api-key                  "$API_KEY" \
    --enable-chunked-prefill \
    $PREFIX_FLAG \
    $EXTRA_FLAGS \
    > "/tmp/vllm_${MODE}.log" 2>&1 &

PID=$!
echo "vLLM started (PID $PID) in background."
echo "Tail log: tail -f /tmp/vllm_${MODE}.log"
echo "Wait for: curl http://localhost:${PORT}/health"
