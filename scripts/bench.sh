#!/usr/bin/env bash
# Run a single benchmark profile.
#
# Usage:
#   ./scripts/bench.sh [OPTIONS]
#
# Examples:
#   ./scripts/bench.sh
#   ./scripts/bench.sh --profile coding-singleturn --concurrency 20
#   ./scripts/bench.sh --backend trtllm --url http://localhost:8000/generate_stream
#
set -euo pipefail

PYTHON="${PYTHON:-$(which python)}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Defaults (override via flags)
URL="http://localhost:8000/v1/chat/completions"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
BACKEND="vllm"
PROFILE="chat-singleturn"
CONCURRENCY=10
NUM_REQUESTS=100
WARMUP=5
ARRIVAL="steady"
TARGET_RATE=10.0
API_KEY="test"
OUTPUT=""
IGNORE_EOS=""
MODE=""
MAX_CONTEXT_TOKENS=""
CONTEXT_SAFETY_MARGIN_TOKENS="${CONTEXT_SAFETY_MARGIN_TOKENS:-256}"
PREFIX_CACHING_STATE="${PREFIX_CACHING_STATE:-auto}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
GPU_MEM="${GPU_MEM:-}"
TP="${TP:-}"

usage() {
  grep '^#' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url)           URL="$2"; shift 2 ;;
    --model)         MODEL="$2"; shift 2 ;;
    --backend)       BACKEND="$2"; shift 2 ;;
    --profile)       PROFILE="$2"; shift 2 ;;
    --concurrency)   CONCURRENCY="$2"; shift 2 ;;
    --num-requests)  NUM_REQUESTS="$2"; shift 2 ;;
    --warmup)        WARMUP="$2"; shift 2 ;;
    --arrival)       ARRIVAL="$2"; shift 2 ;;
    --target-rate)   TARGET_RATE="$2"; shift 2 ;;
    --api-key)       API_KEY="$2"; shift 2 ;;
    --output)        OUTPUT="$2"; shift 2 ;;
    --ignore-eos)    IGNORE_EOS="--ignore-eos"; shift ;;
    --mode)          MODE="$2"; shift 2 ;;
    --max-context-tokens) MAX_CONTEXT_TOKENS="$2"; shift 2 ;;
    --context-safety-margin-tokens) CONTEXT_SAFETY_MARGIN_TOKENS="$2"; shift 2 ;;
    --prefix-caching-state) PREFIX_CACHING_STATE="$2"; shift 2 ;;
    --chunked-prefill) CHUNKED_PREFILL="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEM="$2"; shift 2 ;;
    --tensor-parallel-size) TP="$2"; shift 2 ;;
    -h|--help)       usage ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

if [[ -z "$OUTPUT" ]]; then
  TS=$(date +%Y%m%d_%H%M%S)
  OUTPUT="results/${BACKEND}_${PROFILE}_conc${CONCURRENCY}_${TS}.json"
fi

echo "Backend:     $BACKEND"
echo "URL:         $URL"
echo "Model:       $MODEL"
echo "Profile:     $PROFILE"
echo "Concurrency: $CONCURRENCY"
echo "Requests:    $NUM_REQUESTS"
echo "Output:      $OUTPUT"
echo ""

cd "$REPO_ROOT"

RUNNER_ARGS=(
  --url "$URL"
  --model "$MODEL"
  --backend "$BACKEND"
  --profile "$PROFILE"
  --concurrency "$CONCURRENCY"
  --num-requests "$NUM_REQUESTS"
  --warmup "$WARMUP"
  --arrival "$ARRIVAL"
  --target-rate "$TARGET_RATE"
  --api-key "$API_KEY"
  --prefix-caching-state "$PREFIX_CACHING_STATE"
  --chunked-prefill "$CHUNKED_PREFILL"
  --context-safety-margin-tokens "$CONTEXT_SAFETY_MARGIN_TOKENS"
  --output "$OUTPUT"
)

if [[ -z "$MODE" && "$PROFILE" == *multiturn* ]]; then
  MODE="multi-turn"
elif [[ -z "$MODE" ]]; then
  MODE="single-turn"
fi
RUNNER_ARGS+=(--mode "$MODE")

if [[ -n "$MAX_CONTEXT_TOKENS" ]]; then
  RUNNER_ARGS+=(--max-context-tokens "$MAX_CONTEXT_TOKENS")
elif [[ "$MODE" == "multi-turn" && -n "$MAX_MODEL_LEN" ]]; then
  RUNNER_ARGS+=(--max-context-tokens "$MAX_MODEL_LEN")
fi
if [[ -n "$MAX_MODEL_LEN" ]]; then
  RUNNER_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
fi
if [[ -n "$GPU_MEM" ]]; then
  RUNNER_ARGS+=(--gpu-memory-utilization "$GPU_MEM")
fi
if [[ -n "$TP" ]]; then
  RUNNER_ARGS+=(--tensor-parallel-size "$TP")
fi
if [[ -n "$IGNORE_EOS" ]]; then
  RUNNER_ARGS+=(--ignore-eos)
fi

OPENAI_API_KEY="$API_KEY" "$PYTHON" -m src.benchmark.runner \
  "${RUNNER_ARGS[@]}"
