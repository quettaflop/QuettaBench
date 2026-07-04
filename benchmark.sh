#!/usr/bin/env bash
# =============================================================================
# inference-benchmark — canonical benchmark runs
#
# This script documents all flags used in official benchmark runs.
# Edit the CONFIG section for your server, then run a target:
#
#   ./benchmark.sh stress               # prefill-heavy + decode-heavy sweep
#   ./benchmark.sh chatbot              # canonical natural chat sweep (ShareGPT)
#   ./benchmark.sh production           # all production profiles
#   ./benchmark.sh cross_validate       # cross-validation profile (random tokens)
#
# =============================================================================
set -euo pipefail

PYTHON="${PYTHON:-$(which python)}"
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

# =============================================================================
# CONFIG — edit these for your server
# =============================================================================
URL="${BENCH_URL:-http://localhost:8000/v1/chat/completions}"
MODEL="${BENCH_MODEL:-neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8}"
BACKEND="vllm"          # vllm | sglang | openai | trtllm
API_KEY="test"
WARMUP=5                # warmup requests (excluded from timing)
TIMEOUT=300             # per-request timeout in seconds

# Server flags (for reference — set when launching vLLM):
#   --enable-prefix-caching   → enables APC; TTFT will be lower on repeated prompts
#   --enable-chunked-prefill  → improves throughput under load
#   --tensor-parallel-size N  → for multi-GPU (e.g. tp=2 for 70B on 2x H100)
#   --max-model-len N         → context window limit

# =============================================================================
# CONCURRENCY LEVELS
# =============================================================================
CONC_STANDARD="1 10 20 40 80 120"
CONC_SHORT="1 10 20 40"

# =============================================================================
# HELPER
# =============================================================================
run() {
    local PROFILE="$1"
    local CONC="$2"
    local NREQ="${3:-200}"
    local EXTRA="${4:-}"   # e.g. "--ignore-eos"
    local LABEL="${5:-}"   # output filename label override

    local TAG="${LABEL:-$(echo "$MODEL" | tr '/' '_')_${PROFILE}_conc${CONC}}"
    local OUT="results/${TAG}.json"

    echo ""
    echo "=== profile=$PROFILE conc=$CONC nreq=$NREQ extra='$EXTRA' ==="
    OPENAI_API_KEY="$API_KEY" "$PYTHON" -m src.benchmark.runner \
        --url        "$URL" \
        --model      "$MODEL" \
        --backend    "$BACKEND" \
        --profile    "$PROFILE" \
        --concurrency "$CONC" \
        --num-requests "$NREQ" \
        --warmup     "$WARMUP" \
        --timeout    "$TIMEOUT" \
        --api-key    "$API_KEY" \
        --output     "$OUT" \
        $EXTRA
}

# =============================================================================
# TARGETS
# =============================================================================

stress() {
    # Synthetic stress test (random tokens, --ignore-eos auto-set by stress-test mode).
    # prefill-heavy: high ISL, stresses prefill throughput.
    # decode-heavy: high OSL, stresses decode throughput.
    # TPOT and throughput are the primary metrics; TTFT valid (random tokens, no shared prefix).
    echo "### prefill-heavy + decode-heavy sweep ###"
    echo "Mode:    stress-test (random tokens)"
    echo "Server:  ./scripts/launch_server.sh stress-test --model $MODEL"
    echo "Model:   $MODEL"
    echo "Server:  $URL"
    echo "Backend: $BACKEND"
    echo ""

    for CONC in $CONC_STANDARD; do
        NREQ=200
        [[ "$CONC" -eq 1 ]] && NREQ=50
        run "prefill-heavy" "$CONC" "$NREQ"
    done

    for CONC in $CONC_STANDARD; do
        NREQ=200
        [[ "$CONC" -eq 1 ]] && NREQ=50
        [[ "$CONC" -ge 80 ]] && NREQ=100
        run "decode-heavy" "$CONC" "$NREQ"
    done
}

chatbot() {
    # ShareGPT real text — different prompt every request → TTFT is valid regardless
    # of prefix caching state (no shared prefix across requests).
    # OSL hit rate ~91-93% on 8B/70B.
    echo "### natural chat single-turn sweep (ShareGPT real text) ###"
    echo "Mode:    single-turn"
    echo "Server:  ./scripts/launch_server.sh single-turn --model $MODEL"
    echo "Model:   $MODEL"
    echo "Server:  $URL"
    echo "NOTE: Real text, varied prompts — TTFT valid with or without prefix caching"
    echo ""

    for CONC in $CONC_STANDARD; do
        run "chat-singleturn" "$CONC" "200"
    done
}

production() {
    # New production profiles (real text, varied ISL/OSL).
    # Requires server max_model_len >= 17000 for coding-singleturn (ISL ~17K).
    # TTFT valid (varied prompts, no shared prefix).
    echo "### New production profiles ###"
    echo "Mode:    single-turn"
    echo "Server:  ./scripts/launch_server.sh single-turn --model $MODEL"
    echo "Model:   $MODEL"
    echo "Server:  $URL"
    echo ""

    for PROFILE in chat-singleturn coding-singleturn; do
        for CONC in $CONC_STANDARD; do
            run "$PROFILE" "$CONC" "200"
        done
    done
}

cross_validate() {
    # Random token workload for cross-validation with InferenceX.
    # --ignore-eos: required for FP8 models — without it OSL hit rate is ~37-51%.
    # Server should have prefix caching OFF for comparable results with InferenceX.
    # TPOT is the reliable comparison metric (TTFT differs due to arrival pattern).
    echo "### Cross-validation (random tokens, --ignore-eos) ###"
    echo "Mode:    stress-test"
    echo "Server:  ./scripts/launch_server.sh stress-test --model $MODEL"
    echo "Model:   $MODEL"
    echo "Server:  $URL"
    echo "Workload: random tokens ISL=1024 OSL=1024"
    echo "Flags:   --ignore-eos (auto-set by --mode stress-test; required for FP8)"
    echo "NOTE: Use TPOT for comparison with InferenceX, not TTFT"
    echo "      InferenceX uses --request-rate inf → TTFT not comparable"
    echo ""

    for CONC in 1 10 40 80; do
        run "random-1k" "$CONC" "100" "--ignore-eos --mode stress-test"
    done
}

# =============================================================================
# DISPATCH
# =============================================================================
TARGET="${1:-stress}"

case "$TARGET" in
    stress)            stress ;;
    chatbot)           chatbot ;;
    production)        production ;;
    cross_validate)    cross_validate ;;
    *)
        echo "Unknown target: $TARGET"
        echo "Available: stress | chatbot | production | cross_validate"
        exit 1
        ;;
esac
