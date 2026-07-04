#!/usr/bin/env bash
# Check if a vLLM/SGLang/OpenAI-compatible server is alive.
# Hits /v1/models and prints the available models.
#
# Usage:
#   ./scripts/check_server.sh [URL]
#
# Examples:
#   ./scripts/check_server.sh
#   ./scripts/check_server.sh http://localhost:8001
#
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
BASE_URL="${BASE_URL%/}"  # strip trailing slash

echo "Checking $BASE_URL ..."

RESPONSE=$(curl -sf "$BASE_URL/v1/models" -H "Authorization: Bearer test" 2>&1) || {
  echo "ERROR: Server not reachable at $BASE_URL"
  exit 1
}

echo "Server is up."
echo "$RESPONSE" | python3 -c "
import json, sys
d = json.load(sys.stdin)
models = [m['id'] for m in d.get('data', [])]
print('Available models:')
for m in models:
    print(f'  {m}')
" 2>/dev/null || echo "$RESPONSE"
