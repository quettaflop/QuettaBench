# scripts/fetch_data.sh
# Pull the large trajectory datasets from R2 (they are gitignored).
# Needs an aws profile with R2 credentials (default profile name: r2).

set -euo pipefail

PROFILE="${AWS_PROFILE:-r2}"
ENDPOINT="${R2_ENDPOINT:-https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com}"
BUCKET="${R2_BUCKET:-agent-bench}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/data"

for f in coding_agent_prompts.jsonl osworld_trajectories.jsonl \
         swebench_trajectories.jsonl terminalbench_trajectories.jsonl; do
    aws --profile "$PROFILE" --endpoint-url "$ENDPOINT" \
        s3 cp "s3://$BUCKET/data/$f" "$DEST/$f"
done
echo "datasets in $DEST"
