#!/usr/bin/env python3
"""Pull the gitignored trajectory datasets from R2 into data/ (`just fetch`)."""

import os
import subprocess
import sys
from pathlib import Path

PROFILE = os.environ.get("AWS_PROFILE", "r2")
ENDPOINT = os.environ.get("R2_ENDPOINT", "")
BUCKET = os.environ.get("R2_BUCKET", "agent-bench")
DEST = Path(__file__).resolve().parent

DATASETS = [
    "coding_agent_prompts.jsonl",
    "osworld_trajectories.jsonl",
    "swebench_trajectories.jsonl",
    "terminalbench_trajectories.jsonl",
]
# Raw traces that build_trajectories.py converts: key under the bucket -> name in data/.
TRACES = {
    "traces/deepresearch-9k_DeepSeek-V4-Flash-0731_rtx6000pro/dr9k-v4flash.jsonl": "dr9k-v4flash.jsonl",
}


def aws_s3(*args: str) -> None:
    subprocess.run(
        ["aws", "--profile", PROFILE, "--endpoint-url", ENDPOINT, "s3", *args],
        check=True,
    )


def main() -> None:
    if not ENDPOINT:
        sys.exit("R2_ENDPOINT is unset; run this as just fetch")
    for name in DATASETS:
        aws_s3("cp", f"s3://{BUCKET}/traces/replay-data/{name}", str(DEST / name))
    for key, name in TRACES.items():
        aws_s3("cp", f"s3://{BUCKET}/{key}", str(DEST / name))
    print(f"datasets in {DEST}")


if __name__ == "__main__":
    main()
