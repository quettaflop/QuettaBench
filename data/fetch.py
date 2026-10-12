#!/usr/bin/env python3
"""Pull the large trajectory datasets from R2.

They are gitignored. Needs an aws profile with R2 credentials (default
profile name: r2). Override with AWS_PROFILE, R2_ENDPOINT, R2_BUCKET.

The GCS pulls and the raw-capture builders stay separate scripts:
    python3 data/download_agent_trajectories.py
    python3 data/download_coding_prompts.py
    python3 data/build_chat_trajectories.py
    python3 data/build_claweval_trajectories.py --traces DIR
"""

import os
import subprocess
from pathlib import Path

PROFILE = os.environ.get("AWS_PROFILE", "r2")
ENDPOINT = os.environ.get(
    "R2_ENDPOINT",
    "https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com",
)
BUCKET = os.environ.get("R2_BUCKET", "agent-bench")
DEST = Path(__file__).resolve().parent

DATASETS = [
    "coding_agent_prompts.jsonl",
    "osworld_trajectories.jsonl",
    "swebench_trajectories.jsonl",
    "terminalbench_trajectories.jsonl",
]


def aws_s3(*args: str) -> None:
    subprocess.run(
        ["aws", "--profile", PROFILE, "--endpoint-url", ENDPOINT, "s3", *args],
        check=True,
    )


def main() -> None:
    for name in DATASETS:
        aws_s3("cp", f"s3://{BUCKET}/data/{name}", str(DEST / name))
    print(f"datasets in {DEST}")


if __name__ == "__main__":
    main()
