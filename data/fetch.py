#!/usr/bin/env python3
"""Pull the gitignored trajectory datasets from R2 into data/ (`just fetch`) and check each against MANIFEST.json.

A file whose sha256 differs from the manifest is removed and the fetch fails: a replay on a file the manifest does
not describe is not comparable with one it does."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PROFILE = os.environ.get("AWS_PROFILE", "r2")
ENDPOINT = os.environ.get("R2_ENDPOINT", "")
BUCKET = os.environ.get("R2_BUCKET", "quettabench-prod")
PREFIX = os.environ.get("R2_WORKLOADS_PREFIX", "campaign-2026-10/workloads")
DEST = Path(__file__).resolve().parent
MANIFEST = DEST / "MANIFEST.json"

# Built files the profiles read, then the raw traces build_trajectories.py rebuilds them from.
DATASETS = [
    "coding_agent_prompts.jsonl",
    "swebench_native_trajectories.jsonl",
    "terminalbench_qwen_trajectories.jsonl",
    "deepresearch_trajectories.jsonl",
    "deepsearch_trajectories.jsonl",
]
TRACES = ["dr9k-v4flash.jsonl", "dsqa-v4flash.jsonl"]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest() -> dict:
    return json.loads(MANIFEST.read_text())["files"]


def verify(path: Path, expected: str) -> None:
    got = sha256_of(path)
    if got != expected:
        path.unlink()
        sys.exit(f"{path.name}: sha256 {got} is not the manifest's {expected}; removed")


def aws_s3(*args: str) -> None:
    subprocess.run(
        ["aws", "--profile", PROFILE, "--endpoint-url", ENDPOINT, "s3", *args],
        check=True,
    )


def main() -> None:
    if not ENDPOINT:
        sys.exit("R2_ENDPOINT is unset; run this as just fetch")
    files = manifest()
    for name in DATASETS + TRACES:
        aws_s3("cp", f"s3://{BUCKET}/{PREFIX}/{name}", str(DEST / name))
        verify(DEST / name, files[name]["sha256"])
    print(f"datasets in {DEST}, every sha256 as in {MANIFEST.name}")


if __name__ == "__main__":
    main()
