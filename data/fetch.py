#!/usr/bin/env python3
"""Pull the gitignored trajectory datasets from R2 into data/ (`just fetch`) and check each against MANIFEST.json.

A file whose sha256 differs from the manifest is removed and the fetch fails: a replay on a file the manifest does
not describe is not comparable with one it does. `--check` checks every manifest file already in data/, the ones
build_trajectories.py builds from the raw traces included, and fetches nothing."""

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

# Files the profiles read as fetched, then the raw traces and recorded agent runs build_trajectories.py builds the
# deep research, deep search, SWE-bench and Terminal-bench files from after the fetch (just fetch).
DATASETS = ["coding_agent_prompts.jsonl"]
TRACES = ["dr9k-v4flash.jsonl", "dsqa-v4flash.jsonl"]
# Directory under the workloads prefix -> directory in data/raw/.
RUNS = {
    "sources/swebench-verified_qwen38-flash-next_native_2026-09-10/qwen_native_c8": "swebench_native_c8",
    "sources/terminalbench-2-1_qwen38-flash-next_2026-08-27/jobs": "terminalbench_jobs",
}


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


def check() -> None:
    for name, entry in manifest().items():
        if not (DEST / name).exists():
            sys.exit(f"{name}: not in {DEST}; run just fetch")
        verify(DEST / name, entry["sha256"])
    print(f"every file in {MANIFEST.name} is in {DEST} with its sha256")


def main() -> None:
    if sys.argv[1:] == ["--check"]:
        check()
        return
    if not ENDPOINT:
        sys.exit("R2_ENDPOINT is unset; run this as just fetch")
    files = manifest()
    for name in DATASETS + TRACES:
        aws_s3("cp", f"s3://{BUCKET}/{PREFIX}/{name}", str(DEST / name))
        verify(DEST / name, files[name]["sha256"])
    # A recorded run is many files; the sha256 of what it builds, checked by --check, covers them.
    for key, name in RUNS.items():
        aws_s3("sync", f"s3://{BUCKET}/{PREFIX}/{key}", str(DEST / "raw" / name))
    print(f"datasets in {DEST}, every sha256 as in {MANIFEST.name}; recorded runs in {DEST / 'raw'}")


if __name__ == "__main__":
    main()
