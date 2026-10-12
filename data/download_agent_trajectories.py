#!/usr/bin/env python3
"""Download harbor job trajectories from GCS and write per-source session JSONL (growing message history per turn, osl_tokens estimate).

Usage: python data/download_agent_trajectories.py [--source swebench|terminalbench|all]
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.expanduser("~/gcp-key.json")

try:
    from google.cloud import storage
except ImportError:
    print("Installing google-cloud-storage...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "google-cloud-storage", "-q"], check=True)
    from google.cloud import storage

BUCKET_NAME = "sequrity-experiments"

RUN_PREFIXES = {
    "swebench": [
        "harbor/jobs/codex-gpt5.2-swebench-verified-2026-02-16-1700/",
        "harbor/jobs/codex-gpt5.2-swebench-verified-2026-02-16-1800/",
    ],
    "terminalbench": [
        "harbor/jobs/codex-gpt5.2-terminal-bench-2026-02-16-2300/",
        "harbor/jobs/codex-gpt5.2-terminal-bench-2026-02-17-0955/",
        "harbor/jobs/codex-gpt5.2-terminal-bench-2026-02-17-1446/",
    ],
}

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"


def estimate_tokens(text: str) -> int:
    """Rough token estimate: words * 1.35."""
    return max(1, int(len(text.split()) * 1.35))


def extract_turns_from_trajectory(trajectory: dict) -> list[dict]:
    """Turns of an ATIF trajectory: input messages so far plus an osl estimate of the agent's raw_arguments."""
    steps = trajectory.get("steps", [])
    if not steps:
        return []

    messages: list[dict] = []
    turns: list[dict] = []

    for step in steps:
        source = step.get("source", "")

        if source == "system":
            content = step.get("message", "")
            if content:
                messages.append({"role": "system", "content": content})

        elif source == "user":
            content = step.get("message", "")
            if content:
                messages.append({"role": "user", "content": content})

        elif source == "agent":
            # raw_arguments is the real model output, message is a stub
            raw_args = step.get("extra", {}).get("raw_arguments", "")
            if not raw_args:
                raw_args = step.get("message", "")
            if not raw_args:
                continue

            osl_est = estimate_tokens(raw_args)

            turns.append({
                "turn_idx": len(turns),
                "messages": list(messages),
                "osl_tokens": osl_est,
            })

            messages.append({"role": "assistant", "content": raw_args})

            obs = step.get("observation", {}).get("results", "")
            if isinstance(obs, list) and obs:
                obs_content = obs[0].get("content", "") if isinstance(obs[0], dict) else str(obs[0])
            elif isinstance(obs, str):
                obs_content = obs
            else:
                obs_content = ""
            if obs_content:
                messages.append({"role": "user", "content": obs_content})

    return turns


def list_task_prefixes(client: storage.Client, run_prefix: str) -> list[str]:
    """Task directory prefixes under a run prefix."""
    bucket = client.bucket(BUCKET_NAME)
    it = bucket.list_blobs(prefix=run_prefix, delimiter="/")
    list(it)  # prefixes fill only once the iterator is consumed
    return [p for p in sorted(it.prefixes) if not p.endswith("config.json")]


def download_trajectory(client: storage.Client, task_prefix: str) -> dict | None:
    """Parsed trajectory.json of a task, or None if the download fails."""
    bucket = client.bucket(BUCKET_NAME)
    blob_path = task_prefix + "agent/trajectory.json"
    blob = bucket.blob(blob_path)
    try:
        return json.loads(blob.download_as_text())
    except Exception as e:
        print(f"  WARN: Could not download {blob_path}: {e}")
        return None


def process_run(client: storage.Client, run_prefix: str, source: str) -> list[dict]:
    """Session dicts for the first copy of each task in a run, skipping tasks under 2 turns."""
    run_name = run_prefix.rstrip("/").split("/")[-1]
    print(f"\n--- Run: {run_name} ---")

    task_prefixes = list_task_prefixes(client, run_prefix)
    print(f"  Found {len(task_prefixes)} tasks")

    sessions = []
    seen_tasks: set[str] = set()

    for task_prefix in task_prefixes:
        parts = task_prefix.rstrip("/").split("/")
        task_name = parts[-1]

        # the same task can appear in several runs
        base_task = task_name.rsplit("__", 1)[0] if "__" in task_name else task_name
        if base_task in seen_tasks:
            continue
        seen_tasks.add(base_task)

        trajectory = download_trajectory(client, task_prefix)
        if trajectory is None:
            continue

        turns = extract_turns_from_trajectory(trajectory)
        if len(turns) < 2:
            continue

        sessions.append({
            "session_id": task_name,
            "source": source,
            "run": run_name,
            "num_turns": len(turns),
            "turns": turns,
        })

    print(f"  Extracted {len(sessions)} sessions (min 2 turns)")
    if sessions:
        turn_counts = [s["num_turns"] for s in sessions]
        print(f"  Turn distribution: min={min(turn_counts)}, median={sorted(turn_counts)[len(turn_counts)//2]}, max={max(turn_counts)}")

    return sessions


def main():
    parser = argparse.ArgumentParser(description="Download agent trajectories from GCS")
    parser.add_argument("--source", choices=["swebench", "terminalbench", "all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sources = list(RUN_PREFIXES.keys()) if args.source == "all" else [args.source]

    print(f"Connecting to GCS bucket: {BUCKET_NAME}")
    client = storage.Client()

    for source in sources:
        print(f"\n{'='*60}")
        print(f"Processing {source} trajectories")
        print(f"{'='*60}")

        all_sessions = []
        for run_prefix in RUN_PREFIXES[source]:
            sessions = process_run(client, run_prefix, source)
            all_sessions.extend(sessions)

        seen = set()
        unique_sessions = []
        for s in all_sessions:
            if s["session_id"] not in seen:
                seen.add(s["session_id"])
                unique_sessions.append(s)

        output_path = args.output_dir / f"{source}_trajectories.jsonl"
        print(f"\nWriting {len(unique_sessions)} unique sessions to {output_path}")
        with open(output_path, "w") as f:
            for session in unique_sessions:
                f.write(json.dumps(session) + "\n")

        if unique_sessions:
            turn_counts = [s["num_turns"] for s in unique_sessions]
            print(f"Total sessions: {len(unique_sessions)}")
            print(f"Turn counts: min={min(turn_counts)}, median={sorted(turn_counts)[len(turn_counts)//2]}, max={max(turn_counts)}")
            print("Turn buckets:")
            for lo, hi, label in [(2, 5, "short (2-5)"), (5, 10, "medium (5-10)"), (10, 20, "long (10-20)"), (20, 100, "xl (20+)")]:
                count = sum(1 for t in turn_counts if lo <= t < hi)
                print(f"  {label}: {count} sessions")

    print("\nDone.")


if __name__ == "__main__":
    main()
