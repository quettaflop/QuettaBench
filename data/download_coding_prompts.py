#!/usr/bin/env python3
"""Download coding-singleturn prompts from GCS and write data/coding_agent_prompts.jsonl (system prompt, task description, osl estimate, deduplicated by task name)."""

import json
import os
import re
import sys
from pathlib import Path

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.expanduser("~/gcp-key.json")

try:
    from google.cloud import storage
except ImportError:
    print("Installing google-cloud-storage...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "google-cloud-storage", "-q"], check=True)
    from google.cloud import storage

BUCKET_NAME = "sequrity-experiments"
PREFIXES = [
    "harbor/task_analysis/swebench-verified-sequrity-codex-new23-500/",
    "harbor/task_analysis/swebench-verified-sequrity-codex-new27-500/",
    "harbor/task_analysis/swebench-verified-sequrity-codex-new38-100/",
]

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "coding_agent_prompts.jsonl"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def extract_system_prompt(content: str) -> str:
    """System prompt from _pllm_full_prompt.md: the code block of Message 0, else the first block."""
    msg0_match = re.search(r"## Message 0:.*?\n`{3,4}\n(.*?)\n`{3,4}", content, re.DOTALL)
    if msg0_match:
        return msg0_match.group(1).strip()
    blocks = re.findall(r"`{3,4}\n(.*?)\n`{3,4}", content, re.DOTALL)
    if blocks:
        return blocks[0].strip()
    return ""


def extract_task_description(content: str) -> str:
    """Text between ## Task Description and the next ## header, empty if absent."""
    match = re.search(r"## Task Description\s*\n(.*?)(?=\n## |\Z)", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def estimate_osl_tokens(content: str) -> int:
    """Output tokens as chars / 3.8 of the Program block, else the mean code block, else 200."""
    program_match = re.search(
        r"## (?:PLLM )?Program.*?\n`{3,4}[^\n]*\n(.*?)\n`{3,4}",
        content, re.DOTALL | re.IGNORECASE
    )
    if program_match:
        code = program_match.group(1)
        return max(50, int(len(code) / 3.8))

    blocks = re.findall(r"`{3,4}[^\n]*\n(.*?)\n`{3,4}", content, re.DOTALL)
    if blocks:
        avg_len = sum(len(b) for b in blocks) / len(blocks)
        return max(50, int(avg_len / 3.8))

    return 200


def list_blobs(client: storage.Client, prefix: str):
    """All blobs under a prefix."""
    bucket = client.bucket(BUCKET_NAME)
    return list(bucket.list_blobs(prefix=prefix))


def download_blob_text(blob) -> str:
    """Blob content as UTF-8 text."""
    return blob.download_as_text(encoding="utf-8")


def main():
    print(f"Connecting to GCS bucket: {BUCKET_NAME}")
    client = storage.Client()

    seen_tasks: dict[str, dict] = {}
    run_system_prompts: dict[str, str] = {}

    for prefix in PREFIXES:
        run_name = prefix.rstrip("/").split("/")[-1]
        print(f"\n--- Run: {run_name} ---")

        blobs = list_blobs(client, prefix)
        print(f"  Found {len(blobs)} blobs")

        pllm_blob = None
        task_blobs = []
        for blob in blobs:
            fname = blob.name.split("/")[-1]
            if fname == "_pllm_full_prompt.md":
                pllm_blob = blob
            elif fname.endswith(".md") and not fname.startswith("_"):
                task_blobs.append(blob)

        system_prompt = ""
        if pllm_blob:
            print(f"  Downloading system prompt from: {pllm_blob.name}")
            pllm_content = download_blob_text(pllm_blob)
            system_prompt = extract_system_prompt(pllm_content)
            print(f"  System prompt length: {len(system_prompt):,} chars")
        else:
            print("  WARNING: _pllm_full_prompt.md not found for this run")

        run_system_prompts[prefix] = system_prompt

        new_count = 0
        skip_count = 0
        for blob in task_blobs:
            fname = blob.name.split("/")[-1]
            task_name = fname.replace(".md", "")

            if task_name in seen_tasks:
                skip_count += 1
                continue

            content = download_blob_text(blob)
            user_msg = extract_task_description(content)
            if not user_msg:
                continue

            osl_tokens = estimate_osl_tokens(content)

            seen_tasks[task_name] = {
                "system": system_prompt,
                "user": user_msg,
                "osl_tokens": osl_tokens,
                "task_name": task_name,
                "run": run_name,
            }
            new_count += 1

        print(f"  New tasks added: {new_count}, duplicates skipped: {skip_count}")

    print(f"\nWriting {len(seen_tasks)} unique prompts to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for entry in seen_tasks.values():
            record = {
                "system": entry["system"],
                "user": entry["user"],
                "osl_tokens": entry["osl_tokens"],
            }
            f.write(json.dumps(record) + "\n")

    print(f"\nDone. Saved {len(seen_tasks)} unique prompts.")

    if seen_tasks:
        osl_vals = [e["osl_tokens"] for e in seen_tasks.values()]
        user_lens = [len(e["user"]) for e in seen_tasks.values()]
        sys_lens = [len(e["system"]) for e in seen_tasks.values()]
        print(f"OSL tokens  -- min: {min(osl_vals)}, max: {max(osl_vals)}, avg: {sum(osl_vals)//len(osl_vals)}")
        print(f"User chars  -- min: {min(user_lens)}, max: {max(user_lens)}, avg: {sum(user_lens)//len(user_lens)}")
        print(f"System chars -- min: {min(sys_lens)}, max: {max(sys_lens)}, avg: {sum(sys_lens)//len(sys_lens)}")


if __name__ == "__main__":
    main()
