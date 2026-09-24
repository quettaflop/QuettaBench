#!/usr/bin/env python3
"""Turn benchmarks/CHANGELOG.md entries into a benchmark job matrix.

Mirrors the changelog-triggered trigger model (a change names the workloads and
hardware it affects) without a benchmarking fleet. A new entry emits one job per
(workload, hardware) pair; a malformed entry hard-fails so a typo never silently
skips a benchmark. Until self-hosted GPU runners exist, --dry-run prints the
exact commands as a checklist instead of dispatching, and each real run appends
its summary lines plus input and git shas to results/history.jsonl.

Entry grammar (a markdown list item under a version header):
  - <free text> -- workloads: <a, b> -- hardware: <H200, RTXPRO6000>
"""

import argparse
import json
import re
import sys

ENTRY_RE = re.compile(
    r"workloads:\s*([^|]+?)\s*(?:--|\|)\s*hardware:\s*([^|]+?)\s*$", re.I)
SUMMARY_TAGS = ("META", "WORKLOADSUM", "SERVESUM", "SLASUM",
                "ENERGYSUM", "COSTSUM", "FIDELITY")


def parse_changelog(text):
    """[{workload, hardware}] for every (workload, hardware) pair named by the
    entries. A list item that mentions workloads but does not parse cleanly
    raises, rather than being skipped and quietly dropping a benchmark."""
    jobs = []
    in_entries = False
    for line in text.splitlines():
        if line.startswith("## "):  # entries live under a version header, not the preamble
            in_entries = True
            continue
        if not in_entries:
            continue
        s = line.strip()
        if not s.startswith("- ") or "workloads:" not in s.lower():
            continue
        m = ENTRY_RE.search(s)
        if not m:
            raise ValueError(
                f"malformed changelog entry, need 'workloads: .. -- hardware: ..': {s!r}")
        wls = [w.strip() for w in m.group(1).split(",") if w.strip()]
        hws = [h.strip() for h in m.group(2).split(",") if h.strip()]
        if not wls or not hws:
            raise ValueError(f"entry names no workload or hardware: {s!r}")
        for w in wls:
            for h in hws:
                jobs.append({"workload": w, "hardware": h})
    return jobs


def commands(job):
    """The deterministic command checklist for one job. Fixed given (workload,
    hardware) so a dry run byte-matches a golden and a reviewer can diff it."""
    w, h = job["workload"], job["hardware"]
    return [
        f"bash scripts/crossval/xval.sh {w} $WEIGHTS_{w}",
        (f"python3 scripts/crossval/trace_serve.py traces/{w}.jsonl --model $WEIGHTS_{w} "
         f"--sla-ttft-ms 2000 --sla-tpot-ms 100 --energy --gpu-cost-hr $COST_{h} "
         f"--json results/{w}-{h}.json"),
    ]


def history_record(job, summary_lines, git_sha, input_sha):
    """One results/history.jsonl row: the job, the git and input shas for
    provenance, and the summary lines the run emitted, keyed by their tag."""
    summaries = {}
    for line in summary_lines:
        tag = line.split(" ", 1)[0]
        if tag in SUMMARY_TAGS:
            summaries.setdefault(tag, []).append(line.rstrip("\n"))
    return {"workload": job["workload"], "hardware": job["hardware"],
            "git_sha": git_sha, "input_sha256": input_sha, "summaries": summaries}


def main():
    ap = argparse.ArgumentParser(description="changelog -> benchmark job matrix")
    ap.add_argument("changelog", help="benchmarks/CHANGELOG.md")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the command checklist instead of a job matrix")
    args = ap.parse_args()
    jobs = parse_changelog(open(args.changelog).read())
    if not jobs:
        sys.exit("no benchmark entries found")
    if args.dry_run:
        for job in jobs:
            print(f"# {job['workload']} on {job['hardware']}")
            for c in commands(job):
                print(c)
    else:
        print(json.dumps({"include": jobs}))


if __name__ == "__main__":
    main()
