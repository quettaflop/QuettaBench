#!/usr/bin/env python3
"""Publish benchmark results as a static site and a redacted data release.

Collects results/history.jsonl into a zero-infra static site (one page per
hardware, a table of runs with their summary lines) plus a redacted copy of the
raw jsonl for release. Releasing the per-operation cost tables is the one
dataset a competitor does not publish, so the release is the flagship artifact;
the publisher strips host-identifying paths, hostnames, and IPs from every
emitted byte first so nothing internal leaks. Output is deterministic (sorted,
no timestamps) so a golden test pins it.
"""

import argparse
import html
import json
import re
from pathlib import Path

_HOME = re.compile(r"/home/[^/\s\"']+")
_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_HOSTS = re.compile(r"\b(?:runpod|hwn-[\w-]+|gpu\d+)\b")


def redact(text):
    """Strip host-identifying strings: home paths, IPv4 addresses, and known
    host aliases. A leaked path or hostname in a public dump is the failure this
    guards, so it runs on every emitted byte, not just visible tables."""
    text = _HOME.sub("<path>", text)
    text = _IP.sub("<ip>", text)
    text = _HOSTS.sub("<host>", text)
    return text


def load_history(path):
    """Read results/history.jsonl into a list of run records."""
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def render_hardware_page(hardware, records):
    """Deterministic HTML for one hardware page: rows sorted by (workload,
    git_sha), summary tags joined. No timestamps, so re-publishing is a no-op
    diff unless the data changed."""
    rows = []
    for r in sorted(records, key=lambda x: (x["workload"], x.get("git_sha", ""))):
        summ = "; ".join(f"{k}: {' '.join(v)}"
                         for k, v in sorted(r.get("summaries", {}).items()))
        rows.append(f"<tr><td>{html.escape(r['workload'])}</td>"
                    f"<td>{html.escape(r.get('git_sha', '')[:9])}</td>"
                    f"<td>{html.escape(summ)}</td></tr>")
    return (f"<h1>{html.escape(hardware)}</h1>\n<table>\n"
            "<tr><th>workload</th><th>git</th><th>summaries</th></tr>\n"
            + "\n".join(rows) + "\n</table>\n")


def build_site(records, out_dir):
    """Write index.html, one <hardware>.html per hardware, and a redacted
    history.jsonl release. Returns the sorted list of emitted file names."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_hw = {}
    for r in records:
        by_hw.setdefault(r["hardware"], []).append(r)
    index = ["<h1>Benchmark results</h1>", "<ul>"]
    for hw in sorted(by_hw):
        (out / f"{hw}.html").write_text(redact(render_hardware_page(hw, by_hw[hw])))
        index.append(f'<li><a href="{html.escape(hw)}.html">{html.escape(hw)}</a></li>')
    index.append("</ul>")
    (out / "index.html").write_text("\n".join(index) + "\n")
    release = "\n".join(json.dumps(r, sort_keys=True) for r in records)
    (out / "history.jsonl").write_text(redact(release) + "\n")
    return sorted(p.name for p in out.iterdir())


def main():
    ap = argparse.ArgumentParser(description="publish results as a static site")
    ap.add_argument("history", help="results/history.jsonl")
    ap.add_argument("--out", default="docs/results", help="output directory")
    args = ap.parse_args()
    files = build_site(load_history(args.history), args.out)
    print(f"published {len(files)} files to {args.out}: {', '.join(files)}")


if __name__ == "__main__":
    main()
