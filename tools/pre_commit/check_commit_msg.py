"""commit-msg hook: the owner's commit rules (AGENTS.md, "How to change things").

Usage: check_commit_msg.py <message-file>
"""

import re
import sys

MAX_SUBJECT = 72
SUBJECT = re.compile(r"^[a-z][a-z0-9_./-]*: \S")
# Messages git writes itself, left as git writes them.
GIT_SUBJECT = re.compile(r"^(Merge |Revert \"|fixup! |squash! )")
# Any "<Word>-by:" trailer or "generated with/by" line: commits carry no authorship beyond the author field.
ATTRIBUTION = re.compile(r"^\s*[A-Za-z]+(?:-[A-Za-z]+)*-by:|generated (?:with|by)\b", re.IGNORECASE)
DASHES = ("\u2014", "\u2013")


def problems(message: str) -> list[str]:
    lines = [line for line in message.splitlines() if not line.startswith("#")]
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return ["empty message"]
    found = []
    subject = lines[0]
    if len(subject) > MAX_SUBJECT:
        found.append(f"subject is {len(subject)} chars, limit {MAX_SUBJECT}")
    if not SUBJECT.match(subject) and not GIT_SUBJECT.match(subject):
        found.append("subject is not '<area>: <change>' (lowercase area, colon, space)")
    for n, line in enumerate(lines, 1):
        if ATTRIBUTION.search(line):
            found.append(f"line {n}: attribution trailer: {line.strip()}")
        if any(d in line for d in DASHES):
            found.append(f"line {n}: em-dash or en-dash; write --")
    return found


def main() -> int:
    with open(sys.argv[1], encoding="utf-8") as f:
        found = problems(f.read())
    for p in found:
        print(f"commit message: {p}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
