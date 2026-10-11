"""Fail on an em-dash in the given files (owner rule: write -- instead).

Usage: check_em_dash.py <file>...
"""

import sys

EM_DASH = "—"


def main() -> int:
    bad = 0
    for path in sys.argv[1:]:
        with open(path, encoding="utf-8", errors="replace") as f:
            for n, line in enumerate(f, 1):
                if EM_DASH in line:
                    print(f"{path}:{n}: em-dash; write --")
                    bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
