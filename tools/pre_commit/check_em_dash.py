"""Fail on an em-dash or en-dash in the given files (write -- instead)."""

import sys

DASHES = {"\u2014": "em-dash", "\u2013": "en-dash"}


def main() -> int:
    bad = 0
    for path in sys.argv[1:]:
        with open(path, encoding="utf-8", errors="replace") as f:
            for n, line in enumerate(f, 1):
                for dash, name in DASHES.items():
                    if dash in line:
                        print(f"{path}:{n}: {name}; write -- or 'to'")
                        bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
