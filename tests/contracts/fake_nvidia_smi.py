#!/usr/bin/env python3
"""Stand-in nvidia-smi for power queries: GPU i draws 100 + 10*i watts.

`-i 0,2` limits the reply to those ids; without it the box has four GPUs.
"""

import sys

args = sys.argv[1:]
ids = args[args.index("-i") + 1].split(",") if "-i" in args else ["0", "1", "2", "3"]
print("\n".join(f"{100 + 10 * int(i):.2f}" for i in ids))
