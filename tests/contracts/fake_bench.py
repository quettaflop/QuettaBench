#!/usr/bin/env python3
"""Stand-in decode_loop bench: prints one LOOP line from the QW_* env."""

import os
import subprocess
import sys
import time

if os.environ.get("FAKE_BENCH_CHILD"):
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    with open(os.environ["FAKE_BENCH_CHILD"], "w") as fh:
        fh.write(str(child.pid))

if os.environ.get("FAKE_BENCH_SILENT") != "1":
    print(
        f"LOOP ctx={os.environ['QW_BENCH_PROMPT']} bs={os.environ['QW_BATCH']} "
        f"tp={os.environ['QW_WORLD']} kv=fp16 ms_per_step=3.500 k=100",
        flush=True,
    )
print("bench chatter", file=sys.stderr, flush=True)
time.sleep(float(os.environ.get("FAKE_BENCH_SLEEP", "0")))
sys.exit(int(os.environ.get("FAKE_BENCH_EXIT", "0")))
