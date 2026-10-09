"""Stand-in for a QuettaServe decode_loop bench: one LOOP line from the QW_* env.

FAKE_BENCH_EXIT sets the exit code; FAKE_BENCH_SILENT=1 prints no LOOP line.
"""

import os
import sys

if os.environ.get("FAKE_BENCH_SILENT") != "1":
    print(f"LOOP ctx={os.environ['QW_BENCH_PROMPT']} bs={os.environ['QW_BATCH']} "
          f"tp={os.environ['QW_WORLD']} kv=fp16 ms_per_step=3.500 k=100", flush=True)
print("bench chatter", file=sys.stderr, flush=True)
sys.exit(int(os.environ.get("FAKE_BENCH_EXIT", "0")))
