# profiling/probes/cached_prefill_steps_v3.py
"""Measure cached-prefill step time with independent decode reference.

After prime (prefix warmup), measures pure decode cost from prefix-only
generation.  Then cached prefill = t1(prefix+suffix) - decode_ref.

Raw token IDs, distinct token sets, bucket warmup, scheduler hook.
"""

from __future__ import annotations

import argparse, csv, os, statistics, sys, time
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
for p in [e for e in sys.path if e == _SCRIPT_DIR]:
    sys.path.remove(p)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/data48/kevinlau/models/Llama-3.1-8B-Instruct")
    p.add_argument("--U-values", nargs="*", type=int, default=[64, 128, 256, 512, 1024])
    p.add_argument("--P-values", nargs="*", type=int, default=[512, 1024, 2048, 4096, 8192])
    p.add_argument("--measure-runs", type=int, default=3)
    p.add_argument("--output", type=Path,
                   default=Path("profile_data/results/cached_prefill_v3_H100.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    from vllm.v1.core.sched.scheduler import Scheduler
    sched_log: list[int] = []
    orig_sched = Scheduler.schedule
    def traced_sched(self):
        out = orig_sched(self)
        st = out.num_scheduled_tokens
        sched_log.append(sum(st.values()) if isinstance(st, dict) else 0)
        return out
    Scheduler.schedule = traced_sched

    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, max_model_len=32768, gpu_memory_utilization=0.70,
              max_num_seqs=8, max_num_batched_tokens=16384,
              enable_prefix_caching=True, seed=0)
    params1 = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    params32 = SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True)

    TID = 279
    def mk(n): return [TID] * n

    def timed(ids, params=params1):
        sched_log.clear()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        llm.generate({"prompt_token_ids": ids}, params, use_tqdm=False)
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e)

    # Warm all bucket sizes
    print("Warming CUDA graph buckets...")
    for sz in sorted(set(args.U_values) | set(args.P_values)):
        for _ in range(2):
            timed(mk(sz)); torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "U", "P", "prefill_ms", "decode_ref_ms", "t1_cached_ms",
            "scheduled_tokens", "cache_hit",
        ])
        writer.writeheader()

        for P in args.P_values:
            prefix = mk(P)

            # Prime cache: prefix + 1 suffix token
            timed(prefix + [TID])
            torch.cuda.empty_cache()

            for U in args.U_values:
                suffix = mk(U)

                # Delta method: t1 = prefill + 1 decode, t32 = prefill + 32 decode
                t1_samples, t32_samples, sched_samples = [], [], []
                for _ in range(args.measure_runs):
                    t1 = timed(prefix + suffix, params1)
                    t1_samples.append(t1)
                    sched_samples.append([x for x in sched_log if x > 0])
                    torch.cuda.empty_cache()

                    t32 = timed(prefix + suffix, params32)
                    t32_samples.append(t32)
                    torch.cuda.empty_cache()

                t1_med = statistics.median(t1_samples)
                t32_med = statistics.median(t32_samples)
                decode_ref = (t32_med - t1_med) / 31
                prefill = max(0.001, t1_med - decode_ref)
                sched = sched_samples[0]
                hit = sum(sched) < P + U

                writer.writerow({
                    "U": U, "P": P,
                    "prefill_ms": f"{prefill:.4f}",
                    "decode_ref_ms": f"{decode_ref:.4f}",
                    "t1_cached_ms": f"{t1_med:.4f}",
                    "scheduled_tokens": ",".join(str(x) for x in sched),
                    "cache_hit": str(hit),
                })
                print(f"P={P:>5d} U={U:>4d}: prefill={prefill:.2f}ms "
                      f"d_ref={decode_ref:.1f}ms sched={sched} hit={hit}")

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
