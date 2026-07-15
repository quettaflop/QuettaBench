# profiling/probes/gemm_slice_probe.py
"""Row-parallel GEMM slices missing from the H100 table (o/down at tp2/tp4:
K in {1024, 2048, 3584, 7168}, N=4096), plus on-table anchor shapes to check
harness-vs-NCU fidelity. CUDA-event timed torch.matmul, bf16. latency_us is the
min over reps, matching the table's min-reduce semantics.

  CUDA_VISIBLE_DEVICES=0 python gemm_slice_probe.py --out gemm_slices_H100.csv
"""

import argparse
import csv
import statistics as st

import torch

MS = [1, 64, 116, 256, 500, 1024, 2000, 4096, 8192]  # the table's M axis
SLICES = [(4096, k) for k in (1024, 2048, 3584, 7168)]  # (N, K), o/down at tp4/tp2
ANCHORS = [(1024, 4096), (4096, 4096), (14336, 4096), (4096, 14336)]  # on-table
REPS, WARMUP = 100, 15


def time_gemm(m: int, n: int, k: int) -> tuple[float, float]:
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(WARMUP):
        a @ b
    torch.cuda.synchronize()
    times = []
    for _ in range(REPS):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        a @ b
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)  # us
    return min(times), st.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gemm_slices_H100.csv")
    args = ap.parse_args()

    rows = []
    for role, shapes in (("anchor", ANCHORS), ("slice", SLICES)):
        for n, k in shapes:
            for m in MS:
                lo, med = time_gemm(m, n, k)
                rows.append(dict(M=m, N=n, K=k, dtype_bytes=2,
                                 latency_us=round(lo, 2), latency_med_us=round(med, 2),
                                 role=role, source="torch_matmul_cuda_events"))
                print(f"{role} M={m:>5} N={n:>5} K={k:>5}: min {lo:8.2f} us  med {med:8.2f}", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
