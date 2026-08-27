#!/usr/bin/env python3
"""Gated-DeltaNet (Qwen3.5) kernel-cost grid on vLLM's vendored FLA ops (the same
kernels vllm/sglang serve Qwen3.5 with):

  prefill: fla.ops.chunk_gated_delta_rule        (q,k,v,g,beta -> o)   seq grid
  decode:  fla.ops.fused_recurrent_gated_delta_rule (T=1 + state)      batch grid

Method follows the serving reality (see kernel_data/PROVENANCE.md): decode is
CUDA-graphed -> NCU faithful (`--ncu` -> ncu/gdn/…_decode.csv); prefill is eager ->
always cuda_event -> cuda_event/gdn/…_prefill.csv (the multi-kernel dispatch floor
is real). The delta-rule core is the GDN analog of the flash-attention grid;
projections are GEMMs (gemm table) and conv/gating are elementwise. Emits per-call
median us and us x n_linear (all-GDN-layers convention).

  python gdn_grid_probe.py --hv 32 --dk 128 --dv 128 --n-linear 24 --tag gdn-9b \
      --mode smoke --out-dir <dir>
"""
from __future__ import annotations
import argparse, csv, os, statistics as st, traceback
from pathlib import Path
import torch
import triton
from vllm.model_executor.layers.fla.ops import (
    chunk_gated_delta_rule, fused_recurrent_gated_delta_rule,
)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ncu import ncu_op_us

# chunk_gated_delta_rule's solve_tril merge kernel autotunes on a fresh host (no
# Triton disk cache) and needs a scratch allocator; vLLM sets one at model init, a
# bare probe must too. Without this the prefill loop crashes ("no allocator was
# set") on fresh H100/3090 and writes 0 prefill rows.
triton.set_allocator(lambda size, alignment, stream: torch.empty(size, dtype=torch.int8, device="cuda"))

REPS, WARMUP = 30, 5
# cap at 80: the generic fused_recurrent decode proxy hits an illegal-memory bug at
# B>=120 (ssm_state_indices path) that corrupts the CUDA context and would kill the
# prefill loop too. The faithful fused_sigmoid_gating_delta_rule_update decode (which
# lifts this cap) is a follow-up -> then re-profile decode via gdn-ncu.
BATCH_FULL = [1, 2, 4, 8, 16, 32, 40, 64, 80]
SEQ_FULL   = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
BATCH_SMOKE = [1, 32, 256]
SEQ_SMOKE   = [128, 2048]
# NCU is slow -> curated grids; the loader interpolates. Decode capped at 80 (the
# fused_recurrent proxy hits a synthetic-input bug at B>=120 -- same limit as the
# cuda_event grid). NCU_REPS invocations profiled per point; min = warm steady state.
BATCH_NCU = [1, 8, 32, 80]
NCU_REPS = 5


def _ncu_decode_inner(B, H, K, V) -> str:
    return (
        "import torch;"
        "from vllm.model_executor.layers.fla.ops import fused_recurrent_gated_delta_rule as R;"
        "d=torch.device('cuda');t=torch.bfloat16;"
        f"B,H,K,V={B},{H},{K},{V};s=1.0/(K**0.5);"
        "q=torch.randn(B,1,H,K,device=d,dtype=t);k=torch.randn(B,1,H,K,device=d,dtype=t);"
        "v=torch.randn(B,1,H,V,device=d,dtype=t);"
        "g=-torch.rand(B,1,H,device=d,dtype=torch.float32);beta=torch.rand(B,1,H,device=d,dtype=t);"
        "st=torch.randn(B,H,V,K,device=d,dtype=torch.float32);"
        "f=lambda:R(q,k,v,g,beta,scale=s,initial_state=st,inplace_final_state=False,use_qk_l2norm_in_kernel=True);"
        f"[f() for _ in range({NCU_REPS})];torch.cuda.synchronize()"
    )


def time_call(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)  # ms -> us
    return st.median(ts)


def mk(B, T, H, K, V, dev, dt):
    q = torch.randn(B, T, H, K, device=dev, dtype=dt)
    k = torch.randn(B, T, H, K, device=dev, dtype=dt)
    v = torch.randn(B, T, H, V, device=dev, dtype=dt)
    g = -torch.rand(B, T, H, device=dev, dtype=torch.float32)   # log-space forget gate (<=0)
    beta = torch.rand(B, T, H, device=dev, dtype=dt)            # betas in (0,1)
    return q, k, v, g, beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hv", type=int, default=32)     # linear_num_value_heads
    ap.add_argument("--dk", type=int, default=128)    # linear_key_head_dim
    ap.add_argument("--dv", type=int, default=128)    # linear_value_head_dim
    ap.add_argument("--n-linear", type=int, default=24)
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--gpu-label", required=True, help="device label matching device_spec YAML name, e.g. A100/H100")
    ap.add_argument("--geometry-tag", default="9b", help="GDN head geometry tag, e.g. 9b (H_v=32) / 27b (H_v=48)")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--max-mem-gb", type=float, default=30.0)
    ap.add_argument("--ncu", action="store_true", help="NCU per-kernel timing (the standard); curated grid")
    ap.add_argument("--ncu-bin", default=os.environ.get("NCU_BIN", "ncu"))
    a = ap.parse_args()
    dev = torch.device("cuda"); dt = torch.bfloat16
    H, K, V = a.hv, a.dk, a.dv
    scale = 1.0 / (K ** 0.5)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    # Decode is CUDA-graphed in serving -> NCU (--ncu) is faithful; prefill is eager
    # -> always cuda_event (the multi-kernel dispatch floor is real), never NCU.
    batch_ax = BATCH_NCU if a.ncu else (BATCH_SMOKE if a.mode == "smoke" else BATCH_FULL)
    seq_ax = SEQ_SMOKE if a.mode == "smoke" else SEQ_FULL
    print(f"[gdn{'/ncu' if a.ncu else ''}] dev={torch.cuda.get_device_name(0)} H={H} K={K} V={V} n_linear={a.n_linear}", flush=True)

    # PREFILL grid FIRST: chunk_gated_delta_rule, B=1, T=seq (eager in serving ->
    # always cuda_event). Deliberately before decode: the fragile fused_recurrent
    # decode proxy hits an illegal-memory bug (B=16 on 3090, B>=120 on A100/H100)
    # that corrupts the CUDA context; running prefill first saves the floor grid
    # regardless of where decode dies.
    pre_rows = []
    for T in seq_ax:
        try:
            q, k, v, g, beta = mk(1, T, H, K, V, dev, dt)
            def call():
                return chunk_gated_delta_rule(
                    q, k, v, g, beta, scale=scale, output_final_state=True,
                    use_qk_l2norm_in_kernel=True)
            us = time_call(call)
            pre_rows.append((T, round(us, 3), round(us * a.n_linear, 3)))
            print(f"  prefill T={T:5d}: {us:8.2f} us/call  x{a.n_linear}L={us*a.n_linear:9.1f} us", flush=True)
        except Exception as ex:
            print(f"  prefill T={T} FAILED: {type(ex).__name__}: {ex}", flush=True)
            traceback.print_exc()
            break

    # DECODE grid: fused recurrent, T=1, initial recurrent state [B,H,V,K]
    dec_rows = []
    for B in batch_ax:
        try:
            if a.ncu:
                us = ncu_op_us(_ncu_decode_inner(B, H, K, V), a.ncu_bin, NCU_REPS)
                if us is None:
                    print(f"  ncu MISS decode B={B}", flush=True)
                    continue
            else:
                q, k, v, g, beta = mk(B, 1, H, K, V, dev, dt)
                state = torch.randn(B, H, V, K, device=dev, dtype=torch.float32)
                # NOTE: representative GDN decode-core cost. The faithful serving decode
                # is fused_sigmoid_gating_delta_rule_update (fixed-param, no autotune);
                # the generic recurrent delta-rule is the same math and a valid proxy.
                def call():
                    return fused_recurrent_gated_delta_rule(
                        q, k, v, g, beta, scale=scale, initial_state=state,
                        inplace_final_state=False, use_qk_l2norm_in_kernel=True)
                us = time_call(call)
            dec_rows.append((B, round(us, 3), round(us * a.n_linear, 3)))
            print(f"  {'ncu ' if a.ncu else ''}decode B={B:4d}: {us:8.2f} us/call  x{a.n_linear}L={us*a.n_linear:9.1f} us", flush=True)
        except Exception as ex:
            print(f"  decode B={B} FAILED: {type(ex).__name__}: {ex}", flush=True)
            traceback.print_exc()
            break

    # decode -> {ncu,cuda_event}/gdn/ (follows --ncu); prefill -> cuda_event/gdn/.
    base = f"{a.gpu_label}_{a.geometry_tag}"
    dec_method = "ncu" if a.ncu else "cuda_event"
    ddir = out / dec_method / "gdn"; ddir.mkdir(parents=True, exist_ok=True)
    pdir = out / "cuda_event" / "gdn"; pdir.mkdir(parents=True, exist_ok=True)
    with open(ddir / f"{base}_decode.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["batch", "us_per_call", "us_x_nlinear"]); w.writerows(dec_rows)
    with open(pdir / f"{base}_prefill.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["seq_len", "us_per_call", "us_x_nlinear"]); w.writerows(pre_rows)
    print(f"[gdn] wrote {ddir}/{base}_decode.csv + {pdir}/{base}_prefill.csv ({len(dec_rows)}/{len(pre_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
