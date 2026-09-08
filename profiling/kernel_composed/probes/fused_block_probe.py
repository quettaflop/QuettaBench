#!/usr/bin/env python3
"""Measure ONE real transformer block forward -> a mixed-step profile CSV, so the
kernel-composed step prediction can be cross-validated without a checkpoint.

WHY THIS PROBE EXISTS. Every leaf kernel is measured (gemm, grouped-MoE, flash
attention, elementwise) and `KernelComposed.fused_step_ms` SUMS them into a step
cost. That sum is an assumption -- additivity -- and it had never been checked for
FP8, because checking it normally means serving the real model and diffing against
the engine, which for Qwen3-235B-A22B-FP8 needs 236 GB of weights on disk.

This probe removes the checkpoint from the loop. A block's kernel time does not
depend on the VALUES in its weights, only on shapes and dtypes, so one block built
from random weights runs exactly the kernels the real model would. One FP8 block of
Qwen3-235B is ~2.5 GB, versus 236 GB for the checkpoint. Timing that block and
multiplying by n_layers gives the same quantity `fused_step_ms` predicts, measured
end-to-end through the real ops rather than assembled from parts.

WHAT IT DOES AND DOES NOT VALIDATE. It validates the step-cost layer: whether the
measured leaf grids, composed, equal a real fused forward. It does NOT validate
vLLM's scheduler, host overheads, the frontend model, or the queue sim. That split
is the point: everything above the step cost is precision-independent (a scheduler
does not care about dtype), so if the FP8 step cost is right, the already-calibrated
serving dynamics carry over to FP8 unchanged.

Ops are vLLM's own wherever the grids used vLLM's own -- the block-scaled FP8 linear,
`fused_experts`, `flash_attn_varlen_func` -- so a discrepancy is the composition's,
not a different kernel's.

  CUDA_VISIBLE_DEVICES=1 python fused_block_probe.py --model qwen3-235b-a22b-fp8 --tp 4
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch
import yaml

DEFAULT_PREFILL = [512, 1024, 2048, 4096, 8192]
_VLLM_CTX = None      # see main(): must outlive every CustomOp construction
DEFAULT_BATCH = [0, 1, 8, 16, 32]


def _default_out_dir() -> str:
    for env in ("KERNEL_DATA", "QUETTASIM_DATA"):
        raw = (os.environ.get(env) or "").strip()
        if raw:
            p = Path(raw).expanduser()
            return str(p if env == "KERNEL_DATA" else p / "kernel_data")
    return str(Path(__file__).resolve().parents[2] / "data" / "kernel_data")


class Block:
    """One tp-sharded transformer block: attention + grouped-MoE FFN.

    Mirrors ``sum_kernels._block_linear_us`` + ``_prefill_attn_us`` op for op, in the
    same order, so the measured total is comparable term by term. The MoE router GEMM
    stays BF16 even under an FP8 checkpoint: HF's quantization_config lists every
    ``mlp.gate`` in ``ignored_layers``.
    """

    def __init__(self, spec: dict, tp: int, fp8: bool, dev, block=(128, 128)):
        from vllm.model_executor.layers.fused_moe import fused_experts, fused_topk
        from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
        from vllm.model_executor.layers.quantization.utils.fp8_utils import W8A8BlockFp8LinearOp
        from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

        self.fe, self.ft, self.dev, self.fp8 = fused_experts, fused_topk, dev, fp8
        h = self.h = int(spec["hidden_dim"])
        hd = int(spec["head_dim"])
        nq = int(spec["n_heads"]) // tp
        kv_shards = min(tp, int(spec["kv_heads"]))
        nkv = int(spec["kv_heads"]) // kv_shards
        self.nq, self.nkv, self.hd = nq, nkv, hd
        self.n_layers = int(spec["n_layers"])
        E = self.E = int(spec["n_experts"])
        self.k = int(spec["n_active_experts"])
        inter = int(spec["intermediate_size"]) // tp
        self.qkv_n = (nq + 2 * nkv) * hd
        self.o_k = (nq * hd)

        bf = torch.bfloat16
        mk = lambda *s: (torch.randn(*s, device=dev, dtype=bf) * 0.02)  # noqa: E731
        self.lin_op = None
        if fp8:
            self.lin_op = W8A8BlockFp8LinearOp(GroupShape(*block), GroupShape(1, block[1]))
            self.bn, self.bk = block
            nb = lambda x, b: (x + b - 1) // b  # noqa: E731
            self.w_qkv = mk(self.qkv_n, h).to(torch.float8_e4m3fn)
            self.s_qkv = torch.rand(nb(self.qkv_n, self.bn), nb(h, self.bk),
                                    device=dev, dtype=torch.float32).mul_(.01).add_(1e-4)
            self.w_o = mk(h, self.o_k).to(torch.float8_e4m3fn)
            self.s_o = torch.rand(nb(h, self.bn), nb(self.o_k, self.bk),
                                  device=dev, dtype=torch.float32).mul_(.01).add_(1e-4)
            self.w1 = mk(E, 2 * inter, h).to(torch.float8_e4m3fn)
            self.w2 = mk(E, h, inter).to(torch.float8_e4m3fn)
            s1 = torch.rand(E, nb(2 * inter, self.bn), nb(h, self.bk),
                            device=dev, dtype=torch.float32).mul_(.01).add_(1e-4)
            s2 = torch.rand(E, nb(h, self.bn), nb(inter, self.bk),
                            device=dev, dtype=torch.float32).mul_(.01).add_(1e-4)
            self.qc = fp8_w8a8_moe_quant_config(w1_scale=s1, w2_scale=s2,
                                                block_shape=list(block))
        else:
            self.w_qkv = mk(h, self.qkv_n)
            self.w_o = mk(self.o_k, h)
            self.w1 = mk(E, 2 * inter, h)
            self.w2 = mk(E, h, inter)
            self.qc = None
        self.w_gate = mk(h, E)          # router: BF16 in both cases

    def _lin(self, x, w, s, w_bf):
        if self.fp8:
            return self.lin_op.apply(x, w, s)
        return torch.matmul(x, w_bf)

    def attn_buffers(self, p: int, b: int, ctx: int, dev):
        """Pre-allocate the attention inputs for a (p prefill, b decode) step.

        Prefill and decode are TWO separate flash calls, matching how the composition
        prices them (`_prefill_attn_us` + `_decode_attn_us` read different grids for
        different kernels). The decode rows must attend their FULL context: a single
        varlen call with cu_seqlens_k == cu_seqlens_q would give every decode row
        kv_len=1, reading none of the KV cache -- which is most of a decode step's work.
        """
        out = {}
        if p > 0:
            cu = torch.tensor([0, p], dtype=torch.int32, device=dev)
            out["pf"] = (cu, p)
        if b > 0:
            # b rows of one query token each over `ctx` resident keys.
            kv = torch.randn(b * ctx, self.nkv, self.hd, device=dev, dtype=torch.bfloat16)
            out["dec"] = (
                torch.randn(b, self.nq, self.hd, device=dev, dtype=torch.bfloat16),
                kv, torch.randn_like(kv),
                torch.arange(0, b + 1, dtype=torch.int32, device=dev),
                torch.arange(0, (b + 1) * ctx, ctx, dtype=torch.int32, device=dev),
                ctx,
            )
        return out

    def forward(self, x, ab):
        """One block over `x` (tokens, hidden); `ab` from attn_buffers."""
        from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func
        m = x.shape[0]
        # rmsnorm (fused add+norm in vLLM) x2 around the two sub-layers
        h1 = torch.nn.functional.rms_norm(x, (self.h,))
        qkv = self._lin(h1, self.w_qkv, getattr(self, "s_qkv", None), self.w_qkv)
        q = qkv[:, : self.nq * self.hd].view(m, self.nq, self.hd)
        kv = qkv[:, self.nq * self.hd:].view(m, 2 * self.nkv, self.hd)
        k, v = kv[:, : self.nkv], kv[:, self.nkv:]
        if "pf" in ab:
            cu, ml = ab["pf"]
            flash_attn_varlen_func(q[:ml].contiguous(), k[:ml].contiguous(),
                                   v[:ml].contiguous(), ml, cu, ml, cu_seqlens_k=cu,
                                   causal=True, fa_version=3)
        if "dec" in ab:
            dq, dk, dv, cq, ck, kvl = ab["dec"]
            flash_attn_varlen_func(dq, dk, dv, 1, cq, kvl, cu_seqlens_k=ck,
                                   causal=True, fa_version=3)
        a = q.reshape(m, self.nq * self.hd)
        x = x + self._lin(a, self.w_o, getattr(self, "s_o", None), self.w_o)
        h2 = torch.nn.functional.rms_norm(x, (self.h,))
        gate = torch.matmul(h2, self.w_gate)          # router GEMM, bf16
        tw, tid, *_ = self.ft(h2, gate, self.k, True)
        ff = (self.fe(h2, self.w1, self.w2, tw, tid, quant_config=self.qc) if self.fp8
              else self.fe(h2, self.w1, self.w2, tw, tid))
        return x + ff


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model YAML stem under device_spec/models")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--prefill", default=",".join(map(str, DEFAULT_PREFILL)))
    ap.add_argument("--batch", default=",".join(map(str, DEFAULT_BATCH)))
    ap.add_argument("--decode-ctx", type=int, default=32768)
    ap.add_argument("--gpu-label", default="H200")
    ap.add_argument("--out-dir", default=_default_out_dir())
    a = ap.parse_args()

    from vllm.config import VllmConfig, set_current_vllm_config
    # Hold the context manager for the process lifetime. Calling __enter__() on a
    # temporary lets it be garbage-collected, and closing a @contextmanager generator
    # runs its finally block -- which resets the config contextvar, so the next
    # CustomOp construction fails with "Current vLLM config is not set".
    global _VLLM_CTX
    _VLLM_CTX = set_current_vllm_config(VllmConfig())
    _VLLM_CTX.__enter__()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _graph import graph_time_us

    root = Path(__file__).resolve().parents[2]
    spec = yaml.safe_load((root / "device_spec" / "models" / f"{a.model}.yaml").read_text())
    prec = (spec.get("precision") or {})
    fp8 = str(prec.get("activations", "bfloat16")).startswith("fp8")
    dev = torch.device("cuda")
    blk = Block(spec, a.tp, fp8, dev)
    print(f"[fused_block] {torch.cuda.get_device_name(0)} | {spec['name']} tp{a.tp} "
          f"| fp8={fp8} | {blk.n_layers} layers | heads {blk.nq}q/{blk.nkv}kv/{blk.hd}",
          flush=True)

    rows = []
    for p in [int(x) for x in a.prefill.split(",")] + [0]:
        for b in [int(x) for x in a.batch.split(",")]:
            m = p + b
            if m <= 0:
                continue
            try:
                x = torch.randn(m, blk.h, device=dev, dtype=torch.bfloat16)
                ab = blk.attn_buffers(p, b, a.decode_ctx, dev)
                us = graph_time_us(lambda: blk.forward(x, ab))
            except Exception as ex:                     # noqa: BLE001
                print(f"  skip p={p} b={b}: {type(ex).__name__}: {ex}", flush=True)
                continue
            step_ms = us / 1000.0 * blk.n_layers
            rows.append({"prefill_tokens": p, "decode_batch": b,
                         "step_ms": round(step_ms, 4)})
            print(f"  p={p:5d} b={b:3d}: block {us:9.1f}us -> "
                  f"x{blk.n_layers} = {step_ms:9.2f} ms/step", flush=True)
            del x, ab
            torch.cuda.empty_cache()

    dst = Path(a.out_dir) / "validation"
    dst.mkdir(parents=True, exist_ok=True)
    tag = f"{a.model}_tp{a.tp}_{a.gpu_label}"
    path = dst / f"fused_block_profile_{tag}.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prefill_tokens", "decode_batch", "step_ms"])
        w.writeheader(); w.writerows(rows)
    print(f"[fused_block] wrote {path} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
