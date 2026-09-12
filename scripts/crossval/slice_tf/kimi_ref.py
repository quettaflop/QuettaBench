"""Dump Kimi-K3 reference activations, for the kimi crate's gates.

Runs Moonshot's own modeling code (`modeling_kimi_k3_linear.py`, shipped inside the
checkpoint) on the Kimi-K3-0.40B development replica — 8 layers covering every layer
type: KDA linear attention (0,1,2,4,5,6), MLA/NoPE full attention (3,7), a dense FFN
layer (0, `first_k_dense_replace=1`) and LatentMoE layers (1..7) — and captures the
hidden state entering every block, per-sublayer taps, the KDA recurrent/conv state
after prefill, and prefill + per-step decode logits.

    KIMI_MODEL=/data35/kevinlau/models/Kimi-K3-0.40B \
    PYTHONPATH=/data35/kevinlau/pylibs/fla \
    python kimi_ref.py <out.safetensors> [decode_steps]

    python kimi_ref.py --compare <a.safetensors> <b.safetensors> <out.json>

By default everything is fp32 end to end, TF32 explicitly off, so the candle port can
be gated tightly. `KIMI_REF_DTYPE=bf16` runs the same code in bf16 — the dtype the
real K3 ships in — which is *not* bit-reproducible run to run; calibrate any gate
built on it with `--compare` first, never by reusing the fp32 numbers.

The vision tower is never instantiated: we build `KimiLinearForCausalLM`
(the `text_config` model) directly and drop the `vision_tower.*` / `mm_projector.*`
tensors, exactly as the text-only server will.

Shims (see docs/notes/kimi-k3-scale-plan.md):
  1. `fla-core` is a hard import of the modeling file and is absent from the vllm-v4
     env; it is installed to a side directory and reached via PYTHONPATH. No source
     of Moonshot's is modified.
  2. The modeling file uses a package-relative import (`from .configuration_kimi_k3
     import ...`), and the checkpoint directory name is not a Python identifier. We
     materialize a throwaway package of *byte-identical copies* and import from it.
No monkeypatching of the modeling code itself: every tap is a forward hook.
"""

import hashlib
import json
import os
import shutil
import sys
import time

import torch
from safetensors.torch import load_file, save_file

# 16 fixed ids, all < vocab_size (163840); no tokenizer needed.
PROMPT_IDS = [1, 4321, 100, 65535, 2048, 777, 31415, 9,
              128000, 42, 5, 99991, 1234, 60000, 8, 163839]
DECODE_STEPS = 8

PKG = "kimi_k3_ref_mod"
PKG_FILES = ("configuration_kimi_k3.py", "modeling_kimi_k3_linear.py")

DTYPES = {"f32": torch.float32, "fp32": torch.float32, "bf16": torch.bfloat16}


def _materialize_pkg(model_dir: str, tmp_root: str) -> str:
    """Copy the two modeling files into an importable package (shim 2)."""
    pkg_dir = os.path.join(tmp_root, PKG)
    os.makedirs(pkg_dir, exist_ok=True)
    with open(os.path.join(pkg_dir, "__init__.py"), "w"):
        pass
    for name in PKG_FILES:
        src = os.path.join(model_dir, name)
        dst = os.path.join(pkg_dir, name)
        shutil.copyfile(src, dst)
        with open(src, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()[:16]
        print(f"  {name}: sha256:{digest}")
    if tmp_root not in sys.path:
        sys.path.insert(0, tmp_root)
    return pkg_dir


def build_model(model_dir: str, tmp_root: str):
    print("materializing modeling package from checkpoint (verbatim copies):")
    _materialize_pkg(model_dir, tmp_root)
    cfg_mod = __import__(f"{PKG}.configuration_kimi_k3", fromlist=["KimiLinearConfig"])
    mdl_mod = __import__(f"{PKG}.modeling_kimi_k3_linear",
                         fromlist=["KimiLinearForCausalLM", "KimiDynamicCache"])

    with open(os.path.join(model_dir, "config.json")) as f:
        raw = json.load(f)
    text_cfg = dict(raw["text_config"])
    text_cfg.pop("auto_map", None)
    # The 0.40B leaves `gate_lower_bound` unset, which selects the softplus gate;
    # the real 2.8T K3 sets it to -5.0 and takes the sigmoid branch instead. This
    # override runs the same weights through that branch so the port's
    # implementation of it can be gated too (see kimi/tests/reference.rs).
    lb = os.environ.get("KIMI_REF_GATE_LOWER_BOUND")
    if lb is not None:
        text_cfg["linear_attn_config"] = dict(text_cfg["linear_attn_config"])
        text_cfg["linear_attn_config"]["gate_lower_bound"] = float(lb)
        print(f"gate_lower_bound override: {lb}")
    config = cfg_mod.KimiLinearConfig(**text_cfg)
    # eager attention: the fp32 dev box has no flash-attn2 build, and eager is the
    # deterministic path we want to gate against anyway.
    config._attn_implementation = "eager"
    config.use_cache = True

    # KIMI_REF_DTYPE=bf16 casts every parameter, exactly as `model.to(bf16)` does
    # for the real K3 — norms, A_log and dt_bias included. The Rust port's
    # `load_dtype` makes the same no-exceptions choice, so the two sides agree on
    # *where* the widenings happen (at the use sites the modeling code takes
    # `.float()`), not merely on the final dtype.
    dtype = DTYPES[os.environ.get("KIMI_REF_DTYPE", "f32")]
    with torch.device("cuda"):
        model = mdl_mod.KimiLinearForCausalLM(config)
    model = model.to(dtype)
    print(f"model dtype: {dtype}")

    sd = load_file(os.path.join(model_dir, "model.safetensors"))
    text_sd, dropped = {}, 0
    for k, v in sd.items():
        if k.startswith("language_model."):
            text_sd[k[len("language_model."):]] = v.to(dtype)
        else:
            dropped += 1
    missing, unexpected = model.load_state_dict(text_sd, strict=False)
    print(f"loaded {len(text_sd)} text tensors (dropped {dropped} vision/projector)")
    assert not unexpected, f"unexpected keys: {unexpected[:8]}"
    assert not missing, f"missing keys: {missing[:8]}"
    model.eval()
    return model, config, mdl_mod


def install_taps(model, config, captured):
    """Forward hooks only — the modeling code is untouched."""
    handles = []

    def keep(name):
        def hook(_m, _a, out):
            t = out[0] if isinstance(out, tuple) else out
            captured[name] = t.detach().float().cpu()
        return hook

    def keep_in(name):
        def hook(_m, a, _kw=None):
            captured[name] = a[0].detach().float().cpu()
        return hook

    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_pre_hook(keep_in(f"layer_in_{i}")))
        handles.append(layer.self_attn.register_forward_hook(keep(f"attn_out_{i}")))
        block = getattr(layer, "block_sparse_moe", None) or layer.mlp
        handles.append(block.register_forward_hook(keep(f"ffn_out_{i}")))
        sa = layer.self_attn
        if config.is_kda_layer(i):
            handles.append(sa.q_conv1d.register_forward_hook(keep(f"kda_q_{i}")))
            handles.append(sa.k_conv1d.register_forward_hook(keep(f"kda_k_{i}")))
            handles.append(sa.v_conv1d.register_forward_hook(keep(f"kda_v_{i}")))
            handles.append(sa.f_b_proj.register_forward_hook(keep(f"kda_graw_{i}")))
            handles.append(sa.b_proj.register_forward_hook(keep(f"kda_beta_{i}")))
            handles.append(sa.g_proj.register_forward_hook(keep(f"kda_ogate_{i}")))
            handles.append(sa.o_norm.register_forward_pre_hook(keep_in(f"kda_opre_{i}")))
        else:
            handles.append(sa.q_b_proj.register_forward_hook(keep(f"mla_qb_{i}")))
            handles.append(
                sa.kv_a_proj_with_mqa.register_forward_hook(keep(f"mla_kva_{i}")))
            handles.append(sa.g_proj.register_forward_hook(keep(f"mla_ogate_{i}")))
    handles.append(model.model.norm.register_forward_pre_hook(keep_in("final_pre_norm")))
    return handles


def dump(model_dir, out_path, decode_steps, tmp_root):
    torch.cuda.set_device(0)
    torch.manual_seed(33377335)
    # True fp32: TF32 would give the reference a different rounding than candle's
    # fp32 cublas path and blur the gate.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    model, config, mdl_mod = build_model(model_dir, tmp_root)
    # KimiDeltaAttention.forward picks `mode = "fused_recurrent" if q_len == 1 else
    # self.mode`, i.e. prefill uses the chunked kernel and decode the recurrent one.
    # Both are deterministic here; flipping `self.mode` uses the modeling code's own
    # switch (no logic patched) so we can measure the chunk-vs-recurrent algorithmic
    # gap, which is the dominant error term for a naive-recurrent port.
    kda_mode = os.environ.get("KIMI_REF_KDA_MODE")
    if kda_mode:
        for layer in model.model.layers:
            if hasattr(layer.self_attn, "mode"):
                layer.self_attn.mode = kda_mode
        print(f"KDA prefill mode forced to {kda_mode!r}")
    captured = {}
    handles = install_taps(model, config, captured)

    tensors = {}
    ids = torch.tensor([PROMPT_IDS], dtype=torch.long, device="cuda")
    cache = mdl_mod.KimiDynamicCache(config=config)
    with torch.no_grad():
        out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    logits = out.logits
    for k, v in captured.items():
        tensors[k] = v[0] if v.dim() == 3 else v
    tensors["prefill_logits"] = logits.detach().float().cpu()[0]
    print(f"prefill: {len(PROMPT_IDS)} tokens, logits {tuple(logits.shape)}, "
          f"argmax(last) {int(logits[0, -1].argmax())}")

    # KDA state after prefill: the delta-rule state and the short-conv ring, per layer.
    # `transpose_state_layout=True` in the modeling code, so record the shape too.
    for i in range(config.num_hidden_layers):
        if cache.recurrent_states[i] is not None:
            tensors[f"kda_state_{i}"] = cache.recurrent_states[i].detach().float().cpu()[0]
        if cache.conv_states[i] is not None:
            qc, kc, vc = cache.conv_states[i]
            tensors[f"kda_convq_{i}"] = qc.detach().float().cpu()[0]
            tensors[f"kda_convk_{i}"] = kc.detach().float().cpu()[0]
            tensors[f"kda_convv_{i}"] = vc.detach().float().cpu()[0]

    pos = len(PROMPT_IDS)
    tok = logits[:, -1:].argmax(dim=-1)
    for s in range(decode_steps):
        captured.clear()
        with torch.no_grad():
            out = model(input_ids=tok, past_key_values=cache, use_cache=True)
        step_logits = out.logits
        tensors[f"decode{s}_token"] = tok.detach().cpu()[0].long()
        tensors[f"decode{s}_logits"] = step_logits.detach().float().cpu()[0, -1]
        if s == 0:
            for k, v in captured.items():
                tensors[f"d0_{k}"] = v[0] if v.dim() == 3 else v
        print(f"decode {s}: in={int(tok[0, 0])} argmax={int(step_logits[0, -1].argmax())}")
        pos += 1
        tok = step_logits[:, -1:].argmax(dim=-1)
    for k, v in captured.items():
        tensors[f"dlast_{k}"] = v[0] if v.dim() == 3 else v

    for h in handles:
        h.remove()
    tensors["input_ids"] = torch.tensor(PROMPT_IDS, dtype=torch.int64)
    tensors = {k: v.contiguous() for k, v in tensors.items()}
    save_file(tensors, out_path)
    print(f"wrote {out_path}: {len(tensors)} tensors, {decode_steps} decode steps")


def compare(a_path, b_path, out_json):
    """Run-to-run self-variance of the reference: this calibrates the crate's gates."""
    a, b = load_file(a_path), load_file(b_path)
    assert set(a) == set(b), "dumps have different tensor sets"
    rows, worst = {}, (0.0, "")
    for k in sorted(a):
        x, y = a[k].float(), b[k].float()
        if x.dtype == torch.int64 or "token" in k or k == "input_ids":
            assert torch.equal(a[k], b[k]), f"{k} differs across runs"
            continue
        denom = max(float(y.abs().max()), 1e-3)
        rel = float((x - y).abs().max()) / denom
        rows[k] = rel
        if rel > worst[0]:
            worst = (rel, k)
    summary = {
        "runs": [a_path, b_path],
        "max_rel": worst[0],
        "max_rel_tensor": worst[1],
        "per_tensor_rel": rows,
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"run-to-run self-variance: max rel {worst[0]:.3e} at {worst[1]}")
    for k in ("prefill_logits", "final_pre_norm"):
        if k in rows:
            print(f"  {k}: rel {rows[k]:.3e}")
    print(f"wrote {out_json}")


def verify_kda(dump_path, model_dir):
    """Re-derive the KDA output from the dumped taps with a naive per-token
    recurrence, and check it against what FLA's chunked kernel actually produced.

    This is the formula the Rust port implements, so proving it here against the
    shipped kernel on real weights removes the single biggest porting risk. Written
    against fla/ops/kda/naive.py::naive_recurrent_kda and the fused_recurrent
    kernel body (gate/beta/l2norm are folded into the kernel by the modeling code's
    use_{gate,beta_sigmoid,qk_l2norm}_in_kernel=True flags).
    """
    import torch.nn.functional as F
    d = load_file(dump_path)
    ckpt = load_file(os.path.join(model_dir, "model.safetensors"))
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)["text_config"]
    lin = cfg["linear_attn_config"]
    hd, nh = lin["head_dim"], lin["num_heads"]
    kda_layers = [i - 1 for i in lin["kda_layers"]]  # config is 1-indexed
    scale = hd ** -0.5

    worst = 0.0
    for i in kda_layers:
        q = d[f"kda_q_{i}"].double().view(-1, nh, hd)
        k = d[f"kda_k_{i}"].double().view(-1, nh, hd)
        v = d[f"kda_v_{i}"].double().view(-1, nh, hd)
        g_raw = d[f"kda_graw_{i}"].double().view(-1, nh, hd)
        beta_raw = d[f"kda_beta_{i}"].double().view(-1, nh)
        pre = f"language_model.model.layers.{i}.self_attn."
        A_log = ckpt[pre + "A_log"].double()             # [H]
        dt_bias = ckpt[pre + "dt_bias"].double().view(nh, hd)

        # gate: g = -exp(A_log) * softplus(g_raw + dt_bias)   (gate.py:naive_kda_gate)
        gk = -A_log.view(nh, 1).exp() * F.softplus(g_raw + dt_bias)
        beta = torch.sigmoid(beta_raw)
        # l2norm with eps *inside* the sqrt, then scale on q only
        qn = q / (q.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt() * scale
        kn = k / (k.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()

        S = torch.zeros(nh, hd, hd, dtype=torch.float64)   # [H, K, V]
        o = torch.zeros_like(v)
        for t in range(q.shape[0]):
            S = S * gk[t].exp().unsqueeze(-1)              # decay per key-dim
            u = (v[t] - torch.einsum("hk,hkv->hv", kn[t], S)) * beta[t].unsqueeze(-1)
            S = S + torch.einsum("hk,hv->hkv", kn[t], u)
            o[t] = torch.einsum("hk,hkv->hv", qn[t], S)

        ref = d[f"kda_opre_{i}"].double().view(-1, nh, hd)
        rel = float((o - ref).abs().max()) / max(float(ref.abs().max()), 1e-3)
        # final state, laid out as the cache stores it
        st = d[f"kda_state_{i}"].double()
        srel = min(
            float((S - st).abs().max()) if S.shape == st.shape else float("inf"),
            float((S.transpose(-1, -2) - st).abs().max())
            if S.transpose(-1, -2).shape == st.shape else float("inf"),
        ) / max(float(st.abs().max()), 1e-3)
        layout = "[H,K,V]" if S.shape == st.shape and \
            float((S - st).abs().max()) <= float((S.transpose(-1, -2) - st).abs().max()) \
            else "[H,V,K] (transposed)"
        print(f"layer {i}: naive-recurrent vs chunk_kda rel {rel:.3e} | "
              f"final state rel {srel:.3e}, cache layout {layout} {tuple(st.shape)}")
        worst = max(worst, rel, srel)
    print(f"KDA recurrence verified, worst rel {worst:.3e}")


def bench(model_dir, tmp_root, prompt_len, steps, warm):
    """Decode timing for Moonshot's own modeling code — the baseline the Rust
    crate is measured against.

    This is a *weak* baseline and should be labelled as one: it is a PyTorch eager
    loop, so it is the natural comparison for the crate's eager path and only a
    frame of reference for its graphed path. It is the honest option available,
    because no vLLM release can run this checkpoint (see
    docs/perf/2026-08-30-kimi-k3-decode.md).

    Same protocol as kimi/tests/decode_bench.rs: prefill, warm, then time each
    step individually and fit the cumulative curve, so the slope is the marginal
    per-step cost and r^2 says whether the number is clean enough to quote.
    """
    torch.cuda.set_device(0)
    torch.manual_seed(33377335)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    model, config, mdl_mod = build_model(model_dir, tmp_root)
    ids = torch.tensor(
        [[(i * 137 + 11) % 100000 for i in range(prompt_len)]],
        dtype=torch.long, device="cuda",
    )
    cache = mdl_mod.KimiDynamicCache(config=config)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        model(input_ids=ids, past_key_values=cache, use_cache=True)
    torch.cuda.synchronize()
    print(f"[hf] prefill {prompt_len} tokens: {(time.perf_counter() - t0) * 1e3:.1f} ms")

    tok = torch.tensor([[7]], dtype=torch.long, device="cuda")
    for _ in range(warm):
        with torch.no_grad():
            model(input_ids=tok, past_key_values=cache, use_cache=True)
    torch.cuda.synchronize()

    per_step, cum, total = [], [], 0.0
    for _ in range(steps):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(input_ids=tok, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        total += dt
        per_step.append(dt)
        cum.append(total)

    xs = [float(i) for i in range(steps)]
    n = float(steps)
    def fit(ys):
        mx, my = sum(xs) / n, sum(ys) / n
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        a = my - b * mx
        ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        ss_tot = sum((y - my) ** 2 for y in ys)
        return b, (1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0)
    slope, r2 = fit(cum)
    growth, _ = fit(per_step)
    med = sorted(per_step)[steps // 2]
    print(f"[hf] {next(model.parameters()).dtype} eager decode: {slope:.3f} ms/step (fit) "
          f"| mean {total / steps:.3f} | median {med:.3f} "
          f"| growth {growth * 1e3:+.2f} us/step | r2 {r2:.5f}")
    print(f"[hf] context {prompt_len + warm} -> {prompt_len + warm + steps} over the window")
    if r2 < 0.999:
        print("[hf] r2 < 0.999: the per-step cost is not clean, do not quote it.")


def main():
    if sys.argv[1] == "--bench":
        bench(
            os.environ["KIMI_MODEL"],
            os.environ.get("TMPDIR", "/tmp"),
            int(sys.argv[2]) if len(sys.argv) > 2 else 100,
            int(sys.argv[3]) if len(sys.argv) > 3 else 100,
            int(sys.argv[4]) if len(sys.argv) > 4 else 16,
        )
        return
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3], sys.argv[4])
        return
    if sys.argv[1] == "--verify-kda":
        verify_kda(sys.argv[2], os.environ["KIMI_MODEL"])
        return
    out_path = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else DECODE_STEPS
    model_dir = os.environ["KIMI_MODEL"]
    tmp_root = os.environ.get("TMPDIR", "/tmp")
    dump(model_dir, out_path, steps, tmp_root)


if __name__ == "__main__":
    main()
