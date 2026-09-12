"""Dump the HF reference on the REAL 13-layer Kimi-K3 slice, for the kimi crate's gates.

Runs Moonshot's own modeling code (`modeling_kimi_linear.py`, shipped inside the
checkpoint) on the truncated real checkpoint built by `slice_truncate.py`
(13 layers: KDA layers 0,1,2,4,5,6,8,9,10,12 zero-indexed, MLA layers 3,7,11,
dense FFN layer 0, LatentMoE layers 1..12, attn_res anchors at layers 0 and 12).

    KIMI_SLICE_MODEL=/data35/kevinlau/kimi-slice/truncated \
    PYTHONPATH=/data35/kevinlau/pylibs/fla \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python ref_slice.py <out.safetensors> [decode_steps]

    python ref_slice.py --compare <a.safetensors> <b.safetensors> <out.json>

Tap names, prompt ids and the dump layout are shared with `ref_full.py` (the
0.40B harness) so the crate's gate code reads both dumps identically.

Why this shape (the option analysis, 2026-09-01):
  (a) transformers `run_compressed` execution: DEAD. compressed-tensors 0.17
      removed `CompressedLinear` (`from_linear` raises "no longer supported"),
      and its replacement — `ModelCompressor.add_decompress_hook` — decompresses
      the ENTIRE model to bf16 on the first forward pass. 13 MXFP4 MoE layers
      decompress to ~709 GiB, which fits neither 4x80 GiB HBM nor is worth
      spilling. `run_compressed=False` decompresses at load: same wall. Upstream
      is moving away from quantized execution, so a fresh venv does not help.
  (b) layer-streaming (load one layer, forward, free): works but decode is
      sequential across layers *and* steps, so 48 forwards re-read ~195 GiB of
      shards each -> hours of pure I/O, and a lot of new state-juggling code.
  (c) CHOSEN: packed-resident experts. The packed slice text weights are
      195.3 GiB (measured from shard headers), which fits across GPUs 0-3
      (4x80 GiB) with room for activations. Expert w1/w2/w3 nn.Linear modules
      are swapped for `MXFP4Linear`, which keeps `weight_packed`/`weight_scale`
      uint8-resident and dequantizes per call with compressed-tensors' OWN
      functions (`unpack_fp4_from_uint8` + `decompress_mx_scale` + `dequantize`)
      — the exact composition `NVFP4PackedCompressor.decompress` performs, so
      the nibble order and e8m0 semantics are the library's, not ours. A
      startup gate additionally checks one expert bit-exactly against
      `MXFP4PackedCompressor.decompress` driven by the checkpoint's own
      quantization_config.

MXFP4 facts verified against compressed-tensors 0.17 source:
  - packing: two e2m1 values per uint8, LOW nibble first
    (`unpack_fp4_from_uint8`: `torch.stack((low, high), dim=1)`);
  - group-32 along the input (last) dim; scale uint8 e8m0, value 2^(byte-127)
    decoded in bf16 (`decompress_mx_scale`).

Checkpoint quirks handled here (the Rust loader must mirror them):
  - `self_attn.A_log` is stored as [128] but the model has 96 KDA heads; the
    tail [96:] is exactly zero in every layer (verified) — export-time padding
    to head_dim. We slice `[:num_heads]` and hard-assert the tail is zero.
  - `KimiLinearModel.__init__` force-overrides `_attn_implementation` to
    flash_attention_2 (not installed here, and non-eager is the wrong gate
    target anyway). We set `config._attn_implementation = "eager"` AFTER
    construction; the modeling code reads the config at forward time, so the
    eager mask + eager attention path is taken with zero code modification.

Dtype policy matches `ref_full.py` KIMI_REF_DTYPE=bf16 (the dtype this
checkpoint ships in, and the default here): every non-uint8 tensor is cast to
bf16 at load, norms/A_log/dt_bias included; widenings happen where the modeling
code takes `.float()`. fp32 accumulation is wherever Moonshot's code does it.

Shims (same rules as ref_full.py — no monkeypatching of the modeling code):
  1. fla-core via PYTHONPATH=/data35/kevinlau/pylibs/fla.
  2. modeling files materialized as byte-identical copies in an importable
     package (the checkpoint dir name is not a Python identifier).
  3. Cross-GPU pipeline: layers 0-3 + embed on cuda:0, 4-6 on cuda:1,
     7-9 on cuda:2, 10-12 + final norm + output_attn_res + lm_head on cuda:3.
     A forward pre-hook per decoder layer moves tensor args/kwargs to that
     layer's device (block_residual included); the KimiDynamicCache keeps each
     layer's state on that layer's device untouched.
"""

import hashlib
import json
import os
import shutil
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

import ref_full  # PROMPT_IDS, install_taps, compare — shared with the 0.40B harness

PKG = "kimi_k3_slice_mod"
PKG_FILES = ("configuration_kimi_k3.py", "modeling_kimi_linear.py")

DTYPES = {"f32": torch.float32, "fp32": torch.float32, "bf16": torch.bfloat16}

# layer -> cuda ordinal. Packed MoE layer ~16.6 GiB => ~50 GiB of weights per GPU.
LAYER_DEV = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 2, 8: 2, 9: 2,
             10: 3, 11: 3, 12: 3}
EMBED_DEV, TAIL_DEV = 0, 3  # embed_tokens; final norm/output_attn_res/lm_head


# ---------------------------------------------------------------------------
# MXFP4 expert linear: packed-resident, dequantized per call with
# compressed-tensors' own kernels.
# ---------------------------------------------------------------------------

from compressed_tensors.compressors.mx_utils import decompress_mx_scale
from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
from compressed_tensors.quantization.lifecycle.forward import dequantize as ct_dequantize


class MXFP4Linear(nn.Module):
    """Drop-in for the expert nn.Linear: uint8-packed weights stay resident,
    the bf16 weight is materialized per forward call (~22 MiB) and freed.

    Dequant is compressed-tensors' own `NVFP4PackedCompressor.decompress`
    composition, line for line: unpack (low nibble first) -> e8m0 scale decode
    -> group-32 `dequantize` (args inferred GROUP(32) from the scale shape).
    """

    def __init__(self, out_features: int, in_features: int, device=None):
        super().__init__()
        assert in_features % 32 == 0
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer(
            "weight_packed",
            torch.empty(out_features, in_features // 2, dtype=torch.uint8, device=device),
        )
        self.register_buffer(
            "weight_scale",
            torch.empty(out_features, in_features // 32, dtype=torch.uint8, device=device),
        )

    def dequant(self) -> torch.Tensor:
        w = unpack_fp4_from_uint8(
            self.weight_packed, self.out_features, self.in_features, dtype=torch.bfloat16
        )
        scale = decompress_mx_scale(self.weight_scale)
        return ct_dequantize(x_q=w, scale=scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.dequant().to(x.dtype))

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, mxfp4-pack"


def verify_dequant_against_ct(model_dir: str, lin: "MXFP4Linear") -> None:
    """Bit-exact check of MXFP4Linear.dequant against the canonical
    MXFP4PackedCompressor.decompress, driven by the checkpoint's own
    quantization_config. This is the nibble-order/e8m0 gate."""
    from compressed_tensors.compressors import BaseCompressor
    from compressed_tensors.quantization import QuantizationConfig

    with open(os.path.join(model_dir, "config.json")) as f:
        qc_raw = json.load(f)["text_config"]["quantization_config"]
    qc = QuantizationConfig.model_validate(qc_raw)
    scheme = qc.config_groups["group_0"]
    compressor = BaseCompressor.load_from_registry(qc.format)
    out = compressor.decompress(
        {
            "weight_packed": lin.weight_packed.cpu(),
            "weight_scale": lin.weight_scale.cpu(),
        },
        scheme,
    )
    ours = lin.dequant().cpu()
    ref = out["weight"].to(ours.dtype)
    assert torch.equal(ours, ref), (
        f"MXFP4Linear.dequant diverges from {type(compressor).__name__}.decompress: "
        f"max abs diff {(ours.float() - ref.float()).abs().max()}"
    )
    print(f"mxfp4 dequant gate: bit-exact vs {type(compressor).__name__}.decompress "
          f"({qc.format}) on {tuple(ours.shape)}")


# ---------------------------------------------------------------------------
# Model construction (meta) + streamed multi-shard loading
# ---------------------------------------------------------------------------

def _materialize_pkg(model_dir: str, tmp_root: str) -> None:
    pkg_dir = os.path.join(tmp_root, PKG)
    os.makedirs(pkg_dir, exist_ok=True)
    with open(os.path.join(pkg_dir, "__init__.py"), "w"):
        pass
    for name in PKG_FILES:
        src = os.path.join(model_dir, name)
        shutil.copyfile(src, os.path.join(pkg_dir, name))
        with open(src, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()[:16]
        print(f"  {name}: sha256:{digest}")
    if tmp_root not in sys.path:
        sys.path.insert(0, tmp_root)


def _shim_transformers_compat() -> None:
    """Shim 3: the modeling file is written for transformers ~4.56 and this env
    has 5.12. Two APIs drifted; both are patched on the *transformers* side
    (aliases/adapters only — no Moonshot code is modified, no behavior change):
      - OutputRecorder moved from transformers.utils.generic to
        transformers.utils.output_capturing; re-expose it at the old path.
      - create_causal_mask renamed `input_embeds` -> `inputs_embeds`, dropped
        the (already unused) `cache_position` kwarg, and now calls
        `cache.get_mask_sizes(q_length: int, layer_idx)` where 4.56 passed the
        `cache_position` tensor; install a kwarg adapter, plus a cache proxy
        translating the new get_mask_sizes contract back to the old one
        (KimiDynamicCache only ever reads `cache_position.shape[0]`, which is
        exactly `q_length`)."""
    import inspect

    import transformers.masking_utils as masking_utils
    import transformers.utils.generic as generic

    if not hasattr(generic, "OutputRecorder"):
        from transformers.utils.output_capturing import OutputRecorder

        generic.OutputRecorder = OutputRecorder
        print("shim: transformers.utils.generic.OutputRecorder aliased from "
              "transformers.utils.output_capturing (moved in transformers 5.x)")

    params = inspect.signature(masking_utils.create_causal_mask).parameters
    if "input_embeds" not in params and not getattr(
            masking_utils.create_causal_mask, "_kimi_compat", False):
        orig = masking_utils.create_causal_mask

        class _OldCacheMaskProxy:
            def __init__(self, cache):
                object.__setattr__(self, "_cache", cache)

            def __getattr__(self, name):
                return getattr(self._cache, name)

            def get_mask_sizes(self, q_length: int, layer_idx: int):
                fake_cache_position = torch.empty(q_length, device="meta")
                return self._cache.get_mask_sizes(fake_cache_position, layer_idx)

        def create_causal_mask_compat(*args, **kwargs):
            if "input_embeds" in kwargs:
                kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            kwargs.pop("cache_position", None)  # deprecated/unused in 5.x
            pkv = kwargs.get("past_key_values")
            if pkv is not None and "cache_position" in inspect.signature(
                    pkv.get_mask_sizes).parameters:
                kwargs["past_key_values"] = _OldCacheMaskProxy(pkv)
            return orig(*args, **kwargs)

        create_causal_mask_compat._kimi_compat = True
        masking_utils.create_causal_mask = create_causal_mask_compat
        print("shim: create_causal_mask adapter installed (input_embeds->"
              "inputs_embeds, cache_position dropped, get_mask_sizes contract "
              "translated — all moved/renamed in transformers 5.x)")


def build_meta_model(model_dir: str, tmp_root: str):
    """Build the 13-layer text model on the meta device with MXFP4 expert
    linears, ready for streamed loading (also used by slice_truncate.py
    as the expected-tensor-name oracle)."""
    _shim_transformers_compat()
    print("materializing modeling package from checkpoint (verbatim copies):")
    _materialize_pkg(model_dir, tmp_root)
    cfg_mod = __import__(f"{PKG}.configuration_kimi_k3", fromlist=["KimiLinearConfig"])
    mdl_mod = __import__(f"{PKG}.modeling_kimi_linear",
                         fromlist=["KimiLinearForCausalLM", "KimiDynamicCache"])

    with open(os.path.join(model_dir, "config.json")) as f:
        text_cfg = dict(json.load(f)["text_config"])
    text_cfg.pop("auto_map", None)
    config = cfg_mod.KimiLinearConfig(**text_cfg)
    config._attn_implementation = "eager"
    config.use_cache = True

    with torch.device("meta"):
        model = mdl_mod.KimiLinearForCausalLM(config)
        # Swap expert linears for packed-resident MXFP4 modules. Structure
        # only; Moonshot's forward code is untouched.
        n_swapped = 0
        for layer in model.model.layers:
            moe = getattr(layer, "block_sparse_moe", None)
            if moe is None:
                continue
            for expert in moe.experts:
                for wn in ("w1", "w2", "w3"):
                    lin = getattr(expert, wn)
                    setattr(expert, wn, MXFP4Linear(lin.out_features, lin.in_features))
                    n_swapped += 1
    # The modeling __init__ force-overrides to flash_attention_2; flip the
    # config back AFTER construction (read at forward time — see module doc).
    config._attn_implementation = "eager"
    print(f"meta model built: {config.num_hidden_layers} layers, "
          f"{n_swapped} expert linears -> MXFP4Linear, attn=eager")
    model.eval()
    return model, config, mdl_mod


def _device_for(name: str) -> str:
    if name.startswith("model.layers."):
        return f"cuda:{LAYER_DEV[int(name.split('.')[2])]}"
    if name.startswith("model.embed_tokens"):
        return f"cuda:{EMBED_DEV}"
    return f"cuda:{TAIL_DEV}"  # model.norm, model.output_attn_res_*, lm_head


def _assign(model: nn.Module, name: str, tensor: torch.Tensor) -> None:
    mod = model
    parts = name.split(".")
    for p in parts[:-1]:
        mod = getattr(mod, p)
    leaf = parts[-1]
    if leaf in mod._parameters:
        mod._parameters[leaf] = nn.Parameter(tensor, requires_grad=False)
    elif leaf in mod._buffers:
        expected = mod._buffers[leaf]
        assert expected.shape == tensor.shape and expected.dtype == tensor.dtype, \
            f"{name}: buffer shape/dtype mismatch {tensor.shape} {tensor.dtype}"
        mod._buffers[leaf] = tensor
    else:
        raise KeyError(f"{name}: no parameter or buffer at this path")


def load_weights(model: nn.Module, config, model_dir: str, dtype: torch.dtype) -> None:
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    shards = sorted(set(weight_map.values()))
    num_heads = config.linear_attn_config["num_heads"]

    expected = set(model.state_dict().keys())
    loaded, dropped = set(), 0
    t0 = time.time()
    for si, shard in enumerate(shards):
        with safe_open(os.path.join(model_dir, shard), framework="pt") as f:
            for key in f.keys():
                if not key.startswith("language_model."):
                    dropped += 1  # vision_tower / mm_projector, never instantiated
                    continue
                name = key[len("language_model."):]
                t = f.get_tensor(key)
                if name.endswith("self_attn.A_log") and t.shape[0] != num_heads:
                    # export-time zero-padding of the per-head A_log to head_dim
                    assert torch.all(t[num_heads:] == 0.0), \
                        f"{name}: expected zero tail beyond {num_heads} heads"
                    t = t[:num_heads].clone()
                if t.dtype != torch.uint8:  # packed weights/scales stay uint8
                    t = t.to(dtype)
                _assign(model, name, t.to(_device_for(name)))
                loaded.add(name)
        print(f"  shard {si + 1}/{len(shards)} {shard} "
              f"({time.time() - t0:.0f}s elapsed)", flush=True)

    missing, unexpected = expected - loaded, loaded - expected
    assert not unexpected, f"unexpected keys: {sorted(unexpected)[:8]}"
    assert not missing, f"missing keys: {sorted(missing)[:8]}"
    for n, p in list(model.named_parameters()) + list(model.named_buffers()):
        assert p.device.type != "meta", f"{n} never loaded (still meta)"
    print(f"loaded {len(loaded)} text tensors (dropped {dropped} vision/projector), "
          f"dtype {dtype}, {time.time() - t0:.0f}s")
    for d in sorted({v for v in LAYER_DEV.values()}):
        print(f"  cuda:{d}: {torch.cuda.memory_allocated(d) / 2**30:.1f} GiB weights")


def install_pipeline_hooks(model: nn.Module) -> None:
    """Move each decoder layer's tensor args/kwargs to its device. The custom
    KimiDynamicCache is left alone — every layer's state already lives on that
    layer's device."""

    def move(obj, dev):
        if torch.is_tensor(obj):
            return obj.to(dev)
        if isinstance(obj, tuple):
            return tuple(move(o, dev) for o in obj)
        if isinstance(obj, list):
            return [move(o, dev) for o in obj]
        if isinstance(obj, dict):
            return {k: move(v, dev) for k, v in obj.items()}
        return obj

    for i, layer in enumerate(model.model.layers):
        dev = f"cuda:{LAYER_DEV[i]}"

        def hook(_m, args, kwargs, dev=dev):
            return move(args, dev), move(kwargs, dev)

        layer.register_forward_pre_hook(hook, with_kwargs=True)


def build_model(model_dir: str, tmp_root: str):
    model, config, mdl_mod = build_meta_model(model_dir, tmp_root)
    dtype = DTYPES[os.environ.get("KIMI_REF_DTYPE", "bf16")]
    load_weights(model, config, model_dir, dtype)
    install_pipeline_hooks(model)
    # nibble-order / e8m0 gate on a real expert
    verify_dequant_against_ct(model_dir, model.model.layers[1].block_sparse_moe.experts[0].w1)
    return model, config, mdl_mod


# ---------------------------------------------------------------------------
# Dump (mirrors ref_full.dump: same tap names, same tensor layout)
# ---------------------------------------------------------------------------

def dump(model_dir: str, out_path: str, decode_steps: int, tmp_root: str) -> None:
    torch.manual_seed(33377335)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    model, config, mdl_mod = build_model(model_dir, tmp_root)
    kda_mode = os.environ.get("KIMI_REF_KDA_MODE")
    if kda_mode:
        for layer in model.model.layers:
            if hasattr(layer.self_attn, "mode"):
                layer.self_attn.mode = kda_mode
        print(f"KDA prefill mode forced to {kda_mode!r}")
    captured = {}
    handles = ref_full.install_taps(model, config, captured)

    tensors = {}
    embed_dev = f"cuda:{EMBED_DEV}"
    ids = torch.tensor([ref_full.PROMPT_IDS], dtype=torch.long, device=embed_dev)
    cache = mdl_mod.KimiDynamicCache(config=config)
    with torch.no_grad():
        out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    logits = out.logits
    for k, v in captured.items():
        tensors[k] = v[0] if v.dim() == 3 else v
    tensors["prefill_logits"] = logits.detach().float().cpu()[0]
    print(f"prefill: {len(ref_full.PROMPT_IDS)} tokens, logits {tuple(logits.shape)}, "
          f"argmax(last) {int(logits[0, -1].argmax())}", flush=True)

    for i in range(config.num_hidden_layers):
        if cache.recurrent_states[i] is not None:
            tensors[f"kda_state_{i}"] = cache.recurrent_states[i].detach().float().cpu()[0]
        if cache.conv_states[i] is not None:
            qc, kc, vc = cache.conv_states[i]
            tensors[f"kda_convq_{i}"] = qc.detach().float().cpu()[0]
            tensors[f"kda_convk_{i}"] = kc.detach().float().cpu()[0]
            tensors[f"kda_convv_{i}"] = vc.detach().float().cpu()[0]

    tok = logits[:, -1:].argmax(dim=-1)
    for s in range(decode_steps):
        captured.clear()
        with torch.no_grad():
            out = model(input_ids=tok.to(embed_dev), past_key_values=cache, use_cache=True)
        step_logits = out.logits
        tensors[f"decode{s}_token"] = tok.detach().cpu()[0].long()
        tensors[f"decode{s}_logits"] = step_logits.detach().float().cpu()[0, -1]
        if s == 0:
            for k, v in captured.items():
                tensors[f"d0_{k}"] = v[0] if v.dim() == 3 else v
        print(f"decode {s}: in={int(tok[0, 0])} argmax={int(step_logits[0, -1].argmax())}",
              flush=True)
        tok = step_logits[:, -1:].argmax(dim=-1)
    for k, v in captured.items():
        tensors[f"dlast_{k}"] = v[0] if v.dim() == 3 else v

    for h in handles:
        h.remove()
    tensors["input_ids"] = torch.tensor(ref_full.PROMPT_IDS, dtype=torch.int64)
    tensors = {k: v.contiguous() for k, v in tensors.items()}
    save_file(tensors, out_path)
    print(f"wrote {out_path}: {len(tensors)} tensors, {decode_steps} decode steps")


def main() -> None:
    if sys.argv[1] == "--compare":
        ref_full.compare(sys.argv[2], sys.argv[3], sys.argv[4])
        return
    out_path = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else ref_full.DECODE_STEPS
    dump(
        os.environ["KIMI_SLICE_MODEL"],
        out_path,
        steps,
        os.environ.get("TMPDIR", "/tmp"),
    )


if __name__ == "__main__":
    main()
