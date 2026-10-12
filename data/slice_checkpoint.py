#!/usr/bin/env python3
"""Build a 13-layer HF model dir from the downloaded Kimi-K3 slice (layers 0-12) and check that it loads.
Usage: PYTHONPATH=<fla-dir> python3 data/slice_checkpoint.py <slice-dir>/model <slice-dir>/truncated
"""

import importlib
import json
import os
import shutil
import struct
import sys

N_LAYERS = 13  # layers 0-12 sit in the 16 downloaded shards
# the checkpoint dir name is not a Python identifier, so its modeling code is copied into this package
PKG = "kimi_k3_slice_mod"
PKG_FILES = ("configuration_kimi_k3.py", "modeling_kimi_linear.py")


def shard_headers(path: str) -> dict:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def layer_of(name: str) -> int | None:
    pre = "language_model.model.layers."
    return int(name[len(pre):].split(".")[0]) if name.startswith(pre) else None


def build(src: str, dst: str) -> None:
    os.makedirs(dst, exist_ok=True)

    present = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))
    print(f"symlinking {len(present)} shards")
    for f in present:
        link = os.path.join(dst, f)
        if os.path.islink(link):
            os.unlink(link)
        os.symlink(os.path.join("..", "model", f), link)

    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg["text_config"]
    tc["num_hidden_layers"] = N_LAYERS
    lac = tc["linear_attn_config"]
    lac["full_attn_layers"] = [x for x in lac["full_attn_layers"] if x <= N_LAYERS]
    lac["kda_layers"] = [x for x in lac["kda_layers"] if x <= N_LAYERS]
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"config.json: num_hidden_layers={N_LAYERS}, "
          f"full_attn_layers={lac['full_attn_layers']}, "
          f"kda_layers={lac['kda_layers']}")

    with open(os.path.join(src, "model.safetensors.index.json")) as f:
        index = json.load(f)
    have = set(present)
    wm = index["weight_map"]

    kept, dropped_missing_shard, dropped_deep_layer = {}, 0, 0
    for name, shard in wm.items():
        li = layer_of(name)
        if shard not in have:
            assert li is None or li >= N_LAYERS, \
                f"{name} belongs to the slice but its shard {shard} was not downloaded"
            dropped_missing_shard += 1
            continue
        if li is not None and li >= N_LAYERS:
            dropped_deep_layer += 1
            continue
        kept[name] = shard
    assert dropped_deep_layer == 0, \
        f"{dropped_deep_layer} layer-13+ tensors inside the downloaded shards (unexpected)"

    total_size, checked = 0, 0
    for shard in have:
        hdr = shard_headers(os.path.join(src, shard))
        for name, meta in hdr.items():
            if name in kept:
                assert kept[name] == shard, f"{name}: index says {kept[name]}, header {shard}"
                total_size += meta["data_offsets"][1] - meta["data_offsets"][0]
                checked += 1
    assert checked == len(kept), f"headers cover {checked} of {len(kept)} kept tensors"

    index["weight_map"] = kept
    index["metadata"]["total_size"] = total_size
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"index: kept {len(kept)} tensors ({total_size / 2**30:.1f} GiB), "
          f"dropped {dropped_missing_shard} entries in absent shards")

    skip = set(present) | {"config.json", "model.safetensors.index.json"}
    copied = []
    for f in sorted(os.listdir(src)):
        if f in skip or f.startswith("."):
            continue
        with open(os.path.join(src, f), "rb") as r, open(os.path.join(dst, f), "wb") as w:
            w.write(r.read())
        copied.append(f)
    print(f"copied aux files: {', '.join(copied)}")


def gate_a_autoconfig(dst: str) -> None:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(dst, trust_remote_code=True)
    tc = cfg.text_config
    assert tc.num_hidden_layers == N_LAYERS, tc.num_hidden_layers
    assert tc.linear_attn_config["full_attn_layers"] == [4, 8, 12]
    assert tc.linear_attn_config["kda_layers"] == [1, 2, 3, 5, 6, 7, 9, 10, 11, 13]
    assert tc.quantization_config["format"] == "mxfp4-pack-quantized"
    assert tc.attn_res_block_size == 12
    assert tc.linear_attn_config["gate_lower_bound"] == -5.0
    assert tc.first_k_dense_replace == 1
    print(f"gate A: AutoConfig parses ({type(cfg).__name__}); "
          f"13 layers, MLA at [4,8,12] (1-indexed), quant config intact")


def gate_b_shards_exist(dst: str) -> dict:
    with open(os.path.join(dst, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]
    for shard in set(wm.values()):
        p = os.path.join(dst, shard)
        assert os.path.exists(p), f"weight_map references missing shard {shard}"
        assert os.path.getsize(os.path.realpath(p)) > 0
    print(f"gate B: all {len(set(wm.values()))} referenced shards exist and resolve")
    return wm


def meta_text_model(model_dir: str, tmp_root: str):
    """(model, config): the 13-layer text model on the meta device with expert w1/w2/w3 swapped for packed MXFP4 buffers (weight_packed, weight_scale, uint8)."""
    import torch
    import transformers.utils.generic as generic
    from torch import nn

    if not hasattr(generic, "OutputRecorder"):  # moved in transformers 5.x
        from transformers.utils.output_capturing import OutputRecorder
        generic.OutputRecorder = OutputRecorder

    pkg_dir = os.path.join(tmp_root, PKG)
    os.makedirs(pkg_dir, exist_ok=True)
    open(os.path.join(pkg_dir, "__init__.py"), "w").close()
    for name in PKG_FILES:
        shutil.copyfile(os.path.join(model_dir, name), os.path.join(pkg_dir, name))
    if tmp_root not in sys.path:
        sys.path.insert(0, tmp_root)
    cfg_mod = importlib.import_module(f"{PKG}.configuration_kimi_k3")
    mdl_mod = importlib.import_module(f"{PKG}.modeling_kimi_linear")

    with open(os.path.join(model_dir, "config.json")) as f:
        text_cfg = dict(json.load(f)["text_config"])
    text_cfg.pop("auto_map", None)
    config = cfg_mod.KimiLinearConfig(**text_cfg)
    config._attn_implementation = "eager"

    class PackedMXFP4(nn.Module):
        def __init__(self, out_features, in_features):
            super().__init__()
            self.register_buffer("weight_packed", torch.empty(
                out_features, in_features // 2, dtype=torch.uint8))
            self.register_buffer("weight_scale", torch.empty(
                out_features, in_features // 32, dtype=torch.uint8))

    with torch.device("meta"):
        model = mdl_mod.KimiLinearForCausalLM(config)
        for layer in model.model.layers:
            moe = getattr(layer, "block_sparse_moe", None)
            if moe is None:
                continue
            for expert in moe.experts:
                for wn in ("w1", "w2", "w3"):
                    lin = getattr(expert, wn)
                    setattr(expert, wn, PackedMXFP4(lin.out_features, lin.in_features))
    return model, config


def gate_c_name_diff(dst: str, wm: dict) -> None:
    model, config = meta_text_model(dst, os.environ.get("TMPDIR", "/tmp"))
    expected = {"language_model." + k: tuple(v.shape)
                for k, v in model.state_dict().items()}
    ckpt_text = {k for k in wm if k.startswith("language_model.")}
    ckpt_other = {k for k in wm if not k.startswith("language_model.")}

    missing = sorted(set(expected) - ckpt_text)
    extra = sorted(ckpt_text - set(expected))
    assert not missing, f"{len(missing)} model tensors absent from index: {missing[:8]}"
    assert not extra, f"{len(extra)} index tensors unknown to the model: {extra[:8]}"

    hdr_cache, num_heads = {}, config.linear_attn_config["num_heads"]
    mismatches = []
    for name, shard in wm.items():
        if name not in expected:
            continue
        if shard not in hdr_cache:
            hdr_cache[shard] = shard_headers(os.path.join(dst, shard))
        got = tuple(hdr_cache[shard][name]["shape"])
        want = expected[name]
        if got != want:
            if name.endswith("self_attn.A_log") and want == (num_heads,):
                continue  # zero-padded to head_dim; loader slices (verified there)
            mismatches.append((name, want, got))
    assert not mismatches, f"shape mismatches: {mismatches[:8]}"
    print(f"gate C: name-level diff clean -- {len(expected)} text tensors all "
          f"present with matching shapes (A_log pad exception), "
          f"{len(ckpt_other)} vision/projector tensors alongside")


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    build(src, dst)
    gate_a_autoconfig(dst)
    wm = gate_b_shards_exist(dst)
    gate_c_name_diff(dst, wm)
    print("truncated model dir OK:", dst)


if __name__ == "__main__":
    main()
