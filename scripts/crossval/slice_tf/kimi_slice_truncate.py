"""Build a VALID standalone 13-layer HF model dir from the downloaded Kimi-K3 slice.

The slice at <src> has 16 of 96 shards (layers 0-12 complete, plus
embed/norm/lm_head/output_attn_res in shards 94-96) but a full-model
config.json and a full 96-shard index. This script materializes <dst> as a
directory both HF transformers and the kimi crate can load standalone:

  - the 16 shards are symlinked (relative), not copied (197 GiB);
  - config.json: text_config.num_hidden_layers=13; the 1-INDEXED
    full_attn_layers / kda_layers lists filtered to entries <= 13; everything
    else (quantization_config, situ betas, attn_res_block_size, vision_config,
    gate_lower_bound...) byte-for-byte semantics preserved;
  - model.safetensors.index.json: weight_map subset to tensors that live in the
    16 shards (== layers 0-12 plus the non-layer set plus vision tower — both
    directions verified), metadata.total_size recomputed from the shard
    headers' data_offsets over exactly the kept tensors;
  - every auxiliary python/tokenizer/processor/config file copied verbatim.

Sanity gates (all hard failures):
  A. transformers AutoConfig.from_pretrained(dst, trust_remote_code=True)
     parses, and the truncated fields read back exactly;
  B. every weight_map entry's shard file exists in <dst> (symlinks resolve);
  C. name-level diff against the actual 13-layer text model: the meta-built
     KimiLinearForCausalLM (experts as MXFP4Linear, via kimi_ref_slice) must
     find every one of its state-dict tensors in the subset weight_map and
     vice versa (language_model.* namespace) — zero missing, zero extra —
     with shapes checked against the shard headers. Known exception: A_log is
     stored zero-padded to head_dim (128) while the model has 96 heads; shape
     is allowed to differ there (the loader slices, see kimi_ref_slice.py).

    PYTHONPATH=/data35/kevinlau/pylibs/fla \
    python kimi_slice_truncate.py /data35/kevinlau/kimi-slice/model \
                                  /data35/kevinlau/kimi-slice/truncated
"""

import json
import os
import struct
import sys

N_LAYERS = 13


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

    # --- config.json ---
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

    # --- index ---
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

    # --- aux files ---
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


def gate_c_name_diff(dst: str, wm: dict) -> None:
    import kimi_ref_slice

    model, config, _ = kimi_ref_slice.build_meta_model(
        dst, os.environ.get("TMPDIR", "/tmp"))
    expected = {"language_model." + k: tuple(v.shape)
                for k, v in model.state_dict().items()}
    ckpt_text = {k for k in wm if k.startswith("language_model.")}
    ckpt_other = {k for k in wm if not k.startswith("language_model.")}

    missing = sorted(set(expected) - ckpt_text)
    extra = sorted(ckpt_text - set(expected))
    assert not missing, f"{len(missing)} model tensors absent from index: {missing[:8]}"
    assert not extra, f"{len(extra)} index tensors unknown to the model: {extra[:8]}"

    # shape check against shard headers
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
    print(f"gate C: name-level diff clean — {len(expected)} text tensors all "
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
