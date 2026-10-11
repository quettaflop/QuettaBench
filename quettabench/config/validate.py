"""Serving-topology and expert-parallel rules, checked before anything
launches: the loader refuses a Config that breaks them, the comparison
refuses rows whose topologies differ. Moved unchanged from xval_config.py.
"""

import re


def parse_serving_style(spec):
    """Serving topology for a run, recorded so aggregated and disaggregated
    numbers are never mixed in one table. 'aggregated' (default) is one server
    over tp*pp GPUs. 'disagg:<P>p<D>d' splits prefill and decode onto disjoint
    GPU pools with a KV-transfer connector between them (vLLM PD). 'ep<N>dp<M>'
    is wide expert parallelism (N-way experts across M data-parallel replicas).
    'afd:<A>a<F>f' puts attention and FFN/MoE layers on disjoint pools of A
    and F GPUs (attention-FFN disaggregation).
    Returns {mode, ...degrees}; raises ValueError on a malformed spec so a
    fat-fingered topology fails at parse rather than silently running aggregated."""
    if spec in (None, "", "aggregated"):
        return {"mode": "aggregated"}
    m = re.fullmatch(r"disagg:(\d+)p(\d+)d", spec)
    if m:
        p, d = int(m.group(1)), int(m.group(2))
        if p < 1 or d < 1:
            raise ValueError(f"disagg needs >=1 prefill and >=1 decode: {spec!r}")
        return {"mode": "disagg", "prefill": p, "decode": d}
    m = re.fullmatch(r"afd:(\d+)a(\d+)f", spec)
    if m:
        a, f = int(m.group(1)), int(m.group(2))
        if a < 1 or f < 1:
            raise ValueError(f"afd needs >=1 attention and >=1 FFN GPU: {spec!r}")
        return {"mode": "afd", "attn": a, "ffn": f}
    m = re.fullmatch(r"ep(\d+)dp(\d+)", spec)
    if m:
        ep, dp = int(m.group(1)), int(m.group(2))
        if ep < 1 or dp < 1:
            raise ValueError(f"ep_dp needs >=1 expert-parallel and >=1 data-parallel: {spec!r}")
        return {"mode": "ep_dp", "ep": ep, "dp": dp}
    raise ValueError(f"unknown serving_style {spec!r}; use aggregated, "
                     "disagg:<P>p<D>d, or ep<N>dp<M>")


def device_count(engine):
    """GPUs one run needs: the two pools for a disaggregated style, else tp * pp."""
    style = parse_serving_style(engine.serving_style)
    if style["mode"] == "afd":
        return style["attn"] + style["ffn"]
    if style["mode"] == "disagg":
        return style["prefill"] + style["decode"]
    return engine.tp * engine.pp


def disagg_gpu_sets(style, available):
    """(prefill_ids, decode_ids): disjoint device-id slices for a disagg style.
    The pools never share a GPU because a shared device would let prefill and
    decode contend, voiding the separation the P/D split exists to measure."""
    p, d = style["prefill"], style["decode"]
    if p + d > len(available):
        raise ValueError(f"disagg:{p}p{d}d needs {p + d} GPUs, have {len(available)}")
    prefill, decode = available[:p], available[p:p + d]
    assert not (set(prefill) & set(decode)), "prefill and decode GPU sets overlap"
    return prefill, decode


def check_ep_legal(ep, num_experts):
    """Expert parallelism must evenly divide the model's expert count, else a
    rank ends up with a ragged expert shard the engine cannot place. Raises on
    an illegal degree rather than letting the engine fail deep in load."""
    if num_experts % ep != 0:
        raise ValueError(f"ep={ep} does not divide {num_experts} experts")


def disagg_kv_config(role, rank, connector, kv_port=14579, kv_parallel_size=2):
    """One side of a prefill/decode split as a vLLM kv_transfer_config dict,
    passed to the server as --kv-transfer-config <json>. role is 'kv_producer'
    (prefill) or 'kv_consumer' (decode); the two share kv_port and
    kv_parallel_size so the connector pairs them and take distinct kv_rank. A
    wrong role raises here rather than starting a server that never pairs."""
    if role not in ("kv_producer", "kv_consumer"):
        raise ValueError(f"kv_role must be kv_producer or kv_consumer, got {role!r}")
    return {"kv_connector": connector, "kv_role": role, "kv_rank": rank,
            "kv_parallel_size": kv_parallel_size, "kv_port": kv_port}


def serving_style_compatible(a, b):
    """Two rows compare only when their serving topology matches; a disagg
    number against an aggregated one is a category error the table must refuse,
    the same way an NCCL or KV mismatch withholds a ratio."""
    return parse_serving_style(a)["mode"] == parse_serving_style(b)["mode"]
