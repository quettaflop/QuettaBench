"""Environment variables to Config fields; the only os.environ reads and writes."""

import os

from quettabench.config.schema import ConfigError


def _truthy(raw):
    return raw.strip().lower() in {"1", "true", "on", "yes"}


def _nonempty(raw):
    return bool(raw)


def _attn_ffn(raw):
    """'attn=N,ffn=M' as afd:<N>a<M>f; a missing or extra key raises ValueError."""
    pools = dict(part.split("=", 1) for part in raw.split(","))
    if set(pools) != {"attn", "ffn"}:
        raise ValueError(raw)
    return f"afd:{int(pools['attn'])}a{int(pools['ffn'])}f"


# Two vars for one field with different values are refused.
CLASHES = {"XVAL_TRACE": "XVAL_CORPUS", "XVAL_AFD": "XVAL_SERVING_STYLE"}


# env var -> (Config path, converter, description).
ENV_MAP = {
    "XVAL_LINK_PROFILE": ("hardware.link_profile", str, "interconnect profile name"),
    "XVAL_BACKEND": ("hardware.backend", str, "GPU backend override"),
    "XVAL_PP": ("engine.pp", int, "pipeline-parallel size"),
    "XVAL_EVAL_POS": ("mode.eval_pos", int, "decode position where ms/step is evaluated"),
    "XVAL_CORPUS": ("workload.source", str, "prompt source: text corpus"),
    "XVAL_TRACE": ("workload.source", str, "prompt source: trace"),
    "XVAL_QUANT": ("engine.quant", str, "weight quantization"),
    "XVAL_SERVING_STYLE": ("engine.serving_style", str, "serving style"),
    "XVAL_AFD": ("engine.serving_style", _attn_ffn, "attention/FFN disaggregated pools"),
    "XVAL_ALLOW_METHOD_MISMATCH": (
        "crossval.allow_method_mismatch",
        _nonempty,
        "accept a table whose method differs from the header",
    ),
    "VMIN_ATTEMPTS": ("mode.vmin_attempts", int, "min-batch fit attempts"),
    "PROMPT_MODE": ("workload.prompt_mode", str, "how prompts are built"),
    "GPU_UTIL": ("engine.gpu_util", float, "engine GPU memory fraction"),
    "ENABLE_EP": ("engine.ep", _truthy, "expert parallelism on"),
    "NCCL_ALGO": ("hardware.collective.NCCL_ALGO", str, "NCCL algorithm pin"),
    "NCCL_PROTO": ("hardware.collective.NCCL_PROTO", str, "NCCL protocol pin"),
    # The caller's value beats the hardware yaml for the collective pins.
    "NCCL_P2P_LEVEL": ("hardware.collective.NCCL_P2P_LEVEL", str, "NCCL P2P level pin"),
    "NCCL_IB_DISABLE": ("hardware.collective.NCCL_IB_DISABLE", str, "NCCL InfiniBand off switch"),
    "NCCL_SOCKET_IFNAME": ("hardware.collective.NCCL_SOCKET_IFNAME", str, "NCCL socket interface pin"),
    "LLMSRV_NO_NVLS": ("hardware.collective.LLMSRV_NO_NVLS", str, "disables NVLink SHARP in the engine"),
    "LLMSRV_TWOSHOT_MIN_BYTES": (
        "hardware.collective.LLMSRV_TWOSHOT_MIN_BYTES",
        str,
        "engine two-shot allreduce size threshold",
    ),
    "QS_SHA": ("provenance.engine_sha", str, "QuettaServe commit"),
    "DS_ENGINE_SHA": ("provenance.engine_sha", str, "engine commit, used when QS_SHA is unset"),
    "CUDA_VISIBLE_DEVICES": ("hardware.visible_devices", str, "GPU ids the run may use"),
    "HF_HOME": ("hardware.hf_home", str, "Hugging Face cache dir"),
    # Not a field: input to backend auto-detection.
    "TPU_NAME": ("detect.tpu_name", str, "TPU name"),
    # Not a field: bench_sha where there is no .git (the docker image sets it).
    "QBENCH_SHA": ("detect.bench_sha", str, "bench commit baked into the docker image"),
}


def read_env(env=None, only=None):
    """Clashing vars with different values raise ConfigError."""
    if env is None:
        env = os.environ
    out = {}
    for var, (path, convert, _) in ENV_MAP.items():
        if only is not None and var not in only:
            continue
        raw = env.get(var)
        if raw is None or raw == "":
            continue
        try:
            value = convert(raw)
        except ValueError:
            raise ConfigError(f"env {var}={raw!r}: not a valid {convert.__name__}") from None
        if path in out:
            if var in CLASHES and out[path] != value:
                raise ConfigError(f"env {CLASHES[var]} and {var} are both set; unset one")
            continue
        out[path] = value
    return out


def torchrun_rank():
    rank = os.environ.get("RANK")
    return int(rank) if rank not in (None, "") else None


def with_env(values):
    """For a child that takes a secret through env rather than argv."""
    return {**os.environ, **{k: str(v) for k, v in values.items()}}


def export(values):
    """Returns the names set."""
    names = []
    for var, value in values.items():
        if value:
            os.environ[var] = str(value)
            names.append(var)
    return names
