# Every env var the pre-refactor code reads, mapped to the Config field it
# sets, so old habits keep working. read_env() and export() are the
# package's own reads and writes of os.environ; tests/config/test_no_environ.py
# enforces it, with an allowlist of older readers that only shrinks.
import os

from quettabench.config.schema import ConfigError


def _truthy(raw):
    """serve.py's ENABLE_EP rule: 1/true/on/yes."""
    return raw.strip().lower() in {"1", "true", "on", "yes"}


def _nonempty(raw):
    """crossval's XVAL_ALLOW_METHOD_MISMATCH rule (old table.py): any non-empty value."""
    return bool(raw)


def _attn_ffn(raw):
    """xval_config.afd's XVAL_AFD='attn=N,ffn=M' as the serving style afd:<N>a<M>f.
    A missing or extra key raises instead of being ignored."""
    pools = dict(part.split("=", 1) for part in raw.split(","))
    if set(pools) != {"attn", "ffn"}:
        raise ValueError(raw)
    return f"afd:{int(pools['attn'])}a{int(pools['ffn'])}f"


# A second var that names the same field as an earlier one is refused when
# their values differ, instead of the first silently winning.
CLASHES = {"XVAL_TRACE": "XVAL_CORPUS", "XVAL_AFD": "XVAL_SERVING_STYLE"}


# env var -> (Config path, converter, old reader). The old reader is where
# the pre-refactor code read the var, as file:line at 8bd8bd6; every one of
# those files has since retired.
ENV_MAP = {
    "XVAL_LINK_PROFILE": ("hardware.link_profile", str, "xval_config.py:79,283 vmin_fit.py:27,151"),
    "XVAL_BACKEND": ("hardware.backend", str, "xval_config.py:182"),
    "XVAL_PP": ("engine.pp", int, "xval_config.py:337 vmin_fit.py:122"),
    "XVAL_EVAL_POS": ("mode.eval_pos", int, "vmin_fit.py:118"),
    "XVAL_CORPUS": ("workload.source", str, "vmin_fit.py:55, prompt_mode corpus"),
    "XVAL_TRACE": ("workload.source", str, "vmin_fit.py:137, prompt_mode trace"),
    "XVAL_QUANT": ("engine.quant", str, "table.py:137"),
    "XVAL_SERVING_STYLE": ("engine.serving_style", str, "table.py:129"),
    "XVAL_AFD": ("engine.serving_style", _attn_ffn, "xval_config.py:271 on branch feat/afd, never merged"),
    "XVAL_ALLOW_METHOD_MISMATCH": ("crossval.allow_method_mismatch", _nonempty, "table.py:152"),
    "VMIN_ATTEMPTS": ("mode.vmin_attempts", int, "vmin_fit.py:256"),
    "PROMPT_MODE": ("workload.prompt_mode", str, "vmin_fit.py:130"),
    "GPU_UTIL": ("engine.gpu_util", float, "vmin_fit.py:127"),
    "ENABLE_EP": ("engine.ep", _truthy, "serve.py:1323"),
    "NCCL_ALGO": ("hardware.collective.NCCL_ALGO", str, "vmin_fit.py:30,149 table.py:119"),
    "NCCL_PROTO": ("hardware.collective.NCCL_PROTO", str, "vmin_fit.py:30,150 table.py:120"),
    # The other five collective pins: the caller's value beat the hardware
    # yaml in bench.sh:41-45 and vmin_fit.py:28-30 (as of 8bd8bd6).
    "NCCL_P2P_LEVEL": ("hardware.collective.NCCL_P2P_LEVEL", str, "bench.sh:42 vmin_fit.py:30"),
    "NCCL_IB_DISABLE": ("hardware.collective.NCCL_IB_DISABLE", str, "bench.sh:42 vmin_fit.py:30"),
    "NCCL_SOCKET_IFNAME": ("hardware.collective.NCCL_SOCKET_IFNAME", str,
                           "bench.sh:42 vmin_fit.py:30"),
    "LLMSRV_NO_NVLS": ("hardware.collective.LLMSRV_NO_NVLS", str, "bench.sh:42 vmin_fit.py:30"),
    "LLMSRV_TWOSHOT_MIN_BYTES": ("hardware.collective.LLMSRV_TWOSHOT_MIN_BYTES", str,
                                 "bench.sh:42 vmin_fit.py:30"),
    "QS_SHA": ("provenance.engine_sha", str, "cache.py:46"),
    "DS_ENGINE_SHA": ("provenance.engine_sha", str, "cache.py:46, used when QS_SHA is unset"),
    "CUDA_VISIBLE_DEVICES": ("hardware.visible_devices", str, "trace_serve.py:111,491"),
    "HF_HOME": ("hardware.hf_home", str, "provision.py writes it to .env; just loads it"),
    # Not a field: an input to backend auto-detection (xval_config.py:191).
    "TPU_NAME": ("detect.tpu_name", str, "xval_config.py:191"),
}


def read_env(env=None, only=None):
    """{Config path: value} for every mapped var that is set and non-empty.

    env=None reads the process environment; only, when given, limits the
    read to those var names. Where two vars feed one field the
    first in ENV_MAP order wins (QS_SHA over DS_ENGINE_SHA, as cache.py did),
    except the pairs in CLASHES: XVAL_CORPUS and XVAL_TRACE both set to
    different files is refused, since the source a grid run reads would
    depend on prompt_mode silently, and so is XVAL_AFD beside a different
    XVAL_SERVING_STYLE.
    """
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


def export(values):
    """Put each non-empty value into the process environment unless the
    variable is already set, so the caller's env wins. An in-process engine
    reads its NCCL settings there; vmin_fit did the same setdefault at import."""
    for var, value in values.items():
        if value:
            os.environ.setdefault(var, str(value))
