"""A pasted `vllm serve` recipe: its exports, model and flags, and the layout those flags launch."""

import json
import re
import shlex
from typing import NamedTuple

from quettabench.config.schema import ConfigError

PUNCTUATION = frozenset("();<>|&")
FLAG_LIKE = re.compile(r"-{1,2}[A-Za-z]")
# vLLM's short spellings of the flags below.
ALIASES = {
    "-tp": "--tensor-parallel-size",
    "-pp": "--pipeline-parallel-size",
    "-dp": "--data-parallel-size",
    "-ep": "--enable-expert-parallel",
    "-cc": "--compilation-config",
}
# layout field -> (flag, cast); a field the recipe does not set takes SILENT.
VALUE_FLAGS = {
    "tp": ("--tensor-parallel-size", int),
    "pp": ("--pipeline-parallel-size", int),
    "dp": ("--data-parallel-size", int),
    "dtype": ("--dtype", str),
    "kv_cache_dtype": ("--kv-cache-dtype", str),
    "max_model_len": ("--max-model-len", int),
    "gpu_util": ("--gpu-memory-utilization", float),
}
# None is the engine's own default, which qbench does not know.
SILENT = {
    "tp": 1,
    "pp": 1,
    "dp": 1,
    "dtype": "auto",
    "kv_cache_dtype": "auto",
    "max_model_len": None,
    "gpu_util": None,
    "prefix_caching": None,
    "ep": False,
    "cudagraph": True,  # None: the recipe's compilation config sets it
}
# name -> (flag that turns it on, flag that turns it off)
SWITCH_FLAGS = {
    "ep": ("--enable-expert-parallel", "--no-enable-expert-parallel"),
    "prefix_caching": ("--enable-prefix-caching", "--no-enable-prefix-caching"),
    "eager": ("--enforce-eager", "--no-enforce-eager"),
}
REFUSED = {
    "--data-parallel-size-local": "qbench runs one node; the GPU count is tp * pp * dp",
    "--config": "a config file hides flags from the recipe text; write them in the recipe",
    "--api-key": "the startup log would print it; `export VLLM_API_KEY=...` before the command instead",
}
READ = {
    "--port",
    "--served-model-name",
    *REFUSED,
    *(flag for flag, _ in VALUE_FLAGS.values()),
    *(flag for pair in SWITCH_FLAGS.values() for flag in pair),
}
# `-O3`, `-O 3` and `-O=3` all set the optimization level.
OPTIMIZATION_LEVEL = re.compile(r"-O(\d*)")


class Flag(NamedTuple):
    pos: int  # position in the flags list
    value: str | None  # None: the recipe gives the name alone
    inline: bool  # written `--name=value`


def parse_recipe(text):
    """(exports, model, flags): `export K=V` lines, then exactly one `vllm serve <model> ...` command."""
    text = re.sub(r"\\\n", " ", text.replace("\r\n", "\n").replace("\r", "\n"))
    exports, serve = {}, None
    for line in text.split("\n"):
        words = _words(line)
        if not words:
            continue
        if serve is not None:
            raise ConfigError("engine.recipe: nothing may follow the `vllm serve` command")
        if words[0] == "export":
            for word in words[1:]:
                key, eq, value = word.partition("=")
                if not eq:
                    raise ConfigError(f"engine.recipe: export wants K=V, got {word!r}")
                exports[key] = value
        elif words[:2] == ["vllm", "serve"] and len(words) > 2:
            serve = words
        else:
            raise ConfigError("engine.recipe must be `export K=V` lines, then `vllm serve <model> ...`")
    if serve is None:
        raise ConfigError("engine.recipe has no `vllm serve <model>` command")
    if serve[2].startswith("-"):
        raise ConfigError(f"engine.recipe: the model is positional, `vllm serve <model> ...`; got {serve[2]!r}")
    return exports, serve[2], serve[3:]


def _words(line):
    """The shell words of one logical line, a trailing # comment dropped; shell operators are refused."""
    lexer = shlex.shlex(_without_comment(line), posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        words = list(lexer)
    except ValueError as e:
        raise ConfigError(f"engine.recipe: {e}") from None
    for word in words:
        if set(word) <= PUNCTUATION:
            raise ConfigError(f"engine.recipe: shell operator {word!r}; the recipe is one plain command")
    return words


def _without_comment(line):
    """The line up to a # that begins a word outside quotes, as bash reads it; `a#b` stays whole."""
    quote, starts = None, True
    chars = iter(enumerate(line))
    for i, char in chars:
        if quote:
            if char == quote:
                quote = None
            elif char == "\\" and quote == '"':
                next(chars, None)
        elif char == "#" and starts:
            return line[:i]
        elif char in "'\"":
            quote, starts = char, False
        elif char == "\\":
            next(chars, None)
            starts = False
        else:
            starts = char.isspace()
    return line


def flag_index(flags):
    """{flag name: [Flag, ...]} with `_` read as `-`, the short aliases spelled out and `--name=value` split."""
    index: dict[str, list[Flag]] = {}
    for i, word in enumerate(flags):
        if not FLAG_LIKE.match(word):
            continue
        name, eq, inline = word.partition("=")
        name = ALIASES.get(name) or name.replace("_", "-")
        following = flags[i + 1] if i + 1 < len(flags) else None
        if eq:
            value = inline
        else:
            value = following if following is not None and not FLAG_LIKE.match(following) else None
        index.setdefault(name, []).append(Flag(i, value, bool(eq)))
    return index


def flag_value(index, name):
    """The value of the last `name`; None when it is absent or has none."""
    found = index.get(name)
    return found[-1].value if found else None


def with_port(flags, port):
    """flags with every --port value set to port; a --port is appended when there is none."""
    flags = list(flags)
    found = flag_index(flags).get("--port")
    if not found:
        return [*flags, "--port", str(port)]
    for at in reversed(found):
        if at.inline:
            flags[at.pos] = f"{flags[at.pos].partition('=')[0]}={port}"
        elif at.value is None:
            flags.insert(at.pos + 1, str(port))
        else:
            flags[at.pos + 1] = str(port)
    return flags


def layout(flags):
    """The layout fields as the recipe launches them; ValueError for a spelling qbench cannot read."""
    index = flag_index(flags)
    for name in index:
        if name in REFUSED:
            raise ValueError(f"{name} is refused: {REFUSED[name]}")
        if name.startswith("--") and name not in READ and any(flag.startswith(name) for flag in READ):
            raise ValueError(f"{name} abbreviates a flag qbench reads; spell the flag out")
    out = dict(SILENT)
    for field, (name, cast) in VALUE_FLAGS.items():
        if name not in index:
            continue
        value = flag_value(index, name)
        try:
            out[field] = cast(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise ValueError(f"{name} {value!r} is not a plain {cast.__name__} (no k or M suffixes)") from None
    for field, (on, off) in SWITCH_FLAGS.items():
        found = _switch(index, on, off)
        if field == "eager":
            if found:
                out["cudagraph"] = False
            elif found is None:
                out["cudagraph"] = _compiled_cudagraph(index)
        elif found is not None:
            out[field] = found
    return out


def _compiled_cudagraph(index):
    """Graphs on (True) when the recipe sets no compilation; off when every one says off; else None (the recipe's config sets it)."""
    says = []
    for name, found in index.items():
        if name == "--compilation-config":
            for at in found:
                says.append(_graphs_off(at.value))
        elif name.startswith("--compilation-config."):
            for at in found:
                says.append(name.endswith(".cudagraph-mode") and str(at.value).upper() == "NONE")
        elif (level := OPTIMIZATION_LEVEL.fullmatch(name)) or name == "--optimization-level":
            for at in found:
                says.append((level[1] if level and level[1] else at.value) == "0")
    if not says:
        return True
    return False if all(says) else None


def _graphs_off(text):
    try:
        config = json.loads(text)
    except (TypeError, ValueError):
        return False
    return isinstance(config, dict) and str(config.get("cudagraph_mode", "")).upper() == "NONE"


def _switch(index, on, off):
    """The last of `--on`, `--on=true|false` and `--off` the recipe passes; None when it passes none."""
    events = []
    for at in index.get(on, []):
        events.append((at.pos, _truth(on, at.value) if at.inline else True))
    for at in index.get(off, []):
        if at.inline:
            raise ValueError(f"{off} takes no value")
        events.append((at.pos, False))
    return max(events)[1] if events else None


def _truth(name, value):
    if value not in ("true", "false"):
        raise ValueError(f"{name}={value} must be true or false")
    return value == "true"
