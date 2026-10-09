import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from quettabench.config import env_map
from quettabench.crossval import compare, reader

# The env vars the old table read, now through env_map into Config paths.
ENV = ("XVAL_QUANT", "XVAL_SERVING_STYLE", "NCCL_ALGO", "NCCL_PROTO",
       "XVAL_ALLOW_METHOD_MISMATCH")


def main():
    bench_path, cache_path = sys.argv[1], sys.argv[2]
    if not Path(bench_path).exists():
        sys.exit(f"no bench log at {bench_path}; run `just bench llama` first")
    if not Path(cache_path).exists():
        sys.exit(f"no vLLM baseline at {cache_path}; run `just vllm llama`")
    values = env_map.read_env(only=ENV)
    ours = reader.read_bench_log(bench_path, reader.engine_facts(values))
    if not ours.rows:
        print("".join(n + "\n" for n in ours.notes), end="")
        sys.exit(f"no engine rows parsed from {bench_path}")
    theirs = reader.read_baseline(cache_path)
    report = compare.pair(ours, theirs, values.get("crossval.allow_method_mismatch", False),
                          now=time.time())
    print(compare.render(report), end="")
    if not report.rows:
        sys.exit("no overlapping (ctx, bs, tp) points")


if __name__ == "__main__":
    main()
