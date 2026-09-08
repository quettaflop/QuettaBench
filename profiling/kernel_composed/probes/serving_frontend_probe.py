#!/usr/bin/env python3
"""Live-server probe -> the device-YAML `frontend:` and `serving:` blocks, for a GPU
that has NO ground-truth serving runs yet (for a GPU that HAS GT, prefer the cheaper
profiling/derive_serving_yaml.py, which fits the same blocks from the GT).

Same decomposition as derive_serving_yaml.py -- the frontend/host cost is
(measured latency - kernel-composed compute prediction) -- but the measurements come
from controlled requests against a live vLLM OpenAI server instead of GT JSONs:

  1. single-stream, sweeping prompt length -> floor_ms + new_ms_per_token (TTFT
     residual) and the per-STEP host floor (TPOT residual: whole-step
     kernel_step_overhead_ms, per-layer decode_overhead_ms_per_layer)
  2. APC pairs (same prompt twice)         -> cached_ms_per_token (hit-TTFT residual)
  3. TRUE concurrent herd sweep            -> mult_curve reference points (measured
     median TTFT vs the queue sim's frontend-off prediction of the same herd)

Every prompt carries a unique seed prefix so sweep prompts never share an APC
prefix with each other (a plain "hi hi hi..." sweep silently turns the longer
prompts into partial cache hits). Token counts come from the server's own usage
accounting (stream_options.include_usage), not from the word count.

Start vLLM separately (the GT harness does the same), then point this at it:

  vllm serve <model> --tensor-parallel-size 1 --port 8000 &
  python serving_frontend_probe.py --gpu-label H200 --model Qwen/Qwen3-30B-A3B \
      --model-yaml ../engine/device_spec/models/qwen3-30b-a3b.yaml \
      --base-url http://localhost:8000/v1

Prints the blocks; does NOT edit the YAML. UNVALIDATED without a live server --
run on the target GPU.
"""
from __future__ import annotations
import argparse, json, statistics as st, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / ".." / ".."))
from engine.loaders.config_loader import load_kernel_gpu, load_kernel_model  # noqa: E402
from engine.backends.kernel_composed_cost import KernelComposedCost  # noqa: E402

_SWEEP_PROMPTS = [128, 256, 512, 1024, 2048]   # single-stream prompt lengths (tokens)
_CONC_SWEEP = [1, 2, 5, 10, 20, 40]
_SAMPLES = 3                                    # medians over this many repeats
_DECODE_TOKENS = 64                             # output tokens for a stable TPOT


def _stream_once(base_url: str, model: str, prompt: str,
                 out_tokens: int = 8) -> tuple[float, float, int]:
    """(ttft_ms, tpot_ms, prompt_tokens) for one streamed completion."""
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": out_tokens,
        "stream": True, "temperature": 0.0,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"{base_url}/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first_at = None
    n = 0
    prompt_tokens = 0
    with urllib.request.urlopen(req) as r:
        for line in r:
            if not line.startswith(b"data:") or b"[DONE]" in line:
                continue
            payload = json.loads(line[5:])
            usage = payload.get("usage")
            if usage:                       # final usage-only chunk: no token
                prompt_tokens = int(usage.get("prompt_tokens", 0))
                continue
            if not payload.get("choices"):
                continue
            now = time.perf_counter()
            if first_at is None:
                first_at = now
            n += 1
    last = time.perf_counter()
    ttft_ms = (first_at - t0) * 1000.0
    tpot_ms = ((last - first_at) / max(1, n - 1)) * 1000.0 if n > 1 else 0.0
    return ttft_ms, tpot_ms, prompt_tokens


def _linfit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares y = intercept + slope*x (closed form, no numpy)."""
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / denom if denom else 0.0
    return (sy - slope * sx) / n, slope


def _conc_sweep(a, cost, model):
    """TRUE concurrent herds (unique ~512-token prompts). mult reference =
    (measured median - sim frontend-off median) / the c=1 residual; the sim
    prediction absorbs engine queueing (chunked-prefill serialization), so the
    ratio isolates the FRONTEND's load scaling.

    Each concurrency point runs an UNMEASURED warmup herd first, then the
    measured herd: the first concurrent burst at a new batch shape triggers
    one-time work (triton/moe autotune, graph shapes) that contaminated the
    first version of this sweep by up to 50x at c=2-10."""
    from engine.sim.queue_sim import predict_cell  # noqa: PLC0415
    from engine.loaders.workload import Turn  # noqa: PLC0415

    def herd(c: int) -> list[tuple[float, float, int]]:
        prompts = [_prompt(512) for _ in range(c)]
        with ThreadPoolExecutor(max_workers=c) as ex:
            return list(ex.map(lambda pr: _stream_once(a.base_url, a.model, pr), prompts))

    print("# concurrent herd sweep (512-token prompts; warmup herd per point)", flush=True)
    herd(max(_CONC_SWEEP))                          # global warmup at the largest shape
    conc_rows = []
    for c in _CONC_SWEEP:
        herd(c)                                     # unmeasured per-point warmup
        res = herd(c)
        meas = st.median([r[0] for r in res])
        ptok = int(st.median([r[2] for r in res]))
        turn = Turn(cache_hit_tokens=0.0, new_prefill_tokens=float(ptok), osl_tokens=8)
        pred_ttfts, _ = predict_cell(cost, [turn], c)
        pred = pred_ttfts[0] if pred_ttfts else 0.0
        conc_rows.append((c, meas, pred, meas - pred))
        print(f"  c={c:3d}  measured={meas:8.1f}ms  sim(frontend-off)={pred:8.1f}ms  "
              f"resid={meas - pred:7.1f}ms", flush=True)
    base_resid = max(1e-6, conc_rows[0][3])
    mult_pts = [(c, max(1.0, round(resid / base_resid, 2)))
                for c, _m, _p, resid in conc_rows]
    if a.only_conc:
        print(f"\n  mult_curve: {[[c, m] for c, m in mult_pts]}  # measured resid ratio; lanes flat")
        print(f"  # measured (c, median ttft ms, sim frontend-off ms): "
              f"{[(c, round(m, 1), round(p, 1)) for c, m, p, _ in conc_rows]}")
    return mult_pts, conc_rows


_SEED = [0]


def _prompt(tokens: int, *, reuse: str | None = None) -> str:
    """~`tokens` tokens with a UNIQUE leading seed (no shared APC prefix across
    sweep prompts). ``reuse`` returns exactly the given prompt (APC-hit pairs)."""
    if reuse is not None:
        return reuse
    _SEED[0] += 1
    return f"seed{_SEED[0]:06d} " + "hi " * max(1, tokens - 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-label", required=True)
    ap.add_argument("--model", required=True, help="served model name (OpenAI 'model' field)")
    ap.add_argument("--model-yaml", required=True, help="engine model YAML for the compute prediction")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--out-dir", default=None, help="unused (kept for run_probe.sh parity)")
    ap.add_argument("--skip-conc", action="store_true",
                    help="skip the concurrent herd sweep (parts 1-2 only)")
    ap.add_argument("--only-conc", action="store_true",
                    help="run only the concurrent herd sweep (parts 1-2 already collected)")
    a = ap.parse_args()

    import dataclasses
    from kernel_composed.types import FrontendParams
    gpu = load_kernel_gpu(HERE / ".." / ".." / "device_spec" / f"{a.gpu_label.lower()}.yaml")
    # The residual decomposition needs the COMPUTE-ONLY prediction: strip any
    # frontend block / host floors the YAML already carries (ported or measured),
    # or the sim side would double-count exactly the thing being measured.
    gpu = dataclasses.replace(gpu, frontend=FrontendParams(),
                              kernel_ttft_overhead_ms=0.0, kernel_step_overhead_ms=0.0,
                              kernel_step_overlap_per_ms=0.0,
                              decode_overhead_ms=0.0, decode_overhead_ms_per_layer=0.0)
    model = load_kernel_model(Path(a.model_yaml))
    cost = KernelComposedCost(model, gpu, tp=a.tp)   # compute-only prediction

    # warmup (loads tokenizer caches, CUDA graphs already captured at server start)
    _stream_once(a.base_url, a.model, _prompt(64))

    if a.only_conc:
        _conc_sweep(a, cost, model)
        return

    # 1. single-stream: TTFT residual vs prompt length -> floor + new rate;
    #    TPOT residual -> per-step host floor (whole-step and per-layer forms).
    X, y_ttft, step_resid = [], [], []
    print("# single-stream sweep (miss)", flush=True)
    for p in _SWEEP_PROMPTS:
        tt, tp_, ptoks = zip(*[
            _stream_once(a.base_url, a.model, _prompt(p), out_tokens=_DECODE_TOKENS)
            for _ in range(_SAMPLES)])
        ttft, tpot, ptok = st.median(tt), st.median(tp_), int(st.median(ptoks))
        compute_prefill = cost.prefill_step_ms([ptok])
        compute_decode = cost.decode_step_ms([ptok])
        y_ttft.append(ttft - compute_prefill)
        X.append(float(ptok))
        step_resid.append(tpot - compute_decode)
        print(f"  p={ptok:5d}  ttft={ttft:8.1f}ms (compute {compute_prefill:7.1f})  "
              f"tpot={tpot:6.2f}ms (compute {compute_decode:5.2f})", flush=True)
    intercept, slope = _linfit(X, y_ttft)
    floor, rate_n = max(0.0, intercept), max(0.0, slope)
    step_over = max(0.0, st.median(step_resid))
    dov = step_over / max(1, int(model.n_layers))

    # 2. APC pairs: same prompt twice -> hit-TTFT residual vs cached tokens.
    print("# APC-hit sweep (cached)", flush=True)
    Xc, y_hit = [], []
    for p in _SWEEP_PROMPTS:
        prompt = _prompt(p)
        _stream_once(a.base_url, a.model, prompt)          # prime the cache
        tt, _tp, ptoks = zip(*[
            _stream_once(a.base_url, a.model, _prompt(0, reuse=prompt))
            for _ in range(_SAMPLES)])
        ttft, ptok = st.median(tt), int(st.median(ptoks))
        # vLLM re-prefills the last block of a full-hit prompt; compute is ~one block.
        compute_hit = cost.prefill_step_ms([max(1, int(model.cache_block_size))])
        Xc.append(float(ptok))
        y_hit.append(ttft - compute_hit)
        print(f"  p={ptok:5d}  hit ttft={ttft:8.1f}ms", flush=True)
    _c0, slope_c = _linfit(Xc, y_hit)
    rate_c = max(0.0, slope_c)

    # 3. concurrency sweep (see _conc_sweep).
    if a.skip_conc:
        mult_pts, conc_rows = [(1, 1.0)], []
    else:
        mult_pts, conc_rows = _conc_sweep(a, cost, model)

    print(f"\n[{a.gpu_label} / {a.model} / tp{a.tp}] live-server frontend probe\n")
    print(f"# paste into device_spec/{a.gpu_label.lower()}.yaml (serving_frontend_probe.py):")
    print("frontend:")
    print(f"  floor_ms: {round(floor, 2)}")
    print(f"  new_ms_per_token: {round(rate_n, 6)}")
    print(f"  cached_ms_per_token: {round(rate_c, 6)}")
    print(f"  mult_curve: {[[c, m] for c, m in mult_pts]}  # measured resid ratio; lanes flat")
    print("  lanes_curve: []   # not separable from mult in this probe; leave flat")
    print(f"  # measured (c, median ttft ms, sim frontend-off ms): "
          f"{[(c, round(m, 1), round(p, 1)) for c, m, p, _ in conc_rows]}")
    print("serving:")
    print(f"  decode_overhead_ms_per_layer: {round(dov, 5)}   # = step residual {round(step_over, 3)}ms / {model.n_layers} layers")
    print(f"  ttft_overhead_ms: {round(floor, 2)}")
    print("  max_concurrent_prefills: 2")
    print(f"  kernel_ttft_overhead_ms: {round(floor, 2)}   # per-request first-token host floor")
    print(f"  kernel_step_overhead_ms: {round(step_over, 3)}   # per-step host floor (whole step)")
    print("  kernel_step_overlap_per_ms: 0.0   # measure with a high-conc TPOT sweep before setting")


if __name__ == "__main__":
    main()
