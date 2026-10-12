"""Live-server profiler sources that price against QuettaSim: the device-YAML frontend and serving blocks, and a PD pair's host cost.

Imports QuettaSim's engine package, so native.py loads it only when one of these sources is asked for. Standalone: python -m quettabench.profiling.frontend {frontend|pd_host} --device-yaml <yaml> ..."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import random
import statistics as st
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from engine.backends.kernel_composed_cost import KernelComposedCost
from engine.core.cost import step_host_ms
from engine.factory import build_kernel_composed_cost
from engine.loaders.config_loader import load_kernel_gpu, load_kernel_model
from engine.loaders.workload import Turn
from engine.sim.closed_loop import predict_cell
from engine.sim.disagg import Link
from kernel_composed.types import FrontendParams

from quettabench.engines import launcher

_SWEEP_PROMPTS = [128, 256, 512, 1024, 2048]  # tokens
_CONC_SWEEP = [1, 2, 5, 10, 20, 40]
_SAMPLES = 3  # repeats per point, median taken
_DECODE_TOKENS = 64  # enough output tokens for a stable TPOT


def _stream_once(base_url: str, model: str, prompt: str, out_tokens: int = 8) -> tuple[float, float, int]:
    """(ttft_ms, tpot_ms, prompt_tokens) for one streamed completion."""
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": out_tokens,
            "stream": True,
            "temperature": 0.0,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(f"{base_url}/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first_at = None
    n = 0
    prompt_tokens = 0
    with launcher.urlopen(req) as r:
        for line in r:
            if not line.startswith(b"data:") or b"[DONE]" in line:
                continue
            payload = json.loads(line[5:])
            usage = payload.get("usage")
            if usage:  # the final usage-only chunk carries no token
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
    """Concurrent herds of unique ~512-token prompts; returns (mult_curve points, rows).

    mult is (measured median - sim frontend-off median) over the c=1 residual. Each point runs an unmeasured herd first because a new batch shape triggers one-time autotune work."""

    def herd(c: int) -> list[tuple[float, float, int]]:
        prompts = [_prompt(512) for _ in range(c)]
        with ThreadPoolExecutor(max_workers=c) as ex:
            return list(ex.map(lambda pr: _stream_once(a.base_url, a.model, pr), prompts))

    print("# concurrent herd sweep (512-token prompts; warmup herd per point)", flush=True)
    herd(max(_CONC_SWEEP))
    conc_rows = []
    for c in _CONC_SWEEP:
        herd(c)
        res = herd(c)
        meas = st.median([r[0] for r in res])
        ptok = int(st.median([r[2] for r in res]))
        turn = Turn(cache_hit_tokens=0.0, new_prefill_tokens=float(ptok), osl_tokens=8)
        pred_ttfts, _ = predict_cell(cost, [turn], c)
        pred = pred_ttfts[0] if pred_ttfts else 0.0
        conc_rows.append((c, meas, pred, meas - pred))
        print(
            f"  c={c:3d}  measured={meas:8.1f}ms  sim(frontend-off)={pred:8.1f}ms  resid={meas - pred:7.1f}ms",
            flush=True,
        )
    base_resid = max(1e-6, conc_rows[0][3])
    mult_pts = [(c, max(1.0, round(resid / base_resid, 2))) for c, _m, _p, resid in conc_rows]
    if a.only_conc:
        print(f"\n  mult_curve: {[[c, m] for c, m in mult_pts]}  # measured resid ratio; lanes flat")
        print(
            f"  # measured (c, median ttft ms, sim frontend-off ms): "
            f"{[(c, round(m, 1), round(p, 1)) for c, m, p, _ in conc_rows]}"
        )
    return mult_pts, conc_rows


_SEED = [0]


def _prompt(tokens: int, *, reuse: str | None = None) -> str:
    """~`tokens` tokens behind a unique seed so sweep prompts share no cached prefix; `reuse` returns that prompt unchanged."""
    if reuse is not None:
        return reuse
    _SEED[0] += 1
    return f"seed{_SEED[0]:06d} " + "hi " * max(1, tokens - 4)


def frontend_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-label", required=True)
    ap.add_argument("--model", required=True, help="served model name (OpenAI 'model' field)")
    ap.add_argument("--model-yaml", required=True, help="engine model YAML for the compute prediction")
    ap.add_argument(
        "--device-yaml",
        required=True,
        help="QuettaSim's device_spec/<gpu>.yaml, e.g. ../QuettaSim/device_spec/h200.yaml",
    )
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--skip-conc", action="store_true", help="skip the concurrent herd sweep (parts 1-2 only)")
    ap.add_argument(
        "--only-conc", action="store_true", help="run only the concurrent herd sweep (parts 1-2 already collected)"
    )
    ap.add_argument("--out", default=None, help="write frontend_probe()'s dict here as JSON")
    return ap


def frontend_probe(a):
    """Sweep a.base_url and return {frontend, serving, conc}; conc rows are [c, median ttft ms, sim frontend-off ms].

    Each value is measured latency minus the compute-only prediction."""
    gpu = load_kernel_gpu(Path(a.device_yaml))
    # Strip the host terms the yaml already carries, or the residual counts them twice.
    gpu = dataclasses.replace(
        gpu,
        frontend=FrontendParams(),
        kernel_ttft_overhead_ms=0.0,
        kernel_step_overhead_ms=0.0,
        kernel_step_overlap_per_ms=0.0,
        decode_overhead_ms=0.0,
        decode_overhead_ms_per_layer=0.0,
    )
    model = load_kernel_model(Path(a.model_yaml))
    cost = KernelComposedCost(model, gpu, tp=a.tp)

    _stream_once(a.base_url, a.model, _prompt(64))

    if a.only_conc:
        mult_pts, conc_rows = _conc_sweep(a, cost, model)
        return {
            "frontend": {"mult_curve": [[c, m] for c, m in mult_pts]},
            "conc": [[c, m, p] for c, m, p, _ in conc_rows],
        }

    X, y_ttft, step_resid = [], [], []
    print("# single-stream sweep (miss)", flush=True)
    for p in _SWEEP_PROMPTS:
        tt, tp_, ptoks = zip(
            *[_stream_once(a.base_url, a.model, _prompt(p), out_tokens=_DECODE_TOKENS) for _ in range(_SAMPLES)]
        )
        ttft, tpot, ptok = st.median(tt), st.median(tp_), int(st.median(ptoks))
        compute_prefill = cost.prefill_step_ms([ptok])
        compute_decode = cost.decode_step_ms([ptok])
        y_ttft.append(ttft - compute_prefill)
        X.append(float(ptok))
        step_resid.append(tpot - compute_decode)
        print(
            f"  p={ptok:5d}  ttft={ttft:8.1f}ms (compute {compute_prefill:7.1f})  "
            f"tpot={tpot:6.2f}ms (compute {compute_decode:5.2f})",
            flush=True,
        )
    intercept, slope = _linfit(X, y_ttft)
    floor, rate_n = max(0.0, intercept), max(0.0, slope)
    step_over = max(0.0, st.median(step_resid))
    dov = step_over / max(1, int(model.n_layers))

    print("# APC-hit sweep (cached)", flush=True)
    Xc, y_hit = [], []
    for p in _SWEEP_PROMPTS:
        prompt = _prompt(p)
        _stream_once(a.base_url, a.model, prompt)
        tt, _tp, ptoks = zip(*[_stream_once(a.base_url, a.model, _prompt(0, reuse=prompt)) for _ in range(_SAMPLES)])
        ttft, ptok = st.median(tt), int(st.median(ptoks))
        # vLLM re-prefills the last block of a full hit.
        compute_hit = cost.prefill_step_ms([max(1, int(model.cache_block_size))])
        Xc.append(float(ptok))
        y_hit.append(ttft - compute_hit)
        print(f"  p={ptok:5d}  hit ttft={ttft:8.1f}ms", flush=True)
    _c0, slope_c = _linfit(Xc, y_hit)
    rate_c = max(0.0, slope_c)

    if a.skip_conc:
        mult_pts, conc_rows = [(1, 1.0)], []
    else:
        mult_pts, conc_rows = _conc_sweep(a, cost, model)

    print(f"\n[{a.gpu_label} / {a.model} / tp{a.tp}] live-server frontend probe\n")
    print(f"# paste into device_spec/{a.gpu_label.lower()}.yaml (quettabench.profiling.frontend):")
    print("frontend:")
    print(f"  floor_ms: {round(floor, 2)}")
    print(f"  new_ms_per_token: {round(rate_n, 6)}")
    print(f"  cached_ms_per_token: {round(rate_c, 6)}")
    print(f"  mult_curve: {[[c, m] for c, m in mult_pts]}  # measured resid ratio; lanes flat")
    print("  lanes_curve: []   # not separable from mult in this probe; leave flat")
    print(
        f"  # measured (c, median ttft ms, sim frontend-off ms): "
        f"{[(c, round(m, 1), round(p, 1)) for c, m, p, _ in conc_rows]}"
    )
    print("serving:")
    print(
        f"  decode_overhead_ms_per_layer: {round(dov, 5)}   # = step residual {round(step_over, 3)}ms / {model.n_layers} layers"
    )
    print(f"  ttft_overhead_ms: {round(floor, 2)}")
    print("  max_concurrent_prefills: 2")
    print(f"  kernel_ttft_overhead_ms: {round(floor, 2)}   # per-request first-token host floor")
    print(f"  kernel_step_overhead_ms: {round(step_over, 3)}   # per-step host floor (whole step)")
    print("  kernel_step_overlap_per_ms: 0.0   # measure with a high-conc TPOT sweep before setting")
    return {
        "frontend": {
            "floor_ms": round(floor, 2),
            "new_ms_per_token": round(rate_n, 6),
            "cached_ms_per_token": round(rate_c, 6),
            "mult_curve": [[c, m] for c, m in mult_pts],
            "lanes_curve": [],
        },
        "serving": {
            "decode_overhead_ms_per_layer": round(dov, 5),
            "ttft_overhead_ms": round(floor, 2),
            "max_concurrent_prefills": 2,
            "kernel_ttft_overhead_ms": round(floor, 2),
            "kernel_step_overhead_ms": round(step_over, 3),
            "kernel_step_overlap_per_ms": 0.0,
        },
        "conc": [[c, m, p] for c, m, p, _ in conc_rows],
    }


def frontend_main(argv=None):
    a = frontend_parser().parse_args(argv)
    out = frontend_probe(a)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2) + "\n")


COLS = (
    "gpu",
    "phase",
    "conc",
    "tokens_total",
    "tokens_new_p",
    "tokens_cached_d",
    "prefill_ms",
    "sim_prefill_ms",
    "p_resid_ms",
    "d_first_ms",
    "sim_d_step_ms",
    "sim_pull_ms",
    "d_resid_ms",
    "client_ttft_ms",
)


class Prompter:
    """Random-word prompts of a target token length, checked with the model tokenizer."""

    def __init__(self, tokenizer_name: str, seed: int = 1234):
        from transformers import AutoTokenizer  # imported here: optional dependency

        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.rng = random.Random(seed)
        # Single-token alphabetic words.
        words = []
        for i in range(1000, min(len(self.tok), 60000)):
            s = self.tok.decode([i])
            if s.startswith(" ") and s[1:].isalpha() and 3 <= len(s) <= 9:
                words.append(s[1:])
        self.words = words

    def count(self, messages) -> int:
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if isinstance(ids, dict) or hasattr(ids, "keys"):
            ids = ids["input_ids"]
        if hasattr(ids, "shape"):
            return int(ids.shape[-1])
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return len(ids)

    def make(self, target_tokens: int, tag: str) -> str:
        n = max(8, target_tokens - 24)
        text = f"probe {tag} " + " ".join(self.rng.choice(self.words) for _ in range(n))
        got = self.count([{"role": "user", "content": text}])
        while got > target_tokens and len(text) > 64:
            cut = int(len(text) * (1 - (got - target_tokens) / got * 0.9)) - 8
            text = text[: max(64, cut)]
            got = self.count([{"role": "user", "content": text}])
        return text


def chat_stream(proxy_url: str, model: str, text: str, rid: str, max_tokens: int = 4) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        f"{proxy_url}/v1/chat/completions", data=body, headers={"Content-Type": "application/json", "X-Request-Id": rid}
    )
    t0 = time.perf_counter()
    first = None
    with launcher.urlopen(req, timeout=3600) as r:
        for line in r:
            if line.startswith(b"data:") and first is None and b'"content"' in line:
                first = time.perf_counter()
    t1 = time.perf_counter()
    return {"rid": rid, "client_ttft_ms": ((first or t1) - t0) * 1e3, "wall_ms": (t1 - t0) * 1e3}


def _loglines(path: Path) -> list[str]:
    try:
        return path.read_text().splitlines()
    except FileNotFoundError:
        return []


def proxy_rows(path: Path, since: int) -> list[dict]:
    lines = _loglines(path)[since:]
    return [json.loads(x) for x in lines if x.strip()]


class SimSide:
    def __init__(self, device_yaml: str, model: str, tp: int, link: Link):
        self.p = build_kernel_composed_cost(device_yaml, model, tp=tp, gpu_mem_util=0.92)
        self.d = build_kernel_composed_cost(device_yaml, model, tp=tp, gpu_mem_util=0.92)
        self.tp = tp
        self.link = link
        self.layers = int(self.p.model.n_layers)

    def prefill_ms(self, new: int, cached: int) -> float:
        budget = float(self.p.sched.max_num_batched_tokens or 8192)
        rem, res, total, first = float(new), float(cached), 0.0, True
        while rem > 0:
            ch = min(rem, budget)
            gpu = self.p.fused_step_ms(int(ch), 0, 0) + self.p.cross_attn_ms(int(ch), res)
            total += gpu + step_host_ms(self.p, gpu, ch, after_idle=first)
            res += ch
            rem -= ch
            first = False
        return total

    def d_step_ms(self, ctx: int) -> float:
        gpu = self.d.fused_step_ms(0, 1, (float(ctx),))
        return gpu + step_host_ms(self.d, gpu, 1, after_idle=True)

    def pull_ms(self, pulled_tokens: int, runs: int = 1) -> float:
        nbytes = pulled_tokens * float(self.p.model.kv_bytes_per_token)
        return self.link.transfer_ms(nbytes, streams=self.tp, n_descs=self.layers * runs)


def lstsq(xs, ys):
    """(a, b) of the least-squares line y = a + b x."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    return my - b * mx, b


def pd_host_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Per-request host cost of a PD-disaggregated vLLM pair, by prompt length.")
    ap.add_argument("--gpu-label", required=True)
    ap.add_argument("--proxy-url", default="http://127.0.0.1:8000")
    ap.add_argument("--proxy-log", required=True, help="the proxy's --log JSONL")
    ap.add_argument("--model", default="qwen3-235b", help="served model name")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8")
    ap.add_argument("--model-yaml", default="qwen3-235b-a22b-fp8")
    ap.add_argument(
        "--device-yaml",
        required=True,
        help="QuettaSim's device_spec/<gpu>.yaml, e.g. ../QuettaSim/device_spec/h200.yaml",
    )
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--lengths", type=int, nargs="+", default=[2048, 8192, 16384, 32768, 49152, 65536, 98304, 122880])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument(
        "--turns",
        type=int,
        default=0,
        help="growing-context multi-turn session length (0 = skip). Each turn "
        "appends --turn-new new tokens, so the prefix grows and is unique -- "
        "the faithful per-cached-token host cost.",
    )
    ap.add_argument(
        "--turn-new", type=int, default=1800, help="new tokens appended per turn in --turns mode (E6's median new)"
    )
    ap.add_argument("--herd", type=int, default=16)
    ap.add_argument("--herd-tokens", type=int, default=32768)
    ap.add_argument("--link", default="2.2e11,0,25", help="bw_bytes_per_s,latency_us,per_desc_us of the pull")
    ap.add_argument("--out", help="CSV path; pd_host_main() requires it, pd_host_probe() ignores it")
    return ap


def pd_host_probe(a) -> list[dict]:
    """Run the cold, warm, turn and herd phases through a.proxy_url; one COLS row per scored request.

    Residuals are the proxy log's latency minus the sim prediction: P is prefill_s - sim prefill, D is d_first - sim first step - sim pull."""
    bw, lat, pdu = (float(x) for x in a.link.split(","))
    sim = SimSide(a.device_yaml, a.model_yaml, a.tp, Link(bw, lat, pdu))
    pr = Prompter(a.tokenizer)
    log = Path(a.proxy_log)
    rows: list[dict] = []

    def score(phase: str, conc: int, client: dict, prox: dict) -> dict:
        pu = prox["prefill_usage"]
        total = int(pu["prompt_tokens"])
        cached_p = int(pu.get("cached_tokens") or 0)
        new_p = total - cached_p
        prefill_ms = prox["prefill_s"] * 1e3
        d_first = (prox["first_token_at"] - prox["decode_start"]) * 1e3
        # D holds the previous turn only in the warm phases.
        cached_d = total if phase == "cold" else int(prox.get("_cached_d", 0))
        pulled = max(0, total - (cached_d if phase != "cold" else 0))
        sp = sim.prefill_ms(new_p, cached_p)
        sd = sim.d_step_ms(total)
        spull = sim.pull_ms(pulled)
        r = {
            "gpu": a.gpu_label,
            "phase": phase,
            "conc": conc,
            "tokens_total": total,
            "tokens_new_p": new_p,
            "tokens_cached_d": cached_d if phase != "cold" else 0,
            "prefill_ms": round(prefill_ms, 1),
            "sim_prefill_ms": round(sp, 1),
            "p_resid_ms": round(prefill_ms - sp, 1),
            "d_first_ms": round(d_first, 1),
            "sim_d_step_ms": round(sd, 1),
            "sim_pull_ms": round(spull, 1),
            "d_resid_ms": round(d_first - sd - spull, 1),
            "client_ttft_ms": round(client["client_ttft_ms"], 1),
        }
        print(
            f"  {phase:5s} c{conc:<3} total={total:>7} new_p={new_p:>7} | P {prefill_ms:8.1f} ms "
            f"(sim {sp:7.1f}, resid {r['p_resid_ms']:8.1f}) | D-first {d_first:8.1f} ms "
            f"(step {sd:5.1f} + pull {spull:5.1f}, resid {r['d_resid_ms']:8.1f})",
            flush=True,
        )
        return r

    # Unscored warmup: NIXL handshake and cudagraph paths.
    chat_stream(a.proxy_url, a.model, pr.make(512, "warmup0"), "probe-warmup")
    time.sleep(1.0)

    for L in a.lengths:
        for rep in range(a.repeats):
            text = pr.make(L, f"L{L}r{rep}")
            since = len(_loglines(log))
            c = chat_stream(a.proxy_url, a.model, text, f"probe-cold-L{L}-r{rep}")
            time.sleep(0.3)
            px = proxy_rows(log, since)
            rows.append(score("cold", 1, c, px[-1]))
            cold_total = int(px[-1]["prefill_usage"]["prompt_tokens"])
            since = len(_loglines(log))
            c = chat_stream(
                a.proxy_url, a.model, text + " Now reply with the single word OK.", f"probe-warm-L{L}-r{rep}"
            )
            time.sleep(0.3)
            px = proxy_rows(log, since)
            px[-1]["_cached_d"] = cold_total + 4  # cold prompt plus its 4 generated tokens
            rows.append(score("warm", 1, c, px[-1]))

    if a.turns > 0:
        # One growing session: each turn extends the previous context with fresh words.
        ctx = pr.make(a.turn_new, "T0")
        prev_total = 0
        for t in range(a.turns):
            if t > 0:
                ctx = ctx + " " + " ".join(pr.rng.choice(pr.words) for _ in range(a.turn_new))
            since = len(_loglines(log))
            c = chat_stream(a.proxy_url, a.model, ctx, f"probe-turn-{t}")
            time.sleep(0.3)
            px = proxy_rows(log, since)
            if px:
                px[-1]["_cached_d"] = prev_total
                rows.append(score("turn", 1, c, px[-1]))
                prev_total = int(px[-1]["prefill_usage"]["prompt_tokens"]) + 4

    if a.herd > 1:
        texts = [pr.make(a.herd_tokens, f"H{i}") for i in range(a.herd)]
        totals = []
        for i, t in enumerate(texts):
            since = len(_loglines(log))
            chat_stream(a.proxy_url, a.model, t, f"probe-herdseed-{i}")
            time.sleep(0.2)
            totals.append(int(proxy_rows(log, since)[-1]["prefill_usage"]["prompt_tokens"]))
        since = len(_loglines(log))
        results: list[dict] = [None] * a.herd

        def fire(i):
            results[i] = chat_stream(
                a.proxy_url, a.model, texts[i] + " Now reply with the single word OK.", f"probe-herd-{i}"
            )

        th = [threading.Thread(target=fire, args=(i,)) for i in range(a.herd)]
        for t in th:
            t.start()
        for t in th:
            t.join()
        time.sleep(0.5)
        px = {r["request_id"]: r for r in proxy_rows(log, since)}
        for i in range(a.herd):
            p = px.get(f"probe-herd-{i}")
            if p:
                p["_cached_d"] = totals[i] + 4
                rows.append(score("herd", a.herd, results[i], p))
    return rows


def pd_host_main(argv=None) -> None:
    ap = pd_host_parser()
    a = ap.parse_args(argv)
    if not a.out:
        ap.error("--out is required")
    rows = pd_host_probe(a)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COLS))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")

    print("\nresidual fits (ms = floor + slope * tokens), c1:")
    for phase, xkey, label in (
        ("cold", "tokens_total", "P cold  (all new)"),
        ("warm", "tokens_total", "P warm  (identical resend)"),
        ("turn", "tokens_cached_d", "P turn  (growing prefix)"),
        ("cold", "tokens_total", "D cold"),
        ("warm", "tokens_total", "D warm"),
        ("turn", "tokens_cached_d", "D turn  (growing prefix)"),
    ):
        sel = [r for r in rows if r["phase"] == phase and r["conc"] == 1]
        if len(sel) < 3:
            continue
        ykey = "p_resid_ms" if label.startswith("P") else "d_resid_ms"
        f0, sl = lstsq([r[xkey] for r in sel], [r[ykey] for r in sel])
        print(f"  {label:<22} floor {f0:8.1f} ms   {sl * 1e3:7.2f} us/token   (n={len(sel)})")
    herd = [r for r in rows if r["phase"] == "herd"]
    warm = [
        r
        for r in rows
        if r["phase"] == "warm" and r["conc"] == 1 and abs(r["tokens_total"] - a.herd_tokens) < 0.15 * a.herd_tokens
    ]
    if herd and warm:
        print(
            f"\nherd c{a.herd} at ~{a.herd_tokens} tokens (warm): P resid median "
            f"{st.median(r['p_resid_ms'] for r in herd):.0f} ms vs c1 {st.median(r['p_resid_ms'] for r in warm):.0f} ms; "
            f"D resid median {st.median(r['d_resid_ms'] for r in herd):.0f} ms vs c1 "
            f"{st.median(r['d_resid_ms'] for r in warm):.0f} ms"
        )


MAINS = {"frontend": frontend_main, "pd_host": pd_host_main}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in MAINS:
        sys.exit(f"usage: python -m quettabench.profiling.frontend {{{'|'.join(MAINS)}}} [flags]")
    MAINS[argv[0]](argv[1:])


if __name__ == "__main__":
    main()
