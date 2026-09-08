#!/usr/bin/env python3
"""Per-request HOST cost of a PD-disaggregated vLLM pair, by prompt length.

The kernel-composed sim prices GPU work from measured kernels and the KV hand-off from
kv_transfer_probe.py; what is left in a PD request's TTFT is host work that scales with
the WHOLE prompt -- JSON parsing, chat-template rendering, tokenizing, block hashing,
connector bookkeeping -- paid once on P and once more on D (the proxy re-sends the full
prompt to D after P answers). The device YAML's ``frontend:`` block carries that model;
its H200 values were measured on <= 2048-token prompts, and agentic contexts are
10k-120k tokens. This probe measures the same residual on long prompts, against a live
P/D pair + proxy launched exactly like the GT servers, with CONTROLLED SYNTHETIC prompts
(random vocabulary words, never a GT trace), so nothing here is fitted on evaluation data.

Per prompt length L:
  cold   a unique L-token prompt              -> P prefills everything (new = L)
  warm   the same prompt + one short sentence -> P: prefix hit on L (new ~ 16); D: holds
                                                 the cold turn's KV, pulls only the delta
  turn   a GROWING agentic session: turn t appends ~`--turn-new` genuinely new tokens to
         the running context (like a real tool-call turn), so each turn's prefix is UNIQUE
         and longer than the last. This is the faithful multi-turn host cost -- the `warm`
         phase re-sends the IDENTICAL prompt (APC's best case) and so under-measures the
         per-cached-token host of a real session, where every turn re-tokenizes + re-hashes
         + rebuilds the block table for a growing unique prefix. The pd_sweep GT prefill
         decomposes to ~9.5 us/cached-token at these shapes, ~3x the identical-resend warm
         slope, and this phase is what measures that honestly (on synthetic content).
Both are read from the proxy's per-request log (prefill_s; decode_start -> first_token_at)
and compared with the sim's engine-side prediction for that exact request (chunked
prefill + cross-attention on P; one decode step + the measured pull on D):

  P_host(L, new) = prefill_s      - sim_prefill_ms
  D_host(L)      = d_first_ms     - sim_first_step_ms - sim_pull_ms

A herd round (``--herd`` warm requests at ``--herd-tokens`` fired together) gives the
contention multiplier at that concurrency.

  # servers + proxy up (see scratchpad/pd_probe/launch.sh for the E6-shaped pair)
  /opt/vllm-local/bin/python profiling/probes/pd_host_probe.py --gpu-label H200 \\
      --proxy-log <dir>/proxy_requests.jsonl --out data/kernel_data/cuda_event/frontend/H200_pd_host.csv

Prints per-row residuals and the least-squares floor + per-token slopes for P (cold /
warm) and D; does NOT edit the YAML.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics as st
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / ".." / ".."))
from engine.factory import build_kernel_composed_cost  # noqa: E402
from engine.sim.queue_sim import _step_host_ms  # noqa: E402
from engine.sim.disagg import Link  # noqa: E402

COLS = ("gpu", "phase", "conc", "tokens_total", "tokens_new_p", "tokens_cached_d",
        "prefill_ms", "sim_prefill_ms", "p_resid_ms",
        "d_first_ms", "sim_d_step_ms", "sim_pull_ms", "d_resid_ms", "client_ttft_ms")


# ── prompts ───────────────────────────────────────────────────────────────────
class Prompter:
    """Random-word prompts of a target token length, checked with the model tokenizer."""

    def __init__(self, tokenizer_name: str, seed: int = 1234):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.rng = random.Random(seed)
        # single-token, space-prefixed alphabetic words -> ~1 token per word
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
        n = max(8, target_tokens - 24)                      # leave room for the template
        text = f"probe {tag} " + " ".join(self.rng.choice(self.words) for _ in range(n))
        got = self.count([{"role": "user", "content": text}])
        # random words tokenize near 1:1; trim/pad once to land within a block
        while got > target_tokens and len(text) > 64:
            cut = int(len(text) * (1 - (got - target_tokens) / got * 0.9)) - 8
            text = text[:max(64, cut)]
            got = self.count([{"role": "user", "content": text}])
        return text


# ── requests ──────────────────────────────────────────────────────────────────
def chat_stream(proxy_url: str, model: str, text: str, rid: str, max_tokens: int = 4) -> dict:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": text}],
                       "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(f"{proxy_url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "X-Request-Id": rid})
    t0 = time.perf_counter()
    first = None
    with urllib.request.urlopen(req, timeout=3600) as r:
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


# ── sim-side prediction of the same request ───────────────────────────────────
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
            total += gpu + _step_host_ms(self.p, gpu, ch, after_idle=first)
            res += ch; rem -= ch; first = False
        return total

    def d_step_ms(self, ctx: int) -> float:
        gpu = self.d.fused_step_ms(0, 1, (float(ctx),))
        return gpu + _step_host_ms(self.d, gpu, 1, after_idle=True)

    def pull_ms(self, pulled_tokens: int, runs: int = 1) -> float:
        nbytes = pulled_tokens * float(self.p.model.kv_bytes_per_token)
        return self.link.transfer_ms(nbytes, streams=self.tp, n_descs=self.layers * runs)


def lstsq(xs, ys):
    """y = a + b x by least squares (pure python; tiny n)."""
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    return my - b * mx, b


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu-label", required=True)
    ap.add_argument("--proxy-url", default="http://127.0.0.1:8000")
    ap.add_argument("--proxy-log", required=True, help="the proxy's --log JSONL")
    ap.add_argument("--model", default="qwen3-235b", help="served model name")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8")
    ap.add_argument("--model-yaml", default="qwen3-235b-a22b-fp8")
    ap.add_argument("--device-yaml", default=str(HERE / ".." / ".." / "device_spec" / "h200.yaml"))
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[2048, 8192, 16384, 32768, 49152, 65536, 98304, 122880])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--turns", type=int, default=0,
                    help="growing-context multi-turn session length (0 = skip). Each turn "
                         "appends --turn-new new tokens, so the prefix grows and is unique -- "
                         "the faithful per-cached-token host cost (see module docstring).")
    ap.add_argument("--turn-new", type=int, default=1800,
                    help="new tokens appended per turn in --turns mode (E6's median new)")
    ap.add_argument("--herd", type=int, default=16)
    ap.add_argument("--herd-tokens", type=int, default=32768)
    ap.add_argument("--link", default="2.2e11,0,25", help="bw_bytes_per_s,latency_us,per_desc_us of the pull")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    bw, lat, pdu = (float(x) for x in a.link.split(","))
    sim = SimSide(a.device_yaml, a.model_yaml, a.tp, Link(bw, lat, pdu))
    pr = Prompter(a.tokenizer)
    log = Path(a.proxy_log)
    rows: list[dict] = []

    def score(phase: str, conc: int, client: dict, prox: dict) -> dict:
        pu, du = prox["prefill_usage"], prox.get("decode_usage") or {}
        total = int(pu["prompt_tokens"]); cached_p = int(pu.get("cached_tokens") or 0)
        new_p = total - cached_p
        prefill_ms = prox["prefill_s"] * 1e3
        d_first = (prox["first_token_at"] - prox["decode_start"]) * 1e3
        # D holds the previous turn of this prompt only in the warm phases
        cached_d = total if phase == "cold" else int(prox.get("_cached_d", 0))
        pulled = max(0, total - (cached_d if phase != "cold" else 0))
        sp = sim.prefill_ms(new_p, cached_p)
        sd = sim.d_step_ms(total)
        spull = sim.pull_ms(pulled)
        r = {"gpu": a.gpu_label, "phase": phase, "conc": conc, "tokens_total": total,
             "tokens_new_p": new_p, "tokens_cached_d": cached_d if phase != "cold" else 0,
             "prefill_ms": round(prefill_ms, 1), "sim_prefill_ms": round(sp, 1),
             "p_resid_ms": round(prefill_ms - sp, 1),
             "d_first_ms": round(d_first, 1), "sim_d_step_ms": round(sd, 1),
             "sim_pull_ms": round(spull, 1), "d_resid_ms": round(d_first - sd - spull, 1),
             "client_ttft_ms": round(client["client_ttft_ms"], 1)}
        print(f"  {phase:5s} c{conc:<3} total={total:>7} new_p={new_p:>7} | P {prefill_ms:8.1f} ms "
              f"(sim {sp:7.1f}, resid {r['p_resid_ms']:8.1f}) | D-first {d_first:8.1f} ms "
              f"(step {sd:5.1f} + pull {spull:5.1f}, resid {r['d_resid_ms']:8.1f})", flush=True)
        return r

    # warm the pair once (NIXL handshake, cudagraph paths) -- not scored
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
            # same session, one more sentence: P prefix hit, D holds the cold turn's KV
            since = len(_loglines(log))
            c = chat_stream(a.proxy_url, a.model, text + " Now reply with the single word OK.",
                            f"probe-warm-L{L}-r{rep}")
            time.sleep(0.3)
            px = proxy_rows(log, since)
            px[-1]["_cached_d"] = cold_total + 4          # cold prompt + its 4 generated tokens
            rows.append(score("warm", 1, c, px[-1]))

    if a.turns > 0:
        # One growing session: turn t sends (running context + turn-new fresh words), so the
        # server sees a UNIQUE, lengthening prefix each turn -- APC hits all but the last
        # turn's tail, exactly the pd_sweep regime. This measures host over a real growing
        # context, not the identical-resend best case the `warm` phase captures.
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
        for i, t in enumerate(texts):                      # seed every session at c1
            since = len(_loglines(log))
            chat_stream(a.proxy_url, a.model, t, f"probe-herdseed-{i}")
            time.sleep(0.2)
            totals.append(int(proxy_rows(log, since)[-1]["prefill_usage"]["prompt_tokens"]))
        since = len(_loglines(log))
        results: list[dict] = [None] * a.herd
        def fire(i):
            results[i] = chat_stream(a.proxy_url, a.model, texts[i] + " Now reply with the single word OK.",
                                     f"probe-herd-{i}")
        th = [threading.Thread(target=fire, args=(i,)) for i in range(a.herd)]
        for t in th: t.start()
        for t in th: t.join()
        time.sleep(0.5)
        px = {r["request_id"]: r for r in proxy_rows(log, since)}
        for i in range(a.herd):
            p = px.get(f"probe-herd-{i}")
            if p:
                p["_cached_d"] = totals[i] + 4
                rows.append(score("herd", a.herd, results[i], p))

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COLS)); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")

    print("\nresidual fits (ms = floor + slope * tokens), c1:")
    for phase, xkey, label in (("cold", "tokens_total", "P cold  (all new)"),
                               ("warm", "tokens_total", "P warm  (identical resend)"),
                               ("turn", "tokens_cached_d", "P turn  (growing prefix)"),
                               ("cold", "tokens_total", "D cold"), ("warm", "tokens_total", "D warm"),
                               ("turn", "tokens_cached_d", "D turn  (growing prefix)")):
        sel = [r for r in rows if r["phase"] == phase and r["conc"] == 1]
        if len(sel) < 3:
            continue
        ykey = "p_resid_ms" if label.startswith("P") else "d_resid_ms"
        f0, sl = lstsq([r[xkey] for r in sel], [r[ykey] for r in sel])
        print(f"  {label:<22} floor {f0:8.1f} ms   {sl * 1e3:7.2f} us/token   (n={len(sel)})")
    herd = [r for r in rows if r["phase"] == "herd"]
    warm = [r for r in rows if r["phase"] == "warm" and r["conc"] == 1
            and abs(r["tokens_total"] - a.herd_tokens) < 0.15 * a.herd_tokens]
    if herd and warm:
        print(f"\nherd c{a.herd} at ~{a.herd_tokens} tokens (warm): P resid median "
              f"{st.median(r['p_resid_ms'] for r in herd):.0f} ms vs c1 {st.median(r['p_resid_ms'] for r in warm):.0f} ms; "
              f"D resid median {st.median(r['d_resid_ms'] for r in herd):.0f} ms vs c1 "
              f"{st.median(r['d_resid_ms'] for r in warm):.0f} ms")


if __name__ == "__main__":
    main()
