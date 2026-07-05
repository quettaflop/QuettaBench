#!/usr/bin/env python3
# profiling/probes/live_ttft_probe.py
"""Live vLLM OpenAI-server c1 TTFT probe — does the HTTP/chat path reproduce the serving cached rate 6.1 ms/1k?

Mirrors prefill_stage_split.py but over the FULL serving path (aiohttp streaming chat/completions, same as the
benchmark client). For each (new, cached): prime the cached prefix (a HIT), then measure TTFT of `prefix + a
FRESH new tail` (new = real miss) at concurrency 1, fresh tail per trial. Regress ttft on (new, cached).
Compare the cached slope to offline 2.4 ms/1k (prefill_stage_split) and the fitted serving 6.1 ms/1k.
"""
import asyncio, json, time, random, csv, argparse, statistics as st
import aiohttp

_VOCAB = ("the of and to in a is that for it as was with on be at by this had not are but from or "
          "have an they which one you were her all she there would their we him been has when who "
          "will more no if out so up said what its about into than them can only other new some "
          "could time these two may then do first any my now such like our over man me even most "
          "made after also did many before must through back years where much your way well down").split()
_RNG = random.Random(0)
_WORDS = [_RNG.choice(_VOCAB) for _ in range(40000)]
_TAIL = random.Random(777)


async def ttft_once(session, url, model, content):
    payload = {"model": model, "messages": [{"role": "user", "content": content}],
               "max_tokens": 1, "temperature": 0.0, "stream": True}
    t0 = time.perf_counter()
    async with session.post(url, json=payload, headers={"Authorization": "Bearer test"}) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
        async for raw in resp.content:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            ds = line[len("data:"):].strip()
            if ds == "[DONE]":
                break
            ch = json.loads(ds)
            ch_ch = ch.get("choices", [])
            if ch_ch and ch_ch[0].get("delta", {}).get("content") is not None:
                return (time.perf_counter() - t0) * 1000.0
    return None


async def measure(session, url, model, cached, new, trials):
    prefix = " ".join(_WORDS[:int(cached * 0.96)])
    if cached > 0:
        await ttft_once(session, url, model, prefix)  # prime -> cached blocks resident
    vals = []
    for _ in range(trials + 1):  # +1 warmup
        tail = " ".join(_TAIL.choice(_VOCAB) for _ in range(int(new * 0.96)))
        content = (prefix + " " + tail) if cached > 0 else tail
        vals.append(await ttft_once(session, url, model, content))
    return st.median(vals[1:])


def ols(rows):
    import numpy as np
    X = np.array([[1.0, r["new"], r["cached"]] for r in rows]); y = np.array([r["ttft_ms"] for r in rows])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8771/v1/chat/completions")
    ap.add_argument("--model", default="llama")
    ap.add_argument("--news", default="8,128,512,1024,2048")
    ap.add_argument("--cacheds", default="0,2000,8000,16000")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out", default="live_ttft.csv")
    a = ap.parse_args()
    news = [int(x) for x in a.news.split(",")]; cacheds = [int(x) for x in a.cacheds.split(",")]
    rows = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as s:
        for cached in cacheds:
            for new in news:
                t = await measure(s, a.url, a.model, cached, new, a.trials)
                rows.append({"new": new, "cached": cached, "ttft_ms": round(t, 3)})
                print(f"  new={new:>5} cached={cached:>6}  ttft={t:7.2f} ms", flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["new", "cached", "ttft_ms"]); w.writeheader(); w.writerows(rows)
    b = ols(rows)
    print(f"\n=== live ttft ~ FLOOR + a*new + b*cached ===")
    print(f"  FLOOR={b[0]:7.2f} ms | new={b[1]*1000:7.3f} ms/1k | cached={b[2]*1000:7.3f} ms/1k")
    print(f"  (offline prefill_stage_split: new 25.3, cached 2.37 ms/1k ; fitted serving: new 31, cached 6.1)")


if __name__ == "__main__":
    asyncio.run(main())
