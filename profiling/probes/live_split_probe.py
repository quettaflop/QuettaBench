#!/usr/bin/env python3
# profiling/probes/live_split_probe.py
"""Live concurrency sweep — measure the cached-host SHARED vs PER-REQ split on the REAL serving stack.

The offline batch CSV gave the wrong split (it lacked the HTTP/IPC stack that dominates the cached rate).
This fires B CONCURRENT cache-hit requests (shared primed prefix P + a fresh small new tail each) at the live
vLLM server and measures per-request TTFT, then fits TTFT(B,P) = floor + shared*P + perreq*B*P — the exact
decomposition the headline applies (shared once/step + perreq per request). Reports the shared/perreq split.
"""
import asyncio, json, time, random, csv, argparse, statistics as st
import aiohttp

_VOCAB = ("the of and to in a is that for it as was with on be at by this had not are but from or "
          "have an they which one you were her all she there would their we him been has when who "
          "will more no if out so up said what its about into than them can only other new some").split()
_RNG = random.Random(0)
_WORDS = [_RNG.choice(_VOCAB) for _ in range(40000)]
_TAIL = random.Random(777)
NEW = 8


async def ttft_once(session, url, model, content):
    payload = {"model": model, "messages": [{"role": "user", "content": content}],
               "max_tokens": 1, "temperature": 0.0, "stream": True}
    t0 = time.perf_counter()
    async with session.post(url, json=payload, headers={"Authorization": "Bearer test"}) as resp:
        async for raw in resp.content:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            ds = line[len("data:"):].strip()
            if ds == "[DONE]":
                break
            ch = json.loads(ds); cc = ch.get("choices", [])
            if cc and cc[0].get("delta", {}).get("content") is not None:
                return (time.perf_counter() - t0) * 1000.0
    return None


async def measure(session, url, model, B, P, rounds):
    prefix = " ".join(_WORDS[:int(P * 0.96)])
    await ttft_once(session, url, model, prefix)  # prime the shared prefix (HIT for all B)
    meds = []
    for _ in range(rounds + 1):  # +1 warmup round
        contents = [prefix + " " + " ".join(_TAIL.choice(_VOCAB) for _ in range(int(NEW * 0.96))) for _ in range(B)]
        ts = await asyncio.gather(*[ttft_once(session, url, model, c) for c in contents])
        meds.append(st.median([t for t in ts if t]))
    return st.median(meds[1:])


def fit(rows):
    import numpy as np
    X = np.array([[1.0, r["P"], r["B"] * r["P"]] for r in rows]); y = np.array([r["ttft_ms"] for r in rows])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b  # floor, shared(/tok), perreq(/tok)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8771/v1/chat/completions")
    ap.add_argument("--model", default="llama")
    ap.add_argument("--Bs", default="1,2,4,8,16")
    ap.add_argument("--Ps", default="2000,8000,16000")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default="live_split.csv")
    a = ap.parse_args()
    Bs = [int(x) for x in a.Bs.split(",")]; Ps = [int(x) for x in a.Ps.split(",")]
    rows = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600),
                                     connector=aiohttp.TCPConnector(limit=64)) as s:
        for P in Ps:
            for B in Bs:
                t = await measure(s, a.url, a.model, B, P, a.rounds)
                rows.append({"B": B, "P": P, "ttft_ms": round(t, 3)})
                print(f"  B={B:>3} P={P:>6}  ttft={t:8.2f} ms", flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["B", "P", "ttft_ms"]); w.writeheader(); w.writerows(rows)
    fl, sh, pr = fit(rows)
    tot = sh + pr
    print(f"\n=== TTFT(B,P) = floor + shared*P + perreq*B*P  (live serving stack) ===")
    print(f"  floor={fl:7.2f} ms | shared={sh*1000:7.3f} ms/1k | perreq={pr*1000:7.3f} ms/1k")
    if tot > 0:
        print(f"  B=1 cached rate = {tot*1000:.3f} ms/1k (cf fit 6.1) | SPLIT shared {sh/tot*100:.1f}% / perreq {pr/tot*100:.1f}%  (code: 57/43)")


if __name__ == "__main__":
    asyncio.run(main())
