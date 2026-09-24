#!/usr/bin/env python3
"""Tokenize a real agent-bench trace into the crossval request shape.

Each session turn becomes one request: prompt_text -> prompt_token_ids and the
assistant completion (reasoning_text + content_text) -> output_token_ids, so
trace_serve can measure real acceptance (SPECSUM) against tokens a model actually
produced. Needs the serving model's tokenizer.
"""
import argparse
import json
import sys


def r2_to_requests(records, encode):
    """Flatten agent-bench sessions to requests. encode(text) -> list[int] is the
    model tokenizer; the assistant reference is reasoning_text + content_text."""
    reqs = []
    for rec in records:
        for t in rec.get("turns", []):
            prompt = t.get("prompt_text")
            if not prompt:
                continue
            out = (t.get("reasoning_text") or "") + (t.get("content_text") or "")
            pids = encode(prompt)
            req = {
                "prompt_token_ids": pids,
                "prompt_len": len(pids),
                "session_id": rec.get("session_id"),
                "turn": t.get("turn_index", 0),
                "source": rec.get("model", "R2"),
            }
            if out:
                oids = encode(out)
                req["output_token_ids"] = oids
                req["output_length"] = len(oids)
            reqs.append(req)
    return reqs


def main():
    ap = argparse.ArgumentParser(description="tokenize an agent-bench trace for replay")
    ap.add_argument("trace", help="agent-bench JSONL")
    ap.add_argument("--model", required=True, help="tokenizer or model dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    def encode(s):
        return tok(s, add_special_tokens=False)["input_ids"]
    records = [json.loads(l) for l in open(args.trace) if l.strip()]
    reqs = r2_to_requests(records, encode)
    if not reqs:
        sys.exit(f"{args.trace}: no turns with prompt_text")
    with open(args.out, "w") as fh:
        for r in reqs:
            fh.write(json.dumps(r) + "\n")
    n_out = sum(1 for r in reqs if r.get("output_token_ids"))
    print(f"wrote {len(reqs)} reqs ({n_out} with output tokens) to {args.out}")


if __name__ == "__main__":
    main()
