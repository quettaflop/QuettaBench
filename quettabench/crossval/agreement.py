"""Agreement rows: two runs that returned text or logits, compared request by
request. The comparison arithmetic of the old greedy_agreement.py and
logit_agreement.py; the generation is a serve run with mode.returns.

A returns=text row carries token_ids (and text). A returns=logits row
carries logits = {argmax, top2_gap, logprob}, lists aligned by prompt
position: the argmax id, the top-1 minus top-2 logit gap, and the logprob
of the true next token. theirs is the reference side: its top-2 gap decides
whether an argmax flip is a near tie (the precision floor) or a hard
disagreement, and FIDELITY passes only with zero hard disagreements.

Requests pair by (cell, n-th returned row of that cell) for grid runs and
by request_index (else file order) for the rest.

The FIDELITY gate, vLLM (eager, as logit_agreement.py built it) against the
transformers reference on the same grid prompts:

  qbench run configs/runs/llama.yaml --set mode.grid=smoke --set mode.returns=logits --set 'engine.extra.vllm={enforce_eager: true}' --set run_name=llama-logits
  qbench run configs/runs/llama.yaml --set mode.grid=smoke --set engine.name=reference --set run_name=llama-reference
  qbench crossval results/llama-logits results/llama-reference

Slice verification (the old slice_tf/ harness, tf_verify.py and vllm_tf.py).
On a model cut to N layers (engine.layers, or a checkpoint dir cut the same
way for vLLM) a free-running greedy chain cannot verify numerics: bf16
rounding flips the router within a few steps, and the reference disagrees
with itself across its own kernels. The attainable check is teacher-forced,
which every logits run already is: each position is scored on the same true
prefix, so nothing cascades. The anchor is the reference at float32. The
noise floor is the reference at the engine's dtype (each kernel mode a
run) against that anchor; the engine passes BAND when its argmax flips
against the anchor are no more than the worst floor run's. A floor run is
trusted only once it is deterministic: the same run twice crossvals to
argmax_agree equal to positions and delta max 0.

  qbench run <run>.yaml --set mode.returns=logits --set engine.model=<cut dir> --set run_name=slice-vllm
  qbench run <run>.yaml --set engine.name=reference --set engine.layers=N --set engine.dtype=float32 --set run_name=slice-f32
  qbench run <run>.yaml --set engine.name=reference --set engine.layers=N --set engine.dtype=bfloat16 --set run_name=slice-bf16
  qbench crossval results/slice-vllm results/slice-f32 --floor results/slice-bf16
"""

import statistics

# Reference top-2 gap at or below which an argmax flip is a near tie: one
# bf16 step for logits in [16, 32), where the 2026-09-13 llama check found
# its two flips (V13, probes/runbooks/slice_verification.md).
# logit_agreement.py used < 0.1 and so counted those two ties as hard
# disagreements.
NEAR_TIE_GAP = 0.125
AGREE_MIN = 4        # leading tokens counted as "agrees" in the greedy summary


def returned(rows, name):
    """{request key: value of name} for the rows that carry it."""
    out, seen = {}, {}
    for i, r in enumerate(rows):
        if r.get(name) is None:
            continue
        cell = r.get("cell")
        if cell is not None:
            group = (cell["ctx"], cell["bs"], cell["tp"])
            n = seen[group] = seen.get(group, -1) + 1
            key = f"{group[0]}x{group[1]}" + (f"@tp{group[2]}" if group[2] != 1 else "") + f" #{n}"
        else:
            key = f"req {r.get('request_index', i)}"
        out[key] = r[name]
    return out


def common_prefix(a, b):
    """Leading positions where a and b agree."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def greedy(ours_rows, theirs_rows):
    """One row per request both runs returned tokens for, then a summary."""
    a, b = returned(ours_rows, "token_ids"), returned(theirs_rows, "token_ids")
    rows = []
    for key in sorted(set(a) & set(b)):
        prefix, of = common_prefix(a[key], b[key]), min(len(a[key]), len(b[key]))
        rows.append({"kind": "greedy", "request": key, "prefix": prefix, "of": of,
                     "match_pct": 100.0 * prefix / of if of else 0.0})
    if rows:
        prefixes = [r["prefix"] for r in rows]
        rows.append({"kind": "greedy_summary", "prompts": len(prefixes),
                     "mean_prefix": statistics.mean(prefixes),
                     "median_prefix": statistics.median(prefixes),
                     "agree4plus": sum(1 for p in prefixes if p >= AGREE_MIN)})
    return rows


def logit(ours_rows, theirs_rows):
    """A row per hard disagreement, then the summary with the FIDELITY verdict."""
    a, b = returned(ours_rows, "logits"), returned(theirs_rows, "logits")
    keys = sorted(set(a) & set(b))
    if not keys:
        return []
    rows, total, agree, near_tie, deltas = [], 0, 0, 0, []
    for key in keys:
        ours, ref = a[key], b[key]
        for pos in range(min(len(ours["argmax"]), len(ref["argmax"]))):
            total += 1
            if ours["argmax"][pos] == ref["argmax"][pos]:
                agree += 1
            elif ref["top2_gap"][pos] <= NEAR_TIE_GAP:
                near_tie += 1
            else:
                rows.append({"kind": "logit_disagree", "request": key, "pos": pos,
                             "reference_gap": ref["top2_gap"][pos],
                             "reference": ref["argmax"][pos], "ours": ours["argmax"][pos]})
            if pos < len(ours["logprob"]) and pos < len(ref["logprob"]):
                deltas.append(abs(ours["logprob"][pos] - ref["logprob"][pos]))
    hard = len(rows)
    deltas.sort()
    rows.append({"kind": "logit_summary", "positions": total, "argmax_agree": agree,
                 "near_tie_flips": near_tie, "hard_disagreements": hard,
                 "delta_mean": sum(deltas) / len(deltas) if deltas else None,
                 "delta_p95": deltas[int(0.95 * len(deltas))] if deltas else None,
                 "delta_max": deltas[-1] if deltas else None,
                 "verdict": "SKIP" if not total else "PASS" if hard == 0 else "FAIL"})
    return rows


def flips(rows, anchor_rows):
    """(positions, argmax flips) of rows scored against anchor_rows."""
    summary = logit(rows, anchor_rows)
    if not summary:
        return 0, 0
    return summary[-1]["positions"], summary[-1]["positions"] - summary[-1]["argmax_agree"]


def band(ours_rows, anchor_rows, floor_runs):
    """tf_verify.py's acceptance: ours' argmax flips against the anchor are
    at most the worst floor run's. Withheld when the runs score different
    position counts, since the counts would not be comparable."""
    positions, ours = flips(ours_rows, anchor_rows)
    floors = [flips(rows, anchor_rows) for rows in floor_runs]
    comparable = positions > 0 and all(p == positions for p, _ in floors)
    worst = max(f for _, f in floors)
    return {"kind": "band", "positions": positions, "flips": ours,
            "floor_flips": [f for _, f in floors], "floor": worst,
            "verdict": "WITHHELD" if not comparable else "PASS" if ours <= worst else "FAIL"}


def compare_runs(ours_rows, theirs_rows):
    """Greedy then logit rows; empty when neither run returned anything."""
    return greedy(ours_rows, theirs_rows) + logit(ours_rows, theirs_rows)


def render(rows):
    """The AGREE / DISAGREE / SUMMARY lines and the FIDELITY line the retired agreement scripts and xval.sh printed."""
    lines = []
    for r in rows:
        kind = r["kind"]
        if kind == "greedy":
            lines.append(f"AGREE {r['prefix']:2d}/{r['of']} tokens ({r['match_pct']:5.1f}%) "
                         f"| {r['request']}")
        elif kind == "greedy_summary":
            lines.append(f"SUMMARY prompts={r['prompts']} mean_prefix={r['mean_prefix']:.1f} "
                         f"median_prefix={r['median_prefix']} "
                         f"agree4plus={r['agree4plus']}/{r['prompts']}")
        elif kind == "logit_disagree":
            lines.append(f"DISAGREE {r['request']} pos={r['pos']} "
                         f"reference_gap={r['reference_gap']:.4f} "
                         f"reference={r['reference']} ours={r['ours']}")
        elif kind == "logit_summary":
            if r["positions"]:
                lines.append(f"SUMMARY positions={r['positions']} "
                             f"argmax_agree={r['argmax_agree']} "
                             f"({100.0 * r['argmax_agree'] / r['positions']:.2f}%) "
                             f"near_tie_flips={r['near_tie_flips']} "
                             f"hard_disagreements={r['hard_disagreements']}")
            if r["delta_max"] is not None:
                lines.append(f"SUMMARY true_token_logprob_delta mean={r['delta_mean']:.5f} "
                             f"p95={r['delta_p95']:.5f} max={r['delta_max']:.5f}")
            lines.append(f"FIDELITY verdict={r['verdict']} "
                         f"hard_disagreements={r['hard_disagreements']}")
        elif kind == "band":
            lines.append(f"BAND verdict={r['verdict']} flips={r['flips']} floor={r['floor']} "
                         f"floor_runs={','.join(map(str, r['floor_flips']))} "
                         f"positions={r['positions']}")
    return lines
