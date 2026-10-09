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
"""

import statistics

NEAR_TIE_GAP = 0.1   # reference top-2 gap below which an argmax flip is a near tie
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
            elif ref["top2_gap"][pos] < NEAR_TIE_GAP:
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


def compare_runs(ours_rows, theirs_rows):
    """Greedy then logit rows; empty when neither run returned anything."""
    return greedy(ours_rows, theirs_rows) + logit(ours_rows, theirs_rows)


def render(rows):
    """The old scripts' AGREE / DISAGREE / SUMMARY lines and xval.sh's FIDELITY line."""
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
    return lines
