"""Pair two Sides cell by cell and decide, per row, a ratio or a withhold.

The rules are table.py's, moved: pair cells at matched (ctx, bs, tp);
pair against theirs' ms_at_eval when present, else the raw slope flagged
RAW; withhold on a method mismatch (unless allow_method_mismatch), a KV
dtype, quant or serving-topology mismatch, and on COMM rows (bs >=
comm_bound_bs); flag LOW R2 (< 0.999) and EAGER but keep their ratio; warn,
without withholding, when theirs is older than 14 days or the two sides'
NCCL pins differ. One rule is new: when both sides carry a workload_hash
and the hashes differ, every row is withheld (reason "workload").

render(report) is table.py's text, byte for byte, with the agreement rows
(crossval/agreement.py) printed after the cell table.
"""

from dataclasses import dataclass, field

from quettabench.config.schema import serving_style_compatible
from quettabench.crossval import agreement
from quettabench.crossval.reader import Side, age_days

# Age past which the baseline draws a warning: drivers and kernels move, and
# ratios against a stale vLLM capture mislead.
STALE_DAYS = 14
LOW_R2 = 0.999
# Steady-state marginal decode: the engine's pipelined loop and vLLM's slope
# time the same quantity; the criterion per-step groups do not.
MARGINAL = ("loop", "slope")
KV_SHORT = {"float16": "fp16", "bfloat16": "bf16", "half": "fp16"}


@dataclass
class Report:
    ours: Side
    theirs: Side
    age_days: float
    nccl_mismatch: bool
    serving: tuple          # (ours, theirs) serving style
    quant: tuple            # (ours, theirs) quantization
    serving_mismatch: bool
    quant_mismatch: bool
    method_matched: bool    # after allow_method_mismatch
    workload_mismatch: bool
    comm_bound_bs: int | None
    tps_without_theirs: list
    rows: list              # paired cells; ratio None when withheld
    ours_only: list
    theirs_only: list
    agreement: list = field(default_factory=list)

    @property
    def comparable(self):
        """False when every ratio is withheld for the whole run."""
        return self.method_matched and not (self.serving_mismatch or self.quant_mismatch)


def _short(kv):
    return KV_SHORT.get(kv, kv)


def _nccl_mismatch(ours, theirs):
    """theirs pinned an NCCL algo (else proto) that ours pins differently."""
    o, t = ours.header, theirs.header
    if t["nccl_algo"] and o["nccl_algo"] and t["nccl_algo"] != o["nccl_algo"]:
        return True
    return bool(t["nccl_proto"] and o["nccl_proto"] and t["nccl_proto"] != o["nccl_proto"])


def pair_row(a, b, report):
    """One paired cell: ours row a against theirs row b."""
    bs = a["bs"]
    ms_a = a["ms_at_eval"] or a["ms_per_step"]
    # Pair against the KV-position-normalized number when present, so both
    # columns are decode cost at the same resident KV (ctx+eval_pos).
    ms_b = b["ms_at_eval"] or b["ms_per_step"]
    r2 = b["r2"] if b["r2"] is not None else float("nan")
    flags, withheld = [], []
    if r2 < LOW_R2:
        flags.append("LOW R2")
    # Pre-normalization baseline: ms is the raw slope at the fit centroid, not
    # eval_pos. Marked so a mixed table is never read as matched-KV.
    if not b["ms_at_eval"]:
        flags.append("RAW")
    if a["kv"] and _short(a["kv"]) != _short(b["kv"]):
        flags.append(f"KV {_short(a['kv'])}/{_short(b['kv'])}")
        withheld.append("kv")
    if report.quant_mismatch:
        flags.append(f"QUANT {report.quant[1]}/{report.quant[0]}")
        withheld.append("quant")
    if report.comm_bound_bs is not None and bs >= report.comm_bound_bs:
        flags.append("COMM")
        withheld.append("comm")
    # Eager rows include per-step host enqueue; the flag keeps that visible.
    if a["mode"] == "eager":
        flags.append("EAGER")
    if not report.method_matched:
        withheld.append("method")
    if report.serving_mismatch:
        withheld.append("serving")
    if report.workload_mismatch:
        withheld.append("workload")
    tps_a, tps_b = bs * 1000.0 / ms_a, bs * 1000.0 / ms_b
    return {"group": a["group"], "ctx": a["ctx"], "bs": bs, "tp": a["tp"],
            "ours_ms": ms_a, "theirs_ms": ms_b, "ours_tok_s": tps_a, "theirs_tok_s": tps_b,
            "ratio": None if withheld else tps_a / tps_b, "r2": r2,
            "flags": flags, "withheld": withheld}


def pair(ours, theirs, allow_method_mismatch=False, now=0.0):
    """The Report for ours against theirs; now is epoch seconds for the age."""
    o, t = ours.header, theirs.header
    serving = (o["serving_style"] or "aggregated", t["serving_style"] or "aggregated")
    quant = (o["quant"] or t["quant"], t["quant"])
    hashes = (o["workload_hash"], t["workload_hash"])
    a = {(r["ctx"], r["bs"], r["tp"]): r for r in ours.rows}
    b = {(r["ctx"], r["bs"], r["tp"]): r for r in theirs.rows}
    comm = t["comm_bound_bs"] if t["comm_bound_bs"] is not None else o["comm_bound_bs"]
    report = Report(
        ours=ours, theirs=theirs, age_days=age_days(theirs, now),
        nccl_mismatch=_nccl_mismatch(ours, theirs), serving=serving, quant=quant,
        # Serving topology must match on both sides; a disaggregated engine row
        # against an aggregated baseline is a category error, never a ratio.
        serving_mismatch=not serving_style_compatible(serving[1], serving[0]),
        # Quantization must match; an fp4 engine row against a bf16 baseline is
        # not a fair ratio, so it is flagged per row and withheld, like KV.
        quant_mismatch=quant[0] != quant[1],
        method_matched=(o["method"] in MARGINAL and t["method"] in MARGINAL)
        or bool(allow_method_mismatch),
        workload_mismatch=all(hashes) and hashes[0] != hashes[1],
        comm_bound_bs=None if comm is None else int(comm),
        tps_without_theirs=sorted({k[2] for k in a} - {k[2] for k in b}),
        rows=[], ours_only=sorted(set(a) - set(b)), theirs_only=sorted(set(b) - set(a)))
    report.rows = [pair_row(a[k], b[k], report) for k in sorted(set(a) & set(b))]
    return report


def _cells(keys):
    return ", ".join(f"{c}x{b}" + (f"@tp{t}" if t != 1 else "") for c, b, t in keys)


def render_preamble(report):
    """Parser notes, theirs' header line, the warnings, and the tp check."""
    t, o = report.theirs.header, report.ours.header
    lines = list(report.ours.notes) + list(report.theirs.notes)
    nccl = t["nccl_algo"] or t["nccl_proto"]
    lines.append(
        f"vLLM {t['timestamp']} ({report.age_days:.1f}d)  "
        f"vllm {t['version']}  {t['gpu']}  {t['model']}  "
        f"{t['dtype']}  grid {t['grid']}  {t['graph_mode']}"
        + (f"  bench {t['bench_sha'][:9]}" if t["bench_sha"] else "")
        + (f"  engine {t['engine_sha'][:9]}" if t["engine_sha"] else "")
        + (f"  prompts {t['prompt_mode']}" if t["prompt_mode"] else "")
        + (f"  eval_pos {t['eval_pos']}" if t["eval_pos"] else "")
        + (f"  nccl={t['nccl_algo']}/{t['nccl_proto']}" if nccl else ""))
    if report.age_days > STALE_DAYS:
        lines.append(f"WARNING: baseline older than {STALE_DAYS}d; re-run `just run configs/runs/<wl>.yaml`")
    if report.nccl_mismatch:
        lines.append(f"NCCL MISMATCH engine={o['nccl_algo']}/{o['nccl_proto']} "
                     f"vllm={t['nccl_algo']}/{t['nccl_proto']}")
    if report.serving_mismatch:
        lines.append(f"SERVING MISMATCH engine={report.serving[0]} vllm={report.serving[1]}; "
                     "ratios withheld")
    if report.quant_mismatch:
        lines.append(f"QUANT MISMATCH engine={report.quant[0]} vllm={report.quant[1]}; "
                     "ratios withheld")
    if report.workload_mismatch:
        lines.append(f"WORKLOAD MISMATCH engine={o['workload_hash']} "
                     f"vllm={t['workload_hash']}; ratios withheld")
    if not report.comparable:
        lines.append("engine log times one synchronized step per iteration; the baseline "
                     "is a pipelined slope. Not the same quantity, so ratios are withheld "
                     "(XVAL_ALLOW_METHOD_MISMATCH=1 to force).")
    lines.append("")
    if report.tps_without_theirs:
        base_tps = sorted({r["tp"] for r in report.theirs.rows})
        lines.append(f"tp mismatch: engine rows at tp {report.tps_without_theirs} have no "
                     f"baseline (baseline holds tp {base_tps}); capture one per tp")
    return lines


def render_table(report):
    """The paired rows, the COMM footnote, and the unpaired cells."""
    lines = [(f"{'group':<20} {'ctx':>6} {'bs':>4} {'tp':>4} {'ours ms':>10} {'vllm ms':>9} "
              f"{'ours tok/s':>12} {'vllm tok/s':>11} {'ratio':>7} {'r2':>7}")]
    for r in report.rows:
        ratio = f"{'--':>7}" if r["ratio"] is None else f"{r['ratio']:>6.2f}x"
        lines.append(
            f"{r['group']:<20} {r['ctx']:>6} {r['bs']:>4} {r['tp']:>4} {r['ours_ms']:>10.3f} "
            f"{r['theirs_ms']:>9.3f} {r['ours_tok_s']:>12.0f} {r['theirs_tok_s']:>11.0f} "
            f"{ratio} {r['r2']:>7.5f}" + "".join(f"  {f}" for f in r["flags"]))
    if any("comm" in r["withheld"] for r in report.rows):
        lines.append(f"\nCOMM = collective-bound (bs>={report.comm_bound_bs}): absolute ms is "
                     f"dominated by the shared TP allreduce and is node/NCCL-specific; "
                     f"cross-node reference anchoring is not meaningful for these cells.")
    if report.ours_only:
        lines.append(f"\nours-only: {_cells(report.ours_only)}")
    if report.theirs_only:
        lines.append(f"vLLM-only: {_cells(report.theirs_only)}")
    return lines


def render(report):
    """table.py's stdout for report, then the agreement rows; the table is
    left out when no cell pairs."""
    lines = render_preamble(report)
    if report.rows:
        lines += render_table(report)
    if report.agreement:
        lines += [""] + agreement.render(report.agreement)
    return "".join(line + "\n" for line in lines)
