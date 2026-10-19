"""RunWriter owns results/<run>/: rows as they finish, run.json at the end."""

import dataclasses
import json
import os
import re
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from quettabench.config import env_map

RUN_SCHEMA_VERSION = 1
RUN_STATUSES = ("ok", "partial", "aborted", "failed")
RUN_MARKERS = ("run.json", "requests.jsonl", "engine.json")
# Secret name, after splitting camelCase into words: one of these words anywhere ...
SECRET_WORD = (
    r"secrets?|password|passwd|pass|passphrase|credentials?|bearer|auth|oauth"
    r"|authorization|cookie"
)
# ... or one of these as its last word (HF_TOKEN yes, num_key_value_heads no) ...
SECRET_LAST = r"token|apikey|pat"
# ... or key(s) as the last word after one of these (OPENAI_API_KEY yes, cache_keys no).
KEY_OWNER = r"api|secret|private|access|auth|ssh|encryption|signing"
# AWS_ACCESS_KEY_ID names the key pair.
KEY_ID = r"access[_.-]key[_.-]id"
SECRET_NAME = re.compile(
    rf"(?:^|[_.-])(?:{SECRET_WORD})(?:$|[_.-])|(?:^|[_.-])(?:{SECRET_LAST})$"
    rf"|(?:^|[_.-])(?:{KEY_OWNER})[_.-]keys?$|(?:^|[_.-]){KEY_ID}$",
    re.IGNORECASE,
)
# ... except a tokenizer's special tokens, which name a vocabulary entry.
SPECIAL_TOKEN = re.compile(r"(?:eos|bos|pad|unk|sep|cls|mask)[_.-]token", re.IGNORECASE)
CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
# One `Name: value` HTTP header as an argv item (curl -H).
HEADER = re.compile(r"\s*(?P<name>[A-Za-z0-9_.-]+)\s*:\s*\S.*", re.DOTALL)
# A flag, or a NAME=value whose NAME is secret: a command line.
FLAG = re.compile(r"(^|\s)-")
ASSIGN = re.compile(r"(?:^|[\s,])([A-Za-z_][A-Za-z0-9_.-]*)=")
# One NAME=value of a comma-separated list (A=1,HF_TOKEN=X).
LISTED = re.compile(r"(^|,)([A-Za-z_][A-Za-z0-9_.-]*)=([^,]*)")
# NAME=value as a shell word, value quoted or not (HF_TOKEN='X Y').
SHELL_ASSIGN = re.compile(r"""(^|[\s;&|(])([A-Za-z_][A-Za-z0-9_.-]*)=((?:'[^']*'|"(?:[^"\\]|\\.)*"|[^\s'";&|()<>])+)""")
# The password in scheme://user:password@host.
URL_PASSWORD = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*://[^\s/:@]*:)[^\s/@]+@")
# Secret words that hide the next word in a command line too (Bearer X).
COMMAND_SECRETS = ("bearer", "authorization")
# Shell tokens _redact_command keeps as written.
PUNCTUATION = re.compile(r"[();<>|&]+")
REPO = Path(__file__).resolve().parents[2]


def secret(name):
    words = CAMEL.sub("_", str(name).lstrip("-"))
    return not SPECIAL_TOKEN.fullmatch(words) and bool(SECRET_NAME.search(words))


def _secret_header(text):
    header = HEADER.fullmatch(text)
    return f"{header['name']}: REDACTED" if header and secret(header["name"]) else None


def redact_argv(argv, command=False):
    """Covers flag, NAME=X, header, comma-list and URL forms."""
    # Only a flag, bearer or authorization hides the next item, so a bare "pass" hides nothing.
    out, hide = [], False
    for arg in argv:
        if isinstance(arg, (dict, list, tuple)):
            out.append(redact(arg))
            hide = False
            continue
        text = str(arg)
        if URL_PASSWORD.search(text):
            arg = text = URL_PASSWORD.sub(r"\1REDACTED@", text)
        if hide and not text.startswith("-"):
            out.append("REDACTED")
            hide = False
            continue
        hide = False
        header = _secret_header(text)
        if header:
            out.append(header)
            continue
        if text[:1] in ("{", "["):
            try:
                doc = json.loads(text)
            except ValueError:
                doc = None
            if isinstance(doc, (dict, list)):
                hidden = redact(doc)
                out.append(arg if hidden == doc else json.dumps(hidden, ensure_ascii=False))
                continue
        name, eq, value = text.partition("=")
        if eq and _secret_header(value):
            arg = f"{name}={_secret_header(value)}"
        elif any(c.isspace() for c in text) and not (text.startswith("-") and eq and secret(name)):
            arg = _redact_command(text)
        elif secret(name):
            arg = f"{name}=REDACTED" if eq else arg
            hide = not eq and (not command or text.startswith("-") or text.lower() in COMMAND_SECRETS)
        elif eq and "," in value:
            arg = LISTED.sub(lambda m: f"{m[1]}{m[2]}=REDACTED" if secret(m[2]) else m[0], text)
        out.append(arg)
    return out


def _redact_command(text):
    """A secret NAME=value word loses its whole value, quoted parts included."""
    words = SHELL_ASSIGN.sub(lambda m: f"{m[1]}{m[2]}=REDACTED" if secret(m[2]) else m[0], text)
    lexer = shlex.shlex(words, posix=True, punctuation_chars=True)
    lexer.whitespace_split, lexer.commenters = True, ""
    try:
        parts = list(lexer)
    except ValueError:
        parts = words.split()
    hidden = redact_argv(parts, command=True)
    if hidden == parts and words == text:
        return text
    return " ".join(p if PUNCTUATION.fullmatch(p) else shlex.quote(p) for p in hidden)


def redact(obj):
    """The one rule for everything under results/."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if secret(k):
                out[k] = v if isinstance(v, (dict, list, tuple)) and not v else "REDACTED"
            elif isinstance(v, str) and (v[:1] in ("{", "[") or _secret_header(v)):
                out[k] = redact_argv([v])[0]
            elif isinstance(v, str) and (FLAG.search(v) or any(map(secret, ASSIGN.findall(v)))):
                out[k] = _redact_command(v)
            else:
                out[k] = redact(v)
        return out
    if isinstance(obj, (list, tuple)):
        return redact_argv(obj)
    if isinstance(obj, str):
        return URL_PASSWORD.sub(r"\1REDACTED@", obj)
    return obj


def exit_status(exc):
    """aborted on Ctrl-C, ok on a clean SystemExit, failed otherwise."""
    if isinstance(exc, SystemExit) and exc.code in (None, 0):
        return "ok", None
    status = "aborted" if isinstance(exc, KeyboardInterrupt) else "failed"
    return status, f"{type(exc).__name__}: {exc}"


def write_atomic(path, text):
    """Via <path>.tmp and os.replace, so a poller sees the old file or the whole new one."""
    tmp = Path(f"{path}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def stamp_pulled(path):
    doc = json.loads(Path(path).read_text())
    if not isinstance(doc, dict) or not isinstance(doc.setdefault("provenance", {}), dict):
        raise TypeError("run.json or its provenance is not an object")
    doc["provenance"]["pulled"] = _now()
    write_atomic(path, json.dumps(doc, indent=2) + "\n")


def bench_sha(repo=REPO):
    """Else QBENCH_SHA (set in the image), else empty; never raises."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:
        sha = ""
    return sha or env_map.read_env(only=("QBENCH_SHA",)).get("detect.bench_sha", "")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _ms(seconds, digits=2):
    return round(seconds * 1000, digits)


# Written only when not None; seconds become milliseconds, rates keep four digits.
OPTIONAL_MS = (
    "sent_at",
    "scheduled_at",
    "dispatch_started_at",
    "semaphore_acquired_at",
    "completed_at",
    "client_schedule_delay",
    "client_queue_wait",
    "client_request_wall",
    "client_stall",
    "client_stall_ttft",
)
OPTIONAL_PLAIN = ("request_index", "max_tokens_requested", "message_count", "prompt_chars")
OPTIONAL_CACHE = (
    "session_id",
    "turn_index",
    "previous_context_tokens",
    "total_context_tokens",
    "new_prefill_tokens",
    "cached_context_tokens",
    "cache_hit_rate",
    "cache_estimate_source",
    "cache_block_size",
    "block_aligned_cached_context_tokens",
    "block_aligned_new_prefill_tokens",
    "block_aligned_cache_hit_rate",
    "uncached_prefix_tail_tokens",
    "total_context_blocks",
    "cached_context_blocks",
    "new_prefill_blocks",
)
RATES = ("cache_hit_rate", "block_aligned_cache_hit_rate")
NEW_FIELDS = (
    "ttft_basis",
    "ttft_rule",
    "accepted_tokens",
    "server_cached_tokens",
    "cell",
    "batch_wall_s",
    "logits",
    "stream_closed_s",
    "chunk_chars",
    "chunk_tokens",
    "planned_digest",
    "sent_digest",
)


def jsonl_row(r, **extra):
    row = {
        "success": r.success,
        "ttft_ms": _ms(r.ttft) if r.ttft else None,
        "tpot_ms": _ms(r.tpot) if r.tpot else None,
        "itl_ms": [_ms(t) for t in r.itl] if r.itl else [],
        "e2el_ms": _ms(r.e2el) if r.e2el else None,
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "error": r.error,
    }
    if r.error_kind is not None:
        row["error_kind"] = r.error_kind
    row["usage_reported"] = r.usage_reported
    if r.excluded_from_summary:
        row["excluded_from_summary"] = True
    for name in OPTIONAL_PLAIN:
        if getattr(r, name) is not None:
            row[name] = getattr(r, name)
    for name in OPTIONAL_MS:
        if getattr(r, name + "_s") is not None:
            row[name + "_ms"] = _ms(getattr(r, name + "_s"))
    # The close on the run clock; stream_closed_s counts from the request's own clock start, sent_at.
    if r.sent_at_s is not None and r.stream_closed_s is not None:
        row["stream_closed_ms"] = _ms(r.sent_at_s + r.stream_closed_s)
    for name in OPTIONAL_CACHE:
        value = getattr(r, name)
        if value is not None:
            row[name] = round(value, 4) if name in RATES else value
    if r.request_metadata:
        row["request_metadata"] = r.request_metadata
    for name in NEW_FIELDS:
        if getattr(r, name, None) is not None:
            row[name] = getattr(r, name)
    # The per-chunk series are arrays in memory; JSON gets lists, with null for a chunk without a count (a
    # negative entry, summary.NO_COUNT; results.py is shipped to probe hosts without summary.py).
    if "chunk_chars" in row:
        row["chunk_chars"] = list(row["chunk_chars"])
    if "chunk_tokens" in row:
        row["chunk_tokens"] = [None if t < 0 else t for t in row["chunk_tokens"]]
    row.update(extra)
    return row


def measured_row(row, **extra):
    """Success and error first, like a RequestResult row."""
    return {"success": True, "error": None, **row, **extra}


def provenance(cfg, handle=None, driver=None, launch_env=None):
    engine, hardware = cfg.engine, cfg.hardware
    return {
        "gpu": getattr(handle, "gpu", "") or "",
        "engine": engine.name,
        "engine_version": getattr(driver, "version", None),
        "bench_sha": bench_sha(),
        "engine_sha": cfg.provenance.get("engine_sha"),
        "nccl_algo": hardware.collective.get("NCCL_ALGO"),
        "nccl_proto": hardware.collective.get("NCCL_PROTO"),
        "link_profile": hardware.link_profile,
        "serving_style": engine.serving_style,
        "quant": engine.quant,
        "kv_cache_dtype": engine.kv_cache_dtype,
        "launch_env": redact(launch_env or {}),
    }


class RunWriter:
    """results_dir None writes nothing."""

    def __init__(self, results_dir, cfg, *, handle=None, driver=None, launch_env=None, exclusive=False):
        self.dir = Path(results_dir) if results_dir else None
        self.cfg = cfg
        self.provenance = provenance(cfg, handle, driver, launch_env)
        self.provenance["started"] = _now()
        self.rows = 0
        self.finished = False
        self._fh = None
        self._probe_fh = None
        self._row_errors = []
        self._fill: tuple | None = None
        if self.dir is not None:
            self.dir.parent.mkdir(parents=True, exist_ok=True)
            self.dir.mkdir(exist_ok=not exclusive)
            self._fh = (self.dir / "requests.jsonl").open("a")
            # A flush can block for 100 ms on a network disk; on the event loop that stalls every stream.
            self._row_pool = ThreadPoolExecutor(max_workers=1)

    def _write_row(self, row):
        assert self._fh is not None
        try:
            self._fh.write(json.dumps(row) + "\n")
            self._fh.flush()
        except OSError as e:
            self._row_errors.append(e)

    def _close_rows(self):
        assert self._fh is not None
        self._row_pool.shutdown(wait=True)
        self._fh.close()
        self._fh = None
        if self._fill is not None and self.dir is not None:
            field, by_index = self._fill
            path = self.dir / "requests.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            for row in rows:
                if row.get("request_index") in by_index:
                    row[field] = by_index[row["request_index"]]
            write_atomic(path, "".join(json.dumps(row) + "\n" for row in rows))

    def fill_rows(self, field, by_index):
        """Sets field on every row by its request_index when the rows close, for a value known only after the
        run; the rows are streamed as requests complete."""
        self._fill = (field, by_index)

    def discard(self):
        """For a run this process does not record."""
        self._close_probe_rows()
        if self._fh is not None and self.dir is not None:
            self._close_rows()
            if not self.rows:
                (self.dir / "requests.jsonl").unlink(missing_ok=True)
                if not any(self.dir.iterdir()):
                    self.dir.rmdir()

    def bind(self, handle=None, driver=None):
        self.provenance["gpu"] = getattr(handle, "gpu", "") or ""
        self.provenance["engine_version"] = getattr(driver, "version", None)

    def fail(self, exc):
        """Unless finish() already wrote one."""
        if not self.finished:
            status, reason = exit_status(exc)
            self.finish(None, status=status, reason=reason)

    def append(self, result, **extra):
        if self._row_errors:
            raise self._row_errors[0]
        self.rows += 1
        row = measured_row(result, **extra) if isinstance(result, dict) else jsonl_row(result, **extra)
        if self._fh is not None:
            self._row_pool.submit(self._write_row, row)

    def probe_row(self, row):
        """Flushed, so a crash keeps every row measured."""
        line = json.dumps(dataclasses.asdict(row)) + "\n"
        if self.dir is not None:
            if self._probe_fh is None:
                self._probe_fh = (self.dir / "rows.jsonl").open("a")
            self._probe_fh.write(line)
            self._probe_fh.flush()

    def _close_probe_rows(self):
        if self._probe_fh is not None:
            self._probe_fh.close()
            self._probe_fh = None

    def finish(self, summary, steps=None, status="ok", reason=None):
        if status not in RUN_STATUSES:
            raise ValueError(f"run_status {status!r}; use one of {RUN_STATUSES}")
        self.provenance["finished"] = _now()
        self.finished = True
        cfg = self.cfg.to_dict()
        origins = cfg.pop("origins")
        run = {
            "schema_version": RUN_SCHEMA_VERSION,
            "provenance": self.provenance,
            "config": redact(cfg),
            "origins": origins,
            "summary": summary,
            "steps": steps,
            "run_status": {"status": status, "reason": reason},
        }
        self._close_probe_rows()
        if self._fh is not None and self.dir is not None:
            self._close_rows()
            if self._row_errors and status == "ok":
                run["run_status"] = {
                    "status": "failed",
                    "reason": f"requests.jsonl write failed: {self._row_errors[0]}",
                }
            write_atomic(self.dir / "run.json", json.dumps(run, indent=2, default=str) + "\n")
        return run


# Format pinned by tests/golden.
PCTS = (50, 90, 95, 99)


def workload_line(w):
    line = (
        f"WORKLOAD_CHECK plan={w['plan_hash']} expected={w['expected']} executed={w['executed_hash']} "
        f"match={int(w['match'])} n={w['n']}/{w['of']}"
    )
    if w.get("turns_not_sent"):
        line += f" not_sent={len(w['turns_not_sent'])}"
    if m := w.get("first_mismatch"):
        where = ":".join(str(v) for k, v in m.items() if k != "reason")
        line += f" first_mismatch={where} ({m['reason']})"
    return line


def goodput_probe_line(p):
    return (
        f"GOODPUT_PROBE speed={p['speed']:g} verdict={p['verdict']} "
        f"bound={','.join(p['bound']) or 'none'} req_s={p['req_s']:.3f}"
    )


def summary_tag_lines(summary):
    lines = []
    w = summary.get("workload")
    if w is not None:
        lines.append(workload_line(w))
        if not w["match"]:
            void = "sweep" if "goodput" in summary else "comparison"
            lines.append(f"WORKLOAD_MISMATCH replayed workload differs from the trace; {void} is void")
    s = summary.get("serve")
    if s is not None:
        tp = s["tpot_ms"]
        lines.append(
            f"SERVE_SUMMARY n={s['n']} "
            + (
                " ".join(f"ttft_ms_p{q}={s['ttft_ms'][f'p{q}']:.1f}" for q in PCTS)
                if s["ttft_ms"]
                else "ttft_ms=withheld"
            )
            + " "
            + (
                " ".join(f"steady_ttft_ms_p{q}={st[f'p{q}']:.1f}" for q in PCTS) + " "
                if (st := s.get("steady_ttft_ms"))
                else ""
            )
            + (
                " ".join(f"tpot_ms_p{q}={tp[f'p{q}']:.3f}" for q in PCTS) + " "
                if tp
                else ("tpot_ms=withheld " if not s["ttft_ms"] else "tpot_ms_p50=- ")
            )
            + f"req_s={s['req_s']:.3f} tok_s={s['tok_s']:.1f}"
        )
    sla = summary.get("sla")
    if sla is not None:
        tpot = "-" if sla["tpot_p99"] is None else f"{sla['tpot_p99']:.3f}"
        ttft = "-" if sla["ttft_p99"] is None else f"{sla['ttft_p99']:.1f}"
        lines.append(
            f"SLA_SUMMARY verdict={sla['verdict']} bound={','.join(sla['bound']) or 'none'} "
            f"ttft_p99={ttft}/{sla['sla_ttft_ms'] or '-'} "
            f"tpot_p99={tpot}/{sla['sla_tpot_ms'] or '-'} "
            f"rate_req_s={sla['rate_req_s']:.3f}"
        )
    b = summary.get("burst")
    if b is not None and b["peak_ttft_p99"] is not None:
        rec = "na" if b["recovery_s"] is None else f"{b['recovery_s']:.1f}"
        lines.append(
            f"BURST_SUMMARY peak_ttft_p99={b['peak_ttft_p99']:.1f} recovery_s={rec} target_ms={b['target_ms']}"
        )
    spec = summary.get("spec_decode")
    if spec is not None and spec.get("source") == "server_metrics":
        lines.append(
            f"SPEC_DECODE_SUMMARY spec=measured source=server_metrics "
            f"accept_len={spec['accept_len']:.2f} accept_rate={spec['accept_rate']:.3f} "
            f"drafts={spec['drafts']} draft_tokens={spec['draft_tokens']} "
            f"accepted_tokens={spec['accepted_tokens']}"
        )
    elif spec is not None:
        lines.append(
            f"SPEC_DECODE_SUMMARY spec=measured accept_len={spec['accept_len']:.2f} "
            f"accept_rate={spec['accept_rate']:.3f} n={spec['n']}"
            if spec["spec"] == "measured"
            else "SPEC_DECODE_SUMMARY spec=off note=needs-real-text-trace"
        )
    g = summary.get("goodput")
    if g is not None and (w is None or w["match"]):
        bound = ",".join(g["fail_bound"])
        lines.append(
            f"GOODPUT_SUMMARY max_req_s={g['max_req_s']:.3f} speed={g['speed']:g} "
            f"bound={bound or 'none-within-range'} iterations={g['iterations']}"
            if g["max_req_s"] is not None
            else f"GOODPUT_SUMMARY max_req_s=none bound={bound or 'na'} "
            f"iterations={g['iterations']} note=no-passing-speed-within-budget"
        )
    return lines


def print_summary_tag_lines(summary):
    for line in summary_tag_lines(summary):
        print(line, flush=True)


# Format pinned by tests/golden.
def grid_event_line(run, event):
    ctx, bs = event["ctx"], event["bs"]
    if event["event"] == "skip_maxlen":
        return f"SKIP ctx={ctx} bs={bs} reason=maxlen"
    if event["event"] == "skip_kv":
        return f"SKIP ctx={ctx} bs={bs} need_kv={event['need']} cap={event['cap']}"
    if event["event"] == "cycled":
        return f"NOTE ctx={ctx} bs={bs} trace cycled (fewer reqs than bs)"
    s = event["step"]
    return (
        f"RESULT mode={run['meta']['mode']} tp={s['tp']} kv={run['kv']} ctx={ctx} bs={bs} "
        f"ms_per_step={s['ms_per_step']:.3f} ms_at_eval={s['ms_at_eval']:.3f} "
        f"eval_pos={s['eval_pos']} mean_steps={s['mean_steps']:.0f} r2={s['r2']:.5f} "
        f"prefill_s={s['prefill_s']:.2f} kv_gib={s['kv_gib']:.1f} "
        f"attn_share={s['attn_share']:.3f} raw={[round(y, 2) for y in s['raw']]}"
    )


def grid_tag_lines(run):
    """META through DONE."""
    lines = []
    t = run.get("trace")
    if t is not None:
        lines.append(f"META trace={t['path']} reqs={t['reqs']} workload_hash={t['workload_hash']}")
    lines += [f"META {k}={v}" for k, v in run["meta"].items()]
    c = run.get("capacity")
    if c is not None:
        lines.append(
            f"CAPACITY mode={c['mode']} tp={c['tp']} dtype={c['dtype']} "
            f"kv_bytes_per_token={c['kv_bytes_per_token']} free_gib={'-' if c['free_gib'] is None else format(c['free_gib'], '.0f')} "
            f"kv_tokens={c['kv_tokens']}"
        )
    if run.get("warm_ok"):
        lines.append("WARM_OK")
    lines += [grid_event_line(run, e) for e in run.get("events", ())]
    if run.get("done"):
        lines.append("DONE")
    return lines


def print_grid_tag_lines(run, shown=0):
    """Returns the new shown count."""
    lines = grid_tag_lines(run)
    for line in lines[shown:]:
        print(line, flush=True)
    return len(lines)
