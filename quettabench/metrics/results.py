"""The one results writer.

RunWriter owns results/<run>/: requests.jsonl gets one line per request the
moment it finishes (append-only, flushed, so a crash keeps every row
already written) and run.json is written once at the end:

  {schema_version, provenance, config, origins, summary, steps, run_status}

run_status is {"status": ok | partial | aborted | failed, "reason": ...}.
Provenance carries the launch env with every key matching
TOKEN|KEY|SECRET|HF_ written as REDACTED.

write_legacy_results is the old --output layout (benchmark_schema_version 5,
config, summary, per_request), kept until commit 7 removes the old CLI.
"""

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# v5: greedy-by-default + seed; failed requests keep partial timings and error_kind.
BENCHMARK_SCHEMA_VERSION = 5
RUN_SCHEMA_VERSION = 1
RUN_STATUSES = ("ok", "partial", "aborted", "failed")
SECRET_NAME = re.compile(r"TOKEN|KEY|SECRET|HF_")
REPO = Path(__file__).resolve().parents[2]


def redact(env):
    """env with the value of every secret-looking key replaced by REDACTED."""
    return {k: "REDACTED" if SECRET_NAME.search(k) else v for k, v in (env or {}).items()}


def bench_sha(repo=REPO):
    """This checkout's commit; empty when there is no git or no repo (an rsync
    copy), never a crash. Same rule as crossval/cache.py."""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                              capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return ""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _ms(seconds, digits=2):
    return round(seconds * 1000, digits)


# RequestResult field -> per_request key, written only when not None.
# Seconds become milliseconds; rates keep four digits.
OPTIONAL_MS = ("scheduled_at", "dispatch_started_at", "semaphore_acquired_at", "completed_at",
               "client_schedule_delay", "client_queue_wait", "client_request_wall")
OPTIONAL_PLAIN = ("request_index", "max_tokens_requested", "message_count", "prompt_chars")
OPTIONAL_CACHE = ("session_id", "turn_index", "previous_context_tokens", "total_context_tokens",
                  "new_prefill_tokens", "cached_context_tokens", "cache_hit_rate",
                  "cache_estimate_source", "cache_block_size",
                  "block_aligned_cached_context_tokens", "block_aligned_new_prefill_tokens",
                  "block_aligned_cache_hit_rate", "uncached_prefix_tail_tokens",
                  "total_context_blocks", "cached_context_blocks", "new_prefill_blocks")
RATES = ("cache_hit_rate", "block_aligned_cache_hit_rate")
# Fields the target schema adds; requests.jsonl only, never the old layout.
NEW_FIELDS = ("ttft_basis", "ttft_rule", "accepted_tokens", "cell")


def request_row(r):
    """One per_request entry of the old layout, keys in the old order."""
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
    for name in OPTIONAL_CACHE:
        value = getattr(r, name)
        if value is not None:
            row[name] = round(value, 4) if name in RATES else value
    if r.request_metadata:
        row["request_metadata"] = r.request_metadata
    return row


def jsonl_row(r, **extra):
    """request_row plus the target-schema fields that are set, plus extra."""
    row = request_row(r)
    for name in NEW_FIELDS:
        if getattr(r, name, None) is not None:
            row[name] = getattr(r, name)
    row.update(extra)
    return row


def provenance(cfg, handle=None, driver=None, launch_env=None):
    """Where a run's numbers came from: hardware, engine, code and launch env."""
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
        "kv_dtype": engine.kv_dtype,
        "launch_env": redact(launch_env),
    }


class RunWriter:
    """results/<run>/requests.jsonl as rows finish, run.json at finish().

    results_dir None writes nothing: the run still goes through one writer
    whether or not the caller asked for files.
    """

    def __init__(self, results_dir, cfg, *, handle=None, driver=None, launch_env=None):
        self.dir = Path(results_dir) if results_dir else None
        self.cfg = cfg
        self.provenance = provenance(cfg, handle, driver, launch_env)
        self.provenance["started"] = _now()
        self.rows = 0
        self._fh = None
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._fh = (self.dir / "requests.jsonl").open("a")

    def append(self, result, **extra):
        """Write one request's row now and flush it to disk."""
        self.rows += 1
        if self._fh is not None:
            self._fh.write(json.dumps(jsonl_row(result, **extra)) + "\n")
            self._fh.flush()

    def finish(self, summary, steps=None, status="ok", reason=None):
        """Write run.json; returns its content. summary is a plain dict."""
        if status not in RUN_STATUSES:
            raise ValueError(f"run_status {status!r}; use one of {RUN_STATUSES}")
        self.provenance["finished"] = _now()
        cfg = self.cfg.to_dict()
        origins = cfg.pop("origins")
        run = {
            "schema_version": RUN_SCHEMA_VERSION,
            "provenance": self.provenance,
            "config": cfg,
            "origins": origins,
            "summary": summary,
            "steps": steps,
            "run_status": {"status": status, "reason": reason},
        }
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            (self.dir / "run.json").write_text(json.dumps(run, indent=2, default=str) + "\n")
        return run


def write_legacy_results(summary, results, output_path: str, config: dict):
    """The old --output file: {config, summary, per_request}."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # Stamp the schema version so result files are self-describing (v4 =
    # fixed streaming client + coverage-guarded tpot). setdefault keeps any
    # explicitly provided value.
    config = dict(config)
    config.setdefault("benchmark_schema_version", BENCHMARK_SCHEMA_VERSION)
    output = {
        "config": config,
        "summary": summary.to_dict(),
        "per_request": [request_row(r) for r in results if r is not None],
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {output_path}")


def per_turn_output_path(output: str) -> str:
    """Per-turn breakdown path for a results file; always differs from ``output``."""
    p = Path(output)
    stem = p.stem if p.suffix else p.name
    return str(p.with_name(stem + "_per_turn.json"))


def write_legacy_per_turn(turn_summaries, output: str, config: dict):
    """The old <output>_per_turn.json: {config, per_turn}."""
    turn_output = per_turn_output_path(output)
    Path(turn_output).parent.mkdir(parents=True, exist_ok=True)
    with open(turn_output, "w") as f:
        json.dump({
            "config": config,
            "per_turn": [ts.to_dict() for ts in turn_summaries],
        }, f, indent=2)
    print(f"Per-turn results saved to: {turn_output}")
