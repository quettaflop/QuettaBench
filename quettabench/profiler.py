"""Attach-only profiler: read results/<run>/, run the asked-for sources, and
merge their sections into results/<run>/profile.json.

It never launches an engine and never drives the measured load. It finds
the server through engine.json {endpoint, pid, launch_spec, instrumentation}
and the run through run.json {config, summary}; run.json is only read.

  metrics      native: one scrape of <endpoint>/metrics + prefix_cache_hit_rate
  energy       wrapped: board power from attach until run.json appears (the
               run finished), or for energy_s seconds; joules per output token
               only for the run window
  cost         wrapped: USD at gpu_cost_hr over run.json's wall time, tp * pp GPUs
  gsm8k        wrapped: greedy GSM8k exact match through the endpoint (its own
               sidecar requests, sent after energy stops, as trace_serve did)
  frontend, pd_host
               native: the live-server probes against the endpoint
  kv_transfer  native: the NIXL probe's CSV, written by its own torchrun run
  kernels      native: per-kernel GPU time per decode step from the armed nsys
               trace, optionally inside an NVTX window less another
  traces       always: the nsys / torch files instrumentation armed at launch

A source that fails is recorded under `errors` and the others still run.
A later profile of the same run replaces sections of the same name.
"""

import argparse
import json
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from quettabench.profiling import native, wrapped

SOURCES = ("metrics", "energy", "cost", "gsm8k", "frontend", "pd_host", "kv_transfer", "kernels")
# energy first (it waits for the run to end), sidecar requests next, file reads last
ORDER = ("energy", "metrics", "gsm8k", "frontend", "pd_host", "cost", "kv_transfer", "kernels")
SECTION = {"gsm8k": "accuracy"}


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def run_totals(run):
    """(output tokens, wall seconds) of a run.json summary."""
    s = run.get("summary") or {}
    return (s.get("out_tokens", s.get("total_output_tokens", 0)),
            s.get("wall_s", s.get("duration_s", 0.0)))


def check_args(sources, args):
    unknown = sorted(set(sources) - set(SOURCES))
    if unknown:
        raise ValueError(f"unknown source(s) {unknown}; known: {', '.join(SOURCES)}")
    needs = {"cost": ("gpu_cost_hr",), "gsm8k": ("gsm8k_file", "gsm8k_n"),
             "kv_transfer": ("kv_transfer_csv",), "kernels": ("kernel_steps",)}
    for source in sources:
        missing = [k for k in needs.get(source, ()) if args.get(k) is None]
        if missing:
            raise ValueError(f"source {source} needs {', '.join(missing)}")


def measure_energy(results_dir, engine, config, nvidia_smi="nvidia-smi", interval_s=1.0,
                   energy_s=None, poll_s=0.5, timeout_s=86400):
    """Sample until run.json appears, or for energy_s seconds when given.
    Returns (samples, window): window "run" or "fixed"."""
    run_json = Path(results_dir) / "run.json"
    if energy_s is None and run_json.exists():
        raise ValueError("run.json already exists: the run is over; pass energy_s "
                         "for a fixed window")
    sampler = wrapped.PowerSampler(wrapped.energy_devices(engine, config),
                                   nvidia_smi=nvidia_smi, interval_s=interval_s).start()
    deadline = time.monotonic() + (energy_s if energy_s is not None else timeout_s)
    while time.monotonic() < deadline and (energy_s is not None or not run_json.exists()):
        time.sleep(poll_s)
    return sampler.stop(), "fixed" if energy_s is not None else "run"


def _endpoint(engine):
    if not engine or not engine.get("endpoint"):
        raise ValueError("no engine.json endpoint to attach to")
    return engine["endpoint"]


def run_source(source, results_dir, engine, args, energy):
    """One source's section. Reads run.json fresh: energy may have waited for it."""
    run = read_json(Path(results_dir) / "run.json") or {}
    config = run.get("config") or {}
    model = args.get("served_model") or (config.get("engine") or {}).get("model")
    if source == "energy":
        samples, window = energy
        devices = wrapped.energy_devices(engine, config)
        section = wrapped.energy_section(samples, args.get("interval_s", 1.0),
                                         run_totals(run)[0], devices)
        if window == "fixed":
            section["joules_per_token"] = None  # the window is not the run's
        return {**section, "window": window}
    if source == "metrics":
        return native.metrics_section(native.scrape_metrics(_endpoint(engine)))
    if source == "cost":
        if not run:
            raise ValueError("cost needs run.json")
        out_tokens, wall_s = run_totals(run)
        e = config.get("engine") or {}
        return wrapped.cost_section(args["gpu_cost_hr"], e.get("tp", 1) * e.get("pp", 1),
                                    wall_s, out_tokens)
    if source == "gsm8k":
        items = wrapped.load_gsm8k(args["gsm8k_file"], args.get("tokenizer") or model,
                                   args["gsm8k_n"], tokenizer=args.get("gsm8k_tokenizer"))
        generate = args.get("gsm8k_generate") or wrapped.http_generate(_endpoint(engine), model)
        return wrapped.gsm8k_accuracy(generate, items)
    if source == "frontend":
        return native.serving_frontend(_endpoint(engine), model, args.get("frontend_argv"))
    if source == "pd_host":
        return native.pd_host(_endpoint(engine), model, args.get("pd_host_argv"))
    if source == "kv_transfer":
        return native.kv_transfer(args["kv_transfer_csv"])
    if source == "kernels":
        return native.kernel_summary(results_dir, args["kernel_steps"], args.get("kernel_window"),
                                     args.get("kernel_subtract"))
    raise ValueError(f"unknown source {source}")


def profile(results_dir, sources, **source_args):
    """Run sources against results_dir's run; merge into profile.json and return it.

    source_args: nvidia_smi, interval_s, energy_s (energy); gpu_cost_hr (cost);
    gsm8k_file, gsm8k_n, tokenizer, gsm8k_tokenizer, gsm8k_generate (gsm8k);
    served_model (overrides config engine.model for requests); frontend_argv,
    pd_host_argv (the probes' own flags); kv_transfer_csv; kernel_steps,
    kernel_window, kernel_subtract (kernels).
    """
    check_args(sources, source_args)
    results_dir = Path(results_dir)
    engine = read_json(results_dir / "engine.json") or {}
    out = read_json(results_dir / "profile.json") or {}
    errors = {}
    energy = None
    for source in [s for s in ORDER if s in sources]:
        try:
            if source == "energy":
                run = read_json(results_dir / "run.json") or {}
                energy = measure_energy(
                    results_dir, engine, run.get("config"),
                    **{k: source_args[k] for k in ("nvidia_smi", "interval_s", "energy_s")
                       if k in source_args})
            out[SECTION.get(source, source)] = run_source(source, results_dir, engine,
                                                          source_args, energy)
        except Exception as e:  # noqa: BLE001 -- one broken source must not lose the others
            errors[source] = f"{type(e).__name__}: {e}"
    out["traces"] = native.collect_armed_traces(results_dir, engine.get("instrumentation", "none"))
    out["sources_run"] = sorted(set(out.get("sources_run", [])) | (set(sources) - set(errors)))
    out["errors"] = errors
    out["profiled_at"] = datetime.now(timezone.utc).isoformat()
    (results_dir / "profile.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return out


def legacy_profile_lines(profile):
    """ENERGYSUM, ACCSUM and COSTSUM in trace_serve's formats and order, for
    the sections profile has (ENERGYSUM only for a run window)."""
    lines = []
    e = profile.get("energy")
    if e is not None and e.get("joules_per_token") is not None:
        lines.append(f"ENERGYSUM joules={e['joules']:.0f} "
                     f"joules_per_token={e['joules_per_token']:.3f} scope=gpu_board "
                     f"devices={e['devices']} samples={e['samples']}")
    a = profile.get("accuracy")
    if a is not None:
        lines.append(f"ACCSUM dataset=gsm8k exact_match={a['exact_match']:.3f} n={a['n']}")
    c = profile.get("cost")
    if c is not None:
        lines.append(f"COSTSUM usd={c['usd']:.4f} usd_per_m_out_tokens={c['usd_per_m_out_tokens']:.2f} "
                     f"gpu_cost_hr={c['gpu_cost_hr']} devices={c['devices']} wall_s={c['wall_s']:.1f}")
    return lines


def print_legacy_profile_lines(profile):
    for line in legacy_profile_lines(profile):
        print(line, flush=True)


def get_args(argv=None):
    ap = argparse.ArgumentParser(prog="qbench profile", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="results/<run> directory")
    ap.add_argument("--source", action="append", required=True, choices=SOURCES)
    ap.add_argument("--nvidia-smi", default="nvidia-smi")
    ap.add_argument("--interval-s", type=float, default=1.0)
    ap.add_argument("--energy-s", type=float, help="fixed energy window instead of the run's")
    ap.add_argument("--gpu-cost-hr", type=float, help="USD per GPU-hour")
    ap.add_argument("--gsm8k-file", help="GSM8k JSONL ({question, answer})")
    ap.add_argument("--gsm8k-n", type=int)
    ap.add_argument("--tokenizer", help="gsm8k tokenizer; default config engine.model")
    ap.add_argument("--served-model", help="OpenAI model name; default config engine.model")
    ap.add_argument("--frontend-args", default="", help='serving_frontend flags, e.g. "--gpu-label H200"')
    ap.add_argument("--pd-host-args", default="", help="pd_host flags")
    ap.add_argument("--kv-transfer-csv", help="the CSV kv_transfer's torchrun run wrote")
    ap.add_argument("--kernel-steps", type=int, help="decode steps the kernels window holds")
    ap.add_argument("--kernel-window", help="NVTX range the kernels source reads (default all)")
    ap.add_argument("--kernel-subtract", help="NVTX range whose kernel totals come off first")
    return ap, ap.parse_args(argv)


def main(argv=None):
    ap, a = get_args(argv)
    args = {"nvidia_smi": a.nvidia_smi, "interval_s": a.interval_s, "energy_s": a.energy_s,
            "gpu_cost_hr": a.gpu_cost_hr, "gsm8k_file": a.gsm8k_file, "gsm8k_n": a.gsm8k_n,
            "tokenizer": a.tokenizer, "served_model": a.served_model,
            "frontend_argv": shlex.split(a.frontend_args),
            "pd_host_argv": shlex.split(a.pd_host_args), "kv_transfer_csv": a.kv_transfer_csv,
            "kernel_steps": a.kernel_steps, "kernel_window": a.kernel_window,
            "kernel_subtract": a.kernel_subtract}
    try:
        out = profile(a.run, a.source, **args)
    except ValueError as e:
        ap.error(str(e))
    print_legacy_profile_lines(out)
    for source, message in out["errors"].items():
        print(f"PROFILE_ERROR source={source} {message}", file=sys.stderr)
    return 1 if out["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
