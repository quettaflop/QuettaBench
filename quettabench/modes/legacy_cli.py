"""The old runner command line: argparse, flag checks, the --mode side
effects and the results-file config dict. serve.main drives it; commit 7
removes it with the old CLI.

--grid is the old vmin_fit.py command line (workload name, grid name,
--model, --nograph, --allow-unverified): grid_translate resolves the
workload as vmin_fit did, its exit messages included.
"""

import argparse
import sys
from pathlib import Path

from ..config import env_map, xval_config
from ..config.schema import Config, EngineCfg, GridCfg, HardwareCfg, ModeCfg, WorkloadCfg
from ..engines import SUPPORTED_BACKENDS, get_backend
from ..engines import vllm as vllm_adapter
from ..engines.drivers import InProcessDriver
from ..metrics.results import RunWriter
from ..workloads.arrival import ARRIVAL_PATTERNS, arrival_spec
from ..workloads.grid import PROMPT_MODES, GridTrace
from ..workloads.profile_trace import ProfileSessionTrace, ProfileTrace
from ..workloads.profiles import get_profile
from ..workloads.sessions import (
    TurnPacing,
    _load_source_session_ids,
    resolve_multi_turn_num_sessions,
)

# Closed loop: a fixed population of `concurrency` in-flight requests, each
# replaced only when the previous one finishes. Offered load is a consequence of
# server speed, so the server can never fall behind.
# Open loop: requests arrive on a clock at --target-rate regardless of whether
# the server keeps up, so queues can grow without bound. This is the one that
# exposes saturation.
LOAD_MODES = ("closed-loop", "open-loop")
OPEN_LOOP_ARRIVALS = ("poisson", "ramp")
TURN_PACINGS = ("per-session", "interleaved")
# Flags added after the old results layout froze; kept out of its config dict.
NEW_FLAGS = ("results_dir", "sla_ttft_ms", "sla_tpot_ms", "goodput_sweep",
             "grid", "workload", "nograph", "allow_unverified", "dump_tokens")


def _resolve_cache_state(cli_value: str, profile) -> tuple[str, str]:
    if cli_value != "auto":
        return cli_value, "cli"
    return (
        "expected_on" if profile.prefix_caching_required else "expected_off",
        "profile_default",
    )


def _resolve_tri_state(cli_value: str) -> tuple[str, str]:
    if cli_value != "auto":
        return cli_value, "cli"
    return "unknown", "not_reported"


def normalize_dashboard_scope(scope: str) -> str:
    if scope in {"latest", "synthetic", "synthetic-distributional", "synthetic_distributional"}:
        return "synthetic_distributional"
    if scope in {"archive", "trace_replay"}:
        return "trace_replay"
    if scope in {"current", "canonical", "fixed", "fixed-grid", "mse", "archived"}:
        return "archived"
    return scope


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="inference-benchmark runner")
    parser.add_argument("--url", required=False, help="Server endpoint URL")
    parser.add_argument("--model", required=False)
    parser.add_argument("--backend", default="vllm", choices=SUPPORTED_BACKENDS,
                        help="Backend type (vllm/sglang/openai → /v1/chat/completions, "
                             "vllm-completions → /v1/completions)")
    parser.add_argument("--profile", default="chat-singleturn", help="Workload profile name")
    parser.add_argument("--workload-file", default="",
                        help="Override the profile file_path, for trace variants")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--multi-turn-sessions", type=int, default=None,
                        help="Override number of multi-turn sessions to load/sample. Floored at --concurrency.")
    parser.add_argument("--max-turn-index", type=int, default=None,
                        help="For multi-turn runs, stop after this zero-based turn index.")
    parser.add_argument("--source-session-ids-file", default=None,
                        help="Validation mode: source-lock distributional multi-turn sampling to these source_session_id values.")
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--api-key", default="test")
    parser.add_argument("--arrival", default="steady", choices=ARRIVAL_PATTERNS)
    parser.add_argument("--turn-pacing", default="per-session",
                        choices=list(TURN_PACINGS),
                        help="closed-loop multi-turn scheduling: 'per-session' (default) "
                             "runs a session's turns back to back with no cross-session "
                             "barrier; 'interleaved' is a turn-aligned herd (global "
                             "barrier per turn — schedule artifacts in TTFT/TPOT, not "
                             "production traffic). Ignored under --open-loop.")
    parser.add_argument("--tool-wait-ms", type=float, default=0.0,
                        help="Fixed sleep after each turn before the next turn in the "
                             "same session (ms). Default 0 (auto-mode / no tool gap). "
                             "Ignored under --turn-pacing interleaved.")
    parser.add_argument("--human-wait-ms", type=float, default=0.0,
                        help="Extra sleep after each turn for a human-approval gap (ms). "
                             "Default 0. Added to --tool-wait-ms. Ignored under "
                             "--turn-pacing interleaved.")
    parser.add_argument("--use-recorded-waits", action="store_true",
                        help="Add per-turn tool_wait_after_ms / human_wait_after_ms from "
                             "the trajectory (tool_ms / human_ms aliases accepted). "
                             "Default off. Still adds the CLI --tool-wait-ms / "
                             "--human-wait-ms values.")
    parser.add_argument("--use-recorded-arrivals", action="store_true",
                        help="Start each session at its recorded arrival_time_ms instead "
                             "of t=0 (closed-loop) or a generated Poisson/ramp "
                             "(open-loop). Default off. Ignored under interleaved.")
    parser.add_argument("--target-rate", type=float, default=10.0, help="req/s for poisson/ramp")
    load_group = parser.add_mutually_exclusive_group()
    load_group.add_argument(
        "--closed-loop", dest="load_mode", action="store_const", const="closed-loop",
        help="Closed loop (default): hold --concurrency requests in flight, each replaced "
             "only when the previous finishes. Offered load is capped by server speed, so "
             "the server can never fall behind. Implies --arrival steady. Multi-turn "
             "defaults to --turn-pacing per-session.")
    load_group.add_argument(
        "--open-loop", dest="load_mode", action="store_const", const="open-loop",
        help="Open loop: fire arrivals on a clock at --target-rate with NO client-side "
             "concurrency cap, so an overloaded server queues instead of back-pressuring "
             "the client. Requires --arrival poisson|ramp. Single-turn: independent "
             "requests. Multi-turn: session starts are open-loop; turns stay sequential "
             "inside each session.")
    parser.set_defaults(load_mode="closed-loop")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature. Default 0.0 (greedy) so output lengths "
                             "are reproducible across runs.")
    parser.add_argument("--sampling-seed", type=int, default=None,
                        help="Seed forwarded to the server sampler. Defaults to --seed; "
                             "pass -1 to omit it entirely.")
    parser.add_argument("--exact-output-length", action="store_true",
                        help="Pin generated length to the profile's per-request max_tokens "
                             "(sends min_tokens=max_tokens and ignore_eos). Makes measured OSL "
                             "equal planned OSL instead of wherever the model chose to stop.")
    parser.add_argument("--reset-prefix-cache", action="store_true",
                        help="POST the server's prefix-cache reset endpoint before the run. "
                             "Sweeps replay byte-identical prompts across cells, so without "
                             "this every cell after the first is measured warm.")
    parser.add_argument("--warmup-concurrency", type=int, default=None, metavar="N",
                        help="Batch width to warm at. Defaults to --concurrency under "
                             "closed loop (where that IS the in-flight cap), and to "
                             "min(--num-requests, 64) under open loop (where it is not). "
                             "Engines that JIT kernels per batch shape compile mid-run for "
                             "any width warmup missed.")
    parser.add_argument("--discard-first", type=int, default=0, metavar="N",
                        help="Hold the first N requests out of the summary (they stay in "
                             "per_request). Removes the startup wave. Only meaningful once "
                             "--num-requests is several times --concurrency; at 2x there is "
                             "no settled middle left to keep. Single-turn only.")
    parser.add_argument("--allow-missing-usage", action="store_true",
                        help="Permit a run where no response carried a usage block. Token "
                             "counts and throughputs will be zero-filled, not measured.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output", default="results/latest.json")
    parser.add_argument("--sla-ttft-ms", type=float, default=None,
                        help="SLA: max p99 TTFT ms; adds SLASUM and BURSTSUM.")
    parser.add_argument("--sla-tpot-ms", type=float, default=None,
                        help="SLA: max p99 TPOT ms/token; adds SLASUM.")
    parser.add_argument("--goodput-sweep", action="store_true",
                        help="Bisect arrival speed to the SLA PASS/FAIL boundary (max 6 "
                             "probes); needs --open-loop and an --sla-* bound.")
    parser.add_argument("--results-dir", default=None,
                        help="Also write the new layout (run.json, requests.jsonl) to this "
                             "directory, results/<run>/ by convention.")
    parser.add_argument("--max-context-tokens", type=int, default=None,
                        help="Optional cap for distributional multi-turn synthetic prompt context")
    parser.add_argument("--context-safety-margin-tokens", type=int, default=256,
                        help="Reserved token headroom under --max-context-tokens for output and tokenizer mismatch")
    parser.add_argument("--prefix-cache-block-size", type=int, default=16,
                        help="KV prefix-cache block size in tokens for block-aligned cache estimates")
    parser.add_argument("--prefix-caching-state", choices=["auto", "on", "off", "unknown"],
                        default="auto",
                        help="Metadata only: actual server prefix-cache state when known")
    parser.add_argument("--chunked-prefill", choices=["auto", "on", "off", "unknown"],
                        default="auto",
                        help="Metadata only: actual server chunked-prefill state when known")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Metadata only: server --max-model-len")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None,
                        help="Metadata only: server GPU memory utilization target")
    parser.add_argument("--tensor-parallel-size", type=int, default=None,
                        help="Required: server tensor parallel size. Recorded in the result "
                             "config and used to compose the parallelism label (1gpu/tp/tp+ep).")
    parser.add_argument("--dtype", default=None,
                        help="Metadata only: server compute dtype")
    parser.add_argument("--kv-cache-dtype", default=None,
                        help="Metadata only: server KV-cache dtype")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None,
                        help="Metadata only: server max_num_batched_tokens")
    parser.add_argument("--max-num-seqs", type=int, default=None,
                        help="Metadata only: server max_num_seqs")
    parser.add_argument("--ignore-eos", action="store_true",
                        help="Pass ignore_eos=true to vLLM (needed for FP8 models with random token workloads)")
    parser.add_argument("--mode", choices=["stress-test", "single-turn", "multi-turn"],
                        help="Benchmark mode (sets profile defaults and required flags). "
                             "Use --profile for a specific profile within a mode.")
    parser.add_argument(
        "--scope",
        choices=[
            "synthetic_distributional",
            "synthetic-distributional",
            "trace_replay",
            "archived",
            "synthetic",
            "latest",
            "current",
            "canonical",
            "archive",
            "fixed",
            "fixed-grid",
            "mse",
            "moe_ep",
        ],
        default=None,
        help="Dashboard scope override (default: *-synth→synthetic_distributional, active→archived, inactive→trace_replay)",
    )
    parser.add_argument("--min-success-rate", type=float, default=0.75, dest="min_success_rate",
                        help="Minimum success rate (0.0-1.0). Runs below this threshold exit with an error. Default: 0.75")
    parser.add_argument("--list-profiles", action="store_true", help="List available profiles and exit")
    parser.add_argument("--include-inactive", action="store_true",
                        help="With --list-profiles, include legacy/inactive profiles")
    parser.add_argument("--trace-request-ids", action="store_true",
                        help="Send stable request_id/X-Request-Id values for vLLM engine tracing.")
    parser.add_argument("--agent-type", type=str, default=None, help="Filter profiles by agent type")
    parser.add_argument("--turn-style", type=str, default=None, help="Filter profiles by turn style")
    parser.add_argument("--serving-style", type=str, default=None, help="Filter profiles by serving style")
    parser.add_argument("--data-source", type=str, default=None, help="Filter profiles by data source")
    grid = parser.add_argument_group("grid runs (the old modes/vmin_fit.py)")
    grid.add_argument("--grid", nargs="?", const="", default=None, metavar="NAME",
                      help="Time the workload's (ctx, bs) grid in-process; NAME overrides "
                           "the workload's own grid")
    grid.add_argument("--workload", help="Grid runs: workload name from configs/workloads.json")
    grid.add_argument("--nograph", action="store_true", help="Grid runs: disable CUDA graphs")
    grid.add_argument("--allow-unverified", action="store_true",
                      help="Grid runs: run a workload whose verified flag is false (bring-up only)")
    grid.add_argument("--dump-tokens", action="store_true",
                      help="Grid runs: also decode 64 greedy tokens per cell and write their "
                           "token ids as rows (see --results-dir)")
    args = parser.parse_args(argv)
    if args.profile == "mooncake-trace" and args.backend != "vllm-completions":
        parser.error("--profile mooncake-trace sends token id prompts; "
                     "use --backend vllm-completions")
    return args



def list_profiles(args) -> None:
    """--list-profiles: print the filtered profile table."""
    from ..workloads.profiles import filter_profiles, PROFILES
    filtered = filter_profiles(
        agent_type=args.agent_type,
        turn_style=args.turn_style,
        serving_style=args.serving_style,
        data_source=args.data_source,
        include_inactive=args.include_inactive,
    )
    print(f"\n{'Name':<30} {'Agent Type':<18} {'Turn Style':<14} {'Serving':<20} {'Data Source':<12} {'ISL':<6} {'OSL':<6}")
    print("-" * 110)
    for name, p in sorted(filtered.items()):
        print(f"{name:<30} {p.agent_type:<18} {p.turn_style:<14} {p.serving_style:<20} {p.data_source:<12} {p.isl_tokens:<6} {p.osl_tokens:<6}")
    inactive_note = " including inactive" if args.include_inactive else ""
    print(f"\n{len(filtered)} profiles shown{inactive_note} (of {len(PROFILES)} total)")
    if any([args.agent_type, args.turn_style, args.serving_style, args.data_source]):
        active = []
        if args.agent_type: active.append(f"agent_type={args.agent_type}")
        if args.turn_style: active.append(f"turn_style={args.turn_style}")
        if args.serving_style: active.append(f"serving_style={args.serving_style}")
        if args.data_source: active.append(f"data_source={args.data_source}")
        print(f"Filters: {', '.join(active)}")


def check_args(args) -> None:
    """Refuse bad flag combinations (exit 1) and fill the derived defaults:
    warmup width and sampling seed."""
    # --url and --model are required for actual benchmark runs
    if not args.url or not args.model:
        print("Error: --url and --model are required for benchmark runs.")
        print("Use --list-profiles to browse profiles without a server.")
        sys.exit(1)

    # --tensor-parallel-size is load-bearing (it composes the parallelism label
    # recorded in the result config), so refuse to guess it. Silently defaulting
    # to 1 would mislabel every multi-GPU sweep.
    if args.tensor_parallel_size is None:
        print("Error: --tensor-parallel-size is required for benchmark runs "
              "(pass 1 for single-GPU serving).")
        sys.exit(1)
    if args.tensor_parallel_size < 1:
        print(f"Error: --tensor-parallel-size must be >= 1, got {args.tensor_parallel_size}.")
        sys.exit(1)

    # Load mode and arrival pattern have to agree. "steady" means every request
    # is scheduled at t=0 and the semaphore paces them, which is the definition
    # of closed loop; a rate process only means anything with the cap removed.
    if args.load_mode == "open-loop":
        if args.use_recorded_arrivals:
            # Recorded arrival_time_ms replaces the generated pattern, so the
            # poisson|ramp requirement (and --target-rate) does not apply.
            if args.arrival in OPEN_LOOP_ARRIVALS:
                print("WARNING: --use-recorded-arrivals replays recorded "
                      "arrival_time_ms; --arrival and --target-rate are ignored.")
        elif args.arrival not in OPEN_LOOP_ARRIVALS:
            print(f"Error: --open-loop requires --arrival {'|'.join(OPEN_LOOP_ARRIVALS)}, "
                  f"got '{args.arrival}'. 'steady' schedules everything at t=0, which is "
                  f"closed-loop by construction. Multi-turn trajectory replays can "
                  f"exempt this with --use-recorded-arrivals.")
            sys.exit(1)
        elif args.target_rate <= 0:
            print(f"Error: --open-loop requires a positive --target-rate, got {args.target_rate}.")
            sys.exit(1)
    elif args.arrival in OPEN_LOOP_ARRIVALS:
        print(f"Error: --arrival {args.arrival} describes open-loop offered load, but the run "
              f"is closed-loop, so --concurrency would still cap it. Pass --open-loop to "
              f"remove the cap, or use --arrival steady.")
        sys.exit(1)

    if args.goodput_sweep and (args.load_mode != "open-loop"
                               or not (args.sla_ttft_ms or args.sla_tpot_ms)):
        print("Error: --goodput-sweep sweeps arrival speed against an SLA; it needs "
              "--open-loop and --sla-ttft-ms and/or --sla-tpot-ms.")
        sys.exit(1)

    if args.tool_wait_ms < 0 or args.human_wait_ms < 0:
        print("Error: --tool-wait-ms and --human-wait-ms must be >= 0.")
        sys.exit(1)

    if args.concurrency < 1:
        print(f"Error: --concurrency must be >= 1, got {args.concurrency}.")
        sys.exit(1)
    if args.num_requests < 1:
        print(f"Error: --num-requests must be >= 1, got {args.num_requests}.")
        sys.exit(1)
    if args.warmup < 0:
        print(f"Error: --warmup must be >= 0, got {args.warmup}.")
        sys.exit(1)
    if args.max_turn_index is not None and args.max_turn_index < 0:
        print(f"Error: --max-turn-index must be >= 0, got {args.max_turn_index}.")
        sys.exit(1)
    if not 0.0 <= args.min_success_rate <= 1.0:
        print(f"Error: --min-success-rate must be in [0, 1], got {args.min_success_rate}.")
        sys.exit(1)

    # Warmup width. Under closed loop --concurrency is the real in-flight cap, so
    # it is the right width. Under open loop it caps nothing, so sizing warmup
    # from it can miss the batch shapes the run reaches -- default to the upper
    # bound on in-flight (num_requests), capped to keep warmup affordable.
    if args.warmup_concurrency is None:
        if args.load_mode == "open-loop":
            args.warmup_concurrency = max(1, min(args.num_requests, 64))
        else:
            args.warmup_concurrency = args.concurrency
    elif args.warmup_concurrency < 1:
        print(f"Error: --warmup-concurrency must be >= 1, got {args.warmup_concurrency}.")
        sys.exit(1)

    # -1 is the explicit opt-out; otherwise the sampler shares the workload seed
    # so identical prompts produce identical generations.
    if args.sampling_seed is None:
        args.sampling_seed = args.seed
    elif args.sampling_seed < 0:
        args.sampling_seed = None


def apply_mode(args) -> None:
    """--mode side effects: multi-turn and stress-test swap the default profile,
    stress-test forces ignore_eos; each prints what it needs from the server."""
    if not args.mode:
        return
    if args.mode == "multi-turn":
        print("NOTE: multi-turn mode requires server launched with --enable-prefix-caching (vLLM)")
        if args.profile == "chat-singleturn":  # default — override for multi-turn
            args.profile = "chat-multiturn"
    if args.mode == "stress-test":
        if not args.ignore_eos:
            print("NOTE: stress-test mode auto-enables --ignore-eos (required for FP8 models)")
            args.ignore_eos = True
        if args.profile == "chat-singleturn":  # default — override for stress-test
            args.profile = "random-1k"
    if args.mode == "single-turn":
        print("NOTE: single-turn mode requires server launched with --enable-prefix-caching (vLLM)")
        print("      or radix cache (SGLang default). See scripts/launch_server.sh")


def build_config(args):
    """The results-file config dict for a checked run, plus what the run needs
    beyond args: (config, profile, num_sessions, source_session_ids)."""
    profile = get_profile(args.profile)
    profile_name = profile.name
    if args.discard_first < 0:
        print(f"Error: --discard-first must be >= 0, got {args.discard_first}.")
        sys.exit(1)
    if args.discard_first and profile.mode == "multi-turn":
        print(f"Error: --discard-first is not supported for multi-turn profile "
              f"'{profile_name}'. Every turn has its own barrier transient, so a global "
              f"first-N has no consistent meaning; use the per-turn breakdown instead.")
        sys.exit(1)
    if args.discard_first >= args.num_requests and profile.mode != "multi-turn":
        print(f"Error: --discard-first {args.discard_first} would hold out every request "
              f"of {args.num_requests}.")
        sys.exit(1)
    if args.multi_turn_sessions is not None and args.multi_turn_sessions <= 0:
        print("Error: --multi-turn-sessions must be positive when provided.")
        sys.exit(1)
    effective_num_sessions, num_sessions_source = resolve_multi_turn_num_sessions(
        profile,
        args.concurrency,
        args.multi_turn_sessions,
    )
    source_session_ids = _load_source_session_ids(args.source_session_ids_file)
    if source_session_ids is not None and profile.dataset != "distributional-multi-turn":
        print("--source-session-ids-file is only valid for distributional multi-turn profiles.")
        sys.exit(1)
    if source_session_ids is not None:
        effective_num_sessions = len(source_session_ids)
        num_sessions_source = "source_session_ids_file"
    prefix_caching_state, prefix_caching_state_source = _resolve_cache_state(
        args.prefix_caching_state,
        profile,
    )
    chunked_prefill_state, chunked_prefill_state_source = _resolve_tri_state(
        args.chunked_prefill,
    )
    scope = args.scope
    if scope is None:
        scope = "synthetic_distributional" if profile_name.endswith("-synth") else ("archived" if profile.active else "trace_replay")
    else:
        scope = normalize_dashboard_scope(scope)
    # Expert parallelism is enabled by the launcher via ENABLE_EP (moe_ep scope).
    # Record it explicitly so EP-on runs are labelled in the result data itself,
    # not just inferred from the scope. ep_size mirrors the launcher's --ep-size $TP.
    enable_ep = env_map.read_env(only=("ENABLE_EP",)).get("engine.ep", False)
    ep_size = args.tensor_parallel_size if enable_ep else 1
    # Canonical parallelism-strategy label composed from the active axes:
    # "1gpu" (single GPU), "tp", "tp+ep". Mirrors compile_sweep.parallelism_label.
    _par_axes = (["tp"] if args.tensor_parallel_size > 1 else []) + (["ep"] if enable_ep else [])
    parallelism = "+".join(_par_axes) or "1gpu"
    config = {
        **{k: v for k, v in vars(args).items() if k not in NEW_FLAGS},
        "profile": profile_name,
        "mode": args.mode or profile.mode,
        "dashboard_scope": scope,
        "enable_ep": enable_ep,
        "ep_size": ep_size,
        "parallelism": parallelism,
        "turn_pacing": args.turn_pacing,
        "tool_wait_ms": args.tool_wait_ms,
        "human_wait_ms": args.human_wait_ms,
        "use_recorded_waits": args.use_recorded_waits,
        "use_recorded_arrivals": args.use_recorded_arrivals,
        "reply_feedback": True,
        "load_mode": args.load_mode,
        "sampling": {
            "temperature": args.temperature,
            "sampling_seed": args.sampling_seed,
            "exact_output_length": args.exact_output_length,
            "ignore_eos": args.ignore_eos or args.exact_output_length,
            "output_length_controlled": args.exact_output_length,
        },
        "prefix_cache_reset_before_run": args.reset_prefix_cache,
        "warmup_style": "profile_shaped_at_concurrency" if args.warmup else "none",
        "discard_first": args.discard_first,
        "warmup_concurrency": args.warmup_concurrency,
        "requests_per_concurrency_slot": (
            round(args.num_requests / args.concurrency, 2) if args.concurrency else None
        ),
        "profile_metadata": {
            "dataset": profile.dataset,
            "agent_type": profile.agent_type,
            "turn_style": profile.turn_style,
            "serving_style": profile.serving_style,
            "data_source": profile.data_source,
            "active": profile.active,
            "prefix_caching_required": profile.prefix_caching_required,
            "isl_tokens": profile.isl_tokens,
            "osl_tokens": profile.osl_tokens,
            "min_turns": profile.min_turns,
            "max_turns": profile.max_turns,
            "num_sessions": effective_num_sessions,
            "profile_num_sessions": profile.num_sessions,
            "num_sessions_source": num_sessions_source,
            "source_session_ids_count": (
                len(source_session_ids) if source_session_ids is not None else None
            ),
        },
        "prediction_metadata": {
            "prefix_caching_state": prefix_caching_state,
            "prefix_caching_state_source": prefix_caching_state_source,
            "prefix_cache_block_size": args.prefix_cache_block_size,
            "chunked_prefill": chunked_prefill_state,
            "chunked_prefill_source": chunked_prefill_state_source,
            "max_context_tokens": args.max_context_tokens,
            "context_safety_margin_tokens": args.context_safety_margin_tokens,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.dtype,
            "kv_cache_dtype": args.kv_cache_dtype,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "logical_cache_estimate": "previous_prompt_tokens",
            "block_aligned_cache_estimate": "floor(previous_context / block_size) * block_size",
            "engine_cache_telemetry": "not_available",
        },
    }
    return config, profile, effective_num_sessions, source_session_ids


def arrival_of(args, multi_turn):
    """The run's arrival spec. Open-loop multi-turn with recorded arrivals
    ignores the schedule but has always reported --target-rate as offered,
    so it keeps a poisson spec at that rate."""
    if multi_turn and args.load_mode == "open-loop" and args.use_recorded_arrivals \
            and args.arrival == "steady":
        return f"poisson:{args.target_rate}"
    return arrival_spec(args.arrival, args.target_rate)


def run_config(args, profile, enable_ep):
    """The Config the old flags describe; engine fields are metadata only,
    since the old runner never launches the server."""
    multi = profile.mode == "multi-turn"
    engine = {k: v for k, v in {
        "model": args.model, "tp": args.tensor_parallel_size, "ep": enable_ep,
        "dtype": args.dtype, "kv_dtype": args.kv_cache_dtype,
        "max_model_len": args.max_model_len, "gpu_util": args.gpu_memory_utilization,
    }.items() if v is not None}
    vllm_extra = {k: getattr(args, k) for k in ("max_num_seqs", "max_num_batched_tokens")
                  if getattr(args, k) is not None}
    mode = ModeCfg(
        serve_profile=args.mode, load="open" if args.load_mode == "open-loop" else "closed",
        concurrency=args.concurrency, warmup=args.warmup,
        warmup_concurrency=args.warmup_concurrency, discard_first=args.discard_first,
        ignore_eos=args.ignore_eos, temperature=args.temperature,
        sampling_seed=args.sampling_seed, exact_output_length=args.exact_output_length,
        timeout=args.timeout, min_success_rate=args.min_success_rate,
        reset_prefix_cache=args.reset_prefix_cache,
        allow_missing_usage=args.allow_missing_usage,
        trace_request_ids=args.trace_request_ids,
        prefix_cache_block_size=args.prefix_cache_block_size,
        sla_ttft_ms=args.sla_ttft_ms, sla_tpot_ms=args.sla_tpot_ms,
        goodput_sweep=args.goodput_sweep)
    workload = WorkloadCfg(kind="sessions" if multi else "live", source=profile.name,
                           arrival=arrival_of(args, multi), seed=args.seed,
                           num_requests=args.num_requests)
    cfg = Config(engine=EngineCfg(**engine, extra={"vllm": vllm_extra} if vllm_extra else {}),
                 workload=workload, mode=mode, hardware=HardwareCfg(),
                 run_name=Path(args.results_dir).name if args.results_dir else "")
    origins = {f"engine.{k}": "legacy_cli" for k in engine}
    for section in ("mode", "workload"):
        origins.update({f"{section}.{k}": "legacy_cli" for k in vars(getattr(cfg, section))})
    return Config(**{**vars(cfg), "origins": dict(sorted(origins.items()))})


def translate(args):
    """(old config dict, Config, Trace, Driver, RunWriter) for a checked run."""
    config, profile, num_sessions, source_session_ids = build_config(args)
    cfg = run_config(args, profile, config["enable_ep"])
    dataset_kw = {"max_context_tokens": args.max_context_tokens, "random_seed": args.seed,
                  "context_safety_margin_tokens": args.context_safety_margin_tokens,
                  "tokenizer_name": args.model}
    if profile.mode == "multi-turn":
        pacing = TurnPacing(args.turn_pacing, args.tool_wait_ms, args.human_wait_ms,
                            args.use_recorded_waits, args.use_recorded_arrivals)
        trace = ProfileSessionTrace(profile, warmup=args.warmup,
                                    max_turn_index=args.max_turn_index, pacing=pacing,
                                    num_sessions=num_sessions,
                                    source_session_ids=source_session_ids, **dataset_kw)
    else:
        if (args.use_recorded_arrivals or args.use_recorded_waits
                or args.tool_wait_ms or args.human_wait_ms):
            print("WARNING: --use-recorded-arrivals / --use-recorded-waits / "
                  "--tool-wait-ms / --human-wait-ms apply to multi-turn "
                  "profiles only; ignoring them.")
        trace = ProfileTrace(profile, args.num_requests, warmup=args.warmup,
                             workload_file=args.workload_file, **dataset_kw)
    driver = get_backend(args.backend, url=args.url, model=args.model, api_key=args.api_key,
                         timeout=args.timeout)
    return config, cfg, trace, driver, RunWriter(args.results_dir, cfg, driver=driver)


def grid_config(args):
    """The Config of a --grid run, resolved from the workload record the way
    vmin_fit resolved it: env knobs (XVAL_PP, GPU_UTIL, PROMPT_MODE,
    VMIN_ATTEMPTS, XVAL_EVAL_POS, XVAL_CORPUS/XVAL_TRACE, NCCL_*, QS_SHA)
    through env_map, the collective block for the link profile from
    xval_config. Exits with vmin_fit's messages on a bad name."""
    workloads = xval_config.workloads()
    name = args.workload
    wl = workloads.get(name)
    if wl is None:
        sys.exit(f"unknown crate {name!r}; workloads: {', '.join(workloads)}")
    if not wl.get("verified") and not args.allow_unverified:
        sys.exit(f"{name} is not validated yet; pass --allow-unverified to bring it up")
    grids = {k.lower(): v for k, v in xval_config.grids().items()}
    grid = (args.grid or wl["grid"]).lower()
    if grid not in grids:
        sys.exit(f"unknown grid {grid!r}; grids: {', '.join(grids)}")
    found = env_map.read_env()
    prompt_mode = found.get("workload.prompt_mode", wl.get("prompt_mode", "distinct"))
    if prompt_mode not in PROMPT_MODES:
        sys.exit(f"unknown prompt_mode {prompt_mode!r}; use distinct, clone, corpus, synth or trace")

    link = found.get("hardware.link_profile") or xval_config.link_profile()
    # The caller's NCCL_ALGO/PROTO win over the yaml, as vmin_fit's setdefault let them.
    collective = {**xval_config.collective(profile=link),
                  **{k: found[f"hardware.collective.{k}"] for k in ("NCCL_ALGO", "NCCL_PROTO")
                     if f"hardware.collective.{k}" in found}}
    engine = {"model": args.model, "model_name": wl["name"], "tp": int(wl.get("tp", 1)),
              "pp": found.get("engine.pp", int(wl.get("pp", 1))),
              "ep": bool(wl.get("expert_parallel")), "dtype": wl["dtype"],
              "kv_dtype": wl.get("kv_dtype") or "auto", "quant": wl.get("quant") or None,
              "max_model_len": int(wl["maxlen"]),
              "gpu_util": found.get("engine.gpu_util", float(wl.get("gpu_util", 0.90))),
              "cudagraph": not args.nograph, "weights_gib": float(wl["weights_gib"]),
              "kv_bytes": int(wl["kv_bytes"]) if wl.get("kv_bytes") else None,
              "verified": bool(wl.get("verified"))}
    mode = {"grid": grid, "returns": "text" if args.dump_tokens else "none",
            **{k: found[f"mode.{k}"] for k in ("vmin_attempts", "eval_pos") if f"mode.{k}" in found}}
    workload = {"kind": "grid", "prompt_mode": prompt_mode, "source": found.get("workload.source", "")}
    hardware = {"link_profile": link, "collective": {k: v for k, v in collective.items() if v},
                "timing_steps": xval_config.run_params()["timing_steps"]}
    steps = tuple(xval_config.step_points())
    cells = tuple(tuple(c) for c in grids[grid])
    over = tuple(c for c in cells if c[0] + max(steps) + 2 > engine["max_model_len"])
    origins = {f"{section}.{k}": "legacy_cli" for section, part in
               (("engine", engine), ("mode", mode), ("workload", workload), ("hardware", hardware))
               for k in part}
    origins.update({path: "env" for path in found})
    engine_sha = found.get("provenance.engine_sha")
    return Config(engine=EngineCfg(**engine), workload=WorkloadCfg(**workload),
                  mode=ModeCfg(**mode), hardware=HardwareCfg(**hardware),
                  grid=GridCfg(cells=cells, step_points=steps, cells_over_maxlen=over),
                  run_name=name if not args.grid else f"{name}-{args.grid}",
                  provenance={"engine_sha": engine_sha} if engine_sha else {},
                  origins=dict(sorted(origins.items())))


def grid_translate(args):
    """(Config, GridTrace, InProcessDriver, RunWriter) for a --grid run. The
    collective env goes into this process first: the engine's NCCL reads it."""
    cfg = grid_config(args)
    env_map.export(cfg.hardware.collective)
    driver = InProcessDriver(vllm_adapter.ADAPTER.translate(cfg))
    trace = GridTrace.from_config(cfg, tokenizer=driver.tokenizer)
    return cfg, trace, driver, RunWriter(args.results_dir, cfg, driver=driver)
