Architecture
============

A benchmark run is a pipeline from workload definition to scored
results. Modules are listed in the order a request flows through them.

Workloads
---------

:mod:`src.workloads.profiles`
    Registry of workload profiles. A profile names its dataset, data
    file, length bounds, execution mode and whether the server must
    run with prefix caching.

:mod:`src.workloads.dataset`
    Dataset factory and classes. A dataset yields one request at a
    time: messages or token ids, a max_tokens target and request
    metadata.

:mod:`src.workloads.mooncake`
    Mooncake production trace support. Parses the trace and expands
    hash ids into deterministic 512 token blocks so prefix sharing
    survives replay.

:mod:`src.workloads.distributional`
    Distributional synthetic multi turn sessions, sampled from the
    compact trace distributions loaded by
    :mod:`src.workloads.trace_distributions`.

:mod:`src.workloads.arrival`
    Arrival schedules: steady, poisson and ramp.

Execution
---------

:mod:`src.benchmark.runner`
    CLI entry point. Builds the dataset and arrival schedule, warms up
    the server, dispatches requests closed loop or open loop, and
    writes the results JSON.

:mod:`src.modes`
    The three execution modes (stress test, single turn, multi turn)
    and the flags each requires.

:mod:`src.engines`
    Backend registry. openai, vllm and sglang post chat completions;
    vllm-completions posts to /v1/completions and accepts token id
    prompts for exact trace replay.

:mod:`src.benchmark.metrics`
    Per request results and aggregation: p50, p90 and p99 for TTFT,
    TPOT, ITL and E2EL.

:mod:`src.benchmark.server_control`
    Server state control between sweep cells, for example prefix
    cache resets.

Scripts
-------

Standalone tools, run from the repo root:

``scripts/synthesize_mooncake_trace.py``
    Samples a synthetic Mooncake style trace from a real one.

``scripts/audit_live_replay.py``
    Audits a replay result against its source trace with the server's
    reported token counts.

``scripts/score_replay_records.py``
    Compares two replay results and applies the acceptance gates.
