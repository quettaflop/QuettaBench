# Benchmark changelog

Each entry names the workloads and hardware a change affects. The bench-dispatch
workflow parses new entries on push to main and emits one benchmark job per
(workload, hardware) pair. Grammar per entry (a list item):

    - <what changed> -- workloads: <a, b> -- hardware: <H200, RTXPRO6000>

## unreleased

- serving replay energy and cost columns added -- workloads: llama -- hardware: H200
