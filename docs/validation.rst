Validation
==========

QuettaBench checks its own measurements four ways: delivery audits
against the server's token accounting, run to run repeatability,
agreement with an independent tool on identical prompts, and gates on
synthetic traces against their source. The numbers below come from the
validation run on a single H200 with Llama 3.1 8B on vLLM 0.19.0 with
prefix caching enabled, against the first 1000 records of the Mooncake
conversation trace.

Delivery
--------

Server reported token counts match the trace exactly on every audited
request, about 7000 requests across runs. A replay fails if any
request's input or output length differs from the trace, if usage is
missing, or if more than 1 percent of requests fail.

Repeatability
-------------

Two replays of the same trace agree within 2 percent on all reported
statistics.

Cross tool agreement
--------------------

QuettaBench and vllm bench serve measured the same prompts, sent one
at a time so queueing does not affect the numbers, and agree within
1.2 percent. Values are milliseconds:

.. list-table::
   :header-rows: 1

   * - Workload
     - Metric
     - QuettaBench
     - vllm bench serve
   * - trace prompts
     - TTFT mean / median
     - 519.65 / 244.43
     - 522.07 / 243.67
   * - trace prompts
     - TPOT mean / median
     - 5.35 / 5.22
     - 5.39 / 5.27
   * - synthetic prompts
     - TTFT mean / median
     - 653.60 / 191.23
     - 657.62 / 189.34
   * - synthetic prompts
     - TPOT mean / median
     - 5.28 / 5.19
     - 5.34 / 5.24

The two workloads are different prompt samples, so compare tools
within a row. vllm bench serve does not report E2EL.

Synthetic traces
----------------

A synthetic trace replayed against its source stays within 1.1 percent
on ISL, OSL, TTFT, TPOT and E2EL, with identical input and output
length distributions.

Reproduce
---------

Replay with ``--profile mooncake-trace --backend vllm-completions``,
audit with ``scripts/audit_live_replay.py``, compare with
``scripts/score_replay_records.py``. Commands are in :doc:`usage`.
