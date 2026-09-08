QuettaBench
===========

QuettaBench is a benchmark for LLM serving engines. It drives any
OpenAI compatible server with realistic workloads and reports per
request latency: time to first token (TTFT), time per output token
(TPOT), inter token latency (ITL) and end to end latency (E2EL),
each with mean, p50, p90 and p99.

What it does
------------

* Replays workload profiles built from real agent data: SWE-Bench
  planning calls, terminal agent sessions, computer use sessions and
  ShareGPT chat, single turn and multi turn.
* Replays the public Mooncake production trace with exact prefix
  sharing: hash ids expand to deterministic token blocks, so prefixes
  shared in the trace stay shared byte for byte on the server.
* Generates synthetic traces that keep a source trace's length
  distributions and prefix sharing structure, then gates them against
  the real replay with acceptance scripts.
* Drives load closed loop (a fixed population of in flight requests)
  or open loop (timed arrivals at a target rate, so queues can grow
  and saturation becomes visible).
* Audits delivery against the server's own token accounting: a replay
  passes only when the server reports exactly the trace's input and
  output lengths for every request.

How a run works
---------------

A profile from :mod:`src.workloads.profiles` selects a dataset from
:mod:`src.workloads.dataset`, which yields one request per record. An
arrival schedule from :mod:`src.workloads.arrival` decides when each
request dispatches. The runner, :mod:`src.benchmark.runner`, sends
them through a backend from :mod:`src.engines` and aggregates per
request results with :mod:`src.benchmark.metrics` into a results JSON.

Quickstart
----------

.. code-block:: bash

   pip install -r requirements.txt

   # server side, for example vLLM with prefix caching
   vllm serve meta-llama/Llama-3.1-8B --enable-prefix-caching

   # client side: 100 chat requests, 10 in flight
   python -m src.benchmark.runner \
       --url http://localhost:8000/v1/chat/completions \
       --model meta-llama/Llama-3.1-8B \
       --profile chat-singleturn \
       --num-requests 100 --concurrency 10 \
       --output results/run.json

See :doc:`usage` for trace replay and the synthetic trace workflow,
and :doc:`architecture` for a module map in the order a request flows
through them.

This site is rebuilt from source docstrings on every push to main.
Keep docstrings in Google style and new modules appear here
automatically.

.. toctree::
   :maxdepth: 1
   :caption: Guide

   architecture
   usage
   validation

.. toctree::
   :maxdepth: 3
   :caption: API reference

   api/src

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
