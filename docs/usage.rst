Usage
=====

Install
-------

.. code-block:: bash

   pip install -r requirements.txt

Datasets referenced by profiles live outside git; see
``scripts/fetch_data.sh``.

Standard profile run
--------------------

Start any OpenAI compatible server with prefix caching enabled, then:

.. code-block:: bash

   python -m src.benchmark.runner \
       --url http://localhost:8000/v1/chat/completions \
       --model meta-llama/Llama-3.1-8B \
       --profile chat-singleturn \
       --num-requests 100 --concurrency 10 \
       --output results/run.json

``--list-profiles`` prints every available profile. Closed loop is
the default: ``--concurrency`` requests stay in flight, so the server
is never offered more load than it can absorb. Open loop dispatches
on a clock instead, which exposes saturation:

.. code-block:: bash

   python -m src.benchmark.runner ... \
       --open-loop --arrival poisson --target-rate 2.0

Mooncake trace replay
---------------------

The mooncake-trace profile sends token id prompts, which only the
vllm-completions backend accepts:

.. code-block:: bash

   python -m src.benchmark.runner \
       --url http://localhost:8000/v1/completions \
       --model meta-llama/Llama-3.1-8B \
       --backend vllm-completions \
       --profile mooncake-trace \
       --workload-file data/mooncake/conversation_trace.jsonl \
       --num-requests 1000 \
       --exact-output-length \
       --output results/replay.json

``--exact-output-length`` pins each response to the trace's output
length, which the delivery audit requires.

Synthetic trace workflow
------------------------

.. code-block:: bash

   # sample a synthetic trace like the real one
   python scripts/synthesize_mooncake_trace.py \
       --trace data/mooncake/conversation_trace.jsonl \
       --out data/mooncake/synthetic.jsonl --seed 7

   # audit a replay: server reported token counts must equal the trace
   python scripts/audit_live_replay.py \
       --result results/replay.json \
       --trace data/mooncake/conversation_trace.jsonl

   # gate two replays against each other
   python scripts/score_replay_records.py \
       --a results/replay.json --b results/synthetic_replay.json \
       --trace-a data/mooncake/conversation_trace.jsonl \
       --trace-b data/mooncake/synthetic.jsonl

Results
-------

The output JSON holds the run configuration, aggregate latency
statistics and a per_request list with one record per request:
success, TTFT, TPOT and E2EL in milliseconds, the server reported
token counts and dispatch timing.

Troubleshooting
---------------

``--profile mooncake-trace sends token id prompts; use --backend vllm-completions``
    The mooncake profile delivers prompts as token ids, which the chat
    backends cannot send. Add ``--backend vllm-completions`` and point
    ``--url`` at ``/v1/completions``.

Audit fails with requests lacking server usage
    The audit trusts only the server's own token accounting, reported
    on the final streaming chunk. The vllm-completions backend always
    requests it; a server that never reports usage cannot pass.

HTTP 400 from /v1/chat/completions on a base model
    Base models without a chat template reject chat requests. Use an
    instruct model, or the vllm-completions backend with raw text
    prompts.

Missing data files
    Datasets referenced by profiles live outside git. Fetch them with
    ``scripts/fetch_data.sh``.

Profile requires prefix caching
    Profiles marked prefix caching required expect the server launched
    with ``--enable-prefix-caching``.
