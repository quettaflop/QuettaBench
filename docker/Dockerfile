# qbench plus the vLLM it measures, so a fresh GPU box needs only docker.
# Build from the repo root: just image (passes QBENCH_SHA)
ARG VLLM_IMAGE=vllm/vllm-openai:v0.26.0
FROM ${VLLM_IMAGE}

# NIXL for the kv_transfer probe (cuda only); vLLM's NixlConnector uses the same package. Empty NIXL skips it on rocm, tpu and neuron bases.
ARG NIXL=1.3.1
RUN if [ -n "$NIXL" ]; then pip install --no-cache-dir nixl==$NIXL; fi

WORKDIR /opt/qbench
COPY pyproject.toml README.md ./
COPY quettabench quettabench
COPY configs configs
COPY data data
# Editable, so configs/ stays next to the package as in a checkout; the package finds it by __file__.
RUN pip install --no-cache-dir -e .

# No .git in the image: results.bench_sha reads this instead.
ARG QBENCH_SHA=""
ENV QBENCH_SHA=$QBENCH_SHA

ENTRYPOINT ["qbench"]
