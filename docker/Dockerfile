# qbench plus the vLLM it measures.
# Build from the repo root: just image
ARG VLLM_IMAGE=vllm/vllm-openai:v0.26.0
FROM ${VLLM_IMAGE}

# NIXL for the kv_transfer probe (cuda only); empty skips it on rocm, tpu and neuron.
ARG NIXL=1.3.1
RUN if [ -n "$NIXL" ]; then pip install --no-cache-dir nixl==$NIXL; fi

WORKDIR /opt/qbench
COPY pyproject.toml README.md ./
COPY quettabench quettabench
COPY configs configs
COPY data data
# Editable, so configs/ stays next to the package.
RUN pip install --no-cache-dir -e .

# No .git in the image: results.bench_sha reads this.
ARG QBENCH_SHA=""
ENV QBENCH_SHA=$QBENCH_SHA

ENTRYPOINT ["qbench"]
