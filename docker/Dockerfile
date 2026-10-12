# qbench plus the vLLM it measures, so a fresh GPU box needs only docker.
# Build from the repo root: just image (passes QBENCH_SHA)
ARG VLLM_IMAGE=vllm/vllm-openai:v0.26.0
FROM ${VLLM_IMAGE}

# NIXL for the kv_transfer probe; vLLM's NixlConnector uses the same package. v0.26.0 ships 1.3.1.
RUN pip install --no-cache-dir nixl==1.3.1

WORKDIR /opt/qbench
COPY pyproject.toml requirements.txt README.md ./
COPY quettabench quettabench
COPY configs configs
COPY data data
# Editable, so configs/ stays next to the package as in a checkout; the package finds it by __file__.
RUN pip install --no-cache-dir -e .

# No .git in the image: results.bench_sha reads this instead.
ARG QBENCH_SHA=""
ENV QBENCH_SHA=$QBENCH_SHA

ENTRYPOINT ["qbench"]
