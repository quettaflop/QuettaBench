# qbench plus the vLLM it measures, so a fresh GPU box needs only docker.
# Build from the repo root: docker build -f docker/Dockerfile -t qbench:$(git rev-parse --short HEAD) .
ARG VLLM_IMAGE=vllm/vllm-openai:v0.26.0
FROM ${VLLM_IMAGE}

# NIXL for the kv_transfer probe; vLLM's NixlConnector uses the same package.
RUN pip install --no-cache-dir nixl

WORKDIR /opt/qbench
COPY pyproject.toml requirements.txt README.md ./
COPY quettabench quettabench
RUN pip install --no-cache-dir .
COPY configs configs
COPY data data

ENTRYPOINT ["qbench"]
