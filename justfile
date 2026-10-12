# Each recipe is one command line; the logic lives in quettabench/.

set dotenv-load := true
# "$@" passes recipe args on unsplit.
set positional-arguments

# Kernel table label: just gpu=<gpu> <recipe>
gpu := ""
# cuda | rocm | tpu | neuron: picks the device-visibility variable and the gpu-info tool.
backend := env_var_or_default("QBENCH_BACKEND", "cuda")
visible_devices_env := if backend == "cuda" { "CUDA_VISIBLE_DEVICES" } else if backend == "rocm" { "HIP_VISIBLE_DEVICES" } else if backend == "tpu" { "TPU_VISIBLE_CHIPS" } else if backend == "neuron" { "NEURON_VISIBLE_DEVICES" } else { error("QBENCH_BACKEND must be cuda, rocm, tpu or neuron") }
gpu_query_cmd := if backend == "cuda" { "nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv" } else if backend == "rocm" { "amd-smi list" } else if backend == "tpu" { "tpu-info" } else { "neuron-ls" }
gpu_id := env_var_or_default("GPU_ID", "0")
python_bin := env_var_or_default("PYTHON_BIN", "python3")
quettasim_root := env_var_or_default("QUETTASIM", "../QuettaSim")
remote_host := env_var_or_default("REMOTE_HOST", "")
gpu_ids := env_var_or_default("GPU_IDS", "")
nproc := env_var_or_default("NPROC", "1")
ncu_bin := env_var_or_default("NCU_BIN", "")
probe_results_dir := "results/probes"
# Head config for flash and gdn: just n_heads=64 n_kv_heads=8 flash <gpu> --ncu
head_config := "tp1"
n_heads := "32"
n_kv_heads := "8"
head_dim := "128"
layers := "32"
gdn_geometry := "9b"
value_heads := "32"
key_dim := "128"
value_dim := "128"
gdn_layers := "24"
grid_size := "full"

r2_endpoint := env_var_or_default("R2_ENDPOINT", "")
r2_bucket := env_var_or_default("R2_BUCKET", "")
# Backup of results/probes/ only, never QuettaSim's kernel_data.
r2_prefix := "profiling/qbench_probes"
r2_profile := env_var_or_default("AWS_PROFILE", "r2")

default:
    @just --list

# Set up this box for the run yamls (--dry-run prints the plan; writes .env)
provision *args:
    {{python_bin}} -m quettabench.provision {{args}}

# One run yaml to results/<run>/
run cfg *args:
    {{python_bin}} -m quettabench.cli run {{cfg}} {{args}}

# Resolved Config (with origins) and LaunchSpec; starts nothing
dry-run cfg *args:
    {{python_bin}} -m quettabench.cli run {{cfg}} --dry-run {{args}}

# Same as run; the matrix: block in the run yaml makes one results/<run>/ per point
sweep cfg *args:
    {{python_bin}} -m quettabench.cli run {{cfg}} {{args}}

# Attach to a results/<run>/ and append profile.json
profile *args:
    {{python_bin}} -m quettabench.cli profile {{args}}

# Compare results dirs at matched cells
compare *args:
    {{python_bin}} -m quettabench.cli compare {{args}}

test:
    python3 -m pytest tests -q

# CI hooks when pre-commit is installed, else ruff alone.
lint:
    if command -v pre-commit >/dev/null; then pre-commit run --all-files --hook-stage manual; else ruff check quettabench tests; fi

# Install the pre-commit and commit-msg hooks into .git
hooks:
    pre-commit install

# The GPUs the probes will target (local, or REMOTE_HOST).
gpu-info:
    #!/usr/bin/env bash
    if [ -n "${REMOTE_HOST:-}" ]; then
        ssh -o BatchMode=yes "$REMOTE_HOST" {{gpu_query_cmd}}
    else
        {{gpu_query_cmd}}
    fi

# Any probe by module name: just probe gemm "--gpu-label <gpu> --ncu"
probe name *args:
    {{visible_devices_env}}={{gpu_id}} {{python_bin}} -m quettabench.probes.{{name}} {{args}}

# Run a probe on REMOTE_HOST and pull the run back (NPROC>1: torchrun): just remote-probe gemm <gpu> --ncu
remote-probe name gpu=gpu *args:
    python3 -m quettabench.probes.base remote-probe --host "{{remote_host}}" --py "{{python_bin}}" --gpu-id "{{gpu_id}}" --gpu-ids "{{gpu_ids}}" --nproc "{{nproc}}" --ncu-bin "{{ncu_bin}}" -- "$@"

# qbench and vLLM in one image, qbench:<short sha>: just image [<vLLM image>] [build args]
image base="" *args:
    base="$1"; shift; docker build --network host ${base:+--build-arg "VLLM_IMAGE=$base"} ${HTTPS_PROXY:+--build-arg HTTPS_PROXY} ${HTTP_PROXY:+--build-arg HTTP_PROXY} ${NO_PROXY:+--build-arg NO_PROXY} --build-arg QBENCH_SHA=$(git rev-parse --short HEAD) "$@" -f docker/Dockerfile -t qbench:$(git rev-parse --short HEAD) .

# QuettaSim tables from results dirs: just export --results results/<run> --dest <dir>
export *args:
    {{python_bin}} -m quettabench.cli export {{args}}

# CUDA-event timing; --ncu needs a real ncu in NCU_BIN, --wide adds the wide shapes
gemm gpu=gpu *args:
    {{visible_devices_env}}={{gpu_id}} {{python_bin}} -m quettabench.probes.gemm --gpu-label {{gpu}} {{args}}

elementwise gpu=gpu:
    {{visible_devices_env}}={{gpu_id}} {{python_bin}} -m quettabench.probes.elementwise --gpu-label {{gpu}}

flash gpu=gpu *args:
    {{visible_devices_env}}={{gpu_id}} {{python_bin}} -m quettabench.probes.flash_attn --gpu-label {{gpu}} --head-config {{head_config}} --n-heads {{n_heads}} --n-kv-heads {{n_kv_heads}} --head-dim {{head_dim}} --layers {{layers}} {{args}}

moe gpu=gpu geometries="qwen3-30b-a3b,mixtral-8x7b,gpt-oss-20b,gpt-oss-120b" tp="1,2,4":
    {{visible_devices_env}}={{gpu_id}} {{python_bin}} -m quettabench.probes.moe --gpu-label {{gpu}} --geometries {{geometries}} --tp {{tp}}

# One MoE block's input, output and routing decision, for truncated-model verification
moe-routing model gpu=gpu layer="1" *args:
    {{visible_devices_env}}={{gpu_id}} {{python_bin}} -m quettabench.probes.moe --routing --gpu-label {{gpu}} --model {{model}} --layer {{layer}} {{args}}

gdn gpu=gpu *args:
    {{visible_devices_env}}={{gpu_id}} {{python_bin}} -m quettabench.probes.gdn --gpu-label {{gpu}} --gdn-geometry {{gdn_geometry}} --value-heads {{value_heads}} --key-dim {{key_dim}} --value-dim {{value_dim}} --gdn-layers {{gdn_layers}} --grid-size {{grid_size}} {{args}}

# NCCL grid; also prints tp_comm scalars as a YAML block.
collectives gpu=gpu world="2" hidden="4096" layers="32" ops="all_reduce,all_gather,reduce_scatter,all_to_all" gpu_ids="0,1":
    {{visible_devices_env}}={{gpu_ids}} {{python_bin}} -m torch.distributed.run --nproc_per_node={{world}} -m quettabench.probes.collective --gpu-label {{gpu}} --hidden {{hidden}} --layers {{layers}} --ops {{ops}}

# Chunked-prefill attention grid, one head config per run.
chunked-prefill-attn gpu=gpu n_heads="16" n_kv_heads="2" head_dim="128":
    {{visible_devices_env}}={{gpu_id}} {{python_bin}} -m quettabench.probes.chunked_prefill_attn --gpu-label {{gpu}} --n-heads {{n_heads}} --n-kv-heads {{n_kv_heads}} --head-dim {{head_dim}}

# TP all-reduce through vLLM's GroupCoordinator next to plain NCCL.
vllm-allreduce gpu=gpu world="2" gpu_ids="0,1":
    {{visible_devices_env}}={{gpu_ids}} {{python_bin}} -m torch.distributed.run --nproc_per_node={{world}} -m quettabench.probes.vllm_allreduce --gpu-label {{gpu}}

# Parallelism-switch cost (mode: tp | dp | ep | cross_node | plan); cross_node needs 2-node torchrun.
reparallel gpu=gpu world="8" mode="tp" gpu_ids="0,1,2,3,4,5,6,7":
    {{visible_devices_env}}={{gpu_ids}} {{python_bin}} -m torch.distributed.run --nproc_per_node={{world}} -m quettabench.probes.reparallel --gpu-label {{gpu}} --mode {{mode}}

_default_model_yaml := quettasim_root / "device_spec/models/qwen3.5-9b.yaml"

# Frontend section of profile.json from the run's live vLLM server
serving-profile run gpu=gpu model_yaml=_default_model_yaml tp="1" *args:
    {{python_bin}} -m quettabench.cli profile --run {{run}} --source frontend --frontend-args "--gpu-label {{gpu}} --device-yaml {{quettasim_root}}/device_spec/{{lowercase(gpu)}}.yaml --model-yaml {{model_yaml}} --tp {{tp}}" {{args}}

profile-gpu gpu=gpu:
    just gemm {{gpu}} --ncu
    just gpu={{gpu}} elementwise
    @echo ""
    @echo "GPU-generic tables done for {{gpu}}. Head-config-specific next:"
    @echo "  just n_heads=<Hq> n_kv_heads=<Hkv> head_dim=<head_dim> flash {{gpu}}"
    @echo "  just gdn {{gpu}}     # Qwen3.5 hybrids only"
    @echo "Then route the tables to QuettaSim: just export --only kernel_tables --results results/probes --kernel-dest <QuettaSim data root, e.g. ../QuettaSim/data>"

# Trajectory datasets from R2 into data/
fetch:
    R2_ENDPOINT={{r2_endpoint}} AWS_PROFILE={{r2_profile}} {{python_bin}} data/fetch.py

mirror-preview:
    aws --profile {{r2_profile}} --endpoint-url {{r2_endpoint}} s3 sync {{probe_results_dir}}/ s3://{{r2_bucket}}/{{r2_prefix}}/ --dryrun --exclude '*/__pycache__/*' --exclude '*.pyc'

mirror-up:
    aws --profile {{r2_profile}} --endpoint-url {{r2_endpoint}} s3 sync {{probe_results_dir}}/ s3://{{r2_bucket}}/{{r2_prefix}}/ --exclude '*/__pycache__/*' --exclude '*.pyc' --only-show-errors
    @echo "{{probe_results_dir}} -> s3://{{r2_bucket}}/{{r2_prefix}}/"

mirror-down:
    aws --profile {{r2_profile}} --endpoint-url {{r2_endpoint}} s3 sync s3://{{r2_bucket}}/{{r2_prefix}}/ {{probe_results_dir}}/ --exclude '*/__pycache__/*' --only-show-errors
    @echo "s3://{{r2_bucket}}/{{r2_prefix}}/ -> {{probe_results_dir}}"
