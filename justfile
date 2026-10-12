# QuettaBench recipes: each is one command line; the logic lives in quettabench/.
#
#   just provision configs/runs/llama.yaml --set hardware.provision.py=/venv/bin/python
#   just dry configs/runs/llama.yaml --set engine.model=/weights/llama
#   just run configs/runs/llama.yaml --set engine.model=/weights/llama
#   just profile --run results/llama --source metrics
#   just xval results/<ours> results/<theirs>
#   just export --results results/<run> --dest <export dir>
#   just image
#
# .env (written by `just provision`) sets PYTHON_BIN and the cache dirs. Each probe run writes results/probes/<gpu>/<probe>/<run-id>/ (pass --out through `just probe` to change the parent).
# `just export` is the only route from results/ to QuettaSim; the probes' kernel tables go by its kernel_tables builder into <QuettaSim data root>/kernel_data/ (--kernel-dest).
#
#   REMOTE_HOST=<host>  GPU_ID=0  PYTHON_BIN=~/miniconda3/envs/vllm/bin/python
#
#   just gpu-info
#   just gpu=<gpu> profile-gpu
#   just gpu=<gpu> flash tag=tp1 nh=32 nkv=8 hd=128
#   just probe gemm "--gpu-label <gpu> --ncu"
#   just export --only kernel_tables --results results/probes --kernel-dest ../QuettaSim/data

set dotenv-load := true
# Recipe args reach the shell as $1, $2, ... too, so "$@" passes them on unsplit.
set positional-arguments

# The label the kernel tables are keyed by; name it per run: just gpu=<gpu> <recipe>
gpu := ""
gpu_id := env_var_or_default("GPU_ID", "0")
py := env_var_or_default("PYTHON_BIN", "python3")
sim_root := env_var_or_default("QUETTASIM", "../QuettaSim")
remote_host := env_var_or_default("REMOTE_HOST", "")
gpu_ids := env_var_or_default("GPU_IDS", "")
nproc := env_var_or_default("NPROC", "1")
ncu_bin := env_var_or_default("NCU_BIN", "")
kdata := "results/probes"

r2_endpoint := env_var_or_default("R2_ENDPOINT", "https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com")
r2_bucket := env_var_or_default("R2_BUCKET", "")
# The probes' own prefix: a backup of results/probes/, never QuettaSim's profiling/kernel_data/.
r2_prefix := "profiling/qbench_probes"
r2_profile := env_var_or_default("AWS_PROFILE", "r2")

default:
    @just --list

# Check and set up this box for the given run yamls (--dry-run prints the plan; writes .env)
provision *args:
    {{py}} -m quettabench.provision {{args}}

# One run yaml to results/<run>/
run cfg *args:
    {{py}} -m quettabench.cli run {{cfg}} {{args}}

# The run's resolved Config (with origins) and LaunchSpec; starts nothing
dry cfg *args:
    {{py}} -m quettabench.cli run {{cfg}} --dry-run {{args}}

# Same as run: a sweep is the matrix: block in the run yaml, one results/<run>/ per point
sweep cfg *args:
    {{py}} -m quettabench.cli run {{cfg}} {{args}}

# Attach to a finished or running results/<run>/ and append profile.json
profile *args:
    {{py}} -m quettabench.cli profile {{args}}

# Compare results dirs (QuettaServe vs vLLM at matched cells)
xval *args:
    {{py}} -m quettabench.cli xval {{args}}

test:
    python3 -m pytest tests -q

# The CI hooks when pre-commit is installed, else ruff alone. mypy is not part of the gate yet.
lint:
    if command -v pre-commit >/dev/null; then pre-commit run --all-files --hook-stage manual; else ruff check quettabench tests; fi

# Install the pre-commit and commit-msg hooks into .git
hooks:
    pre-commit install

# Show the GPU(s) the probes will target (local, or REMOTE_HOST if set).
gpu-info:
    #!/usr/bin/env bash
    if [ -n "${REMOTE_HOST:-}" ]; then
        ssh -o BatchMode=yes "$REMOTE_HOST" nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
    else
        nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
    fi

# Any probe by module name, its flags in one quoted argument: just probe gemm "--gpu-label <gpu> --ncu"
probe name *args:
    CUDA_VISIBLE_DEVICES={{gpu_id}} {{py}} -m quettabench.probes.{{name}} {{args}}

# Run a probe on REMOTE_HOST in its own mktemp dir (probes/base.py remote_probe): ship the probes package and what its run.json needs, run it (NPROC>1 = one rank per GPU in GPU_IDS through torchrun) with --run-id from this machine's clock, always pull the run into results/probes/<gpu>/<probe>/<run-id>/, then remove that remote dir. The probe's flags follow the GPU as they are: just remote-probe gemm <gpu> --ncu --model-shapes llama8b
remote-probe name gpu=gpu *args:
    python3 -m quettabench.probes.base remote-probe --host "{{remote_host}}" --py "{{py}}" --gpu-id "{{gpu_id}}" --gpu-ids "{{gpu_ids}}" --nproc "{{nproc}}" --ncu-bin "{{ncu_bin}}" -- "$@"

# qbench and vLLM in one image, qbench:<short sha>: just image [<platform vLLM image, see README>] [docker build args]
image base="" *args:
    base="$1"; shift; docker build --network host ${base:+--build-arg "VLLM_IMAGE=$base"} ${HTTPS_PROXY:+--build-arg HTTPS_PROXY} ${HTTP_PROXY:+--build-arg HTTP_PROXY} ${NO_PROXY:+--build-arg NO_PROXY} --build-arg QBENCH_SHA=$(git rev-parse --short HEAD) "$@" -f docker/Dockerfile -t qbench:$(git rev-parse --short HEAD) .

# QuettaSim tables from results dirs: just export --results results/<run> --dest <dir> [--kernel-dest <QuettaSim data root>]
export *args:
    {{py}} -m quettabench.cli export {{args}}

# Each probe run writes its rows to rows.jsonl; a row's timing (ncu = graphed-decode-faithful,
# cuda_event = eager-prefill-faithful) picks its QuettaSim dir at export. The QuettaSim loader
# reads decode+gemm+elementwise from ncu, prefill from cuda_event.

gemm-cuda-event gpu=gpu:
    CUDA_VISIBLE_DEVICES={{gpu_id}} {{py}} -m quettabench.probes.gemm --gpu-label {{gpu}}

gemm-cuda-event-wide gpu=gpu:
    CUDA_VISIBLE_DEVICES={{gpu_id}} {{py}} -m quettabench.probes.gemm --gpu-label {{gpu}} --wide

# NCU GEMM (the STANDARD). Needs a real ncu: NCU_BIN=/opt/nvidia/nsight-compute/.../ncu
gemm-ncu gpu=gpu:
    CUDA_VISIBLE_DEVICES={{gpu_id}} {{py}} -m quettabench.probes.gemm --gpu-label {{gpu}} --ncu

gemm-ncu-wide gpu=gpu:
    CUDA_VISIBLE_DEVICES={{gpu_id}} {{py}} -m quettabench.probes.gemm --gpu-label {{gpu}} --ncu --wide

elementwise gpu=gpu:
    CUDA_VISIBLE_DEVICES={{gpu_id}} {{py}} -m quettabench.probes.elementwise --gpu-label {{gpu}}

flash gpu=gpu tag="tp1" nh="32" nkv="8" hd="128" layers="32":
    CUDA_VISIBLE_DEVICES={{gpu_id}} {{py}} -m quettabench.probes.flash_attn --gpu-label {{gpu}} --tag {{tag}} --n-heads {{nh}} --n-kv-heads {{nkv}} --head-dim {{hd}} --layers {{layers}}

flash-ncu gpu=gpu tag="tp1" nh="32" nkv="8" hd="128" layers="32":
    CUDA_VISIBLE_DEVICES={{gpu_id}} {{py}} -m quettabench.probes.flash_attn --gpu-label {{gpu}} --tag {{tag}} --n-heads {{nh}} --n-kv-heads {{nkv}} --head-dim {{hd}} --layers {{layers}} --ncu

moe gpu=gpu geoms="qwen3-30b-a3b,mixtral-8x7b,gpt-oss-20b,gpt-oss-120b" tp="1,2,4":
    CUDA_VISIBLE_DEVICES={{gpu_id}} {{py}} -m quettabench.probes.moe --gpu-label {{gpu}} --geometries {{geoms}} --tp {{tp}}

# One MoE block's input, output and routing decision -> results/probes/<gpu>/moe_routing/<run-id>/ (slice verification)
moe-routing model gpu=gpu layer="1" *args:
    CUDA_VISIBLE_DEVICES={{gpu_id}} {{py}} -m quettabench.probes.moe --routing --gpu-label {{gpu}} --model {{model}} --layer {{layer}} {{args}}

gdn gpu=gpu geom="9b" hv="32" dk="128" dv="128" nlin="24" mode="full":
    CUDA_VISIBLE_DEVICES={{gpu_id}} {{py}} -m quettabench.probes.gdn --gpu-label {{gpu}} --geometry-tag {{geom}} --hv {{hv}} --dk {{dk}} --dv {{dv}} --n-linear {{nlin}} --mode {{mode}}

gdn-ncu gpu=gpu geom="9b" hv="32" dk="128" dv="128" nlin="24":
    CUDA_VISIBLE_DEVICES={{gpu_id}} {{py}} -m quettabench.probes.gdn --gpu-label {{gpu}} --geometry-tag {{geom}} --hv {{hv}} --dk {{dk}} --dv {{dv}} --n-linear {{nlin}} --ncu

# NCCL grid -> results/probes/<gpu>/collective/<run-id>/rows.jsonl. Also prints a legacy tp_comm: YAML block.
collectives gpu=gpu world="2" hidden="4096" layers="32" ops="all_reduce,all_gather,reduce_scatter,all_to_all" gpu_ids="0,1":
    CUDA_VISIBLE_DEVICES={{gpu_ids}} {{py}} -m torch.distributed.run --nproc_per_node={{world}} -m quettabench.probes.collective --gpu-label {{gpu}} --hidden {{hidden}} --layers {{layers}} --ops {{ops}}

# Chunked-prefill (q over resident) attention grid, one head config per run.
cross-attn gpu=gpu nh="16" nkv="2" hd="128":
    CUDA_VISIBLE_DEVICES={{gpu_id}} {{py}} -m quettabench.probes.cross_attn --gpu-label {{gpu}} --n-heads {{nh}} --n-kv-heads {{nkv}} --head-dim {{hd}}

# TP all-reduce through vLLM's OWN GroupCoordinator (custom allreduce on), next to plain NCCL.
vllm-allreduce gpu=gpu world="2" gpu_ids="0,1":
    CUDA_VISIBLE_DEVICES={{gpu_ids}} {{py}} -m torch.distributed.run --nproc_per_node={{world}} -m quettabench.probes.vllm_allreduce --gpu-label {{gpu}}

# In-place parallelism-switch cost (mode: tp | dp | ep | xnode | plan). Cross-node
# physics (xnode) needs a 2-node torchrun -- see the probe header.
reparallel gpu=gpu world="8" mode="tp" gpu_ids="0,1,2,3,4,5,6,7":
    CUDA_VISIBLE_DEVICES={{gpu_ids}} {{py}} -m torch.distributed.run --nproc_per_node={{world}} -m quettabench.probes.reparallel --gpu-label {{gpu}} --mode {{mode}}

_default_model_yaml := sim_root / "device_spec/models/qwen3.5-9b.yaml"

# The live vLLM server of a results/<run>/ -> its profile.json frontend section (no GT),
# through qbench profile. GT-fit stays in QuettaSim:
#   just -f $QUETTASIM/justfile serving-fit gpu=X model=...
serving-profile run gpu=gpu model_yaml=_default_model_yaml tp="1" *args:
    {{py}} -m quettabench.cli profile --run {{run}} --source frontend --frontend-args "--gpu-label {{gpu}} --device-yaml {{sim_root}}/device_spec/{{lowercase(gpu)}}.yaml --model-yaml {{model_yaml}} --tp {{tp}}" {{args}}

profile-gpu gpu=gpu:
    just gpu={{gpu}} gemm-ncu
    just gpu={{gpu}} elementwise
    @echo ""
    @echo "GPU-generic tables done for {{gpu}}. Head-config-specific next:"
    @echo "  just gpu={{gpu}} flash tag=tp1 nh=<Hq> nkv=<Hkv> hd=<head_dim>"
    @echo "  just gpu={{gpu}} gdn  geom=9b     # Qwen3.5 hybrids only"
    @echo "Then route the tables to QuettaSim: just export --only kernel_tables --results results/probes --kernel-dest <QuettaSim data root, e.g. ../QuettaSim/data>"

# The trajectory datasets from R2 into data/ (gitignored)
fetch:
    R2_ENDPOINT={{r2_endpoint}} AWS_PROFILE={{r2_profile}} {{py}} data/fetch.py

mirror-preview:
    aws --profile {{r2_profile}} --endpoint-url {{r2_endpoint}} s3 sync {{kdata}}/ s3://{{r2_bucket}}/{{r2_prefix}}/ --dryrun --exclude '*/__pycache__/*' --exclude '*.pyc'

mirror-up:
    aws --profile {{r2_profile}} --endpoint-url {{r2_endpoint}} s3 sync {{kdata}}/ s3://{{r2_bucket}}/{{r2_prefix}}/ --exclude '*/__pycache__/*' --exclude '*.pyc' --only-show-errors
    @echo "{{kdata}} -> s3://{{r2_bucket}}/{{r2_prefix}}/"

mirror-down:
    aws --profile {{r2_profile}} --endpoint-url {{r2_endpoint}} s3 sync s3://{{r2_bucket}}/{{r2_prefix}}/ {{kdata}}/ --exclude '*/__pycache__/*' --only-show-errors
    @echo "s3://{{r2_bucket}}/{{r2_prefix}}/ -> {{kdata}}"
