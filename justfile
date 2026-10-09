# QuettaBench recipes. Probe and export recipes only; run/profile/crossval land with the CLI.
#
# Probes write under results/probes/<gpu>/ (pass --out through `just probe` to change that).
# `just export` is the only route from results/ to QuettaSim.
#
#   REMOTE_HOST=a100  GPU_ID=0  PYTHON_BIN=~/miniconda3/envs/vllm/bin/python
#
#   just gpu-info
#   just gpu=A100 profile-gpu
#   just gpu=A100 flash tag=tp1 nh=32 nkv=8 hd=128
#   just probe gemm "--gpu-label A100 --ncu"
#   just export --dest ../QuettaSim/engine/data

set dotenv-load := true

gpu := "A100"
gpu_id := env_var_or_default("GPU_ID", "0")
py := env_var_or_default("PYTHON_BIN", "python3")
sim_root := env_var_or_default("QUETTASIM", "../QuettaSim")
kdata := "results/probes"

r2_endpoint := env_var_or_default("R2_ENDPOINT", "")
r2_bucket := env_var_or_default("R2_BUCKET", "")
r2_prefix := env_var_or_default("R2_KERNEL_PREFIX", "profiling/kernel_data")
r2_profile := env_var_or_default("AWS_PROFILE", "r2")

default:
    @just --list

# Show the GPU(s) the probes will target (local, or REMOTE_HOST if set).
gpu-info:
    #!/usr/bin/env bash
    if [ -n "${REMOTE_HOST:-}" ]; then
        ssh -o BatchMode=yes "$REMOTE_HOST" nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
    else
        nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
    fi

# Any probe by module name, its flags in one quoted argument: just probe gemm "--gpu-label A100 --ncu"
probe name *args:
    CUDA_VISIBLE_DEVICES={{gpu_id}} {{py}} -m quettabench.probes.{{name}} {{args}}

# Run a probe on REMOTE_HOST: ship the probes package, run it there (NPROC>1 = one rank per
# GPU in GPU_IDS through torchrun), pull the tables into results/probes/<gpu>/.
remote-probe name gpu=gpu *args:
    #!/usr/bin/env bash
    set -euo pipefail
    : "${REMOTE_HOST:?set REMOTE_HOST to the ssh host or alias}"
    rdir=/tmp/qbench-probes
    ssh -o BatchMode=yes "$REMOTE_HOST" "mkdir -p $rdir/out"
    rsync -a --relative -e 'ssh -o BatchMode=yes' quettabench/__init__.py quettabench/probes/ "$REMOTE_HOST:$rdir/"
    if [ "${NPROC:-1}" -gt 1 ]; then
        launch="CUDA_VISIBLE_DEVICES=${GPU_IDS:-{{gpu_id}}} {{py}} -m torch.distributed.run --nproc_per_node=$NPROC -m quettabench.probes.{{name}}"
    else
        launch="CUDA_VISIBLE_DEVICES={{gpu_id}} ${NCU_BIN:+NCU_BIN=$NCU_BIN} {{py}} -m quettabench.probes.{{name}}"
    fi
    ssh -o BatchMode=yes "$REMOTE_HOST" "cd $rdir && $launch --gpu-label {{gpu}} --out $rdir/out {{args}}"
    mkdir -p results/probes/{{gpu}}
    rsync -a -e 'ssh -o BatchMode=yes' "$REMOTE_HOST:$rdir/out/" results/probes/{{gpu}}/

# QuettaSim tables from results dirs: just export --results results/<run> --dest <dir>
export *args:
    {{py}} -m quettabench.export.quettasim {{args}}

# Storage is method-explicit: ncu/ = graphed-decode-faithful, cuda_event/ =
# eager-prefill-faithful. The QuettaSim loader reads decode+gemm+elementwise from
# ncu/, prefill from cuda_event/.

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

gdn gpu=gpu geom="9b" hv="32" dk="128" dv="128" nlin="24" mode="full":
    CUDA_VISIBLE_DEVICES={{gpu_id}} {{py}} -m quettabench.probes.gdn --gpu-label {{gpu}} --geometry-tag {{geom}} --hv {{hv}} --dk {{dk}} --dv {{dv}} --n-linear {{nlin}} --mode {{mode}}

gdn-ncu gpu=gpu geom="9b" hv="32" dk="128" dv="128" nlin="24":
    CUDA_VISIBLE_DEVICES={{gpu_id}} {{py}} -m quettabench.probes.gdn --gpu-label {{gpu}} --geometry-tag {{geom}} --hv {{hv}} --dk {{dk}} --dv {{dv}} --n-linear {{nlin}} --ncu

# NCCL grid -> results/probes/<gpu>/ncu/collectives/. Also prints a legacy tp_comm: YAML block.
collectives gpu=gpu world="2" hidden="4096" layers="32" ops="all_reduce,all_gather,reduce_scatter,all_to_all" gpu_ids="0,1":
    CUDA_VISIBLE_DEVICES={{gpu_ids}} {{py}} -m torch.distributed.run --nproc_per_node={{world}} -m quettabench.probes.collective --gpu-label {{gpu}} --hidden {{hidden}} --layers {{layers}} --ops {{ops}}

# Alias: all-reduce microbench used to be a separate probe file.
tp-comm gpu=gpu tp="2" hidden="4096" layers="32" gpu_ids="0,1":
    just collectives gpu={{gpu}} world={{tp}} hidden={{hidden}} layers={{layers}} gpu_ids={{gpu_ids}}

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

# Live vLLM server -> frontend: + serving: YAML (no GT). GT-fit stays in QuettaSim:
#   just -f $QUETTASIM/justfile serving-fit gpu=X model=...
serving-profile gpu=gpu model="Qwen3.5-9B" model_yaml=_default_model_yaml base_url="http://localhost:8000/v1" tp="1":
    {{py}} profiling/kernel_composed/probes/serving_frontend_probe.py --gpu-label {{gpu}} --model {{model}} \
        --model-yaml {{model_yaml}} --base-url {{base_url}} --tp {{tp}}

profile-gpu gpu=gpu:
    just gpu={{gpu}} gemm-ncu
    just gpu={{gpu}} elementwise
    @echo ""
    @echo "GPU-generic tables done for {{gpu}}. Head-config-specific next:"
    @echo "  just gpu={{gpu}} flash tag=tp1 nh=<Hq> nkv=<Hkv> hd=<head_dim>"
    @echo "  just gpu={{gpu}} gdn  geom=9b     # Qwen3.5 hybrids only"
    @echo "Then route the tables to QuettaSim: just export --dest <dir>"

mirror-preview:
    aws --profile {{r2_profile}} --endpoint-url {{r2_endpoint}} s3 sync {{kdata}}/ s3://{{r2_bucket}}/{{r2_prefix}}/ --dryrun --exclude '*/__pycache__/*' --exclude '*.pyc'

mirror-up:
    aws --profile {{r2_profile}} --endpoint-url {{r2_endpoint}} s3 sync {{kdata}}/ s3://{{r2_bucket}}/{{r2_prefix}}/ --exclude '*/__pycache__/*' --exclude '*.pyc' --only-show-errors
    @echo "{{kdata}} -> s3://{{r2_bucket}}/{{r2_prefix}}/"

mirror-up-prune:
    aws --profile {{r2_profile}} --endpoint-url {{r2_endpoint}} s3 sync {{kdata}}/ s3://{{r2_bucket}}/{{r2_prefix}}/ --delete --exclude '*/__pycache__/*' --exclude '*.pyc' --only-show-errors
    @echo "{{kdata}} -> s3://{{r2_bucket}}/{{r2_prefix}}/ (pruned)"

mirror-down:
    aws --profile {{r2_profile}} --endpoint-url {{r2_endpoint}} s3 sync s3://{{r2_bucket}}/{{r2_prefix}}/ {{kdata}}/ --exclude '*/__pycache__/*' --only-show-errors
    @echo "s3://{{r2_bucket}}/{{r2_prefix}}/ -> {{kdata}}"
