#!/usr/bin/env bash
# profiling/runbooks/preflight_a100_roofline_utils.sh
# Preflight for the A100 roofline-utils measurement (L6, audit-v2 G7).
# See profiling/docs/a100_roofline_utils_runbook.md. Run LOCALLY; it ssh-es to 'a100'.
#
# Hard gates (no override flags — the measurement is latency-sensitive and was
# explicitly deferred because of GPU contention):
#   1. host reachable
#   2. ZERO compute apps on the whole host (sibling-GPU HBM/PCIe/host contention moves
#      6-20 ms step walls; one free GPU is NOT sufficient)
#   3. vllm env present and importable
#   4. Llama-3.1-8B-Instruct weights present under /data/models (Rule #1: NO downloads)
#   5. >= 20 GiB free on /data for the run dir + traces
set -u
HOST=a100
ENV_PY=/home/kevinlau/miniconda3/envs/vllm/bin/python
MODELS_DIR=/data/models
FAIL=0
note() { printf '%s\n' "$*"; }
check() { # check <name> <0|1> <detail>
  if [ "$2" -eq 0 ]; then note "PASS  $1  $3"; else note "FAIL  $1  $3"; FAIL=1; fi
}

# 1. reachability
if out=$(ssh -o ConnectTimeout=10 "$HOST" hostname 2>&1); then
  check reachability 0 "$out"
else
  check reachability 1 "$out"
  note "aborting: host unreachable"; exit 1
fi

# 2. whole-host GPU quiet (the deferral condition)
apps=$(ssh "$HOST" "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader" 2>&1)
n_apps=$(printf '%s' "$apps" | grep -c . || true)
if [ "$n_apps" -eq 0 ]; then
  check gpu_quiet 0 "no compute apps on any GPU"
else
  check gpu_quiet 1 "$n_apps compute app(s) running — measurement stays DEFERRED:"
  printf '%s\n' "$apps" | sed 's/^/        /'
fi

# 3. env
vllm_ver=$(ssh "$HOST" "$ENV_PY -c 'import vllm; print(vllm.__version__)'" 2>&1)
if [ $? -eq 0 ]; then check vllm_env 0 "vllm $vllm_ver ($ENV_PY)"; else check vllm_env 1 "$vllm_ver"; fi

# 4. weights (Rule #1: report, never download)
model_path=$(ssh "$HOST" "ls -d $MODELS_DIR/*[Ll]lama-3.1-8[Bb]-Instruct* 2>/dev/null | head -1")
if [ -n "$model_path" ]; then
  sz=$(ssh "$HOST" "du -sh '$model_path' 2>/dev/null | cut -f1")
  check model_weights 0 "$model_path ($sz)"
else
  check model_weights 1 "no Llama-3.1-8B-Instruct under $MODELS_DIR — STOP, do not download (Rule #1); report instead"
fi

# 5. disk
free_gb=$(ssh "$HOST" "df -BG --output=avail /data 2>/dev/null | tail -1 | tr -dc 0-9")
if [ -n "${free_gb:-}" ] && [ "$free_gb" -ge 20 ]; then
  check disk 0 "${free_gb}G free on /data"
else
  check disk 1 "free space on /data: ${free_gb:-unknown}G (< 20G)"
fi

if [ "$FAIL" -eq 0 ]; then
  note "PREFLIGHT PASS — proceed with the runbook (fresh run dir, one GPU, leave clean)."
else
  note "PREFLIGHT FAIL — measurement stays deferred."
  exit 1
fi
