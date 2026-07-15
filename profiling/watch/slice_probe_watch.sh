# profiling/watch/slice_probe_watch.sh
# Waits for 2 truly free GPUs (mem<1GiB, util<5%, 3 consecutive 2-min samples),
# then runs gemm_slice_probe (1 GPU) + custom_allreduce_probe (2 GPUs) and exits.
# Run on the h100 box:  nohup bash slice_probe_watch.sh > /data48/kevinlau/herd_probe/slice_watch.log 2>&1 &

PY=$HOME/miniconda3/envs/vllm/bin/python
DIR=$(cd "$(dirname "$0")" && pwd)
OUT=/data48/kevinlau/herd_probe
mkdir -p "$OUT"

# root fs is 100% full (2026-07-04) — keep every write off it
export TMPDIR=/data48/kevinlau/tmp
export XDG_CACHE_HOME=/data48/kevinlau/tmp/cache
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"

declare -A streak
while true; do
  free=()
  while IFS=', ' read -r idx mem util; do
    if [ "$mem" -lt 1024 ] && [ "$util" -lt 5 ]; then
      streak[$idx]=$(( ${streak[$idx]:-0} + 1 ))
    else
      streak[$idx]=0
    fi
    [ "${streak[$idx]}" -ge 3 ] && free+=("$idx")
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)

  if [ "${#free[@]}" -ge 2 ]; then
    g1=${free[0]}; g2=${free[1]}
    echo "$(date '+%F %T') firing on GPUs $g1,$g2"
    CUDA_VISIBLE_DEVICES=$g1 "$PY" "$DIR/gemm_slice_probe.py" \
      --out "$OUT/gemm_slices_H100.csv" && echo "gemm probe OK" || echo "gemm probe FAILED"
    CUDA_VISIBLE_DEVICES=$g1,$g2 "$PY" "$DIR/custom_allreduce_probe.py" \
      --out "$OUT/custom_allreduce_H100_tp2.json" && echo "ar probe OK" || echo "ar probe FAILED"
    echo "$(date '+%F %T') done"
    exit 0
  fi
  echo "$(date '+%F %T') waiting (streaks: $(for k in "${!streak[@]}"; do printf '%s:%s ' "$k" "${streak[$k]}"; done))"
  sleep 120
done
