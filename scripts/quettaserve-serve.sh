#!/usr/bin/env bash
# QuettaServe's OpenAI-compatible server: api, then one dist secondary per GPU, all in this process group, so the
# launcher's signal to the group stops every one of them. Exits with the status of the first of them to exit.
#
#   scripts/quettaserve-serve.sh --bin-dir DIR --model DIR --tp N --pp N --http-port PORT
#       [--kv-memory-fraction F] [--stagger SECONDS] [api flag ...]
#
# CUDA_VISIBLE_DEVICES names at least tp*pp GPUs. Secondary g runs on GPU g and the primary shares GPU 0. The primary
# listens for secondaries on PORT+1, secondary g on PORT+2+g. Secondaries start SECONDS apart (default 3) once the
# primary listens, in GPU order: ranks follow connection order, so rank i is GPU i. --kv-memory-fraction goes to every
# secondary; every other flag (--num-cache-slots, --prefix-cache, --served-model-name, ...) goes to api.
set -u

die() {
  echo "quettaserve-serve: $*" >&2
  exit 2
}

bin= model= tp= pp= http= kvf= stagger=3
api_args=()
while (($#)); do
  case $1 in
  --bin-dir | --model | --tp | --pp | --http-port | --kv-memory-fraction | --stagger)
    (($# >= 2)) || die "$1 needs a value"
    case $1 in
    --bin-dir) bin=$2 ;;
    --model) model=$2 ;;
    --tp) tp=$2 ;;
    --pp) pp=$2 ;;
    --http-port) http=$2 ;;
    --kv-memory-fraction) kvf=$2 ;;
    --stagger) stagger=$2 ;;
    esac
    shift 2
    ;;
  *)
    api_args+=("$1")
    shift
    ;;
  esac
done
[[ -n $bin && -n $model && -n $tp && -n $pp && -n $http ]] || die "--bin-dir, --model, --tp, --pp and --http-port are required"
[[ -x $bin/api && -x $bin/dist ]] || die "no executable api and dist in $bin"
[[ $tp =~ ^[1-9][0-9]*$ && $pp =~ ^[1-9][0-9]*$ && $http =~ ^[1-9][0-9]*$ ]] || die "--tp, --pp and --http-port must be positive integers"
n=$((tp * pp))
IFS=, read -ra gpus <<<"${CUDA_VISIBLE_DEVICES:-}"
((${#gpus[@]} >= n)) || die "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} names fewer than tp*pp=$n GPUs"
peer=$((http + 1))
command -v ss >/dev/null || die "needs ss (iproute2) to see the primary listen"

listening() {
  [[ -n $(ss -Hltn "sport = :$1") ]]
}

for ((p = http; p <= peer + n; p++)); do
  ! listening "$p" || die "port $p is already in use"
done

pids=()
stop() {
  kill "${pids[@]}" 2>/dev/null
  wait
}
trap 'stop; exit 143' TERM INT

"$bin/api" --gpu 0 --model "$model" --tokenizer "$model/tokenizer.json" --port "$peer" --num-nodes $((n + 1)) \
  --tensor-parallel "$tp" --http-addr "127.0.0.1:$http" "${api_args[@]}" &
pids+=($!)
until listening "$peer"; do
  if ! kill -0 "${pids[0]}" 2>/dev/null; then
    wait "${pids[0]}"
    exit
  fi
  sleep 0.5
done
for ((g = 0; g < n; g++)); do
  # dist's --model is a required global flag its subcommand does not see when given first; MODEL is.
  MODEL=$model "$bin/dist" --gpu "$g" secondary --primary "127.0.0.1:$peer" --port $((peer + 1 + g)) \
    ${kvf:+--kv-memory-fraction "$kvf"} &
  pids+=($!)
  sleep "$stagger"
done
wait -n
status=$?
stop
exit "$status"
