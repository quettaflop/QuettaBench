#!/usr/bin/env bash
# Sweep the qwen3 grid through the engine's decode_loop test into one log.
# Run from a QuettaServe checkout (QS_DIR overrides); QW_BENCH_BIN points at a
# prebuilt binary when cargo is absent.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${QW_CKPT:?set QW_CKPT (path to the Qwen3 checkpoint)}"
QS="${QS_DIR:-.}"
if [ -z "${QW_BENCH_BIN:-}" ] && [ ! -f "$QS/qwen/Cargo.toml" ]; then
  echo "no qwen crate under '$QS'; run from the QuettaServe root, set QS_DIR, or set QW_BENCH_BIN" >&2
  exit 1
fi
OUT="${1:-qwen3-bench.log}"

# Link profile picks the collective env, same rule as ds_bench.sh: PCIe boxes
# get the tuned pins, NVLink boxes get none. Override with XVAL_LINK_PROFILE.
if [ -z "${XVAL_LINK_PROFILE:-}" ]; then
  # Capture then glob-match. Piping into `grep -q` makes grep close the pipe on
  # first match; nvidia-smi then takes SIGPIPE (141) and pipefail reads that as
  # a failure, so every box misdetects as pcie. No pipe, no SIGPIPE.
  _topo="$(nvidia-smi topo -m 2>/dev/null || true)"
  case "$_topo" in
    *NV[0-9]*) XVAL_LINK_PROFILE=nvlink ;;
    *)         XVAL_LINK_PROFILE=pcie ;;
  esac
fi
export XVAL_LINK_PROFILE

# Read run params from xval.yaml; falls back to hardcoded defaults when absent.
read -r _STEPS _HEADROOM < <(python3 -c "
import sys; sys.path.insert(0,'$HERE')
from xval_config import run_params
p = run_params()
print(p['timing_steps'], p['max_seq_headroom'])
")
STEPS="${QW_TIMING_STEPS:-$_STEPS}"
_MAX_SEQ_HEADROOM="${QW_MAX_SEQ_HEADROOM:-$_HEADROOM}"

# Cells and tp come from the qwen3 workload in xval.yaml / workloads.json.
CELLS="$(python3 -c "
import sys; sys.path.insert(0,'$HERE')
from xval_config import workloads, grids
wl = workloads()['qwen3']
for ctx, bs in grids()[wl['grid']]:
    print(ctx, bs)
")"
WORLD="${QW_WORLD:-$(python3 -c "
import sys; sys.path.insert(0,'$HERE')
from xval_config import workloads
print(workloads()['qwen3'].get('tp', 1))
")}"

# Collective env from xval.yaml, resolved through the link profile; empty
# values mean "leave unset" and are not exported (see ds_bench.sh).
while IFS='=' read -r _k _v; do
  case "$_k" in
    NCCL_ALGO)             _YAML_NCCL_ALGO="$_v" ;;
    NCCL_PROTO)            _YAML_NCCL_PROTO="$_v" ;;
    NCCL_P2P_LEVEL)        _YAML_NCCL_P2P_LEVEL="$_v" ;;
    NCCL_IB_DISABLE)       _YAML_NCCL_IB_DISABLE="$_v" ;;
    NCCL_SOCKET_IFNAME)    _YAML_NCCL_SOCKET_IFNAME="$_v" ;;
    LLMSRV_NO_NVLS)        _YAML_LLMSRV_NO_NVLS="$_v" ;;
    LLMSRV_TWOSHOT_MIN_BYTES) _YAML_LLMSRV_TWOSHOT_MIN_BYTES="$_v" ;;
  esac
done < <(python3 -c "
import sys; sys.path.insert(0,'$HERE')
from xval_config import collective
for k, v in collective().items():
    print(k + '=' + v)
")
NCCL_ALGO="${NCCL_ALGO:-$_YAML_NCCL_ALGO}"
NCCL_PROTO="${NCCL_PROTO:-$_YAML_NCCL_PROTO}"
NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-$_YAML_NCCL_P2P_LEVEL}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-$_YAML_NCCL_IB_DISABLE}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$_YAML_NCCL_SOCKET_IFNAME}"
LLMSRV_NO_NVLS="${LLMSRV_NO_NVLS:-$_YAML_LLMSRV_NO_NVLS}"
LLMSRV_TWOSHOT_MIN_BYTES="${LLMSRV_TWOSHOT_MIN_BYTES:-$_YAML_LLMSRV_TWOSHOT_MIN_BYTES}"
for _var in NCCL_ALGO NCCL_PROTO NCCL_P2P_LEVEL NCCL_IB_DISABLE \
  NCCL_SOCKET_IFNAME LLMSRV_NO_NVLS LLMSRV_TWOSHOT_MIN_BYTES; do
  if [ -n "${!_var}" ]; then export "$_var"; fi
done

# Kernel suite: the engine picks decode kernels from env; unset means the
# Baseline recurrence kernel, which is NOT what the published numbers use.
# exact = the vLLM-parity suite, but its GDN cubins must first be exported on
# the target GPU (qwen/KERNELS.md: scripts/export_vllm_gdn_triton.py under a
# vLLM 0.27.1 python) and QS_VLLM_GDN_DIR pointed at them; without the dir
# the exact loader panics at startup. The default therefore selects exact
# only when the dir is present, else the tiled rust suite. QS_CUSTOM_AR=1
# pairs the P2P all-reduce at tp>=2 (the engine's default there is NCCL).
# Caller env overrides everything.
if [ -z "${QS_KERNELS:-}" ]; then
  if [ -n "${QS_VLLM_GDN_DIR:-}" ]; then QS_KERNELS=exact; else QS_KERNELS=rust; fi
fi
export QS_KERNELS
if [ -n "${QS_VLLM_GDN_DIR:-}" ]; then export QS_VLLM_GDN_DIR; fi
if [ "$WORLD" -ge 2 ]; then
  QS_CUSTOM_AR="${QS_CUSTOM_AR:-1}"
  export QS_CUSTOM_AR
fi

if [ -z "$CELLS" ]; then
  echo "no cells from workloads grid; check xval_config workloads['qwen3']['grid']" >&2
  exit 1
fi

: > "$OUT"
FAILED=0
while read -r ctx bs; do
  # max_seq must satisfy: prompt + steps + 8 <= max_seq.
  max_seq=$((ctx + STEPS + _MAX_SEQ_HEADROOM))
  echo ">>> ctx=$ctx bs=$bs world=$WORLD max_seq=$max_seq link=$XVAL_LINK_PROFILE nccl=${NCCL_ALGO:-auto}/${NCCL_PROTO:-auto} kern=$QS_KERNELS ar=${QS_CUSTOM_AR:-0} gdn_dir=${QS_VLLM_GDN_DIR:-none}" | tee -a "$OUT" >&2
  before=$(wc -l < "$OUT")
  (
    cd "$QS" || exit 1
    export QW_CKPT QW_WORLD="$WORLD" QW_BATCH="$bs" \
      QW_BENCH_PROMPT="$ctx" QW_MAX_SEQ="$max_seq" QW_TIMING_STEPS="$STEPS"
    if [ -n "${QW_BENCH_BIN:-}" ]; then
      exec "$QW_BENCH_BIN" --nocapture
    fi
    exec cargo test -p qwen --test decode_loop --release -- --nocapture
  ) 2>&1 | tee -a "$OUT"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "FAIL ctx=$ctx bs=$bs (bench exited $rc; see $OUT)" | tee -a "$OUT" >&2
    exit "$rc"
  fi
  if ! tail -n +"$((before + 1))" "$OUT" | grep -q "^LOOP "; then
    echo "FAIL ctx=$ctx bs=$bs (no LOOP line; see $OUT)" | tee -a "$OUT" >&2
    FAILED=$((FAILED + 1))
  fi
done <<< "$CELLS"
echo "wrote $OUT" >&2
[ "$FAILED" -eq 0 ] || { echo "$FAILED cell(s) failed" >&2; exit 1; }
