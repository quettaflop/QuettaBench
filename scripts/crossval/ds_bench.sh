#!/usr/bin/env bash
# Sweep the deepseek grid through the engine's batch_bench test into one log.
# Run from a QuettaServe checkout (QS_DIR overrides); DS_BENCH_BIN points at a
# prebuilt binary when cargo is absent.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${DS_CKPT:?set DS_CKPT (the mp<world> checkpoint)}"
: "${DS_CFG:?set DS_CFG (the original inference config.json)}"
QS="${QS_DIR:-.}"
if [ -z "${DS_BENCH_BIN:-}" ] && [ ! -f "$QS/deepseek/Cargo.toml" ]; then
  echo "no deepseek crate under '$QS'; run from the QuettaServe root, set QS_DIR, or set DS_BENCH_BIN" >&2
  exit 1
fi
OUT="${1:-deepseek-bench.log}"
# DS_LAYERS would truncate the model.
unset DS_LAYERS

# Link profile picks the collective env. PCIe boxes get the tuned pins; NVLink
# boxes get none, because NCCL's NVLS/LL128 selection and the engine's
# peer-allreduce beat the pins there while vLLM's TP path ignores NCCL_ALGO,
# so pinning would slow only the engine. Override with XVAL_LINK_PROFILE.
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

# Read run params and collective env from xval.yaml in one Python call each.
# xval_config.py falls back to hardcoded defaults when xval.yaml is absent.
# Caller env still overrides via ${VAR:-yaml_value} below.
read -r _STEPS _HEADROOM < <(python3 -c "
import sys; sys.path.insert(0,'$HERE')
from xval_config import run_params
p = run_params()
print(p['timing_steps'], p['max_seq_headroom'])
")
STEPS="${DS_TIMING_STEPS:-$_STEPS}"
_MAX_SEQ_HEADROOM="${DS_MAX_SEQ_HEADROOM:-$_HEADROOM}"

# Cells and tp come from xval.yaml (via xval_config); falls back to workloads.json when absent.
CELLS="$(python3 -c "
import sys; sys.path.insert(0,'$HERE')
from xval_config import workloads, grids
wl = workloads()['deepseek']
for ctx, bs in grids()[wl['grid']]:
    print(ctx, bs)
")"
WORLD="${DS_WORLD:-$(python3 -c "
import sys; sys.path.insert(0,'$HERE')
from xval_config import workloads
print(workloads()['deepseek'].get('tp', 1))
")}"

# Collective env from xval.yaml; one Python call emits 7 lines: KEY=value.
# Values contain no newlines; NCCL_ALGO contains semicolons (safe for read -d).
# pcie profile: reduces stay Tree on both sides for impartiality; all-gather
# must be Ring because NCCL has no Tree all-gather; without SYS, cross-socket
# tree links fall back to TCP (~4x slower). nvlink profile: values arrive
# empty and are not exported, so NCCL and the engine keep their defaults.
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
# Empty means "leave unset": an exported empty NCCL var is not the same as an
# absent one, so only non-empty values are exported.
for _var in NCCL_ALGO NCCL_PROTO NCCL_P2P_LEVEL NCCL_IB_DISABLE \
  NCCL_SOCKET_IFNAME LLMSRV_NO_NVLS LLMSRV_TWOSHOT_MIN_BYTES; do
  if [ -n "${!_var}" ]; then export "$_var"; fi
done

if [ -z "$CELLS" ]; then
  echo "no cells from workloads grid; check xval_config workloads['deepseek']['grid']" >&2
  exit 1
fi

: > "$OUT"
FAILED=0
while read -r ctx bs; do
  # The bench asserts prompt + steps + 8 <= max_seq.
  max_seq=$((ctx + STEPS + _MAX_SEQ_HEADROOM))
  echo ">>> ctx=$ctx bs=$bs world=$WORLD max_seq=$max_seq link=$XVAL_LINK_PROFILE nccl=${NCCL_ALGO:-auto}/${NCCL_PROTO:-auto}" | tee -a "$OUT" >&2
  before=$(wc -l < "$OUT")
  (
    cd "$QS" || exit 1
    # Collective vars were exported above only when non-empty; the subshell
    # inherits them, so re-exporting (and re-setting empties) is avoided here.
    export DS_CKPT DS_CFG DS_WORLD="$WORLD" DS_BATCH="$bs" \
      DS_BENCH_PROMPT="$ctx" DS_MAX_SEQ="$max_seq" DS_TIMING_STEPS="$STEPS"
    if [ -n "${DS_BENCH_BIN:-}" ]; then
      exec "$DS_BENCH_BIN" --nocapture
    fi
    exec cargo test -p deepseek --test batch_bench --release -- --nocapture
  ) 2>&1 | tee -a "$OUT"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "FAIL ctx=$ctx bs=$bs (bench exited $rc; see $OUT)" | tee -a "$OUT" >&2
    exit "$rc"
  fi
  if ! tail -n +"$((before + 1))" "$OUT" | grep -q "^\[batch_bench\] .*layers=all"; then
    echo "FAIL ctx=$ctx bs=$bs (no full-model batch_bench line; see $OUT)" | tee -a "$OUT" >&2
    FAILED=$((FAILED + 1))
  fi
done <<< "$CELLS"
echo "wrote $OUT" >&2
[ "$FAILED" -eq 0 ] || { echo "$FAILED cell(s) failed" >&2; exit 1; }
