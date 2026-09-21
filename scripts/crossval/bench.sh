#!/usr/bin/env bash
# bench.sh <workload> [out-log]: sweep one workload's grid through its engine
# bench. The per-model pieces (crate, test, env prefix, pass pattern, family)
# come from the workload's "bench" section in workloads.json / xval.yaml.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WL="${1:?workload (a workloads.json entry with a bench section)}"
OUT="${2:-$WL-bench.log}"

mapfile -t _B < <(python3 -c "
import sys; sys.path.insert(0,'$HERE')
from xval_config import workloads
wl = workloads().get('$WL') or sys.exit('unknown workload $WL')
b = wl.get('bench') or sys.exit('workload $WL has no bench config')
print(b['crate']); print(b['test']); print(b['env']); print(b['family']); print(b['ok'])
print(wl.get('tp', 1))
")
[ "${#_B[@]}" -eq 6 ] || exit 1
CRATE="${_B[0]}"; TEST="${_B[1]}"; P="${_B[2]}"; FAMILY="${_B[3]}"; OK="${_B[4]}"; TP="${_B[5]}"

_V="${P}_CKPT"; CKPT="${!_V:-}"
[ -n "$CKPT" ] || { echo "set ${P}_CKPT" >&2; exit 1; }
CFG=""
if [ "$FAMILY" = moe ]; then
  _V="${P}_CFG"; CFG="${!_V:-}"
  [ -n "$CFG" ] || { echo "set ${P}_CFG" >&2; exit 1; }
  unset "${P}_LAYERS"
fi

QS="${QS_DIR:-.}"
_V="${P}_BENCH_BIN"; BIN="${!_V:-}"
if [ -z "$BIN" ] && [ ! -f "$QS/$CRATE/Cargo.toml" ]; then
  echo "no $CRATE crate under '$QS'; run from the QuettaServe root, set QS_DIR, or set ${P}_BENCH_BIN" >&2
  exit 1
fi

if [ -z "${XVAL_LINK_PROFILE:-}" ]; then
  # Capture then match; piping to grep -q would SIGPIPE nvidia-smi and pipefail misreads it.
  _topo="$(nvidia-smi topo -m 2>/dev/null || true)"
  case "$_topo" in
    *NV[0-9]*) XVAL_LINK_PROFILE=nvlink ;;
    *)         XVAL_LINK_PROFILE=pcie ;;
  esac
fi
export XVAL_LINK_PROFILE

read -r _STEPS _HEADROOM < <(python3 -c "
import sys; sys.path.insert(0,'$HERE')
from xval_config import run_params
p = run_params()
print(p['timing_steps'], p['max_seq_headroom'])
")
_V="${P}_TIMING_STEPS"; STEPS="${!_V:-$_STEPS}"
_V="${P}_MAX_SEQ_HEADROOM"; HEADROOM="${!_V:-$_HEADROOM}"
_V="${P}_WORLD"; WORLD="${!_V:-$TP}"

CELLS="$(python3 -c "
import sys; sys.path.insert(0,'$HERE')
from xval_config import workloads, grids
for ctx, bs in grids()[workloads()['$WL']['grid']]:
    print(ctx, bs)
")"
[ -n "$CELLS" ] || { echo "no cells for workload $WL" >&2; exit 1; }

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

MODE=""
if [ "$FAMILY" = gdn ]; then
  # exact needs GDN cubins in QS_VLLM_GDN_DIR or the loader panics; default exact only when set.
  if [ -z "${QS_KERNELS:-}" ]; then
    if [ -n "${QS_VLLM_GDN_DIR:-}" ]; then QS_KERNELS=exact; else QS_KERNELS=rust; fi
  fi
  export QS_KERNELS
  if [ -n "${QS_VLLM_GDN_DIR:-}" ]; then export QS_VLLM_GDN_DIR; fi
  if [ "$WORLD" -ge 2 ]; then export QS_CUSTOM_AR="${QS_CUSTOM_AR:-1}"; fi
  _V="${P}_MODE"; MODE="${!_V:-}"
  if [ -z "$MODE" ]; then
    if [ "$QS_KERNELS" = exact ]; then MODE=eager; else MODE=graph; fi
  fi
  export "${P}_MODE=$MODE"
fi

: > "$OUT"
FAILED=0
while read -r ctx bs; do
  max_seq=$((ctx + STEPS + HEADROOM))
  hdr=">>> ctx=$ctx bs=$bs world=$WORLD max_seq=$max_seq link=$XVAL_LINK_PROFILE nccl=${NCCL_ALGO:-auto}/${NCCL_PROTO:-auto}"
  if [ "$FAMILY" = gdn ]; then
    hdr="$hdr kern=$QS_KERNELS ar=${QS_CUSTOM_AR:-0} mode=$MODE gdn_dir=${QS_VLLM_GDN_DIR:-none}"
  fi
  echo "$hdr" | tee -a "$OUT" >&2
  before=$(wc -l < "$OUT")
  (
    cd "$QS" || exit 1
    export "${P}_CKPT=$CKPT" "${P}_WORLD=$WORLD" "${P}_BATCH=$bs" \
      "${P}_BENCH_PROMPT=$ctx" "${P}_MAX_SEQ=$max_seq" "${P}_TIMING_STEPS=$STEPS"
    if [ -n "$CFG" ]; then export "${P}_CFG=$CFG"; fi
    if [ -n "$BIN" ]; then exec "$BIN" --nocapture; fi
    exec cargo test -p "$CRATE" --test "$TEST" --release -- --nocapture
  ) 2>&1 | tee -a "$OUT"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "FAIL ctx=$ctx bs=$bs (bench exited $rc; see $OUT)" | tee -a "$OUT" >&2
    exit "$rc"
  fi
  if ! tail -n +"$((before + 1))" "$OUT" | grep -Eq "$OK"; then
    echo "FAIL ctx=$ctx bs=$bs (no bench result line; see $OUT)" | tee -a "$OUT" >&2
    FAILED=$((FAILED + 1))
  fi
done <<< "$CELLS"
echo "wrote $OUT" >&2
[ "$FAILED" -eq 0 ] || { echo "$FAILED cell(s) failed" >&2; exit 1; }
