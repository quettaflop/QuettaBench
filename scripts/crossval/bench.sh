#!/usr/bin/env bash
# bench.sh <workload> [out-log]: sweep one workload's grid through its engine
# bench. The per-model pieces (crate, test, env prefix, pass pattern, family)
# come from the workload's "bench" section in workloads.json / xval.yaml.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WL="${1:?workload (a workloads.json entry with a bench section)}"
OUT="${2:-$WL-bench.log}"

mapfile -t _B < <(python3 "$HERE/xval_config.py" bench "$WL")
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

export XVAL_LINK_PROFILE="${XVAL_LINK_PROFILE:-$(python3 "$HERE/xval_config.py" link-profile)}"

read -r _STEPS _HEADROOM < <(python3 "$HERE/xval_config.py" run-params)
_V="${P}_TIMING_STEPS"; STEPS="${!_V:-$_STEPS}"
_V="${P}_MAX_SEQ_HEADROOM"; HEADROOM="${!_V:-$_HEADROOM}"
_V="${P}_WORLD"; WORLD="${!_V:-$TP}"

CELLS="$(python3 "$HERE/xval_config.py" cells "$WL")"
[ -n "$CELLS" ] || { echo "no cells for workload $WL" >&2; exit 1; }

# Collective env: caller env wins over yaml; empty values are never exported.
while IFS='=' read -r _k _v; do
  if [ -n "${!_k:-}" ]; then _v="${!_k}"; fi
  if [ -n "$_v" ]; then export "$_k=$_v"; fi
done < <(python3 "$HERE/xval_config.py" collective)

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
