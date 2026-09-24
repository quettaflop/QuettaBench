#!/usr/bin/env bash
# oneshot.sh <workload...>: empty container to full cross-validation in one pass.
# Provisions via bootstrap.sh, then for each workload sweeps the engine and runs
# the baseline, checks, table and (if XVAL_SERVE_TRACE is set) the serving replay
# with the workload-identity gate via xval.sh. XVAL_DRYRUN=1 stops after the plan.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ "$#" -ge 1 ] || { echo "usage: oneshot.sh <workload> [workload...]" >&2; exit 1; }

"$HERE/bootstrap.sh" "$@"
[ "${XVAL_DRYRUN:-0}" = 1 ] && exit 0
# shellcheck source=/dev/null
. "$HERE/.xval_env"

for WL in "$@"; do
  key="${WL//[^A-Za-z0-9]/_}"
  _WV="XVAL_WEIGHTS_$key"; MDIR="${!_WV:?bootstrap resolved no weights for $WL}"
  echo "=== $WL ==="
  LOG="$HERE/$WL-bench.log"
  "$HERE/bench.sh" "$WL" "$LOG"
  PY="$XVAL_PY" "$HERE/xval.sh" "$WL" "$MDIR" "$LOG"
done
