#!/usr/bin/env bash
# bootstrap.sh [workload...]: provision a fresh GPU container for the
# cross-validator and write the resolved env to .xval_env, which the run
# scripts source. Idempotent: every step checks before acting. XVAL_DRYRUN=1
# prints the plan and writes nothing.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cfg() { python3 "$HERE/xval_config.py" "$@"; }
ENVFILE="$HERE/.xval_env"
DRY="${XVAL_DRYRUN:-0}"
MISSING=0

log()  { echo "[bootstrap] $*" >&2; }
run()  { if [ "$DRY" = 1 ]; then log "would: $*"; else log "+ $*"; eval "$@"; fi; }
need() { log "MISSING: $*"; MISSING=1; }
: > "$ENVFILE.tmp"
save() { printf 'export %s=%q\n' "$1" "$2" >> "$ENVFILE.tmp"; }

# 1. Caches off the small container root; torch.compile and the vLLM cache overflow it.
CACHE_ROOT="${XVAL_CACHE_ROOT:-$(cfg provision cache_root)}"
run "mkdir -p '$CACHE_ROOT'/vllm '$CACHE_ROOT'/torch '$CACHE_ROOT'/hf '$CACHE_ROOT'/tmp '$CACHE_ROOT'/models"
save VLLM_CACHE_ROOT "$CACHE_ROOT/vllm"
save XDG_CACHE_HOME "$CACHE_ROOT/vllm"
save TORCHINDUCTOR_CACHE_DIR "$CACHE_ROOT/torch"
save HF_HOME "$CACHE_ROOT/hf"
save TMPDIR "$CACHE_ROOT/tmp"

# 2. A python that has vLLM: provision.py / XVAL_PY, else pip install provision.vllm_spec.
PY="${XVAL_PY:-$(cfg provision py)}"; PY="${PY:-python3}"
if "$PY" -m pip show vllm >/dev/null 2>&1; then
  log "vllm python ok: $PY"
else
  SPEC="$(cfg provision vllm_spec)"
  if [ -n "$SPEC" ]; then run "'$PY' -m pip install $SPEC" || need "pip install '$SPEC' failed"
  else need "no vllm in '$PY'; set provision.py to a vllm python or provision.vllm_spec to install one"; fi
fi
save XVAL_PY "$PY"

# 3. Blackwell needs NCCL >= the runtime floor; prepend provision.nccl_lib when given.
NCCL_LIB="${XVAL_NCCL_LIB:-$(cfg provision nccl_lib)}"
if [ -n "$NCCL_LIB" ]; then
  [ -d "$NCCL_LIB" ] || need "provision.nccl_lib '$NCCL_LIB' does not exist"
  save LD_LIBRARY_PATH "$NCCL_LIB:${LD_LIBRARY_PATH:-}"
fi

# 4. QuettaServe checkout for the engine benches, cloned when QS_DIR has none.
QS="${QS_DIR:-$HERE/QuettaServe}"
if [ ! -d "$QS/.git" ] && [ -n "$(cfg provision qs_repo)" ]; then
  REPO="$(cfg provision qs_repo)"; COMMIT="$(cfg provision qs_commit)"
  run "git clone '$REPO' '$QS'" || need "git clone '$REPO' failed"
  [ -n "$COMMIT" ] && run "git -C '$QS' checkout '$COMMIT'"
fi

# 5. Per workload: engine bench binary, engine checkpoint (+moe config), vLLM weights.
for WL in "$@"; do
  key="${WL//[^A-Za-z0-9]/_}"
  mapfile -t B < <(cfg bench "$WL")
  [ "${#B[@]}" -ge 4 ] || { need "workload $WL has no bench section"; continue; }
  CRATE="${B[0]}"; TEST="${B[1]}"; P="${B[2]}"; FAMILY="${B[3]}"

  _V="${P}_BENCH_BIN"; BIN="${!_V:-}"
  if [ -n "$BIN" ] && [ -x "$BIN" ]; then
    save "${P}_BENCH_BIN" "$BIN"; log "$WL: bench bin $BIN"
  elif [ "$DRY" = 1 ]; then
    log "would: build $CRATE/$TEST in $QS"
  elif [ -f "$QS/$CRATE/Cargo.toml" ]; then
    if ( cd "$QS" && cargo test -p "$CRATE" --test "$TEST" --release --no-run ) >&2; then
      BUILT="$(ls -t "$QS"/target/release/deps/"$TEST"-* 2>/dev/null | grep -vE '\.(d|o)$' | head -1)"
      if [ -n "$BUILT" ]; then save "${P}_BENCH_BIN" "$BUILT"; log "$WL: built $BUILT"
      else need "$WL: no $TEST binary after build"; fi
    else need "$WL: cargo build failed"; fi
  else
    need "$WL engine: set ${P}_BENCH_BIN, or provision.qs_repo, or QS_DIR to a QuettaServe checkout"
  fi

  _CK="XVAL_CKPT_$key"; CK="${!_CK:-}"
  if [ -n "$CK" ]; then save "${P}_CKPT" "$CK"; else need "$WL engine checkpoint: set XVAL_CKPT_$key"; fi
  if [ "$FAMILY" = moe ]; then
    _CF="XVAL_CFG_$key"; CF="${!_CF:-}"
    if [ -n "$CF" ]; then save "${P}_CFG" "$CF"; else need "$WL moe config: set XVAL_CFG_$key"; fi
  fi

  _WV="XVAL_WEIGHTS_$key"; WPATH="${!_WV:-}"
  _HV="XVAL_HF_$key"; HFID="${!_HV:-}"
  if [ -n "$WPATH" ]; then
    save "$_WV" "$WPATH"
  elif [ -n "$HFID" ]; then
    DEST="$CACHE_ROOT/models/$WL"
    if [ -d "$DEST" ] && [ -n "$(ls -A "$DEST" 2>/dev/null)" ]; then log "$WL: weights $DEST"
    else run "'$PY' -m huggingface_hub.commands.huggingface_cli download '$HFID' --local-dir '$DEST'" \
      || need "$WL: hf download $HFID failed (HF_TOKEN set?)"; fi
    save "$_WV" "$DEST"
  else
    need "$WL weights: set XVAL_WEIGHTS_$key=<dir> or XVAL_HF_$key=<hf-repo-id>"
  fi
done

if [ "$MISSING" != 0 ]; then log "provisioning incomplete; resolve the MISSING items above"; rm -f "$ENVFILE.tmp"; exit 1; fi
if [ "$DRY" = 1 ]; then log "dry run: plan only, no .xval_env written"; rm -f "$ENVFILE.tmp"; exit 0; fi
mv "$ENVFILE.tmp" "$ENVFILE"
log "wrote $ENVFILE; run oneshot.sh or xval.sh next"
