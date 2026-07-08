#!/usr/bin/env bash
# Cron-driven benchmark orchestrator.
# - Compiles sweep.yaml into a scoped benchmark job manifest
# - Per host: if host idle, fire next pending job; if host busy, skip
# - Detects completed sweeps (rsync + s3 sync, mark done)
# - Detects OOM failures (parse vllm log), retries once with reduced max_len
# - Idempotent — safe to run on cron
#
# Cron line:
#   */30 * * * * BENCH_STATE_ROOT=/mnt/100g/agent-bench/state bash /root/QuettaBench/scripts/bench_orchestrator.sh >> /tmp/bench_orchestrator.cron.log 2>&1
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
JOBS_CONFIG="${BENCH_JOBS_CONFIG:-$REPO_ROOT/inference-benchmark/scripts/sweep.yaml}"
EXPLICIT_JOBS_FILE="${BENCH_JOBS_FILE:-}"
JOBS_CACHE_DIR="${BENCH_JOBS_CACHE_DIR:-/tmp/bench_jobs}"
JOBS_MANIFEST="${BENCH_JOBS_MANIFEST:-}"
STATE_ROOT="${BENCH_STATE_ROOT:-/mnt/100g/agent-bench/state}"
LEGACY_STATE_ROOT="${BENCH_LEGACY_STATE_ROOT:-/tmp/bench_jobs/state}"
LOCAL_RESULTS_ROOT="${BENCH_RESULTS_ROOT:-${BENCHMARK_RESULTS_DIR:-/mnt/100g/agent-bench/results}}"
CONTROL_DIR="${BENCH_CONTROL_DIR:-$STATE_ROOT/control}"
DRAINED_HOSTS_FILE="${BENCH_DRAINED_HOSTS_FILE:-$CONTROL_DIR/drained-hosts.txt}"
BLOCKED_GPUS_FILE="${BENCH_BLOCKED_GPUS_FILE:-$CONTROL_DIR/blocked-gpus.txt}"
REMOTE_TMP_ROOT="${BENCH_REMOTE_TMP:-/tmp}"
REMOTE_CODE_ROOT="${BENCH_REMOTE_ROOT:-/tmp/inference-benchmark}"
H100_REMOTE_TMP_ROOT="${BENCH_H100_REMOTE_TMP:-/data48/tmp}"
H100_REMOTE_CODE_ROOT="${BENCH_H100_REMOTE_ROOT:-$H100_REMOTE_TMP_ROOT/inference-benchmark}"
LOG="${BENCH_ORCHESTRATOR_LOG:-/tmp/bench_orchestrator.log}"
EP="${R2_ENDPOINT:-https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com}"
BUCKET="${R2_BUCKET:-agent-bench}"
PROFILE="${AWS_PROFILE:-r2}"
# Raw benchmark outputs are namespaced so trace replay, synthetic
# distributional, and retired archived runs do not overwrite each other in R2.
# Override only for one-off maintenance, e.g. RESULT_SCOPE=archived/foo.
DEFAULT_RESULT_SCOPE="${RESULT_SCOPE:-}"
DRY_RUN="${BENCH_ORCHESTRATOR_DRY_RUN:-0}"
SKIP_REMOTE_PROBE="${BENCH_ORCHESTRATOR_SKIP_REMOTE_PROBE:-0}"
FLEXIBLE_PINNED_GPUS="${BENCH_FLEXIBLE_PINNED_GPUS:-1}"
MAX_DISPATCHES="${BENCH_ORCHESTRATOR_MAX_DISPATCHES:-0}"
DISPATCHES=0
RECLAIM_BEFORE_DISPATCH="${BENCH_RECLAIM_BEFORE_DISPATCH:-0}"
RECLAIM_EXECUTE="${BENCH_RECLAIM_EXECUTE:-0}"
RECLAIM_CONFIG="${BENCH_RECLAIM_CONFIG:-$REPO_ROOT/inference-benchmark/scripts/gpu_cleanup.json}"
RECONCILE_COVERAGE_BEFORE_DISPATCH="${BENCH_RECONCILE_COVERAGE_BEFORE_DISPATCH:-1}"
COVERAGE_RESET_STATUSES="${BENCH_COVERAGE_RESET_STATUSES:-done,skipped,failed,known_oom}"
COVERAGE_MAX_REQUEUES="${BENCH_COVERAGE_MAX_REQUEUES:-1}"
if ! [[ "$COVERAGE_MAX_REQUEUES" =~ ^-?[0-9]+$ ]]; then
    COVERAGE_MAX_REQUEUES=1
fi
# Runtime override via a control file (no restart needed; mirrors
# drained-hosts.txt / blocked-gpus.txt). Lets ops raise the coverage requeue
# budget on the fly so incomplete jobs keep auto-requeuing onto idle GPUs.
MAX_REQUEUES_FILE="${BENCH_MAX_REQUEUES_FILE:-$CONTROL_DIR/max-coverage-requeues}"
if [[ -f "$MAX_REQUEUES_FILE" ]]; then
    _mrq="$(tr -dc '0-9-' < "$MAX_REQUEUES_FILE" 2>/dev/null)"
    [[ "$_mrq" =~ ^-?[0-9]+$ ]] && COVERAGE_MAX_REQUEUES="$_mrq"
fi
MAX_OOM_RETRIES="${BENCH_ORCHESTRATOR_MAX_OOM_RETRIES:-3}"
if ! [[ "$MAX_OOM_RETRIES" =~ ^[0-9]+$ ]]; then
    MAX_OOM_RETRIES=3
fi
MAX_INCOMPLETE_RETRIES="${BENCH_ORCHESTRATOR_MAX_INCOMPLETE_RETRIES:-2}"
if ! [[ "$MAX_INCOMPLETE_RETRIES" =~ ^[0-9]+$ ]] || [[ "$MAX_INCOMPLETE_RETRIES" -lt 1 ]]; then
    MAX_INCOMPLETE_RETRIES=2
fi

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }
truthy() { [[ "${1:-}" == "1" || "${1:-}" == "true" || "${1:-}" == "yes" ]]; }
dry_run() { truthy "$DRY_RUN"; }
safe_name() { printf '%s' "${1:-all}" | tr -c 'A-Za-z0-9_.-' '_'; }
b64_arg() { base64 | tr -d '\n'; }

host_drained() {
    local host="$1"
    [[ -f "$DRAINED_HOSTS_FILE" ]] || return 1
    awk -v host="$host" '
        /^[[:space:]]*(#|$)/ { next }
        $1 == host { found = 1 }
        END { exit found ? 0 : 1 }
    ' "$DRAINED_HOSTS_FILE"
}

load_blocked_gpus() {
    [[ -f "$BLOCKED_GPUS_FILE" ]] || return 0
    while IFS=' ' read -r host gpu; do
        [[ -n "$host" && "$gpu" =~ ^[0-9]+$ ]] || continue
        HOST_BLOCKED_GPUS[$host]="${HOST_BLOCKED_GPUS[$host]:-} $gpu "
    done < <(
        awk '
            /^[[:space:]]*(#|$)/ { next }
            {
                host = $1
                gpu = $2
                if (index(host, ":") > 0 && gpu == "") {
                    split(host, parts, ":")
                    host = parts[1]
                    gpu = parts[2]
                }
                if (host != "" && gpu ~ /^[0-9]+$/) {
                    print host, gpu
                }
            }
        ' "$BLOCKED_GPUS_FILE"
    )
}

canonical_scope() {
    case "${1:-}" in
        synthetic|latest|synthetic-distributional|synthetic_distributional) echo "synthetic_distributional" ;;
        archive|trace_replay) echo "trace_replay" ;;
        current|canonical|fixed|fixed-grid|mse|archived) echo "archived" ;;
        *) echo "${1:-}" ;;
    esac
}

state_scope_aliases() {
    case "$(canonical_scope "$1")" in
        synthetic_distributional) echo "synthetic_distributional synthetic latest synthetic-distributional" ;;
        trace_replay) echo "trace_replay archive" ;;
        archived) echo "archived current canonical fixed fixed-grid mse" ;;
        *) echo "$1" ;;
    esac
}

LOCK_FILE="/tmp/bench_orchestrator.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    log "another bench_orchestrator.sh tick is already running; exiting"
    exit 0
fi

JOBS_FILE="$EXPLICIT_JOBS_FILE"
if [[ -z "$JOBS_FILE" ]]; then
    if [[ -z "${BENCH_JOBS_SCOPE:-}" ]]; then
        log "BENCH_JOBS_SCOPE is required when compiling jobs from $JOBS_CONFIG"
        exit 1
    fi
    REQUESTED_JOBS_SCOPE="$BENCH_JOBS_SCOPE"
    SAFE_SCOPE="$(safe_name "$REQUESTED_JOBS_SCOPE")"
    mkdir -p "$JOBS_CACHE_DIR"
    JOBS_FILE="$JOBS_CACHE_DIR/bench_jobs.${SAFE_SCOPE}.txt"
    JOBS_MANIFEST="${JOBS_MANIFEST:-$JOBS_CACHE_DIR/bench_jobs.${SAFE_SCOPE}.json}"
    mkdir -p "$(dirname "$JOBS_FILE")" "$(dirname "$JOBS_MANIFEST")"
    if ! python3 "$REPO_ROOT/inference-benchmark/scripts/compile_sweep.py" \
        --yaml "$JOBS_CONFIG" --scope "$REQUESTED_JOBS_SCOPE" --format text --out "$JOBS_FILE" >> "$LOG" 2>&1; then
        log "failed to compile job rows from $JOBS_CONFIG scope=$REQUESTED_JOBS_SCOPE"
        exit 1
    fi
    if ! python3 "$REPO_ROOT/inference-benchmark/scripts/compile_sweep.py" \
        --yaml "$JOBS_CONFIG" --scope "$REQUESTED_JOBS_SCOPE" --format json --out "$JOBS_MANIFEST" >> "$LOG" 2>&1; then
        log "failed to compile job manifest from $JOBS_CONFIG scope=$REQUESTED_JOBS_SCOPE"
        exit 1
    fi
    log "compiled jobs from config=$JOBS_CONFIG scope=$REQUESTED_JOBS_SCOPE manifest=$JOBS_MANIFEST rows=$JOBS_FILE"
else
    log "using explicit legacy jobs file=$JOBS_FILE"
fi

RAW_JOBS_SCOPE=$(awk -F': ' '/^# SCOPE:/ {print $2; exit}' "$JOBS_FILE" 2>/dev/null || true)
RAW_EXPECTED_JOBS_SCOPE="${BENCH_JOBS_SCOPE:-${RAW_JOBS_SCOPE:-fixed}}"
JOBS_SCOPE="$(canonical_scope "$RAW_JOBS_SCOPE")"
EXPECTED_JOBS_SCOPE="$(canonical_scope "$RAW_EXPECTED_JOBS_SCOPE")"
if [[ "$EXPECTED_JOBS_SCOPE" != "all" ]]; then
    if [[ "$JOBS_SCOPE" != "$EXPECTED_JOBS_SCOPE" ]]; then
        log "refusing to run: $JOBS_FILE has scope='${RAW_JOBS_SCOPE:-missing}' normalized='${JOBS_SCOPE:-missing}', expected '${RAW_EXPECTED_JOBS_SCOPE}' normalized='$EXPECTED_JOBS_SCOPE'"
        exit 1
    fi
fi
STATE_SCOPE="$EXPECTED_JOBS_SCOPE"
if [[ "$STATE_SCOPE" == "all" ]]; then
    STATE_SCOPE="${JOBS_SCOPE:-all}"
fi
STATE_DIR="$STATE_ROOT/$STATE_SCOPE"
LEGACY_STATE_DIR="$LEGACY_STATE_ROOT/$STATE_SCOPE"
RUNS_DIR="$STATE_DIR/runs"
mkdir -p "$STATE_DIR"
mkdir -p "$LOCAL_RESULTS_ROOT"
log "using jobs scope=${JOBS_SCOPE:-missing} expected_scope=$EXPECTED_JOBS_SCOPE state_dir=$STATE_DIR results_root=$LOCAL_RESULTS_ROOT"
dry_run && log "dry-run enabled: no state writes, rsync, R2 upload, sweep-state publish, or remote launch"
if [[ "$MAX_DISPATCHES" =~ ^[0-9]+$ && "$MAX_DISPATCHES" -gt 0 ]]; then
    log "max dispatches for this tick: $MAX_DISPATCHES"
fi
mapfile -t JOB_HOSTS < <(awk -F'|' '!/^#/ && NF >= 10 {gsub(/[[:space:]]/, "", $1); if ($1 != "") print $1}' "$JOBS_FILE" | sort -u)

if truthy "$RECONCILE_COVERAGE_BEFORE_DISPATCH"; then
    case "$STATE_SCOPE" in
        synthetic_distributional)
            DEFAULT_COVERAGE_DATA="$REPO_ROOT/inference-benchmark/dashboard/public/data.synthetic_distributional.json"
            ;;
        trace_replay)
            DEFAULT_COVERAGE_DATA="$REPO_ROOT/inference-benchmark/dashboard/public/data.trace_replay.json"
            ;;
        archived)
            DEFAULT_COVERAGE_DATA="$REPO_ROOT/inference-benchmark/dashboard/public/data.archived.json"
            ;;
        *)
            DEFAULT_COVERAGE_DATA="$REPO_ROOT/inference-benchmark/dashboard/public/data.json"
            ;;
    esac
    COVERAGE_DATA="${BENCH_COVERAGE_DATA:-$DEFAULT_COVERAGE_DATA}"
    COVERAGE_REPORT="${BENCH_COVERAGE_REPORT:-/tmp/sweep-coverage-reconcile-${STATE_SCOPE}.md}"
    COVERAGE_MISSING_JOBS="${BENCH_COVERAGE_MISSING_JOBS:-/tmp/bench_jobs/missing_${STATE_SCOPE}_bench_jobs.txt}"
    COVERAGE_BLOCKERS_JSON="${BENCH_COVERAGE_BLOCKERS_JSON:-$REPO_ROOT/inference-benchmark/dashboard/dist/coverage-blockers.${STATE_SCOPE}.json}"
    COVERAGE_SWEEP_STATE_OUT="${BENCH_COVERAGE_SWEEP_STATE_OUT:-$REPO_ROOT/inference-benchmark/dashboard/public/sweep-state.json}"
    if dry_run; then
        log "dry-run: skipping coverage reconcile preflight"
    elif [[ ! -s "$COVERAGE_DATA" ]]; then
        log "coverage reconcile preflight skipped: missing data file $COVERAGE_DATA"
    else
        log "running coverage reconcile preflight data=$COVERAGE_DATA reset_statuses=$COVERAGE_RESET_STATUSES max_requeues=$COVERAGE_MAX_REQUEUES"
        if ! python3 "$REPO_ROOT/inference-benchmark/scripts/reconcile_sweep_coverage.py" \
            --scope "$STATE_SCOPE" \
            --data "$COVERAGE_DATA" \
            --sweep-yaml "$JOBS_CONFIG" \
            --bench-jobs "$JOBS_FILE" \
            --state-dir "$STATE_ROOT" \
            --report "$COVERAGE_REPORT" \
            --write-missing-jobs "$COVERAGE_MISSING_JOBS" \
            --write-blockers-json "$COVERAGE_BLOCKERS_JSON" \
            --write-sweep-state \
            --sweep-state-out "$COVERAGE_SWEEP_STATE_OUT" \
            --reset-stale \
            --reset-statuses "$COVERAGE_RESET_STATUSES" \
            --max-coverage-requeues "$COVERAGE_MAX_REQUEUES" \
            --limit 20 >> "$LOG" 2>&1; then
            log "coverage reconcile preflight failed; continuing without automatic coverage requeue"
        fi
    fi
fi

if truthy "$RECLAIM_BEFORE_DISPATCH"; then
    RECLAIM_MODE=(--dry-run)
    if truthy "$RECLAIM_EXECUTE" && ! dry_run; then
        RECLAIM_MODE=(--execute)
    fi
    log "running GPU reclaim preflight mode=${RECLAIM_MODE[*]} config=$RECLAIM_CONFIG"
    if ! python3 "$REPO_ROOT/inference-benchmark/scripts/clean_orphan_gpus.py" \
        --config "$RECLAIM_CONFIG" \
        --jobs-file "$JOBS_FILE" \
        --scope "$STATE_SCOPE" \
        --state-dir "$STATE_ROOT" \
        "${RECLAIM_MODE[@]}" >> "$LOG" 2>&1; then
        log "GPU reclaim preflight failed; continuing without reclaim"
    fi
fi

host_prefix() {
    case "$1" in
        a100)  echo "a100"  ;;
        3090)   echo "3090"  ;;
        2080ti) echo "2080ti" ;;
        *)      echo "$1"    ;;
    esac
}

host_python() {
    # args: host [backend]
    local host="$1" backend="${2:-vllm}"
    if [[ "$backend" == "sglang" ]]; then
        case "$host" in
            a100)       echo "/data/kevinlau/miniconda3/envs/sglang/bin/python" ;;
            3090|2080ti) echo "/home/kevinlau/miniconda3/envs/sglang/bin/python" ;;
            h100)        echo "/data/kevinlau/miniconda3/envs/sglang/bin/python" ;;
            h100-2)      echo "/home/kevinlau/miniconda3/envs/sglang/bin/python" ;;
        esac
    else
        case "$host" in
            a100)       echo "/data/kevinlau/miniconda3/bin/python" ;;
            3090|2080ti) echo "/home/kevinlau/miniconda3/envs/vllm/bin/python" ;;
            h100|h100-2) echo "/home/kevinlau/miniconda3/envs/vllm/bin/python" ;;
        esac
    fi
}

remote_tmp_root() {
    case "$1" in
        h100|h100-2) echo "$H100_REMOTE_TMP_ROOT" ;;
        *) echo "$REMOTE_TMP_ROOT" ;;
    esac
}

remote_code_root() {
    case "$1" in
        h100|h100-2) echo "$H100_REMOTE_CODE_ROOT" ;;
        *) echo "$REMOTE_CODE_ROOT" ;;
    esac
}

remote_results_root() {
    echo "$(remote_tmp_root "$1")/results"
}

# job_id keeps the legacy "host_model_tpN_mode" shape for vllm so existing
# state files in /tmp/bench_jobs/state/ remain valid. sglang cells get a
# "_sglang" suffix to disambiguate from the vllm run of the same cell.
job_id() {
    local jid="${1}_${2}_tp${3}_${4}"
    if [[ "${5:-vllm}" != "vllm" ]]; then
        jid="${jid}_${5}"
    fi
    echo "$jid"
}

extra_env_value() {
    local key="$1" text="${2:-}" part
    for part in $text; do
        if [[ "$part" == "$key="* ]]; then
            echo "${part#*=}"
            return
        fi
    done
}

remove_extra_env_key() {
    local key="$1" text="${2:-}" part
    local out=()
    for part in $text; do
        [[ "$part" == "$key="* ]] && continue
        out+=("$part")
    done
    echo "${out[*]}"
}

row_result_scope() {
    local extra_env="${1:-}" scope
    scope=$(extra_env_value "RESULT_SCOPE" "$extra_env")
    [[ -n "$scope" ]] && { echo "$scope"; return; }
    scope=$(extra_env_value "DASHBOARD_SCOPE" "$extra_env")
    [[ -n "$scope" ]] && { echo "$scope"; return; }
    scope=$(extra_env_value "SCOPE" "$extra_env")
    [[ -n "$scope" ]] && { echo "$scope"; return; }
    [[ -n "$DEFAULT_RESULT_SCOPE" ]] && { echo "$DEFAULT_RESULT_SCOPE"; return; }
    [[ "$JOBS_SCOPE" != "all" && -n "$JOBS_SCOPE" ]] && { echo "$JOBS_SCOPE"; return; }
    [[ "$EXPECTED_JOBS_SCOPE" != "all" && -n "$EXPECTED_JOBS_SCOPE" ]] && { echo "$EXPECTED_JOBS_SCOPE"; return; }
    echo "current"
}

dashboard_scope_for() {
    case "$1" in
        synthetic|latest|synthetic-distributional|synthetic_distributional) echo "synthetic_distributional" ;;
        archive|trace_replay) echo "trace_replay" ;;
        current|canonical|fixed|fixed-grid|mse|archived|archived/*) echo "archived" ;;
        *) echo "$1" ;;
    esac
}

storage_scope_for() {
    case "$1" in
        synthetic|latest|synthetic-distributional|synthetic_distributional) echo "synthetic_distributional" ;;
        archive|trace_replay) echo "trace_replay" ;;
        current|canonical) echo "archived/canonical" ;;
        fixed|fixed-grid) echo "archived/fixed-grid" ;;
        mse) echo "archived/mse" ;;
        archived) echo "archived" ;;
        archived/*) echo "$1" ;;
        *) echo "$1" ;;
    esac
}

expected_output_summary() {
    local dir="$1" short="$2" tp="$3" backend="$4" mode="$5" concs="$6" profiles="$7"
    local profile conc file total=0 present=0
    local missing_sample=()
    local missing_all=()
    for profile in $profiles; do
        for conc in $concs; do
            total=$((total + 1))
            if [[ "$mode" == "multi" ]]; then
                file="$dir/${profile}_conc${conc}.json"
            else
                file="$dir/${short}_tp${tp}_${backend}_${profile}_conc${conc}.json"
            fi
            if [[ -s "$file" ]]; then
                present=$((present + 1))
            else
                missing_all+=("$(basename "$file")")
                if [[ "${#missing_sample[@]}" -lt 4 ]]; then
                    missing_sample+=("$(basename "$file")")
                fi
            fi
        done
    done
    EXPECTED_OUTPUT_TOTAL="$total"
    EXPECTED_OUTPUT_PRESENT="$present"
    EXPECTED_OUTPUT_MISSING_SAMPLE="${missing_sample[*]:-}"
    EXPECTED_OUTPUT_MISSING_ALL="${missing_all[*]:-}"
}

oom_log_on_host() {
    local host="$1" port="$2"
    local log_path
    log_path="$(remote_tmp_root "$host")/vllm_${port}.log"
    ssh "$host" "grep -l -i -E 'OutOfMemoryError|CUDA out of memory|out of memory|No available memory for the cache blocks|Available KV cache memory: -|larger than the available KV cache memory|estimated maximum model length|max seq len .*larger than' '$log_path' /tmp/vllm_${port}.log 2>/dev/null" < /dev/null || true
}

oom_max_len_hint_on_host() {
    local host="$1" port="$2"
    local log_path
    log_path="$(remote_tmp_root "$host")/vllm_${port}.log"
    ssh "$host" "grep -oi -E 'estimated maximum model length is [0-9]+' '$log_path' /tmp/vllm_${port}.log 2>/dev/null | tail -1 | awk '{print \$NF}'" < /dev/null || true
}

sweep_log_done_on_host() {
    local host="$1" remote_log="$2" result
    [[ -n "$remote_log" ]] || return 1
    result=$(ssh "$host" "if grep -q 'done; results in ' '$remote_log' 2>/dev/null; then echo yes; fi" < /dev/null 2>/dev/null || true)
    [[ "$result" == "yes" ]]
}

next_oom_max_len() {
    local max_len="$1" hint="${2:-}" next
    next=$((max_len / 2))
    if [[ "$hint" =~ ^[0-9]+$ && "$hint" -gt 0 && "$hint" -lt "$next" ]]; then
        # Keep retries on the existing 2K/4K/8K/... grid while respecting
        # vLLM's own max-length estimate when it provides one.
        local rounded=2048
        while [[ $((rounded * 2)) -le "$hint" ]]; do
            rounded=$((rounded * 2))
        done
        next="$rounded"
    fi
    [[ "$next" -lt 2048 ]] && next=2048
    echo "$next"
}

can_retry_oom() {
    local oom="$1" attempt="$2" max_len="$3"
    [[ -n "$oom" && "$attempt" -lt "$MAX_OOM_RETRIES" && "$max_len" -gt 2048 ]]
}

failure_log_summary_on_host() {
    local host="$1" remote_log="$2"
    [[ -n "$remote_log" ]] || return 0
    ssh "$host" "grep -E 'ABORT: (Success rate|No requests completed)|\\[warn\\] (bench|mt-bench) failed' '$remote_log' 2>/dev/null | tail -6 | paste -sd '; ' -" < /dev/null 2>/dev/null || true
}

# Best-effort structured failure_class from the server/bench logs, so the
# coverage classifier no longer has to guess from a collapsed reason string.
# (RFC: docs/coverage-classification-rfc.md.) Echoes one of model_missing|
# oom_kv_cache|engine_crash|low_success_rate|requests_aborted, or "" if unknown.
failure_class_on_host() {
    local host="$1" port="$2" remote_log="$3" oom_hint="$4" detail="$5" log_path sig
    [[ -n "$oom_hint" ]] && { echo "oom_kv_cache"; return; }
    case "${detail,,}" in
        *"success rate"*) echo "low_success_rate"; return;;
    esac
    log_path="$(remote_tmp_root "$host")/vllm_${port}.log"
    sig=$(ssh "$host" "grep -hoiE \"Can't load the configuration|Repo id must be in the form|No available memory for the cache blocks|CUDA out of memory|out of memory|Engine core initialization failed|EngineCore failed to start|Success rate .* below minimum|No requests completed\" '$log_path' \"/tmp/vllm_${port}.log\" '$remote_log' 2>/dev/null | head -1" < /dev/null 2>/dev/null || true)
    case "${sig,,}" in
        *"can't load the configuration"*|*"repo id must be"*) echo "model_missing";;
        *"no available memory for the cache blocks"*|*"cuda out of memory"*|*"out of memory"*) echo "oom_kv_cache";;
        *"engine core initialization failed"*|*"enginecore failed"*) echo "engine_crash";;
        *"success rate"*) echo "low_success_rate";;
        *"no requests completed"*) echo "requests_aborted";;
        *) echo "";;
    esac
}

# Preflight (RFC §4.2): is the model actually staged on the host? Echoes
# present|missing|unknown. Only checks absolute local paths -- HF repo ids
# download on demand so are never "missing" -- and only reports `missing` when
# the SSH succeeded and the dir is genuinely absent (transient SSH errors ->
# unknown, so we never false-flag a reachable model).
model_present_on_host() {
    local host="$1" model_path="$2" out
    [[ "$model_path" == /* ]] || { echo "unknown"; return; }
    out=$(ssh "$host" "if test -d '$model_path' && { test -f '$model_path/config.json' || ls '$model_path'/*.safetensors >/dev/null 2>&1; }; then echo present; else echo missing; fi" < /dev/null 2>/dev/null || echo unknown)
    case "$out" in
        present|missing) echo "$out";;
        *) echo "unknown";;
    esac
}

# Per-cell outcomes (RFC §4.5): parse the bench log for per-(profile,concurrency)
# failures so a job that serves at low concurrency but fails at high concurrency
# records distinct classes. Echoes a JSON map "<profile>|<conc>" -> failure_class
# (only failed cells; succeeded cells have output JSONs). Best-effort; {} on error.
cell_outcomes_on_host() {
    local host="$1" remote_log="$2"
    [[ -n "$remote_log" ]] || { echo "{}"; return; }
    ssh "$host" "cat '$remote_log' 2>/dev/null" < /dev/null 2>/dev/null | python3 -c '
import sys, json, re
blocks, cur = {}, None
for line in sys.stdin:
    m = re.search(r"=== profile=(\S+) conc=(\d+)", line)
    if m:
        cur = (m.group(1), int(m.group(2)))
        blocks.setdefault(cur, [])
        continue
    if cur is not None:
        blocks[cur].append(line)
def classify(text):
    t = text.lower()
    if "success rate" in t and "below min" in t:
        return "low_success_rate"
    if ("no requests completed" in t or "requests failed" in t
            or "server may not be functional" in t):
        return "requests_aborted"
    return "unknown"
out = {}
for (profile, conc), lines in blocks.items():
    low = " ".join(lines).lower()
    if "bench failed for" in low or "abort:" in low:
        out["%s|%d" % (profile, conc)] = classify(low)
print(json.dumps(out))
' 2>/dev/null || echo "{}"
}

state_read_file() {
    local jid="$1" suffix="$2" primary scope candidate
    primary="$STATE_DIR/${jid}.${suffix}"
    if [[ -f "$primary" ]]; then
        echo "$primary"
        return
    fi
    for scope in $(state_scope_aliases "$STATE_SCOPE"); do
        candidate="$STATE_ROOT/$scope/${jid}.${suffix}"
        if [[ "$candidate" != "$primary" && -f "$candidate" ]]; then
            echo "$candidate"
            return
        fi
    done
    for scope in $(state_scope_aliases "$STATE_SCOPE"); do
        candidate="$LEGACY_STATE_ROOT/$scope/${jid}.${suffix}"
        if [[ -f "$candidate" ]]; then
            echo "$candidate"
            return
        fi
    done
    candidate="$LEGACY_STATE_ROOT/${jid}.${suffix}"
    if [[ -f "$candidate" ]]; then
        echo "$candidate"
        return
    fi
    echo "$primary"
}

read_status()  { cat "$(state_read_file "$1" status)" 2>/dev/null || echo "pending"; }
write_state_value() {
    local jid="$1" suffix="$2" value="$3"
    if dry_run; then
        log "$jid: dry-run would write ${suffix}=$value"
    else
        echo "$value" > "$STATE_DIR/${jid}.${suffix}"
    fi
}
remove_state_file() {
    local jid="$1" suffix="$2"
    if dry_run; then
        log "$jid: dry-run would remove ${suffix}"
    else
        rm -f "$STATE_DIR/${jid}.${suffix}"
    fi
}
write_status() { write_state_value "$1" status "$2"; }
read_attempt() { cat "$(state_read_file "$1" attempt)" 2>/dev/null || echo "0"; }
bump_attempt() { local n=$(($(read_attempt "$1") + 1)); write_state_value "$1" attempt "$n"; }
read_signature() { cat "$(state_read_file "$1" signature)" 2>/dev/null || true; }
write_signature() { write_state_value "$1" signature "$2"; }

write_failure_metadata() {
    local jid="$1" status="$2" attempt="$3" max_attempts="$4" present="$5" total="$6"
    local missing_outputs="$7" reason="$8" remote_log="$9" mirror_status="${10}"
    # Structured outcome (RFC: docs/coverage-classification-rfc.md). Optional and
    # additive: when empty, reconcile derives failure_class from the reason text.
    local failure_class="${11:-}" gpu_mem_util="${12:-}" cell_outcomes_json="${13:-}"
    local path="$STATE_DIR/${jid}.failure.json"
    if dry_run; then
        log "$jid: dry-run would write failure metadata status=$status class=${failure_class:-?} attempt=$attempt reason=${reason:0:160}"
        return
    fi
    python3 - "$path" "$jid" "$status" "$attempt" "$max_attempts" "$present" "$total" \
        "$missing_outputs" "$reason" "$remote_log" "$mirror_status" "$failure_class" "$gpu_mem_util" "$cell_outcomes_json" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    path,
    jid,
    status,
    attempt,
    max_attempts,
    present,
    total,
    missing_outputs,
    reason,
    remote_log,
    mirror_status,
    failure_class,
    gpu_mem_util,
    cell_outcomes_json,
) = sys.argv[1:]

def to_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None

def to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

evidence = {
    "outputs_present": to_int(present),
    "outputs_expected": to_int(total),
}
if to_float(gpu_mem_util) is not None:
    evidence["gpu_mem_util"] = to_float(gpu_mem_util)

try:
    cell_outcomes = json.loads(cell_outcomes_json) if cell_outcomes_json else {}
    if not isinstance(cell_outcomes, dict):
        cell_outcomes = {}
except (ValueError, TypeError):
    cell_outcomes = {}

payload = {
    "job_id": jid,
    "status": status,
    "kind": "incomplete_outputs",
    "failure_class": (failure_class or None),
    "evidence": evidence,
    "cell_outcomes": cell_outcomes,
    "attempt": to_int(attempt),
    "max_attempts": to_int(max_attempts),
    "expected_outputs_present": to_int(present),
    "expected_outputs_total": to_int(total),
    "missing_outputs": [part for part in missing_outputs.split() if part],
    "reason": reason,
    "remote_log": remote_log,
    "mirror_status": mirror_status,
    "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}

target = Path(path)
target.parent.mkdir(parents=True, exist_ok=True)
tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, target)
PY
}

generate_run_id() {
    local jid="$1" suffix safe_jid
    safe_jid="$(safe_name "$jid")"
    suffix="$(od -An -N4 -tx1 /dev/urandom 2>/dev/null | tr -d ' \n')"
    [[ -z "$suffix" ]] && suffix="${RANDOM}${RANDOM}"
    echo "run_$(date -u +%Y%m%dT%H%M%SZ)_${safe_jid}_${suffix}"
}

write_run_record() {
    local run_id="$1" jid="$2" status="$3" host="$4" gpus="$5" port="$6" backend="$7"
    local storage_scope="$8" dashboard_scope="$9" model_path="${10}" tp="${11}" short="${12}" mode="${13}"
    local concs="${14}" profiles="${15}" remote_pid="${16}" remote_log="${17}" started_at="${18}"
    local path="$RUNS_DIR/${run_id}.json"
    if dry_run; then
        log "$jid: dry-run would write run lease $path status=$status run_id=$run_id"
        return
    fi
    mkdir -p "$RUNS_DIR"
    python3 - "$path" "$run_id" "$jid" "$status" "$host" "$gpus" "$port" "$backend" \
        "$storage_scope" "$dashboard_scope" "$model_path" "$tp" "$short" "$mode" \
        "$concs" "$profiles" "$remote_pid" "$remote_log" "$started_at" <<'PY'
import json
import os
import sys
from pathlib import Path

(
    path,
    run_id,
    jid,
    status,
    host,
    gpus,
    port,
    backend,
    storage_scope,
    dashboard_scope,
    model_path,
    tp,
    short,
    mode,
    concs,
    profiles,
    remote_pid,
    remote_log,
    started_at,
) = sys.argv[1:]

payload = {
    "run_id": run_id,
    "job_id": jid,
    "status": status,
    "host": host,
    "gpus": [part for part in gpus.replace(",", " ").split() if part],
    "port": port,
    "backend": backend,
    "storage_scope": storage_scope,
    "dashboard_scope": dashboard_scope,
    "model_path": model_path,
    "model_short": short,
    "tp": int(tp),
    "mode": mode,
    "concurrencies": [int(part) for part in concs.split() if part.isdigit()],
    "profiles": [part for part in profiles.split() if part],
    "remote_launcher_pid": remote_pid,
    "remote_log": remote_log,
    "started_at": started_at,
}

target = Path(path)
target.parent.mkdir(parents=True, exist_ok=True)
tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, target)
PY
}

update_run_record_status() {
    local run_id="$1" jid="$2" status="$3" reason="$4" attempt="$5" present="$6" total="$7" missing_outputs="$8" remote_log="$9"
    [[ -n "$run_id" ]] || return 0
    local path="$RUNS_DIR/${run_id}.json"
    if dry_run; then
        log "$jid: dry-run would update run lease $path status=$status"
        return
    fi
    python3 - "$path" "$run_id" "$jid" "$status" "$reason" "$attempt" "$present" "$total" "$missing_outputs" "$remote_log" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path, run_id, jid, status, reason, attempt, present, total, missing_outputs, remote_log = sys.argv[1:]

def to_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None

target = Path(path)
try:
    payload = json.loads(target.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    payload = {"run_id": run_id, "job_id": jid}

payload["run_id"] = payload.get("run_id") or run_id
payload["job_id"] = payload.get("job_id") or jid
payload["status"] = status
payload["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
if status not in {"dispatching", "running"}:
    payload["finalized_at"] = payload["updated_at"]
if reason:
    payload["failure"] = {
        "reason": reason,
        "attempt": to_int(attempt),
        "expected_outputs_present": to_int(present),
        "expected_outputs_total": to_int(total),
        "missing_outputs": [part for part in missing_outputs.split() if part],
        "remote_log": remote_log,
    }

target.parent.mkdir(parents=True, exist_ok=True)
tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, target)
PY
}

update_current_run_record_status() {
    local jid="$1" status="$2" reason="$3" attempt="$4" present="$5" total="$6" missing_outputs="$7" remote_log="$8"
    local run_id
    run_id=$(cat "$(state_read_file "$jid" run_id)" 2>/dev/null || true)
    update_run_record_status "$run_id" "$jid" "$status" "$reason" "$attempt" "$present" "$total" "$missing_outputs" "$remote_log"
}

# Phase 1: multi-slot GPU scheduling — scan per-GPU and per-port usage.
declare -A HOST_GPU_COUNT=( [a100]=8 [3090]=8 [2080ti]=8 [h100]=8 [h100-2]=4 )
PORT_RANGE=(8089 8090 8091 8092 8093 8094 8095 8096)
GPU_BUSY_MEM_MIB=${BENCH_GPU_BUSY_MEM_MIB:-512}
[[ "$GPU_BUSY_MEM_MIB" =~ ^[0-9]+$ ]] || GPU_BUSY_MEM_MIB=512
declare -A HOST_USED_GPUS
declare -A HOST_USED_PORTS
declare -A HOST_OBSERVED_GPUS
declare -A HOST_OBSERVED_PORTS
declare -A HOST_PORT_CMDS
declare -A HOST_DRAIN_LOGGED
declare -A HOST_BLOCKED_GPUS

while IFS='=' read -r host count; do
    [[ -n "$host" && "$count" =~ ^[0-9]+$ ]] && HOST_GPU_COUNT[$host]="$count"
done < <(python3 "$REPO_ROOT/inference-benchmark/scripts/compile_sweep.py" --yaml "$JOBS_CONFIG" --list-host-gpu-counts 2>/dev/null || true)

load_blocked_gpus

if truthy "$SKIP_REMOTE_PROBE"; then
    log "remote slot probing disabled by BENCH_ORCHESTRATOR_SKIP_REMOTE_PROBE"
else
    for HOST in "${JOB_HOSTS[@]}"; do
        SLOT_INFO=$(ssh -o ConnectTimeout=5 -o BatchMode=yes "$HOST" '
            busy_mem_threshold='"$GPU_BUSY_MEM_MIB"'
            echo "GPUS:$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F", " -v threshold="$busy_mem_threshold" "\$2 > threshold {printf \$1\" \"}")"
            ports=""
            for p in 8089 8090 8091 8092 8093 8094 8095 8096; do
                line=$(ss -ltnp 2>/dev/null | awk -v port="$p" "{ n=split(\$4, parts, \":\"); if (parts[n] == port) { print; exit } }")
                if [ -n "$line" ]; then
                    ports="${ports}${p} "
                    pid=$(printf "%s\n" "$line" | sed -n "s/.*pid=\([0-9][0-9]*\).*/\1/p" | head -1)
                    cmd=""
                    if [ -n "$pid" ]; then
                        cmd=$(ps -o args= -p "$pid" 2>/dev/null | head -1)
                    fi
                    printf "PORTCMD:%s|%s\n" "$p" "$cmd"
                fi
            done
            echo "PORTS:${ports}"
        ' 2>/dev/null || true)
        HOST_USED_GPUS[$HOST]=$(echo "$SLOT_INFO" | grep "^GPUS:" | sed 's/^GPUS://')
        HOST_USED_PORTS[$HOST]=$(echo "$SLOT_INFO" | grep "^PORTS:" | sed 's/^PORTS://')
        HOST_OBSERVED_GPUS[$HOST]="${HOST_USED_GPUS[$HOST]:-}"
        HOST_OBSERVED_PORTS[$HOST]="${HOST_USED_PORTS[$HOST]:-}"
        while IFS= read -r port_line; do
            port_part="${port_line#PORTCMD:}"
            port="${port_part%%|*}"
            cmd="${port_part#*|}"
            [[ -n "$port" ]] && HOST_PORT_CMDS["$HOST:$port"]="$cmd"
        done < <(echo "$SLOT_INFO" | grep "^PORTCMD:")
    done
fi

for HOST in "${JOB_HOSTS[@]}"; do
    if [[ -n "${HOST_BLOCKED_GPUS[$HOST]:-}" ]]; then
        HOST_USED_GPUS[$HOST]="${HOST_USED_GPUS[$HOST]:-} ${HOST_BLOCKED_GPUS[$HOST]}"
    fi
    log "slots $HOST: used_gpus=[${HOST_USED_GPUS[$HOST]:-}] blocked_gpus=[${HOST_BLOCKED_GPUS[$HOST]:-}] used_ports=[${HOST_USED_PORTS[$HOST]:-}]"
done

find_free_gpus() {
    local host="$1" needed="$2"
    local total=${HOST_GPU_COUNT[$host]:-0}
    local used=" ${HOST_USED_GPUS[$host]:-} "
    local free=()
    for ((i=0; i<total; i++)); do
        [[ "$used" == *" $i "* ]] && continue
        free+=("$i")
        [[ ${#free[@]} -ge $needed ]] && break
    done
    [[ ${#free[@]} -ge $needed ]] && { IFS=,; echo "${free[*]}"; } || true
}

find_free_port() {
    local host="$1"
    local used=" ${HOST_USED_PORTS[$host]:-} "
    for p in "${PORT_RANGE[@]}"; do
        [[ "$used" == *" $p "* ]] && continue
        echo "$p"
        return
    done
}

claim_slot() {
    local host="$1" gpus="$2" port="$3"
    HOST_USED_GPUS[$host]="${HOST_USED_GPUS[$host]:-} ${gpus//,/ } "
    HOST_USED_PORTS[$host]="${HOST_USED_PORTS[$host]:-} $port "
}

release_unobserved_slot() {
    local host="$1" gpus="$2" port="$3"
    local used_gpus=" ${HOST_USED_GPUS[$host]:-} "
    local observed_gpus=" ${HOST_OBSERVED_GPUS[$host]:-} ${HOST_BLOCKED_GPUS[$host]:-} "
    local g
    for g in ${gpus//,/ }; do
        [[ -z "$g" ]] && continue
        [[ "$observed_gpus" == *" $g "* ]] && continue
        used_gpus="${used_gpus// $g / }"
    done
    HOST_USED_GPUS[$host]="$used_gpus"

    local used_ports=" ${HOST_USED_PORTS[$host]:-} "
    local observed_ports=" ${HOST_OBSERVED_PORTS[$host]:-} "
    if [[ -n "$port" && "$observed_ports" != *" $port "* ]]; then
        used_ports="${used_ports// $port / }"
    fi
    HOST_USED_PORTS[$host]="$used_ports"
}

job_warmup_timeout() {
    local backend="$1"
    if [[ "$backend" == "sglang" ]]; then
        echo 900
    else
        echo 600
    fi
}

port_matches_job() {
    local host="$1" port="$2" model_path="$3"
    local cmd="${HOST_PORT_CMDS["$host:$port"]:-}"
    local model_base="${model_path##*/}"
    [[ -z "$cmd" ]] && return 1
    [[ -n "$model_path" && "$cmd" == *"$model_path"* ]] && return 0
    [[ -n "$model_base" && "$cmd" == *"$model_base"* ]] && return 0
    return 1
}

port_owner_summary() {
    local host="$1" port="$2"
    local cmd="${HOST_PORT_CMDS["$host:$port"]:-}"
    if [[ -z "$cmd" ]]; then
        echo "no listener command captured"
    else
        echo "${cmd:0:160}"
    fi
}

# Reserve recently dispatched running jobs from state before considering
# pending rows. This protects jobs that are still loading and have not opened a
# port or allocated noticeable GPU memory yet.
while IFS='|' read -r HOST MODEL_PATH TP SHORT MODE BACKEND MAX_LEN GPU_MEM CONCS PROFILES EXTRA_ENV || [[ -n "$HOST" ]]; do
    HOST=$(echo "$HOST" | tr -d ' ')
    [[ -z "$HOST" || "${HOST:0:1}" == "#" ]] && continue
    : "${BACKEND:=vllm}"
    JID=$(job_id "$HOST" "$SHORT" "$TP" "$MODE" "$BACKEND")
    STATUS=$(read_status "$JID")
    [[ "$STATUS" != "running" ]] && continue

    JOB_PORT=$(cat "$(state_read_file "$JID" port)" 2>/dev/null || true)
    JOB_GPUS=$(cat "$(state_read_file "$JID" gpus)" 2>/dev/null || true)
    [[ -z "$JOB_PORT$JOB_GPUS" ]] && continue

    STATUS_FILE="$(state_read_file "$JID" status)"
    AGE=999999999
    [[ -f "$STATUS_FILE" ]] && AGE=$(( $(date +%s) - $(stat -c %Y "$STATUS_FILE") ))
    WARMUP_TIMEOUT=$(job_warmup_timeout "$BACKEND")

    if ! truthy "$SKIP_REMOTE_PROBE"; then
        if [[ -n "$JOB_PORT" && " ${HOST_OBSERVED_PORTS[$HOST]:-} " == *" $JOB_PORT "* ]]; then
            if ! port_matches_job "$HOST" "$JOB_PORT" "$MODEL_PATH"; then
                log "$JID: not reserving stale recorded slot on $HOST:$JOB_PORT; listener belongs to $(port_owner_summary "$HOST" "$JOB_PORT")"
                continue
            fi
        elif [[ "$AGE" -ge "$WARMUP_TIMEOUT" ]]; then
            log "$JID: not reserving old recorded slot on $HOST port=$JOB_PORT gpus=[$JOB_GPUS] age=${AGE}s; no matching listener"
            continue
        fi
    fi

    claim_slot "$HOST" "$JOB_GPUS" "$JOB_PORT"
    log "$JID: reserving recorded running slot on $HOST port=$JOB_PORT gpus=[$JOB_GPUS] age=${AGE}s"
done < "$JOBS_FILE"

# Phase 2: scan jobs, decide actions.
while IFS='|' read -r HOST MODEL_PATH TP SHORT MODE BACKEND MAX_LEN GPU_MEM CONCS PROFILES EXTRA_ENV || [[ -n "$HOST" ]]; do
    HOST=$(echo "$HOST" | tr -d ' ')
    [[ -z "$HOST" || "${HOST:0:1}" == "#" ]] && continue

    : "${BACKEND:=vllm}"  # default if column missing (legacy rows)
    ROW_RESULT_SCOPE=$(row_result_scope "$EXTRA_ENV")
    ROW_DASHBOARD_SCOPE=$(dashboard_scope_for "$ROW_RESULT_SCOPE")
    ROW_STORAGE_SCOPE=$(storage_scope_for "$ROW_RESULT_SCOPE")
    JID=$(job_id "$HOST" "$SHORT" "$TP" "$MODE" "$BACKEND")
    STATUS=$(read_status "$JID")
    JOB_SIGNATURE="${ROW_STORAGE_SCOPE}|${ROW_DASHBOARD_SCOPE}|${MAX_LEN}|${GPU_MEM}|${CONCS}|${PROFILES}|${EXTRA_ENV}"
    OLD_SIGNATURE=$(read_signature "$JID")
    if [[ "$STATUS" =~ ^(done|skipped|failed)$ && -n "$OLD_SIGNATURE" && "$OLD_SIGNATURE" != "$JOB_SIGNATURE" ]]; then
        log "$JID: job shape changed since terminal $STATUS; retrying as pending"
        STATUS="pending"
        write_status "$JID" pending
        write_state_value "$JID" attempt "0"
        remove_state_file "$JID" max_len_override
        remove_state_file "$JID" reason
        remove_state_file "$JID" failure.json
    elif [[ "$STATUS" =~ ^(skipped|failed)$ && -z "$OLD_SIGNATURE" && "$MODE" == "multi" && "$PROFILES" != *"swebench-multiturn"* && "$PROFILES" != *"terminalbench-multiturn"* ]]; then
        log "$JID: legacy terminal $STATUS predates profile filtering; retrying reduced-profile job"
        STATUS="pending"
        write_status "$JID" pending
        write_state_value "$JID" attempt "0"
        remove_state_file "$JID" max_len_override
        remove_state_file "$JID" reason
        remove_state_file "$JID" failure.json
    fi
    PREFIX=$(host_prefix "$HOST")
    RESULT_DIR_NAME="${PREFIX}_${SHORT}_tp${TP}_${BACKEND}"
    REMOTE_TMP_FOR_HOST="$(remote_tmp_root "$HOST")"
    REMOTE_CODE_FOR_HOST="$(remote_code_root "$HOST")"
    REMOTE_RESULTS_FOR_HOST="$(remote_results_root "$HOST")"
    OUT_DIR_REMOTE="$REMOTE_RESULTS_FOR_HOST/${ROW_STORAGE_SCOPE}/${RESULT_DIR_NAME}"
    # Fallback for jobs launched before RESULT_SCOPE existed; completed jobs
    # still upload into the normalized R2 namespace to avoid legacy collisions.
    LEGACY_SCOPED_OUT_DIR_REMOTE="/tmp/results/${ROW_STORAGE_SCOPE}/${RESULT_DIR_NAME}"
    LEGACY_OUT_DIR_REMOTE="/tmp/results/${RESULT_DIR_NAME}"
    R2_DIR="${ROW_STORAGE_SCOPE}/${RESULT_DIR_NAME}"
    OUT_DIR_LOCAL="$LOCAL_RESULTS_ROOT/$R2_DIR"
    RUN_MAX_LEN="$MAX_LEN"
    OVERRIDE_FILE="$(state_read_file "$JID" max_len_override)"
    if [[ -f "$OVERRIDE_FILE" ]]; then
        RUN_MAX_LEN=$(cat "$OVERRIDE_FILE")
    fi
    if [[ "$STATUS" == "pending" ]]; then
        expected_output_summary "$OUT_DIR_LOCAL" "$SHORT" "$TP" "$BACKEND" "$MODE" "$CONCS" "$PROFILES"
        if [[ "$EXPECTED_OUTPUT_TOTAL" -gt 0 && "$EXPECTED_OUTPUT_PRESENT" -eq "$EXPECTED_OUTPUT_TOTAL" ]]; then
            if dry_run; then
                log "$JID: dry-run would mark DONE from local cache ($EXPECTED_OUTPUT_PRESENT/$EXPECTED_OUTPUT_TOTAL expected outputs)"
            else
                write_signature "$JID" "$JOB_SIGNATURE"
                write_status "$JID" done
                remove_state_file "$JID" reason
                remove_state_file "$JID" failure.json
                log "$JID: DONE from local cache ($EXPECTED_OUTPUT_PRESENT/$EXPECTED_OUTPUT_TOTAL expected outputs); skipping dispatch"
            fi
            continue
        fi
    fi

    case "$STATUS" in
        done|skipped|failed)
            continue
            ;;
        running)
            JOB_PORT=$(cat "$(state_read_file "$JID" port)" 2>/dev/null || echo "8089")
            JOB_GPUS=$(cat "$(state_read_file "$JID" gpus)" 2>/dev/null || true)
            JOB_REMOTE_LOG="$REMOTE_TMP_FOR_HOST/bench_${SHORT}_tp${TP}_${MODE}_${BACKEND}_p${JOB_PORT}.log"
            EXPECTED_OUTPUT_TOTAL=0
            EXPECTED_OUTPUT_PRESENT=0
            EXPECTED_OUTPUT_MISSING_SAMPLE=""
            EXPECTED_OUTPUT_MISSING_ALL=""
            RELEASE_RECORDED_SLOT=0
            if [[ " ${HOST_OBSERVED_PORTS[$HOST]:-} " == *" $JOB_PORT "* ]]; then
                if port_matches_job "$HOST" "$JOB_PORT" "$MODEL_PATH"; then
                    if sweep_log_done_on_host "$HOST" "$JOB_REMOTE_LOG"; then
                        log "$JID: sweep log $JOB_REMOTE_LOG is complete but $HOST:$JOB_PORT still listens; finalizing"
                    else
                        log "$JID: still running on $HOST:$JOB_PORT"
                        continue
                    fi
                else
                    log "$JID: recorded port $HOST:$JOB_PORT is held by a different command ($(port_owner_summary "$HOST" "$JOB_PORT")); finalizing stale state"
                fi
            fi
            # Grace period: weight-load + CUDA graph compilation.
            # vllm: ~3-5 min typical, 10 min max.
            # sglang: aggressive torch compilation, 10-15 min for large/MoE models.
            WARMUP_TIMEOUT=$(job_warmup_timeout "$BACKEND")
            STATUS_FILE="$(state_read_file "$JID" status)"
            AGE=0
            if [[ -f "$STATUS_FILE" ]]; then
                AGE=$(( $(date +%s) - $(stat -c %Y "$STATUS_FILE") ))
                if [[ "$AGE" -lt "$WARMUP_TIMEOUT" ]]; then
                    if [[ "$BACKEND" == "sglang" ]]; then
                        SCRIPT_NAME="sweep_all_profiles_sglang.sh"
                        [[ "$MODE" == "multi" ]] && SCRIPT_NAME="sweep_multiturn_profiles_sglang.sh"
                    else
                        SCRIPT_NAME="sweep_all_profiles.sh"
                        [[ "$MODE" == "multi" ]] && SCRIPT_NAME="sweep_multiturn_profiles.sh"
                    fi
                    if truthy "$SKIP_REMOTE_PROBE"; then
                        REMOTE_SCRIPT_ALIVE=""
                    else
                        REMOTE_SCRIPT_ALIVE=$(ssh "$HOST" "ps -eo args= | awk -v script='$REMOTE_CODE_FOR_HOST/scripts/${SCRIPT_NAME}' -v needle=' ${TP} ${SHORT} ${BACKEND} ' -v concs=' ${CONCS} ' '\$1 == \"bash\" && \$2 == script && index(\$0, needle) && index(\$0, concs) { found=1 } END { exit found ? 0 : 1 }' && echo yes" < /dev/null 2>/dev/null || true)
                    fi
                    if [[ "$REMOTE_SCRIPT_ALIVE" == "yes" ]]; then
                        log "$JID: dispatched ${AGE}s ago (<$(( WARMUP_TIMEOUT / 60 ))min), still warming up on port $JOB_PORT"
                        JOB_GPUS=$(cat "$(state_read_file "$JID" gpus)" 2>/dev/null || true)
                        [[ -n "$JOB_GPUS" ]] && HOST_USED_GPUS[$HOST]="${HOST_USED_GPUS[$HOST]:-} ${JOB_GPUS//,/ } "
                        HOST_USED_PORTS[$HOST]="${HOST_USED_PORTS[$HOST]:-} $JOB_PORT "
                        continue
                    fi
                    log "$JID: no listener and no live sweep process after ${AGE}s; finalizing early"
                fi
            fi
            if [[ " ${HOST_OBSERVED_PORTS[$HOST]:-} " != *" $JOB_PORT "* ]]; then
                RELEASE_RECORDED_SLOT=1
            fi
            log "$JID: slot idle after ${AGE}s warmup ($BACKEND) — finalizing"
            if dry_run; then
                log "$JID: dry-run would inspect remote outputs and update terminal state"
                continue
            fi
            # All ssh/rsync/aws calls inside this `while read ... done <JOBS`
            # loop must close stdin (`< /dev/null`), otherwise they consume
            # the jobs file and iteration ends early — 2080ti rows were
            # silently skipped on any tick that also dispatched a 3090 job.
            REMOTE_SYNC_DIR=$(ssh "$HOST" "for d in '$OUT_DIR_REMOTE' '$LEGACY_SCOPED_OUT_DIR_REMOTE' '$LEGACY_OUT_DIR_REMOTE'; do if [ -d \"\$d\" ] && [ \$(ls -1 \"\$d\" 2>/dev/null | wc -l) -gt 0 ]; then echo \"\$d\"; break; fi; done" < /dev/null)
            if [[ -n "$REMOTE_SYNC_DIR" ]]; then
                COUNT=$(ssh "$HOST" "ls '$REMOTE_SYNC_DIR' 2>/dev/null | wc -l" < /dev/null)
                mkdir -p "$OUT_DIR_LOCAL"
                if rsync -az "$HOST:$REMOTE_SYNC_DIR/" "$OUT_DIR_LOCAL/" < /dev/null >> "$LOG" 2>&1; then
                    expected_output_summary "$OUT_DIR_LOCAL" "$SHORT" "$TP" "$BACKEND" "$MODE" "$CONCS" "$PROFILES"
                    MIRROR_STATUS="not_mirrored"
                    if [[ "$EXPECTED_OUTPUT_PRESENT" -gt 0 ]]; then
                        if aws --profile "$PROFILE" --endpoint-url "$EP" s3 sync \
                            "$OUT_DIR_LOCAL/" "s3://$BUCKET/results/$R2_DIR/" < /dev/null >> "$LOG" 2>&1; then
                            MIRROR_STATUS="r2_mirrored"
                        else
                            MIRROR_STATUS="r2_mirror_failed"
                        fi
                    fi
                    if [[ "$EXPECTED_OUTPUT_PRESENT" -eq "$EXPECTED_OUTPUT_TOTAL" ]]; then
                        write_status "$JID" done
                        remove_state_file "$JID" reason
                        remove_state_file "$JID" failure.json
                        update_current_run_record_status "$JID" "done" "" "$(read_attempt "$JID")" "$EXPECTED_OUTPUT_PRESENT" "$EXPECTED_OUTPUT_TOTAL" "" "$JOB_REMOTE_LOG"
                        log "$JID: DONE ($EXPECTED_OUTPUT_PRESENT/$EXPECTED_OUTPUT_TOTAL expected outputs; $COUNT files copied to $OUT_DIR_LOCAL, warmup=${AGE}s backend=$BACKEND mirror=$MIRROR_STATUS)"
                    elif [[ "$EXPECTED_OUTPUT_PRESENT" -gt 0 ]]; then
                        ATT=$(read_attempt "$JID")
                        NEXT_ATT=$((ATT + 1))
                        FAILURE_DETAIL=$(failure_log_summary_on_host "$HOST" "$JOB_REMOTE_LOG")
                        [[ -z "$FAILURE_DETAIL" ]] && FAILURE_DETAIL="expected outputs missing after sweep completion"
                        if [[ -n "${EXPECTED_OUTPUT_MISSING_SAMPLE:-}" ]]; then
                            FAILURE_DETAIL="$FAILURE_DETAIL; missing=${EXPECTED_OUTPUT_MISSING_SAMPLE}"
                        fi
                        write_state_value "$JID" attempt "$NEXT_ATT"
                        if [[ "$NEXT_ATT" -ge "$MAX_INCOMPLETE_RETRIES" ]]; then
                            REASON="retry limit reached after ${NEXT_ATT}/${MAX_INCOMPLETE_RETRIES} incomplete attempts: $FAILURE_DETAIL"
                            write_state_value "$JID" reason "$REASON"
                            FAILURE_CLASS=$(failure_class_on_host "$HOST" "$JOB_PORT" "$JOB_REMOTE_LOG" "${OOM:-}" "${FAILURE_DETAIL:-}")
                            CELL_OUTCOMES=$(cell_outcomes_on_host "$HOST" "$JOB_REMOTE_LOG")
                            write_failure_metadata "$JID" "skipped" "$NEXT_ATT" "$MAX_INCOMPLETE_RETRIES" "$EXPECTED_OUTPUT_PRESENT" "$EXPECTED_OUTPUT_TOTAL" "${EXPECTED_OUTPUT_MISSING_ALL:-}" "$REASON" "$JOB_REMOTE_LOG" "$MIRROR_STATUS" "$FAILURE_CLASS" "${GPU_MEM:-}" "$CELL_OUTCOMES"
                            write_status "$JID" skipped
                            update_current_run_record_status "$JID" "skipped" "$REASON" "$NEXT_ATT" "$EXPECTED_OUTPUT_PRESENT" "$EXPECTED_OUTPUT_TOTAL" "${EXPECTED_OUTPUT_MISSING_ALL:-}" "$JOB_REMOTE_LOG"
                            log "$JID: SKIPPED incomplete retry limit ($EXPECTED_OUTPUT_PRESENT/$EXPECTED_OUTPUT_TOTAL expected outputs; attempt=$NEXT_ATT/$MAX_INCOMPLETE_RETRIES; missing=${EXPECTED_OUTPUT_MISSING_SAMPLE:-unknown}; $COUNT files copied to $OUT_DIR_LOCAL, warmup=${AGE}s backend=$BACKEND mirror=$MIRROR_STATUS)"
                        else
                            REASON="incomplete attempt ${NEXT_ATT}/${MAX_INCOMPLETE_RETRIES}: $FAILURE_DETAIL"
                            write_state_value "$JID" reason "$REASON"
                            write_failure_metadata "$JID" "pending" "$NEXT_ATT" "$MAX_INCOMPLETE_RETRIES" "$EXPECTED_OUTPUT_PRESENT" "$EXPECTED_OUTPUT_TOTAL" "${EXPECTED_OUTPUT_MISSING_ALL:-}" "$REASON" "$JOB_REMOTE_LOG" "$MIRROR_STATUS"
                            write_status "$JID" pending
                            update_current_run_record_status "$JID" "pending" "$REASON" "$NEXT_ATT" "$EXPECTED_OUTPUT_PRESENT" "$EXPECTED_OUTPUT_TOTAL" "${EXPECTED_OUTPUT_MISSING_ALL:-}" "$JOB_REMOTE_LOG"
                            log "$JID: INCOMPLETE ($EXPECTED_OUTPUT_PRESENT/$EXPECTED_OUTPUT_TOTAL expected outputs; attempt=$NEXT_ATT/$MAX_INCOMPLETE_RETRIES; missing=${EXPECTED_OUTPUT_MISSING_SAMPLE:-unknown}; $COUNT files copied to $OUT_DIR_LOCAL, warmup=${AGE}s backend=$BACKEND mirror=$MIRROR_STATUS); retry pending"
                        fi
                    else
                        OOM=$(oom_log_on_host "$HOST" "$JOB_PORT")
                        ATT=$(read_attempt "$JID")
                        if can_retry_oom "$OOM" "$ATT" "$RUN_MAX_LEN"; then
                            HINT=$(oom_max_len_hint_on_host "$HOST" "$JOB_PORT")
                            bump_attempt "$JID"
                            NEXT_ATT=$(read_attempt "$JID")
                            write_status "$JID" pending
                            NEW_MAX=$(next_oom_max_len "$RUN_MAX_LEN" "$HINT")
                            write_state_value "$JID" max_len_override "$NEW_MAX"
                            REASON="OOM/KV-cache failure; retrying with max_len=$NEW_MAX${HINT:+ (vllm estimate=$HINT)}"
                            write_state_value "$JID" reason "$REASON"
                            update_current_run_record_status "$JID" "pending" "$REASON" "$NEXT_ATT" "$EXPECTED_OUTPUT_PRESENT" "$EXPECTED_OUTPUT_TOTAL" "${EXPECTED_OUTPUT_MISSING_ALL:-}" "$JOB_REMOTE_LOG"
                            log "$JID: zero expected outputs after copying $COUNT files; OOM detected, retry with max_len=$NEW_MAX${HINT:+ (vllm estimate=$HINT)}"
                        else
                            write_status "$JID" skipped
                            REASON="zero expected outputs and retry limit exhausted or no retryable OOM; attempt=$ATT oom_log=$OOM"
                            write_state_value "$JID" reason "$REASON"
                            FAILURE_CLASS=$(failure_class_on_host "$HOST" "$JOB_PORT" "$JOB_REMOTE_LOG" "${OOM:-}" "${FAILURE_DETAIL:-}")
                            CELL_OUTCOMES=$(cell_outcomes_on_host "$HOST" "$JOB_REMOTE_LOG")
                            write_failure_metadata "$JID" "skipped" "$ATT" "$MAX_OOM_RETRIES" "$EXPECTED_OUTPUT_PRESENT" "$EXPECTED_OUTPUT_TOTAL" "${EXPECTED_OUTPUT_MISSING_ALL:-}" "$REASON" "$JOB_REMOTE_LOG" "$MIRROR_STATUS" "$FAILURE_CLASS" "${GPU_MEM:-}" "$CELL_OUTCOMES"
                            update_current_run_record_status "$JID" "skipped" "$REASON" "$ATT" "$EXPECTED_OUTPUT_PRESENT" "$EXPECTED_OUTPUT_TOTAL" "${EXPECTED_OUTPUT_MISSING_ALL:-}" "$JOB_REMOTE_LOG"
                            log "$JID: SKIPPED (0/$EXPECTED_OUTPUT_TOTAL expected outputs; copied only stale/non-matching files from $REMOTE_SYNC_DIR; attempt=$ATT, oom_log=$OOM)"
                        fi
                    fi
                else
                    write_status "$JID" pending
                    REASON="local rsync failed for $REMOTE_SYNC_DIR -> $OUT_DIR_LOCAL"
                    write_state_value "$JID" reason "$REASON"
                    update_current_run_record_status "$JID" "pending" "$REASON" "$(read_attempt "$JID")" "0" "$EXPECTED_OUTPUT_TOTAL" "${EXPECTED_OUTPUT_MISSING_ALL:-}" "$JOB_REMOTE_LOG"
                    log "$JID: local rsync failed for $REMOTE_SYNC_DIR -> $OUT_DIR_LOCAL; leaving pending"
                fi
            else
                # Widen detection to include vLLM's KV-cache budget failures,
                # which are reported as ValueError rather than torch OOM.
                OOM=$(oom_log_on_host "$HOST" "$JOB_PORT")
                ATT=$(read_attempt "$JID")
                if can_retry_oom "$OOM" "$ATT" "$RUN_MAX_LEN"; then
                    HINT=$(oom_max_len_hint_on_host "$HOST" "$JOB_PORT")
                    bump_attempt "$JID"
                    NEXT_ATT=$(read_attempt "$JID")
                    write_status "$JID" pending
                    NEW_MAX=$(next_oom_max_len "$RUN_MAX_LEN" "$HINT")
                    write_state_value "$JID" max_len_override "$NEW_MAX"
                    REASON="OOM/KV-cache failure before result directory; retrying with max_len=$NEW_MAX${HINT:+ (vllm estimate=$HINT)}"
                    write_state_value "$JID" reason "$REASON"
                    update_current_run_record_status "$JID" "pending" "$REASON" "$NEXT_ATT" "0" "$EXPECTED_OUTPUT_TOTAL" "${EXPECTED_OUTPUT_MISSING_ALL:-}" "$JOB_REMOTE_LOG"
                    log "$JID: OOM detected, retry with max_len=$NEW_MAX${HINT:+ (vllm estimate=$HINT)}"
                else
                    write_status "$JID" skipped
                    REASON="zero results and retry limit exhausted or no retryable OOM; attempt=$ATT oom_log=$OOM"
                    write_state_value "$JID" reason "$REASON"
                    FAILURE_CLASS=$(failure_class_on_host "$HOST" "$JOB_PORT" "$JOB_REMOTE_LOG" "${OOM:-}" "${FAILURE_DETAIL:-}")
                    CELL_OUTCOMES=$(cell_outcomes_on_host "$HOST" "$JOB_REMOTE_LOG")
                    write_failure_metadata "$JID" "skipped" "$ATT" "$MAX_OOM_RETRIES" "0" "$EXPECTED_OUTPUT_TOTAL" "${EXPECTED_OUTPUT_MISSING_ALL:-}" "$REASON" "$JOB_REMOTE_LOG" "not_mirrored" "$FAILURE_CLASS" "${GPU_MEM:-}" "$CELL_OUTCOMES"
                    update_current_run_record_status "$JID" "skipped" "$REASON" "$ATT" "0" "$EXPECTED_OUTPUT_TOTAL" "${EXPECTED_OUTPUT_MISSING_ALL:-}" "$JOB_REMOTE_LOG"
                    log "$JID: SKIPPED (zero results, attempt=$ATT, oom_log=$OOM)"
                fi
            fi
            if [[ "$RELEASE_RECORDED_SLOT" == "1" ]]; then
                release_unobserved_slot "$HOST" "$JOB_GPUS" "$JOB_PORT"
                log "$JID: released unobserved recorded slot on $HOST port=$JOB_PORT gpus=[$JOB_GPUS]"
            fi
            ;;
        pending)
            if host_drained "$HOST"; then
                if [[ -z "${HOST_DRAIN_LOGGED[$HOST]:-}" ]]; then
                    log "host $HOST is drained; preserving running jobs but skipping new dispatches"
                    HOST_DRAIN_LOGGED[$HOST]=1
                fi
                continue
            fi

            if [[ "$MAX_DISPATCHES" =~ ^[0-9]+$ && "$MAX_DISPATCHES" -gt 0 && "$DISPATCHES" -ge "$MAX_DISPATCHES" ]]; then
                continue
            fi

            # Preflight (RFC §4.2): don't claim a GPU slot or launch a server for
            # a model that isn't staged. Emit a structured model_missing outcome
            # and skip -- the dashboard surfaces "model not staged" instead of a
            # generic "zero results" after a wasted launch.
            if [[ "$(model_present_on_host "$HOST" "$MODEL_PATH")" == "missing" ]]; then
                REASON="preflight: model not staged at $MODEL_PATH on $HOST"
                write_state_value "$JID" reason "$REASON"
                write_failure_metadata "$JID" "skipped" "0" "$MAX_OOM_RETRIES" "0" "0" "" "$REASON" "" "not_mirrored" "model_missing" "${GPU_MEM:-}"
                write_status "$JID" skipped
                log "$JID: PREFLIGHT model_missing — $MODEL_PATH absent on $HOST; skipping dispatch (no server launched)"
                continue
            fi

            # Extract explicit CUDA_VISIBLE_DEVICES from extra_env if present
            CELL_CVD=""
            if [[ "$EXTRA_ENV" == *CUDA_VISIBLE_DEVICES=* ]]; then
                CELL_CVD=$(echo "$EXTRA_ENV" | sed -n 's/.*CUDA_VISIBLE_DEVICES=\([^ ]*\).*/\1/p')
            fi

            if [[ -n "$CELL_CVD" ]]; then
                # Treat legacy CUDA_VISIBLE_DEVICES rows as preferred placement.
                # If the requested devices are busy and flexible placement is
                # enabled, fall back to any free GPU set of the same TP width.
                SLOT_FREE=1
                USED=" ${HOST_USED_GPUS[$HOST]:-} "
                for g in ${CELL_CVD//,/ }; do
                    [[ "$USED" == *" $g "* ]] && SLOT_FREE=0 && break
                done
                if [[ "$SLOT_FREE" == "1" ]]; then
                    SLOT_GPUS="$CELL_CVD"
                elif truthy "$FLEXIBLE_PINNED_GPUS"; then
                    SLOT_GPUS=$(find_free_gpus "$HOST" "$TP")
                    [[ -z "$SLOT_GPUS" ]] && continue
                    log "$JID: preferred CUDA_VISIBLE_DEVICES=[$CELL_CVD] busy; flexing to [$SLOT_GPUS]"
                else
                    continue
                fi
            else
                SLOT_GPUS=$(find_free_gpus "$HOST" "$TP")
                [[ -z "$SLOT_GPUS" ]] && continue
            fi

            SLOT_PORT=$(find_free_port "$HOST")
            [[ -z "$SLOT_PORT" ]] && continue
            RUN_ID=$(generate_run_id "$JID")
            RUN_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

            PY=$(extra_env_value "PYTHON_BIN" "$EXTRA_ENV")
            [[ -z "$PY" ]] && PY=$(host_python "$HOST" "$BACKEND")
            if [[ "$BACKEND" == "sglang" ]]; then
                SCRIPT="sweep_all_profiles_sglang.sh"
                [[ "$MODE" == "multi" ]] && SCRIPT="sweep_multiturn_profiles_sglang.sh"
            else
                SCRIPT="sweep_all_profiles.sh"
                [[ "$MODE" == "multi" ]] && SCRIPT="sweep_multiturn_profiles.sh"
            fi

            claim_slot "$HOST" "$SLOT_GPUS" "$SLOT_PORT"
            write_state_value "$JID" port "$SLOT_PORT"
            write_state_value "$JID" gpus "$SLOT_GPUS"
            write_state_value "$JID" run_id "$RUN_ID"
            write_signature "$JID" "$JOB_SIGNATURE"
            remove_state_file "$JID" reason
            remove_state_file "$JID" failure.json

            log "$JID: dispatching run_id=$RUN_ID on $HOST:$SLOT_PORT gpus=[$SLOT_GPUS] ($BACKEND, scope=$ROW_DASHBOARD_SCOPE, storage=$ROW_STORAGE_SCOPE, max_len=$RUN_MAX_LEN, mode=$MODE)"
            write_status "$JID" running

            # Build env with the scheduler's chosen slot. Strip legacy
            # CUDA_VISIBLE_DEVICES from EXTRA_ENV so preferred pins cannot
            # override a flexible placement decision at launch time.
            SLOT_ENV="PORT=$SLOT_PORT CUDA_VISIBLE_DEVICES=$SLOT_GPUS BENCH_RUN_ID=$RUN_ID BENCH_JOB_ID=$JID BENCH_SCOPE=$ROW_STORAGE_SCOPE BENCH_PORT=$SLOT_PORT BENCH_GPUS=$SLOT_GPUS"
            LAUNCH_EXTRA_ENV=$(remove_extra_env_key "CUDA_VISIBLE_DEVICES" "$EXTRA_ENV")

            CMD="$SLOT_ENV BENCH_REMOTE_TMP=${REMOTE_TMP_FOR_HOST} BENCH_REMOTE_ROOT=${REMOTE_CODE_FOR_HOST} ${LAUNCH_EXTRA_ENV} RESULT_SCOPE=${ROW_STORAGE_SCOPE} DASHBOARD_SCOPE=${ROW_DASHBOARD_SCOPE} bash ${REMOTE_CODE_FOR_HOST}/scripts/${SCRIPT} \
                ${MODEL_PATH} ${TP} ${SHORT} ${BACKEND} ${OUT_DIR_REMOTE} \
                ${PY} ${GPU_MEM} ${RUN_MAX_LEN} \"${CONCS}\" \"${PROFILES}\""
            REMOTE_LOG="$REMOTE_TMP_FOR_HOST/bench_${SHORT}_tp${TP}_${MODE}_${BACKEND}_p${SLOT_PORT}.log"
            write_run_record "$RUN_ID" "$JID" "dispatching" "$HOST" "$SLOT_GPUS" "$SLOT_PORT" "$BACKEND" \
                "$ROW_STORAGE_SCOPE" "$ROW_DASHBOARD_SCOPE" "$MODEL_PATH" "$TP" "$SHORT" "$MODE" \
                "$CONCS" "$PROFILES" "" "$REMOTE_LOG" "$RUN_STARTED_AT"
            if dry_run; then
                log "$JID: dry-run would run on $HOST: setsid bash -c '$CMD' > '$REMOTE_LOG' 2>&1 </dev/null &"
            else
                REMOTE_TMP_Q="$(printf "%q" "$REMOTE_TMP_FOR_HOST")"
                PY_Q="$(printf "%q" "$PY")"
                CMD_B64="$(printf '%s' "$CMD" | b64_arg)"
                LOG_B64="$(printf '%s' "$REMOTE_LOG" | b64_arg)"
                PY_LAUNCHER='import base64,os,subprocess,sys;cmd=base64.b64decode(sys.argv[1]).decode();log=base64.b64decode(sys.argv[2]).decode();os.makedirs(os.path.dirname(log) or ".",exist_ok=True);f=open(log,"ab",buffering=0);p=subprocess.Popen(["bash","-lc",cmd],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True);print(p.pid,flush=True)'
                PY_LAUNCHER_Q="$(printf "%q" "$PY_LAUNCHER")"
                REMOTE_PID=$(ssh "$HOST" "mkdir -p -- $REMOTE_TMP_Q && $PY_Q -c $PY_LAUNCHER_Q $CMD_B64 $LOG_B64" < /dev/null)
                write_state_value "$JID" launcher_pid "$REMOTE_PID"
                write_run_record "$RUN_ID" "$JID" "running" "$HOST" "$SLOT_GPUS" "$SLOT_PORT" "$BACKEND" \
                    "$ROW_STORAGE_SCOPE" "$ROW_DASHBOARD_SCOPE" "$MODEL_PATH" "$TP" "$SHORT" "$MODE" \
                    "$CONCS" "$PROFILES" "$REMOTE_PID" "$REMOTE_LOG" "$RUN_STARTED_AT"
                log "$JID: dispatched run_id=$RUN_ID launcher_pid=${REMOTE_PID:-unknown}"
            fi
            DISPATCHES=$((DISPATCHES + 1))
            ;;
    esac
done < "$JOBS_FILE"

if dry_run; then
    log "dry-run: skipping sweep-state publish"
elif truthy "${BENCH_ORCHESTRATOR_SKIP_PUBLISH:-0}"; then
    log "BENCH_ORCHESTRATOR_SKIP_PUBLISH enabled: skipping sweep-state publish"
else
    # Publish sweep-state.json to R2 so the dashboard reflects the latest cell
    # status (pending/running/done/skipped/known_oom). Non-fatal — if this
    # fails, the tick still succeeds; the next tick will republish.
    python3 "$REPO_ROOT/inference-benchmark/scripts/publish_sweep_state.py" \
        --state-dir "$STATE_ROOT" \
        --endpoint "$EP" --bucket "$BUCKET" --profile "$PROFILE" \
        >> "$LOG" 2>&1 || log "publish_sweep_state.py failed"
fi

log "tick complete"
