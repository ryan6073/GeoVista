#!/usr/bin/env bash
# After the 1k (max_pixels≈1M) XLRS run finishes, immediately start the 2k run.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/srv/user/zhujs/miniconda3/envs/ultrars/bin/python}"
API_BASE="${API_BASE:-http://127.0.0.1:8110/v1}"
MODEL="${MODEL:-agent_zoom_step_1120}"
WORKERS="${WORKERS:-6}"
MAX_TURNS="${MAX_TURNS:-16}"

DIR_1K="${DIR_1K:-results/xlrs_georeflex_boost_v2}"
DIR_2K="${DIR_2K:-results/xlrs_georeflex_boost_2k}"
LOG_CHAIN="${DIR_1K}/logs/chain_1k_then_2k.log"
JSONL_1K="${DIR_1K}/${MODEL}_xlrs.jsonl"
PID_1K="${PID_1K:-}"

mkdir -p "${DIR_1K}/logs" "${DIR_2K}/logs"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_CHAIN"; }

wait_1k_done() {
  log "Waiting for 1k run to finish (dir=${DIR_1K}, pid=${PID_1K:-auto})"
  while true; do
    alive=0
    if [[ -n "${PID_1K}" ]] && kill -0 "${PID_1K}" 2>/dev/null; then
      alive=1
    elif pgrep -f "run_xlrs_main.py .*${DIR_1K}" >/dev/null 2>&1; then
      alive=1
    fi

    n=0
    [[ -f "$JSONL_1K" ]] && n=$(wc -l <"$JSONL_1K" | tr -d ' ')
    if [[ "$alive" -eq 0 && "$n" -ge 3080 ]]; then
      log "1k finished: lines=${n}"
      return 0
    fi
    if [[ "$alive" -eq 0 && "$n" -gt 0 ]]; then
      # process gone but short of 3080 — still proceed (partial) unless empty
      log "1k process exited early: lines=${n}; starting 2k anyway"
      return 0
    fi
    if (( n % 200 == 0 )) && [[ "$n" -gt 0 ]]; then
      log "1k progress: ${n}/3080 (still running)"
    fi
    sleep 60
  done
}

start_2k() {
  log "Starting 2k run → ${DIR_2K}"
  log "  GEOVISTA_MAX_RESOLUTION=2048 GEOVISTA_MAX_PIXELS=4194304"
  export GEOVISTA_MAX_RESOLUTION=2048
  export GEOVISTA_MAX_PIXELS=4194304
  nohup "$PY" -u eval/run_xlrs_main.py \
    --api-base "$API_BASE" \
    --model-name "$MODEL" \
    --max-turns "$MAX_TURNS" \
    --workers "$WORKERS" \
    --work-dir "$DIR_2K" \
    >"${DIR_2K}/logs/xlrs_eval.log" 2>&1 &
  echo $! >"${DIR_2K}/logs/eval.pid"
  log "2k PID=$(cat "${DIR_2K}/logs/eval.pid")"
  cat >"${DIR_2K}/EXPERIMENT.md" <<EOF
# XLRS GeoReflex-boost 2k

Chained after 1k run (\`${DIR_1K}\`).

- \`GEOVISTA_MAX_RESOLUTION=2048\`
- \`GEOVISTA_MAX_PIXELS=4194304\` (≈2k×2k area)
- Same GeoReflex crop/SOP fixes as v2
- Model: ${MODEL} @ ${API_BASE}
EOF
}

wait_1k_done
start_2k
log "Chain complete; 2k is running."
