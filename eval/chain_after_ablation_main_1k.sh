#!/usr/bin/env bash
# Wait for XLRS ablation `full` to finish, then run XHR → XLRS → LRS under
# corrected 1k + discard-repeat → force. Prevents the old chain from using main.py.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/srv/user/zhujs/miniconda3/envs/ultrars/bin/python}"
API_BASE="${API_BASE:-http://127.0.0.1:8110/v1}"
MODEL="${MODEL:-agent_zoom_step_1120}"
WORKERS="${WORKERS:-6}"
MAX_TURNS="${MAX_TURNS:-16}"

ABLATION_DIR="${ABLATION_DIR:-results/ablation_prompt_1k}"
FULL_JSONL="${ABLATION_DIR}/${MODEL}_xlrs_full.jsonl"
BENCH_DIR="${BENCH_DIR:-results/main_1k_discard_force}"
LOG="${ABLATION_DIR}/logs/after_ablation_main_1k.log"
mkdir -p "${ABLATION_DIR}/logs" "$BENCH_DIR"

export GEOVISTA_MAX_PIXELS="${GEOVISTA_MAX_PIXELS:-1003520}"
unset GEOVISTA_MAX_RESOLUTION || true

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "Waiting for ablation full → 3080 lines and ablation process exit…"
while true; do
  n=0
  [[ -f "$FULL_JSONL" ]] && n=$(wc -l <"$FULL_JSONL" | tr -d ' ')
  alive=0
  pgrep -f "run_ablation_xlrs_1k.py" >/dev/null 2>&1 && alive=1
  if [[ "$n" -ge 3080 && "$alive" -eq 0 ]]; then
    log "Ablation full done (n=$n). Starting main-1k benches."
    break
  fi
  if (( n % 200 == 0 )) && [[ "$n" -gt 0 ]]; then
    log "full progress ${n}/3080 (ablation_alive=$alive)"
  fi
  sleep 45
done

# Kill any legacy phase-2 main.py the old chain may have just launched.
pkill -f "eval/main.py --data xhr lrs" 2>/dev/null || true
sleep 2

log "=== XHR (RSHR) → XLRS → LRS | discard-repeat→force | 1k ==="
"$PY" -u eval/run_main_1k_benches.py \
  --api-base "$API_BASE" \
  --model-name "$MODEL" \
  --max-turns "$MAX_TURNS" \
  --workers "$WORKERS" \
  --work-dir "$BENCH_DIR" \
  --data xhr xlrs lrs \
  2>&1 | tee -a "$LOG"

log "ALL DONE → $BENCH_DIR"
