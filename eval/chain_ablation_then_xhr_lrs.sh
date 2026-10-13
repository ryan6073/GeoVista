#!/usr/bin/env bash
# 1) XLRS ablation under corrected 1k
# 2) XHR → XLRS → LRS with discard-repeat→force + 1k (see run_main_1k_benches.py)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/srv/user/zhujs/miniconda3/envs/ultrars/bin/python}"
API_BASE="${API_BASE:-http://127.0.0.1:8110/v1}"
MODEL="${MODEL:-agent_zoom_step_1120}"
WORKERS="${WORKERS:-6}"
MAX_TURNS="${MAX_TURNS:-16}"
MAX_TOOL_CALLS="${MAX_TOOL_CALLS:-8}"

ABLATION_DIR="${ABLATION_DIR:-results/ablation_prompt_1k}"
BENCH_DIR="${BENCH_DIR:-results/main_1k_discard_force}"
LOG_DIR="${ABLATION_DIR}/logs"
mkdir -p "$LOG_DIR" "$BENCH_DIR"

export GEOVISTA_MAX_PIXELS="${GEOVISTA_MAX_PIXELS:-1003520}"
unset GEOVISTA_MAX_RESOLUTION || true

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/chain.log"; }

log "=== Phase 1: XLRS ablation 1k ==="
"$PY" -u eval/run_ablation_xlrs_1k.py \
  --api-base "$API_BASE" \
  --model-name "$MODEL" \
  --max-turns "$MAX_TURNS" \
  --max-tool-calls "$MAX_TOOL_CALLS" \
  --workers "$WORKERS" \
  --work-dir "$ABLATION_DIR" \
  2>&1 | tee -a "$LOG_DIR/ablation_xlrs.log"

log "=== Phase 2: XHR → XLRS → LRS (1k, discard-repeat→force) ==="
"$PY" -u eval/run_main_1k_benches.py \
  --api-base "$API_BASE" \
  --model-name "$MODEL" \
  --max-turns "$MAX_TURNS" \
  --workers "$WORKERS" \
  --work-dir "$BENCH_DIR" \
  --data xhr xlrs lrs \
  2>&1 | tee -a "$LOG_DIR/main_1k_benches.log"

log "=== ALL DONE ==="
log "Ablation: $ABLATION_DIR"
log "Main-1k:  $BENCH_DIR"
