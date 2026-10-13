#!/usr/bin/env bash
# Run XLRS twice more (r2, r3) with same stack as main_1k_discard_force.
# Run1 already lives in results/main_1k_discard_force.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/srv/user/zhujs/miniconda3/envs/ultrars/bin/python}"
API_BASE="${API_BASE:-http://127.0.0.1:8110/v1}"
MODEL="${MODEL:-agent_zoom_step_1120}"
WORKERS="${WORKERS:-6}"
MAX_TURNS="${MAX_TURNS:-16}"

R1="${R1:-results/main_1k_discard_force}"
R2="${R2:-results/main_1k_discard_force_xlrs_r2}"
R3="${R3:-results/main_1k_discard_force_xlrs_r3}"
LOG="${LOG:-results/main_1k_discard_force_xlrs_repeats.log}"
mkdir -p "$(dirname "$LOG")" "$R2" "$R3"

export GEOVISTA_MAX_PIXELS="${GEOVISTA_MAX_PIXELS:-1003520}"
unset GEOVISTA_MAX_RESOLUTION || true

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

run_one() {
  local tag=$1 dir=$2
  mkdir -p "$dir"
  log "======== START $tag → $dir ========"
  "$PY" -u eval/run_main_1k_benches.py \
    --api-base "$API_BASE" \
    --model-name "$MODEL" \
    --max-turns "$MAX_TURNS" \
    --workers "$WORKERS" \
    --work-dir "$dir" \
    --data xlrs \
    2>&1 | tee -a "$LOG"
  local acc="n/a"
  if [[ -f "$dir/${MODEL}_xlrs_report.txt" ]]; then
    acc=$(grep -E 'OVERALL ACCURACY' "$dir/${MODEL}_xlrs_report.txt" | tail -1)
  fi
  log "======== DONE $tag | $acc ========"
}

log "XLRS discard-force repeats: r2 then r3 (r1=$R1)"
run_one r2 "$R2"
run_one r3 "$R3"

acc1=$(grep -E 'OVERALL ACCURACY' "$R1/${MODEL}_xlrs_report.txt" | tail -1 || echo n/a)
acc2=$(grep -E 'OVERALL ACCURACY' "$R2/${MODEL}_xlrs_report.txt" | tail -1 || echo n/a)
acc3=$(grep -E 'OVERALL ACCURACY' "$R3/${MODEL}_xlrs_report.txt" | tail -1 || echo n/a)

cat > results/main_1k_discard_force_xlrs_REPEATS.md <<EOF
# XLRS ×3 (1k, discard-repeat→force)

| run | work-dir | result |
|-----|----------|--------|
| 1 | $R1 | $acc1 |
| 2 | $R2 | $acc2 |
| 3 | $R3 | $acc3 |

Log: $LOG
EOF

log "======== ALL 3 XLRS REPEATS COMPLETE ========"
log "summary → results/main_1k_discard_force_xlrs_REPEATS.md"
