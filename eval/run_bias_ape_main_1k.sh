#!/usr/bin/env bash
# Evaluate the three bias-study checkpoints and APE-GRO in parallel.
# Uses the current corrected-1k + discard-repeat->force settings.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/srv/user/zhujs/miniconda3/envs/ultrars/bin/python}"
OUT_ROOT="${OUT_ROOT:-results/bias_ape_1k_eval}"
WORKERS="${WORKERS:-6}"
MAX_TURNS="${MAX_TURNS:-16}"
mkdir -p "$OUT_ROOT/logs"

export GEOVISTA_MAX_PIXELS="${GEOVISTA_MAX_PIXELS:-1003520}"
unset GEOVISTA_MAX_RESOLUTION || true

declare -a SPECS=(
  "bias_acc|8111"
  "bias_iou|8112"
  "bias_plan|8113"
  "ape_gro|8110"
)

pids=()
for spec in "${SPECS[@]}"; do
  IFS='|' read -r model port <<<"$spec"
  work_dir="$OUT_ROOT/$model"
  log="$OUT_ROOT/logs/$model.log"
  mkdir -p "$work_dir"
  "$PY" -u eval/run_main_1k_benches.py \
    --api-base "http://127.0.0.1:${port}/v1" \
    --model-name "$model" \
    --max-turns "$MAX_TURNS" \
    --workers "$WORKERS" \
    --work-dir "$work_dir" \
    --data xlrs xhr lrs \
    >"$log" 2>&1 &
  pid=$!
  echo "$pid" >"$OUT_ROOT/logs/$model.pid"
  echo "started $model port=$port pid=$pid log=$log"
  pids+=("$pid")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if (( failed )); then
  echo "One or more evaluations failed; inspect $OUT_ROOT/logs"
  exit 1
fi
echo "All bias-study + APE-GRO evaluations completed."
