#!/usr/bin/env bash
# Prompt-only component ablation sweep (same checkpoint / tool budget).
# Usage:
#   bash eval/run_ablation_prompt.sh GeoEyes http://127.0.0.1:8006/v1 xlrs xhr lrs
set -euo pipefail

MODEL="${1:?model name, e.g. GeoEyes}"
API_BASE="${2:?api base, e.g. http://127.0.0.1:8006/v1}"
shift 2
DATASETS=("${@:-xlrs xhr lrs}")

MAX_TOOL_CALLS="${MAX_TOOL_CALLS:-8}"
MAX_TURNS="${MAX_TURNS:-20}"
NPROC="${NPROC:-5}"
WORK_DIR="${WORK_DIR:-./results/ablation_prompt}"
MODE="${MODE:-all}"

VARIANTS=(single_path multi_roi evidence full)

cd "$(dirname "$0")/.."
mkdir -p "$WORK_DIR"

echo "Model=$MODEL API=$API_BASE Data=${DATASETS[*]}"
echo "max_tool_calls=$MAX_TOOL_CALLS max_turns=$MAX_TURNS work_dir=$WORK_DIR"
echo "NOTE: This is inference-time prompt ablation on one checkpoint (diagnostic)."

for v in "${VARIANTS[@]}"; do
  echo ""
  echo "======== Ablation: $v ========"
  python eval/main.py \
    --data "${DATASETS[@]}" \
    --model "$MODEL" \
    --mode "$MODE" \
    --api-base "$API_BASE" \
    --api-nproc "$NPROC" \
    --work-dir "$WORK_DIR" \
    --ablation "$v" \
    --max-tool-calls "$MAX_TOOL_CALLS" \
    --max-turns "$MAX_TURNS"
done

echo ""
echo "Done. Results under: $WORK_DIR"
echo "Files look like: ${MODEL}_<dataset>_<variant>.jsonl / *_report.txt"
