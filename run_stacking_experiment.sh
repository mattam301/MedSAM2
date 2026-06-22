#!/usr/bin/env bash
set -euo pipefail

# Core stacking experiment:
#   base                : pretrained checkpoint on single slices
#   stacked_causal      : stacked checkpoint on N-slice clips
#   stacked_bidir       : same stacked checkpoint with bidirectional inference
#
# Main modes:
#   MODE=eval       prepare a 0/10 selected-set split and evaluate immediately
#   MODE=train-eval prepare a train/test split, finetune stacked, then evaluate

PYTHON="${PYTHON:-python}"
EXPERIMENT_NAME="${1:-}"

if [ -z "$EXPERIMENT_NAME" ]; then
  echo "Usage: $0 <experiment_name> [extra_prepare_args...]"
  echo
  echo "Examples:"
  echo "  MODE=eval $0 flare_selected --datasets flare --group-by-filename"
  echo "  MODE=train-eval $0 stack_comparison_e10 --max-cases-per-dataset 1000"
  echo
  echo "Environment knobs:"
  echo "  MODE=eval|train-eval                  default: eval"
  echo "  SUITE=core|full                       default: core"
  echo "  DATASETS_ROOT=data/new_datasets"
  echo "  BASE_CHECKPOINT=checkpoints/sam2.1_hiera_tiny.pt"
  echo "  INFER_CONFIG=sam2/configs/sam2.1_hiera_t512.yaml"
  echo "  IMAGE_CHANNEL_INDEX=0"
  echo "  NUM_GPUS=1"
  echo "  WINDOW_SIZE=8"
  echo "  WINDOW_STRIDE=8"
  echo "  CONTEXT_SLICE_INTERVAL=2              fixed clip slice spacing"
  echo "  CONTEXT_INTERVAL_MODE=fixed|dynamic"
  echo "  MEMORY_STRIDE=1                       SAM2 memory-bank temporal stride"
  echo "  STACK_CHECKPOINT=<path>               useful with MODE=eval"
  echo "  PREPARE=auto|always|never             default: auto"
  exit 1
fi
shift

ARGS=(
  "$EXPERIMENT_NAME"
  --mode "${MODE:-eval}"
  --suite "${SUITE:-core}"
  --python "$PYTHON"
  --datasets-root "${DATASETS_ROOT:-data/new_datasets}"
  --base-checkpoint "${BASE_CHECKPOINT:-checkpoints/sam2.1_hiera_tiny.pt}"
  --infer-config "${INFER_CONFIG:-sam2/configs/sam2.1_hiera_t512.yaml}"
  --image-channel-index "${IMAGE_CHANNEL_INDEX:-0}"
  --num-gpus "${NUM_GPUS:-1}"
  --window-size "${WINDOW_SIZE:-6}"
  --window-stride "${WINDOW_STRIDE:-6}"
  --context-slice-interval "${CONTEXT_SLICE_INTERVAL:-1}"
  --context-interval-mode "${CONTEXT_INTERVAL_MODE:-dynamic}"
  --memory-stride "${MEMORY_STRIDE:-1}"
  --prepare "${PREPARE:-auto}"
  --max-cases-per-dataset "${MAX_CASES:-5000}"
)
if [ -n "${STACK_CHECKPOINT:-}" ]; then
  ARGS+=(--stack-checkpoint "$STACK_CHECKPOINT")
fi

"$PYTHON" scripts/run_experiment.py "${ARGS[@]}" "$@"
