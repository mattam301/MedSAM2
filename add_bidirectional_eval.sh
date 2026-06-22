#!/usr/bin/env bash
set -euo pipefail

# Full ablation suite on an existing or selected-set experiment:
#   base
#   stacked_causal
#   stacked_bidir
#   stacked_bidir_unique
#   stacked_bidir_boot
#   stacked_bidir_boot_unique

PYTHON="${PYTHON:-python}"
EXPERIMENT_NAME="${1:-flare_v2}"
shift || true

ARGS=(
  "$EXPERIMENT_NAME"
  --mode "${MODE:-eval}"
  --suite full
  --python "$PYTHON"
  --datasets-root "${DATASETS_ROOT:-data/new_datasets}"
  --base-checkpoint "${BASE_CHECKPOINT:-checkpoints/sam2.1_hiera_tiny.pt}"
  --infer-config "${INFER_CONFIG:-sam2/configs/sam2.1_hiera_t512.yaml}"
  --image-channel-index "${IMAGE_CHANNEL_INDEX:-0}"
  --window-size "${WINDOW_SIZE:-8}"
  --window-stride "${WINDOW_STRIDE:-8}"
  --context-slice-interval "${CONTEXT_SLICE_INTERVAL:-1}"
  --context-interval-mode "${CONTEXT_INTERVAL_MODE:-fixed}"
  --memory-stride "${MEMORY_STRIDE:-1}"
  --prepare "${PREPARE:-auto}"
  --new-bidir-threshold "${NEW_BIDIR_THRESHOLD:-1.0}"
  --new-bidir-max-context "${NEW_BIDIR_MAX_CONTEXT:-3}"
  --min-component-volume "${MIN_COMPONENT_VOLUME:-100}"
)

if [ "${NEW_BIDIR_UNIQUE_RESIDUAL:-true}" = "false" ]; then
  ARGS+=(--no-new-bidir-unique-residual)
fi
if [ "${DISABLE_3D_FILTER:-false}" = "true" ]; then
  ARGS+=(--disable-3d-filter)
fi
if [ -n "${STACK_CHECKPOINT:-}" ]; then
  ARGS+=(--stack-checkpoint "$STACK_CHECKPOINT")
fi
if [ -n "${MEMORY_DECOMPOSER_CHECKPOINT:-}" ]; then
  ARGS+=(--memory-decomposer-checkpoint "$MEMORY_DECOMPOSER_CHECKPOINT")
fi

"$PYTHON" scripts/run_experiment.py "${ARGS[@]}" "$@"
