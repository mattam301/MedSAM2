# Bidirectional Attention Tests

Smoke tests for `sam2/bidirectional_video_predictor.py`.  All tests are CPU-only
and require no checkpoints or GPU, completing in under 10 seconds.

## Prerequisites

```bash
conda activate medsam2
pip install pytest
```

## Run

```bash
# From the repo root (MedSAM2/)
pytest tests/bidirectional/ -v
```

## Test modules

| File | What it tests |
|------|---------------|
| `test_bidirectional_predictor.py` | Full suite – see below |

## Test classes

### `TestSliceConsistencyLoss`
Unit tests for the `slice_consistency_loss` helper:
- Returns 0 for volumes with < 3 slices
- Returns 0 for a constant (perfectly-consistent) volume
- Returns > 0 for an alternating (maximally-inconsistent) volume
- `weight` parameter scales the loss linearly
- Gradient flows back to `pred_masks` without NaN
- Output is a scalar tensor
- Reversing slice order gives the same loss (symmetric identity warp)

### `TestBidirectionalPredictorClass`
Class-structure checks:
- `BidirectionalSAM2VideoPredictorNPZ` is a subclass of `SAM2VideoPredictorNPZ`
- All three required methods are present
- `_use_bidirectional_memory` flag defaults to `False` in `__init__`
- `build_bidir_sam2_video_predictor_npz` is callable
- No new `nn.Parameter` class attributes (zero new parameters)

### `TestMemoryRoutingFlag`
Verifies the dispatch logic in `_prepare_memory_conditioned_features`:
- Routes to bidirectional method when `_use_bidirectional_memory=True`
- Stays on the causal (super) path when `_use_bidirectional_memory=False`

### `TestBidirectionalMemoryAssembly`
Checks memory token count via an injected `memory_attention` mock:
- Middle frame receives **more** memory tokens than the causal maximum
- Last frame (no future) receives **≤ causal maximum** tokens
- Initial conditioning frame receives exactly **1** dummy token (`no_mem_embed`)

### `TestPropagateControlFlow`
Integration smoke tests for the two-pass loop (no real inference):
- Yields exactly `num_frames` results
- `_use_bidirectional_memory` is `False` after a clean run
- `_use_bidirectional_memory` is `False` even after a Pass-2 exception (try/finally)
- Conditioning frames are **skipped** in Pass 2 (call count verifications)

### `TestTemporalPESymmetry`
Validates that all temporal positional encoding indices stay in `[0, num_maskmem-1]`
for both past and future directions.

## Design Notes

All tests mock `_run_single_frame_inference` so they exercise only the new
bidirectional control-flow and memory-assembly logic, not the SAM2 backbone.
This keeps the tests fast and dependency-free.
