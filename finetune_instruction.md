# Fine-tune Instructions

This guide explains how to use the new scripts for leave-one-dataset-out domain adaptation with 3 datasets stored under `data/new_datasets/`.

## What This Pipeline Does

- Uses 2 datasets for fine-tuning.
- Splits each fine-tune dataset into `70% train / 30% test`.
- Keeps the 3rd dataset fully unseen.
- Trains with the supported MedSAM2 fine-tuning pipeline.
- Evaluates on:
  - the held-out `30%` test split from each fine-tune dataset
  - the fully unseen dataset

## Expected Dataset Layout

Put your 3 datasets here:

```text
MedSAM2/data/new_datasets/
  DatasetA/
    case_001.npz
    case_002.npz
  DatasetB/
    case_001.npz
    ...
  DatasetC/
    case_001.npz
    ...
```

Each `.npz` file should contain at least:

- `imgs`: shape `(D, H, W)`, grayscale volume
- `gts`: shape `(D, H, W)`, segmentation labels

The evaluation script does not require `recist`. It generates prompts from `gts`.

The pipeline now also accepts slice-based NPZs:

- `imgs`: `(H, W)`
- `imgs`: `(H, W, 1)`
- `imgs`: `(H, W, 3)` for BraTS-like multi-modal slices
- `gts`: `(H, W)`

For multi-channel slice inputs, the loader keeps one channel and ignores the others. By default it uses channel `0`, and you can change that with `IMAGE_CHANNEL_INDEX`.

## Files Added

- `run_new_dataset_finetune.sh`
- `scripts/prepare_domain_adaptation_experiment.py`
- `scripts/eval_npz_dataset.py`

## Prerequisites

- Activate the MedSAM2 environment first.
- Make sure training dependencies are installed.
- Make sure the base checkpoint exists:

```bash
bash download.sh
```

The default training checkpoint used by this workflow is:

```text
checkpoints/sam2.1_hiera_tiny.pt
```

## Easiest Way To Run

From the repo root:

```bash
bash run_new_dataset_finetune.sh <held_out_dataset_name>
```

Example:

```bash
bash run_new_dataset_finetune.sh DatasetC
```

This will:

- create deterministic split manifests
- generate a training config
- auto-switch to `num_frames=1` if the fine-tune datasets are slice-based
- launch fine-tuning
- print the evaluation commands at the end

## What Gets Generated

For an experiment like `lodo_datasetc`, the pipeline creates:

- `data/new_datasets/_splits/lodo_datasetc/manifests/`
- `data/new_datasets/_splits/lodo_datasetc/experiment.json`
- `data/new_datasets/_splits/lodo_datasetc/commands.txt`
- `sam2/configs/generated/lodo_datasetc.yaml`
- `exp_log/lodo_datasetc/`

## Prepare Only

If you want to inspect the split and generated config before training:

```bash
python scripts/prepare_domain_adaptation_experiment.py --held-out DatasetC
```

Then train manually:

```bash
python -m training.train \
  -c configs/generated/lodo_datasetc.yaml \
  --output-path exp_log/lodo_datasetc \
  --use-cluster 0 \
  --num-gpus 1 \
  --num-nodes 1
```

## Manual Evaluation

After training, evaluate a dataset split like this:

```bash
python scripts/eval_npz_dataset.py \
  --checkpoint exp_log/lodo_datasetc/checkpoints/checkpoint.pt \
  --cfg sam2/configs/sam2.1_hiera_t512.yaml \
  --dataset-dir data/new_datasets/DatasetA \
  --file-list data/new_datasets/_splits/lodo_datasetc/manifests/dataseta_test.txt \
  --output-dir exp_log/lodo_datasetc/eval/dataseta
```

To save predicted segmentations too:

```bash
python scripts/eval_npz_dataset.py \
  --checkpoint exp_log/lodo_datasetc/checkpoints/checkpoint.pt \
  --cfg sam2/configs/sam2.1_hiera_t512.yaml \
  --dataset-dir data/new_datasets/DatasetC \
  --file-list data/new_datasets/_splits/lodo_datasetc/manifests/datasetc_unseen.txt \
  --output-dir exp_log/lodo_datasetc/eval/datasetc \
  --save-preds
```

## Prompt Type During Evaluation

Supported prompt modes:

- `--prompt-type box`
- `--prompt-type point`
- `--prompt-type mask`

Default is `box`.

## Useful Environment Overrides

You can customize the shell runner with environment variables:

```bash
NUM_GPUS=2 NUM_EPOCHS=50 BATCH_SIZE=4 bash run_new_dataset_finetune.sh DatasetC
```

Supported overrides include:

- `DATASETS_ROOT`
- `BASE_CONFIG`
- `BASE_CHECKPOINT`
- `INFER_CONFIG`
- `TRAIN_RATIO`
- `SEED`
- `NUM_EPOCHS`
- `BATCH_SIZE`
- `NUM_WORKERS`
- `NUM_FRAMES`
- `MAX_NUM_OBJECTS`
- `BASE_LR`
- `VISION_LR`
- `DATASET_MULTIPLIER`
- `IMAGE_CHANNEL_INDEX`
- `NUM_GPUS`

## Important Note

This workflow uses the supported MedSAM2 fine-tuning path based on `training/train.py`.

It does not fine-tune the new bidirectional predictor end to end. The bidirectional code currently fits best as an inference-side experiment, while this training pipeline is the stable baseline path for MRI/CT adaptation.
