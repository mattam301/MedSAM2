#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS_ROOT = REPO_ROOT / "data" / "new_datasets"
DEFAULT_BASE_CONFIG = REPO_ROOT / "sam2" / "configs" / "sam2.1_hiera_tiny_finetune512.yaml"
DEFAULT_INFER_CONFIG = REPO_ROOT / "sam2" / "configs" / "sam2.1_hiera_t512.yaml"
DEFAULT_BASE_CHECKPOINT = REPO_ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"
GENERATED_CONFIG_DIR = REPO_ROOT / "sam2" / "configs" / "generated"
HYDRA_GLOBAL_PACKAGE_HEADER = "# @package _global_\n\n"


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip()).strip("_").lower()


def relative_case_id(npz_path: Path, dataset_dir: Path) -> str:
    return npz_path.relative_to(dataset_dir).with_suffix("").as_posix()


def discover_dataset_dirs(datasets_root: Path) -> dict[str, Path]:
    dataset_dirs: dict[str, Path] = {}
    for child in sorted(datasets_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name.startswith("_"):
            continue
        if any(child.rglob("*.npz")):
            dataset_dirs[child.name] = child
    return dataset_dirs


def cap_cases(
    case_ids: list[str],
    max_cases: int | None,
    seed: int,
) -> list[str]:
    """Deterministically subsample *case_ids* when they exceed *max_cases*."""
    if max_cases is None or len(case_ids) <= max_cases:
        return case_ids
    sampled = case_ids[:]
    random.Random(seed).shuffle(sampled)
    return sorted(sampled[:max_cases])


def split_cases(case_ids: list[str], train_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    if not case_ids:
        return [], []

    shuffled = case_ids[:]
    random.Random(seed).shuffle(shuffled)

    if len(shuffled) == 1:
        return sorted(shuffled), []

    train_count = int(len(shuffled) * train_ratio)
    train_count = max(1, min(train_count, len(shuffled) - 1))

    train_ids = sorted(shuffled[:train_count])
    test_ids = sorted(shuffled[train_count:])
    return train_ids, test_ids


def write_manifest(path: Path, case_ids: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(case_ids)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def detect_case_mode(npz_path: Path) -> str:
    with np.load(npz_path, allow_pickle=True) as data:
        if "imgs" not in data.files or "gts" not in data.files:
            raise ValueError(f"{npz_path} must contain 'imgs' and 'gts'. Found: {data.files}")
        gts = np.asarray(data["gts"])
    if gts.ndim == 2:
        return "slice"
    if gts.ndim == 3:
        return "volume"
    raise ValueError(f"Unsupported mask shape in {npz_path}: {gts.shape}")


def infer_dataset_mode(dataset_dir: Path, case_ids: list[str]) -> str:
    for case_id in case_ids:
        npz_path = dataset_dir / f"{case_id}.npz"
        if npz_path.exists():
            return detect_case_mode(npz_path)
    return "unknown"


def build_dataset_entry(
    folder: Path,
    file_list_txt: Path,
    multiplier: int,
    image_channel_index: int,
) -> dict:
    return {
        "_target_": "training.dataset.vos_dataset.VOSDataset",
        "transforms": "${vos.train_transforms}",
        "training": True,
        "video_dataset": {
            "_target_": "training.dataset.vos_raw_dataset.NPZRawDataset",
            "folder": str(folder.resolve()),
            "file_list_txt": str(file_list_txt.resolve()),
            "image_channel_index": image_channel_index,
        },
        "sampler": {
            "_target_": "training.dataset.vos_sampler.RandomUniformSampler",
            "num_frames": "${scratch.num_frames}",
            "max_num_objects": "${scratch.max_num_objects}",
        },
        "multiplier": multiplier,
    }


def save_generated_hydra_config(path: Path, cfg, OmegaConf) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    yaml_text = OmegaConf.to_yaml(cfg, resolve=False)
    path.write_text(HYDRA_GLOBAL_PACKAGE_HEADER + yaml_text, encoding="utf-8")
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    if first_line != "# @package _global_":
        raise RuntimeError(
            f"Generated config header check failed for {path}. "
            "Expected '# @package _global_' on the first line."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic 70/30 split manifests and a generated MedSAM2 "
            "fine-tuning config for leave-one-dataset-out experiments."
        )
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=DEFAULT_DATASETS_ROOT,
        help="Root folder containing one subdirectory per dataset.",
    )
    parser.add_argument(
        "--held-out",
        required=True,
        help="Dataset name to keep fully unseen during fine-tuning.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Optional experiment name. Defaults to lodo_<held_out>.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of each fine-tune dataset used for training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic splitting.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=DEFAULT_BASE_CONFIG,
        help="Base MedSAM2 fine-tune config to clone and rewrite.",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=DEFAULT_BASE_CHECKPOINT,
        help="Base checkpoint used to initialize training weights.",
    )
    parser.add_argument(
        "--infer-config",
        type=Path,
        default=DEFAULT_INFER_CONFIG,
        help="Inference config used in the generated evaluation commands.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=30,
        help="Fine-tuning epochs written into the generated config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="train_video_batch_size written into the generated config.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="num_train_workers written into the generated config.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Number of frames sampled per training sample.",
    )
    parser.add_argument(
        "--max-num-objects",
        type=int,
        default=5,
        help="Maximum objects sampled per training sample.",
    )
    parser.add_argument(
        "--base-lr",
        type=float,
        default=5e-5,
        help="Base learning rate for non-vision parameters.",
    )
    parser.add_argument(
        "--vision-lr",
        type=float,
        default=3e-5,
        help="Learning rate for image encoder parameters.",
    )
    parser.add_argument(
        "--dataset-multiplier",
        type=int,
        default=1,
        help="Repeat factor multiplier for each fine-tune dataset entry.",
    )
    parser.add_argument(
        "--image-channel-index",
        type=int,
        default=0,
        help=(
            "Channel index to use when an NPZ stores a 2D image as HxWxC or a volume "
            "as DxHxWxC. Useful for multi-modal BraTS-like files."
        ),
    )
    parser.add_argument(
        "--max-cases-per-dataset",
        type=int,
        default=None,
        help=(
            "Maximum number of cases to use from any single dataset. "
            "Datasets with more cases are deterministically subsampled "
            "before the train/test split. Useful when a dataset has "
            ">20 k samples and you want to keep experiments tractable. "
            "By default no cap is applied."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise SystemExit(
            "omegaconf is not installed in the current Python environment. "
            "Activate the MedSAM2 environment first, then rerun this script."
        ) from exc

    datasets_root = args.datasets_root.resolve()
    dataset_dirs = discover_dataset_dirs(datasets_root)

    if not dataset_dirs:
        raise SystemExit(
            f"No dataset folders with .npz files were found under {datasets_root}.\n"
            "Expected layout:\n"
            "  data/new_datasets/<dataset_a>/**/*.npz\n"
            "  data/new_datasets/<dataset_b>/**/*.npz\n"
            "  data/new_datasets/<dataset_c>/**/*.npz"
        )

    if args.held_out not in dataset_dirs:
        discovered = ", ".join(dataset_dirs) or "<none>"
        raise SystemExit(
            f"Held-out dataset '{args.held_out}' was not found.\n"
            f"Discovered datasets: {discovered}"
        )

    fine_tune_dataset_names = [name for name in dataset_dirs if name != args.held_out]
    if len(fine_tune_dataset_names) != 2:
        raise SystemExit(
            "This helper expects exactly 3 dataset folders so it can use 2 for "
            "fine-tuning and 1 as the unseen dataset."
        )

    experiment_name = args.experiment_name or f"lodo_{slugify(args.held_out)}"
    split_root = datasets_root / "_splits" / experiment_name
    manifest_root = split_root / "manifests"
    generated_config_path = GENERATED_CONFIG_DIR / f"{experiment_name}.yaml"
    generated_config_name = f"configs/generated/{generated_config_path.name}"
    output_path = REPO_ROOT / "exp_log" / experiment_name
    default_trained_checkpoint = output_path / "checkpoints" / "checkpoint.pt"

    split_meta: dict[str, dict] = {}
    dataset_entries = []
    experiment_mode = "volume"

    for index, dataset_name in enumerate(sorted(fine_tune_dataset_names)):
        dataset_dir = dataset_dirs[dataset_name]
        all_case_ids = sorted(
            relative_case_id(path, dataset_dir) for path in dataset_dir.rglob("*.npz")
        )

        # ---- cap large datasets before splitting ----
        case_ids = cap_cases(
            all_case_ids,
            args.max_cases_per_dataset,
            args.seed + index,
        )
        if len(case_ids) < len(all_case_ids):
            print(
                f"[cap] {dataset_name}: subsampled {len(all_case_ids)} → "
                f"{len(case_ids)} cases (max-cases-per-dataset={args.max_cases_per_dataset})"
            )

        train_ids, test_ids = split_cases(case_ids, args.train_ratio, args.seed + index)
        dataset_mode = infer_dataset_mode(dataset_dir, case_ids)
        if dataset_mode == "slice":
            experiment_mode = "slice"

        train_manifest = manifest_root / f"{slugify(dataset_name)}_train.txt"
        test_manifest = manifest_root / f"{slugify(dataset_name)}_test.txt"
        write_manifest(train_manifest, train_ids)
        write_manifest(test_manifest, test_ids)

        dataset_entries.append(
            build_dataset_entry(
                dataset_dir,
                train_manifest,
                args.dataset_multiplier,
                args.image_channel_index,
            )
        )
        split_meta[dataset_name] = {
            "dataset_dir": str(dataset_dir.resolve()),
            "mode": dataset_mode,
            "num_cases_total": len(all_case_ids),
            "num_cases_after_cap": len(case_ids),
            "train_cases": len(train_ids),
            "test_cases": len(test_ids),
            "train_manifest": str(train_manifest.resolve()),
            "test_manifest": str(test_manifest.resolve()),
        }

    # ---- held-out dataset (also capped for evaluation tractability) ----
    held_out_dir = dataset_dirs[args.held_out]
    all_held_out_case_ids = sorted(
        relative_case_id(path, held_out_dir) for path in held_out_dir.rglob("*.npz")
    )
    held_out_case_ids = cap_cases(
        all_held_out_case_ids,
        args.max_cases_per_dataset,
        args.seed,
    )
    if len(held_out_case_ids) < len(all_held_out_case_ids):
        print(
            f"[cap] {args.held_out} (held-out): subsampled "
            f"{len(all_held_out_case_ids)} → {len(held_out_case_ids)} cases "
            f"(max-cases-per-dataset={args.max_cases_per_dataset})"
        )

    held_out_manifest = manifest_root / f"{slugify(args.held_out)}_unseen.txt"
    write_manifest(held_out_manifest, held_out_case_ids)
    split_meta[args.held_out] = {
        "dataset_dir": str(held_out_dir.resolve()),
        "mode": infer_dataset_mode(held_out_dir, held_out_case_ids),
        "num_cases_total": len(all_held_out_case_ids),
        "num_cases_after_cap": len(held_out_case_ids),
        "train_cases": 0,
        "test_cases": len(held_out_case_ids),
        "train_manifest": None,
        "test_manifest": str(held_out_manifest.resolve()),
    }

    cfg = OmegaConf.load(args.base_config)
    cfg.scratch.train_video_batch_size = args.batch_size
    cfg.scratch.num_train_workers = args.num_workers
    cfg.scratch.num_frames = 1 if experiment_mode == "slice" else args.num_frames
    cfg.scratch.max_num_objects = args.max_num_objects
    cfg.scratch.num_epochs = args.num_epochs
    cfg.scratch.base_lr = args.base_lr
    cfg.scratch.vision_lr = args.vision_lr
    cfg.trainer.data.train.datasets[0].dataset.datasets = OmegaConf.create(dataset_entries)
    cfg.trainer.checkpoint.model_weight_initializer.state_dict.checkpoint_path = str(
        args.base_checkpoint.resolve()
    )
    cfg.launcher.experiment_log_dir = str(output_path)
    cfg.launcher.num_nodes = 1
    cfg.submitit.use_cluster = False
    if experiment_mode == "slice":
        cfg.trainer.model.num_frames_to_correct_for_train = 1
        cfg.trainer.model.num_frames_to_correct_for_eval = 1
        cfg.trainer.model.num_init_cond_frames_for_train = 1
        cfg.trainer.model.num_init_cond_frames_for_eval = 1
        cfg.trainer.model.rand_frames_to_correct_for_train = False
        cfg.trainer.model.rand_init_cond_frames_for_train = False

    save_generated_hydra_config(generated_config_path, cfg, OmegaConf)

    train_command = (
        f"python -m training.train -c {generated_config_name} "
        f"--output-path {output_path} --use-cluster 0 --num-gpus 1 --num-nodes 1"
    )

    eval_commands = []
    for dataset_name in sorted(fine_tune_dataset_names):
        manifest = split_meta[dataset_name]["test_manifest"]
        dataset_dir = split_meta[dataset_name]["dataset_dir"]
        eval_commands.append(
            "python scripts/eval_npz_dataset.py "
            f"--checkpoint {default_trained_checkpoint} "
            f"--cfg {args.infer_config.resolve()} "
            f"--dataset-dir {dataset_dir} "
            f"--file-list {manifest} "
            f"--image-channel-index {args.image_channel_index} "
            f"--output-dir {output_path / 'eval' / slugify(dataset_name)}"
        )
    eval_commands.append(
        "python scripts/eval_npz_dataset.py "
        f"--checkpoint {default_trained_checkpoint} "
        f"--cfg {args.infer_config.resolve()} "
        f"--dataset-dir {held_out_dir.resolve()} "
        f"--file-list {held_out_manifest.resolve()} "
        f"--image-channel-index {args.image_channel_index} "
        f"--output-dir {output_path / 'eval' / slugify(args.held_out)}"
    )

    commands_path = split_root / "commands.txt"
    commands_path.parent.mkdir(parents=True, exist_ok=True)
    commands_text = (
        "# Train\n"
        f"{train_command}\n\n"
        "# Evaluate seen test splits + unseen dataset\n"
        + "\n".join(eval_commands)
        + "\n"
    )
    commands_path.write_text(commands_text, encoding="utf-8")

    metadata = {
        "experiment_name": experiment_name,
        "experiment_mode": experiment_mode,
        "datasets_root": str(datasets_root),
        "held_out_dataset": args.held_out,
        "fine_tune_datasets": sorted(fine_tune_dataset_names),
        "max_cases_per_dataset": args.max_cases_per_dataset,
        "image_channel_index": args.image_channel_index,
        "generated_config": str(generated_config_path.resolve()),
        "generated_config_name": generated_config_name,
        "base_config": str(args.base_config.resolve()),
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "inference_config": str(args.infer_config.resolve()),
        "output_path": str(output_path.resolve()),
        "train_command": train_command,
        "eval_commands": eval_commands,
        "splits": split_meta,
    }
    metadata_path = split_root / "experiment.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nPrepared experiment: {experiment_name}")
    print(f"Experiment mode : {experiment_mode}")
    print(f"Max cases/dataset: {args.max_cases_per_dataset or 'unlimited'}")
    print(f"Generated config : {generated_config_path}")
    print(f"Metadata        : {metadata_path}")
    print(f"Commands        : {commands_path}")
    print()
    for dataset_name, info in split_meta.items():
        cap_note = ""
        if info["num_cases_after_cap"] < info["num_cases_total"]:
            cap_note = f" (capped from {info['num_cases_total']})"
        print(
            f"{dataset_name}: mode={info['mode']} "
            f"used={info['num_cases_after_cap']}{cap_note} "
            f"train={info['train_cases']} test={info['test_cases']}"
        )


if __name__ == "__main__":
    main()