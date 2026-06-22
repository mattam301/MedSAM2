#!/usr/bin/env python3
"""Prepare a 2D-vs-stacked finetuning comparison experiment.

For each eligible dataset (datasets whose consecutive slices are spatially
coherent), this script:

1.  Discovers all .npz slice files.
2.  Detects volume boundaries by looking for large image-intensity jumps
    between consecutive files.
3.  Groups consecutive slices within the same detected volume into
    pseudo-volumes of size ``--num-frames`` (default 8).
4.  Writes the grouped pseudo-volumes as new .npz files with
    ``imgs`` shape (D, H, W, C) and ``gts`` shape (D, H, W).
5.  Creates 70/30 train/test manifests for both the original 2D slices
    and the stacked pseudo-volumes.
6.  Generates two Hydra training configs:
    - one for 2D slice finetuning  (num_frames=1)
    - one for stacked finetuning   (num_frames=N)
7.  Writes an experiment.json with full metadata.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS_ROOT = REPO_ROOT / "data" / "new_datasets"
DEFAULT_BASE_CONFIG = REPO_ROOT / "sam2" / "configs" / "sam2.1_hiera_tiny_finetune512.yaml"
DEFAULT_INFER_CONFIG = REPO_ROOT / "sam2" / "configs" / "sam2.1_hiera_t512.yaml"
DEFAULT_BASE_CHECKPOINT = REPO_ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"
GENERATED_CONFIG_DIR = REPO_ROOT / "sam2" / "configs" / "generated"
HYDRA_GLOBAL_PACKAGE_HEADER = "# @package _global_\n\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip()).strip("_").lower()


def relative_case_id(npz_path: Path, dataset_dir: Path) -> str:
    return npz_path.relative_to(dataset_dir).with_suffix("").as_posix()


def write_manifest(path: Path, case_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(case_ids)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def split_ids(ids: list[str], train_ratio: float, seed: int):
    shuffled = ids[:]
    random.Random(seed).shuffle(shuffled)
    if train_ratio <= 0:
        return [], sorted(shuffled)
    if train_ratio >= 1:
        return sorted(shuffled), []
    if len(shuffled) <= 1:
        return sorted(shuffled), []
    n = max(1, min(int(len(shuffled) * train_ratio), len(shuffled) - 1))
    return sorted(shuffled[:n]), sorted(shuffled[n:])


def cap_ids(ids: list[str], max_count: Optional[int], seed: int) -> list[str]:
    if max_count is None or len(ids) <= max_count:
        return ids
    sampled = ids[:]
    random.Random(seed).shuffle(sampled)
    return sorted(sampled[:max_count])


# ---------------------------------------------------------------------------
# Volume boundary detection
# ---------------------------------------------------------------------------

def load_slice_stats(path: Path):
    """Return (mean_pixel_value, spatial_shape, mask_fraction) for one .npz."""
    with np.load(path, allow_pickle=True) as data:
        imgs = np.asarray(data["imgs"], dtype=np.float32)
        gts = np.asarray(data["gts"], dtype=np.float32)
    if imgs.ndim == 3:
        imgs = imgs.mean(axis=-1)  # collapse channel
    return imgs.mean(), imgs.shape, float((gts > 0).sum() / max(gts.size, 1))


def detect_volume_boundaries(
    files: list[Path],
    jump_threshold: float = 30.0,
) -> list[list[Path]]:
    """Split an ordered list of .npz paths into groups of consecutive slices
    that belong to the same detected volume.

    A new volume is started whenever:
    - the spatial shape changes, OR
    - the mean pixel intensity jumps by more than *jump_threshold*.
    """
    if not files:
        return []

    groups: list[list[Path]] = []
    current_group: list[Path] = [files[0]]
    prev_mean, prev_shape, _ = load_slice_stats(files[0])

    for f in files[1:]:
        cur_mean, cur_shape, _ = load_slice_stats(f)
        diff = abs(cur_mean - prev_mean)

        if cur_shape != prev_shape or diff > jump_threshold:
            groups.append(current_group)
            current_group = [f]
        else:
            current_group.append(f)

        prev_mean = cur_mean
        prev_shape = cur_shape

    if current_group:
        groups.append(current_group)

    return groups


# just for flare, alternate for ``detect volumn by boundary''
def group_by_case_prefix(files: list[Path], dataset_dir: Path) -> list[list[Path]]:
    """Group files by case ID (e.g., FLARE22_Tr_0001) extracted from filename."""
    from collections import defaultdict
    groups = defaultdict(list)
    
    for f in files:
        # Extract case ID: FLARE22_Tr_0001_0005.npz -> FLARE22_Tr_0001
        match = re.match(r"(.+)_\d{4}", f.stem)
        if match:
            case_id = match.group(1)
        else:
            case_id = f.stem  # fallback
        groups[case_id].append(f)
    
    # Sort each group by slice index
    result = []
    for case_id in sorted(groups.keys()):
        group = sorted(groups[case_id], key=lambda x: x.stem)
        result.append(group)
    return result

def group_files_by_case(files: list[Path]) -> list[list[Path]]:
    """Group slice files by case ID extracted from filename.
    
    Works for filenames like:
      FLARE22_Tr_0001_0005.npz  → case=FLARE22_Tr_0001, slice=0005
      case_001_slice_042.npz    → case=case_001, slice=042
    """
    from collections import defaultdict
    import re
    
    groups = defaultdict(list)
    for f in files:
        # Strip trailing _NNNN (slice index)
        m = re.match(r"^(.+?)[_-](\d+)$", f.stem)
        if m:
            case_id = m.group(1)
            slice_idx = int(m.group(2))
        else:
            case_id = f.stem
            slice_idx = 0
        groups[case_id].append((slice_idx, f))
    
    # Sort by slice index within each case
    result = []
    for case_id in sorted(groups.keys()):
        sorted_slices = sorted(groups[case_id], key=lambda x: x[0])
        result.append([f for _, f in sorted_slices])
    
    return result

# ---------------------------------------------------------------------------
# Pseudo-volume stacking
# ---------------------------------------------------------------------------

def stack_slices_to_volume(slice_paths: list[Path], image_channel_index: int) -> tuple:
 
    """Load N slice .npz files and stack them into a pseudo-volume.

    Returns
    -------
    imgs : ndarray, shape (N, H, W, C) or (N, H, W)
    gts  : ndarray, shape (N, H, W)
    """
    all_imgs = []
    all_gts = []

    for p in slice_paths:
        with np.load(p, allow_pickle=True) as data:
            img = np.asarray(data["imgs"])
            gt = np.asarray(data["gts"])

        # Handle channel selection for multi-channel images
        if img.ndim == 3 and img.shape[-1] > 1:
            img = img[:, :, image_channel_index: image_channel_index + 1]

        all_imgs.append(img)
        all_gts.append(gt)

    imgs = np.stack(all_imgs, axis=0)  # (N, H, W, C) or (N, H, W)
    gts = np.stack(all_gts, axis=0)    # (N, H, W)

    return imgs, gts


def inspect_npz_layout(npz_path: Path) -> tuple[tuple[int, ...], tuple[int, ...]]:
    with np.load(npz_path, allow_pickle=True) as data:
        imgs = np.asarray(data["imgs"])
        gts = np.asarray(data["gts"])
    return imgs.shape, gts.shape


def dataset_is_volume_layout(sample_npz: Path) -> bool:
    _, gts_shape = inspect_npz_layout(sample_npz)
    return len(gts_shape) == 3


def create_stacked_dataset(
    volume_groups: list[list[Path]],
    num_frames: int,
    output_dir: Path,
    dataset_name: str,
    image_channel_index: int,
) -> list[str]:
    """Create stacked pseudo-volume .npz files from detected volume groups.

    Each volume group is split into non-overlapping windows of size
    *num_frames*.  Remainders at the end of a group are kept if they
    have >= 2 slices (padded by repeating the last slice).

    Returns a sorted list of case IDs (relative to *output_dir*).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    case_ids = []
    vol_idx = 0

    for group in volume_groups:
        # Split group into windows of num_frames
        for start in range(0, len(group), num_frames):
            window = group[start: start + num_frames]

            if len(window) < 2:
                continue  # skip single-slice remainders

            # Pad short windows by repeating last slice
            while len(window) < num_frames:
                window.append(window[-1])

            try:
                imgs, gts = stack_slices_to_volume(window, image_channel_index)
            except Exception as e:
                print(f"  WARNING: skipping window at vol {vol_idx}: {e}")
                continue

            case_id = f"{slugify(dataset_name)}_vol{vol_idx:05d}"
            out_path = output_dir / f"{case_id}.npz"
            np.savez_compressed(out_path, imgs=imgs, gts=gts)
            case_ids.append(case_id)
            vol_idx += 1

    return sorted(case_ids)


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

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


def save_hydra_config(path: Path, cfg, OmegaConf) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    yaml_text = OmegaConf.to_yaml(cfg, resolve=False)
    path.write_text(HYDRA_GLOBAL_PACKAGE_HEADER + yaml_text, encoding="utf-8")


def generate_training_config(
    args,
    experiment_tag: str,
    dataset_entries: list[dict],
    num_frames: int,
    output_path: Path,
    OmegaConf,
):
    """Generate and save a Hydra training config."""
    cfg = OmegaConf.load(args.base_config)

    cfg.scratch.train_video_batch_size = args.batch_size
    cfg.scratch.num_train_workers = args.num_workers
    cfg.scratch.num_frames = num_frames
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

    if num_frames == 1:
        cfg.trainer.model.num_frames_to_correct_for_train = 1
        cfg.trainer.model.num_frames_to_correct_for_eval = 1
        cfg.trainer.model.num_init_cond_frames_for_train = 1
        cfg.trainer.model.num_init_cond_frames_for_eval = 1
        cfg.trainer.model.rand_frames_to_correct_for_train = False
        cfg.trainer.model.rand_init_cond_frames_for_train = False

    config_path = GENERATED_CONFIG_DIR / f"{experiment_tag}.yaml"
    save_hydra_config(config_path, cfg, OmegaConf)
    return config_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare 2D-vs-stacked finetuning comparison experiment."
    )
    p.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["npz_brats"],
        help="Dataset folder names to include.",
    )
    p.add_argument("--experiment-name", default="stack_comparison")
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=25)
    p.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    p.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    p.add_argument("--infer-config", type=Path, default=DEFAULT_INFER_CONFIG)
    p.add_argument("--num-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--max-num-objects", type=int, default=5)
    p.add_argument("--base-lr", type=float, default=5e-5)
    p.add_argument("--vision-lr", type=float, default=3e-5)
    p.add_argument("--dataset-multiplier", type=int, default=1)
    p.add_argument("--image-channel-index", type=int, default=0)
    p.add_argument("--max-cases-per-dataset", type=int, default=2000)
    p.add_argument(
        "--jump-threshold",
        type=float,
        default=30.0,
        help="Mean pixel intensity jump that triggers a volume boundary.",
    )
    p.add_argument(
    "--group-by-filename", action="store_true",
    help="Group slices by filename prefix (CASE_NNNN) instead of intensity detection. "
         "Recommended for FLARE22 and other datasets with consistent naming.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if not 0.0 <= args.train_ratio <= 1.0:
        raise SystemExit("--train-ratio must be between 0.0 and 1.0")

    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise SystemExit(
            "omegaconf is not installed. Activate the MedSAM2 environment first."
        ) from exc

    datasets_root = args.datasets_root.resolve()
    experiment_name = args.experiment_name
    exp_root = datasets_root / "_splits" / experiment_name
    stacked_data_root = datasets_root / "_stacked" / experiment_name
    manifest_root = exp_root / "manifests"
    exp_log_base = REPO_ROOT / "exp_log" / experiment_name

    slice_tag = f"{experiment_name}_slice"
    stack_tag = f"{experiment_name}_stacked"

    # -----------------------------------------------------------------------
    # Per-dataset processing
    # -----------------------------------------------------------------------
    slice_dataset_entries = []
    stack_dataset_entries = []
    split_meta: dict[str, dict] = {}

    for idx, ds_name in enumerate(sorted(args.datasets)):
        ds_dir = datasets_root / ds_name
        if not ds_dir.is_dir():
            raise SystemExit(f"Dataset directory not found: {ds_dir}")

        all_files = sorted(ds_dir.glob("*.npz"))

        if args.group_by_filename and args.max_cases_per_dataset:
            # Group by case first, then cap number of cases
            grouped = group_files_by_case(all_files)
            if len(grouped) > args.max_cases_per_dataset:
                random.Random(args.seed + idx).shuffle(grouped)
                grouped = grouped[:args.max_cases_per_dataset]
            # Flatten back to file list
            all_files = sorted([f for group in grouped for f in group])
            print(f"  Capped to {len(grouped)} cases ({len(all_files)} slices)")

        all_case_ids = sorted(relative_case_id(f, ds_dir) for f in all_files)

        if not all_case_ids:
            print(f"WARNING: {ds_name} has no .npz files, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {ds_name} ({len(all_case_ids)} slices)")
        print(f"{'='*60}")

        sample_npz = ds_dir / f"{all_case_ids[0]}.npz"
        is_volume_layout = dataset_is_volume_layout(sample_npz)
        if is_volume_layout:
            print("  Detected volume-layout NPZs; skipping pseudo-volume re-stacking.")

        # --- 2D slice split ---
        # After the above, BEFORE split_ids:
        if args.group_by_filename:
            # Split by case to prevent leakage
            grouped = group_files_by_case([ds_dir / f"{cid}.npz" for cid in all_case_ids])
            rng = random.Random(args.seed + idx)
            rng.shuffle(grouped)
            if args.train_ratio <= 0:
                n_train = 0
            elif args.train_ratio >= 1:
                n_train = len(grouped)
            else:
                n_train = max(1, min(int(len(grouped) * args.train_ratio), len(grouped) - 1))
            train_groups = grouped[:n_train]
            test_groups = grouped[n_train:]
            train_ids = sorted([relative_case_id(f, ds_dir) for g in train_groups for f in g])
            test_ids = sorted([relative_case_id(f, ds_dir) for g in test_groups for f in g])
        else:
            train_ids, test_ids = split_ids(all_case_ids, args.train_ratio, args.seed + idx)
        # train_ids, test_ids = split_ids(all_case_ids, args.train_ratio, args.seed + idx)

        slice_train_manifest = manifest_root / f"{slugify(ds_name)}_slice_train.txt"
        slice_test_manifest = manifest_root / f"{slugify(ds_name)}_slice_test.txt"
        write_manifest(slice_train_manifest, train_ids)
        write_manifest(slice_test_manifest, test_ids)

        slice_dataset_entries.append(
            build_dataset_entry(
                ds_dir, slice_train_manifest, args.dataset_multiplier, args.image_channel_index
            )
        )

        # --- Detect volume boundaries ---
        stacked_train_manifest = manifest_root / f"{slugify(ds_name)}_stacked_train.txt"
        if is_volume_layout:
            stacked_dir = ds_dir
            volume_groups = [[ds_dir / f"{cid}.npz"] for cid in train_ids]
            stacked_case_ids = train_ids
            write_manifest(stacked_train_manifest, stacked_case_ids)
            print("  Using source volumes directly for stacked training.")
        else:
            print(f"  Detecting volume boundaries (threshold={args.jump_threshold})...")
            train_files = sorted(ds_dir / f"{cid}.npz" for cid in train_ids)

            # NEW: Use filename-based grouping if requested
            if args.group_by_filename:
                print(f"  Grouping by filename pattern...")
                volume_groups = group_files_by_case(train_files)
            else:
                print(f"  Detecting volume boundaries (threshold={args.jump_threshold})...")
                volume_groups = detect_volume_boundaries(train_files, args.jump_threshold)

            group_sizes = [len(g) for g in volume_groups]
            if group_sizes:
                print(f"  Volume sizes: min={min(group_sizes)}, max={max(group_sizes)}, "
                      f"median={sorted(group_sizes)[len(group_sizes)//2]}")

            # --- Create stacked pseudo-volumes ---
            stacked_dir = stacked_data_root / ds_name
            print(f"  Creating stacked pseudo-volumes (window={args.num_frames})...")
            stacked_case_ids = create_stacked_dataset(
                volume_groups,
                args.num_frames,
                stacked_dir,
                ds_name,
                args.image_channel_index,
            )
            print(f"  Created {len(stacked_case_ids)} pseudo-volumes")

            write_manifest(stacked_train_manifest, stacked_case_ids)

        stack_dataset_entries.append(
            build_dataset_entry(
                stacked_dir, stacked_train_manifest, args.dataset_multiplier,
                args.image_channel_index,
            )
        )

        # --- Store metadata ---
        split_meta[ds_name] = {
            "dataset_dir": str(ds_dir.resolve()),
            "layout": "volume" if is_volume_layout else "slice",
            "num_slices_total": len(all_case_ids),
            "train_slices": len(train_ids),
            "test_slices": len(test_ids),
            "num_detected_volumes": len(volume_groups),
            "num_stacked_volumes": len(stacked_case_ids),
            "stacked_dir": str(stacked_dir.resolve()),
            "slice_train_manifest": str(slice_train_manifest.resolve()),
            "slice_test_manifest": str(slice_test_manifest.resolve()),
            "stacked_train_manifest": str(stacked_train_manifest.resolve()),
        }

    # -----------------------------------------------------------------------
    # Generate training configs
    # -----------------------------------------------------------------------
    print(f"\nGenerating training configs...")

    slice_output = exp_log_base / "slice"
    stack_output = exp_log_base / "stacked"

    slice_config = generate_training_config(
        args,
        slice_tag,
        slice_dataset_entries,
        num_frames=1,
        output_path=slice_output,
        OmegaConf=OmegaConf,
    )

    stack_config = generate_training_config(
        args,
        stack_tag,
        stack_dataset_entries,
        num_frames=args.num_frames,
        output_path=stack_output,
        OmegaConf=OmegaConf,
    )
    

    # -----------------------------------------------------------------------
    # Save experiment metadata
    # -----------------------------------------------------------------------
    metadata = {
        "experiment_name": experiment_name,
        "datasets": sorted(args.datasets),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "num_frames": args.num_frames,
        "jump_threshold": args.jump_threshold,
        "max_cases_per_dataset": args.max_cases_per_dataset,
        "image_channel_index": args.image_channel_index,
        "slice_config": str(slice_config.resolve()),
        "slice_config_name": f"configs/generated/{slice_config.name}",
        "stack_config": str(stack_config.resolve()),
        "stack_config_name": f"configs/generated/{stack_config.name}",
        "slice_output": str(slice_output.resolve()),
        "stack_output": str(stack_output.resolve()),
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "infer_config": str(args.infer_config.resolve()),
        "splits": split_meta,
    }

    metadata_path = exp_root / "experiment.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nExperiment prepared: {experiment_name}")
    print(f"  Metadata        : {metadata_path}")
    print(f"  Slice config    : {slice_config}")
    print(f"  Stacked config  : {stack_config}")

    for ds_name, info in split_meta.items():
        print(
            f"  {ds_name}: "
            f"slices={info['num_slices_total']} "
            f"train={info['train_slices']} test={info['test_slices']} "
            f"volumes={info['num_detected_volumes']} "
            f"stacked={info['num_stacked_volumes']}"
        )


if __name__ == "__main__":
    main()
