#!/usr/bin/env python3
"""
multi_axis_pipeline.py

Prepare, train, and evaluate multi-axis SAM2 experiments.

Core idea:
    Original stacked SAM2 uses only Z as the sequence/video dimension.
    This script also creates pseudo-videos along Y and X:

        Z-axis clips: imgs[z, :, :]
        Y-axis clips: imgs[:, y, :]
        X-axis clips: imgs[:, :, x]

Training variants generated:
    - all_axes : train on Z + Y + X pseudo-volumes
    - z_only   : train only on Z pseudo-volumes
    - y_only   : train only on Y pseudo-volumes
    - x_only   : train only on X pseudo-volumes

Evaluation variants:
    - base
    - z_only_causal
    - z_only_bidir
    - all_axes_causal
    - all_axes_bidir
    - optional y_only/x_only causal/bidir

Important:
    Your training entrypoint is:

        training/train.py

    NOT:

        sam2/sam2/training/train.py

Hydra config names must be passed as:

        -c configs/generated/<config_name_without_yaml>

    NOT:

        -c sam2/configs/generated/<config_name>.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np


# -------------------------------------------------------------------------
# Repo/root discovery
# -------------------------------------------------------------------------

def find_repo_root() -> Path:
    """Find MedSAM2 repo root robustly whether this file is in root/ or scripts/."""
    here = Path(__file__).resolve()
    candidates = [here.parent] + list(here.parents)
    for cand in candidates:
        if (cand / "training" / "train.py").is_file() and (cand / "sam2" / "configs").is_dir():
            return cand
    raise RuntimeError(
        "Could not find repo root. Expected to find training/train.py and sam2/configs."
    )


REPO_ROOT = find_repo_root()

DEFAULT_DATASETS_ROOT = REPO_ROOT / "data" / "new_datasets"
DEFAULT_BASE_CONFIG = REPO_ROOT / "sam2" / "configs" / "sam2.1_hiera_tiny_finetune512.yaml"
DEFAULT_INFER_CONFIG = REPO_ROOT / "sam2" / "configs" / "sam2.1_hiera_t512.yaml"
DEFAULT_BASE_CHECKPOINT = REPO_ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"
GENERATED_CONFIG_DIR = REPO_ROOT / "sam2" / "configs" / "generated"
HYDRA_GLOBAL_PACKAGE_HEADER = "# @package _global_\n\n"


# -------------------------------------------------------------------------
# Per-dataset Z caps
# -------------------------------------------------------------------------

MAX_Z_SLICES = {
    "flare": 100,
    "npz_brats": 150,
    "brainmets": 150,
    "lgg": 10,
}
DEFAULT_MAX_Z = 150


# -------------------------------------------------------------------------
# Model specs for comparison
# -------------------------------------------------------------------------

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "base": {
        "label": "Base",
        "input": "slice",
        "kind": "base",
    },
    "z_only_causal": {
        "label": "Z-only+Causal",
        "input": "Z clip",
        "kind": "causal",
        "family": "z_only",
        "eval_axis": "z",
    },
    "z_only_bidir": {
        "label": "Z-only+Bidir",
        "input": "Z clip",
        "kind": "bidir",
        "family": "z_only",
        "eval_axis": "z",
    },
    "y_only_causal": {
        "label": "Y-only+Causal",
        "input": "Y clip",
        "kind": "causal",
        "family": "y_only",
        "eval_axis": "y",
    },
    "y_only_bidir": {
        "label": "Y-only+Bidir",
        "input": "Y clip",
        "kind": "bidir",
        "family": "y_only",
        "eval_axis": "y",
    },
    "x_only_causal": {
        "label": "X-only+Causal",
        "input": "X clip",
        "kind": "causal",
        "family": "x_only",
        "eval_axis": "x",
    },
    "x_only_bidir": {
        "label": "X-only+Bidir",
        "input": "X clip",
        "kind": "bidir",
        "family": "x_only",
        "eval_axis": "x",
    },
    "all_axes_causal": {
        "label": "AllAxes+Causal",
        "input": "Z clip",
        "kind": "causal",
        "family": "all_axes",
    },
    "all_axes_bidir": {
        "label": "AllAxes+Bidir",
        "input": "Z clip",
        "kind": "bidir",
        "family": "all_axes",
    },
    "all_axes_fused_causal": {
        "label": "AllAxes+Fused+Causal",
        "input": "XYZ volume",
        "kind": "fused_causal",
        "family": "all_axes",
    },
    "all_axes_fused_bidir": {
        "label": "AllAxes+Fused+Bidir",
        "input": "XYZ volume",
        "kind": "fused_bidir",
        "family": "all_axes",
    },
}


# -------------------------------------------------------------------------
# Generic helpers
# -------------------------------------------------------------------------

def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip()).strip("_").lower()


def natural_sort_key(value: str | Path):
    value = str(value)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", value)]


def run(cmd: list[str], env: Optional[dict[str, str]] = None) -> None:
    print("\n$ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def write_manifest(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(ids)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def relative_npz_id(npz_path: Path, dataset_dir: Path) -> str:
    return npz_path.relative_to(dataset_dir).with_suffix("").as_posix()


def split_ids(ids: list[str], train_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    ids = ids[:]
    random.Random(seed).shuffle(ids)

    if train_ratio <= 0:
        return [], sorted(ids)
    if train_ratio >= 1:
        return sorted(ids), []
    if len(ids) <= 1:
        return sorted(ids), []

    n_train = max(1, min(int(len(ids) * train_ratio), len(ids) - 1))
    return sorted(ids[:n_train]), sorted(ids[n_train:])


def hydra_config_name(config_path: Path) -> str:
    """
    Convert:
        /repo/sam2/configs/generated/foo.yaml

    to Hydra config name:
        configs/generated/foo
    """
    return f"configs/generated/{config_path.stem}"


# -------------------------------------------------------------------------
# Case grouping / loading
# -------------------------------------------------------------------------

def group_files_by_case(files: list[Path]) -> dict[str, list[Path]]:
    """
    Group slice files by case ID.

    Examples:
        FLARE22_Tr_0001_0005.npz -> case FLARE22_Tr_0001, slice 0005
        patientA_042.npz         -> case patientA, slice 042
    """
    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)

    for f in files:
        m = re.match(r"^(.+?)[_-](\d+)$", f.stem)
        if m:
            case_id = m.group(1)
            slice_idx = int(m.group(2))
        else:
            case_id = f.stem
            slice_idx = 0

        groups[case_id].append((slice_idx, f))

    result: dict[str, list[Path]] = {}
    for case_id in sorted(groups.keys(), key=natural_sort_key):
        result[case_id] = [
            f for _, f in sorted(groups[case_id], key=lambda x: x[0])
        ]
    return result


def load_case_volume(
    case_files: list[Path],
    channel_index: int = 0,
    max_slices: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a case as a 3D volume.

    Supports:
        slice-layout:
            each file imgs=(H,W,C) or (H,W), gts=(H,W)

        volume-layout:
            one file imgs=(D,H,W,C) or (D,H,W), gts=(D,H,W)

    Returns:
        imgs: (D,H,W), float32
        gts : (D,H,W), float32
    """
    if not case_files:
        raise ValueError("Empty case_files")

    # Check if first file is already volume-layout
    with np.load(case_files[0], allow_pickle=True) as data:
        first_imgs = np.asarray(data["imgs"])
        first_gts = np.asarray(data["gts"])

    if first_gts.ndim == 3:
        imgs = first_imgs.astype(np.float32)
        gts = first_gts.astype(np.float32)

        if imgs.ndim == 4:
            imgs = imgs[..., channel_index]
        elif imgs.ndim != 3:
            raise ValueError(f"Unsupported volume imgs shape: {imgs.shape}")

        if max_slices is not None:
            imgs = imgs[:max_slices]
            gts = gts[:max_slices]

        return imgs, gts

    # Otherwise stack slice-layout files
    imgs_list: list[np.ndarray] = []
    gts_list: list[np.ndarray] = []

    for idx, f in enumerate(case_files):
        if max_slices is not None and idx >= max_slices:
            break

        with np.load(f, allow_pickle=True) as data:
            img = np.asarray(data["imgs"], dtype=np.float32)
            gt = np.asarray(data["gts"], dtype=np.float32)

        if img.ndim == 3:
            img = img[..., channel_index] if img.shape[-1] > 1 else img[..., 0]
        elif img.ndim != 2:
            raise ValueError(f"Unsupported slice imgs shape {img.shape} in {f}")

        if gt.ndim != 2:
            raise ValueError(f"Unsupported slice gts shape {gt.shape} in {f}")

        imgs_list.append(img)
        gts_list.append(gt)

    if not imgs_list:
        raise ValueError("No slices loaded")

    return np.stack(imgs_list, axis=0), np.stack(gts_list, axis=0)


# -------------------------------------------------------------------------
# Resize helpers
# -------------------------------------------------------------------------

def resize_image_2d(frame: np.ndarray, target_size: int) -> np.ndarray:
    """Resize image frame preserving float intensity as much as possible."""
    if frame.shape == (target_size, target_size):
        return frame.astype(np.float32)

    try:
        import torch
        import torch.nn.functional as F

        x = torch.as_tensor(frame, dtype=torch.float32)[None, None]
        y = F.interpolate(
            x,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )
        return y[0, 0].cpu().numpy().astype(np.float32)
    except Exception:
        from PIL import Image

        fmin, fmax = float(frame.min()), float(frame.max())
        if fmax > fmin:
            tmp = ((frame - fmin) / (fmax - fmin) * 255).astype(np.uint8)
        else:
            tmp = np.zeros_like(frame, dtype=np.uint8)

        out = np.array(
            Image.fromarray(tmp).resize(
                (target_size, target_size),
                Image.BILINEAR,
            )
        ).astype(np.float32)

        if fmax > fmin:
            out = out / 255.0 * (fmax - fmin) + fmin
        return out.astype(np.float32)


def resize_mask_2d(mask: np.ndarray, target_size: int) -> np.ndarray:
    """Resize mask using nearest neighbor."""
    if mask.shape == (target_size, target_size):
        return mask.astype(np.float32)

    try:
        import torch
        import torch.nn.functional as F

        x = torch.as_tensor(mask, dtype=torch.float32)[None, None]
        y = F.interpolate(
            x,
            size=(target_size, target_size),
            mode="nearest",
        )
        return y[0, 0].cpu().numpy().astype(np.float32)
    except Exception:
        from PIL import Image

        out = np.array(
            Image.fromarray(mask.astype(np.uint8)).resize(
                (target_size, target_size),
                Image.NEAREST,
            )
        )
        return out.astype(np.float32)


# -------------------------------------------------------------------------
# Multi-axis pseudo-volume generation
# -------------------------------------------------------------------------

def extract_axis_frame(
    imgs: np.ndarray,
    gts: np.ndarray,
    axis: int,
    idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract one 2D frame from a 3D volume.

    imgs/gts shape: (D,H,W)

    axis:
        0 -> Z: imgs[idx,:,:]    shape (H,W)
        1 -> Y: imgs[:,idx,:]    shape (D,W)
        2 -> X: imgs[:,:,idx]    shape (D,H)
    """
    if axis == 0:
        return imgs[idx], gts[idx]
    if axis == 1:
        return imgs[:, idx, :], gts[:, idx, :]
    if axis == 2:
        return imgs[:, :, idx], gts[:, :, idx]
    raise ValueError(f"Unknown axis: {axis}")


def create_pseudo_volumes_along_axis(
    imgs: np.ndarray,
    gts: np.ndarray,
    axis: int,
    num_frames: int,
    stride: int,
    output_dir: Path,
    case_id: str,
    axis_name: str,
    target_size: int = 512,
    min_mask_frames: int = 1,
) -> int:
    """
    Create sliding-window pseudo-volumes along one axis.

    Output imgs shape:
        (num_frames, target_size, target_size)

    Output gts shape:
        (num_frames, target_size, target_size)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    D, H, W = imgs.shape
    axis_len = [D, H, W][axis]

    if axis_len < num_frames:
        return 0

    count = 0

    for start in range(0, axis_len - num_frames + 1, stride):
        frames_img: list[np.ndarray] = []
        frames_gt: list[np.ndarray] = []

        for offset in range(num_frames):
            idx = start + offset
            frame, mask = extract_axis_frame(imgs, gts, axis, idx)

            frame = resize_image_2d(frame, target_size)
            mask = resize_mask_2d(mask, target_size)

            frames_img.append(frame)
            frames_gt.append(mask)

        vol_img = np.stack(frames_img, axis=0)
        vol_gt = np.stack(frames_gt, axis=0)

        frames_with_mask = int((vol_gt > 0).any(axis=(1, 2)).sum())
        if frames_with_mask < min_mask_frames:
            continue

        out_name = f"{slugify(case_id)}_{axis_name}{count:04d}.npz"
        plane_shape = extract_axis_frame(imgs, gts, axis, start)[0].shape
        np.savez_compressed(
            output_dir / out_name,
            imgs=vol_img,
            gts=vol_gt,
            case_id=np.array(case_id),
            axis=np.array(axis_name),
            axis_index=np.array(axis, dtype=np.int16),
            start=np.array(start, dtype=np.int32),
            volume_shape=np.array(imgs.shape, dtype=np.int32),
            plane_shape=np.array(plane_shape, dtype=np.int32),
            target_size=np.array(target_size, dtype=np.int32),
        )
        count += 1

    return count


def generate_multi_axis_for_dataset(
    ds_name: str,
    train_case_groups: dict[str, list[Path]],
    output_root: Path,
    axes: list[str],
    num_frames: int,
    stride: int,
    target_size: int,
    channel_index: int,
    max_slices_per_case: int,
    min_mask_frames: int,
    overwrite: bool,
) -> dict[str, int]:
    """Generate Z/Y/X pseudo-volumes for one dataset."""
    if overwrite and output_root.exists():
        print(f"  Removing old generated data: {output_root}")
        shutil.rmtree(output_root)

    axis_to_int = {"z": 0, "y": 1, "x": 2}
    counts = {"z": 0, "y": 0, "x": 0}

    print(f"  Generating multi-axis data for {len(train_case_groups)} train cases")
    print(f"  max_slices_per_case={max_slices_per_case}")

    for case_idx, (case_id, case_files) in enumerate(train_case_groups.items(), start=1):
        print(f"    [{case_idx}/{len(train_case_groups)}] {case_id} ({len(case_files)} files)")

        try:
            imgs, gts = load_case_volume(
                case_files,
                channel_index=channel_index,
                max_slices=max_slices_per_case,
            )
        except Exception as e:
            print(f"      WARNING: failed to load case {case_id}: {e}")
            continue

        if (gts > 0).sum() == 0:
            print("      Skipping: no mask")
            continue

        print(f"      loaded volume: {imgs.shape}, mask voxels={(gts > 0).sum():,}")

        for axis_name in axes:
            axis = axis_to_int[axis_name]
            axis_dir = output_root / axis_name

            n = create_pseudo_volumes_along_axis(
                imgs=imgs,
                gts=gts,
                axis=axis,
                num_frames=num_frames,
                stride=stride,
                output_dir=axis_dir,
                case_id=case_id,
                axis_name=axis_name,
                target_size=target_size,
                min_mask_frames=min_mask_frames,
            )
            counts[axis_name] += n
            print(f"      {axis_name}: {n} clips")

    print("  Dataset total:")
    for ax in axes:
        print(f"    {ax}: {counts[ax]}")
    print(f"    combined: {sum(counts.values())}")

    return counts


# -------------------------------------------------------------------------
# Hydra config generation
# -------------------------------------------------------------------------

def build_dataset_entry(
    folder: Path,
    file_list_txt: Path,
    multiplier: int,
    image_channel_index: int,
) -> dict[str, Any]:
    """
    Build a standard VOSDataset entry.

    Important:
        Do NOT include axis=... here.
        VOSDataset.__init__ does not accept axis.
    """
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
        "multiplier": int(multiplier),
    }


def save_hydra_config(path: Path, cfg, OmegaConf) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = OmegaConf.to_yaml(cfg, resolve=False)
    path.write_text(HYDRA_GLOBAL_PACKAGE_HEADER + text, encoding="utf-8")


def generate_training_config(
    args: argparse.Namespace,
    config_stem: str,
    dataset_entries: list[dict[str, Any]],
    output_path: Path,
    OmegaConf,
) -> Path:
    """Generate one training config."""
    if not dataset_entries:
        raise RuntimeError(f"No dataset entries for config {config_stem}")

    cfg = OmegaConf.load(args.base_config)

    cfg.scratch.train_video_batch_size = args.batch_size
    cfg.scratch.num_train_workers = args.num_workers
    cfg.scratch.num_frames = args.num_frames
    cfg.scratch.max_num_objects = args.max_num_objects
    cfg.scratch.num_epochs = args.num_epochs
    cfg.scratch.base_lr = args.base_lr
    cfg.scratch.vision_lr = args.vision_lr
    cfg.scratch.paired_axis_training = bool(args.paired_axis_training)
    cfg.scratch.consistency_loss_weight = float(args.consistency_loss_weight)

    if args.paired_axis_training:
        cfg.trainer._target_ = "training.multi_axis_trainer.MultiAxisTrainer"
        cfg.trainer.consistency_loss_weight = float(args.consistency_loss_weight)

    cfg.trainer.data.train.datasets[0].dataset.datasets = OmegaConf.create(dataset_entries)

    cfg.trainer.checkpoint.model_weight_initializer.state_dict.checkpoint_path = str(
        args.base_checkpoint.resolve()
    )

    cfg.launcher.experiment_log_dir = str(output_path.resolve())
    cfg.launcher.num_nodes = 1
    cfg.submitit.use_cluster = False

    cfg.trainer.model.num_frames_to_correct_for_train = args.num_frames
    cfg.trainer.model.num_frames_to_correct_for_eval = args.num_frames
    cfg.trainer.model.num_init_cond_frames_for_train = min(3, args.num_frames)
    cfg.trainer.model.num_init_cond_frames_for_eval = min(3, args.num_frames)

    config_path = GENERATED_CONFIG_DIR / f"{config_stem}.yaml"
    save_hydra_config(config_path, cfg, OmegaConf)
    return config_path


# -------------------------------------------------------------------------
# Evaluation clip building
# -------------------------------------------------------------------------

def extract_case_and_index(rel: str) -> tuple[str, int]:
    path = Path(rel)
    stem = path.name
    parent = str(path.parent) if str(path.parent) != "." else ""

    match = re.match(r"^(.*?)(\d+)$", stem)
    prefix = match.group(1).rstrip("_-") if match else stem
    index = int(match.group(2)) if match else 0

    case_key = f"{parent}/{prefix}" if parent and prefix else (parent or prefix or stem)
    return case_key, index


def select_window(
    items: list[tuple[int, Path]],
    start: int,
    window_size: int,
    fixed_interval: int,
    interval_mode: str,
) -> list[Path]:
    if interval_mode == "dynamic":
        remaining = max(len(items) - start, 1)
        interval = max(1, math.ceil((remaining - 1) / max(window_size - 1, 1)))
    else:
        interval = max(1, fixed_interval)

    selected: list[Path] = []
    last_index = len(items) - 1

    for offset in range(window_size):
        selected.append(items[min(start + offset * interval, last_index)][1])

    return selected


def build_clip_dataset(
    dataset_dir: Path,
    manifest_path: Path,
    out_npz_dir: Path,
    out_manifest: Path,
    window_size: int,
    window_stride: int,
    slice_interval: int,
    interval_mode: str,
    force: bool = False,
) -> int:
    """
    Build Z-axis evaluation clips from original test slice manifest.
    """
    if out_manifest.exists() and not force:
        lines = [
            x.strip()
            for x in out_manifest.read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
        return len(lines)

    out_npz_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        line.strip()
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise RuntimeError(f"Empty manifest: {manifest_path}")

    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)

    for rel in lines:
        key, index = extract_case_and_index(rel)
        suffix = "" if rel.endswith(".npz") else ".npz"
        groups[key].append((index, dataset_dir / f"{rel}{suffix}"))

    manifest_entries: list[str] = []

    for key, items in sorted(groups.items()):
        items.sort(key=lambda x: x[0])
        safe_key = re.sub(r"[^a-zA-Z0-9._/-]+", "_", key).strip("/")

        for clip_id, start in enumerate(range(0, len(items), window_stride)):
            window_paths = select_window(
                items,
                start=start,
                window_size=window_size,
                fixed_interval=slice_interval,
                interval_mode=interval_mode,
            )

            imgs_list: list[np.ndarray] = []
            gts_list: list[np.ndarray] = []

            for npz_path in window_paths:
                with np.load(npz_path, allow_pickle=True) as data:
                    imgs_list.append(np.asarray(data["imgs"]))
                    gts_list.append(np.asarray(data["gts"]))

            clip_rel = f"{safe_key}/clip_{clip_id:04d}"
            clip_out = out_npz_dir / f"{clip_rel}.npz"
            clip_out.parent.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(
                clip_out,
                imgs=np.stack(imgs_list, axis=0),
                gts=np.stack(gts_list, axis=0),
            )

            manifest_entries.append(clip_rel)

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text("\n".join(manifest_entries) + "\n", encoding="utf-8")

    return len(manifest_entries)


# -------------------------------------------------------------------------
# Metrics / comparison
# -------------------------------------------------------------------------

def metric_value(metrics: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key in metrics and metrics[key] is not None:
            return float(metrics[key])
    return float("nan")


def fmt(value: float) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) and not math.isnan(value) else "N/A"


def collect_metrics(eval_root: Path, model_order: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    by_dataset: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)

    for tag in model_order:
        for metrics_path in sorted((eval_root / tag).rglob("summary.json")):
            metrics = load_json(metrics_path)
            dataset = metrics_path.parent.name

            by_dataset[dataset][tag] = {
                "dice": metric_value(metrics, "dice", "dsc", "mean_dice", "meanDice"),
                "iou": metric_value(metrics, "iou", "jaccard", "mean_iou", "meanIoU"),
                "smoothness": metric_value(metrics, "smoothness", "mean_smoothness"),
            }

    return by_dataset


def write_comparison(eval_root: Path, comparison_dir: Path, model_order: list[str], name: str) -> None:
    comparison_dir.mkdir(parents=True, exist_ok=True)
    by_dataset = collect_metrics(eval_root, model_order)

    if not by_dataset:
        print("No metrics found; comparison skipped.")
        return

    width_ds = 24
    width_model = 26
    sep = "=" * 95
    thin = "-" * 95

    print(f"\n{sep}")
    print("RESULTS")
    print(sep)
    print(
        f"{'Dataset':<{width_ds}} {'Model':<{width_model}} "
        f"{'Input':<10} {'Dice':>8} {'IoU':>8} {'Smooth':>8}"
    )
    print(thin)

    for dataset in sorted(by_dataset):
        for tag in model_order:
            if tag not in by_dataset[dataset]:
                continue

            spec = MODEL_SPECS[tag]
            row = by_dataset[dataset][tag]

            print(
                f"{dataset:<{width_ds}} {spec['label']:<{width_model}} "
                f"{spec['input']:<10} {fmt(row['dice']):>8} "
                f"{fmt(row['iou']):>8} {fmt(row['smoothness']):>8}"
            )
        print(thin)

    print("\nImprovement over Z-only+Causal if available")
    print(thin)
    print(f"{'Dataset':<{width_ds}} {'Model':<{width_model}} {'Delta Dice':>12} {'Delta IoU':>12}")

    for dataset in sorted(by_dataset):
        if "z_only_causal" not in by_dataset[dataset]:
            continue

        base = by_dataset[dataset]["z_only_causal"]

        for tag in model_order:
            if tag == "z_only_causal" or tag not in by_dataset[dataset]:
                continue

            row = by_dataset[dataset][tag]
            print(
                f"{dataset:<{width_ds}} {MODEL_SPECS[tag]['label']:<{width_model}} "
                f"{row['dice'] - base['dice']:+12.4f} "
                f"{row['iou'] - base['iou']:+12.4f}"
            )
        print(thin)

    print("\nCross-dataset averages")
    print(thin)
    print(f"{'Model':<{width_model}} {'Mean Dice':>12} {'Mean IoU':>12} {'Mean Smooth':>12}")

    for tag in model_order:
        rows = [models[tag] for models in by_dataset.values() if tag in models]
        if not rows:
            continue

        means = {}
        for key in ("dice", "iou", "smoothness"):
            valid = [row[key] for row in rows if not math.isnan(row[key])]
            means[key] = sum(valid) / max(len(valid), 1)

        print(
            f"{MODEL_SPECS[tag]['label']:<{width_model}} "
            f"{fmt(means['dice']):>12} "
            f"{fmt(means['iou']):>12} "
            f"{fmt(means['smoothness']):>12}"
        )

    out = comparison_dir / f"{name}_comparison.json"
    write_json(
        out,
        {
            "model_info": {
                tag: {
                    "label": MODEL_SPECS[tag]["label"],
                    "input": MODEL_SPECS[tag]["input"],
                    "kind": MODEL_SPECS[tag]["kind"],
                }
                for tag in model_order
            },
            "by_dataset": by_dataset,
        },
    )

    print(f"\nSaved comparison: {out}")


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare/train/evaluate multi-axis SAM2 experiments."
    )

    p.add_argument("--mode", choices=["prepare", "train", "eval", "train-eval", "all"], default="prepare")
    p.add_argument("--prepare", choices=["auto", "always", "never"], default="auto")

    p.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    p.add_argument("--datasets", nargs="+", default=["flare"])
    p.add_argument("--experiment-name", default="multi_axis")
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=25)

    p.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    p.add_argument("--infer-config", type=Path, default=DEFAULT_INFER_CONFIG)
    p.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)

    # Training hyperparams
    p.add_argument("--num-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--max-num-objects", type=int, default=5)
    p.add_argument("--base-lr", type=float, default=5e-5)
    p.add_argument("--vision-lr", type=float, default=3e-5)
    p.add_argument("--paired-axis-training", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--consistency-loss-weight", type=float, default=0.0)

    # Data generation
    p.add_argument("--axes", nargs="+", default=["z", "y", "x"], choices=["z", "y", "x"])
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--target-size", type=int, default=512)
    p.add_argument("--channel-index", type=int, default=0)
    p.add_argument("--max-slices-per-case", type=int, default=None)
    p.add_argument("--max-cases-per-dataset", type=int, default=None)
    p.add_argument("--min-mask-frames", type=int, default=1)
    p.add_argument("--dataset-multiplier", type=int, default=1)
    p.add_argument("--balance-axes", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--max-axis-multiplier", type=int, default=10)
    p.add_argument("--overwrite-generated", action=argparse.BooleanOptionalAction, default=True)

    # Train/eval execution
    p.add_argument("--python", default=os.environ.get("PYTHON", sys.executable))
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--num-nodes", type=int, default=1)
    p.add_argument("--force-train", action="store_true")

    p.add_argument(
        "--train-models",
        nargs="+",
        default=["z_only", "all_axes"],
        choices=["z_only", "y_only", "x_only", "all_axes"],
    )

    p.add_argument(
        "--eval-models",
        nargs="+",
        default=["base", "z_only", "all_axes"],
        choices=["base", "z_only", "y_only", "x_only", "all_axes"],
    )

    p.add_argument("--eval-causal", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval-bidir", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval-fused", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--fusion", choices=["mean_logit", "mean_prob", "agreement"], default="mean_logit")

    p.add_argument("--eval-root-name", default=None)
    p.add_argument("--eval-window-size", type=int, default=None)
    p.add_argument("--eval-window-stride", type=int, default=None)
    p.add_argument("--context-slice-interval", type=int, default=1)
    p.add_argument("--context-interval-mode", choices=["fixed", "dynamic"], default="dynamic")
    p.add_argument("--memory-stride", type=int, default=1)
    p.add_argument("--min-component-volume", type=int, default=100)
    p.add_argument("--disable-3d-filter", action="store_true")
    p.add_argument("--force-rebuild-eval-clips", action="store_true")

    # Optional checkpoint overrides
    p.add_argument("--z-checkpoint", type=Path, default=None)
    p.add_argument("--y-checkpoint", type=Path, default=None)
    p.add_argument("--x-checkpoint", type=Path, default=None)
    p.add_argument("--all-axes-checkpoint", type=Path, default=None)

    args = p.parse_args()

    if args.eval_window_size is None:
        args.eval_window_size = args.num_frames
    if args.eval_window_stride is None:
        args.eval_window_stride = args.eval_window_size

    return args


# -------------------------------------------------------------------------
# Preparation
# -------------------------------------------------------------------------

def metadata_is_usable(path: Path) -> bool:
    if not path.exists():
        return False

    try:
        meta = load_json(path)
    except Exception:
        return False

    if not meta.get("splits"):
        return False
    if not meta.get("config_names"):
        return False
    if not meta.get("outputs"):
        return False

    return True


def should_prepare(args: argparse.Namespace, meta_path: Path) -> bool:
    if args.prepare == "always":
        return True
    if args.prepare == "never":
        return False
    return not metadata_is_usable(meta_path)


def prepare_experiment(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise SystemExit("omegaconf is required in the MedSAM2 environment.") from exc

    datasets_root = args.datasets_root.resolve()
    experiment_name = args.experiment_name

    split_root = datasets_root / "_splits" / experiment_name
    manifest_root = split_root / "manifests"
    multi_axis_root = datasets_root / "_multi_axis" / experiment_name
    exp_log_base = REPO_ROOT / "exp_log" / experiment_name

    active_axes = [a for a in args.axes if a in {"z", "y", "x"}]

    print(f"\n{'=' * 70}")
    print("PREPARING MULTI-AXIS EXPERIMENT")
    print(f"{'=' * 70}")
    print(f"experiment : {experiment_name}")
    print(f"datasets   : {args.datasets}")
    print(f"axes       : {active_axes}")
    print(f"frames     : {args.num_frames}")
    print(f"stride     : {args.stride}")
    print(f"target     : {args.target_size}")
    print(f"{'=' * 70}\n")

    split_meta: dict[str, Any] = {}
    axis_entry_specs: list[dict[str, Any]] = []
    global_axis_counts = {"z": 0, "y": 0, "x": 0}

    for ds_idx, ds_name in enumerate(sorted(args.datasets)):
        ds_dir = datasets_root / ds_name
        if not ds_dir.is_dir():
            print(f"WARNING: missing dataset directory: {ds_dir}")
            continue

        all_files = sorted(ds_dir.glob("*.npz"), key=lambda x: natural_sort_key(x.name))
        if not all_files:
            print(f"WARNING: no npz files in {ds_dir}")
            continue

        case_groups = group_files_by_case(all_files)
        all_case_keys = sorted(case_groups.keys(), key=natural_sort_key)

        if args.max_cases_per_dataset is not None and len(all_case_keys) > args.max_cases_per_dataset:
            rng = random.Random(args.seed + ds_idx)
            sampled = all_case_keys[:]
            rng.shuffle(sampled)
            all_case_keys = sorted(sampled[: args.max_cases_per_dataset], key=natural_sort_key)
            case_groups = {k: case_groups[k] for k in all_case_keys}

        train_case_keys, test_case_keys = split_ids(
            all_case_keys,
            train_ratio=args.train_ratio,
            seed=args.seed + ds_idx,
        )

        train_case_groups = {k: case_groups[k] for k in train_case_keys}
        test_case_groups = {k: case_groups[k] for k in test_case_keys}

        train_slice_ids = sorted(
            relative_npz_id(f, ds_dir)
            for k in train_case_keys
            for f in case_groups[k]
        )
        test_slice_ids = sorted(
            relative_npz_id(f, ds_dir)
            for k in test_case_keys
            for f in case_groups[k]
        )

        ds_slug = slugify(ds_name)

        slice_train_manifest = manifest_root / f"{ds_slug}_slice_train.txt"
        slice_test_manifest = manifest_root / f"{ds_slug}_slice_test.txt"

        write_manifest(slice_train_manifest, train_slice_ids)
        write_manifest(slice_test_manifest, test_slice_ids)

        max_z = (
            args.max_slices_per_case
            if args.max_slices_per_case is not None
            else MAX_Z_SLICES.get(ds_name, DEFAULT_MAX_Z)
        )

        print(f"\nDataset: {ds_name}")
        print(f"  cases total : {len(all_case_keys)}")
        print(f"  train cases : {len(train_case_keys)}")
        print(f"  test cases  : {len(test_case_keys)}")
        print(f"  train slices: {len(train_slice_ids)}")
        print(f"  test slices : {len(test_slice_ids)}")
        print(f"  max_z       : {max_z}")

        axis_counts = generate_multi_axis_for_dataset(
            ds_name=ds_name,
            train_case_groups=train_case_groups,
            output_root=multi_axis_root / ds_name,
            axes=active_axes,
            num_frames=args.num_frames,
            stride=args.stride,
            target_size=args.target_size,
            channel_index=args.channel_index,
            max_slices_per_case=max_z,
            min_mask_frames=args.min_mask_frames,
            overwrite=args.overwrite_generated,
        )

        axis_manifests: dict[str, str] = {}
        axis_dirs: dict[str, str] = {}

        for axis_name in active_axes:
            axis_dir = multi_axis_root / ds_name / axis_name
            axis_dirs[axis_name] = str(axis_dir.resolve())

            if axis_dir.is_dir():
                axis_ids = sorted(
                    [f.relative_to(axis_dir).with_suffix("").as_posix() for f in axis_dir.rglob("*.npz")],
                    key=natural_sort_key,
                )
            else:
                axis_ids = []

            global_axis_counts[axis_name] += len(axis_ids)

            axis_manifest = manifest_root / f"{ds_slug}_{axis_name}_train.txt"
            write_manifest(axis_manifest, axis_ids)
            axis_manifests[axis_name] = str(axis_manifest.resolve())

            if axis_ids:
                axis_entry_specs.append(
                    {
                        "dataset": ds_name,
                        "axis": axis_name,
                        "folder": axis_dir,
                        "manifest": axis_manifest,
                        "count": len(axis_ids),
                    }
                )

        split_meta[ds_name] = {
            "dataset_dir": str(ds_dir.resolve()),
            "num_cases_total": len(all_case_keys),
            "train_cases": len(train_case_keys),
            "test_cases": len(test_case_keys),
            "train_slices": len(train_slice_ids),
            "test_slices": len(test_slice_ids),
            "max_z": max_z,
            "slice_train_manifest": str(slice_train_manifest.resolve()),
            "slice_test_manifest": str(slice_test_manifest.resolve()),
            "axis_counts": axis_counts,
            "axis_dirs": axis_dirs,
            "axis_manifests": axis_manifests,
        }

    if not axis_entry_specs:
        raise RuntimeError("No multi-axis training clips were generated.")

    # Axis balancing for all_axes config
    axis_multipliers = {"z": args.dataset_multiplier, "y": args.dataset_multiplier, "x": args.dataset_multiplier}

    if args.balance_axes:
        nonzero = {ax: c for ax, c in global_axis_counts.items() if c > 0}
        if nonzero:
            max_count = max(nonzero.values())
            for ax, count in nonzero.items():
                axis_multipliers[ax] = min(
                    args.max_axis_multiplier,
                    max(1, math.ceil(max_count / max(count, 1))),
                )

    print("\nGlobal generated clip counts:")
    for ax in active_axes:
        print(f"  {ax}: {global_axis_counts[ax]} clips, all_axes multiplier={axis_multipliers[ax]}")

    # Build dataset entries
    all_axes_entries: list[dict[str, Any]] = []
    axis_only_entries: dict[str, list[dict[str, Any]]] = {"z": [], "y": [], "x": []}

    for spec in axis_entry_specs:
        ax = spec["axis"]
        folder = Path(spec["folder"])
        manifest = Path(spec["manifest"])

        all_axes_entries.append(
            build_dataset_entry(
                folder=folder,
                file_list_txt=manifest,
                multiplier=axis_multipliers[ax],
                image_channel_index=args.channel_index,
            )
        )

        axis_only_entries[ax].append(
            build_dataset_entry(
                folder=folder,
                file_list_txt=manifest,
                multiplier=args.dataset_multiplier,
                image_channel_index=args.channel_index,
            )
        )

    # Output locations
    outputs = {
        "all_axes": str((exp_log_base / "multi_axis").resolve()),
        "z_only": str((exp_log_base / "z_only").resolve()),
        "y_only": str((exp_log_base / "y_only").resolve()),
        "x_only": str((exp_log_base / "x_only").resolve()),
    }

    # Generate configs
    config_paths: dict[str, str] = {}
    config_names: dict[str, str] = {}

    all_axes_config = generate_training_config(
        args=args,
        config_stem=f"{experiment_name}_all_axes",
        dataset_entries=all_axes_entries,
        output_path=Path(outputs["all_axes"]),
        OmegaConf=OmegaConf,
    )
    config_paths["all_axes"] = str(all_axes_config.resolve())
    config_names["all_axes"] = hydra_config_name(all_axes_config)

    for ax in active_axes:
        key = f"{ax}_only"
        entries = axis_only_entries[ax]
        if not entries:
            continue

        cfg_path = generate_training_config(
            args=args,
            config_stem=f"{experiment_name}_{key}",
            dataset_entries=entries,
            output_path=Path(outputs[key]),
            OmegaConf=OmegaConf,
        )
        config_paths[key] = str(cfg_path.resolve())
        config_names[key] = hydra_config_name(cfg_path)

    checkpoint_paths = {
        key: str((Path(out) / "checkpoints" / "checkpoint.pt").resolve())
        for key, out in outputs.items()
    }

    metadata = {
        "experiment_name": experiment_name,
        "strategy": "multi_axis",
        "datasets": sorted(args.datasets),
        "active_axes": active_axes,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "num_frames": args.num_frames,
        "stride": args.stride,
        "target_size": args.target_size,
        "global_axis_counts": global_axis_counts,
        "axis_multipliers": axis_multipliers,
        "paired_axis_training": bool(args.paired_axis_training),
        "consistency_loss_weight": float(args.consistency_loss_weight),
        "config_paths": config_paths,
        "config_names": config_names,
        "outputs": outputs,
        "checkpoint_paths": checkpoint_paths,
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "infer_config": str(args.infer_config.resolve()),
        "splits": split_meta,

        # compatibility aliases
        "stack_config_name": config_names.get("all_axes"),
        "stack_output": outputs.get("all_axes"),
        "multi_axis_config_name": config_names.get("all_axes"),
        "comparison_config_names": {
            "z": config_names.get("z_only"),
            "y": config_names.get("y_only"),
            "x": config_names.get("x_only"),
        },
    }

    meta_path = split_root / "experiment_multi_axis.json"
    write_json(meta_path, metadata)

    print(f"\nSaved metadata: {meta_path}")

    print_training_commands(metadata)

    print("\nEvaluation example:")
    print(
        f"python {Path(__file__).name} "
        f"--mode eval --experiment-name {experiment_name} "
        f"--datasets-root {datasets_root} "
        f"--eval-models base z_only all_axes --eval-bidir"
    )

    return metadata


# -------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------

def print_training_commands(meta: dict[str, Any]) -> None:
    print(f"\n{'=' * 70}")
    print("TRAINING COMMANDS")
    print(f"{'=' * 70}")

    config_names = meta["config_names"]
    outputs = meta["outputs"]

    for family in ["all_axes", "z_only", "y_only", "x_only"]:
        if family not in config_names:
            continue

        print(f"\n# {family}")
        print("python training/train.py \\")
        print(f"  -c {config_names[family]} \\")
        print(f"  --output-path {outputs[family]} \\")
        print("  --use-cluster 0 --num-gpus 1 --num-nodes 1")


def train_family(args: argparse.Namespace, meta: dict[str, Any], family: str) -> None:
    config_names = meta["config_names"]
    outputs = meta["outputs"]
    ckpts = meta["checkpoint_paths"]

    if family not in config_names:
        print(f"WARNING: no config for {family}; skipping training")
        return

    ckpt = Path(ckpts[family])
    if ckpt.exists() and not args.force_train:
        print(f"Using existing checkpoint for {family}: {ckpt}")
        return

    cmd = [
        args.python,
        "training/train.py",
        "-c",
        config_names[family],
        "--output-path",
        outputs[family],
        "--use-cluster",
        "0",
        "--num-gpus",
        str(args.num_gpus),
        "--num-nodes",
        str(args.num_nodes),
    ]
    run(cmd)


def run_training(args: argparse.Namespace, meta: dict[str, Any]) -> None:
    print(f"\n{'=' * 70}")
    print("TRAINING")
    print(f"{'=' * 70}")

    for family in args.train_models:
        train_family(args, meta, family)


# -------------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------------

def checkpoint_for_family(args: argparse.Namespace, meta: dict[str, Any], family: str) -> Path:
    if family == "base":
        return Path(meta.get("base_checkpoint", str(args.base_checkpoint)))

    override_map = {
        "z_only": args.z_checkpoint,
        "y_only": args.y_checkpoint,
        "x_only": args.x_checkpoint,
        "all_axes": args.all_axes_checkpoint,
    }

    if override_map.get(family) is not None:
        return override_map[family]

    ckpts = meta.get("checkpoint_paths", {})
    if family in ckpts:
        return Path(ckpts[family])

    outputs = meta.get("outputs", {})
    if family in outputs:
        return Path(outputs[family]) / "checkpoints" / "checkpoint.pt"

    raise KeyError(f"Cannot resolve checkpoint for family: {family}")


def eval_one(
    args: argparse.Namespace,
    tag: str,
    checkpoint: Path,
    infer_config: Path,
    dataset_dir: Path,
    manifest: Path,
    output_dir: Path,
) -> None:
    spec = MODEL_SPECS[tag]

    if spec.get("eval_axis") in {"y", "x"}:
        mode = "bidir" if spec["kind"] == "bidir" else "causal"
        cmd = [
            args.python,
            "scripts/eval_multi_axis_fused.py",
            "--checkpoint",
            str(checkpoint),
            "--cfg",
            str(infer_config),
            "--dataset-dir",
            str(dataset_dir),
            "--file-list",
            str(manifest),
            "--image-channel-index",
            str(args.channel_index),
            "--memory-temporal-stride-for-eval",
            str(args.memory_stride),
            "--output-dir",
            str(output_dir),
            "--mode",
            mode,
            "--fusion",
            args.fusion,
            "--axes",
            spec["eval_axis"],
        ]
        if args.disable_3d_filter:
            cmd.append("--disable-3d-filter")
    elif spec["kind"] in {"base", "causal"}:
        cmd = [
            args.python,
            "scripts/eval_npz_dataset.py",
            "--checkpoint",
            str(checkpoint),
            "--cfg",
            str(infer_config),
            "--dataset-dir",
            str(dataset_dir),
            "--file-list",
            str(manifest),
            "--image-channel-index",
            str(args.channel_index),
            "--memory-temporal-stride-for-eval",
            str(args.memory_stride),
            "--output-dir",
            str(output_dir),
        ]
    elif spec["kind"] == "bidir":
        cmd = [
            args.python,
            "scripts/eval_npz_dataset_bidirectional.py",
            "--checkpoint",
            str(checkpoint),
            "--cfg",
            str(infer_config),
            "--dataset-dir",
            str(dataset_dir),
            "--file-list",
            str(manifest),
            "--image-channel-index",
            str(args.channel_index),
            "--memory-temporal-stride-for-eval",
            str(args.memory_stride),
            "--min-component-volume",
            str(args.min_component_volume),
            "--output-dir",
            str(output_dir),
        ]
        if args.disable_3d_filter:
            cmd.append("--no-3d-filter")
    else:
        mode = "bidir" if spec["kind"] == "fused_bidir" else "causal"
        cmd = [
            args.python,
            "scripts/eval_multi_axis_fused.py",
            "--checkpoint",
            str(checkpoint),
            "--cfg",
            str(infer_config),
            "--dataset-dir",
            str(dataset_dir),
            "--file-list",
            str(manifest),
            "--image-channel-index",
            str(args.channel_index),
            "--memory-temporal-stride-for-eval",
            str(args.memory_stride),
            "--output-dir",
            str(output_dir),
            "--mode",
            mode,
            "--fusion",
            args.fusion,
            "--axes",
            "z",
            "y",
            "x",
        ]
        if args.disable_3d_filter:
            cmd.append("--disable-3d-filter")

    run(cmd)


def build_model_order(args: argparse.Namespace) -> list[str]:
    order: list[str] = []

    if "base" in args.eval_models:
        order.append("base")

    for family in ["z_only", "y_only", "x_only", "all_axes"]:
        if family not in args.eval_models:
            continue

        if args.eval_causal:
            order.append(f"{family}_causal")
        if args.eval_bidir:
            order.append(f"{family}_bidir")

        if family == "all_axes" and args.eval_fused:
            if args.eval_causal:
                order.append("all_axes_fused_causal")
            if args.eval_bidir:
                order.append("all_axes_fused_bidir")

    return order


def run_evaluation(args: argparse.Namespace, meta: dict[str, Any]) -> None:
    print(f"\n{'=' * 70}")
    print("EVALUATION")
    print(f"{'=' * 70}")

    exp_log = REPO_ROOT / "exp_log" / args.experiment_name
    eval_root = exp_log / (args.eval_root_name or "eval_multi_axis")
    comparison_dir = exp_log / "comparison"

    infer_config = Path(meta.get("infer_config", str(args.infer_config)))
    model_order = build_model_order(args)

    interval_suffix = (
        f"{args.context_interval_mode}{args.context_slice_interval}"
        if args.context_interval_mode == "fixed"
        else "dynamic"
    )

    tmp_clip_root = exp_log / (
        f"tmp_eval_clips_w{args.eval_window_size}_s{args.eval_window_stride}_i{interval_suffix}"
    )

    print(f"eval root: {eval_root}")
    print(f"models   : {model_order}")

    for ds_name, info in meta["splits"].items():
        ds_slug = slugify(ds_name)
        dataset_dir = Path(info["dataset_dir"])
        slice_test_manifest = Path(info["slice_test_manifest"])

        if not slice_test_manifest.exists():
            print(f"WARNING: missing test manifest for {ds_name}: {slice_test_manifest}")
            continue

        print(f"\nDataset: {ds_name}")

        clip_npz_dir = tmp_clip_root / ds_slug / "npz"
        clip_manifest = tmp_clip_root / ds_slug / f"{ds_slug}_stacked_test.txt"

        clip_count = build_clip_dataset(
            dataset_dir=dataset_dir,
            manifest_path=slice_test_manifest,
            out_npz_dir=clip_npz_dir,
            out_manifest=clip_manifest,
            window_size=args.eval_window_size,
            window_stride=args.eval_window_stride,
            slice_interval=args.context_slice_interval,
            interval_mode=args.context_interval_mode,
            force=args.force_rebuild_eval_clips,
        )
        print(f"  eval clips: {clip_count} -> {clip_manifest}")

        for tag in model_order:
            spec = MODEL_SPECS[tag]

            if tag == "base":
                checkpoint = checkpoint_for_family(args, meta, "base")
                eval_dataset_dir = dataset_dir
                eval_manifest = slice_test_manifest
            elif spec["kind"] in {"fused_causal", "fused_bidir"} or spec.get("eval_axis") in {"y", "x"}:
                family = spec["family"]
                checkpoint = checkpoint_for_family(args, meta, family)
                eval_dataset_dir = dataset_dir
                eval_manifest = slice_test_manifest
            else:
                family = spec["family"]
                checkpoint = checkpoint_for_family(args, meta, family)
                eval_dataset_dir = clip_npz_dir
                eval_manifest = clip_manifest

            if not checkpoint.exists():
                print(f"  WARNING: checkpoint missing for {tag}: {checkpoint}; skipping")
                continue

            print(f"  Evaluating {tag} with checkpoint: {checkpoint}")

            eval_one(
                args=args,
                tag=tag,
                checkpoint=checkpoint,
                infer_config=infer_config,
                dataset_dir=eval_dataset_dir,
                manifest=eval_manifest,
                output_dir=eval_root / tag / ds_slug,
            )

    write_comparison(
        eval_root=eval_root,
        comparison_dir=comparison_dir,
        model_order=model_order,
        name="multi_axis",
    )


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning")
    os.environ.setdefault("SAM2_LOG_LEVEL", "ERROR")

    meta_path = args.datasets_root / "_splits" / args.experiment_name / "experiment_multi_axis.json"

    if should_prepare(args, meta_path):
        meta = prepare_experiment(args)
    else:
        print(f"Using existing metadata: {meta_path}")
        meta = load_json(meta_path)

    if args.mode in {"train", "train-eval", "all"}:
        run_training(args, meta)

    if args.mode in {"eval", "train-eval", "all"}:
        run_evaluation(args, meta)

    if args.mode == "prepare":
        print("\nPrepare complete.")
        print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
