#!/usr/bin/env python3
"""Evaluate SAM2 with BIDIRECTIONAL inference on NPZ datasets.

Key differences from eval_npz_dataset.py:
  - Uses propagate_in_video_bidirectional() (two-pass algorithm).
  - Optional GeoSAM2-style bootstrap: the prompt frame is prepended to the
    image tensor so the memory bank is never empty when the real prompt frame
    is processed (--bootstrap flag).
  - Memory redundancy is scored using Pass 1 mask predictions.
  - Optional 3-D connected-component post-filter (--no-3d-filter to disable).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
torch = None
Image = None
tqdm  = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run MedSAM2 bidirectional inference on an NPZ dataset and compute "
            "segmentation metrics."
        )
    )
    p.add_argument("--checkpoint",  required=True, type=Path)
    p.add_argument(
        "--cfg",
        type=Path,
        default=REPO_ROOT / "sam2" / "configs" / "sam2.1_hiera_t512.yaml",
    )
    p.add_argument("--dataset-dir", required=True, type=Path)
    p.add_argument("--file-list",   type=Path, default=None)
    p.add_argument("--output-dir",  required=True, type=Path)
    p.add_argument(
        "--prompt-type", choices=["box", "point", "mask"], default="box"
    )
    p.add_argument("--save-preds",  action="store_true")
    p.add_argument("--max-cases",   type=int, default=None)
    p.add_argument("--image-channel-index", type=int, default=0)
    p.add_argument(
        "--memory-temporal-stride-for-eval",
        type=int,
        default=None,
        help=(
            "Override SAM2 memory-bank temporal stride during evaluation. "
            "Use 1 for adjacent memory frames; larger values sample wider context."
        ),
    )

    # ── GeoSAM2 bootstrap ────────────────────────────────────────────────
    boot = p.add_argument_group(
        "GeoSAM2 Bootstrap",
        description=(
            "Prepend the prompt frame to the image tensor before inference. "
            "The duplicated frame pre-populates the memory bank so the memory "
            "is never empty when the real prompt frame is processed. "
            "This is the approach described in GeoSAM2 (frame repetition). "
            "Zero extra propagation passes are needed."
        ),
    )
    boot.add_argument(
        "--bootstrap",
        action="store_true",
        help="Enable GeoSAM2-style frame-repetition bootstrap (default: off).",
    )

    # ── Memory / redundancy ──────────────────────────────────────────────
    mem = p.add_argument_group("Memory redundancy")
    mem.add_argument(
        "--memory-redundancy-threshold",
        type=float,
        default=1.0,
        help=(
            "Mask-based redundancy threshold.  Memory frames whose redundancy "
            "score exceeds this value are pruned.  1.0 = no pruning (default)."
        ),
    )
    mem.add_argument(
        "--memory-max-unique-context-frames",
        type=int,
        default=None,
    )
    mem.add_argument("--memory-unique-residual", action="store_true")
    mem.add_argument(
        "--memory-decomposer-type",
        choices=["heuristic", "learned"],
        default="heuristic",
    )
    mem.add_argument("--memory-decomposer-use-augmentation", action="store_true")
    mem.add_argument("--memory-decomposer-checkpoint", type=Path, default=None)

    # ── Post-processing ──────────────────────────────────────────────────
    post = p.add_argument_group("Post-processing")
    post.add_argument("--no-3d-filter", action="store_true")
    post.add_argument("--min-component-volume", type=int, default=100)

    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def resolve_config_path(path: Path) -> str:
    return f"//{path.resolve()}"


def load_case_paths(dataset_dir: Path, file_list: Path | None) -> list[Path]:
    if file_list is None:
        return sorted(dataset_dir.rglob("*.npz"))
    paths: list[Path] = []
    for line in file_list.read_text(encoding="utf-8").splitlines():
        case_id = line.strip()
        if not case_id:
            continue
        suffix = "" if case_id.endswith(".npz") else ".npz"
        paths.append((dataset_dir / f"{case_id}{suffix}").resolve())
    return paths


def resize_grayscale_to_rgb_and_resize(
    array: np.ndarray, image_size: int
) -> np.ndarray:
    resized = np.empty(
        (array.shape[0], 3, image_size, image_size), dtype=np.float32
    )
    for i, frame in enumerate(array):
        img = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
        img = img.resize((image_size, image_size), resample=Image.BILINEAR)
        resized[i] = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)
    return resized


def normalize_npz_case(
    npz_path: Path,
    imgs: np.ndarray,
    gts: np.ndarray,
    image_channel_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    imgs = np.asarray(imgs)
    gts  = np.asarray(gts)

    if gts.ndim == 2:
        if imgs.ndim == 2:
            imgs = imgs[None, ...]
        elif imgs.ndim == 3 and imgs.shape[:2] == gts.shape:
            ch   = min(image_channel_index, imgs.shape[2] - 1)
            imgs = imgs[..., ch][None, ...]
        else:
            raise ValueError(
                f"Unsupported 2-D layout in {npz_path}: "
                f"imgs={imgs.shape}, gts={gts.shape}"
            )
        gts = gts[None, ...]
    elif gts.ndim == 3:
        if imgs.ndim == 3 and imgs.shape == gts.shape:
            pass
        elif imgs.ndim == 4 and imgs.shape[:3] == gts.shape:
            ch   = min(image_channel_index, imgs.shape[3] - 1)
            imgs = imgs[..., ch]
        else:
            raise ValueError(
                f"Unsupported 3-D layout in {npz_path}: "
                f"imgs={imgs.shape}, gts={gts.shape}"
            )
    else:
        raise ValueError(
            f"Unsupported mask layout in {npz_path}: gts={gts.shape}"
        )

    if imgs.shape != gts.shape:
        raise ValueError(
            f"Shape mismatch in {npz_path}: imgs={imgs.shape}, gts={gts.shape}"
        )
    return imgs, gts


def preprocess_volume(
    imgs_3d: np.ndarray, image_size: int, device: str
) -> "torch.Tensor":
    if imgs_3d.shape[1:] == (image_size, image_size):
        images = np.repeat(imgs_3d[:, None], 3, axis=1).astype(np.float32)
    else:
        images = resize_grayscale_to_rgb_and_resize(imgs_3d, image_size)
    images  /= 255.0
    tensor   = torch.from_numpy(images).to(device)
    img_mean = torch.tensor(
        (0.485, 0.456, 0.406), dtype=torch.float32, device=device
    )[:, None, None]
    img_std = torch.tensor(
        (0.229, 0.224, 0.225), dtype=torch.float32, device=device
    )[:, None, None]
    tensor.sub_(img_mean).div_(img_std)
    return tensor


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def get_bbox(mask_2d: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask_2d > 0)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def get_center_point(mask_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask_2d > 0)
    return (
        np.array([[float(xs.mean()), float(ys.mean())]], dtype=np.float32),
        np.array([1], dtype=np.int32),
    )


def get_prompt_slice(mask_3d: np.ndarray) -> int:
    non_empty = np.where(
        mask_3d.reshape(mask_3d.shape[0], -1).any(axis=1)
    )[0]
    if len(non_empty) == 0:
        raise ValueError("Object mask is empty across the full volume.")
    return int(non_empty[len(non_empty) // 2])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / denom)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def surface_smoothness(pred_binary: np.ndarray) -> float:
    """Slice-to-slice area consistency. Higher = smoother 3-D surface."""
    areas = pred_binary.reshape(pred_binary.shape[0], -1).sum(axis=1).astype(float)
    if areas.max() == 0:
        return 0.0
    changes = np.abs(np.diff(areas)) / areas[:-1].clip(min=1)
    return float(1.0 - np.clip(changes.mean(), 0, 1))


# ---------------------------------------------------------------------------
# 3-D post-filter
# ---------------------------------------------------------------------------

def connected_component_filter(
    pred_scores: np.ndarray,
    min_component_volume: int = 100,
) -> np.ndarray:
    try:
        from scipy import ndimage
    except ImportError:
        return pred_scores
    binary = pred_scores > 0.0
    if not binary.any():
        return pred_scores
    labeled, n = ndimage.label(binary)
    if n <= 1:
        return pred_scores
    filtered = pred_scores.copy()
    for i in range(1, n + 1):
        if (labeled == i).sum() < min_component_volume:
            filtered[labeled == i] = -1e9
    return filtered


# ---------------------------------------------------------------------------
# Single-label bidirectional inference
# ---------------------------------------------------------------------------

def run_single_label_bidirectional(
    predictor,
    images: "torch.Tensor",
    video_height: int,
    video_width: int,
    label_mask: np.ndarray,
    prompt_type: str,
    autocast_device: str,
    use_bootstrap: bool = False,
    use_3d_filter: bool = True,
    min_component_volume: int = 100,
) -> np.ndarray:
    """Run two-pass bidirectional inference for one binary label.

    When use_bootstrap=True the prompt frame is prepended to the image
    tensor (GeoSAM2 frame-repetition strategy) so the memory bank holds
    one entry when the real prompt frame is processed.  No extra
    propagation passes are required.

    Returns:
        pred_scores: logit volume [D, H, W] in the original frame space.
    """
    z_mid = get_prompt_slice(label_mask)
    prompt_slice_mask = label_mask[z_mid].astype(np.uint8)
    num_original_frames = label_mask.shape[0]
    pred_scores = np.full(
        label_mask.shape, fill_value=-1e9, dtype=np.float32
    )

    with (
        torch.inference_mode(),
        torch.autocast(autocast_device, dtype=torch.bfloat16),
    ):
        # ── Optionally prepend prompt frame (GeoSAM2 bootstrap) ─────────
        if use_bootstrap:
            images_for_state, adjusted_prompt_idx = (
                predictor.prepare_bootstrap_images(images, z_mid)
            )
            bootstrap_offset = 1
        else:
            images_for_state   = images
            adjusted_prompt_idx = z_mid
            bootstrap_offset   = 0

        inference_state = predictor.init_state(
            images_for_state, video_height, video_width
        )

        # ── Build initial prompt ─────────────────────────────────────────
        if prompt_type == "mask":
            mask_prompt = prompt_slice_mask
        elif prompt_type == "box":
            box = get_bbox(prompt_slice_mask)
            _, _, out_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=adjusted_prompt_idx,
                obj_id=1,
                box=box,
            )
            mask_prompt = (
                (out_logits[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)
            )
        else:  # point
            points, labels = get_center_point(prompt_slice_mask)
            _, _, out_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=adjusted_prompt_idx,
                obj_id=1,
                points=points,
                labels=labels,
            )
            mask_prompt = (
                (out_logits[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)
            )

        predictor.add_new_mask(
            inference_state,
            frame_idx=adjusted_prompt_idx,
            obj_id=1,
            mask=mask_prompt,
        )

        # ── Two-pass bidirectional propagation ───────────────────────────
        # bootstrap_frame_offset tells propagate_in_video_bidirectional to
        # skip the virtual frame in Pass 2 and shift yielded indices back.
        for out_frame_idx, _, out_mask_logits in (
            predictor.propagate_in_video_bidirectional(
                inference_state,
                start_frame_idx=0,
                bootstrap_frame_offset=bootstrap_offset,
            )
        ):
            # out_frame_idx is already in original space (0 … N-1)
            if 0 <= out_frame_idx < num_original_frames:
                pred_scores[out_frame_idx] = (
                    out_mask_logits[0, 0].detach().cpu().numpy()
                )

        predictor.reset_state(inference_state)

    if use_3d_filter:
        pred_scores = connected_component_filter(pred_scores, min_component_volume)

    return pred_scores


# ---------------------------------------------------------------------------
# Case-level evaluation
# ---------------------------------------------------------------------------

def evaluate_case(
    predictor,
    npz_path: Path,
    prompt_type: str,
    device: str,
    autocast_device: str,
    image_channel_index: int,
    use_bootstrap: bool = False,
    use_3d_filter: bool = True,
    min_component_volume: int = 100,
) -> tuple[np.ndarray, list[dict]]:
    data = np.load(npz_path, allow_pickle=True)
    if "imgs" not in data.files or "gts" not in data.files:
        raise ValueError(
            f"{npz_path} must contain 'imgs' and 'gts'. Found: {data.files}"
        )

    imgs_3d, gts_3d = normalize_npz_case(
        npz_path, data["imgs"], data["gts"], image_channel_index
    )
    video_height, video_width = imgs_3d.shape[1:]
    images  = preprocess_volume(imgs_3d, predictor.image_size, device)
    labels  = [int(lbl) for lbl in np.unique(gts_3d) if lbl != 0]

    pred_label_map = np.zeros(gts_3d.shape, dtype=np.uint16)
    score_stack: list[np.ndarray] = []
    label_order: list[int]        = []
    case_metrics: list[dict]      = []

    for label in labels:
        label_mask = (gts_3d == label).astype(np.uint8)
        if label_mask.sum() == 0:
            continue

        scores = run_single_label_bidirectional(
            predictor=predictor,
            images=images,
            video_height=video_height,
            video_width=video_width,
            label_mask=label_mask,
            prompt_type=prompt_type,
            autocast_device=autocast_device,
            use_bootstrap=use_bootstrap,
            use_3d_filter=use_3d_filter,
            min_component_volume=min_component_volume,
        )
        pred_binary = scores > 0.0
        case_metrics.append(
            {
                "case":        npz_path.name,
                "label":       label,
                "dice":        dice_score(pred_binary, label_mask),
                "iou":         iou_score(pred_binary, label_mask),
                "smoothness":  surface_smoothness(pred_binary),
                "gt_voxels":   int(label_mask.sum()),
                "pred_voxels": int(pred_binary.sum()),
            }
        )
        score_stack.append(scores)
        label_order.append(label)

    if score_stack:
        stacked      = np.stack(score_stack, axis=0)
        winner_index = np.argmax(stacked, axis=0)
        winner_score = np.take_along_axis(stacked, winner_index[None], axis=0)[0]
        positive     = winner_score > 0.0
        label_lookup = np.array(label_order, dtype=np.uint16)
        pred_label_map[positive] = label_lookup[winner_index[positive]]

    return pred_label_map, case_metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global torch, Image, tqdm
    args = parse_args()

    try:
        import torch as _torch
        from PIL import Image as _Image
        from tqdm import tqdm as _tqdm
        from sam2.bidirectional_video_predictor import (
            build_bidir_sam2_video_predictor_npz,
        )
        from sam2.build_sam import get_best_available_device
    except ImportError as exc:
        raise SystemExit(
            "MedSAM2 dependencies not available. "
            "Activate the MedSAM2 environment first."
        ) from exc

    torch = _torch
    Image = _Image
    tqdm  = _tqdm

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    case_paths = load_case_paths(
        args.dataset_dir.resolve(),
        args.file_list.resolve() if args.file_list else None,
    )
    if args.max_cases is not None:
        case_paths = case_paths[: args.max_cases]
    if not case_paths:
        raise SystemExit("No NPZ files were selected for evaluation.")

    device          = get_best_available_device()
    autocast_device = "cuda" if device == "cuda" else "cpu"

    print("Building BIDIRECTIONAL SAM2 predictor...")
    predictor = build_bidir_sam2_video_predictor_npz(
        resolve_config_path(args.cfg),
        ckpt_path=str(args.checkpoint.resolve()),
        device=device,
    )
    predictor.memory_redundancy_threshold      = args.memory_redundancy_threshold
    predictor.memory_max_unique_context_frames = args.memory_max_unique_context_frames
    predictor.memory_unique_residual           = args.memory_unique_residual
    if args.memory_temporal_stride_for_eval is not None:
        predictor.memory_temporal_stride_for_eval = args.memory_temporal_stride_for_eval

    if args.memory_decomposer_type != "heuristic" or args.memory_decomposer_checkpoint:
        predictor.decomposer_type = args.memory_decomposer_type

    if args.memory_decomposer_use_augmentation:
        from sam2.memory_decomposer import create_memory_decomposer
        predictor.memory_decomposer = create_memory_decomposer(
            decomposer_type=args.memory_decomposer_type,
            memory_feat_dim=getattr(predictor, "mem_dim", 64),
            hidden_dim=128,
            use_augmentation=True,
            augmentation_scale=0.1,
        ).to(device)

    if args.memory_decomposer_checkpoint:
        import torch as _torch
        from sam2.memory_decomposer import create_memory_decomposer
        ckpt       = _torch.load(
            args.memory_decomposer_checkpoint.resolve(), map_location=device
        )
        state_dict = ckpt.get("state_dict", ckpt)
        cfg_d      = ckpt.get("config", {})
        predictor.memory_decomposer = create_memory_decomposer(
            decomposer_type="learned",
            memory_feat_dim=cfg_d.get("memory_feat_dim", 64),
            hidden_dim=cfg_d.get("hidden_dim", 128),
            use_augmentation=cfg_d.get("use_augmentation", False),
            augmentation_scale=cfg_d.get("augmentation_scale", 0.1),
            max_score_delta=cfg_d.get("max_score_delta", 0.5),
            residual_scale=cfg_d.get("residual_scale", 0.5),
        )
        predictor.memory_decomposer.load_state_dict(state_dict, strict=True)
        predictor.memory_decomposer = predictor.memory_decomposer.to(device)
        predictor._decomposer_type = "learned"
        if not predictor.memory_unique_residual:
            print(
                "WARNING: learned decomposer loaded but --memory-unique-residual "
                "is not set; only redundancy pruning will use it."
            )

    use_3d_filter = not args.no_3d_filter

    print(f"✓ Bidirectional inference (two-pass)")
    print(f"  Bootstrap (GeoSAM2)  : {args.bootstrap}")
    print(f"  Redundancy threshold : {args.memory_redundancy_threshold}")
    print(f"  Max context frames   : {args.memory_max_unique_context_frames}")
    print(f"  Unique residual      : {args.memory_unique_residual}")
    print(f"  Memory stride        : {predictor.memory_temporal_stride_for_eval}")
    print(f"  Decomposer           : {predictor.decomposer_type}")
    print(
        f"  3-D filter           : "
        f"{'enabled (min ' + str(args.min_component_volume) + ' vx)' if use_3d_filter else 'disabled'}"
    )

    all_metrics: list[dict] = []
    predictions_dir = output_dir / "predictions"
    if args.save_preds:
        predictions_dir.mkdir(parents=True, exist_ok=True)

    for npz_path in tqdm(case_paths, desc="Evaluating (bidirectional)"):
        pred_label_map, case_metrics = evaluate_case(
            predictor=predictor,
            npz_path=npz_path,
            prompt_type=args.prompt_type,
            device=device,
            autocast_device=autocast_device,
            image_channel_index=args.image_channel_index,
            use_bootstrap=args.bootstrap,
            use_3d_filter=use_3d_filter,
            min_component_volume=args.min_component_volume,
        )
        all_metrics.extend(case_metrics)
        if args.save_preds:
            np.savez_compressed(
                predictions_dir / npz_path.name, segs=pred_label_map
            )

    metrics_path = output_dir / "case_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case", "label", "dice", "iou", "smoothness",
                "gt_voxels", "pred_voxels",
            ],
        )
        writer.writeheader()
        writer.writerows(all_metrics)

    summary = {
        "num_cases":  len(case_paths),
        "num_labels": len(all_metrics),
        "mean_dice":  float(np.mean([r["dice"] for r in all_metrics])) if all_metrics else None,
        "mean_iou":   float(np.mean([r["iou"]  for r in all_metrics])) if all_metrics else None,
        "mean_smoothness": float(np.mean([r["smoothness"] for r in all_metrics])) if all_metrics else None,
        "checkpoint": str(args.checkpoint.resolve()),
        "config":     str(args.cfg.resolve()),
        "dataset_dir": str(args.dataset_dir.resolve()),
        "file_list":  str(args.file_list.resolve()) if args.file_list else None,
        "prompt_type": args.prompt_type,
        "image_channel_index": args.image_channel_index,
        "inference_mode": "bidirectional_two_pass",
        "bootstrap":   args.bootstrap,
        "memory_temporal_stride_for_eval": predictor.memory_temporal_stride_for_eval,
        "memory_redundancy_threshold":      args.memory_redundancy_threshold,
        "memory_max_unique_context_frames": args.memory_max_unique_context_frames,
        "memory_unique_residual":           args.memory_unique_residual,
        "memory_decomposer_type":           predictor.decomposer_type,
        "memory_decomposer_checkpoint": (
            str(args.memory_decomposer_checkpoint.resolve())
            if args.memory_decomposer_checkpoint else None
        ),
        "use_3d_filter":        use_3d_filter,
        "min_component_volume": args.min_component_volume,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Case metrics     : {metrics_path}")
    print(f"Summary          : {summary_path}")
    if summary["mean_dice"] is not None:
        print(f"Mean Dice        : {summary['mean_dice']:.4f}")
        print(f"Mean IoU         : {summary['mean_iou']:.4f}")
        print(f"Mean Smoothness  : {summary['mean_smoothness']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
