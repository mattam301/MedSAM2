#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
torch = None
Image = None
tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MedSAM2 on an NPZ dataset split using GT-derived prompts and "
            "compute basic segmentation metrics."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Checkpoint to load. Works with trainer checkpoints that store weights under 'model'.",
    )
    parser.add_argument(
        "--cfg",
        type=Path,
        default=REPO_ROOT / "sam2" / "configs" / "sam2.1_hiera_t512.yaml",
        help="Inference config file.",
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="Folder containing NPZ files.",
    )
    parser.add_argument(
        "--file-list",
        type=Path,
        default=None,
        help="Optional manifest listing relative NPZ paths without extension.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for metrics and optional prediction dumps.",
    )
    parser.add_argument(
        "--prompt-type",
        choices=["box", "point", "mask"],
        default="box",
        help="Prompt simulation strategy derived from GT masks.",
    )
    parser.add_argument(
        "--save-preds",
        action="store_true",
        help="Save predicted label volumes as compressed NPZ files.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional cap on the number of cases to evaluate.",
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
        "--memory-temporal-stride-for-eval",
        type=int,
        default=None,
        help=(
            "Override SAM2 memory-bank temporal stride during evaluation. "
            "Use 1 for adjacent memory frames; larger values sample wider context."
        ),
    )
    return parser.parse_args()


def resolve_config_path(path: Path) -> str:
    path = path.resolve()
    return f"//{path}"


def load_case_paths(dataset_dir: Path, file_list: Path | None) -> list[Path]:
    if file_list is None:
        return sorted(dataset_dir.rglob("*.npz"))

    case_paths: list[Path] = []
    for line in file_list.read_text(encoding="utf-8").splitlines():
        case_id = line.strip()
        if not case_id:
            continue
        suffix = "" if case_id.endswith(".npz") else ".npz"
        case_paths.append((dataset_dir / f"{case_id}{suffix}").resolve())
    return case_paths


def resize_grayscale_to_rgb_and_resize(array: np.ndarray, image_size: int) -> np.ndarray:
    resized = np.empty((array.shape[0], 3, image_size, image_size), dtype=np.float32)
    for index, frame in enumerate(array):
        image = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
        image = image.resize((image_size, image_size), resample=Image.BILINEAR)
        resized[index] = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
    return resized


def normalize_npz_case(
    npz_path: Path,
    imgs: np.ndarray,
    gts: np.ndarray,
    image_channel_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    imgs = np.asarray(imgs)
    gts = np.asarray(gts)

    if gts.ndim == 2:
        if imgs.ndim == 2:
            imgs = imgs[None, ...]
        elif imgs.ndim == 3 and imgs.shape[:2] == gts.shape and imgs.shape[2] >= 1:
            channel_index = min(image_channel_index, imgs.shape[2] - 1)
            imgs = imgs[..., channel_index][None, ...]
        else:
            raise ValueError(
                f"Unsupported 2D NPZ layout in {npz_path}: imgs={imgs.shape}, gts={gts.shape}"
            )
        gts = gts[None, ...]
    elif gts.ndim == 3:
        if imgs.ndim == 3 and imgs.shape == gts.shape:
            pass
        elif imgs.ndim == 4 and imgs.shape[:3] == gts.shape and imgs.shape[3] >= 1:
            channel_index = min(image_channel_index, imgs.shape[3] - 1)
            imgs = imgs[..., channel_index]
        else:
            raise ValueError(
                f"Unsupported 3D NPZ layout in {npz_path}: imgs={imgs.shape}, gts={gts.shape}"
            )
    else:
        raise ValueError(f"Unsupported mask layout in {npz_path}: gts={gts.shape}")

    if imgs.shape != gts.shape:
        raise ValueError(
            f"Normalized shapes do not match in {npz_path}: imgs={imgs.shape}, gts={gts.shape}"
        )
    return imgs, gts


def preprocess_volume(imgs_3d: np.ndarray, image_size: int, device: str) -> torch.Tensor:
    if imgs_3d.shape[1:] == (image_size, image_size):
        images = np.repeat(imgs_3d[:, None], 3, axis=1).astype(np.float32)
    else:
        images = resize_grayscale_to_rgb_and_resize(imgs_3d, image_size)
    images /= 255.0
    tensor = torch.from_numpy(images).to(device)
    img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32, device=device)[:, None, None]
    img_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32, device=device)[:, None, None]
    tensor.sub_(img_mean).div_(img_std)
    return tensor


def get_bbox(mask_2d: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask_2d > 0)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def get_center_point(mask_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask_2d > 0)
    center_x = float(xs.mean())
    center_y = float(ys.mean())
    return np.array([[center_x, center_y]], dtype=np.float32), np.array([1], dtype=np.int32)


def get_prompt_slice(mask_3d: np.ndarray) -> int:
    non_empty = np.where(mask_3d.reshape(mask_3d.shape[0], -1).any(axis=1))[0]
    if len(non_empty) == 0:
        raise ValueError("Object mask is empty across the full volume.")
    return int(non_empty[len(non_empty) // 2])


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    inter = np.logical_and(pred, gt).sum()
    return float((2.0 * inter) / denom)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(pred, gt).sum()
    return float(inter / union)


def run_single_label(
    predictor,
    images: torch.Tensor,
    video_height: int,
    video_width: int,
    label_mask: np.ndarray,
    prompt_type: str,
    autocast_device: str,
) -> np.ndarray:
    z_mid = get_prompt_slice(label_mask)
    prompt_slice_mask = label_mask[z_mid].astype(np.uint8)
    pred_scores = np.full(label_mask.shape, fill_value=-1e9, dtype=np.float32)

    with torch.inference_mode(), torch.autocast(autocast_device, dtype=torch.bfloat16):
        inference_state = predictor.init_state(images, video_height, video_width)

        if prompt_type == "mask":
            mask_prompt = prompt_slice_mask
        elif prompt_type == "box":
            box = get_bbox(prompt_slice_mask)
            _, _, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=z_mid,
                obj_id=1,
                box=box,
            )
            mask_prompt = (out_mask_logits[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)
        else:
            points, labels = get_center_point(prompt_slice_mask)
            _, _, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=z_mid,
                obj_id=1,
                points=points,
                labels=labels,
            )
            mask_prompt = (out_mask_logits[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)

        _, _, masks = predictor.add_new_mask(
            inference_state,
            frame_idx=z_mid,
            obj_id=1,
            mask=mask_prompt,
        )
        pred_scores[z_mid] = masks[0, 0].detach().cpu().numpy()

        for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
            inference_state, start_frame_idx=z_mid, reverse=False
        ):
            pred_scores[out_frame_idx] = out_mask_logits[0, 0].detach().cpu().numpy()

        predictor.reset_state(inference_state)
        inference_state = predictor.init_state(images, video_height, video_width)
        predictor.add_new_mask(
            inference_state,
            frame_idx=z_mid,
            obj_id=1,
            mask=mask_prompt,
        )
        for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
            inference_state, start_frame_idx=z_mid, reverse=True
        ):
            pred_scores[out_frame_idx] = out_mask_logits[0, 0].detach().cpu().numpy()
        predictor.reset_state(inference_state)

    return pred_scores


def evaluate_case(
    predictor,
    npz_path: Path,
    prompt_type: str,
    device: str,
    autocast_device: str,
    image_channel_index: int,
) -> tuple[np.ndarray, list[dict]]:
    data = np.load(npz_path, allow_pickle=True)
    if "imgs" not in data.files or "gts" not in data.files:
        raise ValueError(f"{npz_path} must contain 'imgs' and 'gts'. Found: {data.files}")

    imgs_3d, gts_3d = normalize_npz_case(
        npz_path,
        data["imgs"],
        data["gts"],
        image_channel_index=image_channel_index,
    )
    video_height, video_width = imgs_3d.shape[1:]
    images = preprocess_volume(imgs_3d, predictor.image_size, device)
    labels = [int(label) for label in np.unique(gts_3d) if label != 0]

    pred_label_map = np.zeros(gts_3d.shape, dtype=np.uint16)
    score_stack = []
    label_order = []
    case_metrics: list[dict] = []

    for label in labels:
        label_mask = (gts_3d == label).astype(np.uint8)
        if label_mask.sum() == 0:
            continue
        scores = run_single_label(
            predictor=predictor,
            images=images,
            video_height=video_height,
            video_width=video_width,
            label_mask=label_mask,
            prompt_type=prompt_type,
            autocast_device=autocast_device,
        )
        pred_binary = scores > 0.0
        case_metrics.append(
            {
                "case": npz_path.name,
                "label": label,
                "dice": dice_score(pred_binary, label_mask),
                "iou": iou_score(pred_binary, label_mask),
                "gt_voxels": int(label_mask.sum()),
                "pred_voxels": int(pred_binary.sum()),
            }
        )
        score_stack.append(scores)
        label_order.append(label)

    if score_stack:
        stacked = np.stack(score_stack, axis=0)
        winner_index = np.argmax(stacked, axis=0)
        winner_score = np.take_along_axis(stacked, winner_index[None], axis=0)[0]
        positive = winner_score > 0.0
        label_lookup = np.array(label_order, dtype=np.uint16)
        pred_label_map[positive] = label_lookup[winner_index[positive]]

    return pred_label_map, case_metrics


def main() -> None:
    global torch, Image, tqdm
    args = parse_args()

    try:
        import torch as _torch
        from PIL import Image as _Image
        from tqdm import tqdm as _tqdm
        from sam2.build_sam import build_sam2_video_predictor_npz, get_best_available_device
    except ImportError as exc:
        raise SystemExit(
            "MedSAM2 dependencies are not available in the current Python environment. "
            "Activate the MedSAM2 environment first, then rerun this script."
        ) from exc
    torch = _torch
    Image = _Image
    tqdm = _tqdm

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    case_paths = load_case_paths(args.dataset_dir.resolve(), args.file_list.resolve() if args.file_list else None)
    if args.max_cases is not None:
        case_paths = case_paths[: args.max_cases]

    if not case_paths:
        raise SystemExit("No NPZ files were selected for evaluation.")

    device = get_best_available_device()
    autocast_device = "cuda" if device == "cuda" else "cpu"
    predictor = build_sam2_video_predictor_npz(
        resolve_config_path(args.cfg),
        ckpt_path=str(args.checkpoint.resolve()),
        device=device,
    )
    if args.memory_temporal_stride_for_eval is not None:
        predictor.memory_temporal_stride_for_eval = args.memory_temporal_stride_for_eval

    all_metrics: list[dict] = []
    predictions_dir = output_dir / "predictions"
    if args.save_preds:
        predictions_dir.mkdir(parents=True, exist_ok=True)

    for npz_path in tqdm(case_paths, desc="Evaluating NPZ cases"):
        pred_label_map, case_metrics = evaluate_case(
            predictor=predictor,
            npz_path=npz_path,
            prompt_type=args.prompt_type,
            device=device,
            autocast_device=autocast_device,
            image_channel_index=args.image_channel_index,
        )
        all_metrics.extend(case_metrics)

        if args.save_preds:
            np.savez_compressed(predictions_dir / npz_path.name, segs=pred_label_map)

    metrics_path = output_dir / "case_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case", "label", "dice", "iou", "gt_voxels", "pred_voxels"],
        )
        writer.writeheader()
        writer.writerows(all_metrics)

    summary = {
        "num_cases": len(case_paths),
        "num_labels": len(all_metrics),
        "mean_dice": float(np.mean([row["dice"] for row in all_metrics])) if all_metrics else None,
        "mean_iou": float(np.mean([row["iou"] for row in all_metrics])) if all_metrics else None,
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(args.cfg.resolve()),
        "dataset_dir": str(args.dataset_dir.resolve()),
        "file_list": str(args.file_list.resolve()) if args.file_list else None,
        "prompt_type": args.prompt_type,
        "image_channel_index": args.image_channel_index,
        "memory_temporal_stride_for_eval": predictor.memory_temporal_stride_for_eval,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved case metrics: {metrics_path}")
    print(f"Saved summary     : {summary_path}")
    if summary["mean_dice"] is not None:
        print(f"Mean Dice         : {summary['mean_dice']:.4f}")
        print(f"Mean IoU          : {summary['mean_iou']:.4f}")


if __name__ == "__main__":
    main()
