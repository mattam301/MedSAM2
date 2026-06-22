#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run full-volume Z/Y/X SAM2 inference and fuse logits in 3D."
    )
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--z-checkpoint", type=Path, default=None)
    p.add_argument("--y-checkpoint", type=Path, default=None)
    p.add_argument("--x-checkpoint", type=Path, default=None)
    p.add_argument("--cfg", type=Path, default=REPO_ROOT / "sam2" / "configs" / "sam2.1_hiera_t512.yaml")
    p.add_argument("--dataset-dir", required=True, type=Path)
    p.add_argument("--file-list", type=Path, default=None)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--axes", nargs="+", choices=["z", "y", "x"], default=["z", "y", "x"])
    p.add_argument("--mode", choices=["causal", "bidir"], default="causal")
    p.add_argument("--fusion", choices=["mean_logit", "mean_prob", "agreement"], default="mean_logit")
    p.add_argument("--prompt-type", choices=["box", "point", "mask"], default="box")
    p.add_argument("--save-preds", action="store_true")
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--image-channel-index", type=int, default=0)
    p.add_argument("--memory-temporal-stride-for-eval", type=int, default=None)
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--disable-3d-filter", action="store_true")
    p.add_argument("--min-component-volume", type=int, default=100)
    return p.parse_args()


def orient_volume(volume: np.ndarray, axis: str) -> np.ndarray:
    if axis == "z":
        return volume
    if axis == "y":
        return np.transpose(volume, (1, 0, 2))
    if axis == "x":
        return np.transpose(volume, (2, 0, 1))
    raise ValueError(f"Unknown axis: {axis}")


def restore_axis(scores: np.ndarray, axis: str) -> np.ndarray:
    if axis == "z":
        return scores
    if axis == "y":
        return np.transpose(scores, (1, 0, 2))
    if axis == "x":
        return np.transpose(scores, (1, 2, 0))
    raise ValueError(f"Unknown axis: {axis}")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / denom)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def surface_smoothness(pred_binary: np.ndarray) -> float:
    areas = pred_binary.reshape(pred_binary.shape[0], -1).sum(axis=1).astype(float)
    if areas.max() == 0:
        return 1.0
    diffs = np.abs(np.diff(areas))
    return float(1.0 / (1.0 + diffs.mean() / (areas.max() + 1e-6)))


def resolve_config_path(path: Path) -> str:
    return f"//{path.resolve()}"


def checkpoint_for_axis(args: argparse.Namespace, axis: str) -> Path:
    override = {
        "z": args.z_checkpoint,
        "y": args.y_checkpoint,
        "x": args.x_checkpoint,
    }[axis]
    return override if override is not None else args.checkpoint


def build_predictor(args: argparse.Namespace, checkpoint: Path, device: str):
    if args.mode == "bidir":
        from sam2.bidirectional_video_predictor import build_bidir_sam2_video_predictor_npz

        predictor = build_bidir_sam2_video_predictor_npz(
            resolve_config_path(args.cfg),
            ckpt_path=str(checkpoint.resolve()),
            device=device,
        )
    else:
        from sam2.build_sam import build_sam2_video_predictor_npz

        predictor = build_sam2_video_predictor_npz(
            resolve_config_path(args.cfg),
            ckpt_path=str(checkpoint.resolve()),
            device=device,
        )
    if args.memory_temporal_stride_for_eval is not None:
        predictor.memory_temporal_stride_for_eval = args.memory_temporal_stride_for_eval
    return predictor


def run_axis_label(
    eval_mod,
    predictor,
    imgs_3d: np.ndarray,
    label_mask: np.ndarray,
    axis: str,
    prompt_type: str,
    device: str,
    autocast_device: str,
    args: argparse.Namespace,
) -> np.ndarray:
    imgs_axis = orient_volume(imgs_3d, axis)
    mask_axis = orient_volume(label_mask, axis)
    video_height, video_width = imgs_axis.shape[1:]
    images = eval_mod.preprocess_volume(imgs_axis, predictor.image_size, device)

    if args.mode == "bidir":
        scores_axis = eval_mod.run_single_label_bidirectional(
            predictor=predictor,
            images=images,
            video_height=video_height,
            video_width=video_width,
            label_mask=mask_axis,
            prompt_type=prompt_type,
            autocast_device=autocast_device,
            use_bootstrap=args.bootstrap,
            use_3d_filter=not args.disable_3d_filter,
            min_component_volume=args.min_component_volume,
        )
    else:
        scores_axis = eval_mod.run_single_label(
            predictor=predictor,
            images=images,
            video_height=video_height,
            video_width=video_width,
            label_mask=mask_axis,
            prompt_type=prompt_type,
            autocast_device=autocast_device,
        )
    return restore_axis(scores_axis, axis)


def fuse_scores(axis_scores: list[np.ndarray], fusion: str) -> np.ndarray:
    stacked = np.stack(axis_scores, axis=0)
    if fusion == "mean_prob":
        probs = sigmoid(stacked).mean(axis=0)
        return probs - 0.5
    if fusion == "agreement":
        votes = (stacked > 0).sum(axis=0)
        return votes.astype(np.float32) - (stacked.shape[0] / 2.0)
    return stacked.mean(axis=0)


def evaluate_case(args, eval_mod, predictors, npz_path: Path, device: str, autocast_device: str):
    data = np.load(npz_path, allow_pickle=True)
    imgs_3d, gts_3d = eval_mod.normalize_npz_case(
        npz_path,
        data["imgs"],
        data["gts"],
        image_channel_index=args.image_channel_index,
    )
    labels = [int(label) for label in np.unique(gts_3d) if label != 0]

    pred_label_map = np.zeros(gts_3d.shape, dtype=np.uint16)
    score_stack = []
    label_order = []
    case_metrics = []

    for label in labels:
        label_mask = (gts_3d == label).astype(np.uint8)
        axis_scores = []
        for axis in args.axes:
            scores = run_axis_label(
                eval_mod=eval_mod,
                predictor=predictors[axis],
                imgs_3d=imgs_3d,
                label_mask=label_mask,
                axis=axis,
                prompt_type=args.prompt_type,
                device=device,
                autocast_device=autocast_device,
                args=args,
            )
            axis_scores.append(scores)

        fused = fuse_scores(axis_scores, args.fusion)
        pred_binary = fused > 0.0
        case_metrics.append(
            {
                "case": npz_path.name,
                "label": label,
                "dice": dice_score(pred_binary, label_mask),
                "iou": iou_score(pred_binary, label_mask),
                "smoothness": surface_smoothness(pred_binary),
                "gt_voxels": int(label_mask.sum()),
                "pred_voxels": int(pred_binary.sum()),
            }
        )
        score_stack.append(fused)
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
    args = parse_args()

    try:
        import torch as _torch
        from PIL import Image as _Image
        from tqdm import tqdm
        from sam2.build_sam import get_best_available_device
        if args.mode == "bidir":
            import scripts.eval_npz_dataset_bidirectional as eval_mod
        else:
            import scripts.eval_npz_dataset as eval_mod
    except ImportError as exc:
        raise SystemExit("MedSAM2 dependencies are not available in this environment.") from exc

    eval_mod.torch = _torch
    eval_mod.Image = _Image

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    case_paths = eval_mod.load_case_paths(
        args.dataset_dir.resolve(),
        args.file_list.resolve() if args.file_list else None,
    )
    if args.max_cases is not None:
        case_paths = case_paths[: args.max_cases]
    if not case_paths:
        raise SystemExit("No NPZ files were selected for evaluation.")

    device = get_best_available_device()
    autocast_device = "cuda" if device == "cuda" else "cpu"

    predictors = {}
    for axis in args.axes:
        ckpt = checkpoint_for_axis(args, axis)
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint for axis {axis}: {ckpt}")
        predictors[axis] = build_predictor(args, ckpt, device)

    all_metrics = []
    predictions_dir = output_dir / "predictions"
    if args.save_preds:
        predictions_dir.mkdir(parents=True, exist_ok=True)

    for npz_path in tqdm(case_paths, desc=f"Evaluating multi-axis {args.mode}"):
        pred_label_map, case_metrics = evaluate_case(
            args=args,
            eval_mod=eval_mod,
            predictors=predictors,
            npz_path=npz_path,
            device=device,
            autocast_device=autocast_device,
        )
        all_metrics.extend(case_metrics)
        if args.save_preds:
            np.savez_compressed(predictions_dir / npz_path.name, segs=pred_label_map)

    metrics_path = output_dir / "case_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case", "label", "dice", "iou", "smoothness", "gt_voxels", "pred_voxels"],
        )
        writer.writeheader()
        writer.writerows(all_metrics)

    summary = {
        "num_cases": len(case_paths),
        "num_labels": len(all_metrics),
        "mean_dice": float(np.mean([r["dice"] for r in all_metrics])) if all_metrics else None,
        "mean_iou": float(np.mean([r["iou"] for r in all_metrics])) if all_metrics else None,
        "mean_smoothness": float(np.mean([r["smoothness"] for r in all_metrics])) if all_metrics else None,
        "checkpoint": str(args.checkpoint.resolve()),
        "z_checkpoint": str(args.z_checkpoint.resolve()) if args.z_checkpoint else None,
        "y_checkpoint": str(args.y_checkpoint.resolve()) if args.y_checkpoint else None,
        "x_checkpoint": str(args.x_checkpoint.resolve()) if args.x_checkpoint else None,
        "config": str(args.cfg.resolve()),
        "dataset_dir": str(args.dataset_dir.resolve()),
        "file_list": str(args.file_list.resolve()) if args.file_list else None,
        "prompt_type": args.prompt_type,
        "image_channel_index": args.image_channel_index,
        "inference_mode": f"multi_axis_{args.mode}",
        "axes": args.axes,
        "fusion": args.fusion,
        "bootstrap": args.bootstrap,
        "memory_temporal_stride_for_eval": args.memory_temporal_stride_for_eval,
        "use_3d_filter": not args.disable_3d_filter,
        "min_component_volume": args.min_component_volume,
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
