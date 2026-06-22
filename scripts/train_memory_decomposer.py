#!/usr/bin/env python3
"""Train the learned memory decomposer on stacked NPZ clips.

This is an auxiliary trainer for the bidirectional inference decomposer. It
freezes SAM2, collects mask-memory features from prompted training clips, and
trains the small learned decomposer heads with GT-derived proxy targets:

  * redundancy scorer: predicts slice-mask Dice between current and memory frame
  * unique extractor: predicts memory features restricted to GT-unique regions

The output checkpoint can be loaded during bidirectional evaluation with
``--memory-decomposer-checkpoint``.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = None
tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train learned memory decomposer heads from stacked NPZ clips."
    )
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument(
        "--splits-root",
        type=Path,
        default=REPO_ROOT / "data" / "new_datasets" / "_splits",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--cfg",
        type=Path,
        default=None,
        help=(
            "Inference config. If omitted with --experiment-name, uses experiment.json; "
            "otherwise defaults to sam2/configs/sam2.1_hiera_t512.yaml."
        ),
    )
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--file-list", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt-type", choices=["box", "point", "mask"], default="box")
    parser.add_argument("--image-channel-index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-pairs-per-label", type=int, default=32)
    parser.add_argument("--temporal-radius", type=int, default=4)
    parser.add_argument("--score-loss-weight", type=float, default=1.0)
    parser.add_argument("--unique-loss-weight", type=float, default=0.25)
    parser.add_argument("--memory-feat-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-score-delta", type=float, default=0.5)
    parser.add_argument("--residual-scale", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_config_path(path: Path) -> str:
    return f"//{path.resolve()}"


def resolve_training_inputs(args: argparse.Namespace) -> tuple[Path, Path, list[Path]]:
    if args.experiment_name:
        meta_path = args.splits_root / args.experiment_name / "experiment.json"
        if not meta_path.exists():
            raise SystemExit(f"Experiment metadata not found: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        checkpoint = (
            args.checkpoint
            if args.checkpoint is not None
            else Path(meta["stack_output"]) / "checkpoints" / "checkpoint.pt"
        )
        cfg = args.cfg if args.cfg is not None else Path(meta["infer_config"])

        case_paths = []
        for _name, info in meta["splits"].items():
            stacked_dir = Path(info["stacked_dir"])
            manifest = Path(info["stacked_train_manifest"])
            case_paths.extend(load_manifest_paths(stacked_dir, manifest))
        return checkpoint, cfg, case_paths

    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required when --experiment-name is not used.")
    if args.dataset_dir is None or args.file_list is None:
        raise SystemExit(
            "--dataset-dir and --file-list are required when --experiment-name is not used."
        )
    cfg = args.cfg or REPO_ROOT / "sam2" / "configs" / "sam2.1_hiera_t512.yaml"
    return args.checkpoint, cfg, load_manifest_paths(args.dataset_dir, args.file_list)


def load_manifest_paths(dataset_dir: Path, file_list: Path) -> list[Path]:
    case_paths = []
    for line in file_list.read_text(encoding="utf-8").splitlines():
        case_id = line.strip()
        if not case_id:
            continue
        suffix = "" if case_id.endswith(".npz") else ".npz"
        case_paths.append((dataset_dir / f"{case_id}{suffix}").resolve())
    return case_paths


def jsonable_args(args: argparse.Namespace) -> dict:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def mask_dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    denom = int(a.sum() + b.sum())
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / denom)


def resize_mask(mask: np.ndarray, size: tuple[int, int], device: str):
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None].to(device)
    return torch.nn.functional.interpolate(tensor, size=size, mode="nearest")


def load_label_outputs(
    predictor,
    images,
    video_height: int,
    video_width: int,
    label_mask: np.ndarray,
    prompt_type: str,
    autocast_device: str,
):
    from eval_npz_dataset_bidirectional import (
        get_bbox,
        get_center_point,
        get_prompt_slice,
    )

    z_mid = get_prompt_slice(label_mask)
    prompt_slice_mask = label_mask[z_mid].astype(np.uint8)

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
            mask_prompt = (
                (out_mask_logits[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)
            )
        else:
            points, labels = get_center_point(prompt_slice_mask)
            _, _, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=z_mid,
                obj_id=1,
                points=points,
                labels=labels,
            )
            mask_prompt = (
                (out_mask_logits[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)
            )

        predictor.add_new_mask(
            inference_state,
            frame_idx=z_mid,
            obj_id=1,
            mask=mask_prompt,
        )

        for _frame_idx, _obj_ids, _mask_logits in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=0,
        ):
            pass

        outputs = {}
        output_dict = inference_state["output_dict"]
        for frame_idx, out in output_dict["cond_frame_outputs"].items():
            outputs[int(frame_idx)] = out
        for frame_idx, out in output_dict["non_cond_frame_outputs"].items():
            outputs[int(frame_idx)] = out

        raw_features = {
            frame_idx: out["maskmem_features"]
            for frame_idx, out in outputs.items()
            if out.get("maskmem_features") is not None
        }
        predictor.reset_state(inference_state)

    features = {
        frame_idx: feat.detach().clone().float()
        for frame_idx, feat in raw_features.items()
    }
    return features


def train_pair(
    decomposer,
    optimizer,
    current_feats,
    memory_feats,
    current_mask: np.ndarray,
    memory_mask: np.ndarray,
    score_target: float,
    score_loss_weight: float,
    unique_loss_weight: float,
) -> tuple[float, float, float]:
    from sam2.memory_decomposer import _match_current_to_memory, _pool_current_features

    target_device = next(decomposer.parameters()).device
    current_feats = current_feats.to(target_device, dtype=torch.float32)
    memory_feats = memory_feats.to(target_device, dtype=torch.float32)

    with torch.no_grad():
        baseline = decomposer.baseline.decompose(current_feats, memory_feats)

    current_vec = _pool_current_features(current_feats)
    memory_avg = memory_feats.mean(dim=[2, 3])
    current_vec, memory_avg = _match_current_to_memory(current_vec, memory_avg)
    concat_features = torch.cat([current_vec, memory_avg], dim=-1)

    score_delta = decomposer.redundancy_scorer(concat_features).mean()
    baseline_score = torch.as_tensor(
        baseline["redundancy_score"],
        device=target_device,
        dtype=torch.float32,
    )
    pred_score = (baseline_score + decomposer.max_score_delta * score_delta).clamp(
        -1.0, 1.0
    )
    target_score = torch.tensor(score_target, device=target_device, dtype=torch.float32)
    score_loss = torch.nn.functional.mse_loss(pred_score, target_score)

    _, _, feat_h, feat_w = memory_feats.shape
    current_small = resize_mask(current_mask, (feat_h, feat_w), target_device)
    memory_small = resize_mask(memory_mask, (feat_h, feat_w), target_device)
    unique_weight = (memory_small * (1.0 - current_small)).to(memory_feats.dtype)
    unique_target = memory_feats * unique_weight

    unique_delta = decomposer.unique_extractor(memory_feats)
    unique_pred = baseline["unique_residual"] + (
        decomposer.residual_scale * memory_feats * unique_delta
    )
    unique_loss = torch.nn.functional.smooth_l1_loss(unique_pred, unique_target)

    loss = score_loss_weight * score_loss + unique_loss_weight * unique_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return float(loss.item()), float(score_loss.item()), float(unique_loss.item())


def main() -> None:
    global torch, tqdm
    args = parse_args()

    try:
        import torch as _torch
        from PIL import Image as _Image
        from tqdm import tqdm as _tqdm
        import eval_npz_dataset_bidirectional as eval_bidir
        from eval_npz_dataset_bidirectional import normalize_npz_case, preprocess_volume
        from sam2.bidirectional_video_predictor import (
            build_bidir_sam2_video_predictor_npz,
        )
        from sam2.build_sam import get_best_available_device
        from sam2.memory_decomposer import LearnedMemoryDecomposer
    except ImportError as exc:
        raise SystemExit(
            "MedSAM2 dependencies are not available. Activate the medsam2 env first."
        ) from exc

    torch = _torch
    tqdm = _tqdm
    eval_bidir.torch = _torch
    eval_bidir.Image = _Image
    eval_bidir.tqdm = _tqdm
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_best_available_device()
    autocast_device = "cuda" if device == "cuda" else "cpu"

    checkpoint_path, cfg_path, case_paths = resolve_training_inputs(args)
    if args.max_cases is not None:
        case_paths = case_paths[: args.max_cases]
    if not case_paths:
        raise SystemExit("No NPZ files selected for decomposer training.")

    print("Building frozen SAM2 predictor for feature extraction...")
    predictor = build_bidir_sam2_video_predictor_npz(
        resolve_config_path(cfg_path),
        ckpt_path=str(checkpoint_path.resolve()),
        device=device,
    )
    predictor.eval()
    for param in predictor.parameters():
        param.requires_grad_(False)

    decomposer = LearnedMemoryDecomposer(
        memory_feat_dim=args.memory_feat_dim,
        hidden_dim=args.hidden_dim,
        max_score_delta=args.max_score_delta,
        residual_scale=args.residual_scale,
    ).to(device=device, dtype=torch.float32)
    decomposer.train()

    optimizer = torch.optim.AdamW(
        decomposer.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        random.shuffle(case_paths)
        total_loss = total_score = total_unique = 0.0
        num_pairs = 0

        pbar = tqdm(case_paths, desc=f"Decomposer train epoch {epoch}/{args.epochs}")
        for npz_path in pbar:
            data = np.load(npz_path, allow_pickle=True)
            imgs_3d, gts_3d = normalize_npz_case(
                npz_path,
                data["imgs"],
                data["gts"],
                image_channel_index=args.image_channel_index,
            )
            video_height, video_width = imgs_3d.shape[1:]
            images = preprocess_volume(imgs_3d, predictor.image_size, device)
            labels = [int(label) for label in np.unique(gts_3d) if label != 0]

            for label in labels:
                label_mask = (gts_3d == label).astype(np.uint8)
                if label_mask.sum() == 0:
                    continue

                features = load_label_outputs(
                    predictor=predictor,
                    images=images,
                    video_height=video_height,
                    video_width=video_width,
                    label_mask=label_mask,
                    prompt_type=args.prompt_type,
                    autocast_device=autocast_device,
                )
                frame_ids = sorted(features)
                pairs = [
                    (i, j)
                    for i in frame_ids
                    for j in frame_ids
                    if i != j and abs(i - j) <= args.temporal_radius
                ]
                if len(pairs) > args.max_pairs_per_label:
                    pairs = random.sample(pairs, args.max_pairs_per_label)

                for current_idx, memory_idx in pairs:
                    score_target = mask_dice(
                        label_mask[current_idx],
                        label_mask[memory_idx],
                    )
                    loss, score_loss, unique_loss = train_pair(
                        decomposer=decomposer,
                        optimizer=optimizer,
                        current_feats=features[current_idx],
                        memory_feats=features[memory_idx],
                        current_mask=label_mask[current_idx],
                        memory_mask=label_mask[memory_idx],
                        score_target=score_target,
                        score_loss_weight=args.score_loss_weight,
                        unique_loss_weight=args.unique_loss_weight,
                    )
                    total_loss += loss
                    total_score += score_loss
                    total_unique += unique_loss
                    num_pairs += 1

            if num_pairs:
                pbar.set_postfix(loss=f"{total_loss / num_pairs:.4f}")

        epoch_stats = {
            "epoch": epoch,
            "pairs": num_pairs,
            "loss": total_loss / max(num_pairs, 1),
            "score_loss": total_score / max(num_pairs, 1),
            "unique_loss": total_unique / max(num_pairs, 1),
        }
        history.append(epoch_stats)
        print(
            f"epoch={epoch} pairs={num_pairs} "
            f"loss={epoch_stats['loss']:.6f} "
            f"score={epoch_stats['score_loss']:.6f} "
            f"unique={epoch_stats['unique_loss']:.6f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": decomposer.state_dict(),
        "config": {
            "memory_feat_dim": args.memory_feat_dim,
            "hidden_dim": args.hidden_dim,
            "use_augmentation": False,
            "augmentation_scale": 0.1,
            "max_score_delta": args.max_score_delta,
            "residual_scale": args.residual_scale,
        },
        "train_args": jsonable_args(args),
        "checkpoint": str(checkpoint_path.resolve()),
        "cfg": str(cfg_path.resolve()),
        "history": history,
    }
    torch.save(checkpoint, args.output)

    history_path = args.output.with_suffix(".json")
    history_path.write_text(
        json.dumps({"history": history, "args": jsonable_args(args)}, indent=2)
    )
    print(f"Saved decomposer checkpoint: {args.output}")
    print(f"Saved training history     : {history_path}")


if __name__ == "__main__":
    main()
