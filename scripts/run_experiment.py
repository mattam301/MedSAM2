#!/usr/bin/env python3
"""Shared runner for MedSAM2 stacking, bidirectional, and multi‑axis experiments."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_SPECS: dict[str, dict[str, Any]] = {
    # Original models
    "base": {
        "label": "Base (pretrained)",
        "input": "1-slice",
        "kind": "base",
    },
    "stacked_causal": {
        "label": "Stacked+Causal",
        "input": "clip",
        "kind": "causal",
    },
    "stacked_bidir": {
        "label": "Stacked+Bidir",
        "input": "clip",
        "kind": "bidir",
    },
    "stacked_bidir_unique": {
        "label": "Stacked+Bidir+Unique",
        "input": "clip",
        "kind": "bidir",
        "unique": True,
    },
    "stacked_bidir_boot": {
        "label": "Stacked+Bidir+Boot",
        "input": "clip",
        "kind": "bidir",
        "bootstrap": True,
    },
    "stacked_bidir_boot_unique": {
        "label": "Stacked+Bidir+Boot+Unique",
        "input": "clip",
        "kind": "bidir",
        "bootstrap": True,
        "unique": True,
    },
    # ------------------------------------------------------------------
    # Multi‑axis variants (Strategy 1+2)
    # ------------------------------------------------------------------
    "multi_axis_causal": {
        "label": "MultiAxis+Causal",
        "input": "clip",
        "kind": "causal",
    },
    "multi_axis_bidir": {
        "label": "MultiAxis+Bidir",
        "input": "clip",
        "kind": "bidir",
    },
}

MODEL_ORDERS = {
    "core": ["base", "stacked_causal", "stacked_bidir"],
    "full": [
        "base",
        "stacked_causal",
        "stacked_bidir",
        "stacked_bidir_unique",
        "stacked_bidir_boot",
        "stacked_bidir_boot_unique",
    ],
    "multi_axis": [
        "base",
        "stacked_causal",
        "stacked_bidir",
        "multi_axis_causal",
        "multi_axis_bidir",
    ],
}


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip()).strip("_").lower()


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("\n$ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_dataset_dir(meta: dict[str, Any], dataset_slug: str) -> Path | None:
    for name, info in meta["splits"].items():
        if slugify(name) == dataset_slug:
            return Path(info["dataset_dir"])
    return None


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
) -> int:
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
            imgs_list, gts_list = [], []
            for npz_path in window_paths:
                with np.load(npz_path, allow_pickle=True) as data:
                    imgs_list.append(data["imgs"])
                    gts_list.append(data["gts"])
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


def metric_value(metrics: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return float(value)
    return float("nan")


def fmt(value: float) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) and not math.isnan(value) else "N/A"


def collect_metrics(eval_root: Path, model_order: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    by_dataset: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for tag in model_order:
        for metrics_path in sorted((eval_root / tag).rglob("summary.json")):
            metrics = load_json(metrics_path)
            by_dataset[metrics_path.parent.name][tag] = {
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

    width_ds, width_model = 24, 30
    sep = "=" * 98
    thin = "-" * 98

    print(f"\n{sep}")
    print("RESULTS")
    print(sep)
    print(
        f"{'Dataset':<{width_ds}} {'Model':<{width_model}} "
        f"{'Input':<8} {'Dice':>8} {'IoU':>8} {'Smooth':>8}"
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
                f"{spec['input']:<8} {fmt(row['dice']):>8} "
                f"{fmt(row['iou']):>8} {fmt(row['smoothness']):>8}"
            )
        print(thin)

    print("\nImprovement over base")
    print(thin)
    print(f"{'Dataset':<{width_ds}} {'Model':<{width_model}} {'Delta Dice':>12} {'Delta IoU':>12}")
    for dataset in sorted(by_dataset):
        if "base" not in by_dataset[dataset]:
            continue
        base = by_dataset[dataset]["base"]
        for tag in model_order:
            if tag == "base" or tag not in by_dataset[dataset]:
                continue
            row = by_dataset[dataset][tag]
            print(
                f"{dataset:<{width_ds}} {MODEL_SPECS[tag]['label']:<{width_model}} "
                f"{row['dice'] - base['dice']:+12.4f} {row['iou'] - base['iou']:+12.4f}"
            )
        print(thin)

    print("\nCross-dataset averages")
    print(thin)
    print(f"{'Model':<{width_model}} {'Mean Dice':>12} {'Mean IoU':>12} {'Mean Smooth':>12}")
    for tag in model_order:
        rows = [models[tag] for models in by_dataset.values() if tag in models]
        if not rows:
            continue
        means = {
            key: sum(row[key] for row in rows if not math.isnan(row[key])) / max(
                sum(1 for row in rows if not math.isnan(row[key])), 1
            )
            for key in ("dice", "iou", "smoothness")
        }
        print(
            f"{MODEL_SPECS[tag]['label']:<{width_model}} "
            f"{fmt(means['dice']):>12} {fmt(means['iou']):>12} {fmt(means['smoothness']):>12}"
        )

    out = comparison_dir / f"{name}_comparison.json"
    out.write_text(
        json.dumps(
            {
                "model_info": {
                    tag: {
                        "label": MODEL_SPECS[tag]["label"],
                        "input": MODEL_SPECS[tag]["input"],
                    }
                    for tag in model_order
                },
                "by_dataset": by_dataset,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved comparison: {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, optionally train, and evaluate MedSAM2 experiments "
            "(stacking, bidirectional, multi-axis). "
            "Unknown arguments are forwarded to the preparation script."
        )
    )
    parser.add_argument("experiment_name")
    parser.add_argument("--mode", choices=["eval", "train-eval"], default="eval")
    parser.add_argument("--suite", choices=["core", "full", "multi_axis"], default="full")
    parser.add_argument("--python", default=os.environ.get("PYTHON", sys.executable))
    parser.add_argument("--datasets-root", type=Path, default=Path("data/new_datasets"))
    parser.add_argument("--base-checkpoint", type=Path, default=Path("checkpoints/sam2.1_hiera_tiny.pt"))
    parser.add_argument("--infer-config", type=Path, default=Path("sam2/configs/sam2.1_hiera_t512.yaml"))
    parser.add_argument("--image-channel-index", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--context-slice-interval", type=int, default=1)
    parser.add_argument("--context-interval-mode", choices=["fixed", "dynamic"], default="fixed")
    parser.add_argument("--memory-stride", type=int, default=1)
    parser.add_argument("--eval-root-name", default=None)
    parser.add_argument("--stack-checkpoint", type=Path, default=None)
    parser.add_argument("--prepare", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--new-bidir-threshold", type=float, default=1.0)
    parser.add_argument("--new-bidir-max-context", type=int, default=3)
    parser.add_argument("--new-bidir-unique-residual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-3d-filter", action="store_true")
    parser.add_argument("--min-component-volume", type=int, default=100)
    parser.add_argument("--memory-decomposer-checkpoint", type=Path, default=None)
    # Multi‑axis specific arguments (forwarded to prepare script)
    parser.add_argument("--multi-axis-stride", type=int, default=4,
                        help="Stride for pseudo-volume generation along Y/X")
    parser.add_argument("--max-slices-per-case", type=int, default=None,
                        help="Cap Z slices per case for multi‑axis generation")
    args, extra = parser.parse_known_args()
    args.prepare_args = extra
    return args


def prepare_experiment(args: argparse.Namespace, meta_json: Path) -> None:
    """Prepare the standard Z‑stacked experiment."""
    should_prepare = args.prepare == "always" or (args.prepare == "auto" and not meta_json.exists())
    if not should_prepare:
        print(f"Using existing metadata: {meta_json}")
        return

    train_ratio = "0.0" if args.mode == "eval" else None
    cmd = [
        args.python,
        "scripts/prepare_stacking_experiment.py",
        "--datasets-root",
        str(args.datasets_root),
        "--experiment-name",
        args.experiment_name,
        "--base-checkpoint",
        str(args.base_checkpoint),
        "--infer-config",
        str(args.infer_config),
        "--image-channel-index",
        str(args.image_channel_index),
        "--num-frames",
        str(args.window_size),
    ]
    if train_ratio is not None:
        cmd.extend(["--train-ratio", train_ratio])
    cmd.extend(args.prepare_args)
    run(cmd)


def prepare_multi_axis_experiment(args: argparse.Namespace, meta_json: Path) -> None:
    """Prepare the multi‑axis data and training config."""
    should_prepare = args.prepare == "always" or (args.prepare == "auto" and not meta_json.exists())
    if not should_prepare:
        print(f"Using existing multi‑axis metadata: {meta_json}")
        return

    train_ratio = "0.7" if args.mode == "train-eval" else "0.0"
    cmd = [
        args.python,
        "scripts/multi_axis_pipeline.py",
        "--datasets-root", str(args.datasets_root),
        "--experiment-name", args.experiment_name,
        "--base-checkpoint", str(args.base_checkpoint),
        "--infer-config", str(args.infer_config),
        "--image-channel-index", str(args.image_channel_index),
        "--num-frames", str(args.window_size),
        "--stride", str(args.multi_axis_stride),
        "--axes", "z", "y", "x",
        "--train-ratio", train_ratio,
    ]
    if args.max_slices_per_case is not None:
        cmd.extend(["--max-slices-per-case", str(args.max_slices_per_case)])
    # Forward any extra args that the multi‑axis script understands
    cmd.extend(args.prepare_args)
    run(cmd)


def train_model(config_name: str, output_path: str, num_gpus: int, python: str) -> None:
    """Train a model using the given Hydra config.
    
    Args:
        config_name: Absolute or relative path to config YAML (e.g., sam2/configs/generated/multi_axis_multiaxis.yaml)
        output_path: Where to save checkpoints
        num_gpus: Number of GPUs
        python: Python executable
    """
    run([
        python,
        "training/train.py",
        "-c", config_name,
        "--output-path", output_path,
        "--use-cluster", "0",
        "--num-gpus", str(num_gpus),
        "--num-nodes", "1",
    ])

def eval_model(
    args: argparse.Namespace,
    tag: str,
    spec: dict[str, Any],
    checkpoint: Path,
    cfg: Path,
    dataset_dir: Path,
    manifest: Path,
    output_dir: Path,
) -> None:
    """Evaluate a single model variant."""
    if spec["kind"] == "base" or spec["kind"] == "causal":
        cmd = [
            args.python,
            "scripts/eval_npz_dataset.py",
            "--checkpoint", str(checkpoint),
            "--cfg", str(cfg),
            "--dataset-dir", str(dataset_dir),
            "--file-list", str(manifest),
            "--image-channel-index", str(args.image_channel_index),
            "--memory-temporal-stride-for-eval", str(args.memory_stride),
            "--output-dir", str(output_dir),
        ]
    else:  # bidir variants
        cmd = [
            args.python,
            "scripts/eval_npz_dataset_bidirectional.py",
            "--checkpoint", str(checkpoint),
            "--cfg", str(cfg),
            "--dataset-dir", str(dataset_dir),
            "--file-list", str(manifest),
            "--image-channel-index", str(args.image_channel_index),
            "--memory-temporal-stride-for-eval", str(args.memory_stride),
            "--min-component-volume", str(args.min_component_volume),
            "--output-dir", str(output_dir),
        ]
        if args.disable_3d_filter:
            cmd.append("--no-3d-filter")
        if spec.get("bootstrap"):
            cmd.append("--bootstrap")
        if spec.get("unique"):
            cmd.extend([
                "--memory-redundancy-threshold", str(args.new_bidir_threshold),
                "--memory-max-unique-context-frames", str(args.new_bidir_max_context),
            ])
            if args.new_bidir_unique_residual:
                cmd.append("--memory-unique-residual")
        if args.memory_decomposer_checkpoint:
            cmd.extend(["--memory-decomposer-checkpoint", str(args.memory_decomposer_checkpoint)])
    run(cmd)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning")
    os.environ.setdefault("SAM2_LOG_LEVEL", "ERROR")

    splits_root = args.datasets_root / "_splits" / args.experiment_name
    exp_log = Path("exp_log") / args.experiment_name
    eval_root = exp_log / (args.eval_root_name or ("eval_selected" if args.mode == "eval" else "eval"))
    comparison_dir = exp_log / "comparison"

    # ---------- Basic setup ----------
    is_multi_axis = (args.suite == "multi_axis")
    meta_json = splits_root / "experiment.json"
    multi_axis_meta_json = splits_root / "experiment_multi_axis.json"

    # 1. Prepare standard Z‑stacked data (always needed for base/stacked baselines)
    prepare_experiment(args, meta_json)
    if not meta_json.exists():
        raise SystemExit(f"Experiment metadata not found: {meta_json}")
    meta = load_json(meta_json)

    # 2. (Optionally) prepare multi‑axis data
    if is_multi_axis:
        prepare_multi_axis_experiment(args, multi_axis_meta_json)
        if not multi_axis_meta_json.exists():
            raise SystemExit(f"Multi‑axis metadata not found: {multi_axis_meta_json}")
        multi_meta = load_json(multi_axis_meta_json)
    else:
        multi_meta = {}

    # ---------- Checkpoints ----------
    base_checkpoint = Path(meta.get("base_checkpoint", str(args.base_checkpoint)))
    infer_config = Path(meta.get("infer_config", str(args.infer_config)))

    # Stacked checkpoint (from standard stacked training)
    stack_ckpt = args.stack_checkpoint or Path(meta["stack_output"]) / "checkpoints" / "checkpoint.pt"

    # Multi‑axis checkpoint (if suite includes multi‑axis models)
    if is_multi_axis:
        multi_axis_ckpt = Path(multi_meta["stack_output"]) / "checkpoints" / "checkpoint.pt"
    else:
        multi_axis_ckpt = Path("/dev/null")

    # ---------- Training ----------
    if args.mode == "train-eval":
        # Train stacked model if needed
        if not stack_ckpt.exists():
            print("Training stacked model...")
            train_model(meta["stack_config_name"], meta["stack_output"], args.num_gpus, args.python)
        else:
            print(f"Using existing stacked checkpoint: {stack_ckpt}")

        # Train multi‑axis model if needed
        if is_multi_axis and not multi_axis_ckpt.exists():
            print("Training multi‑axis model...")
            train_model(multi_meta["stack_config_name"], multi_meta["stack_output"], args.num_gpus, args.python)
        elif is_multi_axis:
            print(f"Using existing multi‑axis checkpoint: {multi_axis_ckpt}")

    # ---------- Evaluation ----------
    model_order = MODEL_ORDERS[args.suite]
    manifests = sorted((splits_root / "manifests").glob("*_slice_test.txt"))
    if not manifests:
        raise SystemExit(f"No *_slice_test.txt manifests found under {splits_root / 'manifests'}")

    interval_suffix = (
        f"{args.context_interval_mode}{args.context_slice_interval}"
        if args.context_interval_mode == "fixed"
        else "dynamic"
    )
    tmp_clip_root = exp_log / (
        f"tmp_stacked_test_w{args.window_size}_s{args.window_stride}_i{interval_suffix}"
    )

    print("\nExperiment")
    print(f"  name             : {args.experiment_name}")
    print(f"  mode             : {args.mode}")
    print(f"  suite            : {args.suite}")
    print(f"  window           : {args.window_size}, stride {args.window_stride}")
    print(f"  clip interval    : {args.context_interval_mode}, value {args.context_slice_interval}")
    print(f"  memory stride    : {args.memory_stride}")
    print(f"  eval root        : {eval_root}")
    if is_multi_axis:
        print(f"  multi‑axis stride: {args.multi_axis_stride}")

    for manifest in manifests:
        dataset_slug = manifest.stem.removesuffix("_slice_test")
        dataset_dir = resolve_dataset_dir(meta, dataset_slug)
        if dataset_dir is None:
            print(f"WARNING: cannot resolve dataset for {dataset_slug}; skipping")
            continue

        print(f"\nDataset: {dataset_slug}")
        # Build clip dataset (shared by stacked and multi‑axis)
        clip_npz_dir = tmp_clip_root / dataset_slug / "npz"
        clip_manifest = tmp_clip_root / dataset_slug / f"{dataset_slug}_stacked_test.txt"
        if clip_manifest.exists():
            clip_count = len([x for x in clip_manifest.read_text(encoding="utf-8").splitlines() if x.strip()])
            print(f"  Reusing {clip_count} clips: {clip_manifest}")
        else:
            clip_count = build_clip_dataset(
                dataset_dir=dataset_dir,
                manifest_path=manifest,
                out_npz_dir=clip_npz_dir,
                out_manifest=clip_manifest,
                window_size=args.window_size,
                window_stride=args.window_stride,
                slice_interval=args.context_slice_interval,
                interval_mode=args.context_interval_mode,
            )
            print(f"  Built {clip_count} clips: {clip_manifest}")

        for tag in model_order:
            spec = MODEL_SPECS[tag]
            # Determine checkpoint and dataset
            if spec["kind"] == "base":
                checkpoint = base_checkpoint
                eval_dataset_dir = dataset_dir
                eval_manifest = manifest
            elif tag.startswith("multi_axis"):
                if not multi_axis_ckpt.exists():
                    continue
                checkpoint = multi_axis_ckpt
                eval_dataset_dir = clip_npz_dir
                eval_manifest = clip_manifest
            else:  # stacked variants
                if not stack_ckpt.exists():
                    continue
                checkpoint = stack_ckpt
                eval_dataset_dir = clip_npz_dir
                eval_manifest = clip_manifest

            print(f"  Evaluating {tag}...")
            eval_model(
                args=args,
                tag=tag,
                spec=spec,
                checkpoint=checkpoint,
                cfg=infer_config,
                dataset_dir=eval_dataset_dir,
                manifest=eval_manifest,
                output_dir=eval_root / tag / dataset_slug,
            )

    write_comparison(eval_root, comparison_dir, model_order, args.suite)


if __name__ == "__main__":
    main()