#!/usr/bin/env python3
"""Convert PPMI NIfTI volumes into flat per-slice NPZ files.

This format matches the existing stacked-experiment pipeline:
  - each .npz stores one 2-D slice
  - imgs has shape (H, W, 1)
  - gts has shape (H, W)

PPMI appears to be image-only in the downloaded tree, so this script writes
zero-valued masks unless an optional labels tree is provided.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = REPO_ROOT / "PPMI"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "new_datasets" / "ppmi_slices"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PPMI NIfTI volumes to flat per-slice NPZ files."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory containing PPMI .nii / .nii.gz volumes.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where converted slice .npz files will be written.",
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=None,
        help=(
            "Optional parallel labels root. If present, the script will look for "
            "a matching NIfTI mask with the same relative path."
        ),
    )
    parser.add_argument(
        "--percentile-low",
        type=float,
        default=1.0,
        help="Lower percentile used for MRI intensity normalization.",
    )
    parser.add_argument(
        "--percentile-high",
        type=float,
        default=99.0,
        help="Upper percentile used for MRI intensity normalization.",
    )
    parser.add_argument(
        "--keep-tree",
        action="store_true",
        help="Preserve the input directory tree under the output root.",
    )
    return parser.parse_args()


def find_nifti_files(root: Path) -> list[Path]:
    files = list(root.rglob("*.nii")) + list(root.rglob("*.nii.gz"))
    return sorted({path.resolve() for path in files if path.is_file()}, key=lambda p: p.as_posix())


def strip_nii_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def normalize_volume(volume: np.ndarray, low_pct: float, high_pct: float) -> tuple[np.ndarray, float, float]:
    volume = np.asarray(volume, dtype=np.float32)
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        raise ValueError("Volume contains no finite voxels.")

    sample = finite[finite != 0]
    if sample.size == 0:
        sample = finite

    lo = float(np.percentile(sample, low_pct))
    hi = float(np.percentile(sample, high_pct))
    if hi <= lo:
        hi = lo + 1.0

    clipped = np.clip(volume, lo, hi)
    scaled = ((clipped - lo) / (hi - lo) * 255.0).round().astype(np.uint8)
    return scaled, lo, hi


def output_slice_path(
    input_path: Path,
    input_root: Path,
    output_root: Path,
    slice_idx: int,
    keep_tree: bool,
) -> Path:
    rel_path = input_path.relative_to(input_root)
    stem = strip_nii_suffix(rel_path)
    if keep_tree:
        return output_root / rel_path.with_name(f"{stem}_{slice_idx:04d}.npz")
    flat_name = rel_path.as_posix().replace(".nii.gz", "").replace(".nii", "").replace("/", "__")
    return output_root / f"{flat_name}_{slice_idx:04d}.npz"


def load_optional_mask(input_path: Path, input_root: Path, labels_root: Path | None) -> np.ndarray | None:
    if labels_root is None:
        return None

    rel_path = input_path.relative_to(input_root)
    rel_dir = rel_path.parent
    base_name = strip_nii_suffix(rel_path)
    for suffix in (".nii.gz", ".nii"):
        candidate = labels_root / rel_dir / f"{base_name}{suffix}"
        if candidate.exists():
            return np.asarray(sitk.GetArrayFromImage(sitk.ReadImage(str(candidate))))
    return None


def main() -> None:
    args = parse_args()
    if not args.input_root.exists():
        raise SystemExit(f"Input root not found: {args.input_root}")

    input_root = args.input_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    nifti_files = find_nifti_files(input_root)
    if not nifti_files:
        raise SystemExit(f"No .nii or .nii.gz files found under {input_root}")

    written = 0
    source_volumes = 0
    dummy_labels = 0
    reused_labels = 0

    for input_path in nifti_files:
        image = sitk.ReadImage(str(input_path))
        volume = np.asarray(sitk.GetArrayFromImage(image))
        if volume.ndim != 3:
            volume = np.squeeze(volume)
        if volume.ndim != 3:
            raise ValueError(f"Unsupported volume rank in {input_path}: shape={volume.shape}")

        imgs, window_low, window_high = normalize_volume(
            volume,
            low_pct=args.percentile_low,
            high_pct=args.percentile_high,
        )

        mask_volume = load_optional_mask(input_path, input_root, args.labels_root)
        if mask_volume is not None:
            mask_volume = np.asarray(mask_volume)
            if mask_volume.ndim != 3:
                mask_volume = np.squeeze(mask_volume)
            if mask_volume.shape != imgs.shape:
                raise ValueError(
                    f"Shape mismatch for {input_path}: imgs={imgs.shape}, gts={mask_volume.shape}"
                )
            reused_labels += 1
        else:
            dummy_labels += 1

        spacing_xyz = np.asarray(image.GetSpacing(), dtype=np.float32)
        origin_xyz = np.asarray(image.GetOrigin(), dtype=np.float32)
        direction_xyz = np.asarray(image.GetDirection(), dtype=np.float32)

        source_volumes += 1
        for slice_idx in range(imgs.shape[0]):
            out_path = output_slice_path(
                input_path=input_path,
                input_root=input_root,
                output_root=args.output_root,
                slice_idx=slice_idx,
                keep_tree=args.keep_tree,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img_slice = imgs[slice_idx]
            if img_slice.ndim == 2:
                img_slice = img_slice[..., None]
            elif img_slice.ndim == 3 and img_slice.shape[-1] != 1:
                img_slice = img_slice[..., :1]

            if mask_volume is None:
                gt_slice = np.zeros(imgs.shape[1:], dtype=np.uint8)
            else:
                gt_slice = np.asarray(mask_volume[slice_idx], dtype=np.uint8)

            np.savez_compressed(
                out_path,
                imgs=img_slice,
                gts=gt_slice,
                spacing=spacing_xyz[::-1],
                origin=origin_xyz,
                direction=direction_xyz,
                source_path=str(input_path),
                source_relpath=input_path.relative_to(input_root).as_posix(),
                slice_index=np.array(slice_idx, dtype=np.int32),
                has_labels=np.array(mask_volume is not None, dtype=np.bool_),
                intensity_window=np.array([window_low, window_high], dtype=np.float32),
            )
            written += 1

    print()
    print(f"Done. Source volumes:     {source_volumes}")
    print(f"Slices written:          {written}")
    print(f"With labels reused:       {reused_labels}")
    print(f"With dummy masks:         {dummy_labels}")
    print(f"Output root:              {args.output_root}")


if __name__ == "__main__":
    main()