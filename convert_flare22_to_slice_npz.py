#!/usr/bin/env python3
"""Convert FLARE22 NIfTI volumes to flat per-slice NPZ files.

Output: data/new_datasets/npz_flare22/FLARE22_Tr_XXXX_ZZZZ.npz (flat structure)
"""
from pathlib import Path
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
import re

def parse_args():
    p = argparse.ArgumentParser(description="Convert FLARE22 to flat NPZ for MedSAM2")
    p.add_argument("--images-dir", type=Path, required=True, 
                   help="Path to FLARE22Train/images/")
    p.add_argument("--labels-dir", type=Path, required=True,
                   help="Path to FLARE22Train/labels/")
    p.add_argument("--output-dir", type=Path, default="data/new_datasets/npz_flare22",
                   help="Output directory (flat structure)")
    p.add_argument("--ct-window-low", type=float, default=-200)
    p.add_argument("--ct-window-high", type=float, default=300)
    p.add_argument("--skip-empty-slices", action="store_true")
    return p.parse_args()

def resolve_label_path(labels_dir: Path, case_id: str) -> Path | None:
    """Handle FLARE22 label directory quirk."""
    candidates = [
        labels_dir / f"{case_id}.nii",
        labels_dir / f"{case_id}.nii.gz",
        labels_dir / f"{case_id}.nii" / f"{case_id}.nii",
        labels_dir / f"{case_id}.nii" / f"{case_id}.nii.gz",
        labels_dir / case_id / f"{case_id}.nii",
        labels_dir / case_id / f"{case_id}.nii.gz",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None

def ct_window_normalize(volume: np.ndarray, low: float, high: float) -> np.ndarray:
    volume = np.clip(volume, low, high)
    volume = (volume - low) / (high - low)
    return (volume * 255.0).astype(np.float32)

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove any existing files to avoid mixing
    existing = list(args.output_dir.glob("*.npz"))
    if existing:
        print(f"Removing {len(existing)} existing files in {args.output_dir}")
        for f in existing:
            f.unlink()

    image_paths = sorted(args.images_dir.glob("*.nii")) + \
                  sorted(args.images_dir.glob("*.nii.gz"))
    
    if not image_paths:
        raise SystemExit(f"No .nii files found in {args.images_dir}")

    total_slices = 0
    total_skipped = 0

    for img_path in tqdm(image_paths, desc="Converting FLARE22"):
        # Extract case ID
        stem = img_path.name.replace(".nii.gz", "").replace(".nii", "")
        case_id = stem.replace("_0000", "")

        lab_path = resolve_label_path(args.labels_dir, case_id)
        if lab_path is None:
            print(f"⚠️  No label for {case_id}, skipping")
            continue

        # Load volumes
        img_nii = nib.load(str(img_path))
        img_nii = nib.as_closest_canonical(img_nii)
        img_vol = img_nii.get_fdata().astype(np.float32)
        
        lab_nii = nib.load(str(lab_path))
        lab_nii = nib.as_closest_canonical(lab_nii)
        lab_vol = lab_nii.get_fdata().astype(np.uint8)

        if img_vol.shape != lab_vol.shape:
            print(f"⚠️  Shape mismatch {case_id}, skipping")
            continue

        # Windowing
        img_vol = ct_window_normalize(img_vol, args.ct_window_low, args.ct_window_high)
        
        # Slice along Z (axis 2 after canonical)
        num_slices = img_vol.shape[2]
        
        for z in range(num_slices):
            img_slice = img_vol[:, :, z]  # [H, W]
            lab_slice = lab_vol[:, :, z]  # [H, W]
            
            if args.skip_empty_slices and lab_slice.max() == 0:
                total_skipped += 1
                continue
            
            # Flat output: case_slice.npz
            out_name = f"{case_id}_{z:04d}.npz"
            out_path = args.output_dir / out_name
            
            # Ensure channel dim [H, W, 1]
            if img_slice.ndim == 2:
                img_slice = img_slice[..., np.newaxis]
                
            np.savez_compressed(out_path, imgs=img_slice, gts=lab_slice)
            total_slices += 1
    
    print(f"\n✅ Done!")
    print(f"   Cases processed: {len(image_paths)}")
    print(f"   Slices written:  {total_slices} to {args.output_dir}")
    print(f"   Skipped empty:   {total_skipped}")
    
    # Verify
    check = list(args.output_dir.glob("*.npz"))
    print(f"   Verification: {len(check)} .npz files exist")

if __name__ == "__main__":
    main()