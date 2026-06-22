#!/usr/bin/env python3
"""
test_multi_axis_pipeline.py

Quick test to verify the pipeline works end-to-end before full training.
"""

from pathlib import Path
import sys

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image


def test_data_generation():
    """Test that multi-axis pseudo-volumes are created correctly."""
    print("=" * 60)
    print("TEST: Multi-Axis Data Generation")
    print("=" * 60)
    
    # Find generated data
    multi_axis_root = Path("data/new_datasets/_multi_axis")
    if not multi_axis_root.exists():
        print("No multi-axis data found. Run multi_axis_pipeline.py first.")
        return
    
    for ds_dir in sorted(multi_axis_root.iterdir()):
        if not ds_dir.is_dir():
            continue
        
        print(f"\nDataset: {ds_dir.name}")
        
        for axis_name in ['z', 'y', 'x']:
            axis_dir = ds_dir / axis_name
            if not axis_dir.is_dir():
                continue
            
            npz_files = sorted(axis_dir.glob("*.npz"))
            if not npz_files:
                print(f"  {axis_name}: No volumes")
                continue
            
            # Load first volume
            sample = npz_files[0]
            with np.load(sample) as data:
                imgs = data['imgs']
                gts = data['gts']
            
            print(f"  {axis_name}: {len(npz_files)} volumes")
            print(f"    Sample: {sample.name}")
            print(f"    Shape: imgs={imgs.shape}, gts={gts.shape}")
            print(f"    Mask frames: {(gts > 0).any(axis=(1,2)).sum()}/{imgs.shape[0]}")
            
            # Verify shape consistency
            expected_shape = (8, 512, 512)  # num_frames, target_size, target_size
            if imgs.shape != expected_shape:
                print(f"    WARNING: Expected shape {expected_shape}, got {imgs.shape}")


def test_axis_sampling():
    """Test that axis-aware sampling works (requires torch)."""
    print("\n" + "=" * 60)
    print("TEST: Axis-Aware Sampling")
    print("=" * 60)
    
    try:
        import torch
        from collections import Counter
    except ImportError:
        print("PyTorch not available, skipping sampling test.")
        return
    
    # Simple mock datasets
    class MockDataset:
        def __init__(self, name, size):
            self.name = name
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return {'axis': self.name, 'index': idx}
    
    # Import our custom dataset (adjust path as needed)
    try:
        from sam2.training.dataset.vos_dataset import MultiAxisVOSDataset
    except ImportError:
        print("MultiAxisVOSDataset not found. Make sure it's in vos_dataset.py")
        return
    
    # Create mock datasets
    z_ds = MockDataset('z', 1000)
    y_ds = MockDataset('y', 1000)
    x_ds = MockDataset('x', 1000)
    
    # Create multi-axis dataset
    multi_ds = MultiAxisVOSDataset(
        z_dataset=z_ds,
        y_dataset=y_ds,
        x_dataset=x_ds,
        axis_weights={'z': 0.34, 'y': 0.33, 'x': 0.33},
        training=True,
    )
    
    print(f"Effective dataset size: {len(multi_ds)}")
    print(f"Axis statistics: {multi_ds.get_axis_statistics()}")
    
    # Sample 1000 items and count axis distribution
    axis_counts = Counter()
    for i in range(1000):
        sample = multi_ds[i]
        axis_counts[sample['_axis']] += 1
    
    print(f"\nSampled axis distribution (n=1000):")
    for ax in ['z', 'y', 'x']:
        print(f"  {ax}: {axis_counts[ax]} ({axis_counts[ax]/10:.1f}%)")
    
    target = 333
    if all(abs(axis_counts.get(ax, 0) - target) < 50 for ax in ['z', 'y', 'x']):
        print("✓ Axis sampling is approximately uniform")
    else:
        print("⚠ Axis sampling may be biased")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--test", choices=["data", "sampling", "all"], default="all")
    args = p.parse_args()
    
    if args.test in ["data", "all"]:
        test_data_generation()
    
    if args.test in ["sampling", "all"]:
        test_axis_sampling()