import numpy as np
from pathlib import Path
import shutil

# Configuration
INPUT_DIR = "/cm/archive/tuanma5/large_scale_data/output/npy/lungrads"
OUTPUT_DIR = "/cm/shared/tuanma5/workspace/medsam2_3d/MedSAM2/MedSAM2/data/new_datasets/lungrads"  # Change this!

# Setup paths
input_path = Path(INPUT_DIR)
output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)

gts_dir = input_path / "gts"
imgs_dir = input_path / "imgs"

# Process files
for gts_file in sorted(gts_dir.glob("*.npy")):
    filename = gts_file.stem
    imgs_file = imgs_dir / f"{filename}.npy"
    
    if not imgs_file.exists():
        print(f"Skipping {filename}: no matching img file")
        continue
    
    # Load and merge
    gts = np.load(gts_file)
    imgs = np.load(imgs_file)
    
    np.savez(output_path / f"{filename}.npz", gts=gts, imgs=imgs)
    print(f"Created: {filename}.npz")

print(f"\nDone! Output in: {output_path}")