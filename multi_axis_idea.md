Multi-axis 3D medical segmentation

Training step:

  Paired batch: {
      "z": clip from case_A  (num_frames, H, W)
      "y": clip from case_A  (num_frames, H, W)  <- same case
      "x": clip from case_A  (num_frames, H, W)  <- same case
      "z_start": int   <- where in the volume this clip starts
      "y_start": int
      "x_start": int
      "volume_shape": (D, H, W)
  }

  Forward Z clip → logits_z  shape (num_frames, H, W)
  Forward Y clip → logits_y  shape (num_frames, H, W)
  Forward X clip → logits_x  shape (num_frames, H, W)

  Project each back to 3D volume coordinates:
      P_z[z_start:z_start+F, :, :]  = sigmoid(logits_z)
      P_y[:, y_start:y_start+F, :]  = sigmoid(logits_y)
      P_x[:, :, x_start:x_start+F]  = sigmoid(logits_x)

  Find voxels where ALL THREE axes have predictions:
      overlap_mask = (P_z > 0) & (P_y > 0) & (P_x > 0)

  Consistency loss on overlapping voxels:
      loss_consistency = MSE(P_z[overlap], P_y[overlap])
                       + MSE(P_y[overlap], P_x[overlap])
                       + MSE(P_z[overlap], P_x[overlap])

  Total loss = loss_seg_z + loss_seg_y + loss_seg_x
             + weight * loss_consistency


	
Inference Flow:
For each test case (full volume):

  Pass 1 (Z):
      For each Z window:
          run SAM2 predictor
          collect logits per frame
          place into P_z[D, H, W]

  Pass 2 (Y):
      For each Y window:
          run SAM2 predictor
          collect logits per frame
          place into P_y[D, H, W]

  Pass 3 (X):
      For each X window:
          run SAM2 predictor
          collect logits per frame
          place into P_x[D, H, W]

  Fuse:
      P_fused = (P_z + P_y + P_x) / 3
      pred = (P_fused > 0.5).astype(uint8)

  Evaluate:
      dice(pred, gt)
      iou(pred, gt)
      smoothness(pred)	

Plan:

CREATE (3 files):

  1. training/dataset/paired_axis_dataset.py
         PairedAxisDataset
         - groups clips by case_id (parsed from filename)
         - returns {z, y, x, metadata} per item
         - handles cases where some axes have fewer clips

  2. training/loss_fns_consistency.py
         CrossAxisConsistencyLoss
         - takes logits from Z/Y/X passes + clip metadata
         - projects to 3D volume space
         - computes MSE on overlapping voxels only
         - confidence gating (only where at least one axis is confident)

  3. scripts/eval_multi_axis_fused.py
         - loads full test volume (reconstructed from clips)
         - runs three SAM2 passes (causal or bidir)
         - reconstructs P_z, P_y, P_x volumes
         - fuses and evaluates
         - saves summary.json compatible with existing write_comparison()


MODIFY (3 files):

  4. sam2/configs/sam2.1_hiera_tiny_finetune512.yaml
         - add consistency_loss_weight param to scratch
         - add paired dataset config option

  5. multi_axis_pipeline.py
         - generate_training_config(): option to use PairedAxisDataset
         - run_evaluation(): add eval_multi_axis_fused.py call path
         - MODEL_SPECS: add fused variants

  6. training/trainer.py  (need to see this)
         - add consistency loss call after standard loss
         - OR: wrap in a new MultiAxisTrainer subclass
           so we don't break existing trainer behavior

