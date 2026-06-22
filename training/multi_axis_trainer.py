from __future__ import annotations

import os
from typing import Any

import torch

from training.loss_fns import CORE_LOSS_KEY
from training.loss_fns_consistency import CrossAxisConsistencyLoss
from training.trainer import Trainer


def _core_loss(loss_obj: Any) -> torch.Tensor:
    if isinstance(loss_obj, dict):
        return loss_obj[CORE_LOSS_KEY]
    return loss_obj


def _last_step_logits(outputs: list[dict]) -> torch.Tensor:
    """Extract one high-res logit map per frame from SAM2 training outputs."""
    frames = []
    for frame_out in outputs:
        multimasks = frame_out["multistep_pred_multimasks_high_res"][-1]
        frames.append(multimasks[:, 0].mean(dim=0))
    return torch.stack(frames, dim=0)


class MultiAxisTrainer(Trainer):
    """Trainer extension for pre-collated paired-axis batches.

    The normal MedSAM2 path still uses `training.trainer.Trainer`. This class
    expects a batch shaped like:

        {
          "axes": {"z": BatchedVideoDatapoint, "y": ..., "x": ...},
          "starts": {"z": tensor/int, ...},
          "volume_shape": tensor/list/tuple
        }
    """

    def __init__(
        self,
        *args,
        consistency_loss: CrossAxisConsistencyLoss | None = None,
        consistency_loss_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.consistency_loss_weight = float(consistency_loss_weight)
        self.consistency_loss = consistency_loss or CrossAxisConsistencyLoss(
            weight=self.consistency_loss_weight
        )

    def _step(self, batch, model, phase: str):
        if not isinstance(batch, dict) or "axes" not in batch:
            return super()._step(batch, model, phase)

        outputs_by_axis = {}
        logits_by_axis = {}
        seg_losses = []
        step_losses = {}
        batch_size = 1

        for axis, axis_batch in batch["axes"].items():
            outputs = model(axis_batch)
            outputs_by_axis[axis] = outputs
            targets = axis_batch.masks
            key = axis_batch.dict_key
            loss_obj = self.loss[key](outputs, targets)
            seg_loss = _core_loss(loss_obj)
            seg_losses.append(seg_loss)
            step_losses[f"Losses/{phase}_{key}_{axis}_seg"] = seg_loss
            logits_by_axis[axis] = _last_step_logits(outputs)
            batch_size = len(axis_batch.img_batch)

        loss = torch.stack(seg_losses).mean()
        if self.consistency_loss_weight > 0:
            consistency = self.consistency_loss(
                logits_by_axis=logits_by_axis,
                starts_by_axis=batch["starts"],
                volume_shape=batch["volume_shape"],
            )
            loss = loss + consistency
            step_losses[f"Losses/{phase}_multi_axis_consistency"] = consistency

        loss_str = f"Losses/{phase}_multi_axis_loss"
        loss_log_str = os.path.join("Step_Losses", loss_str)
        if self.steps[phase] % self.logging_conf.log_scalar_frequency == 0:
            self.logger.log(loss_log_str, loss, self.steps[phase])

        self.steps[phase] += 1
        return {loss_str: loss}, batch_size, step_losses

