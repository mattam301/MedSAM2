from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resize_frames(frames: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(frames.shape[-2:]) == size:
        return frames
    return F.interpolate(
        frames[:, None],
        size=size,
        mode="bilinear",
        align_corners=False,
    )[:, 0]


def project_axis_probs(
    logits: torch.Tensor,
    axis: str,
    start: int,
    volume_shape: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project an axis clip of logits into a full 3D probability volume.

    Args:
        logits: Tensor shaped [F, H, W] for one axis clip.
        axis: One of z, y, x.
        start: Clip start index in that axis.
        volume_shape: Original volume shape as (D, H, W).
    """
    d, h, w = volume_shape
    probs = torch.sigmoid(logits.float())
    volume = probs.new_zeros((d, h, w))
    valid = torch.zeros((d, h, w), dtype=torch.bool, device=probs.device)
    frames = min(probs.shape[0], {"z": d, "y": h, "x": w}[axis] - start)
    if frames <= 0:
        return volume, valid

    probs = probs[:frames]
    if axis == "z":
        probs = _resize_frames(probs, (h, w))
        volume[start:start + frames] = probs
        valid[start:start + frames] = True
    elif axis == "y":
        probs = _resize_frames(probs, (d, w))
        volume[:, start:start + frames, :] = probs.permute(1, 0, 2)
        valid[:, start:start + frames, :] = True
    elif axis == "x":
        probs = _resize_frames(probs, (d, h))
        volume[:, :, start:start + frames] = probs.permute(1, 2, 0)
        valid[:, :, start:start + frames] = True
    else:
        raise ValueError(f"Unknown axis: {axis}")
    return volume, valid


class CrossAxisConsistencyLoss(nn.Module):
    """Consistency regularizer for paired Z/Y/X predictions from one volume."""

    def __init__(
        self,
        weight: float = 1.0,
        confidence_threshold: float = 0.15,
        min_overlap_voxels: int = 1,
    ):
        super().__init__()
        self.weight = float(weight)
        self.confidence_threshold = float(confidence_threshold)
        self.min_overlap_voxels = int(min_overlap_voxels)

    def forward(
        self,
        logits_by_axis: Mapping[str, torch.Tensor],
        starts_by_axis: Mapping[str, int | torch.Tensor],
        volume_shape: tuple[int, int, int] | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(volume_shape, torch.Tensor):
            volume_shape = tuple(int(x) for x in volume_shape.detach().cpu().tolist())

        projected = {}
        valid_masks = {}
        for axis, logits in logits_by_axis.items():
            start = starts_by_axis[axis]
            if isinstance(start, torch.Tensor):
                start = int(start.detach().cpu().item())
            projected[axis], valid_masks[axis] = project_axis_probs(
                logits=logits,
                axis=axis,
                start=int(start),
                volume_shape=volume_shape,
            )

        axes = [axis for axis in ("z", "y", "x") if axis in projected]
        if len(axes) < 2:
            first = next(iter(logits_by_axis.values()))
            return first.new_tensor(0.0)

        losses = []
        for left_idx, left in enumerate(axes):
            for right in axes[left_idx + 1:]:
                overlap = valid_masks[left] & valid_masks[right]
                confident = (
                    (projected[left] - 0.5).abs() > self.confidence_threshold
                ) | (
                    (projected[right] - 0.5).abs() > self.confidence_threshold
                )
                mask = overlap & confident
                if int(mask.sum().detach().cpu().item()) < self.min_overlap_voxels:
                    continue
                losses.append(F.mse_loss(projected[left][mask], projected[right][mask]))

        if not losses:
            first = next(iter(logits_by_axis.values()))
            return first.new_tensor(0.0)
        return torch.stack(losses).mean() * self.weight

