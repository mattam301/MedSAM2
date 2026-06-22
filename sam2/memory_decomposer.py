"""Memory decomposition strategies for bidirectional inference.

Two approaches:
  1. Heuristic: deterministic similarity-based filtering, no parameters.
  2. Learned: trainable residual corrections on top of the heuristic baseline.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


def _as_feature_tensor(features: torch.Tensor | list) -> torch.Tensor:
    """Return the last feature tensor when SAM2 passes a feature pyramid."""
    if isinstance(features, (list, tuple)):
        return features[-1]
    return features


def _pool_current_features(current_vision_feats: torch.Tensor) -> torch.Tensor:
    """Pool SAM2 current-frame features to a feature matrix with channel last."""
    if current_vision_feats.dim() == 4:
        # Common CNN layout: (B, C, H, W) -> (B, C)
        return current_vision_feats.mean(dim=[2, 3])
    if current_vision_feats.dim() == 3:
        # Common SAM2 layout: (HW, B, C) -> (B, C)
        return current_vision_feats.mean(dim=0)
    if current_vision_feats.dim() == 1:
        return current_vision_feats.unsqueeze(0)
    return current_vision_feats


def _match_current_to_memory(
    current_vec: torch.Tensor,
    memory_avg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align current and memory pooled features for pairwise operations."""
    if current_vec.dim() == 1:
        current_vec = current_vec.unsqueeze(0)
    if memory_avg.dim() == 1:
        memory_avg = memory_avg.unsqueeze(0)

    if current_vec.size(-1) != memory_avg.size(-1):
        min_dim = min(current_vec.size(-1), memory_avg.size(-1))
        current_vec = current_vec[..., :min_dim]
        memory_avg = memory_avg[..., :min_dim]

    if current_vec.size(0) != memory_avg.size(0):
        current_vec = current_vec.mean(dim=0, keepdim=True)
    current_vec = current_vec.expand_as(memory_avg)
    return current_vec, memory_avg


class MemoryDecomposer(ABC, nn.Module):
    """Abstract base for memory decomposition strategies."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def decompose(
        self,
        current_vision_feats: torch.Tensor | list,
        memory_feats: torch.Tensor,
    ) -> dict:
        """Decompose memory into redundant/unique components.

        Args:
            current_vision_feats: Current frame features (tensor or list).
            memory_feats: Memory frame features, shape (B, C, H, W).

        Returns:
            dict with keys:
              - 'redundancy_score': float in [0, 1]
              - 'unique_residual': torch.Tensor (same shape as memory_feats)
              - 'redundant_component': torch.Tensor (same shape as memory_feats)
        """
        pass


class HeuristicMemoryDecomposer(MemoryDecomposer):
    """Heuristic decomposition using feature similarity and projection."""

    def __init__(self):
        super().__init__()

    def decompose(
        self,
        current_vision_feats: torch.Tensor | list,
        memory_feats: torch.Tensor,
    ) -> dict:
        """Decompose via cosine similarity and vector projection.

        No learnable parameters.
        """
        current_vision_feats = _as_feature_tensor(current_vision_feats)
        current_vision_feats = current_vision_feats.to(
            device=memory_feats.device,
            dtype=memory_feats.dtype,
        )

        current_vec = _pool_current_features(current_vision_feats)
        memory_avg = memory_feats.mean(dim=[2, 3])
        current_vec, memory_avg = _match_current_to_memory(current_vec, memory_avg)

        # Redundancy score: cosine similarity
        similarity = F.cosine_similarity(current_vec, memory_avg, dim=-1, eps=1e-6)
        redundancy_score = float(similarity.mean().clamp(-1.0, 1.0).item())

        # Unique residual: subtract shared component
        curr_norm_sq = (current_vec * current_vec).sum(dim=-1, keepdim=True).clamp(min=1e-6)
        scale = (memory_avg * current_vec).sum(dim=-1, keepdim=True) / curr_norm_sq
        shared = scale.unsqueeze(-1).unsqueeze(-1) * current_vec.unsqueeze(-1).unsqueeze(-1)
        unique_residual = memory_feats - shared

        return {
            "redundancy_score": redundancy_score,
            "unique_residual": unique_residual,
            "redundant_component": shared.expand_as(memory_feats),
        }


class LearnedMemoryDecomposer(MemoryDecomposer):
    """Trainable residual decomposer with a deterministic safe baseline.

    The decomposer is often used at evaluation time without a separate trained
    checkpoint. Randomly initialized networks should not rewrite SAM2 memories,
    so this class starts from the heuristic decomposition and learns bounded
    residual corrections. With the default zero initialization, it is equivalent
    to the heuristic decomposer until its weights are trained or loaded.
    """

    def __init__(
        self,
        memory_feat_dim: int = 64,
        hidden_dim: int = 128,
        use_augmentation: bool = False,
        augmentation_scale: float = 0.1,
        max_score_delta: float = 0.10,
        residual_scale: float = 0.10,
    ):
        """Initialize learned decomposer.

        Args:
            memory_feat_dim: Dimension of memory features (e.g., 64).
            hidden_dim: Hidden dimension for MLP (default 128).
            use_augmentation: Whether to apply perturbation to emphasize uniqueness.
            augmentation_scale: Scale of random perturbation (0.0 to 1.0).
        """
        super().__init__()
        self.memory_feat_dim = memory_feat_dim
        self.hidden_dim = hidden_dim
        self.use_augmentation = use_augmentation
        self.augmentation_scale = augmentation_scale
        self.max_score_delta = max_score_delta
        self.residual_scale = residual_scale
        self.baseline = HeuristicMemoryDecomposer()

        # Predicts a bounded correction to the heuristic redundancy score.
        self.redundancy_scorer = nn.Sequential(
            nn.Linear(memory_feat_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )

        # Predicts a small additive correction to the heuristic unique residual.
        self.unique_extractor = nn.Sequential(
            nn.Conv2d(memory_feat_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, memory_feat_dim, kernel_size=1),
            nn.Tanh(),
        )

        # Optional: learnable augmentation noise generator
        if use_augmentation:
            self.augmentation_generator = nn.Sequential(
                nn.Linear(memory_feat_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, memory_feat_dim),
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize learned residual heads as no-ops."""
        final_score = self.redundancy_scorer[-2]
        nn.init.zeros_(final_score.weight)
        nn.init.zeros_(final_score.bias)

        final_unique = self.unique_extractor[-2]
        nn.init.zeros_(final_unique.weight)
        nn.init.zeros_(final_unique.bias)
        if self.use_augmentation:
            final_aug = self.augmentation_generator[-1]
            nn.init.zeros_(final_aug.weight)
            nn.init.zeros_(final_aug.bias)

    def decompose(
        self,
        current_vision_feats: torch.Tensor | list,
        memory_feats: torch.Tensor,
    ) -> dict:
        """Decompose using learned networks.

        Args:
            current_vision_feats: Current frame features (tensor or list).
            memory_feats: Memory frame features, shape (B, C, H, W).

        Returns:
            dict with redundancy score and decomposed components.
        """
        current_vision_feats = _as_feature_tensor(current_vision_feats)

        target_device = memory_feats.device
        target_dtype = memory_feats.dtype
        first_param = next(self.parameters(), None)
        if first_param is not None and (
            first_param.device != target_device or first_param.dtype != target_dtype
        ):
            self.to(device=target_device, dtype=target_dtype)

        current_vision_feats = current_vision_feats.to(
            device=target_device,
            dtype=target_dtype,
        )
        memory_feats = memory_feats.to(device=target_device, dtype=target_dtype)

        baseline = self.baseline.decompose(current_vision_feats, memory_feats)

        current_vec = _pool_current_features(current_vision_feats)
        memory_avg = memory_feats.mean(dim=[2, 3])  # (B, C)
        current_vec, memory_avg = _match_current_to_memory(current_vec, memory_avg)

        concat_features = torch.cat(
            [current_vec, memory_avg],
            dim=-1,
        )  # (B, C*2)

        score_delta = self.redundancy_scorer(concat_features).mean()
        redundancy_score = float(
            (
                torch.as_tensor(
                    baseline["redundancy_score"],
                    device=target_device,
                    dtype=target_dtype,
                )
                + self.max_score_delta * score_delta
            )
            .clamp(-1.0, 1.0)
            .item()
        )

        unique_delta = self.unique_extractor(memory_feats)
        unique_residual = baseline["unique_residual"] + (
            self.residual_scale * memory_feats * unique_delta
        )

        # Optional training-only augmentation. Disabled during eval.
        if self.use_augmentation and self.training:
            noise = self.augmentation_generator(current_vec.mean(dim=0, keepdim=True))
            noise = noise * self.augmentation_scale
            unique_residual = unique_residual + noise.unsqueeze(-1).unsqueeze(-1)

        redundant_component = memory_feats - unique_residual

        return {
            "redundancy_score": redundancy_score,
            "unique_residual": unique_residual,
            "redundant_component": redundant_component,
        }


def create_memory_decomposer(
    decomposer_type: str = "heuristic",
    **kwargs,
) -> MemoryDecomposer:
    """Factory function to create a memory decomposer.

    Args:
        decomposer_type: "heuristic" or "learned".
        **kwargs: Additional arguments for LearnedMemoryDecomposer.

    Returns:
        MemoryDecomposer instance.
    """
    if decomposer_type == "heuristic":
        return HeuristicMemoryDecomposer()
    elif decomposer_type == "learned":
        return LearnedMemoryDecomposer(**kwargs)
    else:
        raise ValueError(f"Unknown decomposer type: {decomposer_type}")
