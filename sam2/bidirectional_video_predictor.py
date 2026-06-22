"""
Bidirectional Video Predictor for 3D Medical Image Segmentation.

All new architecture code lives in this single file.  Nothing in the original
MedSAM2 / SAM2 source tree is modified.

Design:

  Pass 1  –  Standard SAM2 causal forward sweep.
             Produces  M_i^(0) = SAM2_forward(I_i, {M_{i-k}^(0)})  for all i.

  Pass 2  –  Bidirectional refinement.
             For every non-conditioning slice i the MemoryAttention module is
             re-run with an augmented memory bank that contains *both* past and
             future memories from Pass 1:

               Mem_i = Attn(I_i, {M_{i-1}^(0), M_{i+1}^(0), …})

  Bootstrap (GeoSAM2-style, optional) –
             The prompt frame is prepended as a virtual frame 0 so that when
             the real prompt frame is processed the memory bank already holds
             one entry.  This eliminates the cold-start problem without any
             additional propagation pass.

             Concretely, init_state() receives an image tensor that has the
             prompt frame duplicated at position 0:

               images_boot = [prompt_frame, frame_0, frame_1, …, frame_N-1]

             The prompt is placed on virtual frame 1 (the first real frame).
             All frame indices yielded to the caller are shifted back by –1 so
             the caller never sees the virtual frame.

  Consistency Loss (training) –
             L_cons = || M_i - M_{i-1} ||_1  +  || M_i - M_{i+1} ||_1

  Memory Redundancy Scoring –
             Redundancy is scored using Pass 1 predicted masks (not features):
               redundancy = 1 - novel_pixels / memory_area
             High novelty → keep.  High redundancy → prune.

  Temporal positional encoding –
             Past and future frames at equal distance share the same tpos index
             (symmetric reuse, zero new parameters).
"""

import logging
import torch
import torch.nn.functional as F
from contextlib import contextmanager
from tqdm import tqdm

from sam2.sam2_video_predictor_npz import SAM2VideoPredictorNPZ
from sam2.modeling.sam2_utils import get_1d_sine_pe, select_closest_cond_frames
from sam2.build_sam import get_best_available_device, _load_checkpoint
from sam2.memory_decomposer import create_memory_decomposer


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Consistency loss
# ---------------------------------------------------------------------------

def slice_consistency_loss(
    pred_masks: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    """Identity-warp slice consistency regularisation.

    L_cons = mean_i( |M_i - M_{i-1}| + |M_i - M_{i+1}| )  for i = 1…N-2.

    Args:
        pred_masks: Raw logits [N, 1, H, W].
        weight:     Loss coefficient.

    Returns:
        Scalar tensor; 0 when N < 3.
    """
    if pred_masks.shape[0] < 3:
        return pred_masks.new_zeros(())
    probs    = torch.sigmoid(pred_masks)
    interior = probs[1:-1]
    loss     = (interior - probs[:-2]).abs() + (interior - probs[2:]).abs()
    return weight * loss.mean()


# ---------------------------------------------------------------------------
# Mask-based redundancy scoring
# ---------------------------------------------------------------------------

def _mask_based_redundancy_score(
    current_pred_masks: torch.Tensor,
    memory_pred_masks: torch.Tensor,
    threshold: float = 0.0,
    min_memory_area: int = 1,
) -> float:
    """Fraction of the memory mask already covered by the current mask.

    Returns a float in [0, 1].
      0 → perfectly novel   (memory predicts regions current does not)
      1 → perfectly redundant (memory adds nothing new)
    """
    cur_bin = (current_pred_masks > threshold).float()
    mem_bin = (memory_pred_masks  > threshold).float()
    mem_area = mem_bin.sum().item()
    if mem_area < min_memory_area:
        return 0.5   # neutral: frame predicts nothing
    novel     = (mem_bin * (1.0 - cur_bin)).sum().item()
    return float(1.0 - novel / mem_area)


def _boundary_information_score(memory_pred_masks: torch.Tensor) -> float:
    """Uncertainty + edge strength of a memory frame's predicted mask.

    Higher = more boundary content = more valuable to retain.
    """
    probs    = torch.sigmoid(memory_pred_masks.float())
    mean_unc = (1.0 - (2.0 * (probs - 0.5)).abs()).mean().item()
    edge_x   = (probs[..., :, 1:] - probs[..., :, :-1]).abs().mean().item()
    edge_y   = (probs[..., 1:, :] - probs[..., :-1, :]).abs().mean().item()
    return mean_unc + edge_x + edge_y


# ---------------------------------------------------------------------------
# Bidirectional predictor
# ---------------------------------------------------------------------------

class BidirectionalSAM2VideoPredictorNPZ(SAM2VideoPredictorNPZ):
    """SAM2VideoPredictorNPZ with bidirectional attention and GeoSAM2 bootstrap.

    New public entry-points:
      propagate_in_video_bidirectional()  – two-pass bidirectional inference.

    Bootstrap (GeoSAM2-style) is activated by passing bootstrap=True to
    propagate_in_video_bidirectional().  The caller must have initialised the
    inference state via init_state_with_bootstrap() instead of init_state()
    so that the image tensor already contains the duplicated prompt frame.

    Helper:
      prepare_bootstrap_images()  – prepend the prompt frame to an image tensor.
    """

    def __init__(
        self,
        *args,
        memory_redundancy_threshold: float = 1.0,
        memory_max_unique_context_frames: int | None = None,
        memory_unique_residual: bool = False,
        memory_decomposer_type: str = "heuristic",
        memory_decomposer_use_augmentation: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.memory_redundancy_threshold      = memory_redundancy_threshold
        self.memory_max_unique_context_frames = memory_max_unique_context_frames
        self.memory_unique_residual           = memory_unique_residual

        self._decomposer_type             = memory_decomposer_type
        self._decomposer_use_augmentation = memory_decomposer_use_augmentation
        self.memory_decomposer = create_memory_decomposer(
            decomposer_type=memory_decomposer_type,
            memory_feat_dim=getattr(self, "mem_dim", 64),
            hidden_dim=128,
            use_augmentation=memory_decomposer_use_augmentation,
            augmentation_scale=0.1,
        )

        # Set only inside _bidirectional_context(); never touch directly.
        self._use_bidirectional_memory: bool = False
        self._pass1_output_dict: dict | None = None

    # ------------------------------------------------------------------
    # Decomposer type property
    # ------------------------------------------------------------------

    @property
    def decomposer_type(self) -> str:
        return self._decomposer_type

    @decomposer_type.setter
    def decomposer_type(self, value: str) -> None:
        if value != self._decomposer_type:
            self._decomposer_type = value
            self.memory_decomposer = create_memory_decomposer(
                decomposer_type=value,
                memory_feat_dim=getattr(self, "mem_dim", 64),
                hidden_dim=128,
                use_augmentation=self._decomposer_use_augmentation,
                augmentation_scale=0.1,
            )
            logger.info("Switched to %s memory decomposer", value)

    # ------------------------------------------------------------------
    # Context manager: bidirectional state
    # ------------------------------------------------------------------

    @contextmanager
    def _bidirectional_context(self, pass1_output_dict: dict):
        assert not self._use_bidirectional_memory, (
            "BidirectionalSAM2VideoPredictorNPZ is not re-entrant."
        )
        self._use_bidirectional_memory = True
        self._pass1_output_dict        = pass1_output_dict
        try:
            yield
        finally:
            self._use_bidirectional_memory = False
            self._pass1_output_dict        = None

    # ------------------------------------------------------------------
    # GeoSAM2-style bootstrap helpers
    # ------------------------------------------------------------------

    @staticmethod
    def prepare_bootstrap_images(
        images: torch.Tensor,
        prompt_frame_idx: int,
    ) -> tuple[torch.Tensor, int]:
        """Prepend the prompt frame to the image tensor.

        The duplicated frame occupies index 0; every original frame shifts
        right by 1.  The adjusted prompt index is returned so callers can
        pass it straight to add_new_points_or_box / add_new_mask.

        Args:
            images:           Float tensor [N, C, H, W], normalised.
            prompt_frame_idx: Index of the original prompt slice.

        Returns:
            (images_boot, adjusted_prompt_idx)
              images_boot          – tensor [N+1, C, H, W]
              adjusted_prompt_idx  – prompt_frame_idx + 1
        """
        prompt_frame  = images[prompt_frame_idx].unsqueeze(0)   # [1, C, H, W]
        images_boot   = torch.cat([prompt_frame, images], dim=0)
        return images_boot, prompt_frame_idx + 1

    # ------------------------------------------------------------------
    # Override: memory-conditioned features
    # ------------------------------------------------------------------

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,
    ):
        if self._use_bidirectional_memory and self._pass1_output_dict is not None:
            return self._prepare_bidirectional_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                output_dict=output_dict,
                num_frames=num_frames,
            )
        return super()._prepare_memory_conditioned_features(
            frame_idx, is_init_cond_frame,
            current_vision_feats, current_vision_pos_embeds,
            feat_sizes, output_dict, num_frames, track_in_reverse,
        )

    # ------------------------------------------------------------------
    # Mask-based scoring
    # ------------------------------------------------------------------

    def _get_pass1_pred_masks(self, frame_idx: int) -> torch.Tensor | None:
        if self._pass1_output_dict is None:
            return None
        out = (
            self._pass1_output_dict["non_cond_frame_outputs"].get(frame_idx)
            or self._pass1_output_dict["cond_frame_outputs"].get(frame_idx)
        )
        return None if out is None else out.get("pred_masks")

    def _score_memory_entry(
        self, current_frame_idx: int, memory_frame_idx: int
    ) -> tuple[float, float]:
        """(redundancy, boundary_score) for a frame pair.

        Falls back to (0.5, 0.0) when Pass 1 masks are unavailable.
        """
        cur  = self._get_pass1_pred_masks(current_frame_idx)
        mem  = self._get_pass1_pred_masks(memory_frame_idx)
        if cur is None or mem is None:
            return 0.5, 0.0
        return (
            _mask_based_redundancy_score(cur, mem),
            _boundary_information_score(mem),
        )

    # ------------------------------------------------------------------
    # Memory pruning
    # ------------------------------------------------------------------

    def _prune_redundant_memory_entries(
        self,
        current_frame_idx: int,
        t_pos_and_prev: list[tuple],
    ) -> list[tuple]:
        """Keep the most novel memory entries (all must be 3-tuples)."""
        threshold = self.memory_redundancy_threshold
        max_keep  = self.memory_max_unique_context_frames

        if threshold >= 1.0 and max_keep is None:
            return t_pos_and_prev

        cond_entries = [e for e in t_pos_and_prev if e[0] == 0]
        candidates   = [e for e in t_pos_and_prev if e[0] != 0]

        scored: list[tuple[float, tuple]] = []
        for entry in candidates:
            t_pos, prev, is_future = entry
            if prev is None:
                continue
            mem_frame  = (
                current_frame_idx + t_pos if is_future
                else current_frame_idx - t_pos
            )
            red, bnd   = self._score_memory_entry(current_frame_idx, mem_frame)
            combined   = red - 0.2 * bnd   # lower = more valuable
            scored.append((combined, entry))

        scored.sort(key=lambda x: x[0])

        if max_keep is None:
            selected = [e for s, e in scored if s < threshold]
        else:
            preferred = [e for s, e in scored if s < threshold]
            backfill  = [e for s, e in scored if s >= threshold]
            selected  = (preferred + backfill)[:max_keep]

        selected.sort(key=lambda e: abs(e[0]))
        return cond_entries + selected

    # ------------------------------------------------------------------
    # Unique residual
    # ------------------------------------------------------------------

    def _compute_memory_unique_residual(
        self,
        current_vision_feats,
        memory_feats: torch.Tensor,
        alpha: float = 0.3,
    ) -> torch.Tensor:
        result  = self.memory_decomposer.decompose(current_vision_feats, memory_feats)
        unique  = result["unique_residual"]
        blended = (1.0 - alpha) * memory_feats + alpha * unique

        o_mean = memory_feats.mean(dim=(-2, -1), keepdim=True)
        o_std  = memory_feats.std(dim=(-2, -1),  keepdim=True).clamp(min=1e-6)
        b_mean = blended.mean(dim=(-2, -1), keepdim=True)
        b_std  = blended.std(dim=(-2, -1),  keepdim=True).clamp(min=1e-6)
        return (blended - b_mean) / b_std * o_std + o_mean

    # ------------------------------------------------------------------
    # Bidirectional memory assembly
    # ------------------------------------------------------------------

    def _prepare_bidirectional_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
    ):
        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        device = current_vision_feats[-1].device

        if self.num_maskmem == 0:
            return current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)

        to_cat_memory:          list[torch.Tensor] = []
        to_cat_memory_pos_embed: list[torch.Tensor] = []
        num_obj_ptr_tokens = 0
        pass1 = self._pass1_output_dict

        if not is_init_cond_frame:
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond, unselected_cond = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn
            )
            t_pos_and_prevs: list[tuple] = [
                (0, out, False) for out in selected_cond.values()
            ]

            stride = self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos
                prev_frame_idx = (
                    frame_idx - 1 if t_rel == 1
                    else ((frame_idx - 2) // stride) * stride
                         - (t_rel - 2) * stride
                )
                out = (
                    pass1["non_cond_frame_outputs"].get(prev_frame_idx)
                    or unselected_cond.get(prev_frame_idx)
                )
                t_pos_and_prevs.append((t_pos, out, False))

            future_entries: list[tuple] = []
            for k in range(1, self.num_maskmem):
                fidx = frame_idx + k
                if fidx >= num_frames:
                    break
                fut = (
                    pass1["non_cond_frame_outputs"].get(fidx)
                    or output_dict["cond_frame_outputs"].get(fidx)
                )
                if fut is not None:
                    future_entries.append((k, fut, True))

            t_pos_and_prevs = self._prune_redundant_memory_entries(
                frame_idx, t_pos_and_prevs
            )
            future_entries = self._prune_redundant_memory_entries(
                frame_idx, future_entries
            )

            for t_pos, prev, is_future in t_pos_and_prevs + future_entries:
                if prev is None:
                    continue
                feats = prev["maskmem_features"].to(device, non_blocking=True)
                if self.memory_unique_residual:
                    feats = self._compute_memory_unique_residual(
                        current_vision_feats, feats
                    )
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))

                enc = prev["maskmem_pos_enc"][-1].to(device)
                enc = enc.flatten(2).permute(2, 0, 1)
                enc = enc + self.maskmem_tpos_enc[t_pos - 1]  # symmetric
                to_cat_memory_pos_embed.append(enc)

            if self.use_obj_ptrs_in_encoder:
                max_ptrs  = min(num_frames, self.max_obj_ptrs_in_encoder)
                only_past = getattr(self, "only_obj_ptrs_in_the_past_for_eval", True)
                ptr_cond  = (
                    {t: o for t, o in selected_cond.items() if t <= frame_idx}
                    if (not self.training and only_past) else selected_cond
                )
                pos_and_ptrs = [
                    (abs(frame_idx - t), o["obj_ptr"])
                    for t, o in ptr_cond.items()
                ]
                for t_diff in range(1, max_ptrs):
                    t = frame_idx - t_diff
                    if t < 0:
                        break
                    out = (
                        pass1["non_cond_frame_outputs"].get(t)
                        or unselected_cond.get(t)
                    )
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))

                if pos_and_ptrs:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    obj_ptrs = torch.stack(ptrs_list, dim=0)

                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_ptrs - 1
                        proj_tpos  = getattr(self, "proj_tpos_enc_in_obj_ptrs", False)
                        tpos_dim   = C if proj_tpos else self.mem_dim
                        obj_pos    = get_1d_sine_pe(
                            torch.tensor(pos_list, device=device) / t_diff_max,
                            dim=tpos_dim,
                        )
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(
                            len(pos_list), B, self.mem_dim
                        )

                    if self.mem_dim < C:
                        obj_ptrs = obj_ptrs.reshape(
                            -1, B, C // self.mem_dim, self.mem_dim
                        ).permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)

                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]

        else:
            if self.directly_add_no_mem_embed:
                pix = current_vision_feats[-1] + self.no_mem_embed
                return pix.permute(1, 2, 0).view(B, C, H, W)
            to_cat_memory           = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        memory           = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        pix = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        return pix.permute(1, 2, 0).view(B, C, H, W)

    # ------------------------------------------------------------------
    # Two-pass bidirectional propagation
    # ------------------------------------------------------------------

    def propagate_in_video_bidirectional(
        self,
        inference_state,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        bootstrap_frame_offset: int = 0,
    ):
        """Two-pass bidirectional propagation.

        Args:
            inference_state:       Live inference state.
            start_frame_idx:       First frame to process (default: earliest
                                   conditioning frame).
            max_frame_num_to_track: Maximum frames to process.
            bootstrap_frame_offset: Set to 1 when the image tensor was
                                    prepared with prepare_bootstrap_images().
                                    The virtual duplicate frame at index 0 is
                                    processed in Pass 1 to pre-populate the
                                    memory bank but is skipped in Pass 2 and
                                    never yielded to the caller.

        Yields:
            (frame_idx, obj_ids, video_res_masks)
            frame_idx is always in the original (non-bootstrapped) space.
        """
        self.propagate_in_video_preflight(inference_state)

        output_dict             = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        obj_ids                 = inference_state["obj_ids"]
        num_frames              = inference_state["num_frames"]
        batch_size              = self._get_obj_num(inference_state)

        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No prompts found. Add points or a mask first.")

        if start_frame_idx is None:
            # In bootstrapped mode the earliest cond frame is the duplicated
            # prompt at index bootstrap_frame_offset; start from frame 0 of
            # the *internal* sequence so the virtual frame gets processed.
            start_frame_idx = 0
        if max_frame_num_to_track is None:
            max_frame_num_to_track = num_frames

        end_frame_idx    = min(
            start_frame_idx + max_frame_num_to_track, num_frames - 1
        )
        processing_order = range(start_frame_idx, end_frame_idx + 1)

        # ── PASS 1: causal forward sweep ─────────────────────────────────
        pass1_non_cond: dict = {}

        for frame_idx in tqdm(processing_order, desc="Bidir Pass 1/2 (forward)"):
            if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
                storage_key = "cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks  = current_out["pred_masks"]
            elif frame_idx in consolidated_frame_inds["non_cond_frame_outputs"]:
                storage_key = "non_cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks  = current_out["pred_masks"]
            else:
                storage_key = "non_cond_frame_outputs"
                current_out, pred_masks = self._run_single_frame_inference(
                    inference_state=inference_state,
                    output_dict=output_dict,
                    frame_idx=frame_idx,
                    batch_size=batch_size,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=False,
                    run_mem_encoder=True,
                )
                output_dict[storage_key][frame_idx] = current_out

            if storage_key == "non_cond_frame_outputs":
                pass1_non_cond[frame_idx] = current_out

            self._add_output_per_object(
                inference_state, frame_idx, current_out, storage_key
            )
            inference_state["frames_already_tracked"][frame_idx] = {
                "reverse": False
            }

        pass1_output_dict = {
            "cond_frame_outputs":     dict(output_dict["cond_frame_outputs"]),
            "non_cond_frame_outputs": pass1_non_cond,
        }

        # ── PASS 2: bidirectional refinement ─────────────────────────────
        with self._bidirectional_context(pass1_output_dict):
            for frame_idx in tqdm(
                processing_order, desc="Bidir Pass 2/2 (refinement)"
            ):
                # Skip the virtual bootstrap frame; never yield it.
                if frame_idx < bootstrap_frame_offset:
                    continue

                if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
                    current_out = output_dict["cond_frame_outputs"][frame_idx]
                    pred_masks  = current_out["pred_masks"]
                    storage_key = "cond_frame_outputs"
                else:
                    storage_key = "non_cond_frame_outputs"
                    inference_state["frames_already_tracked"].pop(frame_idx, None)
                    output_dict["non_cond_frame_outputs"].pop(frame_idx, None)

                    current_out, pred_masks = self._run_single_frame_inference(
                        inference_state=inference_state,
                        output_dict=output_dict,
                        frame_idx=frame_idx,
                        batch_size=batch_size,
                        is_init_cond_frame=False,
                        point_inputs=None,
                        mask_inputs=None,
                        reverse=False,
                        run_mem_encoder=True,
                    )
                    output_dict[storage_key][frame_idx] = current_out

                self._add_output_per_object(
                    inference_state, frame_idx, current_out, storage_key
                )
                inference_state["frames_already_tracked"][frame_idx] = {
                    "reverse": False
                }

                _, video_res_masks = self._get_orig_video_res_output(
                    inference_state, pred_masks
                )
                # Shift index back to original space for the caller
                yield frame_idx - bootstrap_frame_offset, obj_ids, video_res_masks


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_bidir_sam2_video_predictor_npz(
    config_file: str,
    ckpt_path: str | None = None,
    device: str | None = None,
    mode: str = "eval",
    hydra_overrides_extra: list[str] | None = None,
    apply_postprocessing: bool = True,
    **kwargs,
) -> BidirectionalSAM2VideoPredictorNPZ:
    """Build a BidirectionalSAM2VideoPredictorNPZ from a config + checkpoint."""
    from hydra import compose
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    if hydra_overrides_extra is None:
        hydra_overrides_extra = []

    device = device or get_best_available_device()
    logger.info("build_bidir_sam2_video_predictor_npz: device=%s", device)

    hydra_overrides = [
        "++model._target_="
        "sam2.bidirectional_video_predictor.BidirectionalSAM2VideoPredictorNPZ",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = list(hydra_overrides_extra) + [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model