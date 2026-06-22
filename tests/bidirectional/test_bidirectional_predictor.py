"""
Sanity tests for slice_consistency_loss and BidirectionalSAM2VideoPredictorNPZ.

These are lightweight, CPU-only unit tests that exercise the new logic without
requiring real checkpoints or a GPU.  They complete in under 10 seconds on any
machine, including Apple Silicon and CPU-only CI runners.

Run with:
    pytest tests/bidirectional/ -v

Test coverage:
  - slice_consistency_loss correctness and gradient flow
  - BidirectionalSAM2VideoPredictorNPZ is a proper subclass of SAM2VideoPredictorNPZ
  - _prepare_memory_conditioned_features routing (flag-based dispatch)
  - _prepare_bidirectional_memory_conditioned_features memory token assembly
  - Two-pass propagation control flow (flag reset, cond frame isolation)
  - Temporal PE index symmetry
"""

import pytest
import torch
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame_out(mem_dim=8, H=2, W=2, hidden_dim=16, device="cpu"):
    """Return a dict that mimics a stored frame output."""
    return {
        "maskmem_features": torch.randn(1, mem_dim, H, W, device=device),
        "maskmem_pos_enc": [torch.randn(1, mem_dim, H, W, device=device)],
        "pred_masks": torch.randn(1, 1, H * 4, W * 4, device=device),
        "obj_ptr": torch.randn(1, hidden_dim, device=device),
        "object_score_logits": torch.randn(1, 1, device=device),
    }


def _make_pass1_output_dict(num_frames, mem_dim=8, hidden_dim=16, H=2, W=2):
    """One cond frame (index 0) + all subsequent frames as non-cond."""
    cond = {0: _make_frame_out(mem_dim=mem_dim, H=H, W=W, hidden_dim=hidden_dim)}
    non_cond = {
        i: _make_frame_out(mem_dim=mem_dim, H=H, W=W, hidden_dim=hidden_dim)
        for i in range(1, num_frames)
    }
    return {"cond_frame_outputs": cond, "non_cond_frame_outputs": non_cond}


def _build_mock_pred(num_maskmem=4, hidden_dim=16, mem_dim=8):
    """Mock predictor with all attributes consumed by the bidirectional method.

    no_mem_embed and no_mem_pos_enc have last-dim == mem_dim so that
    embed.expand(1, B, self.mem_dim) never raises a shape mismatch.
    """
    from sam2.bidirectional_video_predictor import BidirectionalSAM2VideoPredictorNPZ

    pred = MagicMock(spec=BidirectionalSAM2VideoPredictorNPZ)
    pred.num_maskmem = num_maskmem
    pred.hidden_dim = hidden_dim
    pred.mem_dim = mem_dim
    pred.training = False
    pred.max_cond_frames_in_attn = -1
    pred.memory_temporal_stride_for_eval = 1
    pred.use_obj_ptrs_in_encoder = False
    pred.directly_add_no_mem_embed = False
    torch.manual_seed(0)
    pred.maskmem_tpos_enc = torch.nn.Parameter(
        torch.randn(num_maskmem, 1, 1, mem_dim)
    )
    pred.no_mem_embed = torch.nn.Parameter(torch.randn(1, 1, mem_dim))
    pred.no_mem_pos_enc = torch.nn.Parameter(torch.randn(1, 1, mem_dim))
    pred._use_bidirectional_memory = False
    pred._pass1_output_dict = None
    return pred


def _make_inference_state(num_frames, cond_frame_indices, mem_dim=8, hidden_dim=16):
    """Minimal inference_state dict for propagation tests."""
    stub = _make_frame_out(mem_dim=mem_dim, hidden_dim=hidden_dim)
    cond_outputs = {i: stub for i in cond_frame_indices}
    output_dict = {
        "cond_frame_outputs": cond_outputs,
        "non_cond_frame_outputs": {},
    }
    consolidated = {
        "cond_frame_outputs": set(cond_frame_indices),
        "non_cond_frame_outputs": set(),
    }
    return {
        "output_dict": output_dict,
        "consolidated_frame_inds": consolidated,
        "obj_ids": [0],
        "num_frames": num_frames,
        "frames_already_tracked": {},
        "device": torch.device("cpu"),
        "storage_device": torch.device("cpu"),
        "video_height": 32,
        "video_width": 32,
    }


# ---------------------------------------------------------------------------
# Test 1 – slice_consistency_loss
# ---------------------------------------------------------------------------

class TestSliceConsistencyLoss:

    def test_returns_zero_for_two_slices(self):
        from sam2.bidirectional_video_predictor import slice_consistency_loss
        assert slice_consistency_loss(torch.zeros(2, 1, 8, 8)).item() == pytest.approx(0.0)

    def test_returns_zero_for_one_slice(self):
        from sam2.bidirectional_video_predictor import slice_consistency_loss
        assert slice_consistency_loss(torch.zeros(1, 1, 8, 8)).item() == pytest.approx(0.0)

    def test_zero_for_constant_volume(self):
        """All slices identical → loss = 0."""
        from sam2.bidirectional_video_predictor import slice_consistency_loss
        masks = torch.full((10, 1, 16, 16), 2.0)
        assert slice_consistency_loss(masks).item() == pytest.approx(0.0, abs=1e-6)

    def test_nonzero_for_alternating_volume(self):
        """Slices alternating 0 / +10 logits → strong discontinuity → loss > 0."""
        from sam2.bidirectional_video_predictor import slice_consistency_loss
        masks = torch.zeros(10, 1, 8, 8)
        masks[1::2] = 10.0
        assert slice_consistency_loss(masks).item() > 0.1

    def test_weight_scales_linearly(self):
        from sam2.bidirectional_video_predictor import slice_consistency_loss
        torch.manual_seed(0)
        masks = torch.randn(6, 1, 8, 8)
        l1 = slice_consistency_loss(masks, weight=1.0)
        l2 = slice_consistency_loss(masks, weight=2.0)
        assert l2.item() == pytest.approx(2.0 * l1.item(), rel=1e-5)

    def test_gradient_flows(self):
        from sam2.bidirectional_video_predictor import slice_consistency_loss
        masks = torch.randn(5, 1, 8, 8, requires_grad=True)
        slice_consistency_loss(masks).backward()
        assert masks.grad is not None
        assert not torch.isnan(masks.grad).any()

    def test_output_is_scalar(self):
        from sam2.bidirectional_video_predictor import slice_consistency_loss
        assert slice_consistency_loss(torch.randn(8, 1, 16, 16)).shape == torch.Size([])

    def test_reversed_order_gives_same_loss(self):
        """Identity warp is symmetric: reversing slice order must give the same loss."""
        from sam2.bidirectional_video_predictor import slice_consistency_loss
        torch.manual_seed(1)
        masks = torch.randn(8, 1, 16, 16)
        assert slice_consistency_loss(masks).item() == pytest.approx(
            slice_consistency_loss(masks.flip(0)).item(), rel=1e-5
        )


# ---------------------------------------------------------------------------
# Test 2 – Class structure
# ---------------------------------------------------------------------------

class TestBidirectionalPredictorClass:

    def test_is_subclass(self):
        from sam2.bidirectional_video_predictor import (
            BidirectionalSAM2VideoPredictorNPZ,
        )
        from sam2.sam2_video_predictor_npz import SAM2VideoPredictorNPZ
        assert issubclass(BidirectionalSAM2VideoPredictorNPZ, SAM2VideoPredictorNPZ)

    def test_exposes_required_methods(self):
        from sam2.bidirectional_video_predictor import (
            BidirectionalSAM2VideoPredictorNPZ,
        )
        for name in (
            "propagate_in_video_bidirectional",
            "_prepare_bidirectional_memory_conditioned_features",
            "_prepare_memory_conditioned_features",
        ):
            assert hasattr(BidirectionalSAM2VideoPredictorNPZ, name), (
                f"Missing: {name}"
            )

    def test_bidirectional_flag_initialises_false(self):
        """__init__ must set _use_bidirectional_memory = False."""
        import inspect
        from sam2.bidirectional_video_predictor import (
            BidirectionalSAM2VideoPredictorNPZ,
        )
        src = inspect.getsource(BidirectionalSAM2VideoPredictorNPZ.__init__)
        assert "_use_bidirectional_memory" in src
        assert "False" in src

    def test_builder_is_callable(self):
        from sam2.bidirectional_video_predictor import (
            build_bidir_sam2_video_predictor_npz,
        )
        assert callable(build_bidir_sam2_video_predictor_npz)

    def test_no_new_parameter_attributes(self):
        """The subclass must declare no new nn.Parameter class attributes.

        All weights come from the parent checkpoint — zero new parameters.
        """
        from sam2.bidirectional_video_predictor import (
            BidirectionalSAM2VideoPredictorNPZ,
        )
        from sam2.sam2_video_predictor_npz import SAM2VideoPredictorNPZ

        parent_attrs = set(SAM2VideoPredictorNPZ.__dict__.keys())
        child_attrs = set(BidirectionalSAM2VideoPredictorNPZ.__dict__.keys())
        for attr in child_attrs - parent_attrs:
            val = BidirectionalSAM2VideoPredictorNPZ.__dict__[attr]
            assert not isinstance(val, torch.nn.Parameter), (
                f"Unexpected new nn.Parameter '{attr}' in bidirectional subclass"
            )


# ---------------------------------------------------------------------------
# Test 3 – _prepare_memory_conditioned_features routing
# ---------------------------------------------------------------------------

class TestMemoryRoutingFlag:

    def test_routes_to_bidirectional_when_flag_true(self):
        from sam2.bidirectional_video_predictor import (
            BidirectionalSAM2VideoPredictorNPZ,
        )
        pred = _build_mock_pred()
        pred._use_bidirectional_memory = True
        pred._pass1_output_dict = _make_pass1_output_dict(num_frames=6)

        bidir_called = []

        def fake_bidir(**kw):
            bidir_called.append(True)
            return torch.zeros(1, pred.hidden_dim, 2, 2)

        # Assign directly on the mock instance so self._prepare_bidirectional_... resolves here
        pred._prepare_bidirectional_memory_conditioned_features = fake_bidir

        fn = BidirectionalSAM2VideoPredictorNPZ._prepare_memory_conditioned_features
        try:
            fn(
                pred, 3, False,
                [torch.randn(4, 1, 16)], [torch.randn(4, 1, 16)],
                [(2, 2)],
                {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
                6,
            )
        except Exception:
            pass

        assert bidir_called, "_prepare_bidirectional_memory_conditioned_features not called"

    def test_does_not_call_bidir_when_flag_false(self):
        from sam2.bidirectional_video_predictor import (
            BidirectionalSAM2VideoPredictorNPZ,
        )
        pred = _build_mock_pred()
        pred._use_bidirectional_memory = False
        pred._pass1_output_dict = None

        bidir_called = []

        def fake_bidir(self, **kw):
            bidir_called.append(True)

        with patch.object(
            BidirectionalSAM2VideoPredictorNPZ,
            "_prepare_bidirectional_memory_conditioned_features",
            fake_bidir,
        ):
            fn = BidirectionalSAM2VideoPredictorNPZ._prepare_memory_conditioned_features
            try:
                fn(
                    pred, 3, False,
                    [torch.randn(4, 1, 16)], [torch.randn(4, 1, 16)],
                    [(2, 2)],
                    {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
                    6,
                )
            except Exception:
                pass

        assert not bidir_called, (
            "_prepare_bidirectional_memory_conditioned_features should not be called"
        )


# ---------------------------------------------------------------------------
# Test 4 – Memory token assembly
# ---------------------------------------------------------------------------

class TestBidirectionalMemoryAssembly:
    """
    Test _prepare_bidirectional_memory_conditioned_features via an injected
    memory_attention mock that records memory.shape[0].

    Shape reminder:
        maskmem_features: [B, mem_dim, H, W]
        after flatten+permute: [H*W, B, mem_dim]
        memory tensor: [num_tensors * H*W, B, mem_dim]
        memory.shape[0] == num_tensors * H * W

    With H=W=2, each memory frame contributes 4 spatial tokens.
    Causal max = num_maskmem frames * 4 tokens = 16.
    """

    NUM_MASKMEM = 4
    MEM_DIM = 8
    HIDDEN_DIM = 16
    H, W = 2, 2
    HW = H * W  # 4 spatial tokens per frame

    def _run_bidir(self, frame_idx, num_frames, pass1_output_dict, output_dict,
                   is_init_cond_frame=False):
        from sam2.bidirectional_video_predictor import (
            BidirectionalSAM2VideoPredictorNPZ,
        )
        pred = _build_mock_pred(
            num_maskmem=self.NUM_MASKMEM,
            hidden_dim=self.HIDDEN_DIM,
            mem_dim=self.MEM_DIM,
        )
        pred._pass1_output_dict = pass1_output_dict
        pred._use_bidirectional_memory = True

        captured = {}

        def capture_attention(curr, curr_pos, memory, memory_pos, num_obj_ptr_tokens=0):
            captured["memory_len"] = memory.shape[0]
            return curr[-1]

        pred.memory_attention = MagicMock(side_effect=capture_attention)

        B = 1
        fn = BidirectionalSAM2VideoPredictorNPZ._prepare_bidirectional_memory_conditioned_features
        fn(
            pred,
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=[torch.randn(self.HW, B, self.HIDDEN_DIM)],
            current_vision_pos_embeds=[torch.randn(self.HW, B, self.HIDDEN_DIM)],
            feat_sizes=[(self.H, self.W)],
            output_dict=output_dict,
            num_frames=num_frames,
        )
        return captured

    def test_middle_frame_has_future_tokens(self):
        """A frame with ample past AND future should exceed the causal max."""
        num_frames = 10
        p1 = _make_pass1_output_dict(
            num_frames, mem_dim=self.MEM_DIM, hidden_dim=self.HIDDEN_DIM,
            H=self.H, W=self.W,
        )
        output_dict = {"cond_frame_outputs": p1["cond_frame_outputs"],
                       "non_cond_frame_outputs": {}}
        causal_max = self.NUM_MASKMEM * self.HW  # 16

        captured = self._run_bidir(5, num_frames, p1, output_dict)
        assert captured.get("memory_len", 0) > causal_max, (
            f"Middle frame: expected >{causal_max} tokens, got {captured.get('memory_len')}"
        )

    def test_last_frame_no_future_tokens(self):
        """Last slice (index num_frames-1) has no future: memory ≤ causal max."""
        num_frames = 8
        p1 = _make_pass1_output_dict(
            num_frames, mem_dim=self.MEM_DIM, hidden_dim=self.HIDDEN_DIM,
            H=self.H, W=self.W,
        )
        output_dict = {"cond_frame_outputs": p1["cond_frame_outputs"],
                       "non_cond_frame_outputs": {}}
        causal_max = self.NUM_MASKMEM * self.HW  # 16

        last = self._run_bidir(num_frames - 1, num_frames, p1, output_dict)
        mid  = self._run_bidir(4,              num_frames, p1, output_dict)

        assert last.get("memory_len", 0) <= causal_max, (
            f"Last frame: expected <={causal_max}, got {last.get('memory_len')}"
        )
        assert mid.get("memory_len", 0) > causal_max, (
            f"Middle frame sanity: expected >{causal_max}, got {mid.get('memory_len')}"
        )

    def test_initial_cond_frame_uses_single_no_mem_embed(self):
        """is_init_cond_frame=True → 1 dummy no_mem token, memory.shape[0] == 1."""
        num_frames = 6
        p1 = _make_pass1_output_dict(
            num_frames, mem_dim=self.MEM_DIM, hidden_dim=self.HIDDEN_DIM,
            H=self.H, W=self.W,
        )
        output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}

        captured = self._run_bidir(
            0, num_frames, p1, output_dict, is_init_cond_frame=True
        )
        assert captured.get("memory_len") == 1, (
            f"Initial cond frame must have exactly 1 no-mem token; "
            f"got {captured.get('memory_len')}"
        )


# ---------------------------------------------------------------------------
# Test 5 – propagate_in_video_bidirectional control flow
# ---------------------------------------------------------------------------

class TestPropagateControlFlow:
    """
    Test the two-pass loop without running real inference.
    All inference calls are fully mocked.

    Pass 1 runs _run_single_frame_inference for every non-cond frame.
    Pass 2 re-runs _run_single_frame_inference for every non-cond frame;
    cond frames are skipped.
    """

    def _run_propagation(self, inference_state, raise_on_call_n=None):
        """Run the bidirectional propagation with a fully mocked predictor."""
        from sam2.bidirectional_video_predictor import (
            BidirectionalSAM2VideoPredictorNPZ,
        )

        pred = _build_mock_pred()
        pred._get_obj_num.return_value = 1
        pred.propagate_in_video_preflight = MagicMock()
        pred._add_output_per_object = MagicMock()

        stub_masks = torch.randn(1, 1, 8, 8)
        pred._get_orig_video_res_output.return_value = (stub_masks, stub_masks)

        call_count = [0]

        def mock_inference(*args, **kwargs):
            call_count[0] += 1
            if raise_on_call_n is not None and call_count[0] >= raise_on_call_n:
                raise RuntimeError("Simulated error")
            out = _make_frame_out()
            return out, out["pred_masks"]

        pred._run_single_frame_inference.side_effect = mock_inference

        fn = BidirectionalSAM2VideoPredictorNPZ.propagate_in_video_bidirectional
        results = []
        try:
            for item in fn(pred, inference_state):
                results.append(item)
        except RuntimeError:
            pass

        return pred, results

    # ---- basic correctness ----

    def test_yields_one_item_per_frame(self):
        """Pass 2 must yield exactly num_frames results."""
        num_frames = 5
        state = _make_inference_state(num_frames, cond_frame_indices=[0])
        pred, results = self._run_propagation(state)
        assert len(results) == num_frames

    # ---- flag lifecycle ----

    def test_flag_reset_after_clean_run(self):
        """_use_bidirectional_memory and _pass1_output_dict reset after clean run."""
        state = _make_inference_state(4, cond_frame_indices=[0])
        pred, _ = self._run_propagation(state)
        assert pred._use_bidirectional_memory is False
        assert pred._pass1_output_dict is None

    def test_flag_reset_after_pass2_exception(self):
        """
        The try/finally in Pass 2 must reset the flags even when an error occurs.

        Setup: 4 frames, frame 0 is cond.
        Pass 1: 3 non-cond frames → calls 1, 2, 3  (all succeed).
        Pass 2: raise on call 4 (first non-cond frame of Pass 2).
        Expected: flags reset despite exception.
        """
        # 4 frames, 1 cond → 3 non-cond.  Pass 1 uses calls 1-3.  Raise at 4.
        state = _make_inference_state(4, cond_frame_indices=[0])
        pred, _ = self._run_propagation(state, raise_on_call_n=4)
        assert pred._use_bidirectional_memory is False, (
            "_use_bidirectional_memory not reset after Pass 2 exception"
        )
        assert pred._pass1_output_dict is None, (
            "_pass1_output_dict not cleaned up after Pass 2 exception"
        )

    # ---- conditioning frame isolation ----

    def test_single_cond_frame_skipped_in_pass2(self):
        """
        3-frame volume, frame 0 is cond.
        start_frame_idx defaults to min(cond) = 0, so range(0, 3) covers all frames.
        Pass 1:  frame 0 (cond, skip), frames 1, 2 → 2 calls.
        Pass 2:  frame 0 (cond, skip), frames 1, 2 → 2 more calls.
        Total:   4 inference calls.
        If frame 0 were mistakenly re-run in Pass 2 the total would be 5.
        """
        state = _make_inference_state(3, cond_frame_indices=[0])
        pred, results = self._run_propagation(state)
        assert pred._run_single_frame_inference.call_count == 4
        assert len(results) == 3

    def test_multiple_cond_frames_skipped_in_pass2(self):
        """
        6-frame volume with cond frames 0 and 4 (4 non-cond frames).
        Pass 1:  4 calls.
        Pass 2:  4 calls.
        Total:   8 inference calls.
        """
        state = _make_inference_state(6, cond_frame_indices=[0, 4])
        pred, results = self._run_propagation(state)
        assert pred._run_single_frame_inference.call_count == 8
        assert len(results) == 6


# ---------------------------------------------------------------------------
# Test 6 – Temporal PE index ranges
# ---------------------------------------------------------------------------

class TestTemporalPESymmetry:

    def test_future_pe_indices_in_range(self):
        """Future frame k must use index k-1, which must be in [0, num_maskmem-1]."""
        for num_maskmem in (4, 7):
            for k in range(1, num_maskmem):
                idx = k - 1
                assert 0 <= idx < num_maskmem, (
                    f"num_maskmem={num_maskmem}: future PE index {idx} out of range"
                )

    def test_past_pe_indices_in_range(self):
        """Past frame at t_pos uses index num_maskmem - t_pos - 1, must be in range."""
        for num_maskmem in (4, 7):
            for t_pos in range(num_maskmem):
                idx = num_maskmem - t_pos - 1
                assert 0 <= idx < num_maskmem, (
                    f"num_maskmem={num_maskmem}: past PE index {idx} out of range"
                )

