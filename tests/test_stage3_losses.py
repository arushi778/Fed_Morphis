"""Comprehensive Stage 3B unit and regression tests for loss functions, weight derivation, and component logging.

Covers:
1. Mandatory baseline preservation (configs/model.yaml unchanged: bce_tversky, 0.5/0.5, alpha=0.3, beta=0.7, smooth=1e-5, pos_weight=None)
2. Hand-calculable mathematical examples for:
   - WeightedBCELoss
   - BalancedDiceLoss
   - FocalTverskyLoss (gamma > 1, gamma == 1, gamma < 1)
   - BCEFocalTverskyLoss (BCE + Focal Tversky)
   - Weighted BCE + Tversky and Weighted BCE + Focal Tversky
3. Independent 4-channel behavior ([MA, HE, EX, SE])
4. All-empty target channels and un-exponentiated L_empty = mean(p^2) in FocalTverskyLoss
5. Single-pixel MA targets and SE-absent batches
6. Extreme logits (+50 and -50) with finite forward/backward gradients
7. FP32 and AMP (float16 / bfloat16 autocast) compatibility
8. Training-only class-weight derivation (derive_training_class_weights):
   - Dampened square-root weighting and linear/log/effective_num modes
   - Configurable min_weight and max_weight clipping
   - Zero-positive-class handling policies ('max_weight', 'min_weight', 'one', 'error')
   - Strict rejection of 'valid', 'val', 'validation', 'test', missing, or empty split records
   - Zero image or mask file I/O during weight derivation
   - Explicit documentation of rectangular FOV crop box vs. circular retinal FOV
   - Distinguishability of manual vs. derived weights in summarize_loss_configuration
9. Configuration validation in src/utils/config.py (gamma, alpha/beta, pos_weight, class_weights, component weights)
10. Trainer and evaluator compatibility and accurate component logging (no silent zero component losses or mislabeling Focal Tversky as Dice)
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.engine.evaluator import evaluate_aupr_dataset, evaluate_segmentation_dataloader
from src.engine.trainer import SupervisedSegmentationTrainer
from src.models.losses import (
    BCEDiceLoss,
    BCEFocalTverskyLoss,
    BCETverskyLoss,
    BalancedDiceLoss,
    CANONICAL_LESION_ORDER,
    FocalTverskyLoss,
    SoftDiceLoss,
    TrainingWeightDerivationError,
    TverskyLoss,
    WeightedBCELoss,
    build_loss_from_config,
    derive_training_class_weights,
    extract_loss_components,
    summarize_loss_configuration,
)
from src.utils.config import (
    ConfigValidationError,
    SUPPORTED_LOSS_TYPES,
    load_yaml_config,
    validate_model_config,
)


# ---------------------------------------------------------------------------
# 1. Mandatory Baseline Preservation
# ---------------------------------------------------------------------------


def test_mandatory_baseline_model_yaml_unchanged() -> None:
    """Verify configs/model.yaml preserves the exact baseline loss configuration."""
    cfg = load_yaml_config("configs/model.yaml", validate=True)
    loss_cfg = cfg["loss"]

    assert loss_cfg["type"] == "bce_tversky"
    assert loss_cfg["bce_weight"] == 0.5
    assert loss_cfg["tversky_weight"] == 0.5
    assert loss_cfg["alpha"] == 0.3
    assert loss_cfg["beta"] == 0.7
    assert loss_cfg["smooth"] == 1e-5
    assert loss_cfg.get("pos_weight") is None

    criterion = build_loss_from_config(cfg)
    assert type(criterion) is BCETverskyLoss
    assert criterion.bce_weight == 0.5
    assert criterion.tversky_weight == 0.5
    assert criterion.alpha == 0.3
    assert criterion.beta == 0.7
    assert criterion.smooth == 1e-5
    assert criterion.pos_weight is None
    assert criterion.class_weights is None


# ---------------------------------------------------------------------------
# 2. Hand-Calculable Mathematical Verification
# ---------------------------------------------------------------------------


def test_hand_calculable_weighted_bce_loss() -> None:
    """Verify WeightedBCELoss against exact hand-calculated values."""
    pos_w = [10.0, 2.0, 4.0, 8.0]
    criterion = WeightedBCELoss(pos_weight=pos_w, bce_weight=1.0)

    # Shape (1, 4, 1, 2): first pixel y=1, second pixel y=0, all logits=0 -> p=0.5
    logits = torch.zeros(1, 4, 1, 2, dtype=torch.float32, requires_grad=True)
    targets = torch.zeros(1, 4, 1, 2, dtype=torch.float32)
    targets[:, :, 0, 0] = 1.0  # pixel 0 is positive, pixel 1 is negative

    # For logit=0: -log(sigmoid(0)) = log(2) = 0.69314718
    # Channel c has 1 positive pixel (weighted by w_c) and 1 negative pixel (weight 1):
    # mean over 2 pixels = 0.5 * (w_c + 1.0) * log(2)
    log2 = math.log(2.0)
    expected_per_ch = torch.tensor([0.5 * (w + 1.0) * log2 for w in pos_w], dtype=torch.float32)
    expected_total = expected_per_ch.mean()

    total, bce_l, sec_l, bce_ch, sec_ch = criterion.forward_tensors(logits, targets)
    assert torch.allclose(bce_ch, expected_per_ch, atol=1e-6)
    assert torch.allclose(bce_l, expected_total, atol=1e-6)
    assert torch.allclose(total, expected_total, atol=1e-6)
    assert float(sec_l.item()) == 0.0
    assert torch.all(sec_ch == 0.0)

    total.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_hand_calculable_balanced_dice_loss() -> None:
    """Verify BalancedDiceLoss with hand-calculable probabilities and per-channel weights."""
    smooth = 1e-5
    cw = [2.5, 0.5, 0.5, 0.5]  # sum = 4.0, mean = 1.0
    criterion = BalancedDiceLoss(class_weights=cw, smooth=smooth, batch_dice=True)
    unweighted = SoftDiceLoss(smooth=smooth, batch_dice=True)

    # Create known probabilities p = 0.8 on foreground (logit = log(4)), p = 0.2 on background (logit = -log(4))
    logit_pos = math.log(4.0)
    logits = torch.full((1, 4, 2, 2), -logit_pos, dtype=torch.float32, requires_grad=True)
    targets = torch.zeros(1, 4, 2, 2, dtype=torch.float32)

    # Channel 0 (MA): 1 positive pixel predicted with p=0.5 (logit=0), 3 negative pixels with p=0.2
    logits_data = logits.detach().clone()
    logits_data[0, 0, 0, 0] = 0.0
    targets[0, 0, 0, 0] = 1.0

    # Channels 1, 2, 3: 2 positive pixels predicted with p=0.8, 2 negative pixels with p=0.2
    for c in (1, 2, 3):
        logits_data[0, c, 0, :] = logit_pos
        targets[0, c, 0, :] = 1.0

    logits_in = logits_data.requires_grad_(True)
    per_ch_unw = unweighted.forward_per_channel(logits_in, targets)
    per_ch_bal = criterion.forward_per_channel(logits_in, targets)

    # Per-channel raw Dice losses must be identical before class weighting
    assert torch.allclose(per_ch_unw, per_ch_bal, atol=1e-7)

    # Hand-check channel 0:
    # p = [0.5, 0.2, 0.2, 0.2], y = [1, 0, 0, 0]
    # intersection = 0.5; sum(p^2) = 0.25 + 3*0.04 = 0.37; sum(y^2) = 1.0 -> cardinality = 1.37
    expected_ch0 = 1.0 - (2.0 * 0.5 + smooth) / (1.37 + smooth)
    assert abs(float(per_ch_bal[0].item()) - expected_ch0) < 1e-6

    # Scalar loss is mean(per_ch * cw)
    cw_t = torch.tensor(cw, dtype=torch.float32)
    expected_scalar = torch.mean(per_ch_bal * cw_t)
    actual_scalar = criterion(logits_in, targets)
    assert torch.allclose(actual_scalar, expected_scalar, atol=1e-6)
    # Because MA (channel 0) has higher error and higher weight (2.5), balanced loss > unweighted loss
    assert float(actual_scalar.item()) > float(unweighted(logits_in, targets).item())


@pytest.mark.parametrize("gamma", [0.1, 0.5, 0.75, 1.0, 4.0 / 3.0, 2.0])
def test_hand_calculable_focal_tversky_loss_across_gammas(gamma: float) -> None:
    """Verify FocalTverskyLoss matches shifted-power (gamma<1) or (1-TI)^gamma (gamma>=1) and mean(p^2) for empty."""
    alpha, beta, smooth = 0.3, 0.7, 1e-5
    ftl = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma, smooth=smooth, batch_tversky=True, eps=1e-6)
    tl = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth, batch_tversky=True)

    # 4 channels: channels 0, 1, 2 present; channel 3 (SE) completely empty (target_sum == 0)
    logits = torch.zeros(1, 4, 2, 2, dtype=torch.float32)
    # Channel 0: p = 0.5 everywhere, 1 positive pixel
    # Channel 1: p = sigmoid(1.0) everywhere, 2 positive pixels
    # Channel 2: p = sigmoid(2.0) on positives, sigmoid(-2.0) on negatives
    # Channel 3 (SE): empty target, logit = 0 -> p = 0.5 -> L_empty = mean(p^2) = 0.25 (un-exponentiated!)
    logits[0, 1] = 1.0
    logits[0, 2, 0, :] = 2.0
    logits[0, 2, 1, :] = -2.0
    logits.requires_grad_(True)

    targets = torch.zeros(1, 4, 2, 2, dtype=torch.float32)
    targets[0, 0, 0, 0] = 1.0
    targets[0, 1, 0, :] = 1.0
    targets[0, 2, 0, :] = 1.0
    # targets[0, 3] remains all zeros

    tl_per_ch = tl.forward_per_channel(logits, targets)
    ftl_per_ch = ftl.forward_per_channel(logits, targets)

    # Present channels 0, 1, 2:
    for c in (0, 1, 2):
        u_c = float(tl_per_ch[c].item())
        if gamma < 1.0:
            expected_c = ((u_c + ftl.eps) ** gamma) - (ftl.eps ** gamma)
        else:
            expected_c = u_c ** gamma
        assert abs(float(ftl_per_ch[c].item()) - expected_c) < 1e-5

    # Absent channel 3 (SE): must be un-exponentiated mean(p^2) = 0.25 regardless of gamma!
    assert abs(float(ftl_per_ch[3].item()) - 0.25) < 1e-6
    assert abs(float(tl_per_ch[3].item()) - 0.25) < 1e-6

    # When gamma == 1.0, FocalTverskyLoss must match TverskyLoss identically in forward and backward
    if gamma == 1.0:
        assert torch.allclose(ftl_per_ch, tl_per_ch, atol=1e-7)
        logits_tl = logits.detach().clone().requires_grad_(True)
        loss_tl = tl(logits_tl, targets)
        loss_tl.backward()

    loss = ftl(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    if gamma == 1.0:
        assert torch.allclose(logits.grad, logits_tl.grad, atol=1e-7)


@pytest.mark.parametrize("gamma", [0.1, 0.5, 0.75, 1.0, 4.0 / 3.0, 2.0])
def test_focal_tversky_zero_error_limit_has_no_artificial_loss_floor(gamma: float) -> None:
    """Verify FocalTverskyLoss has exact zero loss and finite gradients at u == 0 across all gammas."""
    targets = torch.zeros(1, 4, 4, 4, dtype=torch.float32)
    targets[:, :, :2, :2] = 1.0
    # Confident exact predictions (+50 on foreground, -50 on background) -> p = 1.0 on FG, 0.0 on BG
    # Thus FP = 0, FN = 0 -> 1 - TI = 0
    logits = torch.where(targets > 0.5, torch.tensor(50.0), torch.tensor(-50.0)).requires_grad_(True)

    criterion = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=gamma, smooth=1e-5, eps=1e-6)
    loss = criterion(logits, targets)
    assert torch.isfinite(loss)
    # Exact zero loss at u == 0.0 (no non-zero floor even at gamma=0.1)
    assert float(loss.item()) == pytest.approx(0.0, abs=1e-12)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_hand_calculable_bce_focal_tversky_and_weighted_combinations() -> None:
    """Verify BCEFocalTverskyLoss and Weighted BCE + Tversky/Focal Tversky combinations."""
    pos_w = [12.0, 4.0, 4.0, 8.0]
    cw = [1.5, 0.7, 0.8, 1.0]
    logits = torch.randn(2, 4, 8, 8, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, 2, (2, 4, 8, 8)).float()

    criterion = BCEFocalTverskyLoss(
        bce_weight=0.4,
        focal_tversky_weight=0.6,
        alpha=0.3,
        beta=0.7,
        gamma=1.5,
        smooth=1e-5,
        pos_weight=pos_w,
        class_weights=cw,
    )
    wbce = WeightedBCELoss(pos_weight=pos_w)
    ftl = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.5, smooth=1e-5, class_weights=cw)

    tot, bce_l, ftl_l, bce_ch, ftl_ch = criterion.forward_tensors(logits, targets)
    expected_bce = wbce(logits, targets)
    expected_ftl = ftl(logits, targets)
    expected_tot = 0.4 * expected_bce + 0.6 * expected_ftl

    assert torch.allclose(bce_l, expected_bce, atol=1e-6)
    assert torch.allclose(ftl_l, expected_ftl, atol=1e-6)
    assert torch.allclose(tot, expected_tot, atol=1e-6)

    comps = criterion.compute_components(logits, targets)
    assert comps["primary_component_name"] == "bce"
    assert comps["secondary_component_name"] == "focal_tversky"
    assert "focal_tversky_loss" in comps
    assert "per_channel_focal_tversky" in comps
    assert torch.allclose(comps["focal_tversky_loss"], ftl_l.detach(), atol=1e-6)


# ---------------------------------------------------------------------------
# 3. Independent Channel Behavior, Single-Pixel MA, SE-Absent, & Extreme Logits
# ---------------------------------------------------------------------------


def test_four_channel_strict_independence() -> None:
    """Verify modifying logits/targets in one channel never alters loss or gradients of other channels."""
    losses_to_check = [
        WeightedBCELoss(pos_weight=[12.0, 4.0, 4.0, 8.0]),
        BalancedDiceLoss(class_weights=[1.6, 0.6, 0.8, 1.0]),
        FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=4.0 / 3.0),
        BCETverskyLoss(pos_weight=[12.0, 4.0, 4.0, 8.0]),
        BCEFocalTverskyLoss(pos_weight=[12.0, 4.0, 4.0, 8.0], gamma=4.0 / 3.0),
    ]
    torch.manual_seed(99)
    base_logits = torch.randn(2, 4, 8, 8)
    base_targets = torch.randint(0, 2, (2, 4, 8, 8)).float()

    for criterion in losses_to_check:
        l1 = base_logits.clone().requires_grad_(True)
        l2 = base_logits.clone()
        # Perturb only channel 2 (EX)
        l2[:, 2, :, :] = l2[:, 2, :, :] * -3.0 + 5.0
        l2.requires_grad_(True)

        _, bce1, ov1, bce_ch1, ov_ch1 = criterion.forward_tensors(l1, base_targets)
        _, bce2, ov2, bce_ch2, ov_ch2 = criterion.forward_tensors(l2, base_targets)

        for ch_idx in (0, 1, 3):  # MA, HE, SE must be 100% unchanged
            assert torch.allclose(bce_ch1[ch_idx], bce_ch2[ch_idx], atol=1e-7)
            assert torch.allclose(ov_ch1[ch_idx], ov_ch2[ch_idx], atol=1e-7)

        loss1 = criterion(l1, base_targets)
        loss2 = criterion(l2, base_targets)
        loss1.backward()
        loss2.backward()
        assert l1.grad is not None and l2.grad is not None
        for ch_idx in (0, 1, 3):
            assert torch.allclose(l1.grad[:, ch_idx], l2.grad[:, ch_idx], atol=1e-7)


def test_single_pixel_ma_and_se_absent_batch_and_all_empty() -> None:
    """Verify single-pixel MA target, SE-absent batch, and all-empty batch produce finite gradients."""
    candidates = [
        WeightedBCELoss(pos_weight=[15.0, 4.0, 4.0, 10.0]),
        BalancedDiceLoss(class_weights=[2.0, 0.5, 0.5, 1.0]),
        FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=4.0 / 3.0),
        BCEFocalTverskyLoss(pos_weight=[15.0, 4.0, 4.0, 10.0], gamma=4.0 / 3.0),
        BCETverskyLoss(pos_weight=[15.0, 4.0, 4.0, 10.0], alpha=0.3, beta=0.7),
    ]

    # Scenario A: Single-pixel MA target (1 pixel out of 2 * 32 * 32 = 2048) and SE completely absent
    targets_sparse = torch.zeros(2, 4, 32, 32, dtype=torch.float32)
    targets_sparse[0, 0, 16, 16] = 1.0  # Single MA pixel
    targets_sparse[:, 1, 8:12, 8:12] = 1.0  # HE present
    targets_sparse[:, 2, 20:24, 20:24] = 1.0  # EX present
    # Channel 3 (SE) is 0 everywhere across the batch

    # Scenario B: All 4 channels completely empty
    targets_empty = torch.zeros(2, 4, 32, 32, dtype=torch.float32)

    for criterion in candidates:
        for targets in (targets_sparse, targets_empty):
            logits = torch.randn(2, 4, 32, 32, dtype=torch.float32, requires_grad=True)
            loss, bce_l, ov_l, bce_ch, ov_ch = criterion.forward_tensors(logits, targets)
            assert torch.isfinite(loss)
            assert torch.isfinite(bce_ch).all()
            assert torch.isfinite(ov_ch).all()
            loss.backward()
            assert logits.grad is not None
            assert torch.isfinite(logits.grad).all()
            # Gradient on the single MA pixel must be strictly non-zero
            if targets is targets_sparse:
                assert float(logits.grad[0, 0, 16, 16].abs().item()) > 0.0


@pytest.mark.parametrize("extreme_val", [50.0, -50.0])
def test_extreme_logits_plus_minus_50_finite_forward_and_backward(extreme_val: float) -> None:
    """Verify extreme logits (+50 and -50) never overflow/underflow to NaN or Inf."""
    candidates = [
        WeightedBCELoss(pos_weight=[20.0, 4.0, 5.0, 12.0]),
        BalancedDiceLoss(class_weights=[1.8, 0.6, 0.6, 1.0]),
        FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.8),
        FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=4.0 / 3.0),
        BCEFocalTverskyLoss(pos_weight=[20.0, 4.0, 5.0, 12.0], gamma=4.0 / 3.0),
        BCETverskyLoss(pos_weight=[20.0, 4.0, 5.0, 12.0]),
    ]
    targets = torch.zeros(2, 4, 8, 8, dtype=torch.float32)
    targets[:, :3, :4, :4] = 1.0  # Channel 3 empty, Channels 0..2 half positive

    for criterion in candidates:
        logits = torch.full((2, 4, 8, 8), extreme_val, dtype=torch.float32, requires_grad=True)
        loss = criterion(logits, targets)
        assert torch.isfinite(loss), f"Non-finite loss for {type(criterion).__name__} at logit={extreme_val}"
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all(), (
            f"Non-finite grad for {type(criterion).__name__} at logit={extreme_val}"
        )


@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
def test_fp32_and_amp_compatibility(amp_dtype: torch.dtype) -> None:
    """Verify all candidate losses execute in float32 internally when given float16/bfloat16 inputs."""
    candidates = [
        WeightedBCELoss(pos_weight=[12.0, 4.0, 4.0, 8.0]),
        BalancedDiceLoss(class_weights=[1.5, 0.7, 0.8, 1.0]),
        FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=4.0 / 3.0),
        BCEFocalTverskyLoss(pos_weight=[12.0, 4.0, 4.0, 8.0], gamma=4.0 / 3.0),
    ]
    targets = torch.randint(0, 2, (2, 4, 16, 16)).float()

    for criterion in candidates:
        logits_lp = torch.randn(2, 4, 16, 16, dtype=amp_dtype, requires_grad=True)
        loss = criterion(logits_lp, targets)
        assert loss.dtype == torch.float32
        assert torch.isfinite(loss)
        loss.backward()
        assert logits_lp.grad is not None
        assert torch.isfinite(logits_lp.grad).all()


# ---------------------------------------------------------------------------
# 4. Training-Only Class-Weight Derivation & Safeguards
# ---------------------------------------------------------------------------


def _make_synthetic_train_records() -> list[dict[str, Any]]:
    """Build synthetic training records with known dimensions and lesion pixel counts."""
    return [
        {
            "dataset_name": "IDRiD",
            "sample_id": "IDRiD_01",
            "split": "train",
            "image_path": "nonexistent_dir/IDRiD_01.jpg",
            "mask_paths": {"MA": "nonexistent_dir/m1.tif", "HE": None, "EX": None, "SE": None},
            "image_dimensions": {"height": 100, "width": 100, "channels": 3},
            "fov_crop": {"y_min": 10, "y_max": 90, "x_min": 10, "x_max": 90},  # 80x80 = 6400 px
            "lesion_pixel_counts": {"MA": 16, "HE": 400, "EX": 160, "SE": 0},
        },
        {
            "dataset_name": "IDRiD",
            "sample_id": "IDRiD_02",
            "split": "train",
            "image_path": "nonexistent_dir/IDRiD_02.jpg",
            "mask_paths": {"MA": "nonexistent_dir/m2.tif", "HE": None, "EX": None, "SE": None},
            "image_dimensions": {"height": 100, "width": 100, "channels": 3},
            "fov_crop": {"y_min": 0, "y_max": 60, "x_min": 0, "x_max": 60},  # 60x60 = 3600 px
            "lesion_pixel_counts": {"MA": 9, "HE": 600, "EX": 90, "SE": 100},
        },
    ]


def test_derive_training_class_weights_math_and_clipping() -> None:
    """Verify exact sqrt_dampened math, canonical ordering, and [min_weight, max_weight] clipping."""
    records = _make_synthetic_train_records()
    # Total fov_crop_box pixels = 6400 + 3600 = 10,000
    # Positive counts: MA = 25, HE = 1000, EX = 250, SE = 100
    # Negative counts: MA = 9975, HE = 9000, EX = 9750, SE = 9900
    # Raw sqrt(neg/pos):
    #   MA: sqrt(9975 / 25) = sqrt(399) = 19.974984
    #   HE: sqrt(9000 / 1000) = sqrt(9) = 3.0
    #   EX: sqrt(9750 / 250) = sqrt(39) = 6.244998
    #   SE: sqrt(9900 / 100) = sqrt(99) = 9.949874
    res = derive_training_class_weights(
        records,
        mode="sqrt_dampened",
        pixel_domain="fov_crop_box",
        min_weight=4.0,
        max_weight=15.0,
    )

    assert res["channel_order"] == ["MA", "HE", "EX", "SE"]
    assert res["weight_source"] == "derived_from_train_metadata"
    assert res["num_train_records"] == 2
    assert res["total_domain_pixels"] == 10000
    assert res["positive_pixel_counts"] == {"MA": 25, "HE": 1000, "EX": 250, "SE": 100}
    assert res["negative_pixel_counts"] == {"MA": 9975, "HE": 9000, "EX": 9750, "SE": 9900}
    assert res["absent_image_counts"] == {"MA": 0, "HE": 0, "EX": 0, "SE": 1}

    # MA raw ~19.975 clipped to max_weight=15.0; HE raw 3.0 clipped to min_weight=4.0
    assert res["clipped_channels"] == ["MA", "HE"]
    assert res["weights_by_class"]["MA"] == pytest.approx(15.0)
    assert res["weights_by_class"]["HE"] == pytest.approx(4.0)
    assert res["weights_by_class"]["EX"] == pytest.approx(math.sqrt(39.0), abs=1e-5)
    assert res["weights_by_class"]["SE"] == pytest.approx(math.sqrt(99.0), abs=1e-5)
    assert res["weights"] == [
        res["weights_by_class"]["MA"],
        res["weights_by_class"]["HE"],
        res["weights_by_class"]["EX"],
        res["weights_by_class"]["SE"],
    ]
    assert "circular" in res["fov_accounting_note"]
    assert "fov_crop_box" in res["fov_accounting_note"]


def test_derive_training_class_weights_zero_positive_policies() -> None:
    """Verify explicit handling when a lesion class has zero positive pixels across all records."""
    records = _make_synthetic_train_records()
    for r in records:
        r["lesion_pixel_counts"]["SE"] = 0

    res_max = derive_training_class_weights(
        records, min_weight=2.0, max_weight=20.0, zero_positive_policy="max_weight"
    )
    assert res_max["zero_positive_channels"] == ["SE"]
    assert res_max["weights_by_class"]["SE"] == 20.0
    assert res_max["neg_to_pos_ratio_by_class"]["SE"] is None

    res_min = derive_training_class_weights(
        records, min_weight=2.0, max_weight=20.0, zero_positive_policy="min_weight"
    )
    assert res_min["weights_by_class"]["SE"] == 2.0

    res_one = derive_training_class_weights(
        records, min_weight=0.5, max_weight=20.0, zero_positive_policy="one"
    )
    assert res_one["weights_by_class"]["SE"] == 1.0

    with pytest.raises(TrainingWeightDerivationError, match="Channel 'SE' has 0 positive pixels"):
        derive_training_class_weights(records, zero_positive_policy="error")


@pytest.mark.parametrize("forbidden_split", ["valid", "val", "validation", "test", "", "   ", None, 123])
def test_derive_training_class_weights_rejects_non_train_or_missing_splits(
    forbidden_split: Any,
) -> None:
    """Verify validation, test, missing, or ambiguous split records are strictly rejected."""
    records = _make_synthetic_train_records()
    records[1]["split"] = forbidden_split

    with pytest.raises(TrainingWeightDerivationError):
        derive_training_class_weights(records)

    # Also test completely missing 'split' key
    records_no_key = _make_synthetic_train_records()
    del records_no_key[0]["split"]
    with pytest.raises(TrainingWeightDerivationError, match="Missing 'split' metadata"):
        derive_training_class_weights(records_no_key)


def test_derive_training_class_weights_never_loads_images_or_masks() -> None:
    """Verify derive_training_class_weights never calls image/mask loading functions."""
    records = _make_synthetic_train_records()
    with (
        patch("src.data.masks.load_rgb_image", side_effect=AssertionError("Image load forbidden")),
        patch("src.data.masks.load_sample_image_and_mask", side_effect=AssertionError("Mask load forbidden")),
        patch("cv2.imread", side_effect=AssertionError("cv2.imread forbidden")),
    ):
        res = derive_training_class_weights(records)
        assert len(res["weights"]) == 4


def test_manual_vs_derived_weights_distinguishable_in_summary() -> None:
    """Verify summarize_loss_configuration distinguishes manual, derived, and unweighted configs."""
    base_cfg = load_yaml_config("configs/model.yaml")
    s_unweighted = summarize_loss_configuration(base_cfg)
    assert s_unweighted["pos_weight"] is None
    assert s_unweighted["pos_weight_source"] == "none"

    manual_cfg = copy.deepcopy(base_cfg)
    manual_cfg["loss"]["pos_weight"] = [12.0, 4.0, 4.0, 8.0]
    s_manual = summarize_loss_configuration(manual_cfg)
    assert s_manual["pos_weight"] == [12.0, 4.0, 4.0, 8.0]
    assert s_manual["pos_weight_source"] == "manual"

    derived = derive_training_class_weights(_make_synthetic_train_records())
    derived_cfg = copy.deepcopy(base_cfg)
    derived_cfg["loss"]["pos_weight"] = derived["weights"]
    derived_cfg["loss"]["pos_weight_source"] = derived["weight_source"]
    s_derived = summarize_loss_configuration(derived_cfg)
    assert s_derived["pos_weight"] == derived["weights"]
    assert s_derived["pos_weight_source"] == "derived_from_train_metadata"


# ---------------------------------------------------------------------------
# 5. Configuration Validation Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("loss_type", SUPPORTED_LOSS_TYPES)
def test_all_supported_loss_types_validate_and_build(loss_type: str) -> None:
    """Verify every loss type in SUPPORTED_LOSS_TYPES passes validate_model_config and builds cleanly."""
    cfg = load_yaml_config("configs/model.yaml")
    cfg["loss"] = {
        "type": loss_type,
        "bce_weight": 0.5,
        "dice_weight": 0.5,
        "tversky_weight": 0.5,
        "focal_tversky_weight": 0.5,
        "alpha": 0.3,
        "beta": 0.7,
        "gamma": 4.0 / 3.0,
        "smooth": 1e-5,
        "pos_weight": [12.0, 4.0, 4.0, 8.0],
        "class_weights": [1.5, 0.7, 0.8, 1.0],
    }
    validate_model_config(cfg)
    criterion = build_loss_from_config(cfg)
    assert isinstance(criterion, nn.Module)

    logits = torch.randn(2, 4, 16, 16, requires_grad=True)
    targets = torch.randint(0, 2, (2, 4, 16, 16)).float()
    loss = criterion(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_invalid_loss_hyperparameters_rejected_by_config_validation() -> None:
    """Verify invalid gamma, alpha/beta, class_weights, pos_weight, and loss.type are rejected."""
    base = load_yaml_config("configs/model.yaml")

    # Unknown loss type
    bad_type = copy.deepcopy(base)
    bad_type["loss"]["type"] = "nonexistent_loss"
    with pytest.raises(ConfigValidationError, match="loss.type"):
        validate_model_config(bad_type)

    # Non-positive gamma
    for bad_gamma in (0.0, -1.0, float("nan"), float("inf"), True):
        bad_g = copy.deepcopy(base)
        bad_g["loss"]["gamma"] = bad_gamma
        with pytest.raises(ConfigValidationError, match="loss.gamma"):
            validate_model_config(bad_g)

    # alpha + beta == 0
    bad_ab = copy.deepcopy(base)
    bad_ab["loss"]["alpha"] = 0.0
    bad_ab["loss"]["beta"] = 0.0
    with pytest.raises(ConfigValidationError, match=r"alpha.*beta.*must be > 0"):
        validate_model_config(bad_ab)

    # Invalid class_weights
    for bad_cw in ([1.0, 2.0], [1.0, 0.0, 1.0, 1.0], [1.0, -1.0, 1.0, 1.0], [1.0, True, 1.0, 1.0], [1.0, float("inf"), 1.0, 1.0]):
        cfg_cw = copy.deepcopy(base)
        cfg_cw["loss"]["class_weights"] = bad_cw
        with pytest.raises(ConfigValidationError, match="class_weights"):
            validate_model_config(cfg_cw)


# ---------------------------------------------------------------------------
# 6. Trainer and Evaluator Component Logging Compatibility
# ---------------------------------------------------------------------------


class _TinySegDataset(Dataset[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]):
    """Tiny synthetic dataset for verifying trainer and evaluator loss component logging."""

    def __init__(self, split: str = "train", num_samples: int = 2) -> None:
        self.split = split
        self.num_samples = num_samples
        gen = torch.Generator().manual_seed(42)
        self.images = torch.randn(num_samples, 3, 32, 32, generator=gen)
        self.masks = torch.zeros(num_samples, 4, 32, 32, dtype=torch.float32)
        self.masks[:, :3, 8:24, 8:24] = 1.0  # MA, HE, EX present; SE absent
        self.records = [
            {"dataset_name": "DDR", "sample_id": f"S_{split}_{i}", "split": split}
            for i in range(num_samples)
        ]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        return (
            self.images[idx],
            self.masks[idx],
            {
                "dataset_name": "DDR",
                "sample_id": self.records[idx]["sample_id"],
                "split": self.split,
            },
        )


class _TinyConvModel(nn.Module):
    """Lightweight 3->4 conv model for fast trainer/evaluator unit tests."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=1)
        self.pretrained_requested = False
        self.pretrained_weights_loaded = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


@pytest.mark.parametrize(
    ("criterion", "expected_primary", "expected_secondary", "expect_nonzero_bce", "expect_nonzero_overlap"),
    [
        (WeightedBCELoss(pos_weight=[12.0, 4.0, 4.0, 8.0]), "bce", "none", True, False),
        (BalancedDiceLoss(class_weights=[1.5, 0.7, 0.8, 1.0]), "none", "balanced_dice", False, True),
        (FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=4.0 / 3.0), "none", "focal_tversky", False, True),
        (BCEFocalTverskyLoss(pos_weight=[12.0, 4.0, 4.0, 8.0], gamma=4.0 / 3.0), "bce", "focal_tversky", True, True),
        (BCETverskyLoss(pos_weight=[12.0, 4.0, 4.0, 8.0]), "bce", "tversky", True, True),
    ],
)
def test_trainer_and_evaluator_component_logging_and_aupr_compatibility(
    tmp_path: Path,
    criterion: nn.Module,
    expected_primary: str,
    expected_secondary: str,
    expect_nonzero_bce: bool,
    expect_nonzero_overlap: bool,
) -> None:
    """Verify SupervisedSegmentationTrainer, evaluate_segmentation_dataloader, and evaluate_aupr_dataset."""
    torch.manual_seed(42)
    model = _TinyConvModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_ds = _TinySegDataset(split="train", num_samples=2)
    val_ds = _TinySegDataset(split="valid", num_samples=2)
    train_loader = DataLoader(train_ds, batch_size=2)
    val_loader = DataLoader(val_ds, batch_size=2)

    trainer = SupervisedSegmentationTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu",
        checkpoint_dir=tmp_path / "ckpts",
        mixed_precision=False,
    )

    train_summary = trainer.train_one_epoch(train_loader, epoch=1)
    val_summary = trainer.validate_one_epoch(val_loader, epoch=1)

    for summary in (train_summary, val_summary):
        assert summary["primary_loss_component"] == expected_primary
        assert summary["secondary_loss_component"] == expected_secondary
        assert summary["loss_components"]["primary_name"] == expected_primary
        assert summary["loss_components"]["secondary_name"] == expected_secondary
        assert summary["loss"] > 0.0
        if expect_nonzero_bce:
            assert summary["bce_loss"] > 0.0
        else:
            assert summary["bce_loss"] == 0.0
        if expect_nonzero_overlap:
            assert summary["overlap_loss"] > 0.0
            assert summary["dice_loss"] > 0.0
            assert summary[f"{expected_secondary}_loss"] == pytest.approx(summary["overlap_loss"])
        else:
            assert summary["overlap_loss"] == 0.0

    # Verify evaluate_segmentation_dataloader with compute_aupr=True
    eval_res = evaluate_segmentation_dataloader(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device="cpu",
        expected_split="valid",
        compute_aupr=True,
    )
    assert eval_res["primary_loss_component"] == expected_primary
    assert eval_res["secondary_loss_component"] == expected_secondary
    assert eval_res["loss"] > 0.0
    if expect_nonzero_overlap:
        assert eval_res["overlap_loss"] > 0.0
        assert eval_res[f"{expected_secondary}_loss"] == pytest.approx(eval_res["overlap_loss"])
    assert "aupr" in eval_res

    # Verify evaluate_aupr_dataset remains compatible
    aupr_res = evaluate_aupr_dataset(
        model=model,
        dataset=val_ds,
        device="cpu",
        expected_split="valid",
        use_sliding_window=False,
    )
    assert aupr_res["num_samples"] == 2
    assert set(aupr_res["aupr_metrics"]["per_class"].keys()) == {"MA", "HE", "EX", "SE"}


# ---------------------------------------------------------------------------
# 7. Stage 3D Targeted Audit Corrections (S3C-01, S3C-02, S3C-03)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gamma", [0.1, 0.5, 0.75, 1.0, 4.0 / 3.0, 2.0])
def test_s3c01_focal_tversky_near_zero_and_mixed_saturation_bounded_gradients(gamma: float) -> None:
    """Verify S3C-01: bounded dL/du and finite gradients on near-zero error and mixed saturated/non-saturated pixels."""
    eps = 1e-6
    criterion = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=gamma, smooth=1e-5, eps=eps)

    # 1. Direct derivative d/du of _apply_focal_exponent at u = 0.0 and u = 1e-9
    for u_val in (0.0, 1e-12, 1e-9, 1e-6, 1e-3):
        u_t = torch.tensor([u_val], dtype=torch.float32, requires_grad=True)
        out = criterion._apply_focal_exponent(u_t)
        assert torch.isfinite(out).all()
        if u_val == 0.0:
            assert float(out.item()) == pytest.approx(0.0, abs=1e-12)
        out.backward()
        assert u_t.grad is not None
        assert torch.isfinite(u_t.grad).all()
        grad_u = float(u_t.grad.item())
        if gamma < 1.0:
            max_theoretical_slope = gamma * (eps ** (gamma - 1.0))
            assert 0.0 <= grad_u <= max_theoretical_slope * 1.01
        elif gamma == 1.0:
            assert grad_u == pytest.approx(1.0, rel=1e-6)

    # 2. Mixed saturated (+30/-30) and non-saturated (0.0) pixels within the same channel
    # Channel 0..2 have 63 saturated correct pixels and 1 non-saturated pixel (logit = 0.0 -> p = 0.5);
    # Channel 3 is empty (all zeros) with logit = 0.0 -> p = 0.5 -> L_empty = 0.25 un-exponentiated.
    targets = torch.zeros(1, 4, 8, 8, dtype=torch.float32)
    targets[:, :3, :4, :] = 1.0  # 32 positive pixels per channel 0..2
    logits = torch.where(targets > 0.5, torch.tensor(30.0), torch.tensor(-30.0))
    # Set exactly 1 foreground pixel per present channel to non-saturated logit = 1.5
    logits[:, :3, 0, 0] = 1.5
    # Set empty channel 3 to logit = 0.0 everywhere
    logits[:, 3, :, :] = 0.0
    logits.requires_grad_(True)

    per_ch = criterion.forward_per_channel(logits, targets)
    assert torch.isfinite(per_ch).all()
    # Empty channel 3 must remain un-exponentiated mean(p^2) = 0.25 for every gamma
    assert float(per_ch[3].item()) == pytest.approx(0.25, abs=1e-6)

    loss = criterion(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    # Non-saturated pixel (0, 0) must receive a clean, bounded non-zero negative gradient (pushing logit up)
    non_sat_grad = float(logits.grad[0, 0, 0, 0].item())
    assert non_sat_grad < 0.0
    assert abs(non_sat_grad) < 100.0


def test_s3c01_focal_tversky_invalid_eps_rejected() -> None:
    """Verify FocalTverskyLoss and config validation reject non-positive or non-finite eps."""
    for bad_eps in (0.0, -1e-6, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="eps"):
            FocalTverskyLoss(gamma=0.75, eps=bad_eps)  # type: ignore[arg-type]

        cfg = load_yaml_config("configs/model.yaml")
        cfg["loss"]["type"] = "focal_tversky"
        cfg["loss"]["eps"] = bad_eps
        with pytest.raises(ConfigValidationError, match="loss.eps"):
            validate_model_config(cfg)


@pytest.mark.parametrize(
    ("loss_type", "expected_cls", "expect_bce", "expect_overlap"),
    [
        ("bce", WeightedBCELoss, True, False),
        ("weighted_bce", WeightedBCELoss, True, False),
        ("dice", SoftDiceLoss, False, True),
        ("soft_dice", SoftDiceLoss, False, True),
        ("balanced_dice", BalancedDiceLoss, False, True),
        ("tversky", TverskyLoss, False, True),
        ("focal_tversky", FocalTverskyLoss, False, True),
        ("bce_dice", BCEDiceLoss, True, True),
        ("weighted_bce_dice", BCEDiceLoss, True, True),
        ("bce_balanced_dice", BCEDiceLoss, True, True),
        ("bce_tversky", BCETverskyLoss, True, True),
        ("weighted_bce_tversky", BCETverskyLoss, True, True),
        ("bce_focal_tversky", BCEFocalTverskyLoss, True, True),
        ("weighted_bce_focal_tversky", BCEFocalTverskyLoss, True, True),
    ],
)
def test_s3c02_all_supported_loss_types_construct_exact_class_and_ignore_unrelated_fields(
    loss_type: str,
    expected_cls: type[nn.Module],
    expect_bce: bool,
    expect_overlap: bool,
) -> None:
    """Verify S3C-02: every supported loss type builds its exact declared class even when unrelated YAML keys exist."""
    base_cfg = load_yaml_config("configs/model.yaml")
    # Populate unrelated composite fields (as when overriding loss.type on top of configs/model.yaml)
    base_cfg["loss"] = {
        "type": loss_type,
        "bce_weight": 0.5,
        "dice_weight": 0.5,
        "tversky_weight": 0.5,
        "focal_tversky_weight": 0.5,
        "alpha": 0.3,
        "beta": 0.7,
        "gamma": 4.0 / 3.0,
        "smooth": 1e-5,
        "pos_weight": [12.0, 4.0, 4.0, 8.0],
        "class_weights": [1.5, 0.7, 0.8, 1.0],
    }
    validate_model_config(base_cfg)
    criterion = build_loss_from_config(base_cfg)
    assert type(criterion) is expected_cls

    logits = torch.randn(2, 4, 8, 8, requires_grad=True)
    targets = torch.randint(0, 2, (2, 4, 8, 8)).float()
    loss, bce_val, overlap_val, prim_name, sec_name = extract_loss_components(criterion, logits, targets)
    assert torch.isfinite(loss)
    if expect_bce:
        assert bce_val > 0.0
        assert prim_name == "bce"
    else:
        # Standalone overlap losses must have ZERO BCE contribution
        assert bce_val == 0.0
        assert prim_name == "none"

    if expect_overlap:
        assert overlap_val > 0.0
        assert sec_name != "none"
    else:
        assert overlap_val == 0.0
        assert sec_name == "none"

    # Specifically verify standalone Tversky and Focal Tversky are completely invariant to bce_weight and pos_weight
    if not expect_bce:
        cfg_alt = copy.deepcopy(base_cfg)
        cfg_alt["loss"]["bce_weight"] = 999.0
        cfg_alt["loss"]["pos_weight"] = [100.0, 100.0, 100.0, 100.0]
        validate_model_config(cfg_alt)
        crit_alt = build_loss_from_config(cfg_alt)
        loss_alt = crit_alt(logits, targets)
        assert torch.allclose(loss, loss_alt, atol=1e-7)


def test_s3c02_invalid_component_weights_fail_during_config_validation() -> None:
    """Verify S3C-02: invalid component weights fail during validate_model_config, not only in factory."""
    base = load_yaml_config("configs/model.yaml")

    # 1. Standalone BCE / weighted_bce with bce_weight=0.0 even when dice_weight=0.5 is present
    for bce_only_type in ("bce", "weighted_bce"):
        cfg = copy.deepcopy(base)
        cfg["loss"] = {"type": bce_only_type, "bce_weight": 0.0, "dice_weight": 0.5}
        with pytest.raises(ConfigValidationError, match="loss.bce_weight"):
            validate_model_config(cfg)

    # 2. Standalone overlap losses with their own weight explicitly set to 0.0 even when bce_weight=1.0
    for overlap_type, w_key in (
        ("dice", "dice_weight"),
        ("soft_dice", "dice_weight"),
        ("balanced_dice", "dice_weight"),
        ("tversky", "tversky_weight"),
        ("focal_tversky", "focal_tversky_weight"),
    ):
        cfg = copy.deepcopy(base)
        cfg["loss"] = {"type": overlap_type, "bce_weight": 1.0, w_key: 0.0}
        with pytest.raises(ConfigValidationError, match=f"loss\\.{w_key}"):
            validate_model_config(cfg)
        with pytest.raises(ValueError, match=f"loss\\.{w_key}"):
            build_loss_from_config(cfg)

    # 3. Composite bce_dice with bce_weight=0.0, dice_weight=0.0 even if unrelated tversky_weight=0.5 is present
    cfg_comp = copy.deepcopy(base)
    cfg_comp["loss"] = {
        "type": "bce_dice",
        "bce_weight": 0.0,
        "dice_weight": 0.0,
        "tversky_weight": 0.5,
    }
    with pytest.raises(ConfigValidationError, match="must be > 0"):
        validate_model_config(cfg_comp)


def test_s3c03_weight_normalization_and_clipping_semantics() -> None:
    """Verify S3C-03: normalization with no clipping, with clipping, default min_weight, min_weight=None, all-equal."""
    records = _make_synthetic_train_records()

    # 1. Normalization with no clipping (min_weight=None, max_weight=None) -> exact mean 1.0
    res_unclipped = derive_training_class_weights(
        records,
        normalize_mean=True,
        min_weight=None,
        max_weight=None,
    )
    assert res_unclipped["min_weight"] is None
    assert res_unclipped["max_weight"] is None
    assert res_unclipped["clipped_channels"] == []
    assert res_unclipped["clipping_altered_mean"] is False
    assert res_unclipped["weights_mean"] == pytest.approx(1.0, abs=1e-6)
    assert sum(res_unclipped["weights"]) / 4.0 == pytest.approx(1.0, abs=1e-6)

    # 2. Explicit min_weight=None with finite max_weight clipping
    res_max_only = derive_training_class_weights(
        records,
        normalize_mean=True,
        min_weight=None,
        max_weight=1.5,
    )
    assert res_max_only["min_weight"] is None
    assert res_max_only["max_weight"] == 1.5
    assert "MA" in res_max_only["clipped_channels"]
    assert res_max_only["weights_by_class"]["MA"] == pytest.approx(1.5)
    assert res_max_only["weights_by_class"]["HE"] < 1.0  # Not clipped from below
    assert res_max_only["clipping_altered_mean"] is True
    assert res_max_only["weights_mean"] < 1.0

    # 3. Normalization with both min and max clipping (0.5 and 1.5)
    res_both = derive_training_class_weights(
        records,
        normalize_mean=True,
        min_weight=0.5,
        max_weight=1.5,
    )
    assert res_both["weights_by_class"]["MA"] == pytest.approx(1.5)
    assert res_both["weights_by_class"]["HE"] == pytest.approx(0.5)
    assert set(res_both["clipped_channels"]) == {"MA", "HE"}

    # 4. Default minimum weight behavior (min_weight=1.0) with normalize_mean=True:
    # Clipping takes precedence over mean=1.0, clamping below-average channels (HE, EX) up to 1.0
    res_def_min = derive_training_class_weights(records, normalize_mean=True)
    assert res_def_min["min_weight"] == 1.0
    assert all(w >= 1.0 for w in res_def_min["weights"])
    assert res_def_min["clipping_altered_mean"] is True
    assert res_def_min["weights_mean"] > 1.0
    assert set(res_def_min["clipped_channels"]) == {"HE", "EX"}

    # 5. All-equal weights -> mean is 1.0 and clipping does not alter mean even with default min_weight=1.0
    equal_records = _make_synthetic_train_records()
    for r in equal_records:
        r["lesion_pixel_counts"] = {"MA": 100, "HE": 100, "EX": 100, "SE": 100}
    res_equal = derive_training_class_weights(equal_records, normalize_mean=True)
    assert res_equal["weights"] == [1.0, 1.0, 1.0, 1.0]
    assert res_equal["weights_mean"] == pytest.approx(1.0, abs=1e-6)
    assert res_equal["clipping_altered_mean"] is False
    assert res_equal["clipped_channels"] == []


def test_s3c03_invalid_and_contradictory_weight_derivation_parameters_rejected() -> None:
    """Verify S3C-03: contradictory or invalid parameter combinations raise ValueError."""
    records = _make_synthetic_train_records()

    # Contradictory: normalize_mean=True with min_weight > 1.0
    with pytest.raises(ValueError, match="Contradictory parameters.*min_weight"):
        derive_training_class_weights(records, normalize_mean=True, min_weight=1.5, max_weight=10.0)

    # Contradictory: normalize_mean=True with max_weight < 1.0
    with pytest.raises(ValueError, match="Contradictory parameters.*max_weight"):
        derive_training_class_weights(records, normalize_mean=True, min_weight=0.2, max_weight=0.8)

    # Contradictory: zero_positive_policy='min_weight' with min_weight=None
    with pytest.raises(ValueError, match="Contradictory parameters.*min_weight"):
        derive_training_class_weights(records, min_weight=None, zero_positive_policy="min_weight")

    # Contradictory: explicit zero_positive_policy='max_weight' with max_weight=None
    with pytest.raises(ValueError, match="Contradictory parameters.*max_weight"):
        derive_training_class_weights(records, max_weight=None, zero_positive_policy="max_weight")

    # Contradictory: max_weight < min_weight
    with pytest.raises(ValueError, match="max_weight.*must be >= min_weight"):
        derive_training_class_weights(records, min_weight=10.0, max_weight=5.0)

    # Invalid non-positive min_weight or max_weight
    with pytest.raises(ValueError, match="min_weight"):
        derive_training_class_weights(records, min_weight=0.0)
    with pytest.raises(ValueError, match="max_weight"):
        derive_training_class_weights(records, max_weight=-1.0)


# ---------------------------------------------------------------------------
# 8. Stage 3F Targeted Corrections (S3E-01 and S3E-02)
# ---------------------------------------------------------------------------


def test_s3e01_present_vs_all_channel_means_and_clipping_attribution() -> None:
    """Verify S3E-01: present-channel mean vs all-channel mean and accurate clipping_altered_mean."""
    # Exact-square record: fov_crop area = 17 * 100 = 1700
    # MA=100 (raw=4.0), HE=850 (raw=1.0), EX=340 (raw=2.0), SE=170 (raw=3.0)
    base_records = [
        {
            "dataset_name": "DDR",
            "sample_id": "S_EXACT",
            "split": "train",
            "image_dimensions": {"height": 17, "width": 100, "channels": 3},
            "fov_crop": {"y_min": 0, "y_max": 17, "x_min": 0, "x_max": 100},
            "lesion_pixel_counts": {"MA": 100, "HE": 850, "EX": 340, "SE": 170},
        }
    ]

    # 1. All four channels present, normalized without clipping
    res_all_present = derive_training_class_weights(
        base_records,
        normalize_mean=True,
        min_weight=None,
        max_weight=None,
    )
    assert res_all_present["channel_order"] == ["MA", "HE", "EX", "SE"]
    assert res_all_present["present_channels"] == ["MA", "HE", "EX", "SE"]
    assert res_all_present["zero_positive_channels"] == []
    assert res_all_present["weights"] == [1.6, 0.4, 0.8, 1.2]
    assert res_all_present["present_weights_mean"] == pytest.approx(1.0, abs=1e-6)
    assert res_all_present["present_channels_weights_mean"] == pytest.approx(1.0, abs=1e-6)
    assert res_all_present["weights_mean"] == pytest.approx(1.0, abs=1e-6)
    assert res_all_present["all_channels_weights_mean"] == pytest.approx(1.0, abs=1e-6)
    assert res_all_present["clipped_channels"] == []
    assert res_all_present["clipping_altered_mean"] is False

    # 2. One absent channel (SE=0) with zero_positive_policy="max_weight", normalize_mean=True, no clipping on present
    zero_se_records = copy.deepcopy(base_records)
    zero_se_records[0]["lesion_pixel_counts"]["SE"] = 0
    # Present raw weights: MA=4.0, HE=1.0, EX=2.0 -> mean = 7/3 -> normalized = [12/7, 3/7, 6/7] (sum = 3.0)
    # SE = max_weight = 10.0 -> all 4 channels sum = 13.0 -> all-channel mean = 3.25
    res_zero_max = derive_training_class_weights(
        zero_se_records,
        normalize_mean=True,
        min_weight=None,
        max_weight=10.0,
        zero_positive_policy="max_weight",
    )
    assert res_zero_max["present_channels"] == ["MA", "HE", "EX"]
    assert res_zero_max["zero_positive_channels"] == ["SE"]
    assert res_zero_max["present_weights_mean"] == pytest.approx(1.0, abs=1e-6)
    assert res_zero_max["weights_mean"] == pytest.approx(3.25, abs=1e-6)
    assert res_zero_max["all_channels_weights_mean"] == pytest.approx(3.25, abs=1e-6)
    assert res_zero_max["clipped_channels"] == []
    # All-channel mean differs from 1.0 due to SE's zero-positive policy, NOT clipping!
    assert res_zero_max["clipping_altered_mean"] is False

    # 3. One absent channel (SE=0) with zero_positive_policy="min_weight", normalize_mean=True, no clipping on present
    # With min_weight=0.2, present normalized weights [12/7=1.714, 3/7=0.429, 6/7=0.857] are all >= 0.2 (unclipped),
    # while SE gets min_weight=0.2 -> all-channel mean = (3.0 + 0.2) / 4 = 0.8
    res_zero_min = derive_training_class_weights(
        zero_se_records,
        normalize_mean=True,
        min_weight=0.2,
        max_weight=None,
        zero_positive_policy="min_weight",
    )
    assert res_zero_min["present_channels"] == ["MA", "HE", "EX"]
    assert res_zero_min["zero_positive_channels"] == ["SE"]
    assert res_zero_min["weights_by_class"]["SE"] == pytest.approx(0.2, abs=1e-6)
    assert res_zero_min["present_weights_mean"] == pytest.approx(1.0, abs=1e-6)
    assert res_zero_min["weights_mean"] == pytest.approx(0.8, abs=1e-6)
    assert res_zero_min["clipped_channels"] == []
    assert res_zero_min["clipping_altered_mean"] is False

    # 4. Zero-positive channel with normalize_mean=True AND genuine lower-bound clipping (min_weight=1.0 default)
    # Present normalized weights: MA=12/7 (1.714286), HE=3/7 (0.428571 -> clipped to 1.0), EX=6/7 (0.857143 -> clipped to 1.0)
    res_zero_clipped = derive_training_class_weights(
        zero_se_records,
        normalize_mean=True,
        min_weight=1.0,
        max_weight=10.0,
        zero_positive_policy="max_weight",
    )
    assert res_zero_clipped["clipped_channels"] == ["HE", "EX"]
    assert res_zero_clipped["clipping_altered_mean"] is True
    assert res_zero_clipped["pre_clip_present_channels_weights_mean"] == pytest.approx(1.0, abs=1e-6)
    assert res_zero_clipped["present_weights_mean"] == pytest.approx((12.0 / 7.0 + 2.0) / 3.0, abs=1e-5)
    assert res_zero_clipped["weights_mean"] == pytest.approx((12.0 / 7.0 + 2.0 + 10.0) / 4.0, abs=1e-5)


@pytest.mark.parametrize(
    "loss_type",
    ["focal_tversky", "bce_focal_tversky", "weighted_bce_focal_tversky"],
)
@pytest.mark.parametrize(
    ("gamma_val", "eps_val", "expected_gamma", "expected_eps"),
    [
        ("OMIT", "OMIT", 4.0 / 3.0, 1e-6),
        (None, "OMIT", 4.0 / 3.0, 1e-6),
        ("OMIT", None, 4.0 / 3.0, 1e-6),
        (None, None, 4.0 / 3.0, 1e-6),
        (0.75, 1e-5, 0.75, 1e-5),
    ],
)
def test_s3e02_omitted_and_null_gamma_and_eps_consistency(
    loss_type: str,
    gamma_val: Any,
    eps_val: Any,
    expected_gamma: float,
    expected_eps: float,
) -> None:
    """Verify S3E-02: omitted and explicit null gamma/eps validate and build with documented defaults."""
    cfg = load_yaml_config("configs/model.yaml")
    cfg["loss"]["type"] = loss_type
    if gamma_val == "OMIT":
        cfg["loss"].pop("gamma", None)
    else:
        cfg["loss"]["gamma"] = gamma_val

    if eps_val == "OMIT":
        cfg["loss"].pop("eps", None)
    else:
        cfg["loss"]["eps"] = eps_val

    validate_model_config(cfg)
    criterion = build_loss_from_config(cfg)
    summary = summarize_loss_configuration(cfg)

    assert criterion.gamma == pytest.approx(expected_gamma, abs=1e-7)  # type: ignore[union-attr]
    assert criterion.eps == pytest.approx(expected_eps, abs=1e-12)  # type: ignore[union-attr]
    assert summary["gamma"] == pytest.approx(expected_gamma, abs=1e-7)
    assert summary["eps"] == pytest.approx(expected_eps, abs=1e-12)

    logits = torch.randn(1, 4, 8, 8, requires_grad=True)
    targets = torch.randint(0, 2, (1, 4, 8, 8)).float()
    loss = criterion(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize(
    "loss_type",
    ["focal_tversky", "bce_focal_tversky", "weighted_bce_focal_tversky"],
)
@pytest.mark.parametrize("bad_val", [0.0, -0.5, float("nan"), float("inf"), True, "invalid"])
def test_s3e02_non_null_invalid_gamma_and_eps_rejected_by_validator_and_factory(
    loss_type: str,
    bad_val: Any,
) -> None:
    """Verify S3E-02: non-null invalid gamma and eps are rejected by both validate_model_config and build_loss_from_config."""
    base = load_yaml_config("configs/model.yaml")

    cfg_g = copy.deepcopy(base)
    cfg_g["loss"]["type"] = loss_type
    cfg_g["loss"]["gamma"] = bad_val
    with pytest.raises(ConfigValidationError, match="loss.gamma"):
        validate_model_config(cfg_g)
    with pytest.raises(ValueError, match="loss.gamma"):
        build_loss_from_config(cfg_g)

    cfg_e = copy.deepcopy(base)
    cfg_e["loss"]["type"] = loss_type
    cfg_e["loss"]["eps"] = bad_val
    with pytest.raises(ConfigValidationError, match="loss.eps"):
        validate_model_config(cfg_e)
    with pytest.raises(ValueError, match="loss.eps"):
        build_loss_from_config(cfg_e)


