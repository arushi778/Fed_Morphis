"""Focused unit and numerical verification tests for Tversky and BCETversky loss functions."""

import pytest
import torch
from src.models.losses import BCEDiceLoss, BCETverskyLoss, TverskyLoss, build_loss_from_config
from src.utils.config import ConfigValidationError, load_yaml_config, validate_model_config


def test_tversky_loss_ordinary_inputs() -> None:
    """Verify that TverskyLoss computes valid, finite loss on normal random logits."""
    criterion = TverskyLoss(alpha=0.3, beta=0.7, smooth=1e-5)
    logits = torch.randn(2, 4, 32, 32, requires_grad=True)
    targets = torch.randint(0, 2, (2, 4, 32, 32)).float()

    loss = criterion(logits, targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert 0.0 <= loss.item() <= 1.0

    # Verify gradients flow backward
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_tversky_loss_absent_channel() -> None:
    """Verify safe behavior when one or all channels have zero ground-truth positive pixels."""
    criterion = TverskyLoss(alpha=0.3, beta=0.7, smooth=1e-5)
    logits = torch.randn(2, 4, 32, 32, requires_grad=True)
    targets = torch.zeros(2, 4, 32, 32)  # Completely absent

    per_ch_loss = criterion.forward_per_channel(logits, targets)
    assert per_ch_loss.shape == (4,)
    assert torch.isfinite(per_ch_loss).all()
    assert (per_ch_loss >= 0.0).all()

    # When predictions are very confident negatives, absent loss should be near 0
    neg_logits = torch.full((2, 4, 32, 32), -10.0)
    per_ch_neg = criterion.forward_per_channel(neg_logits, targets)
    assert (per_ch_neg < 1e-4).all()


def test_tversky_false_negative_vs_false_positive_asymmetry() -> None:
    """Verify that false negatives (beta=0.7) are penalized strictly more than false positives (alpha=0.3)."""
    criterion = TverskyLoss(alpha=0.3, beta=0.7, smooth=1e-5, batch_tversky=True)

    # 10 true positive pixels
    # Case A: False Negative error (missing 5 positive pixels)
    # y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    # p = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] -> TP=5, FN=5, FP=0
    # Expected TI = 5 / (5 + 0.7*5) = 5 / 8.5 = 0.5882 -> Loss = 0.4118
    target_fn = torch.zeros(1, 1, 15, 1)
    target_fn[:, :, :10, :] = 1.0
    logits_fn = torch.full((1, 1, 15, 1), -100.0)
    logits_fn[:, :, :5, :] = 100.0  # 5 correct, 5 missed

    loss_fn = criterion(logits_fn, target_fn)

    # Case B: False Positive error (5 false positive background pixels)
    # y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    # p = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] -> TP=10, FN=0, FP=5
    # Expected TI = 10 / (10 + 0.3*5) = 10 / 11.5 = 0.8696 -> Loss = 0.1304
    target_fp = torch.zeros(1, 1, 15, 1)
    target_fp[:, :, :10, :] = 1.0
    logits_fp = torch.full((1, 1, 15, 1), -100.0)
    logits_fp[:, :, :15, :] = 100.0  # 10 correct + 5 false positives

    loss_fp = criterion(logits_fp, target_fp)

    # FN loss must be strictly greater than FP loss (specifically ~3x higher)
    assert loss_fn.item() > loss_fp.item()
    assert abs(loss_fn.item() - 0.41176) < 1e-3
    assert abs(loss_fp.item() - 0.13043) < 1e-3


def test_bce_tversky_loss_forward_tensors() -> None:
    """Verify that BCETverskyLoss returns the standard 5-tuple and is an instance of BCEDiceLoss."""
    criterion = BCETverskyLoss(
        bce_weight=0.5,
        tversky_weight=0.5,
        alpha=0.3,
        beta=0.7,
        smooth=1e-5,
    )
    assert isinstance(criterion, BCEDiceLoss)

    logits = torch.randn(2, 4, 32, 32, requires_grad=True)
    targets = torch.randint(0, 2, (2, 4, 32, 32)).float()

    total_loss, bce_loss, tversky_loss, bce_per_ch, tversky_per_ch = criterion.forward_tensors(logits, targets)

    assert torch.isfinite(total_loss)
    assert torch.isfinite(bce_loss)
    assert torch.isfinite(tversky_loss)
    assert bce_per_ch.shape == (4,)
    assert tversky_per_ch.shape == (4,)

    expected_total = 0.5 * bce_loss + 0.5 * tversky_loss
    assert torch.allclose(total_loss, expected_total, atol=1e-6)


def test_bce_tversky_yaml_configuration_parsing() -> None:
    """Verify that configs/model.yaml parses into BCETverskyLoss with correct parameters."""
    cfg = load_yaml_config("configs/model.yaml")
    validate_model_config(cfg)

    loss_cfg = cfg["loss"]
    assert loss_cfg["type"] == "bce_tversky"
    assert loss_cfg["bce_weight"] == 0.5
    assert loss_cfg["tversky_weight"] == 0.5
    assert loss_cfg["alpha"] == 0.3
    assert loss_cfg["beta"] == 0.7
    assert loss_cfg["pos_weight"] is None

    criterion = build_loss_from_config(cfg)
    assert isinstance(criterion, BCETverskyLoss)
    assert criterion.bce_weight == 0.5
    assert criterion.tversky_weight == 0.5
    assert criterion.alpha == 0.3
    assert criterion.beta == 0.7
    assert criterion.pos_weight is None


def test_mixed_precision_stability() -> None:
    """Verify numerical stability under autocast float16/bfloat16 precision."""
    criterion = BCETverskyLoss(alpha=0.3, beta=0.7)
    logits = torch.randn(2, 4, 32, 32, device="cpu", dtype=torch.bfloat16, requires_grad=True)
    targets = torch.randint(0, 2, (2, 4, 32, 32), device="cpu").float()

    loss = criterion(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()

