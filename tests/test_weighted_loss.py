"""Focused unit tests for channel-specific weighted BCE + Soft Dice loss."""

import torch
import pytest
from src.models.losses import BCEDiceLoss, build_loss_from_config
from src.utils.config import ConfigValidationError, load_yaml_config, validate_model_config

def test_pos_weight_yaml_parsing_and_shape():
    """Verify that pos_weight is parsed and registered as a 4D tensor (1, 4, 1, 1)."""
    cfg = load_yaml_config("configs/model.yaml")
    cfg["loss"] = {
        "type": "bce_dice",
        "bce_weight": 0.5,
        "dice_weight": 0.5,
        "smooth": 1e-5,
        "pos_weight": [12.0, 4.0, 4.0, 8.0],
    }
    validate_model_config(cfg)
    
    assert "pos_weight" in cfg["loss"]
    assert cfg["loss"]["pos_weight"] == [12.0, 4.0, 4.0, 8.0]
    
    criterion = build_loss_from_config(cfg)
    assert criterion.pos_weight is not None
    assert criterion.pos_weight.shape == (1, 4, 1, 1)
    assert criterion.pos_weight.squeeze().tolist() == [12.0, 4.0, 4.0, 8.0]

def test_pos_weight_applied_to_bce_only():
    """Verify that pos_weight modifies the BCE loss component but leaves Dice loss completely unchanged."""
    torch.manual_seed(42)
    logits = torch.randn(2, 4, 32, 32)
    targets = torch.randint(0, 2, (2, 4, 32, 32)).float()
    
    unweighted = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5, pos_weight=None)
    weighted = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5, pos_weight=[12.0, 4.0, 4.0, 8.0])
    
    tot_unw, bce_unw, dice_unw, bce_ch_unw, dice_ch_unw = unweighted.forward_tensors(logits, targets)
    tot_w, bce_w, dice_w, bce_ch_w, dice_ch_w = weighted.forward_tensors(logits, targets)
    
    # 1. Dice loss must be completely identical
    assert torch.allclose(dice_unw, dice_w, atol=1e-7)
    assert torch.allclose(dice_ch_unw, dice_ch_w, atol=1e-7)
    
    # 2. BCE loss must be higher due to positive class weighting
    assert bce_w > bce_unw
    
    # 3. Channel-specific scaling: MA (weight 12.0) must have the highest relative boost
    assert bce_ch_w[0] > bce_ch_unw[0]
    assert bce_ch_w[3] > bce_ch_unw[3] # SE (weight 8.0)

def test_pos_weight_exact_numerical_scaling():
    """Verify exact numerical formulation of weighted binary cross-entropy."""
    criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=0.0, pos_weight=[12.0, 4.0, 4.0, 8.0])
    
    # Logit = 0.0 -> sigmoid = 0.5 -> log(0.5) = -0.69314718
    logits = torch.zeros(1, 4, 1, 1)
    targets = torch.ones(1, 4, 1, 1)
    
    _, bce_loss, _, bce_per_ch, _ = criterion.forward_tensors(logits, targets)
    
    expected = torch.tensor([12.0, 4.0, 4.0, 8.0]) * 0.69314718
    assert torch.allclose(bce_per_ch, expected, atol=1e-5)
    
    # For background zeros (targets=0), BCE is unweighted (-log(1 - 0.5) = 0.69314718)
    targets_bg = torch.zeros(1, 4, 1, 1)
    _, _, _, bce_bg_ch, _ = criterion.forward_tensors(logits, targets_bg)
    expected_bg = torch.full((4,), 0.69314718)
    assert torch.allclose(bce_bg_ch, expected_bg, atol=1e-5)

def test_invalid_pos_weight_rejected():
    """Verify that validation rejects invalid pos_weight formats (non-length-4 or non-positive)."""
    cfg = load_yaml_config("configs/model.yaml")
    
    # Case 1: Wrong length
    cfg["loss"]["pos_weight"] = [1.0, 2.0]
    with pytest.raises(ConfigValidationError, match="sequence of 4 positive numbers"):
        validate_model_config(cfg)
        
    # Case 2: Negative weight
    cfg["loss"]["pos_weight"] = [1.0, 2.0, -3.0, 4.0]
    with pytest.raises(ConfigValidationError, match="must be positive numbers > 0"):
        validate_model_config(cfg)

