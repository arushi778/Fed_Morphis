"""Unit tests for validation-based channel-specific threshold optimization."""

from __future__ import annotations

import torch
import pytest

from scripts.optimize_thresholds import (
    evaluate_channel_at_threshold,
    run_threshold_grid_search,
    CANDIDATE_THRESHOLDS,
)
from src.data.masks import CANONICAL_LESION_CHANNELS


def test_evaluate_channel_at_threshold_perfect_prediction():
    """Verify metrics when prediction perfectly matches ground truth."""
    targets = torch.zeros(100, 100, dtype=torch.uint8)
    targets[10:20, 10:20] = 1  # 100 positive pixels

    probs = torch.zeros(100, 100, dtype=torch.float32)
    probs[10:20, 10:20] = 0.9  # well above 0.50

    res = evaluate_channel_at_threshold(probs, targets, threshold=0.50)
    assert res["dice"] == pytest.approx(1.0)
    assert res["iou"] == pytest.approx(1.0)
    assert res["precision"] == pytest.approx(1.0)
    assert res["recall"] == pytest.approx(1.0)
    assert res["tp"] == 100
    assert res["fp"] == 0
    assert res["fn"] == 0


def test_evaluate_channel_threshold_sensitivity():
    """Verify that lowering threshold captures more positive pixels and increases recall."""
    targets = torch.zeros(50, 50, dtype=torch.uint8)
    targets[5:15, 5:15] = 1  # 100 positive pixels

    probs = torch.zeros(50, 50, dtype=torch.float32)
    probs[5:10, 5:15] = 0.6  # 50 pixels high confidence
    probs[10:15, 5:15] = 0.3  # 50 pixels faint confidence

    res_high = evaluate_channel_at_threshold(probs, targets, threshold=0.50)
    res_low = evaluate_channel_at_threshold(probs, targets, threshold=0.25)

    assert res_high["recall"] == pytest.approx(0.50)
    assert res_low["recall"] == pytest.approx(1.00)
    assert res_low["dice"] > res_high["dice"]


def test_run_threshold_grid_search_synthetic():
    """Verify synthetic 2-client grid search executes cleanly and selects optimal thresholds."""
    torch.manual_seed(42)
    num_samples_c0 = 10
    num_samples_c1 = 2
    h, w = 32, 32

    # Client 0 synthetic data
    c0_probs = torch.rand(num_samples_c0, 4, h, w) * 0.4  # mostly low probs
    c0_targets = torch.zeros(num_samples_c0, 4, h, w, dtype=torch.uint8)
    c0_probs[:, 0, 5:10, 5:10] = 0.35
    c0_targets[:, 0, 5:10, 5:10] = 1

    # Client 1 synthetic data
    c1_probs = torch.rand(num_samples_c1, 4, h, w) * 0.4
    c1_targets = torch.zeros(num_samples_c1, 4, h, w, dtype=torch.uint8)
    c1_probs[:, 0, 5:10, 5:10] = 0.35
    c1_targets[:, 0, 5:10, 5:10] = 1

    client_data = {
        "client_0": {"probs": c0_probs, "targets": c0_targets, "dataset": "DDR"},
        "client_1": {"probs": c1_probs, "targets": c1_targets, "dataset": "IDRiD"},
    }

    grid = {
        "MA": [0.20, 0.30, 0.50],
        "HE": [0.30, 0.50],
        "EX": [0.30, 0.50],
        "SE": [0.30, 0.50],
    }

    results = run_threshold_grid_search(client_data, candidate_thresholds=grid)
    assert "baseline_uniform" in results
    assert "channel_optimized" in results
    assert "stability_audit" in results

    opt_th = results["channel_optimized"]["thresholds"]
    assert opt_th["MA"] == 0.30  # captures the 0.35 pixels where 0.50 missed them
    assert results["channel_optimized"]["macro_metrics"]["macro_dice"] >= results["baseline_uniform"]["macro_metrics"]["macro_dice"]

