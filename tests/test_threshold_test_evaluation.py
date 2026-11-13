"""Unit tests for final held-out test evaluation with frozen thresholds."""

from __future__ import annotations

import pytest
import torch

from scripts.evaluate_threshold_test import (
    evaluate_dataset_at_thresholds,
    aggregate_dataset_results,
    FROZEN_OPTIMIZED_THRESHOLDS,
    UNIFORM_BASELINE_THRESHOLDS,
)


def test_evaluate_dataset_at_thresholds_perfect_match():
    """Verify evaluation logic when prediction probabilities exceed thresholds."""
    h, w = 16, 16
    probs = torch.ones(2, 4, h, w, dtype=torch.float32) * 0.8
    targets = torch.ones(2, 4, h, w, dtype=torch.uint8)

    res = evaluate_dataset_at_thresholds(probs, targets, FROZEN_OPTIMIZED_THRESHOLDS)
    assert "macro" in res
    assert "per_class" in res
    assert res["macro"]["dice"] == pytest.approx(1.0)
    assert res["macro"]["iou"] == pytest.approx(1.0)


def test_evaluate_dataset_at_thresholds_varying_thresholds():
    """Verify that lower thresholds capture more predictions than uniform 0.50."""
    h, w = 16, 16
    probs = torch.full((1, 4, h, w), 0.30, dtype=torch.float32)
    targets = torch.ones(1, 4, h, w, dtype=torch.uint8)

    # MA threshold is 0.25 (should be positive), others are >= 0.35 (should be negative)
    res = evaluate_dataset_at_thresholds(probs, targets, FROZEN_OPTIMIZED_THRESHOLDS)
    assert res["per_class"]["MA"]["recall"] == pytest.approx(1.0)
    assert res["per_class"]["HE"]["recall"] == pytest.approx(0.0)
    assert res["per_class"]["EX"]["recall"] == pytest.approx(0.0)
    assert res["per_class"]["SE"]["recall"] == pytest.approx(0.0)


def test_aggregate_dataset_results_weighting():
    """Verify exact sample weighting between two datasets."""
    ds_res = {
        "DDR": {
            "num_samples": 225,
            "loss": 0.20,
            "bce_loss": 0.01,
            "dice_loss": 0.40,
            "eval_metrics": {
                "macro": {"dice": 0.50, "iou": 0.35, "precision": 0.60, "recall": 0.50, "sensitivity": 0.50, "specificity": 0.99},
                "per_class": {
                    ch: {"tp": 100, "fp": 10, "fn": 20, "tn": 1000, "dice": 0.50, "iou": 0.35, "precision": 0.60, "recall": 0.50, "sensitivity": 0.50, "specificity": 0.99}
                    for ch in ["MA", "HE", "EX", "SE"]
                },
            },
        },
        "IDRiD": {
            "num_samples": 27,
            "loss": 0.30,
            "bce_loss": 0.02,
            "dice_loss": 0.50,
            "eval_metrics": {
                "macro": {"dice": 0.40, "iou": 0.25, "precision": 0.50, "recall": 0.40, "sensitivity": 0.40, "specificity": 0.98},
                "per_class": {
                    ch: {"tp": 20, "fp": 5, "fn": 10, "tn": 200, "dice": 0.40, "iou": 0.25, "precision": 0.50, "recall": 0.40, "sensitivity": 0.40, "specificity": 0.98}
                    for ch in ["MA", "HE", "EX", "SE"]
                },
            },
        },
    }

    agg = aggregate_dataset_results(ds_res, FROZEN_OPTIMIZED_THRESHOLDS)
    expected_dice = (225 * 0.50 + 27 * 0.40) / 252
    assert agg["sample_weighted"]["macro"]["dice"] == pytest.approx(expected_dice)
    assert agg["total_samples"] == 252

