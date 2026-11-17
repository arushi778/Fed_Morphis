"""Comprehensive test suite for AUPR and Average Precision (AP) evaluation.

Covers all Stage 2 requirements:
1. Perfect predictions (AP = 1.0, Trapezoidal AUPR = 1.0).
2. Completely incorrect (inverted) predictions.
3. Random predictions converging near baseline prevalence.
4. Highly imbalanced binary masks (e.g. 2 in 20,000 pixels).
5. Lesion class with no positive pixels (target_present=False, ap=None, aupr=None).
6. Lesion class with no predicted positives (ap = prevalence).
7. Four-channel input and strict canonical channel order [MA, HE, EX, SE].
8. Macro metric calculation (macro vs. macro_present_only).
9. Correct accumulation across multiple validation images.
10. FOV masking and prediction/target shape consistency.
11. Strict split verification (rejecting test split records).
12. Compatibility with existing metrics and evaluator.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import auc, average_precision_score, precision_recall_curve
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data.masks import CANONICAL_LESION_CHANNELS
from src.engine.evaluator import (
    SplitMismatchError,
    evaluate_aupr_dataset,
    evaluate_segmentation_dataloader,
)
from src.models.aupr import (
    AUPRTracker,
    CANONICAL_LESION_ORDER,
    DEFAULT_FROZEN_THRESHOLDS,
    DEFAULT_OPERATING_THRESHOLDS,
    PROTOCOL_NAME,
    ChannelPRResult,
    compute_channel_pr_curve,
    compute_multilabel_aupr,
)
from src.models.metrics import SegmentationMetricTracker, compute_segmentation_metrics


# ==============================================================================
# Requirement 1: Perfect Predictions
# ==============================================================================
def test_aupr_perfect_predictions() -> None:
    """Verify that perfect predictions achieve exact AP = 1.0 and Trapezoidal AUPR = 1.0."""
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    # Perfectly separated scores
    y_score = np.array([0.05, 0.10, 0.15, 0.85, 0.90, 0.95], dtype=np.float32)

    res = compute_channel_pr_curve(y_true, y_score, channel_name="MA")

    assert res.target_present is True
    assert res.positive_pixels == 3
    assert res.negative_pixels == 3
    assert res.total_pixels == 6
    assert res.ap == pytest.approx(1.0, abs=1e-5)
    assert res.aupr_trapezoidal == pytest.approx(1.0, abs=1e-5)
    assert res.empty_handling == "standard_non_empty"

    # Operating threshold at 0.50 should have precision=1.0, recall=1.0, f1=1.0
    th_50 = res.operating_thresholds["0.50"]
    assert th_50["precision"] == pytest.approx(1.0)
    assert th_50["recall"] == pytest.approx(1.0)
    assert th_50["f1"] == pytest.approx(1.0)


# ==============================================================================
# Requirement 2: Completely Incorrect (Inverted) Predictions
# ==============================================================================
def test_aupr_completely_incorrect_predictions() -> None:
    """Verify that inverted predictions produce low AP and Trapezoidal AUPR matching exact formula."""
    y_true = np.array([0, 0, 1, 1], dtype=np.int32)
    # Negatives get highest scores, positives get lowest scores
    y_score = np.array([0.95, 0.90, 0.20, 0.10], dtype=np.float32)

    res = compute_channel_pr_curve(y_true, y_score, channel_name="HE")

    # Independent scikit-learn calculation
    expected_ap = float(average_precision_score(y_true, y_score))
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    expected_trapz = float(auc(rec[::-1], prec[::-1]))

    assert res.target_present is True
    assert res.ap == pytest.approx(expected_ap, abs=1e-5)
    assert res.aupr_trapezoidal == pytest.approx(expected_trapz, abs=1e-5)
    # Inverted 2/4 targets: AP is ~0.4167, Trapz is ~0.2917
    assert res.ap < 0.50
    assert res.aupr_trapezoidal < 0.50
    assert res.ap != res.aupr_trapezoidal


# ==============================================================================
# Requirement 3: Random Predictions
# ==============================================================================
def test_aupr_random_predictions_converge_near_prevalence() -> None:
    """Verify that random predictions on a large array produce AP near base prevalence."""
    np.random.seed(42)
    n_samples = 20000
    prevalence = 0.05
    n_pos = int(n_samples * prevalence)  # 1000

    y_true = np.zeros(n_samples, dtype=np.int32)
    pos_idx = np.random.choice(n_samples, size=n_pos, replace=False)
    y_true[pos_idx] = 1

    # Uniform random scores
    y_score = np.random.uniform(0.0, 1.0, size=n_samples).astype(np.float32)

    res = compute_channel_pr_curve(y_true, y_score, channel_name="EX")

    assert res.target_present is True
    assert res.prevalence == pytest.approx(prevalence, abs=1e-4)
    # For uninformative random scores, AP should be close to base prevalence (within 0.03)
    assert abs(res.ap - prevalence) < 0.03


# ==============================================================================
# Requirement 4: Highly Imbalanced Binary Masks
# ==============================================================================
def test_aupr_highly_imbalanced_masks() -> None:
    """Verify behavior on extreme class imbalance typical of microaneurysms (< 0.01% lesion pixels)."""
    n_total = 20000
    n_pos = 2  # Only 2 positive lesion pixels in 20,000 pixels (0.01% prevalence)
    prevalence = n_pos / n_total

    y_true = np.zeros(n_total, dtype=np.int32)
    y_true[42] = 1
    y_true[1337] = 1

    # Case A: Model ranks both positive pixels at the very top
    y_score_top = np.full(n_total, 0.01, dtype=np.float32)
    y_score_top[42] = 0.99
    y_score_top[1337] = 0.98

    res_top = compute_channel_pr_curve(y_true, y_score_top, channel_name="MA")
    assert res_top.ap == pytest.approx(1.0, abs=1e-5)
    assert res_top.aupr_trapezoidal == pytest.approx(1.0, abs=1e-5)

    # Case B: Model ranks 2 false positives ahead of the 2 true positives
    y_score_mixed = np.full(n_total, 0.01, dtype=np.float32)
    y_score_mixed[0] = 0.99   # False positive
    y_score_mixed[1] = 0.98   # False positive
    y_score_mixed[42] = 0.90  # True positive (rank 3, precision = 1/3)
    y_score_mixed[1337] = 0.80 # True positive (rank 4, precision = 2/4 = 0.5)

    res_mixed = compute_channel_pr_curve(y_true, y_score_mixed, channel_name="MA")
    # Expected AP: (1/3 * 0.5) + (2/4 * 0.5) = 1/6 + 1/4 = 5/12 ~= 0.416667
    assert res_mixed.ap == pytest.approx(5.0 / 12.0, abs=1e-4)


# ==============================================================================
# Requirement 5: Lesion Class with No Positive Pixels
# ==============================================================================
def test_aupr_absent_target_class_graceful_handling() -> None:
    """Verify that a channel with zero positive pixels sets target_present=False and ap=None without crashing."""
    y_true = np.zeros(500, dtype=np.int32)
    y_score = np.random.uniform(0.0, 0.8, size=500).astype(np.float32)

    res = compute_channel_pr_curve(y_true, y_score, channel_name="SE")

    assert res.target_present is False
    assert res.positive_pixels == 0
    assert res.negative_pixels == 500
    assert res.prevalence == 0.0
    assert res.ap is None
    assert res.aupr_trapezoidal is None
    assert res.empty_handling == "absent_target_class"
    assert res.curve_summary is None
    assert res.raw_curve_points == 0

    # Operating thresholds should be safely populated without division by zero
    for th_key, th_dict in res.operating_thresholds.items():
        assert th_dict["recall"] == 0.0
        assert th_dict["tp"] == 0
        assert th_dict["fn"] == 0


# ==============================================================================
# Requirement 6: Lesion Class with No Predicted Positives
# ==============================================================================
def test_aupr_no_predicted_positives_flat_predictions() -> None:
    """Verify that when predictions are all zero, AP equals base prevalence."""
    n_pos = 10
    n_neg = 90
    prevalence = n_pos / (n_pos + n_neg)  # 0.10

    y_true = np.zeros(100, dtype=np.int32)
    y_true[:n_pos] = 1

    # Model predicts flat 0.0 for all pixels
    y_score = np.zeros(100, dtype=np.float32)

    res = compute_channel_pr_curve(y_true, y_score, channel_name="EX")

    assert res.target_present is True
    # For completely flat uninformative predictions, AP is theoretically equal to prevalence
    assert res.ap == pytest.approx(prevalence, abs=1e-4)
    assert res.empty_handling == "all_zero_prediction"

    # Non-zero constant flat predictions use standard_non_empty while still yielding AP == prevalence
    res_const = compute_channel_pr_curve(
        y_true, np.full(100, 0.42, dtype=np.float32), channel_name="EX"
    )
    assert res_const.ap == pytest.approx(prevalence, abs=1e-4)
    assert res_const.empty_handling == "standard_non_empty"


# ==============================================================================
# Requirement 7: Four-Channel Input and Canonical Channel Order
# ==============================================================================
def test_aupr_four_channel_canonical_order() -> None:
    """Verify multi-channel evaluation strictly preserves [MA, HE, EX, SE] order with distinct signals."""
    torch.manual_seed(42)
    # Shape: (B=2, C=4, H=16, W=16)
    b, c, h, w = 2, 4, 16, 16
    targets = torch.zeros(b, c, h, w, dtype=torch.float32)
    predictions = torch.full((b, c, h, w), 0.05, dtype=torch.float32)

    # MA (channel 0): Perfect predictions
    targets[:, 0, 2:6, 2:6] = 1.0
    predictions[:, 0, 2:6, 2:6] = 0.95

    # HE (channel 1): Moderate predictions
    targets[:, 1, 4:8, 4:8] = 1.0
    predictions[:, 1, 4:8, 4:8] = 0.60
    predictions[:, 1, 8:12, 8:12] = 0.55  # Some false positives

    # EX (channel 2): Inverted predictions
    targets[:, 2, 0:4, 0:4] = 1.0
    predictions[:, 2, 0:4, 0:4] = 0.01
    predictions[:, 2, 10:14, 10:14] = 0.99

    # SE (channel 3): Completely absent target class
    targets[:, 3, :, :] = 0.0
    predictions[:, 3, :, :] = 0.05

    summary = compute_multilabel_aupr(
        predictions=predictions,
        targets=targets,
        from_logits=False,
    )

    assert summary["protocol"] == PROTOCOL_NAME
    assert summary["channel_order"] == ["MA", "HE", "EX", "SE"]
    assert summary["present_classes"] == ["MA", "HE", "EX"]
    assert summary["absent_classes"] == ["SE"]

    per_class = summary["per_class"]
    assert per_class["MA"]["ap"] == pytest.approx(1.0, abs=1e-4)
    assert per_class["HE"]["ap"] > per_class["EX"]["ap"]
    assert per_class["SE"]["ap"] is None
    assert per_class["SE"]["target_present"] is False


# ==============================================================================
# Requirement 8: Macro Metric Calculation
# ==============================================================================
def test_aupr_macro_metric_calculation() -> None:
    """Verify macro vs. macro_present_only handling when some classes are absent."""
    targets = torch.zeros(1, 4, 10, 10)
    preds = torch.full((1, 4, 10, 10), 0.1)

    # Channel 0 (MA): AP = 1.0
    targets[0, 0, 0:2, 0:2] = 1.0
    preds[0, 0, 0:2, 0:2] = 0.9

    # Channel 1 (HE): AP = 1.0
    targets[0, 1, 0:2, 0:2] = 1.0
    preds[0, 1, 0:2, 0:2] = 0.9

    # Channel 2 (EX): AP = 1.0
    targets[0, 2, 0:2, 0:2] = 1.0
    preds[0, 2, 0:2, 0:2] = 0.9

    # Channel 3 (SE): Absent (0 positives)
    targets[0, 3, :, :] = 0.0
    preds[0, 3, :, :] = 0.1

    summary = compute_multilabel_aupr(preds, targets, from_logits=False)

    # When 1 class is absent, macro across all 4 is None (cannot average undefined)
    assert summary["macro"]["ap"] is None
    assert summary["macro"]["aupr_trapezoidal"] is None

    # macro_present_only should be the mean of the 3 present classes (all 1.0)
    assert summary["macro_present_only"]["ap"] == pytest.approx(1.0, abs=1e-5)
    assert summary["macro_present_only"]["aupr_trapezoidal"] == pytest.approx(1.0, abs=1e-5)

    # If all 4 classes are present:
    targets[0, 3, 0:2, 0:2] = 1.0
    preds[0, 3, 0:2, 0:2] = 0.9
    summary_all = compute_multilabel_aupr(preds, targets, from_logits=False)
    assert summary_all["macro"]["ap"] == pytest.approx(1.0, abs=1e-5)
    assert summary_all["macro_present_only"]["ap"] == pytest.approx(1.0, abs=1e-5)


# ==============================================================================
# Requirement 9: Correct Handling of Multiple Validation Images
# ==============================================================================
def test_aupr_tracker_multiple_images_identical_to_pooled() -> None:
    """Verify that accumulating images sequentially produces identical results to pooled tensors."""
    np.random.seed(123)
    tracker = AUPRTracker(from_logits=False, store_per_image=True)

    n_images = 4
    c, h, w = 4, 16, 16

    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for i in range(n_images):
        p_i = np.random.uniform(0.0, 1.0, (c, h, w)).astype(np.float32)
        t_i = (np.random.uniform(0.0, 1.0, (c, h, w)) > 0.85).astype(np.float32)

        tracker.update(p_i, t_i, sample_ids=[f"val_{i}"])
        all_preds.append(p_i)
        all_targets.append(t_i)

    tracker_res = tracker.compute()

    # Compute directly on pooled arrays
    pooled_preds = np.stack(all_preds, axis=0)  # (N, C, H, W)
    pooled_targets = np.stack(all_targets, axis=0)
    direct_res = compute_multilabel_aupr(pooled_preds, pooled_targets, from_logits=False)

    assert tracker_res["num_samples"] == n_images
    assert len(tracker.per_image_results) == n_images

    for ch in CANONICAL_LESION_ORDER:
        tr_ap = tracker_res["global_pooled"]["per_class"][ch]["ap"]
        dir_ap = direct_res["per_class"][ch]["ap"]
        assert tr_ap == pytest.approx(dir_ap, abs=1e-5)


# ==============================================================================
# Requirement 10: FOV Masking and Shape Consistency
# ==============================================================================
def test_aupr_fov_masking_and_shape_validation() -> None:
    """Verify FOV spatial masking excludes peripheral background and shape errors raise ValueError."""
    h, w = 20, 20
    y_true = np.zeros((h, w), dtype=np.int32)
    y_score = np.zeros((h, w), dtype=np.float32)

    # Place positive lesion inside center
    y_true[10, 10] = 1
    y_score[10, 10] = 0.9

    # Place false positive in black corner outside FOV
    y_score[0, 0] = 0.99
    y_true[0, 0] = 0

    # Circular FOV mask: center is 1, corners are 0
    fov_mask = np.zeros((h, w), dtype=bool)
    fov_mask[5:15, 5:15] = True

    # Without FOV mask: the false positive outside FOV lowers AP
    res_no_fov = compute_channel_pr_curve(y_true, y_score, mask=None)
    assert res_no_fov.ap < 1.0  # False positive at (0, 0) ranked higher

    # With FOV mask: false positive at (0, 0) is ignored!
    res_with_fov = compute_channel_pr_curve(y_true, y_score, mask=fov_mask)
    assert res_with_fov.ap == pytest.approx(1.0, abs=1e-5)

    # Shape mismatch should raise ValueError
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_channel_pr_curve(np.zeros((10, 10)), np.zeros((10, 12)))


# ==============================================================================
# Requirement 11: No Test-Split Records in Validation Evaluator
# ==============================================================================
def test_aupr_evaluator_rejects_test_split() -> None:
    """Verify evaluate_aupr_dataset strictly rejects split='test' and test records."""
    dummy_model = nn.Identity()

    # Attempting to call evaluate_aupr_dataset with expected_split='test' must raise SplitMismatchError
    with pytest.raises(SplitMismatchError, match="restricted to validation/training"):
        evaluate_aupr_dataset(
            model=dummy_model,
            dataset=[],
            expected_split="test",
        )

    # Mock dataset with a test record
    class MockDatasetWithTestRecord(Dataset):
        def __init__(self):
            self.records = [
                {"sample_id": "test_01", "split": "test"},
            ]
        def __len__(self):
            return 1
        def __getitem__(self, idx):
            return torch.zeros(3, 16, 16), torch.zeros(4, 16, 16), {"sample_id": "test_01", "split": "test"}

    with pytest.raises(SplitMismatchError, match="Split mismatch"):
        evaluate_aupr_dataset(
            model=dummy_model,
            dataset=MockDatasetWithTestRecord(),
            expected_split="valid",
            use_sliding_window=False,
        )


# ==============================================================================
# Requirement 12: Compatibility with Existing Metrics and Evaluator
# ==============================================================================
def test_aupr_compatibility_with_existing_evaluator() -> None:
    """Verify evaluate_segmentation_dataloader preserves existing metrics when compute_aupr=True."""
    class DummySegmentationModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Predict high logits for positive regions
            b = x.shape[0]
            out = torch.full((b, 4, 16, 16), -5.0)
            out[:, :, 4:8, 4:8] = 5.0
            return out

    model = DummySegmentationModel()

    class SyntheticDataset(Dataset):
        def __len__(self) -> int:
            return 2
        def __getitem__(self, idx: int):
            img = torch.zeros(3, 16, 16)
            mask = torch.zeros(4, 16, 16)
            mask[:, 4:8, 4:8] = 1.0
            return img, mask, {"sample_id": f"syn_{idx}", "split": "valid"}

    dl = DataLoader(SyntheticDataset(), batch_size=2)

    # Evaluation with compute_aupr=True
    res = evaluate_segmentation_dataloader(
        model=model,
        dataloader=dl,
        device="cpu",
        expected_split="valid",
        compute_aupr=True,
    )

    # Verify standard legacy fields are intact
    assert "dice_macro" in res
    assert "iou_macro" in res
    assert "precision_macro" in res
    assert "recall_macro" in res
    assert "sensitivity_macro" in res
    assert "specificity_macro" in res
    assert res["dice_macro"] == pytest.approx(1.0, abs=1e-4)

    # Verify new AUPR fields are present and accurate
    assert "aupr" in res
    assert "ap_macro" in res
    assert "aupr_macro" in res
    assert res["ap_macro"] == pytest.approx(1.0, abs=1e-4)
    assert res["aupr_macro"] == pytest.approx(1.0, abs=1e-4)
    for ch in CANONICAL_LESION_ORDER:
        assert res["per_class"][ch]["ap"] == pytest.approx(1.0, abs=1e-4)
        assert res["per_class"][ch]["aupr_trapezoidal"] == pytest.approx(1.0, abs=1e-4)


# ==============================================================================
# Stage 2B Regression Tests (F-01 through F-07)
# ==============================================================================


def test_stage2b_sliding_window_evaluator_both_dataset_interfaces() -> None:
    """Verify evaluate_aupr_dataset(..., use_sliding_window=True) works with both dataset interfaces (F-01)."""
    from torch.utils.data import Subset
    from src.engine.sliding_window import SlidingWindowInferencer

    class PositiveCenterModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, _, h, w = x.shape
            out = torch.full((b, 4, h, w), -6.0, dtype=torch.float32)
            out[:, :, 8:24, 8:24] = 6.0
            return out

    model = PositiveCenterModel()
    inferencer = SlidingWindowInferencer(
        model=model,
        patch_size=(32, 32),
        overlap=0.5,
        device="cpu",
        batch_size=2,
    )

    # Interface A: Dataset exposing _load_and_crop_raw_index and .records
    class RawIndexDataset(Dataset):
        def __init__(self) -> None:
            self.records = [
                {"sample_id": "raw_val_01", "split": "valid"},
                {"sample_id": "raw_val_02", "split": "valid"},
            ]

        def __len__(self) -> int:
            return len(self.records)

        def _load_and_crop_raw_index(self, idx: int):
            img = np.full((32, 32, 3), 120, dtype=np.uint8)
            mask = np.zeros((32, 32, 4), dtype=np.uint8)
            mask[8:24, 8:24, :] = 1
            meta = dict(self.records[idx])
            return img, mask, meta

    res_raw = evaluate_aupr_dataset(
        model=model,
        dataset=RawIndexDataset(),
        device="cpu",
        use_sliding_window=True,
        sliding_window_inferencer=inferencer,
        expected_split="valid",
    )
    assert res_raw["status"] == "completed"
    assert res_raw["use_sliding_window"] is True
    assert res_raw["num_samples"] == 2
    assert res_raw["ap_macro"] == pytest.approx(1.0, abs=1e-4)
    assert res_raw["aupr_macro"] == pytest.approx(1.0, abs=1e-4)

    # Also verify Subset wrapping RawIndexDataset works in sliding-window mode
    res_subset = evaluate_aupr_dataset(
        model=model,
        dataset=Subset(RawIndexDataset(), [0]),
        device="cpu",
        use_sliding_window=True,
        sliding_window_inferencer=inferencer,
        expected_split="valid",
    )
    assert res_subset["num_samples"] == 1
    assert res_subset["sample_ids"] == ["raw_val_01"]
    assert res_subset["ap_macro"] == pytest.approx(1.0, abs=1e-4)

    # Interface B: Standard PyTorch Dataset returning (image_tensor, mask_tensor, metadata)
    class StandardTupleDataset(Dataset):
        def __init__(self) -> None:
            self.sample_metadata = [
                {"sample_id": "std_val_0", "split": "valid"},
                {"sample_id": "std_val_1", "split": "valid"},
            ]

        def __len__(self) -> int:
            return 2

        def __getitem__(self, idx: int):
            img = torch.full((3, 32, 32), 0.5, dtype=torch.float32)
            mask = torch.zeros((4, 32, 32), dtype=torch.float32)
            mask[:, 8:24, 8:24] = 1.0
            return img, mask, {"sample_id": f"std_val_{idx}", "split": "valid"}

    res_std = evaluate_aupr_dataset(
        model=model,
        dataset=StandardTupleDataset(),
        device="cpu",
        use_sliding_window=True,
        sliding_window_inferencer=inferencer,
        expected_split="valid",
    )
    assert res_std["status"] == "completed"
    assert res_std["num_samples"] == 2
    assert res_std["sample_ids"] == ["std_val_0", "std_val_1"]
    assert res_std["ap_macro"] == pytest.approx(1.0, abs=1e-4)


def test_stage2b_unconditional_test_split_protection_and_upfront_validation() -> None:
    """Verify F-02: reject expected_split=None, Subset/metadata test records, and validate before inference."""
    from torch.utils.data import Subset

    class CallCountingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.forward_calls = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.forward_calls += 1
            return torch.zeros((x.shape[0], 4, x.shape[2], x.shape[3]), dtype=torch.float32)

    # 1. Reject expected_split=None
    counting_model = CallCountingModel()
    with pytest.raises(SplitMismatchError, match="expected_split=None is prohibited"):
        evaluate_aupr_dataset(
            model=counting_model,
            dataset=[],
            expected_split=None,
        )
    assert counting_model.forward_calls == 0

    # 2. Reject test record even when caller passes expected_split='train', BEFORE inference on sample 0
    class MixedRecordDataset(Dataset):
        def __init__(self) -> None:
            self.records = [
                {"sample_id": "s_0", "split": "train"},
                {"sample_id": "s_1", "split": "test"},
            ]

        def __len__(self) -> int:
            return 2

        def __getitem__(self, idx: int):
            return (
                torch.zeros(3, 16, 16),
                torch.zeros(4, 16, 16),
                dict(self.records[idx]),
            )

    counting_model = CallCountingModel()
    with pytest.raises(SplitMismatchError, match="Split mismatch"):
        evaluate_aupr_dataset(
            model=counting_model,
            dataset=MixedRecordDataset(),
            expected_split="train",
            batch_size=1,
            use_sliding_window=False,
        )
    assert counting_model.forward_calls == 0

    # 3. Reject test record hidden inside a Subset wrapper before inference
    counting_model = CallCountingModel()
    with pytest.raises(SplitMismatchError, match="Split mismatch"):
        evaluate_aupr_dataset(
            model=counting_model,
            dataset=Subset(MixedRecordDataset(), [0, 1]),
            expected_split="train",
            use_sliding_window=True,
        )
    assert counting_model.forward_calls == 0

    # 4. Reject test record in a custom Dataset without .records (via get_sample_metadata) before inference
    class CustomMetadataDatasetWithSecondSampleTest(Dataset):
        def __init__(self) -> None:
            self.getitem_calls = 0

        def __len__(self) -> int:
            return 2

        def get_sample_metadata(self, idx: int) -> dict[str, str]:
            split_val = "valid" if idx == 0 else "test"
            return {"sample_id": f"custom_{idx}", "split": split_val}

        def __getitem__(self, idx: int):
            self.getitem_calls += 1
            split_val = "valid" if idx == 0 else "test"
            return (
                torch.zeros(3, 16, 16),
                torch.zeros(4, 16, 16),
                {"sample_id": f"custom_{idx}", "split": split_val},
            )

    counting_model = CallCountingModel()
    custom_ds = CustomMetadataDatasetWithSecondSampleTest()
    with pytest.raises(SplitMismatchError, match="Split mismatch"):
        evaluate_aupr_dataset(
            model=counting_model,
            dataset=custom_ds,
            expected_split="valid",
            batch_size=1,
            use_sliding_window=False,
        )
    assert counting_model.forward_calls == 0
    assert custom_ds.getitem_calls == 0

    # 5. Fail closed when split identity cannot be established (no .records and no sample metadata)
    class AnonymousDataset(Dataset):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, idx: int):
            return torch.zeros(3, 16, 16), torch.zeros(4, 16, 16)

    counting_model = CallCountingModel()
    with pytest.raises(SplitMismatchError, match="Split identity cannot be established"):
        evaluate_aupr_dataset(
            model=counting_model,
            dataset=AnonymousDataset(),
            expected_split="valid",
        )
    assert counting_model.forward_calls == 0


def test_stage2b_model_training_state_restoration() -> None:
    """Verify F-04: evaluate_aupr_dataset restores model.training after completion and after exception."""
    class StatefulModel(nn.Module):
        def __init__(self, fail: bool = False) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=1)
            self.bn = nn.BatchNorm2d(4)
            self.fail = fail
            self.observed_training_during_forward: list[bool] = []

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.observed_training_during_forward.append(bool(self.training))
            if self.fail:
                raise RuntimeError("Simulated forward failure")
            return self.bn(self.conv(x))

    class SimpleValidDataset(Dataset):
        def __init__(self) -> None:
            self.records = [{"sample_id": "v1", "split": "valid"}]

        def __len__(self) -> int:
            return 1

        def __getitem__(self, idx: int):
            img = torch.zeros(3, 16, 16)
            mask = torch.zeros(4, 16, 16)
            mask[:, 2:6, 2:6] = 1.0
            return img, mask, {"sample_id": "v1", "split": "valid"}

    # 1. Model initially in training mode -> restored to training mode after success
    model_train = StatefulModel(fail=False)
    model_train.train()
    assert model_train.training is True
    evaluate_aupr_dataset(
        model=model_train,
        dataset=SimpleValidDataset(),
        expected_split="valid",
        use_sliding_window=False,
    )
    assert model_train.observed_training_during_forward == [False]
    assert model_train.training is True

    # 2. Model initially in eval mode -> remains in eval mode after success
    model_eval = StatefulModel(fail=False)
    model_eval.eval()
    assert model_eval.training is False
    evaluate_aupr_dataset(
        model=model_eval,
        dataset=SimpleValidDataset(),
        expected_split="valid",
        use_sliding_window=False,
    )
    assert model_eval.observed_training_during_forward == [False]
    assert model_eval.training is False

    # 3. Model initially in training mode -> restored to training mode after exception
    model_fail = StatefulModel(fail=True)
    model_fail.train()
    assert model_fail.training is True
    with pytest.raises(RuntimeError, match="Simulated forward failure"):
        evaluate_aupr_dataset(
            model=model_fail,
            dataset=SimpleValidDataset(),
            expected_split="valid",
            use_sliding_window=False,
        )
    assert model_fail.observed_training_during_forward == [False]
    assert model_fail.training is True


def test_stage2b_all_zero_prediction_handling() -> None:
    """Verify F-05: all-zero predictions on a present class return empty_handling='all_zero_prediction'."""
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
    y_score_zero = np.zeros_like(y_true, dtype=np.float32)

    res = compute_channel_pr_curve(y_true, y_score_zero, channel_name="MA")
    assert res.target_present is True
    assert res.empty_handling == "all_zero_prediction"
    assert res.ap == pytest.approx(0.20, abs=1e-6)
    assert res.aupr_trapezoidal == pytest.approx(0.60, abs=1e-6)

    # Absent target class with all-zero predictions still uses 'absent_target_class'
    y_absent = np.zeros(10, dtype=np.uint8)
    res_absent = compute_channel_pr_curve(y_absent, y_score_zero, channel_name="SE")
    assert res_absent.target_present is False
    assert res_absent.empty_handling == "absent_target_class"
    assert res_absent.ap is None


def test_stage2b_circular_fov_exclusion_in_both_evaluation_modes() -> None:
    """Verify F-06: dark exterior corners outside circular retinal FOV are excluded in both modes."""
    from src.data.fov import compute_retinal_fov_mask
    from src.engine.sliding_window import SlidingWindowInferencer

    h, w = 32, 32
    yy, xx = np.ogrid[:h, :w]
    center_y, center_x, radius = 16, 16, 12
    circle_mask = ((yy - center_y) ** 2 + (xx - center_x) ** 2) <= (radius**2)

    # Raw uint8 RGB image: bright circular fundus (intensity 150) with black exterior corners (intensity 0)
    raw_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    raw_rgb[circle_mask] = 150

    # Ground-truth lesion inside the circular FOV at [14:18, 14:18]
    raw_mask_hwc = np.zeros((h, w, 4), dtype=np.uint8)
    raw_mask_hwc[14:18, 14:18, :] = 1

    fov_detected = compute_retinal_fov_mask(raw_rgb, mask=raw_mask_hwc)
    assert np.array_equal(fov_detected, circle_mask)
    assert bool(fov_detected[0, 0]) is False  # Dark corner is excluded
    assert bool(fov_detected[16, 16]) is True  # Center is included

    # Model predicts true positives in [14:18, 14:18] (logit +4.0 -> prob ~0.982),
    # BUT predicts even stronger false positives in the dark corners [0:3, 0:3] outside FOV (logit +8.0 -> prob ~0.9997)!
    class CornerArtifactModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, _, mh, mw = x.shape
            logits = torch.full((b, 4, mh, mw), -6.0, dtype=torch.float32)
            logits[:, :, 14:18, 14:18] = 4.0
            logits[:, :, 0:3, 0:3] = 8.0  # Strong false positive outside circular FOV
            return logits

    model = CornerArtifactModel()

    # Mode 1: Sliding-window evaluation
    class CircularFOVSlidingDataset(Dataset):
        def __init__(self) -> None:
            self.records = [{"sample_id": "fov_sw_01", "split": "valid"}]

        def __len__(self) -> int:
            return 1

        def _load_and_crop_raw_index(self, idx: int):
            return raw_rgb.copy(), raw_mask_hwc.copy(), {"sample_id": "fov_sw_01", "split": "valid"}

    inferencer = SlidingWindowInferencer(
        model=model, patch_size=(32, 32), overlap=0.5, blending="uniform", device="cpu"
    )
    sw_report = evaluate_aupr_dataset(
        model=model,
        dataset=CircularFOVSlidingDataset(),
        device="cpu",
        use_sliding_window=True,
        sliding_window_inferencer=inferencer,
        expected_split="valid",
    )
    # Because dark corners outside the circular FOV are excluded from AUPR, AP and AUPR are 1.0!
    assert sw_report["ap_macro"] == pytest.approx(1.0, abs=1e-4)
    assert sw_report["aupr_macro"] == pytest.approx(1.0, abs=1e-4)
    assert sw_report["aupr_metrics"]["total_pixels_evaluated"] == int(np.sum(circle_mask))

    # Mode 2: Global-resizing DataLoader evaluation (with ImageNet-normalized tensor)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    norm_chw = ((np.transpose(raw_rgb, (2, 0, 1)).astype(np.float32) / 255.0) - mean) / std
    mask_chw = np.transpose(raw_mask_hwc, (2, 0, 1)).astype(np.float32)

    class CircularFOVDataLoaderDataset(Dataset):
        def __init__(self) -> None:
            self.records = [{"sample_id": "fov_dl_01", "split": "valid"}]

        def __len__(self) -> int:
            return 1

        def __getitem__(self, idx: int):
            return (
                torch.from_numpy(norm_chw),
                torch.from_numpy(mask_chw),
                {"sample_id": "fov_dl_01", "split": "valid"},
            )

    dl_report = evaluate_aupr_dataset(
        model=model,
        dataset=CircularFOVDataLoaderDataset(),
        device="cpu",
        use_sliding_window=False,
        expected_split="valid",
    )
    assert dl_report["ap_macro"] == pytest.approx(1.0, abs=1e-4)
    assert dl_report["aupr_macro"] == pytest.approx(1.0, abs=1e-4)
    assert dl_report["aupr_metrics"]["total_pixels_evaluated"] == int(np.sum(circle_mask))


def test_stage2b_memory_optimization_uint8_and_exact_numerical_equivalence() -> None:
    """Verify F-03: AUPRTracker stores uint8 targets, avoids redundant copies, and matches exact metrics."""
    rng = np.random.default_rng(2026)
    tracker = AUPRTracker(from_logits=False, store_per_image=True)

    preds_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []

    for i in range(3):
        p = rng.uniform(0.0, 1.0, size=(4, 24, 24)).astype(np.float32)
        t = (rng.uniform(0.0, 1.0, size=(4, 24, 24)) > 0.90).astype(np.uint8)
        tracker.update(p, t, sample_ids=[f"mem_{i}"])
        preds_list.append(p)
        targets_list.append(t)

    # Confirm pooled target buffers are stored as 1-byte uint8 and scores as 4-byte float32
    for ch in CANONICAL_LESION_ORDER:
        for arr_t in tracker._pooled_targets[ch]:
            assert arr_t.dtype == np.uint8
            assert arr_t.itemsize == 1
        for arr_s in tracker._pooled_scores[ch]:
            assert arr_s.dtype == np.float32
            assert arr_s.itemsize == 4

    summary = tracker.compute()

    # Compare against independent float64/int32 scikit-learn calculations
    pooled_p = np.concatenate([x.reshape(4, -1) for x in preds_list], axis=1)
    pooled_t = np.concatenate([x.reshape(4, -1) for x in targets_list], axis=1)

    for idx, ch in enumerate(CANONICAL_LESION_ORDER):
        y_true_ref = pooled_t[idx].astype(np.int32)
        y_score_ref = pooled_p[idx].astype(np.float64)
        expected_ap = float(average_precision_score(y_true_ref, y_score_ref))
        prec, rec, _ = precision_recall_curve(y_true_ref, y_score_ref)
        expected_aupr = float(auc(rec[::-1], prec[::-1]))

        assert summary["per_class"][ch]["ap"] == pytest.approx(expected_ap, abs=1e-6)
        assert summary["per_class"][ch]["aupr_trapezoidal"] == pytest.approx(expected_aupr, abs=1e-6)


# ==============================================================================
# Stage 2D Regression Tests: S2C-01, S2C-02, S2C-03, S2C-05
# ==============================================================================


def test_stage2d_s2c01_normalized_fov_bright_pixels_peripheral_and_corner_noise() -> None:
    """Verify S2C-01:
    1. ImageNet-normalized circular FOV images with bright pixels > 2.0 (e.g. raw 250/255 -> ~2.55)
    2. Dim red-dominant peripheral retinal pixels (e.g. R=14, G=0, B=0 > threshold=10)
    3. Low-amplitude background corner noise (e.g. R=6, G=5, B=4 <= threshold=10)
    4. Correct FOV exclusion in global-resizing AUPR evaluation even when bright pixels > 2.0 exist.
    """
    from src.data.fov import compute_retinal_fov_mask

    h, w = 32, 32
    yy, xx = np.ogrid[:h, :w]
    center_y, center_x, radius = 16, 16, 12
    circle_mask = ((yy - center_y) ** 2 + (xx - center_x) ** 2) <= (radius**2)
    inner_bright_mask = ((yy - center_y) ** 2 + (xx - center_x) ** 2) <= (4**2)

    # Construct raw uint8 RGB image:
    # - Exterior dark corners have low-amplitude background noise (6, 5, 4) <= threshold=10
    # - Peripheral retinal ring has dim red-dominant pixels (14, 0, 0) > threshold=10
    # - Central optic disc / bright exudate region has very bright pixels (250, 245, 240)
    #   which after ImageNet normalization reach ~2.55 (> 2.0)
    raw_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    raw_rgb[~circle_mask] = np.array([6, 5, 4], dtype=np.uint8)
    raw_rgb[circle_mask] = np.array([14, 0, 0], dtype=np.uint8)
    raw_rgb[inner_bright_mask] = np.array([250, 245, 240], dtype=np.uint8)

    # 1. Verify raw uint8 FOV detection
    fov_uint8 = compute_retinal_fov_mask(raw_rgb, threshold=10)
    assert np.array_equal(fov_uint8, circle_mask)
    assert bool(fov_uint8[0, 0]) is False  # Low-amplitude corner noise (6,5,4) excluded
    assert bool(fov_uint8[16, 5]) is True  # Dim red-dominant peripheral pixel (14,0,0) included
    assert bool(fov_uint8[16, 16]) is True  # Bright center pixel (250,245,240) included

    # 2. Verify float [0, 1] FOV detection
    float01_rgb = raw_rgb.astype(np.float32) / 255.0
    fov_float01 = compute_retinal_fov_mask(float01_rgb, threshold=10)
    assert np.array_equal(fov_float01, circle_mask)

    # 3. Verify ImageNet-normalized FOV detection (where max pixel > 2.0!)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm_hwc = (float01_rgb - mean) / std
    assert float(np.max(norm_hwc)) > 2.0  # Bright pixels exceed 2.0 (approx 2.55)
    assert float(np.min(norm_hwc)) < -1.5  # Dark corners are negative

    fov_norm_hwc = compute_retinal_fov_mask(norm_hwc, threshold=10)
    assert np.array_equal(fov_norm_hwc, circle_mask)

    norm_chw = np.transpose(norm_hwc, (2, 0, 1))
    fov_norm_chw = compute_retinal_fov_mask(norm_chw, threshold=10)
    assert np.array_equal(fov_norm_chw, circle_mask)

    # 4. Verify global-resizing AUPR evaluation excludes exterior corners even with bright > 2.0 pixels
    mask_chw = np.zeros((4, h, w), dtype=np.float32)
    mask_chw[:, 14:18, 14:18] = 1.0

    class BrightFundusCornerArtifactModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, _, mh, mw = x.shape
            logits = torch.full((b, 4, mh, mw), -6.0, dtype=torch.float32)
            logits[:, :, 14:18, 14:18] = 4.0
            logits[:, :, 0:4, 0:4] = 9.0  # Strong false positive in exterior noisy corner
            return logits

    class BrightNormalizedValidDataset(Dataset):
        def __init__(self) -> None:
            self.records = [{"sample_id": "bright_norm_01", "split": "valid"}]

        def __len__(self) -> int:
            return 1

        def __getitem__(self, idx: int):
            return (
                torch.from_numpy(norm_chw),
                torch.from_numpy(mask_chw),
                {"sample_id": "bright_norm_01", "split": "valid"},
            )

    dl_report = evaluate_aupr_dataset(
        model=BrightFundusCornerArtifactModel(),
        dataset=BrightNormalizedValidDataset(),
        device="cpu",
        use_sliding_window=False,
        expected_split="valid",
    )
    assert dl_report["ap_macro"] == pytest.approx(1.0, abs=1e-4)
    assert dl_report["aupr_macro"] == pytest.approx(1.0, abs=1e-4)
    assert dl_report["aupr_metrics"]["total_pixels_evaluated"] == int(np.sum(circle_mask))


def test_stage2d_s2c02_metadata_only_upfront_verification_zero_getitem_calls() -> None:
    """Verify S2C-02:
    5. Custom datasets without metadata-only interfaces are rejected immediately.
    6. __getitem__ is called zero times during upfront validation.
    7. A test-split record is rejected without loading its image.
    10. Non-indexable datasets without metadata-only split information are rejected.
    """
    from torch.utils.data import DataLoader, IterableDataset
    from src.engine.evaluator import _verify_aupr_dataset_upfront

    class CallTrackingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.forward_calls = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.forward_calls += 1
            return torch.zeros((x.shape[0], 4, x.shape[2], x.shape[3]), dtype=torch.float32)

    # Scenario 5 & 6: Custom dataset that returns split metadata ONLY inside __getitem__
    # (no .records, no .sample_metadata, no .get_sample_metadata) MUST be rejected
    # immediately with 0 __getitem__ calls.
    class GetItemOnlyMetadataDataset(Dataset):
        def __init__(self, split_to_return: str = "valid") -> None:
            self.getitem_calls = 0
            self.split_to_return = split_to_return

        def __len__(self) -> int:
            return 2

        def __getitem__(self, idx: int):
            self.getitem_calls += 1
            raise AssertionError("__getitem__ must never be called during upfront validation!")

    ds_no_meta = GetItemOnlyMetadataDataset("valid")
    model = CallTrackingModel()
    with pytest.raises(SplitMismatchError, match="Split identity cannot be established"):
        evaluate_aupr_dataset(
            model=model,
            dataset=ds_no_meta,
            expected_split="valid",
            use_sliding_window=False,
        )
    assert ds_no_meta.getitem_calls == 0
    assert model.forward_calls == 0

    # Also verify via DataLoader wrapper
    ds_no_meta_dl = GetItemOnlyMetadataDataset("valid")
    with pytest.raises(SplitMismatchError, match="Split identity cannot be established"):
        evaluate_aupr_dataset(
            model=model,
            dataset=DataLoader(ds_no_meta_dl, batch_size=1),
            expected_split="valid",
            use_sliding_window=False,
        )
    assert ds_no_meta_dl.getitem_calls == 0

    # Scenario 6 & 7: Direct check of _verify_aupr_dataset_upfront on valid and test-contaminated
    # datasets across .records, .sample_metadata, and .get_sample_metadata(idx) with 0 __getitem__ calls.
    class RecordsDatasetNeverLoadsPixels(Dataset):
        def __init__(self, splits: list[str]) -> None:
            self.records = [{"sample_id": f"rec_{i}", "split": s} for i, s in enumerate(splits)]
            self.getitem_calls = 0
            self.raw_load_calls = 0

        def __len__(self) -> int:
            return len(self.records)

        def _load_and_crop_raw_index(self, idx: int):
            self.raw_load_calls += 1
            raise AssertionError("_load_and_crop_raw_index must not be called when split validation fails!")

        def __getitem__(self, idx: int):
            self.getitem_calls += 1
            raise AssertionError("__getitem__ must not be called during upfront validation!")

    # Valid records -> _verify_aupr_dataset_upfront succeeds with 0 __getitem__ calls
    valid_rec_ds = RecordsDatasetNeverLoadsPixels(["valid", "valid"])
    _verify_aupr_dataset_upfront(valid_rec_ds, target_split="valid")
    assert valid_rec_ds.getitem_calls == 0
    assert valid_rec_ds.raw_load_calls == 0

    # Test record present -> rejected before any image loading in both DataLoader and sliding-window modes
    test_rec_ds = RecordsDatasetNeverLoadsPixels(["valid", "test"])
    with pytest.raises(SplitMismatchError, match="Split mismatch"):
        evaluate_aupr_dataset(
            model=model,
            dataset=test_rec_ds,
            expected_split="valid",
            use_sliding_window=False,
        )
    assert test_rec_ds.getitem_calls == 0
    assert test_rec_ds.raw_load_calls == 0
    assert model.forward_calls == 0

    with pytest.raises(SplitMismatchError, match="Split mismatch"):
        evaluate_aupr_dataset(
            model=model,
            dataset=test_rec_ds,
            expected_split="valid",
            use_sliding_window=True,
        )
    assert test_rec_ds.getitem_calls == 0
    assert test_rec_ds.raw_load_calls == 0
    assert model.forward_calls == 0

    # Scenario 10: Non-indexable IterableDataset without metadata-only split source is rejected immediately
    class UnverifiedIterableDataset(IterableDataset):
        def __init__(self) -> None:
            self.iter_calls = 0

        def __iter__(self):
            self.iter_calls += 1
            yield (
                torch.zeros(3, 16, 16),
                torch.zeros(4, 16, 16),
                {"sample_id": "iter_0", "split": "valid"},
            )

    iter_ds = UnverifiedIterableDataset()
    with pytest.raises(SplitMismatchError, match="Split identity cannot be established"):
        evaluate_aupr_dataset(
            model=model,
            dataset=DataLoader(iter_ds, batch_size=1),
            expected_split="valid",
            use_sliding_window=False,
        )
    assert iter_ds.iter_calls == 0
    assert model.forward_calls == 0


def test_stage2d_s2c03_nested_subset_and_multisplit_parent_handling() -> None:
    """Verify S2C-03:
    8. Nested Subset indices selecting only 'valid' records from a multi-split parent dataset succeed.
    9. A Subset (or nested Subset) containing a 'test' record is rejected with zero __getitem__ calls.
    """
    from torch.utils.data import Subset
    from src.engine.sliding_window import SlidingWindowInferencer

    class MultiSplitParentDataset(Dataset):
        def __init__(self) -> None:
            # Parent dataset has parent.split = "train" AND contains train, valid, and test records
            self.split = "train"
            self.records = [
                {"sample_id": "p_train_0", "split": "train"},
                {"sample_id": "p_val_1", "split": "valid"},
                {"sample_id": "p_test_2", "split": "test"},
                {"sample_id": "p_val_3", "split": "valid"},
            ]
            self.getitem_calls = 0
            self.raw_load_calls = 0

        def __len__(self) -> int:
            return len(self.records)

        def _load_and_crop_raw_index(self, idx: int):
            self.raw_load_calls += 1
            rec = self.records[idx]
            img = np.full((32, 32, 3), 120, dtype=np.uint8)
            mask = np.zeros((32, 32, 4), dtype=np.uint8)
            mask[8:24, 8:24, :] = 1
            return img, mask, dict(rec)

        def __getitem__(self, idx: int):
            self.getitem_calls += 1
            rec = self.records[idx]
            img = torch.full((3, 32, 32), 0.5, dtype=torch.float32)
            mask = torch.zeros((4, 32, 32), dtype=torch.float32)
            mask[:, 8:24, 8:24] = 1.0
            return img, mask, dict(rec)

    class SimpleMatchModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, _, h, w = x.shape
            logits = torch.full((b, 4, h, w), -6.0, dtype=torch.float32)
            logits[:, :, 8:24, 8:24] = 6.0
            return logits

    model = SimpleMatchModel()

    # Scenario 8: Nested Subset selecting only 'valid' records (parent indices 1 and 3)
    # Outer subset selects [1, 2, 3] -> ["valid", "test", "valid"]
    # Inner nested subset selects [0, 2] of outer -> parent indices [1, 3] -> ["valid", "valid"]
    parent_ds = MultiSplitParentDataset()
    outer_subset = Subset(parent_ds, [1, 2, 3])
    nested_valid_subset = Subset(outer_subset, [0, 2])

    # Works in DataLoader (global resizing) mode
    res_dl = evaluate_aupr_dataset(
        model=model,
        dataset=nested_valid_subset,
        device="cpu",
        use_sliding_window=False,
        expected_split="valid",
    )
    assert res_dl["status"] == "completed"
    assert res_dl["num_samples"] == 2
    assert res_dl["sample_ids"] == ["p_val_1", "p_val_3"]
    assert res_dl["ap_macro"] == pytest.approx(1.0, abs=1e-4)
    assert parent_ds.getitem_calls == 2

    # Works in sliding-window mode
    parent_ds_sw = MultiSplitParentDataset()
    nested_valid_subset_sw = Subset(Subset(parent_ds_sw, [1, 2, 3]), [0, 2])
    inferencer = SlidingWindowInferencer(
        model=model, patch_size=(32, 32), overlap=0.5, blending="uniform", device="cpu"
    )
    res_sw = evaluate_aupr_dataset(
        model=model,
        dataset=nested_valid_subset_sw,
        device="cpu",
        use_sliding_window=True,
        sliding_window_inferencer=inferencer,
        expected_split="valid",
    )
    assert res_sw["status"] == "completed"
    assert res_sw["num_samples"] == 2
    assert res_sw["sample_ids"] == ["p_val_1", "p_val_3"]
    assert res_sw["ap_macro"] == pytest.approx(1.0, abs=1e-4)
    assert parent_ds_sw.raw_load_calls == 2
    assert parent_ds_sw.getitem_calls == 0

    # Scenario 9: A Subset or nested Subset that includes the 'test' record (parent index 2)
    # is rejected upfront with ZERO __getitem__ calls and ZERO _load_and_crop_raw_index calls.
    parent_ds_bad = MultiSplitParentDataset()
    nested_bad_subset = Subset(Subset(parent_ds_bad, [1, 2, 3]), [0, 1])  # maps to parent [1, 2] ('valid', 'test')

    with pytest.raises(SplitMismatchError, match="Split mismatch"):
        evaluate_aupr_dataset(
            model=model,
            dataset=nested_bad_subset,
            device="cpu",
            use_sliding_window=False,
            expected_split="valid",
        )
    assert parent_ds_bad.getitem_calls == 0
    assert parent_ds_bad.raw_load_calls == 0

    with pytest.raises(SplitMismatchError, match="Split mismatch"):
        evaluate_aupr_dataset(
            model=model,
            dataset=nested_bad_subset,
            device="cpu",
            use_sliding_window=True,
            sliding_window_inferencer=inferencer,
            expected_split="valid",
        )
    assert parent_ds_bad.getitem_calls == 0
    assert parent_ds_bad.raw_load_calls == 0


# ==============================================================================
# Stage 2F Regression Tests: S2E-01 & S2E-02
# ==============================================================================


def test_stage2f_s2e01_s2e02_metadata_accessor_bounds_and_subset_index_types() -> None:
    """Verify S2E-01 and S2E-02:
    1. Valid custom dataset implementing only .get_sample_metadata(idx).
    2. Valid one-level Subset using that metadata interface (including NumPy int types).
    3. Negative Subset index rejected before metadata access.
    4. Out-of-range Subset index rejected before metadata access.
    5. Boolean index rejected.
    6. Floating-point index (1.0 and 1.7) rejected.
    7. String index rejected.
    8. Nested Subset validation remains correct (including 3-level nesting).
    9. Both verify_exact_split and evaluate_aupr_dataset reject invalid indices
       before __getitem__, raw image loading, or model forward calls.
    """
    from torch.utils.data import DataLoader, Subset
    from src.engine.evaluator import verify_exact_split
    from src.engine.sliding_window import SlidingWindowInferencer

    class GetSampleMetadataOnlyDataset(Dataset):
        def __init__(self, splits: list[str] | None = None) -> None:
            self._splits = splits or ["valid", "valid", "test"]
            self.metadata_calls: list[int] = []
            self.getitem_calls = 0
            self.raw_load_calls = 0

        def __len__(self) -> int:
            return len(self._splits)

        def get_sample_metadata(self, idx: int) -> dict[str, str]:
            self.metadata_calls.append(idx)
            return {"sample_id": f"meta_only_{idx}", "split": self._splits[idx]}

        def _load_and_crop_raw_index(self, idx: int):
            self.raw_load_calls += 1
            img = np.full((32, 32, 3), 120, dtype=np.uint8)
            mask = np.zeros((32, 32, 4), dtype=np.uint8)
            mask[8:24, 8:24, :] = 1
            return img, mask, {"sample_id": f"meta_only_{idx}", "split": self._splits[idx]}

        def __getitem__(self, idx: int):
            self.getitem_calls += 1
            img = torch.full((3, 32, 32), 0.5, dtype=torch.float32)
            mask = torch.zeros((4, 32, 32), dtype=torch.float32)
            mask[:, 8:24, 8:24] = 1.0
            return img, mask, {"sample_id": f"meta_only_{idx}", "split": self._splits[idx]}

    class CountingMatchModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.forward_calls = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.forward_calls += 1
            b, _, h, w = x.shape
            logits = torch.full((b, 4, h, w), -6.0, dtype=torch.float32)
            logits[:, :, 8:24, 8:24] = 6.0
            return logits

    # 1. Valid custom dataset implementing only .get_sample_metadata(idx)
    ds_all_valid = GetSampleMetadataOnlyDataset(["valid", "valid"])
    model_1 = CountingMatchModel()
    res_direct = evaluate_aupr_dataset(
        model=model_1,
        dataset=ds_all_valid,
        device="cpu",
        use_sliding_window=False,
        expected_split="valid",
    )
    assert res_direct["status"] == "completed"
    assert res_direct["sample_ids"] == ["meta_only_0", "meta_only_1"]
    assert ds_all_valid.getitem_calls == 2
    assert verify_exact_split(DataLoader(ds_all_valid, batch_size=1), expected_split="valid") == 2

    # 2. Valid one-level Subset using .get_sample_metadata(idx), including NumPy int64 index
    ds_mixed = GetSampleMetadataOnlyDataset(["valid", "valid", "test"])
    sub_one_level = Subset(ds_mixed, [0, np.int64(1)])
    model_2 = CountingMatchModel()
    res_sub1 = evaluate_aupr_dataset(
        model=model_2,
        dataset=sub_one_level,
        device="cpu",
        use_sliding_window=False,
        expected_split="valid",
    )
    assert res_sub1["status"] == "completed"
    assert res_sub1["sample_ids"] == ["meta_only_0", "meta_only_1"]
    assert ds_mixed.getitem_calls == 2
    assert verify_exact_split(DataLoader(sub_one_level, batch_size=1), expected_split="valid") == 2

    # 3 & 4. Negative (-1) and out-of-range (3, 99) Subset indices rejected BEFORE get_sample_metadata access,
    #        __getitem__, _load_and_crop_raw_index, or model.forward in both verify_exact_split and evaluate_aupr_dataset
    for bad_indices in ([-1], [0, -1], [3], [0, 99]):
        ds_bounds = GetSampleMetadataOnlyDataset(["valid", "valid", "valid"])
        bad_sub = Subset(ds_bounds, bad_indices)
        model_bounds = CountingMatchModel()

        with pytest.raises(SplitMismatchError, match="out of bounds"):
            verify_exact_split(DataLoader(bad_sub, batch_size=1), expected_split="valid")
        assert ds_bounds.metadata_calls == []
        assert ds_bounds.getitem_calls == 0
        assert ds_bounds.raw_load_calls == 0

        with pytest.raises(SplitMismatchError, match="out of bounds"):
            evaluate_aupr_dataset(
                model=model_bounds,
                dataset=bad_sub,
                device="cpu",
                use_sliding_window=False,
                expected_split="valid",
            )
        assert ds_bounds.metadata_calls == []
        assert ds_bounds.getitem_calls == 0
        assert ds_bounds.raw_load_calls == 0
        assert model_bounds.forward_calls == 0

        with pytest.raises(SplitMismatchError, match="out of bounds"):
            evaluate_aupr_dataset(
                model=model_bounds,
                dataset=bad_sub,
                device="cpu",
                use_sliding_window=True,
                expected_split="valid",
            )
        assert ds_bounds.metadata_calls == []
        assert ds_bounds.getitem_calls == 0
        assert ds_bounds.raw_load_calls == 0
        assert model_bounds.forward_calls == 0

    # 5, 6, 7. Boolean (True, False, np.bool_(True)), float (1.0, 1.7), and string ("0", "1")
    #          Subset indices rejected before metadata access, image loading, or model inference
    invalid_type_cases: list[list[object]] = [
        [True],
        [False],
        [np.bool_(True)],
        [1.0],
        [1.7],
        [0, 1.0],
        ["0"],
        ["1"],
        [None],
    ]
    for bad_type_indices in invalid_type_cases:
        ds_type = GetSampleMetadataOnlyDataset(["valid", "valid", "valid"])
        bad_type_sub = Subset(ds_type, bad_type_indices)  # type: ignore[arg-type]
        model_type = CountingMatchModel()

        with pytest.raises(SplitMismatchError, match="Invalid Subset index"):
            verify_exact_split(DataLoader(bad_type_sub, batch_size=1), expected_split="valid")
        assert ds_type.metadata_calls == []
        assert ds_type.getitem_calls == 0
        assert ds_type.raw_load_calls == 0

        with pytest.raises(SplitMismatchError, match="Invalid Subset index"):
            evaluate_aupr_dataset(
                model=model_type,
                dataset=bad_type_sub,
                device="cpu",
                use_sliding_window=False,
                expected_split="valid",
            )
        assert ds_type.metadata_calls == []
        assert ds_type.getitem_calls == 0
        assert ds_type.raw_load_calls == 0
        assert model_type.forward_calls == 0

        with pytest.raises(SplitMismatchError, match="Invalid Subset index"):
            evaluate_aupr_dataset(
                model=model_type,
                dataset=bad_type_sub,
                device="cpu",
                use_sliding_window=True,
                expected_split="valid",
            )
        assert ds_type.metadata_calls == []
        assert ds_type.getitem_calls == 0
        assert ds_type.raw_load_calls == 0
        assert model_type.forward_calls == 0

    # 8. Three-level nested Subset validation remains correct in both DataLoader and sliding-window modes
    ds_nested = GetSampleMetadataOnlyDataset(["train", "valid", "test", "valid"])
    # Level 1 selects [1, 2, 3] -> ('valid', 'test', 'valid')
    # Level 2 selects [0, 2] of Level 1 -> parent [1, 3] ('valid', 'valid')
    # Level 3 selects [1, 0] of Level 2 -> parent [3, 1] ('valid', 'valid')
    sub3_dl = Subset(Subset(Subset(ds_nested, [1, 2, 3]), [0, 2]), [1, 0])
    assert verify_exact_split(DataLoader(sub3_dl, batch_size=1), expected_split="valid") == 2

    model_nested = CountingMatchModel()
    res_nested_dl = evaluate_aupr_dataset(
        model=model_nested,
        dataset=sub3_dl,
        device="cpu",
        use_sliding_window=False,
        expected_split="valid",
    )
    assert res_nested_dl["sample_ids"] == ["meta_only_3", "meta_only_1"]
    assert ds_nested.getitem_calls == 2

    ds_nested_sw = GetSampleMetadataOnlyDataset(["train", "valid", "test", "valid"])
    sub3_sw = Subset(Subset(Subset(ds_nested_sw, [1, 2, 3]), [0, 2]), [1, 0])
    inferencer = SlidingWindowInferencer(
        model=model_nested, patch_size=(32, 32), overlap=0.5, blending="uniform", device="cpu"
    )
    res_nested_sw = evaluate_aupr_dataset(
        model=model_nested,
        dataset=sub3_sw,
        device="cpu",
        use_sliding_window=True,
        sliding_window_inferencer=inferencer,
        expected_split="valid",
    )
    assert res_nested_sw["sample_ids"] == ["meta_only_3", "meta_only_1"]
    assert ds_nested_sw.raw_load_calls == 2
    assert ds_nested_sw.getitem_calls == 0



