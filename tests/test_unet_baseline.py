"""Unit tests for the Phase 2, Task 1 supervised U-Net + ResNet-50 baseline.

Covers:
1. Model instantiation with pretrained=False and strict pretrained error handling
2. Forward pass output shape (B, 4, H, W) for input (B, 3, H, W)
3. Output contains finite raw logits (not pre-thresholded binary values)
4. Invalid input channel count, dimensions, or spatial resolution raise clear errors
5. Combined BCE + Soft Dice loss returns a finite scalar tensor and component breakdown
6. Backward pass through loss and model produces finite gradients on model parameters
7. Loss remains finite when all target mask channels are empty (0)
8. Loss remains finite when only one lesion channel is non-empty
9. Metric computation for known synthetic ground truth and prediction tensors
10. Metric empty-mask handling follows the documented policy
11. Metric calculation returns per-class values for MA, HE, EX, and SE, plus macro averages
12. One training step updates model parameters
13. One validation step runs in eval mode without updating model parameters
14. Checkpoint save creates expected files (latest_checkpoint.pt and best_checkpoint.pt)
15. Checkpoint load restores model, optimizer, epoch, and best metric state
16. Best-checkpoint logic updates only when the monitored validation metric improves
17. Configuration validation rejects invalid model/loss/training config values
18. GPU forward/backward smoke test when torch.cuda.is_available() is true
19. Strict test-split leakage prevention in trainer
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.engine.evaluator import (
    SplitMismatchError,
    evaluate_checkpoint_on_manifest,
    evaluate_segmentation_dataloader,
    reconstruct_model_from_checkpoint,
    verify_exact_split,
)
from src.engine.trainer import (
    SupervisedSegmentationTrainer,
    TestSplitLeakageError,
    build_optimizer_from_config,
)
from src.models.losses import BCEDiceLoss, SoftDiceLoss, build_loss_from_config
from src.models.metrics import (
    CANONICAL_LESION_ORDER,
    METRIC_NAMES,
    SegmentationMetricTracker,
    compute_segmentation_metrics,
)
from src.models.unet import (
    PretrainedWeightsError,
    ResNet50UNet,
    build_unet_from_config,
)
from src.utils.config import (
    ConfigValidationError,
    load_yaml_config,
    validate_model_config,
)


class SyntheticSegmentationDataset(Dataset[dict[str, Any]]):
    """Fast synthetic dataset yielding (3, H, W) images and (4, H, W) masks."""

    def __init__(
        self,
        num_samples: int = 4,
        image_size: int = 64,
        split: str = "train",
        seed: int = 42,
    ) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.split = split
        gen = torch.Generator().manual_seed(seed)
        self.images = torch.randn(num_samples, 3, image_size, image_size, generator=gen)
        raw_masks = torch.rand(num_samples, 4, image_size, image_size, generator=gen)
        self.masks = (raw_masks > 0.75).float()
        self.records = [
            {"sample_id": f"SYN_{split}_{i:02d}", "split": split}
            for i in range(num_samples)
        ]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "image": self.images[index],
            "mask": self.masks[index],
            "metadata": {
                "sample_id": self.records[index]["sample_id"],
                "split": self.split,
            },
        }


def test_1_model_instantiation_pretrained_false_and_strict_error() -> None:
    """Scenario 1: Model instantiation with pretrained=False and strict error check."""
    model = ResNet50UNet(
        in_channels=3,
        out_channels=4,
        pretrained=False,
        strict_pretrained=False,
    )
    assert model.in_channels == 3
    assert model.out_channels == 4
    assert model.pretrained_requested is False
    assert model.pretrained_weights_loaded is False
    assert model.weights_status == "random_initialization"
    summary = model.get_model_summary()
    assert summary["total_parameters"] > 0
    assert summary["trainable_parameters"] == summary["total_parameters"]
    assert summary["output_channel_order"] == ["MA", "HE", "EX", "SE"]

    # Verify strict_pretrained raises PretrainedWeightsError when loading fails
    with (
        patch.object(Path, "glob", return_value=[]),
        patch(
            "src.models.unet.ResNet50_Weights.DEFAULT.get_state_dict",
            side_effect=RuntimeError("Simulated offline weight download failure"),
        ),
    ):
        with pytest.raises(PretrainedWeightsError, match="Unable to load ImageNet"):
            ResNet50UNet(pretrained=True, strict_pretrained=True)


def test_2_forward_pass_output_shape_and_features() -> None:
    """Scenario 2: Forward pass output shape (B, 4, H, W) for input (B, 3, H, W)."""
    model = ResNet50UNet(in_channels=3, out_channels=4, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        logits = model(x)
        s0, s1, s2, s3, bottleneck = model.forward_features(x)

    assert logits.shape == (2, 4, 64, 64)
    assert s0.shape == (2, 64, 32, 32)
    assert s1.shape == (2, 256, 16, 16)
    assert s2.shape == (2, 512, 8, 8)
    assert s3.shape == (2, 1024, 4, 4)
    assert bottleneck.shape == (2, 2048, 2, 2)


def test_3_output_contains_finite_raw_logits() -> None:
    """Scenario 3: Output contains finite raw logits (not pre-thresholded binary values)."""
    torch.manual_seed(123)
    model = ResNet50UNet(in_channels=3, out_channels=4, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 64, 64) * 3.0
    with torch.no_grad():
        logits = model(x)

    assert torch.isfinite(logits).all()
    unique_vals = torch.unique(logits)
    assert unique_vals.numel() > 10
    # Verify logits are raw unconstrained values (can be < 0)
    assert bool((logits < 0.0).any().item()) is True


def test_4_invalid_input_dimensions_raise_clear_errors() -> None:
    """Scenario 4: Invalid input channel count or spatial dimensions raise clear errors."""
    model = ResNet50UNet(in_channels=3, out_channels=4, pretrained=False)

    # Wrong tensor rank (3D instead of 4D)
    with pytest.raises(ValueError, match="Expected 4D input tensor"):
        model(torch.randn(3, 64, 64))

    # Wrong input channel count (1 instead of 3)
    with pytest.raises(ValueError, match="Expected 3 input channels"):
        model(torch.randn(2, 1, 64, 64))

    # Spatial dimensions not divisible by 32
    with pytest.raises(ValueError, match="divisible by 32"):
        model(torch.randn(2, 3, 50, 50))

    # Spatial dimensions smaller than 32x32
    with pytest.raises(ValueError, match="at least 32x32"):
        model(torch.randn(2, 3, 16, 16))


def test_5_combined_bce_dice_loss_returns_finite_scalar() -> None:
    """Scenario 5: Combined BCE + Soft Dice loss returns a finite scalar tensor."""
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5, smooth=1e-5)
    logits = torch.randn(2, 4, 64, 64, requires_grad=True)
    targets = torch.randint(0, 2, (2, 4, 64, 64)).float()

    loss = criterion(logits, targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert float(loss.item()) > 0.0

    components = criterion.compute_components(logits, targets)
    assert set(components["per_channel_loss"].keys()) == {"MA", "HE", "EX", "SE"}
    assert torch.isfinite(components["bce_loss"])
    assert torch.isfinite(components["dice_loss"])


def test_6_backward_pass_produces_finite_gradients() -> None:
    """Scenario 6: Backward pass through loss and model produces finite gradients."""
    model = ResNet50UNet(in_channels=3, out_channels=4, pretrained=False)
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    x = torch.randn(2, 3, 64, 64)
    targets = torch.randint(0, 2, (2, 4, 64, 64)).float()

    logits = model(x)
    loss = criterion(logits, targets)
    loss.backward()

    grad_norms = [
        p.grad.norm().item()
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    assert len(grad_norms) > 0
    assert all(torch.isfinite(torch.tensor(g)) for g in grad_norms)
    assert any(g > 0.0 for g in grad_norms)


def test_7_loss_finite_when_all_target_channels_empty() -> None:
    """Scenario 7: Loss remains finite when all target mask channels are empty (0)."""
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5, smooth=1e-5)
    logits = torch.randn(2, 4, 64, 64, requires_grad=True)
    empty_targets = torch.zeros(2, 4, 64, 64, dtype=torch.float32)

    loss = criterion(logits, empty_targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_8_loss_finite_when_only_one_lesion_channel_non_empty() -> None:
    """Scenario 8: Loss remains finite when only one lesion channel is non-empty."""
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5, smooth=1e-5)
    logits = torch.randn(2, 4, 64, 64, requires_grad=True)
    targets = torch.zeros(2, 4, 64, 64, dtype=torch.float32)
    # Only channel 0 (MA) has positive lesion pixels
    targets[:, 0, 16:32, 16:32] = 1.0

    components = criterion.compute_components(logits, targets)
    loss = components["loss"]
    assert torch.isfinite(loss)
    for ch in CANONICAL_LESION_ORDER:
        assert components["per_channel_dice"][ch] >= 0.0
        assert components["per_channel_bce"][ch] >= 0.0


def test_9_metrics_exact_match_and_complete_mismatch() -> None:
    """Scenario 9: Exact match gives Dice=1.0, IoU=1.0; complete mismatch gives 0.0."""
    targets = torch.zeros(1, 4, 16, 16, dtype=torch.float32)
    targets[:, :, 4:12, 4:12] = 1.0

    # Exact match using strong logits (+10 for 1, -10 for 0)
    exact_logits = torch.where(targets == 1.0, torch.tensor(10.0), torch.tensor(-10.0))
    exact_report = compute_segmentation_metrics(exact_logits, targets, threshold=0.5)
    assert exact_report["macro"]["dice"] == pytest.approx(1.0)
    assert exact_report["macro"]["iou"] == pytest.approx(1.0)
    assert exact_report["macro"]["precision"] == pytest.approx(1.0)
    assert exact_report["macro"]["recall"] == pytest.approx(1.0)
    assert exact_report["macro"]["sensitivity"] == pytest.approx(1.0)
    assert exact_report["macro"]["specificity"] == pytest.approx(1.0)

    # Complete mismatch using inverted logits
    mismatch_logits = -exact_logits
    mismatch_report = compute_segmentation_metrics(mismatch_logits, targets, threshold=0.5)
    assert mismatch_report["macro"]["dice"] == pytest.approx(0.0)
    assert mismatch_report["macro"]["iou"] == pytest.approx(0.0)
    assert mismatch_report["macro"]["precision"] == pytest.approx(0.0)
    assert mismatch_report["macro"]["recall"] == pytest.approx(0.0)
    assert mismatch_report["macro"]["sensitivity"] == pytest.approx(0.0)
    assert mismatch_report["macro"]["specificity"] == pytest.approx(0.0)


def test_10_metric_empty_mask_handling_policy() -> None:
    """Scenario 10: Metric empty-mask handling follows the documented policy."""
    targets = torch.zeros(1, 4, 16, 16, dtype=torch.float32)
    # Channel 0 (MA): both empty -> 1.0
    # Channel 1 (HE): target empty, prediction positive -> 0.0
    # Channel 2 (EX): target positive, prediction empty -> 0.0
    # Channel 3 (SE): both positive and matching -> 1.0
    targets[0, 2, 2:6, 2:6] = 1.0
    targets[0, 3, 4:8, 4:8] = 1.0

    preds_prob = torch.zeros(1, 4, 16, 16, dtype=torch.float32)
    preds_prob[0, 1, 2:6, 2:6] = 0.9  # FP on empty target
    preds_prob[0, 3, 4:8, 4:8] = 0.9  # TP match on SE

    report = compute_segmentation_metrics(
        preds_prob,
        targets,
        threshold=0.5,
        from_logits=False,
    )

    ma = report["per_class"]["MA"]
    assert ma["empty_handling"] == "both_empty_exact_match"
    assert ma["target_present"] is False
    assert ma["prediction_present"] is False
    assert ma["dice"] == pytest.approx(1.0)
    assert ma["iou"] == pytest.approx(1.0)

    he = report["per_class"]["HE"]
    assert he["empty_handling"] == "target_empty_false_positive"
    assert he["target_present"] is False
    assert he["prediction_present"] is True
    assert he["dice"] == pytest.approx(0.0)
    assert he["iou"] == pytest.approx(0.0)

    ex = report["per_class"]["EX"]
    assert ex["empty_handling"] == "prediction_empty_false_negative"
    assert ex["target_present"] is True
    assert ex["prediction_present"] is False
    assert ex["dice"] == pytest.approx(0.0)
    assert ex["iou"] == pytest.approx(0.0)

    se = report["per_class"]["SE"]
    assert se["empty_handling"] == "standard_non_empty"
    assert se["target_present"] is True
    assert se["prediction_present"] is True
    assert se["dice"] == pytest.approx(1.0)

    assert report["present_classes"] == ["EX", "SE"]
    assert report["absent_classes"] == ["MA", "HE"]
    assert report["macro_present_only"]["dice"] == pytest.approx(0.5)


def test_11_metrics_per_class_and_macro_structure() -> None:
    """Scenario 11: Metric calculation returns MA, HE, EX, SE and macro averages."""
    tracker = SegmentationMetricTracker(threshold=0.5, from_logits=True)
    logits = torch.randn(2, 4, 32, 32)
    targets = torch.randint(0, 2, (2, 4, 32, 32)).float()
    tracker.update(logits, targets)
    summary = tracker.compute()

    assert summary["channel_order"] == ["MA", "HE", "EX", "SE"]
    assert summary["num_samples"] == 2
    assert summary["num_batches"] == 1
    for ch in CANONICAL_LESION_ORDER:
        assert ch in summary["per_class"]
        for m in METRIC_NAMES:
            val = summary["per_class"][ch][m]
            assert 0.0 <= val <= 1.0
    for m in METRIC_NAMES:
        assert 0.0 <= summary["macro"][m] <= 1.0


def test_12_one_training_step_updates_model_parameters(tmp_path: Path) -> None:
    """Scenario 12: One training step updates model parameters."""
    torch.manual_seed(42)
    model = ResNet50UNet(in_channels=3, out_channels=4, pretrained=False)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ds = SyntheticSegmentationDataset(num_samples=2, image_size=64, split="train")
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    initial_params = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }

    trainer = SupervisedSegmentationTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu",
        checkpoint_dir=tmp_path / "ckpts",
        mixed_precision=False,
    )
    train_summary = trainer.train_one_epoch(loader, epoch=1)
    assert torch.isfinite(torch.tensor(train_summary["loss"]))
    assert trainer.global_step == 1

    changed = any(
        not torch.equal(initial_params[name], p.detach())
        for name, p in model.named_parameters()
        if p.requires_grad
    )
    assert changed is True


def test_13_one_validation_step_does_not_update_parameters(tmp_path: Path) -> None:
    """Scenario 13: One validation step runs in eval mode without updating parameters."""
    torch.manual_seed(42)
    model = ResNet50UNet(in_channels=3, out_channels=4, pretrained=False)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ds = SyntheticSegmentationDataset(num_samples=2, image_size=64, split="valid")
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    initial_params = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
    }

    trainer = SupervisedSegmentationTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu",
        checkpoint_dir=tmp_path / "ckpts",
        mixed_precision=False,
    )
    val_summary = trainer.validate_one_epoch(loader, epoch=1)
    assert model.training is False
    assert torch.isfinite(torch.tensor(val_summary["loss"]))

    for name, p in model.named_parameters():
        assert torch.equal(initial_params[name], p.detach())


def test_14_and_15_checkpoint_save_and_load_restores_state(tmp_path: Path) -> None:
    """Scenarios 14 & 15: Checkpoint save creates files and load restores full state."""
    torch.manual_seed(42)
    model = ResNet50UNet(in_channels=3, out_channels=4, pretrained=False)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    train_ds = SyntheticSegmentationDataset(num_samples=2, image_size=64, split="train")
    val_ds = SyntheticSegmentationDataset(num_samples=2, image_size=64, split="valid")
    train_loader = DataLoader(train_ds, batch_size=2)
    val_loader = DataLoader(val_ds, batch_size=2)

    ckpt_dir = tmp_path / "checkpoints"
    trainer = SupervisedSegmentationTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu",
        checkpoint_dir=ckpt_dir,
        seed=99,
        config={"experiment": "unit_test"},
    )

    fit_res = trainer.fit(train_loader, val_loader, num_epochs=1, save_checkpoints=True)
    latest_path = ckpt_dir / "latest_checkpoint.pt"
    best_path = ckpt_dir / "best_checkpoint.pt"
    assert latest_path.is_file()
    assert best_path.is_file()

    # Create a fresh model/optimizer/scheduler and restore state
    model2 = ResNet50UNet(in_channels=3, out_channels=4, pretrained=False)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=1, gamma=0.5)
    trainer2 = SupervisedSegmentationTrainer(
        model=model2,
        criterion=criterion,
        optimizer=optimizer2,
        scheduler=scheduler2,
        device="cpu",
        checkpoint_dir=ckpt_dir,
    )
    loaded = trainer2.load_checkpoint(latest_path)

    assert loaded["seed"] == 99
    assert loaded["config"] == {"experiment": "unit_test"}
    assert trainer2.current_epoch == 1
    assert trainer2.start_epoch == 2
    assert trainer2.global_step == 1
    assert trainer2.best_metric_value == pytest.approx(fit_res["best_metric_value"])
    for (k1, v1), (k2, v2) in zip(
        model.state_dict().items(), model2.state_dict().items()
    ):
        assert k1 == k2
        assert torch.equal(v1, v2)


def test_16_best_checkpoint_updates_only_on_improvement(tmp_path: Path) -> None:
    """Scenario 16: Best-checkpoint logic updates only when monitored metric improves."""
    model = ResNet50UNet(in_channels=3, out_channels=4, pretrained=False)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    train_ds = SyntheticSegmentationDataset(num_samples=2, image_size=64, split="train")
    val_ds = SyntheticSegmentationDataset(num_samples=2, image_size=64, split="valid")
    train_loader = DataLoader(train_ds, batch_size=2)
    val_loader = DataLoader(val_ds, batch_size=2)

    ckpt_dir = tmp_path / "best_ckpt_test"
    trainer = SupervisedSegmentationTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu",
        checkpoint_dir=ckpt_dir,
        monitor_metric="val_dice_macro",
        monitor_mode="max",
    )

    # Mock validate_one_epoch to return higher Dice on epoch 1 (0.75) and lower on epoch 2 (0.40)
    fake_val_epochs = [
        {
            "epoch": 1,
            "loss": 0.3,
            "bce_loss": 0.3,
            "dice_loss": 0.3,
            "dice_macro": 0.75,
            "iou_macro": 0.60,
            "num_samples": 2,
            "num_batches": 1,
            "elapsed_sec": 0.01,
            "metrics": {
                "per_class": {
                    ch: {"dice": 0.75, "iou": 0.60} for ch in CANONICAL_LESION_ORDER
                }
            },
        },
        {
            "epoch": 2,
            "loss": 0.5,
            "bce_loss": 0.5,
            "dice_loss": 0.5,
            "dice_macro": 0.40,
            "iou_macro": 0.25,
            "num_samples": 2,
            "num_batches": 1,
            "elapsed_sec": 0.01,
            "metrics": {
                "per_class": {
                    ch: {"dice": 0.40, "iou": 0.25} for ch in CANONICAL_LESION_ORDER
                }
            },
        },
    ]

    with patch.object(trainer, "validate_one_epoch", side_effect=fake_val_epochs):
        res = trainer.fit(train_loader, val_loader, num_epochs=2, save_checkpoints=True)

    assert res["best_epoch"] == 1
    assert res["best_metric_value"] == pytest.approx(0.75)
    assert res["history"][0]["is_best"] is True
    assert res["history"][1]["is_best"] is False

    best_ckpt = torch.load(ckpt_dir / "best_checkpoint.pt", weights_only=False)
    latest_ckpt = torch.load(ckpt_dir / "latest_checkpoint.pt", weights_only=False)
    assert best_ckpt["epoch"] == 1
    assert latest_ckpt["epoch"] == 2


def test_17_config_validation_rejects_invalid_values() -> None:
    """Scenario 17: Configuration validation rejects invalid model/loss/training values."""
    valid_cfg = load_yaml_config("configs/model.yaml", validate=False)
    validate_model_config(valid_cfg)
    assert valid_cfg["output_channels"] == 4

    # Invalid output_channel_order
    bad_order = dict(valid_cfg, output_channel_order=["MA", "HE", "SE", "EX"])
    with pytest.raises(ConfigValidationError, match="output_channel_order"):
        validate_model_config(bad_order)

    # Both loss weights zero
    bad_loss = dict(
        valid_cfg,
        loss={"bce_weight": 0.0, "dice_weight": 0.0, "smooth": 1e-5},
    )
    with pytest.raises(ConfigValidationError, match="must be > 0"):
        validate_model_config(bad_loss)

    # Invalid optimizer
    bad_train = dict(
        valid_cfg,
        training={**valid_cfg["training"], "optimizer": "rmsprop_invalid"},
    )
    with pytest.raises(ConfigValidationError, match="training.optimizer"):
        validate_model_config(bad_train)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")
def test_18_gpu_forward_backward_amp_smoke_test(tmp_path: Path) -> None:
    """Scenario 18: GPU forward/backward smoke test with mixed precision when CUDA is available."""
    device = torch.device("cuda")
    model = ResNet50UNet(in_channels=3, out_channels=4, pretrained=False)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    train_ds = SyntheticSegmentationDataset(num_samples=2, image_size=64, split="train")
    val_ds = SyntheticSegmentationDataset(num_samples=2, image_size=64, split="valid")
    train_loader = DataLoader(train_ds, batch_size=2)
    val_loader = DataLoader(val_ds, batch_size=2)

    trainer = SupervisedSegmentationTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=tmp_path / "gpu_ckpts",
        mixed_precision=True,
    )
    res = trainer.fit(train_loader, val_loader, num_epochs=1, save_checkpoints=False)
    assert res["epochs_completed"] == 1
    assert torch.isfinite(torch.tensor(res["history"][0]["train_loss"]))
    assert torch.isfinite(torch.tensor(res["history"][0]["val_loss"]))


def test_19_test_split_leakage_rejected(tmp_path: Path) -> None:
    """Verify trainer raises TestSplitLeakageError if test or validation samples leak."""
    model = ResNet50UNet(in_channels=3, out_channels=4, pretrained=False)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    test_ds = SyntheticSegmentationDataset(num_samples=2, image_size=64, split="test")
    test_loader = DataLoader(test_ds, batch_size=2)
    val_ds = SyntheticSegmentationDataset(num_samples=2, image_size=64, split="valid")
    val_loader = DataLoader(val_ds, batch_size=2)

    trainer = SupervisedSegmentationTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu",
        checkpoint_dir=tmp_path / "leak_test",
    )
    with pytest.raises(TestSplitLeakageError, match="Forbidden 'test'-split sample"):
        trainer.train_one_epoch(test_loader, epoch=1)
    with pytest.raises(TestSplitLeakageError, match="Forbidden 'valid'-split sample"):
        trainer.train_one_epoch(val_loader, epoch=1)
    with pytest.raises(TestSplitLeakageError, match="Forbidden 'test'-split sample"):
        trainer.validate_one_epoch(test_loader, epoch=1)


def test_20_reconstruct_model_and_evaluate_synthetic_test_dataloader(
    tmp_path: Path,
) -> None:
    """Scenario 20: Reconstruct model from saved checkpoint and evaluate synthetic test split."""
    torch.manual_seed(42)
    model_cfg = {
        "architecture": "UNet",
        "encoder": "resnet50",
        "pretrained": False,
        "strict_pretrained": False,
        "input_channels": 3,
        "number_of_classes": 4,
        "output_channels": 4,
        "output_channel_order": ["MA", "HE", "EX", "SE"],
        "decoder": {
            "channels": [256, 128, 64, 32, 16],
            "use_batchnorm": True,
            "dropout": 0.1,
            "activation": None,
        },
        "loss": {"type": "bce_dice", "bce_weight": 0.5, "dice_weight": 0.5, "smooth": 1e-5},
        "training": {
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "epochs": 5,
            "mixed_precision": False,
            "gradient_clip_max_norm": 1.0,
            "metric_threshold": 0.5,
            "checkpoint_directory": str(tmp_path / "ckpts"),
            "monitor_metric": "val_dice_macro",
            "monitor_mode": "max",
        },
    }
    model = build_unet_from_config(model_cfg)
    criterion = build_loss_from_config(model_cfg)
    optimizer = build_optimizer_from_config(model, model_cfg)

    trainer = SupervisedSegmentationTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu",
        checkpoint_dir=tmp_path / "ckpts",
        seed=77,
        config={"base": {"random_seed": 77}, "data": {"batch_size": 2}, "model": model_cfg},
    )
    trainer.best_epoch = 12
    trainer.best_metric_value = 0.5432
    ckpt_path = trainer.save_checkpoint(tmp_path / "ckpts/best_checkpoint.pt", epoch=12)

    reconstructed_model, ckpt_dict, saved_cfgs = reconstruct_model_from_checkpoint(
        ckpt_path, device="cpu"
    )
    assert reconstructed_model.training is False
    assert ckpt_dict["epoch"] == 12
    assert ckpt_dict["best_epoch"] == 12
    assert ckpt_dict["seed"] == 77
    assert saved_cfgs["model"]["architecture"] == "UNet"

    for (k1, v1), (k2, v2) in zip(
        model.state_dict().items(), reconstructed_model.state_dict().items()
    ):
        assert k1 == k2
        assert torch.equal(v1, v2)

    test_ds = SyntheticSegmentationDataset(num_samples=4, image_size=64, split="test", seed=99)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)

    params_before = {
        name: p.detach().clone() for name, p in reconstructed_model.named_parameters()
    }
    eval_res = evaluate_segmentation_dataloader(
        model=reconstructed_model,
        dataloader=test_loader,
        criterion=criterion,
        device="cpu",
        mixed_precision=False,
        metric_threshold=0.5,
        expected_split="test",
    )

    # Confirm model parameters were not modified during evaluation
    for name, p in reconstructed_model.named_parameters():
        assert torch.equal(params_before[name], p.detach())

    assert eval_res["split"] == "test"
    assert eval_res["num_samples"] == 4
    assert eval_res["num_batches"] == 2
    assert eval_res["loss"] is not None
    assert 0.0 <= eval_res["dice_macro"] <= 1.0
    assert 0.0 <= eval_res["iou_macro"] <= 1.0
    assert eval_res["dice_macro_present_only"] is not None
    assert eval_res["iou_macro_present_only"] is not None

    for ch in CANONICAL_LESION_ORDER:
        ch_metrics = eval_res["per_class"][ch]
        for key in (
            "dice",
            "iou",
            "precision",
            "recall",
            "sensitivity",
            "specificity",
            "tp",
            "fp",
            "fn",
            "tn",
        ):
            assert key in ch_metrics


def test_21_verify_exact_split_rejects_non_test_samples() -> None:
    """Scenario 21: Held-out test evaluator rejects train or validation samples."""
    model = ResNet50UNet(in_channels=3, out_channels=4, pretrained=False)
    train_ds = SyntheticSegmentationDataset(num_samples=2, image_size=64, split="train")
    val_ds = SyntheticSegmentationDataset(num_samples=2, image_size=64, split="valid")
    test_ds = SyntheticSegmentationDataset(num_samples=2, image_size=64, split="test")

    train_loader = DataLoader(train_ds, batch_size=2)
    val_loader = DataLoader(val_ds, batch_size=2)
    test_loader = DataLoader(test_ds, batch_size=2)

    assert verify_exact_split(test_loader, expected_split="test") == 2

    with pytest.raises(SplitMismatchError, match="expected 'test'"):
        verify_exact_split(train_loader, expected_split="test")

    with pytest.raises(SplitMismatchError, match="expected 'test'"):
        verify_exact_split(val_loader, expected_split="test")

    with pytest.raises(SplitMismatchError, match="expected 'test'"):
        evaluate_segmentation_dataloader(
            model=model,
            dataloader=val_loader,
            expected_split="test",
        )


def test_22_evaluate_checkpoint_on_synthetic_manifest_writes_report(
    tmp_path: Path,
) -> None:
    """Scenario 22: End-to-end synthetic manifest evaluation filters strictly to 'test' and writes JSON report."""
    import json
    import cv2
    import numpy as np

    img_dir = tmp_path / "images"
    msk_dir = tmp_path / "masks"
    img_dir.mkdir(parents=True)
    msk_dir.mkdir(parents=True)

    records: list[dict[str, Any]] = []
    # Create 1 train, 1 valid, and 2 test synthetic records on disk
    sample_specs = [
        ("SYN_TRAIN_01", "train"),
        ("SYN_VALID_01", "valid"),
        ("SYN_TEST_01", "test"),
        ("SYN_TEST_02", "test"),
    ]
    for idx, (sample_id, split_name) in enumerate(sample_specs):
        img = np.full((64, 64, 3), fill_value=60 + idx * 20, dtype=np.uint8)
        img_path = img_dir / f"{sample_id}.png"
        cv2.imwrite(str(img_path), img)

        mask_paths: dict[str, str | None] = {}
        for ch_idx, ch in enumerate(CANONICAL_LESION_ORDER):
            m = np.zeros((64, 64), dtype=np.uint8)
            m[8 + ch_idx * 4 : 20 + ch_idx * 4, 8:24] = 255
            m_path = msk_dir / f"{sample_id}_{ch}.png"
            cv2.imwrite(str(m_path), m)
            mask_paths[ch] = str(m_path)

        records.append(
            {
                "dataset_name": "DDR",
                "sample_id": sample_id,
                "split": split_name,
                "image_path": str(img_path),
                "mask_paths": mask_paths,
                "channel_order": list(CANONICAL_LESION_ORDER),
            }
        )

    manifest_path = tmp_path / "synthetic_ddr_manifest.json"
    manifest_path.write_text(
        json.dumps({"dataset_name": "DDR", "records": records}, indent=2),
        encoding="utf-8",
    )

    model_cfg = load_yaml_config("configs/model.yaml", validate=False)
    model_cfg["pretrained"] = False
    model_cfg["strict_pretrained"] = False
    data_cfg = load_yaml_config("configs/dataset.yaml", validate=False)

    model = build_unet_from_config(model_cfg)
    criterion = build_loss_from_config(model_cfg)
    optimizer = build_optimizer_from_config(model, model_cfg)
    trainer = SupervisedSegmentationTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu",
        checkpoint_dir=tmp_path / "ckpts",
        seed=42,
        config={"base": {"random_seed": 42, "deterministic": True}, "data": data_cfg, "model": model_cfg},
    )
    trainer.best_epoch = 50
    trainer.best_metric_value = 0.558165
    ckpt_path = trainer.save_checkpoint(tmp_path / "ckpts/best_checkpoint.pt", epoch=50)

    report_path = tmp_path / "logs/test_report.json"
    report = evaluate_checkpoint_on_manifest(
        checkpoint_path=ckpt_path,
        manifest_or_records=manifest_path,
        split="test",
        dataset_config=data_cfg,
        model_config=model_cfg,
        batch_size=2,
        image_size=64,
        device="cpu",
        report_path=report_path,
    )

    assert report_path.is_file()
    loaded_json = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded_json["status"] == "completed"
    assert loaded_json["checkpoint"]["epoch"] == 50
    assert loaded_json["dataset"]["split"] == "test"
    assert loaded_json["dataset"]["sample_count"] == 2
    assert loaded_json["evaluation_config"]["seed"] == 42
    assert loaded_json["evaluation_config"]["image_size"] == [64, 64]
    assert set(loaded_json["test_metrics"]["per_class"].keys()) == {"MA", "HE", "EX", "SE"}
    assert report["dataset"]["sample_count"] == 2


def test_23_single_channel_encoder_checkpoint_loading_and_empty_checkpoint_handling(
    tmp_path: Path,
) -> None:
    """Scenario 23: Verify in_channels=1 checkpoint loading preserves conv1 weights and empty dicts fail clearly (L-2)."""
    from src.models.simclr import SimCLRModel

    torch.manual_seed(77)
    src_simclr = SimCLRModel(
        in_channels=1,
        projection_dim=64,
        projection_hidden_dim=128,
        pretrained=False,
    )
    # Assign deterministic pattern to 1-channel conv1 weight so any post-load overwrite is caught
    with torch.no_grad():
        src_simclr.encoder.conv1.weight.fill_(0.25)

    ckpt_1ch_path = tmp_path / "simclr_1ch_encoder.pt"
    torch.save(
        {"encoder_state_dict": src_simclr.export_encoder_state_dict()},
        ckpt_1ch_path,
    )

    unet_1ch = ResNet50UNet(
        in_channels=1,
        out_channels=4,
        pretrained=True,
        strict_pretrained=True,
        encoder_weights_source="ssl_checkpoint",
        pretrained_checkpoint_path=ckpt_1ch_path,
    )
    assert unet_1ch.pretrained_weights_loaded is True
    assert tuple(unet_1ch.encoder_conv1.weight.shape) == (64, 1, 7, 7)
    assert torch.allclose(
        unet_1ch.encoder_conv1.weight,
        torch.full_like(unet_1ch.encoder_conv1.weight, 0.25),
    )

    simclr_1ch = SimCLRModel(
        in_channels=1,
        projection_dim=64,
        projection_hidden_dim=128,
        pretrained=True,
        strict_pretrained=True,
        encoder_weights_source="ssl_checkpoint",
        pretrained_checkpoint_path=ckpt_1ch_path,
    )
    assert simclr_1ch.pretrained_weights_loaded is True
    assert tuple(simclr_1ch.encoder.conv1.weight.shape) == (64, 1, 7, 7)
    assert torch.allclose(
        simclr_1ch.encoder.conv1.weight,
        torch.full_like(simclr_1ch.encoder.conv1.weight, 0.25),
    )

    # Forward pass check on 1-channel input
    unet_1ch.eval()
    with torch.no_grad():
        out = unet_1ch(torch.randn(2, 1, 64, 64))
    assert out.shape == (2, 4, 64, 64)

    # Empty checkpoint dict {} and {"encoder_state_dict": {}} must raise PretrainedWeightsError
    empty_ckpt = tmp_path / "empty_ckpt.pt"
    torch.save({}, empty_ckpt)
    with pytest.raises(PretrainedWeightsError, match="empty"):
        ResNet50UNet(
            in_channels=3,
            out_channels=4,
            pretrained=True,
            strict_pretrained=True,
            encoder_weights_source="ssl_checkpoint",
            pretrained_checkpoint_path=empty_ckpt,
        )

    empty_nested_ckpt = tmp_path / "empty_nested_ckpt.pt"
    torch.save({"encoder_state_dict": {}}, empty_nested_ckpt)
    with pytest.raises(PretrainedWeightsError, match="empty"):
        ResNet50UNet(
            in_channels=3,
            out_channels=4,
            pretrained=True,
            strict_pretrained=True,
            encoder_weights_source="ssl_checkpoint",
            pretrained_checkpoint_path=empty_nested_ckpt,
        )


def test_24_interrupted_and_resumed_training_matches_uninterrupted_training_cpu(
    tmp_path: Path,
) -> None:
    """Scenario 24: Interrupted-and-resumed training reproduces uninterrupted training exactly on CPU (M-3)."""
    from src.utils.seed import set_seed

    # 1. Run 2 epochs uninterrupted with shuffle=True
    set_seed(seed=123, deterministic=True)
    train_ds_full = SyntheticSegmentationDataset(
        num_samples=6, image_size=64, split="train", seed=123
    )
    val_ds_full = SyntheticSegmentationDataset(
        num_samples=2, image_size=64, split="valid", seed=456
    )
    gen_full = torch.Generator().manual_seed(123)
    train_loader_full = DataLoader(
        train_ds_full,
        batch_size=2,
        shuffle=True,
        generator=gen_full,
    )
    val_loader_full = DataLoader(val_ds_full, batch_size=2, shuffle=False)

    set_seed(seed=123, deterministic=True)
    model_full = ResNet50UNet(in_channels=3, out_channels=4, pretrained=False)
    criterion_full = BCEDiceLoss()
    opt_full = torch.optim.AdamW(model_full.parameters(), lr=5e-4, weight_decay=1e-4)
    sched_full = torch.optim.lr_scheduler.StepLR(opt_full, step_size=1, gamma=0.5)

    trainer_full = SupervisedSegmentationTrainer(
        model=model_full,
        criterion=criterion_full,
        optimizer=opt_full,
        scheduler=sched_full,
        device="cpu",
        checkpoint_dir=tmp_path / "full_run",
        mixed_precision=False,
        seed=123,
    )
    res_full = trainer_full.fit(
        train_loader_full, val_loader_full, num_epochs=2, save_checkpoints=True
    )

    # 2. Run 1 epoch, interrupt, reload into fresh instances, and resume for Epoch 2
    set_seed(seed=123, deterministic=True)
    train_ds_part1 = SyntheticSegmentationDataset(
        num_samples=6, image_size=64, split="train", seed=123
    )
    val_ds_part1 = SyntheticSegmentationDataset(
        num_samples=2, image_size=64, split="valid", seed=456
    )
    gen_part1 = torch.Generator().manual_seed(123)
    train_loader_part1 = DataLoader(
        train_ds_part1,
        batch_size=2,
        shuffle=True,
        generator=gen_part1,
    )
    val_loader_part1 = DataLoader(val_ds_part1, batch_size=2, shuffle=False)

    set_seed(seed=123, deterministic=True)
    model_part1 = ResNet50UNet(in_channels=3, out_channels=4, pretrained=False)
    opt_part1 = torch.optim.AdamW(model_part1.parameters(), lr=5e-4, weight_decay=1e-4)
    sched_part1 = torch.optim.lr_scheduler.StepLR(opt_part1, step_size=1, gamma=0.5)

    ckpt_dir_resume = tmp_path / "resume_run"
    trainer_part1 = SupervisedSegmentationTrainer(
        model=model_part1,
        criterion=BCEDiceLoss(),
        optimizer=opt_part1,
        scheduler=sched_part1,
        device="cpu",
        checkpoint_dir=ckpt_dir_resume,
        mixed_precision=False,
        seed=123,
    )
    trainer_part1.fit(
        train_loader_part1, val_loader_part1, num_epochs=1, save_checkpoints=True
    )

    # Simulate fresh process startup with different RNG state before loading checkpoint
    set_seed(seed=999, deterministic=True)
    gen_part2 = torch.Generator().manual_seed(123)
    train_loader_part2 = DataLoader(
        train_ds_part1,
        batch_size=2,
        shuffle=True,
        generator=gen_part2,
    )
    val_loader_part2 = DataLoader(val_ds_part1, batch_size=2, shuffle=False)
    model_part2 = ResNet50UNet(in_channels=3, out_channels=4, pretrained=False)
    opt_part2 = torch.optim.AdamW(model_part2.parameters(), lr=5e-4, weight_decay=1e-4)
    sched_part2 = torch.optim.lr_scheduler.StepLR(opt_part2, step_size=1, gamma=0.5)

    trainer_part2 = SupervisedSegmentationTrainer(
        model=model_part2,
        criterion=BCEDiceLoss(),
        optimizer=opt_part2,
        scheduler=sched_part2,
        device="cpu",
        checkpoint_dir=ckpt_dir_resume,
        mixed_precision=False,
        seed=123,
    )
    loaded_ckpt = trainer_part2.load_checkpoint(ckpt_dir_resume / "latest_checkpoint.pt")
    assert "rng_state" in loaded_ckpt
    res_resumed = trainer_part2.fit(
        train_loader_part2, val_loader_part2, num_epochs=2, save_checkpoints=True
    )

    # Epoch 1 and Epoch 2 losses and metrics must match identically
    assert res_resumed["history"][0]["train_loss"] == pytest.approx(
        res_full["history"][0]["train_loss"], abs=1e-7
    )
    assert res_resumed["history"][1]["train_loss"] == pytest.approx(
        res_full["history"][1]["train_loss"], abs=1e-7
    )
    assert res_resumed["history"][1]["val_loss"] == pytest.approx(
        res_full["history"][1]["val_loss"], abs=1e-7
    )
    assert res_resumed["history"][1]["learning_rate"] == pytest.approx(2.5e-4)
    assert res_resumed["history"][1]["next_learning_rate"] == pytest.approx(1.25e-4)

    for (k1, v1), (k2, v2) in zip(
        model_full.state_dict().items(), model_part2.state_dict().items()
    ):
        assert k1 == k2
        assert torch.allclose(v1, v2, atol=1e-7), f"Parameter mismatch on {k1}"

    # Verify backward compatibility with legacy checkpoints lacking 'rng_state'
    legacy_ckpt_path = tmp_path / "legacy_checkpoint.pt"
    legacy_payload = dict(loaded_ckpt)
    legacy_payload.pop("rng_state", None)
    torch.save(legacy_payload, legacy_ckpt_path)

    trainer_legacy = SupervisedSegmentationTrainer(
        model=ResNet50UNet(in_channels=3, out_channels=4, pretrained=False),
        criterion=BCEDiceLoss(),
        optimizer=opt_part2,
        device="cpu",
        checkpoint_dir=tmp_path / "legacy_run",
        mixed_precision=False,
    )
    loaded_legacy = trainer_legacy.load_checkpoint(legacy_ckpt_path)
    assert "rng_state" not in loaded_legacy
    assert trainer_legacy.start_epoch == 2


