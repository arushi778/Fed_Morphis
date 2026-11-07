"""Unit tests for Phase 3, Task 1: SimCLR Model and NT-Xent Contrastive Learning Components.

Covers:
1. SimCLRModel output shapes for encoder representation h (B, 2048) and projection z (B, D)
2. Configurable projection head dimensions and BatchNorm option
3. Strict separation between ResNet50Encoder and SimCLRProjectionHead (and independence from U-Net decoder)
4. NT-Xent loss finiteness, L2 normalization invariance, and self-similarity exclusion
5. NT-Xent positive-pair alignment behavior (aligned pairs produce lower loss than anti-aligned pairs)
6. NT-Xent batch_size=1 safe & clear error handling and invalid input rejection
7. End-to-end gradient propagation through SimCLR encoder and projection head
8. Bidirectional encoder weight compatibility with ResNet50UNet (loading from supervised baseline checkpoint and exporting to ResNet50UNet)
9. SimCLR configuration loading and validation (configs/ssl.yaml and configs/model.yaml)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models.losses import NTXentLoss, build_nt_xent_loss_from_config
from src.models.simclr import (
    RESNET50_FEATURE_DIM,
    ResNet50Encoder,
    SimCLRModel,
    SimCLRProjectionHead,
    build_simclr_from_config,
)
from src.models.unet import PretrainedWeightsError, ResNet50UNet
from src.utils.config import (
    ConfigValidationError,
    load_yaml_config,
    validate_model_config,
    validate_ssl_config,
)


def test_1_simclr_model_output_shapes_and_configurable_projection() -> None:
    """Verify SimCLRModel returns (h, z) with shapes (B, 2048) and (B, projection_dim)."""
    for proj_dim in (64, 128, 256):
        model = SimCLRModel(
            in_channels=3,
            projection_dim=proj_dim,
            projection_hidden_dim=512,
            projection_use_batchnorm=False,
            pretrained=False,
        )
        model.eval()
        x = torch.randn(3, 3, 64, 64)
        with torch.no_grad():
            h, z = model(x)

        assert h.shape == (3, RESNET50_FEATURE_DIM)
        assert z.shape == (3, proj_dim)
        assert torch.isfinite(h).all()
        assert torch.isfinite(z).all()

    # Verify two-view helper forward_views
    model = SimCLRModel(
        in_channels=3,
        projection_dim=128,
        projection_hidden_dim=256,
        projection_use_batchnorm=True,
        pretrained=False,
    )
    model.eval()
    v1 = torch.randn(2, 3, 64, 64)
    v2 = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        views_out = model.forward_views(v1, v2)

    assert set(views_out.keys()) == {"h1", "z1", "h2", "z2"}
    assert views_out["h1"].shape == (2, 2048)
    assert views_out["h2"].shape == (2, 2048)
    assert views_out["z1"].shape == (2, 128)
    assert views_out["z2"].shape == (2, 128)


def test_2_encoder_and_projection_head_separation() -> None:
    """Verify encoder and projection head are independent submodules with no classification or U-Net decoder layers."""
    model = SimCLRModel(
        in_channels=3,
        projection_dim=128,
        projection_hidden_dim=512,
        pretrained=False,
    )
    model.eval()

    assert isinstance(model.encoder, ResNet50Encoder)
    assert isinstance(model.projection_head, SimCLRProjectionHead)
    assert not hasattr(model.encoder, "fc")
    assert not hasattr(model, "decoder4")
    assert not hasattr(model, "segmentation_head")

    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        h_direct = model.encode(x)
        z_direct = model.project(h_direct)
        h_fwd, z_fwd = model(x)

    assert torch.equal(h_direct, h_fwd)
    assert torch.equal(z_direct, z_fwd)

    summary = model.get_model_summary()
    assert summary["architecture"] == "SimCLR"
    assert summary["backbone"] == "resnet50"
    assert (
        summary["encoder_parameters"] + summary["projection_head_parameters"]
        == summary["total_parameters"]
    )


def test_3_simclr_invalid_inputs_rejected() -> None:
    """Verify invalid image shapes or constructor arguments raise clear ValueErrors."""
    model = SimCLRModel(pretrained=False)

    with pytest.raises(ValueError, match="Expected 4D input tensor"):
        model(torch.randn(3, 64, 64))

    with pytest.raises(ValueError, match="Expected 3 input channels"):
        model(torch.randn(2, 1, 64, 64))

    with pytest.raises(ValueError, match="at least 32x32"):
        model(torch.randn(2, 3, 16, 16))

    with pytest.raises(ValueError, match="projection_dim"):
        SimCLRModel(projection_dim=0, pretrained=False)

    with pytest.raises(ValueError, match="projection_hidden_dim"):
        SimCLRModel(projection_hidden_dim=0, pretrained=False)

    with pytest.raises(ValueError, match="Expected 2D representation tensor"):
        model.project(torch.randn(2, 2048, 1))

    with pytest.raises(ValueError, match="Expected representation dimension 2048"):
        model.project(torch.randn(2, 512))


def test_4_nt_xent_loss_finiteness_normalization_and_self_similarity_exclusion() -> None:
    """Verify NTXentLoss computes finite loss, normalizes embeddings, and excludes self-similarity."""
    loss_fn = NTXentLoss(temperature=0.5, normalize=True)
    torch.manual_seed(42)
    z1 = torch.randn(4, 128)
    z2 = torch.randn(4, 128)

    out = loss_fn.forward_with_metrics(z1, z2)
    loss = out["loss"]
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert float(loss.item()) > 0.0

    # 2B = 8 anchors, each contrasted against 2B - 1 = 7 non-self views
    assert out["logits"].shape == (8, 7)
    assert out["targets"].shape == (8,)

    # L2 normalization invariance: scaling z1 and z2 by positive constants leaves loss unchanged
    loss_scaled = loss_fn(z1 * 10.0, z2 * 0.25)
    assert torch.allclose(loss, loss_scaled, atol=1e-6)


def test_5_nt_xent_positive_pair_alignment_behavior() -> None:
    """Verify aligned positive pairs achieve lower loss and higher accuracy than anti-aligned or random pairs."""
    loss_fn = NTXentLoss(temperature=0.2, normalize=True)

    # Construct 4 orthogonal unit vectors in R^16
    basis = torch.eye(4, 16, dtype=torch.float32)

    # Aligned positive views: z1 == z2 (each sample's two views match perfectly and are orthogonal to negatives)
    aligned_out = loss_fn.forward_with_metrics(basis, basis.clone())

    # Anti-aligned positive views: z2 == -z1
    anti_aligned_out = loss_fn.forward_with_metrics(basis, -basis.clone())

    assert float(aligned_out["loss"].item()) < float(anti_aligned_out["loss"].item())
    assert float(aligned_out["positive_similarity"].item()) == pytest.approx(1.0, abs=1e-5)
    assert float(aligned_out["negative_similarity"].item()) == pytest.approx(0.0, abs=1e-5)
    assert float(aligned_out["contrastive_accuracy"].item()) == pytest.approx(1.0)
    assert float(anti_aligned_out["positive_similarity"].item()) == pytest.approx(-1.0, abs=1e-5)


def test_6_nt_xent_batch_size_one_and_invalid_inputs_handled_clearly() -> None:
    """Verify NTXentLoss handles batch_size=1 and invalid inputs with clear errors."""
    loss_fn = NTXentLoss(temperature=0.5)

    # Batch size 1 has no negative pairs when self-similarity is excluded
    z1_b1 = torch.randn(1, 128)
    z2_b1 = torch.randn(1, 128)
    with pytest.raises(
        ValueError,
        match="NTXentLoss requires batch_size >= 2 to form negative pairs",
    ):
        loss_fn(z1_b1, z2_b1)

    # Shape mismatch between z1 and z2
    with pytest.raises(ValueError, match="Shape mismatch"):
        loss_fn(torch.randn(4, 128), torch.randn(4, 64))

    # Non-2D tensors
    with pytest.raises(ValueError, match="Expected 2D projection tensors"):
        loss_fn(torch.randn(4, 128, 1), torch.randn(4, 128, 1))

    # Non-tensor input
    with pytest.raises(TypeError, match="torch.Tensor"):
        loss_fn([[1.0, 2.0]], [[1.0, 2.0]])  # type: ignore[arg-type]

    # Invalid temperatures
    with pytest.raises(ValueError, match="temperature"):
        NTXentLoss(temperature=0.0)
    with pytest.raises(ValueError, match="temperature"):
        NTXentLoss(temperature=-0.5)
    with pytest.raises(ValueError, match="temperature"):
        NTXentLoss(temperature=float("nan"))


def test_7_gradient_flow_through_simclr_encoder_and_projection_head() -> None:
    """Verify backward pass through NTXentLoss propagates finite gradients to both encoder and projection head."""
    torch.manual_seed(99)
    model = SimCLRModel(
        in_channels=3,
        projection_dim=64,
        projection_hidden_dim=128,
        pretrained=False,
    )
    criterion = NTXentLoss(temperature=0.5)

    view1 = torch.randn(2, 3, 64, 64)
    view2 = torch.randn(2, 3, 64, 64)

    views = model.forward_views(view1, view2)
    loss = criterion(views["z1"], views["z2"])
    loss.backward()

    encoder_grads = [
        p.grad.norm().item()
        for p in model.encoder.parameters()
        if p.requires_grad and p.grad is not None
    ]
    proj_grads = [
        p.grad.norm().item()
        for p in model.projection_head.parameters()
        if p.requires_grad and p.grad is not None
    ]

    assert len(encoder_grads) > 0
    assert len(proj_grads) > 0
    assert all(torch.isfinite(torch.tensor(g)) for g in encoder_grads)
    assert all(torch.isfinite(torch.tensor(g)) for g in proj_grads)
    assert any(g > 0.0 for g in encoder_grads)
    assert any(g > 0.0 for g in proj_grads)


def test_8_supervised_checkpoint_loading_and_unet_transfer_compatibility(
    tmp_path: Path,
) -> None:
    """Verify SimCLRModel loads encoder weights from a ResNet50UNet checkpoint and exports back to ResNet50UNet."""
    torch.manual_seed(17)
    unet = ResNet50UNet(in_channels=3, out_channels=4, pretrained=False)
    unet_ckpt_path = tmp_path / "supervised_best_checkpoint.pt"
    torch.save(
        {
            "epoch": 50,
            "model_state_dict": unet.state_dict(),
        },
        unet_ckpt_path,
    )

    # Load supervised ResNet50UNet checkpoint into SimCLRModel
    simclr = SimCLRModel(
        in_channels=3,
        projection_dim=128,
        projection_hidden_dim=256,
        pretrained=True,
        strict_pretrained=True,
        encoder_weights_source="supervised_checkpoint",
        pretrained_checkpoint_path=unet_ckpt_path,
    )
    assert simclr.pretrained_weights_loaded is True
    assert simclr.weights_status == f"loaded_supervised_checkpoint:{unet_ckpt_path.name}"

    # Verify encoder conv1 and layer4 weights match ResNet50UNet exactly
    assert torch.equal(unet.encoder_conv1.weight, simclr.encoder.conv1.weight)
    for (k1, v1), (k2, v2) in zip(
        unet.encoder_layer4.state_dict().items(),
        simclr.encoder.layer4.state_dict().items(),
    ):
        assert k1 == k2
        assert torch.equal(v1, v2)

    # Now save SimCLR checkpoint and verify ResNet50UNet can load it via encoder_weights_source="ssl_checkpoint"
    ssl_ckpt_path = tmp_path / "simclr_pretrained.pt"
    torch.save(
        {
            "epoch": 50,
            "encoder_state_dict": simclr.export_encoder_state_dict(),
            "model_state_dict": simclr.state_dict(),
        },
        ssl_ckpt_path,
    )

    unet_from_ssl = ResNet50UNet(
        in_channels=3,
        out_channels=4,
        pretrained=True,
        strict_pretrained=True,
        encoder_weights_source="ssl_checkpoint",
        pretrained_checkpoint_path=ssl_ckpt_path,
    )
    assert unet_from_ssl.pretrained_weights_loaded is True
    assert torch.equal(simclr.encoder.conv1.weight, unet_from_ssl.encoder_conv1.weight)

    # Verify strict error when requested checkpoint does not exist
    with pytest.raises(PretrainedWeightsError, match="not found"):
        SimCLRModel(
            pretrained=True,
            strict_pretrained=True,
            encoder_weights_source="supervised_checkpoint",
            pretrained_checkpoint_path=tmp_path / "missing_ckpt.pt",
        )


def test_9_simclr_config_loading_and_validation() -> None:
    """Verify ssl.yaml and model.yaml:simclr validate and build SimCLRModel & NTXentLoss."""
    ssl_cfg = load_yaml_config("configs/ssl.yaml", validate=True)
    model_cfg = load_yaml_config("configs/model.yaml", validate=True)

    assert ssl_cfg["method"] == "SimCLR"
    assert ssl_cfg["projection_dimension"] == 128
    assert ssl_cfg["temperature"] == pytest.approx(0.5)
    assert ssl_cfg["batch_size"] == 16
    assert ssl_cfg["epochs"] == 50
    assert ssl_cfg["optimizer"] == "adamw"
    assert ssl_cfg["learning_rate"] == pytest.approx(0.0003)
    assert ssl_cfg["weight_decay"] == pytest.approx(0.0001)
    assert ssl_cfg["random_seed"] == 42
    assert "augmentations" in ssl_cfg

    model_from_ssl = build_simclr_from_config(ssl_cfg, pretrained_override=False)
    model_from_model_yaml = build_simclr_from_config(model_cfg, pretrained_override=False)
    assert model_from_ssl.projection_dim == 128
    assert model_from_model_yaml.projection_dim == 128

    loss_from_ssl = build_nt_xent_loss_from_config(ssl_cfg)
    assert loss_from_ssl.temperature == pytest.approx(0.5)

    # Reject batch_size < 2 in ssl config
    with pytest.raises(ConfigValidationError, match="batch_size"):
        validate_ssl_config({**ssl_cfg, "batch_size": 1})

    # Reject invalid method
    with pytest.raises(ConfigValidationError, match="method"):
        validate_ssl_config({**ssl_cfg, "method": "UnknownSSL"})

    # Reject invalid crop scale in augmentations
    bad_aug = dict(ssl_cfg["augmentations"], random_resized_crop_scale=[0.9, 0.2])
    with pytest.raises(ConfigValidationError, match="random_resized_crop_scale"):
        validate_ssl_config({**ssl_cfg, "augmentations": bad_aug})

    # Reject invalid simclr block inside model.yaml
    bad_model_cfg = dict(model_cfg, simclr={**model_cfg["simclr"], "temperature": 0.0})
    with pytest.raises(ConfigValidationError, match="temperature"):
        validate_model_config(bad_model_cfg)


def test_10_two_view_transform_and_unlabeled_dataset_no_mask_dependency(
    tmp_path: Path,
) -> None:
    """Verify SimCLRViewTransform and SimCLRUnlabeledDataset generate two distinct views without requiring masks."""
    import cv2
    import numpy as np

    from src.data.dataset import (
        SimCLRUnlabeledDataset,
        build_simclr_dataloader,
    )
    from src.data.transforms import SimCLRViewTransform
    from src.utils.seed import set_seed

    img_dir = tmp_path / "ssl_images"
    img_dir.mkdir(parents=True)

    # Create 4 synthetic gradient/textured images with NO mask files or mask_paths
    records = []
    rng = np.random.default_rng(123)
    for i in range(4):
        arr = rng.integers(20, 240, size=(96, 96, 3), dtype=np.uint8)
        img_p = img_dir / f"UNLABELED_{i:02d}.png"
        cv2.imwrite(str(img_p), arr)
        records.append(
            {
                "dataset_name": "DDR" if i < 2 else "IDRiD",
                "sample_id": f"UNLABELED_{i:02d}",
                "split": "train",
                "image_path": str(img_p),
                # Intentionally omit mask_paths to prove zero mask dependency
            }
        )

    tf = SimCLRViewTransform(
        height=64,
        width=64,
        random_resized_crop_scale=(0.5, 1.0),
        horizontal_flip_prob=0.5,
        vertical_flip_prob=0.5,
        random_rotate90_prob=0.5,
        color_jitter_prob=1.0,
        gaussian_blur_prob=0.5,
    )
    ds = SimCLRUnlabeledDataset(
        manifest_or_records=records,
        transform=tf,
        return_metadata=True,
        filter_train_split=False,
    )
    assert len(ds) == 4
    for rec in ds.records:
        assert "mask_paths" not in rec

    set_seed(42)
    v1, v2, meta = ds[0]
    assert v1.shape == (3, 64, 64)
    assert v2.shape == (3, 64, 64)
    assert v1.dtype == torch.float32
    assert v2.dtype == torch.float32
    assert torch.isfinite(v1).all()
    assert torch.isfinite(v2).all()
    # Two independent stochastic views from the same image must differ
    assert not torch.equal(v1, v2)
    assert meta["sample_id"] == "UNLABELED_00"
    assert meta["split"] == "train"

    # Reproducibility check: resetting seed produces identical (v1, v2)
    set_seed(42)
    v1_repeat, v2_repeat, _ = ds[0]
    assert torch.equal(v1, v1_repeat)
    assert torch.equal(v2, v2_repeat)

    loader = build_simclr_dataloader(
        dataset=ds,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        seed=42,
    )
    b_v1, b_v2, b_meta = next(iter(loader))
    assert b_v1.shape == (2, 3, 64, 64)
    assert b_v2.shape == (2, 3, 64, 64)
    assert len(b_meta["sample_id"]) == 2


def test_11_ssl_split_isolation_and_manifest_selection() -> None:
    """Verify select_ssl_training_records selects strictly 'train' samples and rejects 'valid'/'test'."""
    from src.data.dataset import (
        DatasetSampleError,
        SimCLRUnlabeledDataset,
        select_ssl_training_records,
    )

    ddr_records = select_ssl_training_records("data/processed/manifests/ddr_manifest.json")
    idrid_records = select_ssl_training_records("data/processed/manifests/idrid_manifest.json")
    combined_records = select_ssl_training_records("data/processed/manifests/combined_manifest.json")
    combined_ddr_only = select_ssl_training_records(
        "data/processed/manifests/combined_manifest.json",
        allowed_datasets=["DDR"],
    )

    assert len(ddr_records) == 383
    assert len(idrid_records) == 54
    assert len(combined_records) == 437
    assert len(combined_ddr_only) == 383
    assert all(r["split"] == "train" for r in combined_records)
    assert all("mask_paths" not in r for r in combined_records)

    # Passing a 'valid' or 'test' record with filter_train_split=False must raise DatasetSampleError
    forbidden_val = [
        {
            "dataset_name": "DDR",
            "sample_id": "BAD_VAL_01",
            "split": "valid",
            "image_path": "data/raw/DDR/some_image.jpg",
        }
    ]
    with pytest.raises(DatasetSampleError, match="Forbidden split 'valid'"):
        SimCLRUnlabeledDataset(
            manifest_or_records=forbidden_val,
            filter_train_split=False,
        )

    forbidden_test = [
        {
            "dataset_name": "IDRiD",
            "sample_id": "BAD_TEST_01",
            "split": "test",
            "image_path": "data/raw/IDRiD/some_image.jpg",
        }
    ]
    with pytest.raises(DatasetSampleError, match="Forbidden split 'test'"):
        SimCLRUnlabeledDataset(
            manifest_or_records=forbidden_test,
            filter_train_split=False,
        )


def test_12_simclr_trainer_step_checkpoint_resume_and_strict_pretrained_guard(
    tmp_path: Path,
) -> None:
    """Verify SimCLRTrainer updates parameters, saves/resumes checkpoints, and blocks silent fallback."""
    from unittest.mock import patch
    from torch.utils.data import DataLoader
    from scripts.pretrain_simclr import _SyntheticTwoViewDataset
    from src.data.dataset import simclr_collate_fn
    from src.engine.simclr_trainer import (
        SimCLRTrainer,
        build_ssl_optimizer_from_config,
        build_ssl_scheduler_from_config,
    )
    from src.engine.trainer import TestSplitLeakageError

    ssl_cfg = load_yaml_config("configs/ssl.yaml", validate=True)
    ssl_cfg["pretrained"] = False
    ssl_cfg["learning_rate"] = 1e-3
    ssl_cfg["warmup_epochs"] = 1
    ssl_cfg["epochs"] = 2

    torch.manual_seed(42)
    model = build_simclr_from_config(ssl_cfg, pretrained_override=False)
    criterion = build_nt_xent_loss_from_config(ssl_cfg)
    optimizer = build_ssl_optimizer_from_config(model, ssl_cfg)
    scheduler = build_ssl_scheduler_from_config(optimizer, ssl_cfg, total_epochs=2)

    ds = _SyntheticTwoViewDataset(num_samples=4, image_size=64, seed=42)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=simclr_collate_fn)

    initial_enc = {
        k: v.detach().clone() for k, v in model.encoder.named_parameters() if v.requires_grad
    }
    initial_proj = {
        k: v.detach().clone()
        for k, v in model.projection_head.named_parameters()
        if v.requires_grad
    }

    ckpt_dir = tmp_path / "ssl_ckpts"
    trainer = SimCLRTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu",
        checkpoint_dir=ckpt_dir,
        mixed_precision=False,
        seed=42,
        config={"ssl": ssl_cfg},
    )

    res1 = trainer.fit(loader, num_epochs=1, save_checkpoints=True)
    assert res1["epochs_completed"] == 1
    assert trainer.global_step == 2
    assert (ckpt_dir / "latest_ssl_checkpoint.pt").is_file()
    assert (ckpt_dir / "best_ssl_checkpoint.pt").is_file()
    assert (ckpt_dir / "best_simclr_resnet50.pt").is_file()

    # Verify both encoder and projection head parameters updated
    assert any(
        not torch.equal(initial_enc[k], v.detach())
        for k, v in model.encoder.named_parameters()
        if v.requires_grad
    )
    assert any(
        not torch.equal(initial_proj[k], v.detach())
        for k, v in model.projection_head.named_parameters()
        if v.requires_grad
    )

    # Resume into a fresh trainer for Epoch 2
    model2 = build_simclr_from_config(ssl_cfg, pretrained_override=False)
    optimizer2 = build_ssl_optimizer_from_config(model2, ssl_cfg)
    scheduler2 = build_ssl_scheduler_from_config(optimizer2, ssl_cfg, total_epochs=2)
    trainer2 = SimCLRTrainer(
        model=model2,
        criterion=criterion,
        optimizer=optimizer2,
        scheduler=scheduler2,
        device="cpu",
        checkpoint_dir=ckpt_dir,
        mixed_precision=False,
        seed=42,
        config={"ssl": ssl_cfg},
    )
    loaded = trainer2.load_checkpoint(ckpt_dir / "latest_ssl_checkpoint.pt")
    assert loaded["epoch"] == 1
    assert "encoder_state_dict" in loaded
    assert trainer2.start_epoch == 2
    assert trainer2.global_step == 2

    res2 = trainer2.fit(loader, num_epochs=2, save_checkpoints=True)
    assert res2["epochs_completed"] == 2
    assert trainer2.global_step == 4
    assert len(res2["history"]) == 2

    # Verify silent fallback to random weights is rejected by SimCLRTrainer when pretrained_requested=True
    with (
        patch.object(Path, "glob", return_value=[]),
        patch(
            "src.models.simclr.ResNet50_Weights.DEFAULT.get_state_dict",
            side_effect=RuntimeError("Offline"),
        ),
    ):
        fallback_model = SimCLRModel(pretrained=True, strict_pretrained=False)
        assert fallback_model.pretrained_requested is True
        assert fallback_model.pretrained_weights_loaded is False
        with pytest.raises(
            PretrainedWeightsError,
            match="Refusing to silently fall back to random initialization",
        ):
            SimCLRTrainer(
                model=fallback_model,
                criterion=criterion,
                optimizer=torch.optim.AdamW(fallback_model.parameters(), lr=1e-4),
                device="cpu",
            )

    # Verify test-split leakage is rejected in SimCLRTrainer
    bad_ds = _SyntheticTwoViewDataset(num_samples=2, image_size=64, seed=42)
    bad_ds.records[0]["split"] = "test"
    bad_loader = DataLoader(bad_ds, batch_size=2, collate_fn=simclr_collate_fn)
    with pytest.raises(TestSplitLeakageError, match="Forbidden 'test'-split sample"):
        trainer.train_one_epoch(bad_loader, epoch=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")
def test_13_gpu_simclr_smoke_test(tmp_path: Path) -> None:
    """Run a bounded GPU smoke test verifying forward, loss, backward, optimizer step, and checkpoint writing."""
    from scripts.pretrain_simclr import run_simclr_smoke_test

    base_cfg = load_yaml_config("configs/project.yaml", validate=True)
    data_cfg = load_yaml_config("configs/dataset.yaml", validate=True)
    ssl_cfg = load_yaml_config("configs/ssl.yaml", validate=True)
    ssl_cfg["pretrained"] = False
    ssl_cfg["encoder_weights_source"] = "random"

    report_path = tmp_path / "simclr_gpu_smoke.json"
    report = run_simclr_smoke_test(
        base_cfg=base_cfg,
        data_cfg=data_cfg,
        ssl_cfg=ssl_cfg,
        device=torch.device("cuda"),
        batch_size=4,
        image_size=128,
        checkpoint_dir=tmp_path / "smoke_ckpts",
        report_path=report_path,
    )

    assert report["status"] == "passed"
    assert report["gpu_memory"]["cuda_used"] is True
    assert report["gpu_memory"]["max_memory_allocated_mb"] > 0.0
    assert torch.isfinite(torch.tensor(report["train_loss"]))
    assert Path(report["latest_checkpoint"]).is_file()
    assert Path(report["best_checkpoint"]).is_file()
    assert report_path.is_file()


def test_14_simclr_encoder_loading_into_unet_and_four_channel_output(
    tmp_path: Path,
) -> None:
    """Verify SimCLR encoder checkpoint loads into ResNet50UNet and produces finite 4-channel segmentation output."""
    from src.models.unet import build_unet_from_config

    torch.manual_seed(123)
    simclr = SimCLRModel(
        in_channels=3,
        projection_dim=128,
        projection_hidden_dim=256,
        pretrained=False,
    )
    enc_ckpt_path = tmp_path / "best_simclr_encoder.pt"
    torch.save(
        {
            "epoch": 50,
            "global_step": 2400,
            "best_train_loss": 0.9836,
            "seed": 42,
            "backbone": "resnet50",
            "encoder_state_dict": simclr.export_encoder_state_dict(),
        },
        enc_ckpt_path,
    )

    model_cfg = load_yaml_config("configs/model.yaml", validate=False)
    model_cfg["pretrained"] = True
    model_cfg["strict_pretrained"] = True
    model_cfg["encoder_weights_source"] = "ssl_checkpoint"
    model_cfg["ssl_checkpoint_path"] = str(enc_ckpt_path)
    validate_model_config(model_cfg)

    unet = build_unet_from_config(model_cfg, strict_pretrained_override=True)
    summary = unet.get_model_summary()
    assert unet.pretrained_weights_loaded is True
    assert summary["encoder_weights_source"] == "ssl_checkpoint"
    assert summary["loaded_encoder_parameter_count"] == 318
    assert summary["weights_status"] == f"loaded_checkpoint:{enc_ckpt_path.name}"
    assert summary["out_channels"] == 4
    assert summary["output_channel_order"] == ["MA", "HE", "EX", "SE"]
    assert summary["trainable_parameters"] == summary["total_parameters"]

    # Confirm exact parameter equality between SimCLR encoder and U-Net encoder
    exported_unet_enc = unet.export_encoder_state_dict()
    exported_simclr_enc = simclr.export_encoder_state_dict()
    assert set(exported_unet_enc.keys()) == set(exported_simclr_enc.keys())
    for k in exported_simclr_enc:
        assert torch.equal(exported_simclr_enc[k], exported_unet_enc[k])

    # Confirm 4-channel segmentation forward pass
    unet.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        logits = unet(x)
        probs = unet.predict_probabilities(x)
    assert logits.shape == (2, 4, 64, 64)
    assert probs.shape == (2, 4, 64, 64)
    assert torch.isfinite(logits).all()
    assert torch.all((probs >= 0.0) & (probs <= 1.0))


def test_15_simclr_finetuning_checkpoint_save_resume_and_joint_updates(
    tmp_path: Path,
) -> None:
    """Verify SimCLR-initialized U-Net fine-tunes both encoder & decoder and saves/resumes checkpoints."""
    from torch.utils.data import DataLoader
    from src.engine.trainer import (
        SupervisedSegmentationTrainer,
        build_optimizer_from_config,
    )
    from src.models.losses import build_loss_from_config
    from src.models.unet import build_unet_from_config
    from tests.test_unet_baseline import SyntheticSegmentationDataset

    torch.manual_seed(42)
    simclr = SimCLRModel(pretrained=False)
    enc_ckpt_path = tmp_path / "simclr_encoder.pt"
    torch.save({"encoder_state_dict": simclr.export_encoder_state_dict()}, enc_ckpt_path)

    model_cfg = load_yaml_config("configs/model.yaml", validate=False)
    model_cfg["pretrained"] = True
    model_cfg["strict_pretrained"] = True
    model_cfg["encoder_weights_source"] = "ssl_checkpoint"
    model_cfg["ssl_checkpoint_path"] = str(enc_ckpt_path)
    model_cfg["training"] = {
        **model_cfg["training"],
        "learning_rate": 1e-3,
        "mixed_precision": False,
    }

    unet = build_unet_from_config(model_cfg, strict_pretrained_override=True)
    criterion = build_loss_from_config(model_cfg)
    optimizer = build_optimizer_from_config(unet, model_cfg)

    initial_enc_conv1 = unet.encoder_conv1.weight.detach().clone()
    initial_dec4 = next(unet.decoder4.parameters()).detach().clone()
    initial_head = unet.segmentation_head.weight.detach().clone()

    train_ds = SyntheticSegmentationDataset(num_samples=4, image_size=64, split="train", seed=42)
    val_ds = SyntheticSegmentationDataset(num_samples=2, image_size=64, split="valid", seed=43)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    ckpt_dir = tmp_path / "simclr_finetune_ckpts"
    trainer = SupervisedSegmentationTrainer(
        model=unet,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu",
        checkpoint_dir=ckpt_dir,
        mixed_precision=False,
        seed=42,
        config={"model": model_cfg},
    )

    res1 = trainer.fit(train_loader, val_loader, num_epochs=1, save_checkpoints=True)
    assert res1["epochs_completed"] == 1
    assert (ckpt_dir / "latest_checkpoint.pt").is_file()
    assert (ckpt_dir / "best_checkpoint.pt").is_file()

    # Verify encoder, decoder, and segmentation head all updated jointly
    assert not torch.equal(initial_enc_conv1, unet.encoder_conv1.weight.detach())
    assert not torch.equal(initial_dec4, next(unet.decoder4.parameters()).detach())
    assert not torch.equal(initial_head, unet.segmentation_head.weight.detach())

    # Resume into Epoch 2
    unet2 = build_unet_from_config(model_cfg, strict_pretrained_override=True)
    optimizer2 = build_optimizer_from_config(unet2, model_cfg)
    trainer2 = SupervisedSegmentationTrainer(
        model=unet2,
        criterion=criterion,
        optimizer=optimizer2,
        device="cpu",
        checkpoint_dir=ckpt_dir,
        mixed_precision=False,
        seed=42,
        config={"model": model_cfg},
    )
    loaded = trainer2.load_checkpoint(ckpt_dir / "latest_checkpoint.pt")
    assert loaded["epoch"] == 1
    assert trainer2.start_epoch == 2
    res2 = trainer2.fit(train_loader, val_loader, num_epochs=2, save_checkpoints=True)
    assert res2["epochs_completed"] == 2
    assert len(res2["history"]) == 2


def test_16_validation_comparison_report_generation(tmp_path: Path) -> None:
    """Verify build_validation_comparison_report computes macro & per-class deltas without touching test data."""
    import json
    from src.engine.trainer import build_validation_comparison_report

    baseline_report = {
        "best_epoch": 50,
        "best_metric_value": 0.558165,
        "model": {"weights_status": "loaded_imagenet_cached:resnet50-11ad3fa6.pth"},
        "best_validation_metrics": {
            "epoch": 50,
            "val_loss": 0.235686,
            "val_dice_macro": 0.558165,
            "val_iou_macro": 0.397178,
            "per_class": {
                "MA": {"dice": 0.344349, "iou": 0.207984},
                "HE": {"dice": 0.588948, "iou": 0.417382},
                "EX": {"dice": 0.675494, "iou": 0.509997},
                "SE": {"dice": 0.623869, "iou": 0.453350},
            },
        },
    }
    baseline_path = tmp_path / "baseline_report.json"
    baseline_path.write_text(json.dumps(baseline_report, indent=2), encoding="utf-8")

    candidate_report = {
        "best_epoch": 48,
        "best_metric_value": 0.578165,
        "model": {"weights_status": "loaded_checkpoint:best_simclr_encoder.pt"},
        "best_validation_metrics": {
            "epoch": 48,
            "val_loss": 0.225686,
            "val_dice_macro": 0.578165,
            "val_iou_macro": 0.417178,
            "per_class": {
                "MA": {"dice": 0.364349, "iou": 0.227984},
                "HE": {"dice": 0.608948, "iou": 0.437382},
                "EX": {"dice": 0.685494, "iou": 0.519997},
                "SE": {"dice": 0.653869, "iou": 0.483350},
            },
        },
    }

    comp = build_validation_comparison_report(
        candidate_report=candidate_report,
        baseline_report_or_path=baseline_path,
    )
    assert comp["baseline_best_epoch"] == 50
    assert comp["candidate_best_epoch"] == 48
    assert comp["test_split_evaluated"] is False
    assert "do not establish held-out DDR test-set performance" in comp["validation_scope_note"]
    assert comp["improved_over_baseline_dice"] is True
    assert comp["improved_over_baseline_iou"] is True
    assert comp["macro_metrics"]["val_dice_macro"]["absolute_diff"] == pytest.approx(0.02, abs=1e-6)
    assert comp["macro_metrics"]["val_iou_macro"]["absolute_diff"] == pytest.approx(0.02, abs=1e-6)
    for ch in ("MA", "HE", "EX", "SE"):
        assert ch in comp["per_class_dice"]
        assert comp["per_class_dice"][ch]["absolute_diff"] > 0.0
        assert comp["per_class_dice"][ch]["relative_pct_change"] > 0.0


def test_17_simclr_finetuning_excludes_test_split_and_idrid() -> None:
    """Verify _select_train_val_records selects only DDR train (383) and valid (149) and excludes test (225) and IDRiD."""
    from scripts.train_baseline import _select_train_val_records
    from src.data.manifests import load_manifest_json

    ddr_manifest = load_manifest_json("data/processed/manifests/ddr_manifest.json")
    train_recs, val_recs = _select_train_val_records(
        ddr_manifest["records"],
        allowed_datasets=["DDR"],
    )
    assert len(train_recs) == 383
    assert len(val_recs) == 149
    assert all(r["dataset_name"] == "DDR" and r["split"] == "train" for r in train_recs)
    assert all(r["dataset_name"] == "DDR" and r["split"] == "valid" for r in val_recs)

    combined_manifest = load_manifest_json("data/processed/manifests/combined_manifest.json")
    c_train, c_val = _select_train_val_records(
        combined_manifest["records"],
        allowed_datasets=["DDR"],
    )
    assert len(c_train) == 383
    assert len(c_val) == 149
    assert all(r["dataset_name"] == "DDR" for r in (c_train + c_val))


def test_18_simclr_warmup_learning_rate_logging(tmp_path: Path) -> None:
    """Verify SimCLRTrainer logs the active warmup learning rate used during the epoch and next_learning_rate separately (M-1)."""
    from torch.utils.data import DataLoader
    from scripts.pretrain_simclr import _SyntheticTwoViewDataset
    from src.data.dataset import simclr_collate_fn
    from src.engine.simclr_trainer import (
        SimCLRTrainer,
        build_ssl_optimizer_from_config,
        build_ssl_scheduler_from_config,
    )

    ssl_cfg = load_yaml_config("configs/ssl.yaml", validate=True)
    ssl_cfg["pretrained"] = False
    ssl_cfg["learning_rate"] = 3.0e-4
    ssl_cfg["lr_scheduler"] = "cosine_annealing"
    ssl_cfg["warmup_epochs"] = 5

    torch.manual_seed(42)
    model = build_simclr_from_config(ssl_cfg, pretrained_override=False)
    criterion = build_nt_xent_loss_from_config(ssl_cfg)
    optimizer = build_ssl_optimizer_from_config(model, ssl_cfg)
    scheduler = build_ssl_scheduler_from_config(optimizer, ssl_cfg, total_epochs=10)

    ds = _SyntheticTwoViewDataset(num_samples=4, image_size=64, seed=42)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=simclr_collate_fn)

    trainer = SimCLRTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu",
        checkpoint_dir=tmp_path / "lr_test_ckpts",
        mixed_precision=False,
        seed=42,
    )

    res = trainer.fit(loader, num_epochs=2, save_checkpoints=False)
    ep1, ep2 = res["history"]

    # LinearLR starts at start_factor=0.1 -> 3e-5 during Epoch 1, steps to 8.4e-5 after Epoch 1 (for 5 warmup epochs)
    expected_ep1_active_lr = 3.0e-4 * 0.1
    expected_ep2_active_lr = 3.0e-4 * (0.1 + 0.9 / 5.0)
    expected_ep3_active_lr = 3.0e-4 * (0.1 + 2.0 * 0.9 / 5.0)

    assert ep1["learning_rate"] == pytest.approx(expected_ep1_active_lr, rel=1e-6)
    assert ep1["next_learning_rate"] == pytest.approx(expected_ep2_active_lr, rel=1e-6)
    assert ep2["learning_rate"] == pytest.approx(expected_ep2_active_lr, rel=1e-6)
    assert ep2["next_learning_rate"] == pytest.approx(expected_ep3_active_lr, rel=1e-6)


def test_19_simclr_interrupted_and_resumed_training_reproducibility_cpu(
    tmp_path: Path,
) -> None:
    """Verify SimCLR interrupted-and-resumed training reproduces uninterrupted training exactly on CPU (M-3)."""
    from scripts.pretrain_simclr import _SyntheticTwoViewDataset
    from src.data.dataset import build_simclr_dataloader
    from src.engine.simclr_trainer import (
        SimCLRTrainer,
        build_ssl_optimizer_from_config,
        build_ssl_scheduler_from_config,
    )
    from src.utils.seed import set_seed

    ssl_cfg = load_yaml_config("configs/ssl.yaml", validate=True)
    ssl_cfg["pretrained"] = False
    ssl_cfg["learning_rate"] = 5.0e-4
    ssl_cfg["warmup_epochs"] = 2

    # 1. Run 2 epochs uninterrupted with shuffle=True
    set_seed(seed=77, deterministic=True)
    ds_full = _SyntheticTwoViewDataset(num_samples=6, image_size=64, seed=77)
    loader_full = build_simclr_dataloader(
        dataset=ds_full,
        ssl_config=ssl_cfg,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        seed=77,
    )
    set_seed(seed=77, deterministic=True)
    model_full = build_simclr_from_config(ssl_cfg, pretrained_override=False)
    opt_full = build_ssl_optimizer_from_config(model_full, ssl_cfg)
    sched_full = build_ssl_scheduler_from_config(opt_full, ssl_cfg, total_epochs=4)
    trainer_full = SimCLRTrainer(
        model=model_full,
        criterion=build_nt_xent_loss_from_config(ssl_cfg),
        optimizer=opt_full,
        scheduler=sched_full,
        device="cpu",
        checkpoint_dir=tmp_path / "ssl_full",
        mixed_precision=False,
        seed=77,
    )
    res_full = trainer_full.fit(loader_full, num_epochs=2, save_checkpoints=True)

    # 2. Run 1 epoch, interrupt, reload into fresh instances, and resume for Epoch 2
    set_seed(seed=77, deterministic=True)
    ds_part = _SyntheticTwoViewDataset(num_samples=6, image_size=64, seed=77)
    loader_part1 = build_simclr_dataloader(
        dataset=ds_part,
        ssl_config=ssl_cfg,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        seed=77,
    )
    set_seed(seed=77, deterministic=True)
    model_part1 = build_simclr_from_config(ssl_cfg, pretrained_override=False)
    opt_part1 = build_ssl_optimizer_from_config(model_part1, ssl_cfg)
    sched_part1 = build_ssl_scheduler_from_config(opt_part1, ssl_cfg, total_epochs=4)
    ckpt_dir = tmp_path / "ssl_resume"
    trainer_part1 = SimCLRTrainer(
        model=model_part1,
        criterion=build_nt_xent_loss_from_config(ssl_cfg),
        optimizer=opt_part1,
        scheduler=sched_part1,
        device="cpu",
        checkpoint_dir=ckpt_dir,
        mixed_precision=False,
        seed=77,
    )
    trainer_part1.fit(loader_part1, num_epochs=1, save_checkpoints=True)

    # Perturb RNG state before resuming
    set_seed(seed=999, deterministic=True)
    loader_part2 = build_simclr_dataloader(
        dataset=ds_part,
        ssl_config=ssl_cfg,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        seed=77,
    )
    model_part2 = build_simclr_from_config(ssl_cfg, pretrained_override=False)
    opt_part2 = build_ssl_optimizer_from_config(model_part2, ssl_cfg)
    sched_part2 = build_ssl_scheduler_from_config(opt_part2, ssl_cfg, total_epochs=4)
    trainer_part2 = SimCLRTrainer(
        model=model_part2,
        criterion=build_nt_xent_loss_from_config(ssl_cfg),
        optimizer=opt_part2,
        scheduler=sched_part2,
        device="cpu",
        checkpoint_dir=ckpt_dir,
        mixed_precision=False,
        seed=77,
    )
    loaded = trainer_part2.load_checkpoint(ckpt_dir / "latest_ssl_checkpoint.pt")
    assert "rng_state" in loaded
    res_resumed = trainer_part2.fit(loader_part2, num_epochs=2, save_checkpoints=True)

    assert res_resumed["history"][0]["train_loss"] == pytest.approx(
        res_full["history"][0]["train_loss"], abs=1e-7
    )
    assert res_resumed["history"][1]["train_loss"] == pytest.approx(
        res_full["history"][1]["train_loss"], abs=1e-7
    )
    for (k1, v1), (k2, v2) in zip(
        model_full.state_dict().items(), model_part2.state_dict().items()
    ):
        assert k1 == k2
        assert torch.allclose(v1, v2, atol=1e-7), f"Parameter mismatch on {k1}"




