"""Unit and integration tests for the Flower NumPyClient and parameter serialization (src/federated/client.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flwr.client import Client, NumPyClient
from flwr.common import Parameters
import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.data.dataset import (
    RetinalLesionSegmentationDataset,
    build_segmentation_dataloader,
)
from src.data.masks import CANONICAL_LESION_CHANNELS
from src.data.partitioning import partition_training_records
from src.engine.trainer import TestSplitLeakageError
from src.federated.client import (
    FedMorphisNumPyClient,
    ParameterSerializationError,
    build_federated_client,
    flower_parameters_to_ndarrays,
    get_model_parameters,
    get_model_state_keys,
    model_to_flower_parameters,
    set_model_parameters,
)
from src.models.unet import ResNet50UNet


class _TinyBNUNet(nn.Module):
    """Lightweight segmentation model with Conv2d + BatchNorm2d for fast unit tests."""

    def __init__(self, in_channels: int = 3, out_channels: int = 4) -> None:
        super().__init__()
        self.enc_conv = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, bias=False)
        self.enc_bn = nn.BatchNorm2d(8)
        self.act = nn.ReLU(inplace=True)
        self.head = nn.Conv2d(8, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.act(self.enc_bn(self.enc_conv(x))))


def _create_synthetic_sample(
    root_dir: Path,
    sample_id: str,
    split: str = "train",
    dataset_name: str = "DDR",
    height: int = 32,
    width: int = 32,
    omit_classes: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Create a synthetic retinal image and 4-channel lesion masks on disk."""
    img_arr = np.zeros((height, width, 3), dtype=np.uint8)
    img_arr[4 : height - 4, 4 : width - 4, 0] = 170
    img_arr[4 : height - 4, 4 : width - 4, 1] = 120

    boxes = {
        "MA": (6, 12, 6, 12),
        "HE": (12, 18, 12, 18),
        "EX": (18, 24, 18, 24),
        "SE": (22, 28, 8, 16),
    }

    if dataset_name == "DDR":
        mask_dir = "segmentation label" if split == "valid" else "label"
        img_path = (
            root_dir
            / f"DDR-dataset/lesion_segmentation/{split}/image/{sample_id}.png"
        )
    else:
        split_folder = "a. Training Set" if split == "train" else "b. Testing Set"
        img_path = (
            root_dir
            / f"A. Segmentation/1. Original Images/{split_folder}/{sample_id}.png"
        )

    img_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_arr, mode="RGB").save(img_path)

    mask_paths: dict[str, str | None] = {}
    pixel_counts: dict[str, int] = {}
    present_list: list[str] = []
    missing_list: list[str] = []
    positive_list: list[str] = []

    for cls_name in CANONICAL_LESION_CHANNELS:
        if cls_name in omit_classes:
            mask_paths[cls_name] = None
            pixel_counts[cls_name] = 0
            missing_list.append(cls_name)
            continue

        m_arr = np.zeros((height, width), dtype=np.uint8)
        y0, y1, x0, x1 = boxes[cls_name]
        m_arr[y0:y1, x0:x1] = 255 if dataset_name == "DDR" else 1
        positive_list.append(cls_name)

        if dataset_name == "DDR":
            m_path = (
                root_dir
                / f"DDR-dataset/lesion_segmentation/{split}/{mask_dir}/{cls_name}/{sample_id}.tif"
            )
        else:
            folder_map = {
                "MA": "1. Microaneurysms",
                "HE": "2. Haemorrhages",
                "EX": "3. Hard Exudates",
                "SE": "4. Soft Exudates",
            }
            m_path = (
                root_dir
                / f"A. Segmentation/2. All Segmentation Groundtruths/{split_folder}/{folder_map[cls_name]}/{sample_id}_{cls_name}.tif"
            )

        m_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(m_arr, mode="L").save(m_path)
        mask_paths[cls_name] = str(m_path)
        pixel_counts[cls_name] = int((m_arr > 0).sum())
        present_list.append(cls_name)

    return {
        "dataset_name": dataset_name,
        "sample_id": sample_id,
        "split": split,
        "image_path": str(img_path),
        "mask_paths": mask_paths,
        "channel_order": list(CANONICAL_LESION_CHANNELS),
        "present_lesion_masks": present_list,
        "missing_lesion_masks": missing_list,
        "synthesized_zero_channels": list(missing_list),
        "empty_lesion_masks": [],
        "positive_lesion_masks": positive_list,
        "has_missing_masks": len(missing_list) > 0,
        "lesion_pixel_counts": pixel_counts,
        "image_dimensions": {"height": height, "width": width, "channels": 3},
        "canonical_mask_dimensions": [height, width, 4],
    }


def _build_synthetic_records_and_partition(
    tmp_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build synthetic DDR (4 train, 2 valid, 1 test) and IDRiD (5 train, 0 valid, 1 test) records."""
    records: list[dict[str, Any]] = []
    for i in range(4):
        records.append(
            _create_synthetic_sample(
                tmp_path, f"ddr_tr_{i:02d}", split="train", dataset_name="DDR"
            )
        )
    for i in range(2):
        records.append(
            _create_synthetic_sample(
                tmp_path, f"ddr_val_{i:02d}", split="valid", dataset_name="DDR"
            )
        )
    records.append(
        _create_synthetic_sample(
            tmp_path, "ddr_test_00", split="test", dataset_name="DDR"
        )
    )

    for i in range(5):
        records.append(
            _create_synthetic_sample(
                tmp_path,
                f"IDRiD_{i + 1:02d}",
                split="train",
                dataset_name="IDRiD",
                omit_classes=("SE",),
            )
        )
    records.append(
        _create_synthetic_sample(
            tmp_path,
            "IDRiD_test_01",
            split="test",
            dataset_name="IDRiD",
            omit_classes=("SE",),
        )
    )

    partition = partition_training_records(
        records, num_clients=2, strategy="cross_dataset", seed=42
    )
    return records, partition


def test_parameter_serialization_exact_roundtrip_and_integer_buffers() -> None:
    """Verify exact state_dict round-trip, key ordering, float32/int64 dtypes, and Flower Parameters conversion."""
    model_src = _TinyBNUNet()
    model_src.train()
    # Run forward passes so BatchNorm running_mean, running_var, and num_batches_tracked are non-trivial
    dummy = torch.randn(3, 3, 32, 32)
    _ = model_src(dummy)
    _ = model_src(dummy * 2.0)

    assert int(model_src.enc_bn.num_batches_tracked.item()) == 2
    assert model_src.enc_bn.num_batches_tracked.dtype == torch.int64

    keys = get_model_state_keys(model_src, include_buffers=True)
    assert keys == tuple(model_src.state_dict().keys())
    assert "enc_bn.running_mean" in keys
    assert "enc_bn.running_var" in keys
    assert "enc_bn.num_batches_tracked" in keys

    arrays = get_model_parameters(model_src, include_buffers=True)
    assert len(arrays) == len(keys)

    nb_idx = keys.index("enc_bn.num_batches_tracked")
    assert arrays[nb_idx].dtype == np.int64
    assert arrays[nb_idx].shape == ()
    assert int(arrays[nb_idx].item()) == 2

    # Round-trip through Flower Parameters protobuf
    flwr_params = model_to_flower_parameters(model_src, include_buffers=True)
    assert isinstance(flwr_params, Parameters)
    unpacked_arrays = flower_parameters_to_ndarrays(flwr_params)

    model_dst = _TinyBNUNet()
    # Perturb destination weights and buffers before loading
    for p in model_dst.parameters():
        nn.init.constant_(p, -99.0)
    model_dst.enc_bn.num_batches_tracked.fill_(777)

    set_model_parameters(model_dst, unpacked_arrays, include_buffers=True)

    for k, src_t in model_src.state_dict().items():
        dst_t = model_dst.state_dict()[k]
        assert dst_t.dtype == src_t.dtype
        assert dst_t.shape == src_t.shape
        assert torch.equal(src_t, dst_t), f"Mismatch in restored state_dict key '{k}'"


def test_parameter_validation_rejects_invalid_count_shape_and_dtype() -> None:
    """Verify invalid parameter counts, shapes, dtypes, and non-finite values fail atomically."""
    model = _TinyBNUNet()
    _ = model(torch.randn(2, 3, 32, 32))
    valid_arrays = get_model_parameters(model, include_buffers=True)
    keys = get_model_state_keys(model, include_buffers=True)
    snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # 1. Invalid parameter count (too few / too many)
    with pytest.raises(ParameterSerializationError, match="Parameter count mismatch"):
        set_model_parameters(model, valid_arrays[:-1])

    with pytest.raises(ParameterSerializationError, match="Parameter count mismatch"):
        set_model_parameters(model, [*valid_arrays, valid_arrays[0]])

    # 2. Invalid parameter shape
    bad_shape_arrays = [a.copy() for a in valid_arrays]
    bad_shape_arrays[0] = np.zeros((99, 3, 3, 3), dtype=np.float32)
    with pytest.raises(ParameterSerializationError, match="Shape mismatch"):
        set_model_parameters(model, bad_shape_arrays)

    # 3. Invalid floating-point dtype (float64 instead of float32)
    bad_float_dtype = [a.copy() for a in valid_arrays]
    bad_float_dtype[0] = bad_float_dtype[0].astype(np.float64)
    with pytest.raises(ParameterSerializationError, match="Dtype mismatch"):
        set_model_parameters(model, bad_float_dtype)

    # 4. Invalid integer buffer dtype (float32 passed for int64 num_batches_tracked)
    nb_idx = keys.index("enc_bn.num_batches_tracked")
    bad_int_dtype = [a.copy() for a in valid_arrays]
    bad_int_dtype[nb_idx] = np.array(2.5, dtype=np.float32)
    with pytest.raises(ParameterSerializationError, match="Dtype mismatch"):
        set_model_parameters(model, bad_int_dtype)

    # 5. Negative BatchNorm counter rejected
    neg_nb = [a.copy() for a in valid_arrays]
    neg_nb[nb_idx] = np.array(-5, dtype=np.int64)
    with pytest.raises(
        ParameterSerializationError, match="Negative value detected in BatchNorm counter"
    ):
        set_model_parameters(model, neg_nb)

    # 6. NaN / Inf in floating-point weights rejected
    nan_arrays = [a.copy() for a in valid_arrays]
    nan_arrays[0].flat[0] = np.nan
    with pytest.raises(ParameterSerializationError, match="Non-finite"):
        set_model_parameters(model, nan_arrays)

    # Verify model state remained 100% untouched after all failed loads
    for k, orig_t in snapshot.items():
        assert torch.equal(model.state_dict()[k], orig_t)


def test_numpy_client_fit_evaluate_and_dataset_mapping(tmp_path: Path) -> None:
    """Verify client factory maps client IDs to DDR and IDRiD and executes fit/evaluate on CPU."""
    records, partition = _build_synthetic_records_and_partition(tmp_path)

    # Client 0 -> DDR (4 train, 2 official valid)
    client_ddr = build_federated_client(
        client_id=0,
        partition_manifest=partition,
        source_manifest=records,
        model=_TinyBNUNet(),
        device="cpu",
        local_epochs=2,
        batch_size=2,
        num_workers=0,
        seed=42,
        image_size=(32, 32),
        verify_files_exist=True,
    )
    assert isinstance(client_ddr, NumPyClient)
    assert isinstance(client_ddr, FedMorphisNumPyClient)
    assert isinstance(client_ddr.to_client(), Client)
    assert client_ddr.client_id == "client_0"
    assert client_ddr.dataset_names == ("DDR",)

    props_ddr = client_ddr.get_properties({})
    assert props_ddr["client_id"] == "client_0"
    assert props_ddr["dataset"] == "DDR"
    assert props_ddr["num_train_samples"] == 4
    assert props_ddr["num_val_samples"] == 2

    # Client 1 -> IDRiD (5 official train -> 4 local train + 1 carved valid)
    client_idrid = build_federated_client(
        client_id="client_1",
        partition_manifest=partition,
        source_manifest=records,
        model=_TinyBNUNet(),
        device="cpu",
        local_epochs=1,
        batch_size=2,
        num_workers=0,
        seed=42,
        image_size=(32, 32),
        verify_files_exist=True,
    )
    assert client_idrid.client_id == "client_1"
    assert client_idrid.dataset_names == ("IDRiD",)
    props_idrid = client_idrid.get_properties({})
    assert props_idrid["num_train_samples"] == 4
    assert props_idrid["num_val_samples"] == 1

    # Execute fit on DDR client
    init_params = client_ddr.get_parameters({})
    updated_params, num_train, fit_metrics = client_ddr.fit(
        init_params, {"server_round": 1, "local_epochs": 2}
    )

    assert num_train == 4
    assert len(updated_params) == len(init_params)
    assert any(not np.array_equal(p0, p1) for p0, p1 in zip(init_params, updated_params))
    assert fit_metrics["client_id"] == "client_0"
    assert fit_metrics["dataset"] == "DDR"
    assert fit_metrics["local_epochs"] == 2
    assert np.isfinite(float(fit_metrics["loss"]))
    assert np.isfinite(float(fit_metrics["train_loss"]))
    assert 0.0 <= float(fit_metrics["dice_macro"]) <= 1.0
    assert 0.0 <= float(fit_metrics["iou_macro"]) <= 1.0
    for ch in CANONICAL_LESION_CHANNELS:
        assert f"dice_{ch}" in fit_metrics
        assert f"iou_{ch}" in fit_metrics

    # Execute evaluate on IDRiD client and confirm evaluate does not mutate model weights
    pre_eval_snapshot = [a.copy() for a in updated_params]
    val_loss, num_val, eval_metrics = client_idrid.evaluate(
        updated_params, {"server_round": 1}
    )
    post_eval_params = client_idrid.get_parameters({})

    assert isinstance(val_loss, float)
    assert np.isfinite(val_loss)
    assert num_val == 1
    assert eval_metrics["client_id"] == "client_1"
    assert eval_metrics["dataset"] == "IDRiD"
    assert 0.0 <= float(eval_metrics["dice_macro"]) <= 1.0
    assert 0.0 <= float(eval_metrics["iou_macro"]) <= 1.0
    assert all(
        np.array_equal(a, b) for a, b in zip(pre_eval_snapshot, post_eval_params)
    )


def test_real_resnet50_unet_and_trainer_integration_on_cpu(tmp_path: Path) -> None:
    """Integration test exercising the actual ResNet50UNet and SupervisedSegmentationTrainer via FedMorphisNumPyClient."""
    records, partition = _build_synthetic_records_and_partition(tmp_path)

    client = build_federated_client(
        client_id="DDR",
        partition_manifest=partition,
        source_manifest=records,
        pretrained=False,
        strict_pretrained=False,
        device="cpu",
        local_epochs=1,
        batch_size=2,
        num_workers=0,
        seed=42,
        image_size=(32, 32),
        max_train_batches=1,
        max_val_batches=1,
        verify_files_exist=True,
    )

    assert isinstance(client.model, ResNet50UNet)
    assert client.model.pretrained_weights_loaded is False

    # Verify full ResNet50UNet state_dict round-trip (including all 318+ encoder/decoder tensors & int64 BN buffers)
    initial_params = client.get_parameters({})
    state_keys = get_model_state_keys(client.model, include_buffers=True)
    assert len(initial_params) == len(state_keys)
    assert any(arr.dtype == np.int64 for arr in initial_params)

    updated_params, num_fit_samples, fit_metrics = client.fit(
        initial_params, {"server_round": 1, "local_epochs": 1, "max_batches": 1}
    )
    assert num_fit_samples == 2
    assert len(updated_params) == len(initial_params)
    assert any(
        not np.array_equal(a, b) for a, b in zip(initial_params, updated_params)
    )
    assert np.isfinite(float(fit_metrics["train_loss"]))

    val_loss, num_val_samples, val_metrics = client.evaluate(
        updated_params, {"server_round": 1, "max_val_batches": 1}
    )
    assert np.isfinite(val_loss)
    assert num_val_samples == 2
    assert 0.0 <= float(val_metrics["val_dice_macro"]) <= 1.0
    assert 0.0 <= float(val_metrics["val_iou_macro"]) <= 1.0


def test_client_rejects_test_split_leakage(tmp_path: Path) -> None:
    """Verify FedMorphisNumPyClient rejects any DataLoader containing held-out test records."""
    records, _ = _build_synthetic_records_and_partition(tmp_path)
    train_recs = [r for r in records if r["dataset_name"] == "DDR" and r["split"] == "train"]
    test_recs = [r for r in records if r["dataset_name"] == "DDR" and r["split"] == "test"]

    train_ds = RetinalLesionSegmentationDataset(train_recs, split="train", image_size=(32, 32))
    test_ds = RetinalLesionSegmentationDataset(test_recs, split="test", image_size=(32, 32))

    train_loader = build_segmentation_dataloader(train_ds, batch_size=2, num_workers=0, seed=42)
    test_loader = build_segmentation_dataloader(test_ds, batch_size=1, num_workers=0, seed=42)

    with pytest.raises(TestSplitLeakageError):
        FedMorphisNumPyClient(
            client_id="client_0",
            client_index=0,
            model=_TinyBNUNet(),
            train_loader=train_loader,
            val_loader=test_loader,
            device="cpu",
        )
