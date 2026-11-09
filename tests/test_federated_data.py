"""Unit tests for the federated client data module (src/federated/data.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.masks import CANONICAL_LESION_CHANNELS
from src.data.partitioning import partition_training_records
from src.engine.evaluator import verify_exact_split
from src.engine.trainer import verify_no_test_split_leakage
from src.federated.data import (
    ClientDataBundle,
    FederatedDataError,
    build_all_federated_client_dataloaders,
    build_federated_client_dataloaders,
    carve_client_validation_records,
    resolve_client_id,
    resolve_client_train_val_records,
    verify_client_record_files,
    verify_client_split_integrity,
)
from src.utils.seed import set_seed


def _create_synthetic_sample_on_disk(
    root_dir: Path,
    sample_id: str,
    split: str = "train",
    dataset_name: str = "DDR",
    height: int = 48,
    width: int = 64,
    active_classes: tuple[str, ...] = ("MA", "HE", "EX", "SE"),
    omit_classes: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Create a synthetic retinal image and 4-class lesion masks on disk and return a manifest record."""
    img_arr = np.zeros((height, width, 3), dtype=np.uint8)
    img_arr[4 : height - 4, 6 : width - 6, 0] = 160
    img_arr[4 : height - 4, 6 : width - 6, 1] = 110

    boxes = {
        "MA": (8, 14, 10, 18),
        "HE": (16, 24, 20, 30),
        "EX": (26, 34, 32, 42),
        "SE": (34, 40, 44, 52),
    }
    # Asymmetric bright landmark in top-left of FOV so geometric flips are detectable
    img_arr[6:12, 8:14, 2] = 250

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
    empty_list: list[str] = []

    for cls_name in CANONICAL_LESION_CHANNELS:
        if cls_name in omit_classes:
            mask_paths[cls_name] = None
            pixel_counts[cls_name] = 0
            missing_list.append(cls_name)
            continue

        m_arr = np.zeros((height, width), dtype=np.uint8)
        if cls_name in active_classes:
            by0, by1, bx0, bx1 = boxes[cls_name]
            m_arr[by0:by1, bx0:bx1] = 255 if dataset_name == "DDR" else 1
            positive_list.append(cls_name)
        else:
            empty_list.append(cls_name)

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
        "empty_lesion_masks": empty_list,
        "positive_lesion_masks": positive_list,
        "has_missing_masks": len(missing_list) > 0,
        "lesion_pixel_counts": pixel_counts,
        "image_dimensions": {"height": height, "width": width, "channels": 3},
        "canonical_mask_dimensions": [height, width, 4],
    }


def _build_synthetic_two_client_fixture(tmp_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build synthetic on-disk DDR (train+valid+test) and IDRiD (train+test) records and a 2-client partition."""
    records: list[dict[str, Any]] = []

    # DDR: 6 train, 3 official valid, 2 test
    for i in range(6):
        records.append(
            _create_synthetic_sample_on_disk(
                tmp_path, f"ddr_tr_{i:02d}", split="train", dataset_name="DDR"
            )
        )
    for i in range(3):
        records.append(
            _create_synthetic_sample_on_disk(
                tmp_path, f"ddr_val_{i:02d}", split="valid", dataset_name="DDR"
            )
        )
    for i in range(2):
        records.append(
            _create_synthetic_sample_on_disk(
                tmp_path, f"ddr_te_{i:02d}", split="test", dataset_name="DDR"
            )
        )

    # IDRiD: 10 train (with missing HE/SE on some samples), 0 official valid, 2 test
    for i in range(10):
        omit = ("HE", "SE") if i % 2 == 1 else ("SE",)
        active = ("MA", "EX") if i % 2 == 1 else ("MA", "HE", "EX")
        records.append(
            _create_synthetic_sample_on_disk(
                tmp_path,
                f"IDRiD_{i + 1:02d}",
                split="train",
                dataset_name="IDRiD",
                active_classes=active,
                omit_classes=omit,
            )
        )
    for i in range(2):
        records.append(
            _create_synthetic_sample_on_disk(
                tmp_path,
                f"IDRiD_test_{i + 1:02d}",
                split="test",
                dataset_name="IDRiD",
                active_classes=("MA", "EX"),
                omit_classes=("SE",),
            )
        )

    partition = partition_training_records(
        records, num_clients=2, strategy="cross_dataset", seed=42
    )
    return records, partition


def test_real_manifest_ddr_and_idrid_client_counts_and_disjointness() -> None:
    """Verify configured 2-client DDR (383/149) and IDRiD (44/10) counts and zero leakage on repository manifests."""
    bundles = build_all_federated_client_dataloaders(
        batch_size=8,
        num_workers=0,
        seed=42,
        verify_files_exist=False,
    )
    assert list(bundles.keys()) == ["client_0", "client_1"]

    ddr_bundle = bundles["client_0"]
    idrid_bundle = bundles["client_1"]

    # Client 0: DDR (383 train, 149 official valid)
    assert isinstance(ddr_bundle, ClientDataBundle)
    assert ddr_bundle.client_id == "client_0"
    assert ddr_bundle.client_index == 0
    assert ddr_bundle.dataset_names == ("DDR",)
    assert ddr_bundle.num_train_samples == 383
    assert ddr_bundle.num_val_samples == 149
    assert ddr_bundle.val_split_mode == "official_and_carved"
    assert ddr_bundle.val_source == "official"
    assert ddr_bundle.channel_order == ("MA", "HE", "EX", "SE")
    assert all(
        r["dataset_name"] == "DDR" and r["split"] == "train"
        for r in ddr_bundle.train_records
    )
    assert all(
        r["dataset_name"] == "DDR" and r["split"] == "valid"
        for r in ddr_bundle.val_records
    )

    # Client 1: IDRiD (44 remaining train, 10 deterministically carved valid from 54 official train)
    assert isinstance(idrid_bundle, ClientDataBundle)
    assert idrid_bundle.client_id == "client_1"
    assert idrid_bundle.client_index == 1
    assert idrid_bundle.dataset_names == ("IDRiD",)
    assert idrid_bundle.num_train_samples == 44
    assert idrid_bundle.num_val_samples == 10
    assert idrid_bundle.num_train_samples + idrid_bundle.num_val_samples == 54
    assert idrid_bundle.val_split_mode == "official_and_carved"
    assert idrid_bundle.val_source == "carved_from_train"
    assert idrid_bundle.channel_order == ("MA", "HE", "EX", "SE")
    assert all(
        r["dataset_name"] == "IDRiD" and r["split"] == "train"
        for r in idrid_bundle.train_records
    )
    assert all(
        r["dataset_name"] == "IDRiD"
        and r["split"] == "valid"
        and r.get("original_split") == "train"
        and r.get("val_source") == "carved_from_train"
        for r in idrid_bundle.val_records
    )

    # Verify trainer/evaluator split-hygiene checks pass on both clients
    verify_no_test_split_leakage(ddr_bundle.train_loader, stage_name="train")
    verify_no_test_split_leakage(ddr_bundle.val_loader, stage_name="validation")
    assert verify_exact_split(ddr_bundle.val_loader, expected_split="valid") == 149

    verify_no_test_split_leakage(idrid_bundle.train_loader, stage_name="train")
    verify_no_test_split_leakage(idrid_bundle.val_loader, stage_name="validation")
    assert verify_exact_split(idrid_bundle.val_loader, expected_split="valid") == 10

    # Verify IDRiD train and carved val sets are completely disjoint and conserve all 54 IDRiD train IDs
    idrid_train_ids = {r["sample_id"] for r in idrid_bundle.train_records}
    idrid_val_ids = {r["sample_id"] for r in idrid_bundle.val_records}
    assert len(idrid_train_ids) == 44
    assert len(idrid_val_ids) == 10
    assert idrid_train_ids.isdisjoint(idrid_val_ids)


def test_deterministic_idrid_validation_carve_reproducibility() -> None:
    """Verify the 20% IDRiD validation carve is 100% reproducible under seed 42 and order-invariant."""
    train_a, val_a, _ = resolve_client_train_val_records(
        client_id="IDRiD", seed=42, val_carve_ratio=0.2
    )
    train_b, val_b, _ = resolve_client_train_val_records(
        client_id=1, seed=42, val_carve_ratio=0.2
    )

    assert len(train_a) == 44
    assert len(val_a) == 10
    assert [r["sample_id"] for r in train_a] == [r["sample_id"] for r in train_b]
    assert [r["sample_id"] for r in val_a] == [r["sample_id"] for r in val_b]

    # Re-carving from a reversed copy of the 54 IDRiD train records produces the exact same split
    combined_54 = [dict(r, split="train") for r in (train_a + val_a)]
    rem_rev, val_rev = carve_client_validation_records(
        list(reversed(combined_54)), val_carve_ratio=0.2, seed=42
    )
    assert [r["sample_id"] for r in rem_rev] == [r["sample_id"] for r in train_a]
    assert [r["sample_id"] for r in val_rev] == [r["sample_id"] for r in val_a]

    # Different seed produces a different carved validation subset of size 10
    _, val_diff_seed = carve_client_validation_records(
        combined_54, val_carve_ratio=0.2, seed=999
    )
    assert len(val_diff_seed) == 10
    assert [r["sample_id"] for r in val_diff_seed] != [r["sample_id"] for r in val_a]


def test_synthetic_ddr_and_idrid_client_dataloaders_end_to_end(tmp_path: Path) -> None:
    """Verify DDR and IDRiD client DataLoaders load images and 4-channel masks in [MA, HE, EX, SE] order."""
    records, partition = _build_synthetic_two_client_fixture(tmp_path)

    # Client 0 (DDR): 6 train, 3 official valid
    c0_bundle = build_federated_client_dataloaders(
        client_id=0,
        partition_manifest=partition,
        source_manifest=records,
        batch_size=4,
        num_workers=0,
        seed=42,
        image_size=(32, 32),
        return_metadata=True,
        verify_files_exist=True,
    )
    assert c0_bundle.client_id == "client_0"
    assert c0_bundle.dataset_names == ("DDR",)
    assert c0_bundle.num_train_samples == 6
    assert c0_bundle.num_val_samples == 3
    assert c0_bundle.val_source == "official"

    c0_imgs, c0_masks, c0_meta = next(iter(c0_bundle.train_loader))
    assert c0_imgs.shape == (4, 3, 32, 32)
    assert c0_masks.shape == (4, 4, 32, 32)
    assert c0_imgs.dtype == torch.float32
    assert c0_masks.dtype == torch.float32
    assert set(torch.unique(c0_masks).tolist()).issubset({0.0, 1.0})
    assert all(ch == ["MA", "HE", "EX", "SE"] for ch in c0_meta["channel_order"])
    assert all(ds == "DDR" for ds in c0_meta["dataset_name"])

    # Client 1 (IDRiD): 10 official train -> 8 local train + 2 carved valid (20%)
    c1_bundle = build_federated_client_dataloaders(
        client_id="client_1",
        partition_manifest=partition,
        source_manifest=records,
        batch_size=4,
        num_workers=0,
        seed=42,
        image_size=(32, 32),
        return_metadata=True,
        verify_files_exist=True,
    )
    assert c1_bundle.client_id == "client_1"
    assert c1_bundle.dataset_names == ("IDRiD",)
    assert c1_bundle.num_train_samples == 8
    assert c1_bundle.num_val_samples == 2
    assert c1_bundle.val_source == "carved_from_train"

    c1_val_imgs, c1_val_masks, c1_val_meta = next(iter(c1_bundle.val_loader))
    assert c1_val_imgs.shape == (2, 3, 32, 32)
    assert c1_val_masks.shape == (2, 4, 32, 32)
    assert set(torch.unique(c1_val_masks).tolist()).issubset({0.0, 1.0})
    assert all(ch == ["MA", "HE", "EX", "SE"] for ch in c1_val_meta["channel_order"])
    assert all(ds == "IDRiD" for ds in c1_val_meta["dataset_name"])
    # Channel 3 (SE) was omitted on all synthetic IDRiD samples -> synthesized as all-zero channel
    assert float(c1_val_masks[:, 3, :, :].sum().item()) == 0.0
    assert all("SE" in missing for missing in c1_val_meta["missing_lesion_masks"])


def test_validation_transforms_are_deterministic_while_training_augments(
    tmp_path: Path,
) -> None:
    """Verify validation transforms do not apply training augmentations and remain deterministic."""
    records, partition = _build_synthetic_two_client_fixture(tmp_path)

    bundle = build_federated_client_dataloaders(
        client_id="IDRiD",
        partition_manifest=partition,
        source_manifest=records,
        batch_size=2,
        num_workers=0,
        seed=42,
        image_size=(32, 32),
        verify_files_exist=True,
    )

    # Inspect transform configuration flags directly
    assert getattr(bundle.train_dataset.transform, "is_train", False) is True
    assert getattr(bundle.train_dataset.transform, "augmentation_enabled", False) is True
    assert getattr(bundle.val_dataset.transform, "is_train", True) is False
    assert getattr(bundle.val_dataset.transform, "augmentation_enabled", True) is False

    # Validation dataset returns bit-identical tensors across arbitrary seeds
    set_seed(1)
    val_img_1, val_mask_1 = bundle.val_dataset[0]
    set_seed(99999)
    val_img_2, val_mask_2 = bundle.val_dataset[0]
    assert torch.equal(val_img_1, val_img_2)
    assert torch.equal(val_mask_1, val_mask_2)

    # Training dataset applies stochastic augmentations across repeated calls
    set_seed(42)
    train_samples = [bundle.train_dataset[0][0] for _ in range(8)]
    assert any(not torch.allclose(train_samples[0], s) for s in train_samples[1:])


def test_invalid_client_ids_rejected(tmp_path: Path) -> None:
    """Verify invalid client IDs fail clearly with FederatedDataError."""
    records, partition = _build_synthetic_two_client_fixture(tmp_path)
    clients_map = partition["clients"]

    # Valid resolutions
    assert resolve_client_id(0, clients_map) == ("client_0", 0)
    assert resolve_client_id("0", clients_map) == ("client_0", 0)
    assert resolve_client_id("client_0", clients_map) == ("client_0", 0)
    assert resolve_client_id("DDR", clients_map) == ("client_0", 0)
    assert resolve_client_id(1, clients_map) == ("client_1", 1)
    assert resolve_client_id("1", clients_map) == ("client_1", 1)
    assert resolve_client_id("client_1", clients_map) == ("client_1", 1)
    assert resolve_client_id("IDRiD", clients_map) == ("client_1", 1)

    for bad_id in (-1, 2, 99, True, False, None, "", "   ", "client_99", "UnknownDS"):
        with pytest.raises(FederatedDataError):
            resolve_client_id(bad_id, clients_map)  # type: ignore[arg-type]

    with pytest.raises(FederatedDataError, match="Unknown client_id"):
        build_federated_client_dataloaders(
            client_id="client_99",
            partition_manifest=partition,
            source_manifest=records,
        )


def test_missing_manifest_entries_and_integrity_failures_rejected(
    tmp_path: Path,
) -> None:
    """Verify missing manifest files, missing keys, empty splits, and split leakage fail clearly."""
    records, partition = _build_synthetic_two_client_fixture(tmp_path)

    # 1. Non-existent partition manifest path
    with pytest.raises(FileNotFoundError, match="Partition manifest not found"):
        build_federated_client_dataloaders(
            client_id=0,
            partition_manifest=tmp_path / "nonexistent_partition.json",
        )

    # 2. Missing 'clients' key in partition manifest
    with pytest.raises(FederatedDataError, match="missing a valid 'clients' mapping"):
        build_federated_client_dataloaders(
            client_id=0,
            partition_manifest={"schema_version": "1.0"},
        )

    # 3. Empty client record list
    with pytest.raises(FederatedDataError, match="empty"):
        build_federated_client_dataloaders(
            client_id=0,
            partition_manifest={"clients": {"client_0": []}},
        )

    # 4. Malformed record missing required fields
    with pytest.raises(FederatedDataError, match="missing required field"):
        build_federated_client_dataloaders(
            client_id=0,
            partition_manifest={
                "clients": {"client_0": [{"dataset_name": "DDR", "sample_id": "bad"}]}
            },
        )

    # 5. Cross-client duplicate training record in partition manifest
    dup_rec = dict(partition["clients"]["client_0"][0])
    bad_partition = {
        "clients": {
            "client_0": list(partition["clients"]["client_0"]),
            "client_1": [*partition["clients"]["client_1"], dup_rec],
        }
    }
    with pytest.raises(FederatedDataError, match="Duplicate training sample"):
        build_federated_client_dataloaders(
            client_id=0,
            partition_manifest=bad_partition,
            source_manifest=records,
        )

    # 6. Insufficient training records (< 2) for validation carve
    single_idrid_partition = {
        "clients": {
            "client_0": list(partition["clients"]["client_0"]),
            "client_1": [dict(partition["clients"]["client_1"][0])],
        }
    }
    with pytest.raises(FederatedDataError, match="at least 2 training records"):
        build_federated_client_dataloaders(
            client_id=1,
            partition_manifest=single_idrid_partition,
            source_manifest=records,
        )

    # 7. Test record passed in validation_records override
    test_rec = [r for r in records if r["split"] == "test"][0]
    with pytest.raises(FederatedDataError, match="Forbidden 'test'-split record"):
        build_federated_client_dataloaders(
            client_id=0,
            partition_manifest=partition,
            source_manifest=records,
            validation_records=[test_rec],
        )

    # 8. Overlap between train_records and val_records rejected by verify_client_split_integrity
    train_rec_as_val = dict(partition["clients"]["client_0"][0], split="valid")
    with pytest.raises(FederatedDataError, match="Training and validation splits overlap"):
        verify_client_split_integrity(
            client_id="client_0",
            train_records=partition["clients"]["client_0"],
            val_records=[train_rec_as_val],
        )


def test_missing_data_files_detected_clearly(tmp_path: Path) -> None:
    """Verify missing image files or missing lesion mask files raise FileNotFoundError."""
    records, partition = _build_synthetic_two_client_fixture(tmp_path)

    # Delete one training image file on disk
    victim_record = partition["clients"]["client_0"][0]
    victim_img_path = Path(victim_record["image_path"])
    victim_img_path.unlink()

    with pytest.raises(FileNotFoundError, match="Missing image file"):
        build_federated_client_dataloaders(
            client_id=0,
            partition_manifest=partition,
            source_manifest=records,
            verify_files_exist=True,
        )

    # Also verify Dataset indexing raises FileNotFoundError when verify_files_exist=False
    bundle_lazy = build_federated_client_dataloaders(
        client_id=0,
        partition_manifest=partition,
        source_manifest=records,
        verify_files_exist=False,
    )
    with pytest.raises(FileNotFoundError, match="Source image file not found"):
        _ = bundle_lazy.train_dataset[0]

    # Restore image file, then delete one lesion mask file on disk
    Image.fromarray(np.zeros((48, 64, 3), dtype=np.uint8), mode="RGB").save(
        victim_img_path
    )
    victim_mask_path = Path(victim_record["mask_paths"]["MA"])
    victim_mask_path.unlink()

    with pytest.raises(FileNotFoundError, match="Missing 'MA' mask file"):
        verify_client_record_files(partition["clients"]["client_0"], client_id="client_0")
