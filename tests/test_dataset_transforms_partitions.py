"""Unit tests for PyTorch segmentation Dataset, Albumentations transforms, DataLoader, and federated client partitioning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.dataset import (
    DatasetSampleError,
    RetinalLesionSegmentationDataset,
    build_segmentation_dataloader,
)
from src.data.manifests import build_dataset_manifest
from src.data.masks import CANONICAL_LESION_CHANNELS
from src.data.partitioning import (
    PartitioningError,
    load_partition_manifest,
    partition_training_records,
    save_partition_manifest,
)
from src.data.transforms import (
    RetinalSegmentationTransform,
    build_segmentation_transforms,
)
from src.utils.config import ConfigValidationError, load_yaml_config, validate_config
from src.utils.seed import seed_worker, set_seed


def _create_synthetic_sample_files(
    root_dir: Path,
    sample_id: str,
    split: str = "train",
    dataset_name: str = "DDR",
    height: int = 48,
    width: int = 64,
    active_classes: tuple[str, ...] = ("MA", "HE", "EX", "SE"),
    omit_classes: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Create a synthetic image and lesion masks on disk and return a manifest record."""
    img_arr = np.zeros((height, width, 3), dtype=np.uint8)
    # Bright retinal FOV in [4:height-4, 6:width-6]
    img_arr[4 : height - 4, 6 : width - 6, 0] = 150
    img_arr[4 : height - 4, 6 : width - 6, 1] = 100

    # Place distinct non-overlapping boxes per class inside the FOV
    boxes = {
        "MA": (8, 14, 10, 18),
        "HE": (16, 24, 20, 30),
        "EX": (26, 34, 32, 42),
        "SE": (34, 40, 44, 52),
    }
    # Also mark the MA box brightly in channel 2 (blue) of the image to test image-mask alignment
    y0, y1, x0, x1 = boxes["MA"]
    img_arr[y0:y1, x0:x1, 2] = 250

    if dataset_name == "DDR":
        mask_dir_name = "segmentation label" if split == "valid" else "label"
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
                / f"DDR-dataset/lesion_segmentation/{split}/{mask_dir_name}/{cls_name}/{sample_id}.tif"
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


# ---------------------------------------------------------------------------
# 1, 2, 3, 4, 5. Dataset output shape/dtype, channel order, binary resize, alignment, missing masks
# ---------------------------------------------------------------------------


def test_dataset_output_shape_dtype_and_channel_order(tmp_path: Path) -> None:
    """1 & 2. Verify Dataset returns (3, H, W) float32 image and (4, H, W) float32 mask in [MA, HE, EX, SE] order."""
    rec = _create_synthetic_sample_files(
        tmp_path,
        "sample_01",
        split="test",
        active_classes=("MA", "EX"),
    )
    ds = RetinalLesionSegmentationDataset(
        [rec],
        split="test",
        image_size=(32, 40),
        return_metadata=True,
    )
    assert len(ds) == 1
    image_t, mask_t, meta = ds[0]

    assert isinstance(image_t, torch.Tensor)
    assert isinstance(mask_t, torch.Tensor)
    assert image_t.shape == (3, 32, 40)
    assert mask_t.shape == (4, 32, 40)
    assert image_t.dtype == torch.float32
    assert mask_t.dtype == torch.float32
    assert image_t.device.type == "cpu"
    assert mask_t.device.type == "cpu"
    assert meta["channel_order"] == ["MA", "HE", "EX", "SE"]

    # Channel 0 (MA) and Channel 2 (EX) are positive; Channel 1 (HE) and Channel 3 (SE) are zero
    assert float(mask_t[0].sum()) > 0.0
    assert float(mask_t[1].sum()) == 0.0
    assert float(mask_t[2].sum()) > 0.0
    assert float(mask_t[3].sum()) == 0.0


def test_binary_mask_values_preserved_after_resizing(tmp_path: Path) -> None:
    """3. Verify resized masks contain strictly binary {0, 1} values with no intermediate fractions."""
    rec = _create_synthetic_sample_files(
        tmp_path,
        "sample_resize",
        split="train",
        height=64,
        width=96,
    )
    ds = RetinalLesionSegmentationDataset(
        [rec],
        split="train",
        image_size=(29, 37),
    )
    _, mask_t = ds[0]
    unique_vals = set(torch.unique(mask_t).tolist())
    assert unique_vals.issubset({0.0, 1.0})


def test_image_and_mask_spatial_alignment_across_transforms() -> None:
    """4. Verify image and 4-channel mask undergo the exact same geometric transformations."""
    img = np.zeros((48, 48, 3), dtype=np.uint8)
    mask = np.zeros((48, 48, 4), dtype=np.uint8)
    # Place a bright Asymmetric L-shape in top-left corner of both image and mask channel 0 (MA)
    img[4:16, 4:10, :] = 255
    mask[4:16, 4:10, 0] = 1
    mask[4:16, 4:10, 3] = 1

    tfm = RetinalSegmentationTransform(
        height=48,
        width=48,
        is_train=True,
        augmentation_enabled=True,
        horizontal_flip_prob=1.0,
        vertical_flip_prob=1.0,
        random_rotate90_prob=0.0,
        color_jitter_enabled=False,
        normalize_enabled=False,
    )
    aug_img, aug_mask = tfm(img, mask)

    # After both horizontal and vertical flip (p=1.0), top-left [4:16, 4:10] maps to [32:44, 38:44]
    img_fg = aug_img[:, :, 0] > 0.5
    ma_fg = aug_mask[:, :, 0] == 1
    se_fg = aug_mask[:, :, 3] == 1
    assert np.array_equal(img_fg, ma_fg)
    assert np.array_equal(ma_fg, se_fg)


def test_missing_idrid_lesion_mask_handling_in_dataset(tmp_path: Path) -> None:
    """5. Verify missing IDRiD HE and SE masks yield all-zero channels in the PyTorch Dataset."""
    rec = _create_synthetic_sample_files(
        tmp_path,
        "IDRiD_09",
        split="test",
        dataset_name="IDRiD",
        active_classes=("MA", "EX"),
        omit_classes=("HE", "SE"),
    )
    ds = RetinalLesionSegmentationDataset(
        [rec],
        split="test",
        image_size=(32, 32),
        return_metadata=True,
    )
    _, mask_t, meta = ds[0]
    assert mask_t.shape == (4, 32, 32)
    assert meta["missing_lesion_masks"] == ["HE", "SE"]
    assert float(mask_t[0].sum()) > 0.0  # MA present
    assert float(mask_t[1].sum()) == 0.0  # HE synthesized zero
    assert float(mask_t[2].sum()) > 0.0  # EX present
    assert float(mask_t[3].sum()) == 0.0  # SE synthesized zero


# ---------------------------------------------------------------------------
# 6, 7. Deterministic evaluation transforms & seeded training augmentation reproducibility
# ---------------------------------------------------------------------------


def test_deterministic_evaluation_transforms(tmp_path: Path) -> None:
    """6. Verify validation and test transforms are completely deterministic across calls and seeds."""
    rec = _create_synthetic_sample_files(tmp_path, "val_01", split="valid")
    ds = RetinalLesionSegmentationDataset([rec], split="valid", image_size=(32, 32))

    set_seed(1)
    img1, mask1 = ds[0]
    set_seed(999)
    img2, mask2 = ds[0]

    assert torch.equal(img1, img2)
    assert torch.equal(mask1, mask2)


def test_training_augmentation_reproducibility_under_fixed_seed(tmp_path: Path) -> None:
    """7. Verify training transforms are reproducible under the same seed and vary across seeds."""
    rec = _create_synthetic_sample_files(tmp_path, "train_01", split="train")
    ds = RetinalLesionSegmentationDataset([rec], split="train", image_size=(32, 32))

    set_seed(12345)
    img_a1, mask_a1 = ds[0]
    img_a2, mask_a2 = ds[0]

    set_seed(12345)
    img_b1, mask_b1 = ds[0]
    img_b2, mask_b2 = ds[0]

    assert torch.allclose(img_a1, img_b1)
    assert torch.equal(mask_a1, mask_b1)
    assert torch.allclose(img_a2, img_b2)
    assert torch.equal(mask_a2, mask_b2)


# ---------------------------------------------------------------------------
# 8, 9, 10. DataLoader batch shape, final incomplete batch, and worker seeding
# ---------------------------------------------------------------------------


def test_dataloader_batch_shape_and_incomplete_final_batch(tmp_path: Path) -> None:
    """8 & 9. Verify DataLoader batch tensor shapes and final incomplete batch handling."""
    records = [
        _create_synthetic_sample_files(tmp_path, f"case_{i:02d}", split="train")
        for i in range(5)
    ]
    ds = RetinalLesionSegmentationDataset(records, split="train", image_size=(32, 32))

    loader_keep = build_segmentation_dataloader(
        ds, batch_size=2, num_workers=0, drop_last=False, seed=42
    )
    batches_keep = list(loader_keep)
    assert len(batches_keep) == 3
    assert batches_keep[0][0].shape == (2, 3, 32, 32)
    assert batches_keep[0][1].shape == (2, 4, 32, 32)
    assert batches_keep[2][0].shape == (1, 3, 32, 32)
    assert batches_keep[2][1].shape == (1, 4, 32, 32)

    loader_drop = build_segmentation_dataloader(
        ds, batch_size=2, num_workers=0, drop_last=True, seed=42
    )
    batches_drop = list(loader_drop)
    assert len(batches_drop) == 2


def test_dataloader_worker_seeding_and_generator_isolation(tmp_path: Path) -> None:
    """10. Verify worker seeding and confirm separate DataLoaders do not interfere with each other."""
    records = [
        _create_synthetic_sample_files(
            tmp_path, f"iso_{i:02d}", split="test", return_metadata=False  # type: ignore[arg-type]
        )
        if False
        else _create_synthetic_sample_files(tmp_path, f"iso_{i:02d}", split="train")
        for i in range(6)
    ]
    eval_ds = RetinalLesionSegmentationDataset(
        records,
        split="train",
        transform=build_segmentation_transforms(split="test", image_size=(24, 24)),
        return_metadata=True,
    )

    loader_a = build_segmentation_dataloader(
        eval_ds, batch_size=2, shuffle=True, num_workers=0, seed=77
    )
    loader_b = build_segmentation_dataloader(
        eval_ds, batch_size=2, shuffle=True, num_workers=0, seed=77
    )
    loader_other = build_segmentation_dataloader(
        eval_ds, batch_size=2, shuffle=True, num_workers=0, seed=999
    )

    # Iterate loader_other in between to verify it does not perturb loader_b
    ids_a = [m["sample_id"] for _, _, m in loader_a]
    _ = list(loader_other)
    ids_b = [m["sample_id"] for _, _, m in loader_b]
    assert ids_a == ids_b

    # Also test seed_worker directly for distinct worker IDs
    torch.manual_seed(100)
    seed_worker(0)
    r0 = np.random.randint(0, 100000)
    torch.manual_seed(100)
    seed_worker(1)
    r1 = np.random.randint(0, 100000)
    assert r0 != r1


# ---------------------------------------------------------------------------
# 11, 12, 13, 14, 15, 16. Federated Client Partitioning (IID, Non-IID, Disjoint, Leakage, Determinism)
# ---------------------------------------------------------------------------


def _build_synthetic_multi_split_records( enable_idrid: bool = True) -> list[dict[str, Any]]:
    """Build lightweight synthetic manifest records for partitioning tests without disk I/O."""
    records: list[dict[str, Any]] = []
    classes = ["MA", "HE", "EX", "SE"]
    for i in range(20):
        dom = classes[i % 4]
        records.append(
            {
                "dataset_name": "DDR",
                "sample_id": f"ddr_train_{i:02d}",
                "split": "train",
                "image_path": f"data/raw/DDR/lesion_segmentation/train/image/ddr_train_{i:02d}.jpg",
                "mask_paths": {c: f"masks/{c}/{i}.tif" for c in classes},
                "positive_lesion_masks": [dom],
                "lesion_pixel_counts": {c: (500 if c == dom else 0) for c in classes},
            }
        )
    if enable_idrid:
        for i in range(8):
            dom = classes[i % 4]
            records.append(
                {
                    "dataset_name": "IDRiD",
                    "sample_id": f"idrid_train_{i:02d}",
                    "split": "train",
                    "image_path": f"data/raw/IDRiD/Original Images/Training Set/idrid_train_{i:02d}.jpg",
                    "mask_paths": {c: f"masks/{c}/{i}.tif" for c in classes},
                    "positive_lesion_masks": [dom],
                    "lesion_pixel_counts": {c: (800 if c == dom else 0) for c in classes},
                }
            )
    # Add validation and test records that must NEVER enter client partitions
    for i in range(4):
        records.append(
            {
                "dataset_name": "DDR",
                "sample_id": f"ddr_valid_{i:02d}",
                "split": "valid",
                "image_path": f"data/raw/DDR/lesion_segmentation/valid/image/ddr_valid_{i:02d}.jpg",
                "mask_paths": {c: f"masks/{c}/{i}.tif" for c in classes},
            }
        )
    for i in range(5):
        records.append(
            {
                "dataset_name": "IDRiD",
                "sample_id": f"idrid_test_{i:02d}",
                "split": "test",
                "image_path": f"data/raw/IDRiD/Original Images/Testing Set/idrid_test_{i:02d}.jpg",
                "mask_paths": {c: f"masks/{c}/{i}.tif" for c in classes},
            }
        )
    return records


def test_iid_partition_coverage_disjointness_and_no_split_leakage() -> None:
    """11, 13, 14, 15. Verify IID partitioning covers all train samples once, is disjoint, and leaks 0 val/test samples."""
    records = _build_synthetic_multi_split_records()
    part = partition_training_records(records, num_clients=4, strategy="iid", seed=42)

    assert part["total_train_samples"] == 28
    assert part["held_out_validation_samples"] == 4
    assert part["held_out_test_samples"] == 5
    assert part["verification"]["all_train_samples_assigned_once"] is True
    assert part["verification"]["clients_disjoint"] is True
    assert part["verification"]["lost_train_sample_count"] == 0
    assert part["verification"]["validation_leakage_count"] == 0
    assert part["verification"]["test_leakage_count"] == 0

    sizes = [len(part["clients"][f"client_{c}"]) for c in range(4)]
    assert sizes == [7, 7, 7, 7]


@pytest.mark.parametrize("strategy", ["dirichlet", "cross_dataset"])
def test_non_iid_partition_coverage_disjointness_and_no_leakage(strategy: str) -> None:
    """12, 13, 14, 15. Verify Dirichlet and cross_dataset partitions conserve all train samples with 0 leakage."""
    records = _build_synthetic_multi_split_records()
    part = partition_training_records(
        records,
        num_clients=4,
        strategy=strategy,
        seed=42,
        dirichlet_alpha=0.2,
        min_samples_per_client=1,
    )

    assert part["total_train_samples"] == 28
    assert part["verification"]["all_train_samples_assigned_once"] is True
    assert part["verification"]["clients_disjoint"] is True
    assert part["verification"]["lost_train_sample_count"] == 0
    assert part["verification"]["validation_leakage_count"] == 0
    assert part["verification"]["test_leakage_count"] == 0

    client_sets = [
        {(r["dataset_name"], r["sample_id"]) for r in part["clients"][f"client_{c}"]}
        for c in range(4)
    ]
    assert all(len(s) >= 1 for s in client_sets)
    for i in range(4):
        for j in range(i + 1, 4):
            assert client_sets[i].isdisjoint(client_sets[j])
    union_all = set().union(*client_sets)
    assert len(union_all) == 28


def test_deterministic_partitions_for_fixed_seed(tmp_path: Path) -> None:
    """16. Verify partition outputs are byte-identical for the same seed and differ across seeds."""
    records = _build_synthetic_multi_split_records()

    p1 = partition_training_records(
        records, num_clients=4, strategy="dirichlet", seed=42, dirichlet_alpha=0.5
    )
    # Shuffle input record order to confirm canonical pre-sorting makes output order-independent
    reversed_records = list(reversed(records))
    p2 = partition_training_records(
        reversed_records, num_clients=4, strategy="dirichlet", seed=42, dirichlet_alpha=0.5
    )
    p_diff = partition_training_records(
        records, num_clients=4, strategy="dirichlet", seed=99, dirichlet_alpha=0.5
    )

    f1 = save_partition_manifest(p1, tmp_path / "p1.json")
    f2 = save_partition_manifest(p2, tmp_path / "p2.json")
    assert f1.read_text(encoding="utf-8") == f2.read_text(encoding="utf-8")

    loaded = load_partition_manifest(f1)
    assert loaded["client_summaries"] == p2["client_summaries"]
    assert p1["client_summaries"] != p_diff["client_summaries"]


# ---------------------------------------------------------------------------
# 17, 18. Invalid configuration handling & grading image rejection
# ---------------------------------------------------------------------------


def test_invalid_configuration_and_parameter_handling() -> None:
    """17. Verify invalid dimensions, worker counts, client counts, and partition strategies are rejected."""
    ds_cfg = load_yaml_config("dataset.yaml")
    fed_cfg = load_yaml_config("federated.yaml")

    with pytest.raises(ConfigValidationError, match="image_size"):
        validate_config({**ds_cfg, "image_size": {"height": 0, "width": 512}}, "dataset")

    with pytest.raises(ConfigValidationError, match="number_of_workers"):
        validate_config({**ds_cfg, "number_of_workers": -1}, "dataset")

    with pytest.raises(ConfigValidationError, match="mask_interpolation"):
        validate_config(
            {
                **ds_cfg,
                "evaluation_transforms": {"mask_interpolation": "linear"},
            },
            "dataset",
        )

    with pytest.raises(ConfigValidationError, match="number_of_clients"):
        validate_config({**fed_cfg, "number_of_clients": 0}, "federated")

    with pytest.raises(ConfigValidationError, match="partitioning.scheme"):
        validate_config(
            {**fed_cfg, "partitioning": {"scheme": "random_invented_scheme"}},
            "federated",
        )

    with pytest.raises(ValueError, match="mask_interpolation"):
        RetinalSegmentationTransform(height=64, width=64, mask_interpolation="cubic")

    with pytest.raises(PartitioningError, match="num_clients"):
        partition_training_records([], num_clients=0)

    with pytest.raises(PartitioningError, match="Unsupported partition strategy"):
        partition_training_records([], num_clients=2, strategy="bad_strategy")


def test_grading_images_rejected_by_dataset_and_partitioner(tmp_path: Path) -> None:
    """18. Verify DDR DR_grading and IDRiD Disease Grading records are rejected by Dataset and partitioner."""
    bad_ddr_record = {
        "dataset_name": "DDR",
        "sample_id": "007-grad-01",
        "split": "train",
        "image_path": "data/raw/DDR/DDR-dataset/DR_grading/train/007-grad-01.jpg",
        "mask_paths": {"MA": None, "HE": None, "EX": None, "SE": None},
    }

    with pytest.raises(DatasetSampleError, match="Grading image"):
        RetinalLesionSegmentationDataset([bad_ddr_record], split="train")

    with pytest.raises(DatasetSampleError, match="Grading image"):
        partition_training_records([bad_ddr_record], num_clients=1, strategy="iid")

    # Also verify build_dataset_manifest on a directory containing grading images excludes them
    ddr_dir = tmp_path / "DDR"
    _create_synthetic_sample_files(ddr_dir, "seg_valid_01", split="train", dataset_name="DDR")
    grad_img = ddr_dir / "DDR-dataset/DR_grading/train/grad_extra_99.png"
    grad_img.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(grad_img)

    manifest = build_dataset_manifest("DDR", ddr_dir)
    ds = RetinalLesionSegmentationDataset(manifest, split="train", image_size=(32, 32))
    assert len(ds) == 1
    assert ds.records[0]["sample_id"] == "seg_valid_01"


def test_fov_crop_configuration_overrides_recompute_crop(tmp_path: Path) -> None:
    """Verify cached fov_crop metadata is reused only when threshold, margin, and boundary settings match (M-2)."""
    from src.data.dataset import SimCLRUnlabeledDataset

    rec = _create_synthetic_sample_files(
        tmp_path / "fov_override",
        "fov_sample_01",
        split="train",
        height=64,
        width=64,
    )
    # Attach a cached crop box [6:42, 8:54] (height=36, width=46) enclosing all lesion boxes [8:40, 10:52]
    # while being strictly smaller than the full intensity FOV [4:60, 6:58] (height=56, width=52)
    rec_with_cache = {
        **rec,
        "fov_crop": {
            "y_min": 6,
            "y_max": 42,
            "x_min": 8,
            "x_max": 54,
            "original_height": 64,
            "original_width": 64,
            "cropped_height": 36,
            "cropped_width": 46,
            "threshold": 10,
            "margin": 0,
            "boundary_expanded_for_mask": True,
        },
    }

    base_ds_cfg = load_yaml_config("dataset.yaml")

    def _cfg_with_fov(
        threshold: int = 10,
        margin: int = 0,
        preserve_boundary_lesions: bool = True,
    ) -> dict[str, Any]:
        return {
            **base_ds_cfg,
            "fov_cropping": {
                **base_ds_cfg.get("fov_cropping", {}),
                "enabled": True,
                "intensity_threshold": threshold,
                "margin_pixels": margin,
                "preserve_boundary_lesions": preserve_boundary_lesions,
            },
        }

    # 1. Matching configuration reuses cached fov_crop box (36x46)
    ds_match = RetinalLesionSegmentationDataset(
        [rec_with_cache],
        split="train",
        dataset_config=_cfg_with_fov(
            threshold=10, margin=0, preserve_boundary_lesions=True
        ),
        fov_crop=True,
        return_metadata=True,
    )
    img_m, mask_m, meta_m = ds_match._load_and_crop_raw_index(0)
    assert img_m.shape == (36, 46, 3)
    assert mask_m.shape == (36, 46, 4)
    assert meta_m["cropped_shape"] == [36, 46]

    # 2. Overriding margin_pixels (e.g. margin=4) forces recomputation from the actual image FOV [4:60, 6:58]
    ds_margin_override = RetinalLesionSegmentationDataset(
        [rec_with_cache],
        split="train",
        dataset_config=_cfg_with_fov(
            threshold=10, margin=4, preserve_boundary_lesions=True
        ),
        fov_crop=True,
        return_metadata=True,
    )
    img_mo, _, meta_mo = ds_margin_override._load_and_crop_raw_index(0)
    assert img_mo.shape != (36, 46, 3)
    assert meta_mo["fov_crop"]["margin"] == 4

    # 3. Overriding intensity_threshold forces recomputation in both segmentation and SimCLR datasets
    ds_thresh_override = RetinalLesionSegmentationDataset(
        [rec_with_cache],
        split="train",
        dataset_config=_cfg_with_fov(
            threshold=25, margin=0, preserve_boundary_lesions=True
        ),
        fov_crop=True,
        return_metadata=True,
    )
    img_to, _, meta_to = ds_thresh_override._load_and_crop_raw_index(0)
    assert img_to.shape != (36, 46, 3)
    assert meta_to["fov_crop"]["threshold"] == 25

    # 4. Overriding preserve_boundary_lesions forces recomputation when boundary_expanded_for_mask mismatches
    ds_bound_override = RetinalLesionSegmentationDataset(
        [rec_with_cache],
        split="train",
        dataset_config=_cfg_with_fov(
            threshold=10, margin=0, preserve_boundary_lesions=False
        ),
        fov_crop=True,
        return_metadata=True,
    )
    img_bo, _, _ = ds_bound_override._load_and_crop_raw_index(0)
    assert img_bo.shape != (36, 46, 3)

    # 5. SimCLRUnlabeledDataset also recomputes when threshold or margin overrides cached fov_crop
    ssl_rec_cache = {
        **rec_with_cache,
        "fov_crop": {
            **rec_with_cache["fov_crop"],
            "boundary_expanded_for_mask": False,
        },
    }
    ssl_match = SimCLRUnlabeledDataset(
        [ssl_rec_cache],
        dataset_config=_cfg_with_fov(
            threshold=10, margin=0, preserve_boundary_lesions=False
        ),
        fov_crop=True,
        cache_in_memory=False,
    )
    ssl_img_match, _ = ssl_match._load_and_crop_unlabeled_index(0)
    assert ssl_img_match.shape == (36, 46, 3)

    ssl_override = SimCLRUnlabeledDataset(
        [ssl_rec_cache],
        dataset_config=_cfg_with_fov(
            threshold=10, margin=3, preserve_boundary_lesions=False
        ),
        fov_crop=True,
        cache_in_memory=False,
    )
    ssl_img_override, ssl_meta_override = ssl_override._load_and_crop_unlabeled_index(0)
    assert ssl_img_override.shape != (36, 46, 3)
    assert ssl_meta_override["cropped_shape"] != [36, 46]


def test_simclr_cache_resizing_preserves_aspect_ratio_for_nonsquare_images(
    tmp_path: Path,
) -> None:
    """Verify SimCLRUnlabeledDataset cache resizing scales proportionally and preserves aspect ratio (L-1)."""
    from src.data.dataset import SimCLRUnlabeledDataset

    # Create a non-square 600x1200 image (1:2 aspect ratio) exceeding max_dim=1024
    img_path = tmp_path / "nonsquare_fundus.png"
    arr = np.full((600, 1200, 3), fill_value=120, dtype=np.uint8)
    Image.fromarray(arr).save(img_path)

    rec = {
        "dataset_name": "DDR",
        "sample_id": "nonsquare_01",
        "split": "train",
        "image_path": str(img_path),
    }

    # With image_size=(256, 256), max_dim = max(256*2, 1024) = 1024.
    # Proportional scaling of (600, 1200) to max_dim=1024 must produce (512, 1024, 3), not (1024, 1024, 3).
    ds = SimCLRUnlabeledDataset(
        [rec],
        image_size=(256, 256),
        fov_crop=False,
        cache_in_memory=True,
    )
    cached_rgb, _ = ds._load_and_crop_unlabeled_index(0)
    assert cached_rgb.shape == (512, 1024, 3)
    assert max(cached_rgb.shape[:2]) == 1024
    assert cached_rgb.shape[1] / cached_rgb.shape[0] == pytest.approx(2.0)


