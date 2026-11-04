"""Unit tests for canonical 4-channel mask construction, retinal FOV cropping, and dataset manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from scripts.build_manifests import format_manifest_summary, main as build_manifests_cli_main
from src.data.fov import (
    crop_image_and_mask_to_fov,
    detect_retinal_fov,
)
from src.data.manifests import (
    ManifestValidationError,
    build_configured_manifests,
    build_dataset_manifest,
    load_manifest_json,
    save_manifest_json,
)
from src.data.masks import (
    CANONICAL_LESION_CHANNELS,
    MaskConstructionError,
    build_canonical_lesion_mask,
    convert_raw_mask_to_binary,
    load_sample_image_and_mask,
    load_single_lesion_mask,
)


def _save_synthetic_fundus_image(
    path: Path,
    height: int = 40,
    width: int = 60,
    fov_box: tuple[int, int, int, int] | None = (6, 34, 10, 50),
    fg_value: int = 160,
    bg_value: int = 0,
) -> np.ndarray:
    """Create and save a synthetic RGB fundus image with an optional bright FOV region."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((height, width, 3), bg_value, dtype=np.uint8)
    if fov_box is not None:
        y0, y1, x0, x1 = fov_box
        arr[y0:y1, x0:x1, 0] = fg_value
        arr[y0:y1, x0:x1, 1] = max(0, fg_value - 40)
    # Save as lossless PNG or high-quality TIFF/JPEG depending on suffix
    if path.suffix.lower() in (".jpg", ".jpeg"):
        Image.fromarray(arr, mode="RGB").save(path, quality=100, subsampling=0)
    else:
        Image.fromarray(arr, mode="RGB").save(path)
    return arr


def _save_grayscale_mask(
    path: Path,
    height: int = 40,
    width: int = 60,
    box: tuple[int, int, int, int] | None = (10, 18, 15, 25),
    fg_value: int = 255,
) -> np.ndarray:
    """Create and save a single-channel grayscale lesion mask."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((height, width), dtype=np.uint8)
    if box is not None and fg_value > 0:
        y0, y1, x0, x1 = box
        arr[y0:y1, x0:x1] = fg_value
    Image.fromarray(arr, mode="L").save(path)
    return arr


def _save_rgba_mask(
    path: Path,
    height: int = 40,
    width: int = 60,
    box: tuple[int, int, int, int] | None = (12, 22, 18, 30),
    r_value: int = 255,
    alpha_value: int = 255,
) -> np.ndarray:
    """Create and save a 4-channel RGBA lesion mask matching IDRiD_81_EX.tif layout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    arr[:, :, 3] = alpha_value
    if box is not None and r_value > 0:
        y0, y1, x0, x1 = box
        arr[y0:y1, x0:x1, 0] = r_value
    Image.fromarray(arr, mode="RGBA").save(path)
    return arr


def _snapshot_directory_hashes(root: Path) -> dict[str, tuple[int, str]]:
    """Compute file size and SHA-256 digest for every file under ``root``."""
    snapshot: dict[str, tuple[int, str]] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            data = p.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            snapshot[rel] = (len(data), digest)
    return snapshot


# ---------------------------------------------------------------------------
# 1, 2, 3. DDR train, valid ("segmentation label"), and test mask loading
# ---------------------------------------------------------------------------


def test_ddr_train_mask_loading(tmp_path: Path) -> None:
    """1. Verify DDR training sample image and 4-class label masks load in [MA, HE, EX, SE] order."""
    ddr_train = tmp_path / "DDR-dataset/lesion_segmentation/train"
    img_p = ddr_train / "image/007-0001-100.jpg"
    _save_synthetic_fundus_image(img_p, 32, 48)

    ma_p = ddr_train / "label/MA/007-0001-100.tif"
    he_p = ddr_train / "label/HE/007-0001-100.tif"
    ex_p = ddr_train / "label/EX/007-0001-100.tif"
    se_p = ddr_train / "label/SE/007-0001-100.tif"

    _save_grayscale_mask(ma_p, 32, 48, box=(4, 8, 4, 8), fg_value=255)
    _save_grayscale_mask(he_p, 32, 48, box=(10, 16, 10, 16), fg_value=255)
    _save_grayscale_mask(ex_p, 32, 48, box=None, fg_value=0)  # Explicit all-zero DDR mask
    _save_grayscale_mask(se_p, 32, 48, box=(20, 24, 20, 24), fg_value=255)

    img, res = load_sample_image_and_mask(
        img_p, {"MA": ma_p, "HE": he_p, "EX": ex_p, "SE": se_p}
    )

    assert img.shape == (32, 48, 3)
    assert res.mask.shape == (32, 48, 4)
    assert res.channel_order == ("MA", "HE", "EX", "SE")
    assert res.missing_classes == ()
    assert res.empty_mask_classes == ("EX",)
    assert res.positive_classes == ("MA", "HE", "SE")
    assert res.foreground_pixel_counts == {"MA": 16, "HE": 36, "EX": 0, "SE": 16}


def test_ddr_validation_mask_loading_from_segmentation_label(tmp_path: Path) -> None:
    """2. Verify DDR validation masks under 'valid/segmentation label/' are discovered and loaded."""
    ddr_root = tmp_path / "DDR"
    val_dir = ddr_root / "DDR-dataset/lesion_segmentation/valid"
    _save_synthetic_fundus_image(val_dir / "image/007-val-01.jpg", 36, 52)

    for idx, cls_name in enumerate(CANONICAL_LESION_CHANNELS):
        _save_grayscale_mask(
            val_dir / f"segmentation label/{cls_name}/007-val-01.tif",
            36,
            52,
            box=(2 + idx * 4, 5 + idx * 4, 4, 10),
            fg_value=255,
        )

    manifest = build_dataset_manifest("DDR", ddr_root)
    assert manifest["summary"]["total_samples"] == 1
    assert manifest["summary"]["split_counts"] == {"valid": 1}
    rec = manifest["records"][0]
    assert rec["split"] == "valid"
    assert rec["sample_id"] == "007-val-01"
    assert rec["missing_lesion_masks"] == []
    assert all("segmentation label" in rec["mask_paths"][c] for c in CANONICAL_LESION_CHANNELS)


def test_ddr_test_mask_loading(tmp_path: Path) -> None:
    """3. Verify DDR test masks under 'test/label/' load and preserve split='test'."""
    ddr_root = tmp_path / "DDR"
    test_dir = ddr_root / "DDR-dataset/lesion_segmentation/test"
    _save_synthetic_fundus_image(test_dir / "image/007-test-01.jpg", 32, 48)

    for cls_name in CANONICAL_LESION_CHANNELS:
        _save_grayscale_mask(
            test_dir / f"label/{cls_name}/007-test-01.tif",
            32,
            48,
            box=(8, 14, 10, 20) if cls_name == "MA" else None,
            fg_value=255 if cls_name == "MA" else 0,
        )

    manifest = build_dataset_manifest("DDR", ddr_root)
    assert manifest["summary"]["split_counts"] == {"test": 1}
    rec = manifest["records"][0]
    assert rec["split"] == "test"
    assert rec["positive_lesion_masks"] == ["MA"]
    assert rec["empty_lesion_masks"] == ["HE", "EX", "SE"]
    assert rec["missing_lesion_masks"] == []


# ---------------------------------------------------------------------------
# 4, 5, 6, 7. IDRiD mask loading, missing HE/SE masks, RGBA EX conversion, OD exclusion
# ---------------------------------------------------------------------------


def test_idrid_ma_he_ex_se_mask_loading_and_od_exclusion(tmp_path: Path) -> None:
    """4. Verify IDRiD MA, HE, EX, SE masks load in canonical order while OD is excluded."""
    idrid_root = tmp_path / "IDRiD"
    img_dir = idrid_root / "A. Segmentation/1. Original Images/a. Training Set"
    gt_dir = idrid_root / "A. Segmentation/2. All Segmentation Groundtruths/a. Training Set"

    _save_synthetic_fundus_image(img_dir / "IDRiD_01.jpg", 40, 60)
    _save_grayscale_mask(gt_dir / "1. Microaneurysms/IDRiD_01_MA.tif", 40, 60, (6, 10, 12, 16), 1)
    _save_grayscale_mask(gt_dir / "2. Haemorrhages/IDRiD_01_HE.tif", 40, 60, (12, 18, 14, 20), 1)
    _save_grayscale_mask(gt_dir / "3. Hard Exudates/IDRiD_01_EX.tif", 40, 60, (20, 26, 22, 30), 1)
    _save_grayscale_mask(gt_dir / "4. Soft Exudates/IDRiD_01_SE.tif", 40, 60, (26, 30, 30, 36), 1)
    # Optic Disc mask present on disk, must be excluded from 4-channel lesion target
    _save_grayscale_mask(gt_dir / "5. Optic Disc/IDRiD_01_OD.tif", 40, 60, (10, 30, 10, 30), 1)

    manifest = build_dataset_manifest("IDRiD", idrid_root)
    assert manifest["summary"]["total_samples"] == 1
    assert manifest["summary"]["excluded_od_mask_count"] == 1
    rec = manifest["records"][0]
    assert rec["channel_order"] == ["MA", "HE", "EX", "SE"]
    assert set(rec["mask_paths"].keys()) == {"MA", "HE", "EX", "SE"}
    assert "OD" not in rec["mask_paths"]
    assert rec["canonical_mask_dimensions"] == [40, 60, 4]
    assert rec["lesion_pixel_counts"] == {"MA": 16, "HE": 36, "EX": 48, "SE": 24}


def test_missing_idrid_he_mask_synthesizes_zero_channel(tmp_path: Path) -> None:
    """5. Verify omitted IDRiD HE mask produces an all-zero channel 1 and explicit missing metadata."""
    img_p = tmp_path / "IDRiD_28.jpg"
    _save_synthetic_fundus_image(img_p, 32, 48)
    ma_p = _save_grayscale_mask(tmp_path / "IDRiD_28_MA.tif", 32, 48, (4, 8, 4, 8), 1)
    ex_p = _save_grayscale_mask(tmp_path / "IDRiD_28_EX.tif", 32, 48, (12, 16, 12, 16), 1)
    se_p = _save_grayscale_mask(tmp_path / "IDRiD_28_SE.tif", 32, 48, (20, 24, 20, 24), 1)

    res = build_canonical_lesion_mask(
        (32, 48),
        {
            "MA": tmp_path / "IDRiD_28_MA.tif",
            "HE": None,
            "EX": tmp_path / "IDRiD_28_EX.tif",
            "SE": tmp_path / "IDRiD_28_SE.tif",
        },
    )
    assert res.mask.shape == (32, 48, 4)
    assert int(res.mask[:, :, 1].sum()) == 0
    assert res.missing_classes == ("HE",)
    assert res.synthesized_zero_classes == ("HE",)
    assert res.present_classes == ("MA", "EX", "SE")
    assert np.array_equal(res.mask[:, :, 0], ma_p)
    assert np.array_equal(res.mask[:, :, 2], ex_p)
    assert np.array_equal(res.mask[:, :, 3], se_p)


def test_missing_idrid_se_masks_in_manifest(tmp_path: Path) -> None:
    """6. Verify omitted IDRiD SE masks across samples are handled as valid synthesized zero channels."""
    idrid_root = tmp_path / "IDRiD"
    img_dir = idrid_root / "A. Segmentation/1. Original Images/b. Testing Set"
    gt_dir = idrid_root / "A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set"

    for idx in (1, 2):
        sid = f"IDRiD_0{idx}"
        _save_synthetic_fundus_image(img_dir / f"{sid}.jpg", 32, 48)
        _save_grayscale_mask(gt_dir / f"1. Microaneurysms/{sid}_MA.tif", 32, 48, (6, 10, 10, 14), 1)
        _save_grayscale_mask(gt_dir / f"2. Haemorrhages/{sid}_HE.tif", 32, 48, (12, 16, 14, 18), 1)
        _save_grayscale_mask(gt_dir / f"3. Hard Exudates/{sid}_EX.tif", 32, 48, (18, 22, 20, 24), 1)
        # Omit SE mask for both samples

    manifest = build_dataset_manifest("IDRiD", idrid_root)
    assert manifest["summary"]["total_samples"] == 2
    assert manifest["summary"]["missing_mask_counts_by_class"] == {
        "MA": 0,
        "HE": 0,
        "EX": 0,
        "SE": 2,
    }
    assert manifest["summary"]["synthesized_zero_channel_counts_by_class"] == {
        "MA": 0,
        "HE": 0,
        "EX": 0,
        "SE": 2,
    }
    for rec in manifest["records"]:
        assert rec["missing_lesion_masks"] == ["SE"]
        assert rec["mask_paths"]["SE"] is None
        assert rec["lesion_pixel_counts"]["SE"] == 0


def test_idrid_rgba_ex_mask_conversion_and_ambiguous_multichannel_rejection(
    tmp_path: Path,
) -> None:
    """7. Verify 4-channel RGBA EX mask (R in {0, 255}, G=0, B=0, A=255) converts cleanly to {0, 1}."""
    rgba_path = tmp_path / "IDRiD_81_EX.tif"
    _save_rgba_mask(rgba_path, height=32, width=48, box=(8, 16, 10, 22), r_value=255, alpha_value=255)

    binary = load_single_lesion_mask(rgba_path, expected_shape=(32, 48))
    assert binary.shape == (32, 48)
    assert binary.dtype == np.uint8
    assert set(np.unique(binary).tolist()) == {0, 1}
    # Expected positive count: (16 - 8) * (22 - 10) = 8 * 12 = 96 (NOT 32*48=1536 from alpha=255!)
    assert int(binary.sum()) == 96

    # Also verify an ambiguous multichannel mask where R and G have conflicting annotations is rejected
    ambiguous_rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    ambiguous_rgb[2:6, 2:6, 0] = 255
    ambiguous_rgb[8:12, 8:12, 1] = 255
    with pytest.raises(MaskConstructionError, match="Ambiguous multichannel mask"):
        convert_raw_mask_to_binary(ambiguous_rgb)


# ---------------------------------------------------------------------------
# 8, 9, 10. {0, 1} and {0, 255} normalization, shape/order, and independent overlaps
# ---------------------------------------------------------------------------


def test_zero_one_and_zero_255_normalization_identical(tmp_path: Path) -> None:
    """8. Verify {0, 1} and {0, 255} source masks normalize to identical {0, 1} uint8 arrays."""
    m01_path = tmp_path / "mask_01.tif"
    m255_path = tmp_path / "mask_255.tif"
    _save_grayscale_mask(m01_path, 24, 32, box=(4, 14, 6, 18), fg_value=1)
    _save_grayscale_mask(m255_path, 24, 32, box=(4, 14, 6, 18), fg_value=255)

    b01 = load_single_lesion_mask(m01_path, expected_shape=(24, 32))
    b255 = load_single_lesion_mask(m255_path, expected_shape=(24, 32))

    assert b01.dtype == np.uint8
    assert b255.dtype == np.uint8
    assert set(np.unique(b01).tolist()) == {0, 1}
    assert set(np.unique(b255).tolist()) == {0, 1}
    assert np.array_equal(b01, b255)


def test_output_shape_and_channel_order_enforced(tmp_path: Path) -> None:
    """9. Verify canonical mask always has shape (H, W, 4) and strictly enforces [MA, HE, EX, SE]."""
    ma_p = tmp_path / "ma.tif"
    he_p = tmp_path / "he.tif"
    ex_p = tmp_path / "ex.tif"
    se_p = tmp_path / "se.tif"

    _save_grayscale_mask(ma_p, 20, 30, box=(0, 2, 0, 2), fg_value=1)
    _save_grayscale_mask(he_p, 20, 30, box=(2, 4, 2, 4), fg_value=255)
    _save_grayscale_mask(ex_p, 20, 30, box=(4, 6, 4, 6), fg_value=1)
    _save_grayscale_mask(se_p, 20, 30, box=(6, 8, 6, 8), fg_value=255)

    # Pass dict in shuffled key order; output must still be [MA, HE, EX, SE]
    res = build_canonical_lesion_mask(
        (20, 30),
        {"SE": se_p, "EX": ex_p, "MA": ma_p, "HE": he_p},
    )
    assert res.mask.shape == (20, 30, 4)
    assert res.channel_order == ("MA", "HE", "EX", "SE")
    assert int(res.mask[0, 0, 0]) == 1  # MA at ch 0
    assert int(res.mask[2, 2, 1]) == 1  # HE at ch 1
    assert int(res.mask[4, 4, 2]) == 1  # EX at ch 2
    assert int(res.mask[6, 6, 3]) == 1  # SE at ch 3

    # Verify wrong channel_order is rejected
    with pytest.raises(MaskConstructionError, match="Canonical channel_order"):
        build_canonical_lesion_mask(
            (20, 30),
            {"MA": ma_p},
            channel_order=("EX", "HE", "MA", "SE"),
        )


def test_independent_channels_preserve_spatial_overlaps(tmp_path: Path) -> None:
    """10. Verify overlapping lesions across classes remain independent in their respective channels."""
    ma_p = tmp_path / "overlap_ma.tif"
    he_p = tmp_path / "overlap_he.tif"
    # Overlapping region [6:12, 6:12] is active in both MA and HE
    _save_grayscale_mask(ma_p, 24, 24, box=(4, 12, 4, 12), fg_value=255)
    _save_grayscale_mask(he_p, 24, 24, box=(6, 14, 6, 14), fg_value=255)

    res = build_canonical_lesion_mask(
        (24, 24),
        {"MA": ma_p, "HE": he_p, "EX": None, "SE": None},
    )
    assert res.mask.shape == (24, 24, 4)
    assert int(res.mask[:, :, 0].sum()) == 64
    assert int(res.mask[:, :, 1].sum()) == 64
    # Overlap region is 6x6 = 36 pixels where both channel 0 and channel 1 equal 1
    assert res.overlapping_pixel_count == 36
    assert bool(np.all(res.mask[6:12, 6:12, 0] == 1))
    assert bool(np.all(res.mask[6:12, 6:12, 1] == 1))


# ---------------------------------------------------------------------------
# 11, 12, 13. Retinal FOV detection, synchronized image/mask cropping, and fallback
# ---------------------------------------------------------------------------


def test_fov_detection_on_synthetic_retinal_image() -> None:
    """11. Verify FOV detector locates the bright retinal region while excluding black margins and JPEG noise."""
    img = np.full((64, 80, 3), 4, dtype=np.uint8)  # Background noise = 4 (<= threshold 10)
    img[10:54, 16:68, 0] = 180  # Retinal disk in rows [10..53], cols [16..67]

    fov = detect_retinal_fov(img, threshold=10, margin=0)
    assert fov.fallback_used is False
    assert (fov.y_min, fov.y_max, fov.x_min, fov.x_max) == (10, 54, 16, 68)
    assert (fov.cropped_height, fov.cropped_width) == (44, 52)


def test_image_and_mask_receive_identical_crop_coordinates_and_preserve_boundary_lesions() -> None:
    """12. Verify image and 4-channel mask receive identical crop bounds and boundary lesions are preserved."""
    img = np.zeros((50, 70, 3), dtype=np.uint8)
    img[8:42, 12:58, :] = 150

    mask = np.zeros((50, 70, 4), dtype=np.uint8)
    # Place MA lesion inside FOV and EX lesion right on a dim boundary at row 6..8
    mask[15:20, 20:25, 0] = 1
    mask[6:10, 12:16, 2] = 1

    cropped_img, cropped_mask, fov_meta = crop_image_and_mask_to_fov(
        img, mask, threshold=10, margin=0, preserve_boundary_lesions=True
    )

    assert cropped_mask is not None
    assert cropped_img.shape[:2] == cropped_mask.shape[:2]
    assert (fov_meta.y_min, fov_meta.y_max, fov_meta.x_min, fov_meta.x_max) == (6, 42, 12, 58)
    assert fov_meta.boundary_expanded_for_mask is True
    assert fov_meta.lesion_pixels_outside_fov == 0
    # Total lesion pixels before and after crop are identical
    assert int(cropped_mask.sum()) == int(mask.sum())
    assert set(np.unique(cropped_mask).tolist()) == {0, 1}


def test_fov_fallback_behavior_on_blank_or_degenerate_image() -> None:
    """13. Verify FOV detector falls back gracefully to full image when no valid FOV is detected."""
    blank_img = np.zeros((40, 60, 3), dtype=np.uint8)
    mask = np.zeros((40, 60, 4), dtype=np.uint8)
    mask[5:10, 5:10, 0] = 1

    c_img, c_mask, meta = crop_image_and_mask_to_fov(blank_img, mask, threshold=10)
    assert meta.fallback_used is True
    assert meta.fallback_reason == "no_foreground_detected"
    assert (meta.y_min, meta.y_max, meta.x_min, meta.x_max) == (0, 40, 0, 60)
    assert c_img.shape == (40, 60, 3)
    assert c_mask is not None and c_mask.shape == (40, 60, 4)

    # Also test tiny noise speck (< 1% of image area) fallback
    speck_img = np.zeros((100, 100, 3), dtype=np.uint8)
    speck_img[50, 50, 0] = 200  # 1 pixel out of 10,000 (0.01% < 1%)
    speck_meta = detect_retinal_fov(speck_img, threshold=10, min_fov_fraction=0.01)
    assert speck_meta.fallback_used is True
    assert speck_meta.fallback_reason == "fov_area_below_min_fraction"
    assert (speck_meta.y_min, speck_meta.y_max, speck_meta.x_min, speck_meta.x_max) == (
        0,
        100,
        0,
        100,
    )


# ---------------------------------------------------------------------------
# 14, 15, 16, 17, 18. Manifest determinism, splits, duplicates, grading exclusion, raw preservation
# ---------------------------------------------------------------------------


def test_manifest_determinism_and_split_preservation(tmp_path: Path) -> None:
    """14 & 15. Verify manifest generation is deterministic and preserves train/valid/test splits."""
    ddr_root = tmp_path / "DDR"
    for split, mask_folder, sids in (
        ("test", "label", ["007-test-b", "007-test-a"]),
        ("train", "label", ["007-train-b", "007-train-a"]),
        ("valid", "segmentation label", ["007-val-b", "007-val-a"]),
    ):
        for sid in sids:
            _save_synthetic_fundus_image(
                ddr_root / f"DDR-dataset/lesion_segmentation/{split}/image/{sid}.jpg",
                32,
                48,
            )
            for cls_name in CANONICAL_LESION_CHANNELS:
                _save_grayscale_mask(
                    ddr_root
                    / f"DDR-dataset/lesion_segmentation/{split}/{mask_folder}/{cls_name}/{sid}.tif",
                    32,
                    48,
                    box=(6, 12, 10, 18),
                    fg_value=255,
                )

    m1 = build_dataset_manifest("DDR", ddr_root)
    m2 = build_dataset_manifest("DDR", ddr_root)

    out1 = save_manifest_json(m1, tmp_path / "m1.json")
    out2 = save_manifest_json(m2, tmp_path / "m2.json")
    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")

    loaded = load_manifest_json(out1)
    assert loaded["summary"]["split_counts"] == {"train": 2, "valid": 2, "test": 2}
    observed_order = [(r["split"], r["sample_id"]) for r in loaded["records"]]
    assert observed_order == [
        ("train", "007-train-a"),
        ("train", "007-train-b"),
        ("valid", "007-val-a"),
        ("valid", "007-val-b"),
        ("test", "007-test-a"),
        ("test", "007-test-b"),
    ]


def test_duplicate_sample_id_and_duplicate_mask_detection(tmp_path: Path) -> None:
    """16. Verify duplicate image sample IDs or duplicate lesion masks raise ManifestValidationError."""
    dup_img_root = tmp_path / "dup_img_ds"
    _save_synthetic_fundus_image(dup_img_root / "train/image/case_01.jpg", 32, 32)
    _save_synthetic_fundus_image(dup_img_root / "test/image/case_01.png", 32, 32)
    _save_grayscale_mask(dup_img_root / "train/label/MA/case_01.tif", 32, 32)

    with pytest.raises(ManifestValidationError, match="Duplicate segmentation image sample_id"):
        build_dataset_manifest("DDR", dup_img_root)

    dup_mask_root = tmp_path / "dup_mask_ds"
    _save_synthetic_fundus_image(dup_mask_root / "train/image/case_02.jpg", 32, 32)
    _save_grayscale_mask(dup_mask_root / "train/label/MA/case_02.tif", 32, 32)
    _save_grayscale_mask(dup_mask_root / "train/label/MA/case_02.png", 32, 32)

    with pytest.raises(ManifestValidationError, match="Duplicate 'MA' mask"):
        build_dataset_manifest("DDR", dup_mask_root)


def test_no_unintended_inclusion_of_grading_or_detection_files(tmp_path: Path) -> None:
    """17. Verify DDR DR_grading images, lesion_detection XMLs, and IDRiD grading images are excluded."""
    ddr_root = tmp_path / "DDR"
    _save_synthetic_fundus_image(
        ddr_root / "DDR-dataset/lesion_segmentation/train/image/seg_01.jpg", 32, 48
    )
    for cls_name in CANONICAL_LESION_CHANNELS:
        _save_grayscale_mask(
            ddr_root / f"DDR-dataset/lesion_segmentation/train/label/{cls_name}/seg_01.tif",
            32,
            48,
        )

    # Add grading images (one sharing the segmentation stem, two grading-only) and detection XML
    _save_synthetic_fundus_image(ddr_root / "DDR-dataset/DR_grading/train/seg_01.jpg", 32, 48)
    _save_synthetic_fundus_image(ddr_root / "DDR-dataset/DR_grading/train/grad_only_01.jpg", 32, 48)
    _save_synthetic_fundus_image(ddr_root / "DDR-dataset/DR_grading/test/grad_only_02.jpg", 32, 48)
    xml_path = ddr_root / "DDR-dataset/lesion_detection/train/seg_01.xml"
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text("<annotation></annotation>", encoding="utf-8")

    manifest = build_dataset_manifest("DDR", ddr_root)
    assert manifest["summary"]["total_samples"] == 1
    assert manifest["summary"]["excluded_grading_image_count"] == 3
    assert [r["sample_id"] for r in manifest["records"]] == ["seg_01"]
    assert "lesion_segmentation" in manifest["records"][0]["image_path"]


def test_raw_data_preserved_and_cli_builds_configured_manifests(tmp_path: Path) -> None:
    """18. Verify raw dataset directories remain 100% unmodified after manifest generation."""
    raw_ddr = tmp_path / "raw/DDR"
    raw_idrid = tmp_path / "raw/IDRiD"

    _save_synthetic_fundus_image(
        raw_ddr / "DDR-dataset/lesion_segmentation/train/image/ddr_01.jpg", 32, 48
    )
    for cls_name in CANONICAL_LESION_CHANNELS:
        _save_grayscale_mask(
            raw_ddr / f"DDR-dataset/lesion_segmentation/train/label/{cls_name}/ddr_01.tif",
            32,
            48,
        )

    _save_synthetic_fundus_image(
        raw_idrid / "A. Segmentation/1. Original Images/a. Training Set/IDRiD_01.jpg", 32, 48
    )
    _save_grayscale_mask(
        raw_idrid
        / "A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/1. Microaneurysms/IDRiD_01_MA.tif",
        32,
        48,
        fg_value=1,
    )
    _save_rgba_mask(
        raw_idrid
        / "A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/3. Hard Exudates/IDRiD_01_EX.tif",
        32,
        48,
    )

    before_ddr = _snapshot_directory_hashes(raw_ddr)
    before_idrid = _snapshot_directory_hashes(raw_idrid)

    cfg_path = tmp_path / "dataset_test.yaml"
    manifest_out_dir = tmp_path / "processed/manifests"
    cfg_data = {
        "dataset_names": ["DDR", "IDRiD"],
        "paths": {
            "raw_ddr": str(raw_ddr),
            "raw_idrid": str(raw_idrid),
            "processed": str(tmp_path / "processed"),
            "manifests": str(manifest_out_dir),
            "partitions": str(tmp_path / "partitions"),
        },
        "image_size": {"height": 512, "width": 512},
        "batch_size": 4,
        "number_of_workers": 0,
        "normalization": {"enabled": True},
        "augmentation": {"enabled": True},
        "train_validation_test_split": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15,
        },
    }
    cfg_path.write_text(yaml.safe_dump(cfg_data), encoding="utf-8")

    exit_code = build_manifests_cli_main(
        ["--config", str(cfg_path), "--output-dir", str(manifest_out_dir)]
    )
    assert exit_code == 0

    after_ddr = _snapshot_directory_hashes(raw_ddr)
    after_idrid = _snapshot_directory_hashes(raw_idrid)
    assert before_ddr == after_ddr
    assert before_idrid == after_idrid

    assert (manifest_out_dir / "ddr_manifest.json").exists()
    assert (manifest_out_dir / "idrid_manifest.json").exists()
    assert (manifest_out_dir / "combined_manifest.json").exists()

    manifests = build_configured_manifests(cfg_path, output_dir=manifest_out_dir)
    report_str = format_manifest_summary(manifests)
    assert "Dataset: DDR" in report_str
    assert "Dataset: IDRiD" in report_str


def test_dimension_mismatch_and_unreadable_mask_rejected(tmp_path: Path) -> None:
    """Verify image-mask dimension mismatches and corrupted masks raise clear errors."""
    img_p = tmp_path / "img.jpg"
    _save_synthetic_fundus_image(img_p, 32, 48)
    bad_dim_mask = tmp_path / "wrong_dim.tif"
    _save_grayscale_mask(bad_dim_mask, 16, 16)

    with pytest.raises(MaskConstructionError, match="Mask dimension mismatch"):
        load_single_lesion_mask(bad_dim_mask, expected_shape=(32, 48))

    corrupt_mask = tmp_path / "corrupt.tif"
    corrupt_mask.write_bytes(b"corrupted_tiff_bytes")
    with pytest.raises(MaskConstructionError, match="Unreadable or corrupted"):
        load_single_lesion_mask(corrupt_mask, expected_shape=(32, 48))

