"""Unit tests for dataset inventory, archive detection, and candidate locator utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from scripts.inspect_datasets import (
    SUPPORTED_ARCHIVE_EXTENSIONS,
    SUPPORTED_IMAGE_EXTENSIONS,
    format_inventory_summary,
    get_archive_extraction_hint,
    inspect_dataset_directory,
    is_archive_file,
    is_supported_image_file,
    locate_dataset_candidates,
    main as inspect_cli_main,
)

EXPECTED_INVENTORY_KEYS: set[str] = {
    "dataset_name",
    "root_path",
    "root_exists",
    "status",
    "recognized_layout",
    "archive_files",
    "image_directories",
    "grading_image_directories",
    "mask_directories",
    "other_annotation_files",
    "image_count",
    "grading_image_count",
    "mask_count",
    "empty_mask_count",
    "non_empty_mask_count",
    "readable_image_count",
    "readable_mask_count",
    "unreadable_images",
    "unreadable_masks",
    "image_extensions",
    "mask_extensions",
    "image_extension_counts",
    "mask_extension_counts",
    "image_dimensions",
    "mask_dimensions",
    "image_channel_counts",
    "mask_channel_counts",
    "mask_unique_values",
    "lesion_class_counts",
    "lesion_class_unique_values",
    "paired_sample_count",
    "images_missing_masks",
    "unmatched_masks",
    "duplicate_image_ids",
    "duplicate_mask_entries",
    "dimension_mismatches",
}


def _save_rgb_image(path: Path, height: int = 32, width: int = 48) -> None:
    """Create and save a synthetic RGB image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :, 1] = 128
    Image.fromarray(arr, mode="RGB").save(path)


def _save_mask_image(
    path: Path,
    height: int = 32,
    width: int = 48,
    foreground_value: int = 255,
) -> None:
    """Create and save a synthetic single-channel lesion mask."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((height, width), dtype=np.uint8)
    if foreground_value > 0:
        arr[4:12, 4:12] = foreground_value
    Image.fromarray(arr, mode="L").save(path)


def _write_temp_dataset_config(tmp_path: Path, ddr_dir: Path, idrid_dir: Path) -> Path:
    """Write an isolated dataset.yaml pointing to temporary test directories."""
    cfg_path = tmp_path / "test_dataset.yaml"
    cfg_data = {
        "dataset_names": ["DDR", "IDRiD"],
        "paths": {
            "raw_ddr": str(ddr_dir),
            "raw_idrid": str(idrid_dir),
            "processed": str(tmp_path / "processed"),
            "partitions": str(tmp_path / "partitions"),
        },
        "image_size": {"height": 512, "width": 512},
        "batch_size": 8,
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
    return cfg_path


def test_nonexistent_dataset_directory(tmp_path: Path) -> None:
    """Verify nonexistent directory reports root_exists=False and status='missing_directory'."""
    missing_dir = tmp_path / "does_not_exist"
    report = inspect_dataset_directory("DDR", missing_dir)

    assert set(report.keys()) == EXPECTED_INVENTORY_KEYS
    assert report["dataset_name"] == "DDR"
    assert report["root_exists"] is False
    assert report["status"] == "missing_directory"
    assert report["recognized_layout"] == "none"
    assert report["image_count"] == 0
    assert report["mask_count"] == 0


def test_empty_dataset_directory_with_gitkeep(tmp_path: Path) -> None:
    """Verify empty directory (or directory with only .gitkeep) reports status='empty'."""
    empty_dir = tmp_path / "empty_ddr"
    empty_dir.mkdir(parents=True)
    (empty_dir / ".gitkeep").write_text("", encoding="utf-8")

    report = inspect_dataset_directory("DDR", empty_dir)

    assert set(report.keys()) == EXPECTED_INVENTORY_KEYS
    assert report["root_exists"] is True
    assert report["status"] == "empty"
    assert report["image_count"] == 0
    assert report["mask_count"] == 0
    assert report["paired_sample_count"] == 0


def test_archives_only_directory_detected(tmp_path: Path) -> None:
    """Verify directory containing only unextracted archives reports status='archives_only'."""
    arch_dir = tmp_path / "DDR_raw"
    arch_dir.mkdir(parents=True)
    (arch_dir / "DDR_lesion_segmentation.zip").write_bytes(b"PK\x03\x04dummy")

    report = inspect_dataset_directory("DDR", arch_dir)

    assert report["status"] == "archives_only"
    assert report["archive_files"] == ["DDR_lesion_segmentation.zip"]
    assert report["image_count"] == 0
    assert report["mask_count"] == 0


def test_valid_image_and_mask_discovery_and_pairing_ddr_style(tmp_path: Path) -> None:
    """Verify discovery, dimensions, channels, and pairing on a synthetic DDR-like structure."""
    ddr_root = tmp_path / "DDR"
    _save_rgb_image(ddr_root / "lesion_segmentation/train/image/007_001.jpg", 40, 60)
    _save_rgb_image(ddr_root / "lesion_segmentation/train/image/007_002.png", 40, 60)
    # Also add DR_grading images sharing a stem and a grading-only stem
    _save_rgb_image(ddr_root / "DR_grading/train/007_001.jpg", 40, 60)
    _save_rgb_image(ddr_root / "DR_grading/train/007_999.jpg", 40, 60)

    _save_mask_image(
        ddr_root / "lesion_segmentation/train/label/MA/007_001.tif",
        40,
        60,
        foreground_value=255,
    )
    _save_mask_image(
        ddr_root / "lesion_segmentation/train/label/EX/007_001.tif",
        40,
        60,
        foreground_value=0,
    )
    _save_mask_image(
        ddr_root / "lesion_segmentation/train/label/HE/007_002.tif",
        40,
        60,
        foreground_value=1,
    )

    report = inspect_dataset_directory("DDR", ddr_root)

    assert report["status"] == "present"
    assert report["recognized_layout"] == "ddr_official"
    assert report["image_count"] == 2
    assert report["grading_image_count"] == 2
    assert report["mask_count"] == 3
    assert report["empty_mask_count"] == 1
    assert report["non_empty_mask_count"] == 2
    assert report["readable_image_count"] == 2
    assert report["readable_mask_count"] == 3
    assert report["image_extensions"] == [".jpg", ".png"]
    assert report["mask_extensions"] == [".tif"]
    assert report["image_dimensions"] == {"40x60": 2}
    assert report["mask_dimensions"] == {"40x60": 3}
    assert report["image_channel_counts"] == {3: 2}
    assert report["mask_channel_counts"] == {1: 3}
    assert report["mask_unique_values"] == [0, 1, 255]
    assert report["lesion_class_counts"] == {"EX": 1, "HE": 1, "MA": 1}
    assert report["paired_sample_count"] == 2
    assert report["images_missing_masks"] == []
    assert report["unmatched_masks"] == []
    assert report["duplicate_image_ids"] == []
    assert report["dimension_mismatches"] == []


def test_valid_image_and_mask_pairing_idrid_suffix_style(tmp_path: Path) -> None:
    """Verify IDRiD suffix conventions (e.g., IDRiD_01.jpg <-> IDRiD_01_MA.tif) pair cleanly."""
    idrid_root = tmp_path / "IDRiD"
    _save_rgb_image(
        idrid_root / "1. Original Images/a. Training Set/IDRiD_01.jpg", 32, 32
    )
    _save_mask_image(
        idrid_root
        / "2. All Segmentation Groundtruths/a. Training Set/1. Microaneurysms/IDRiD_01_MA.tif",
        32,
        32,
        foreground_value=255,
    )
    _save_mask_image(
        idrid_root
        / "2. All Segmentation Groundtruths/a. Training Set/4. Soft Exudates/IDRiD_01_SE.tif",
        32,
        32,
        foreground_value=255,
    )

    report = inspect_dataset_directory("IDRiD", idrid_root)

    assert report["status"] == "present"
    assert report["recognized_layout"] == "idrid_official"
    assert report["image_count"] == 1
    assert report["mask_count"] == 2
    assert report["paired_sample_count"] == 1
    assert report["lesion_class_counts"] == {"MA": 1, "SE": 1}
    assert report["images_missing_masks"] == []
    assert report["unmatched_masks"] == []


def test_missing_masks_and_unmatched_masks_detected(tmp_path: Path) -> None:
    """Verify images without masks and masks without images are reported accurately."""
    ds_root = tmp_path / "mismatch_ds"
    _save_rgb_image(ds_root / "images/sample_paired.jpg", 32, 32)
    _save_rgb_image(ds_root / "images/sample_without_mask.jpg", 32, 32)

    _save_mask_image(ds_root / "masks/MA/sample_paired_MA.png", 32, 32)
    _save_mask_image(ds_root / "masks/MA/orphan_mask_99_MA.png", 32, 32)

    report = inspect_dataset_directory("SyntheticDR", ds_root)

    assert report["paired_sample_count"] == 1
    assert report["images_missing_masks"] == ["images/sample_without_mask.jpg"]
    assert report["unmatched_masks"] == ["masks/MA/orphan_mask_99_MA.png"]


def test_dimension_mismatch_detected(tmp_path: Path) -> None:
    """Verify unexpected image vs. mask resolution differences are recorded."""
    ds_root = tmp_path / "dim_mismatch_ds"
    _save_rgb_image(ds_root / "images/case_01.png", height=64, width=64)
    _save_mask_image(ds_root / "masks/EX/case_01.png", height=32, width=48)

    report = inspect_dataset_directory("SyntheticDR", ds_root)

    assert len(report["dimension_mismatches"]) == 1
    mismatch = report["dimension_mismatches"][0]
    assert mismatch["sample_id"] == "case_01"
    assert mismatch["image_size"] == "64x64"
    assert mismatch["mask_size"] == "32x48"


def test_unreadable_image_and_mask_files_detected(tmp_path: Path) -> None:
    """Verify corrupted/unreadable image and mask files are caught without crashing."""
    ds_root = tmp_path / "corrupt_ds"
    (ds_root / "images").mkdir(parents=True)
    (ds_root / "masks/MA").mkdir(parents=True)

    bad_img = ds_root / "images/corrupt_sample.jpg"
    bad_img.write_bytes(b"not a valid jpeg image payload")

    bad_mask = ds_root / "masks/MA/corrupt_sample.png"
    bad_mask.write_bytes(b"not a valid png mask payload")

    report = inspect_dataset_directory("CorruptDR", ds_root)

    assert report["image_count"] == 1
    assert report["readable_image_count"] == 0
    assert report["unreadable_images"] == ["images/corrupt_sample.jpg"]

    assert report["mask_count"] == 1
    assert report["readable_mask_count"] == 0
    assert report["unreadable_masks"] == ["masks/MA/corrupt_sample.png"]


def test_duplicate_filenames_detected(tmp_path: Path) -> None:
    """Verify duplicate image sample IDs and duplicate sample+class masks are flagged."""
    ds_root = tmp_path / "dup_ds"
    _save_rgb_image(ds_root / "images/dup_01.jpg", 16, 16)
    _save_rgb_image(ds_root / "images/dup_01.png", 16, 16)

    _save_mask_image(ds_root / "masks/MA/dup_01_MA.png", 16, 16)
    _save_mask_image(ds_root / "masks/MA/dup_01_MA.tif", 16, 16)

    report = inspect_dataset_directory("DupDR", ds_root)

    assert report["duplicate_image_ids"] == ["dup_01"]
    assert report["duplicate_mask_entries"] == ["dup_01:MA"]


@pytest.mark.parametrize("ext", SUPPORTED_IMAGE_EXTENSIONS)
def test_supported_image_extensions_accepted(tmp_path: Path, ext: str) -> None:
    """Verify all supported raster image extensions are recognized."""
    file_path = tmp_path / f"sample{ext}"
    file_path.write_bytes(b"placeholder")
    assert is_supported_image_file(file_path) is True


@pytest.mark.parametrize("arch_ext", SUPPORTED_ARCHIVE_EXTENSIONS)
def test_supported_archive_extensions_accepted(tmp_path: Path, arch_ext: str) -> None:
    """Verify all supported archive extensions are recognized by is_archive_file."""
    arch_path = tmp_path / f"dataset_bundle{arch_ext}"
    arch_path.write_bytes(b"archive_bytes")
    assert is_archive_file(arch_path) is True
    hint = get_archive_extraction_hint(arch_path, tmp_path / "out")
    assert str(arch_path) in hint


@pytest.mark.parametrize("bad_name", [".gitkeep", ".hidden.jpg", "notes.txt", "doc.pdf"])
def test_unsupported_or_hidden_files_rejected(tmp_path: Path, bad_name: str) -> None:
    """Verify hidden dotfiles and non-image extensions are rejected by is_supported_image_file."""
    file_path = tmp_path / bad_name
    file_path.write_bytes(b"placeholder")
    assert is_supported_image_file(file_path) is False


def test_locate_dataset_candidates_in_explicit_search_dir(tmp_path: Path) -> None:
    """Verify locate_dataset_candidates discovers archives and extracted folders in search_dirs."""
    ddr_empty = tmp_path / "raw_ddr"
    idrid_empty = tmp_path / "raw_idrid"
    ddr_empty.mkdir()
    idrid_empty.mkdir()
    tmp_cfg = _write_temp_dataset_config(tmp_path, ddr_empty, idrid_empty)

    downloads_dir = tmp_path / "user_downloads"
    idrid_zip = downloads_dir / "IDRiD_A_Segmentation.zip"
    idrid_zip.parent.mkdir(parents=True, exist_ok=True)
    idrid_zip.write_bytes(b"PK\x03\x04fakezip")

    extracted_ddr = tmp_path / "DDR_extracted"
    _save_rgb_image(extracted_ddr / "lesion_segmentation/train/image/001.jpg", 32, 32)
    _save_mask_image(extracted_ddr / "lesion_segmentation/train/label/MA/001.tif", 32, 32)

    locator = locate_dataset_candidates(
        search_dirs=[downloads_dir, extracted_ddr],
        dataset_config_path=tmp_cfg,
    )

    assert len(locator["detected_archives"]) == 1
    assert locator["detected_archives"][0]["inferred_dataset"] == "IDRiD"
    assert len(locator["candidate_directories"]) == 1
    assert locator["candidate_directories"][0]["inferred_dataset"] == "DDR"
    assert locator["candidate_directories"][0]["recognized_layout"] == "ddr_official"


def test_cli_main_and_summary_formatting(tmp_path: Path) -> None:
    """Verify CLI main runs cleanly and can export JSON inventory and locator results."""
    ddr_dir = tmp_path / "raw_ddr"
    idrid_dir = tmp_path / "raw_idrid"
    _save_rgb_image(ddr_dir / "lesion_segmentation/train/image/001.jpg", 32, 32)
    _save_mask_image(ddr_dir / "lesion_segmentation/train/label/MA/001.tif", 32, 32)
    idrid_dir.mkdir(parents=True, exist_ok=True)

    tmp_cfg = _write_temp_dataset_config(tmp_path, ddr_dir, idrid_dir)
    json_out = tmp_path / "inventory_out.json"

    exit_code = inspect_cli_main(["--config", str(tmp_cfg), "--json-out", str(json_out)])
    assert exit_code == 0
    assert json_out.exists()

    loaded = json.loads(json_out.read_text(encoding="utf-8"))
    assert "datasets" in loaded
    assert "locator" in loaded
    assert "DDR" in loaded["datasets"]
    assert "IDRiD" in loaded["datasets"]

    summary_text = format_inventory_summary(
        loaded["datasets"], locator_report=loaded["locator"]
    )
    assert "Dataset: DDR" in summary_text
    assert "Dataset: IDRiD" in summary_text
    assert "Recommended Next Action" in summary_text
