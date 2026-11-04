"""Reproducible dataset manifest generation and validation for DDR and IDRiD segmentation.

Builds deterministic, validated segmentation manifests that:
- Preserve the official DDR (``train``, ``valid``, ``test``) and IDRiD (``train``, ``test``)
  split assignments without moving or modifying raw files.
- Recognize DDR's ``valid/segmentation label`` directory alongside ``train/label``
  and ``test/label``.
- Exclude DDR ``DR_grading`` and ``lesion_detection`` files and IDRiD ``OD``
  (Optic Disc) masks from the four-channel ``[MA, HE, EX, SE]`` lesion target.
- Distinguish missing mask files (synthesized zero channels, as in IDRiD ``HE``/``SE``)
  from explicit all-zero mask files on disk (as in DDR absent-lesion ``.tif`` masks).
- Verify image and canonical mask dimensions ``(H, W, 4)`` and compute deterministic
  retinal FOV crop metadata for every sample.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.fov import (
    DEFAULT_FOV_MARGIN,
    DEFAULT_FOV_THRESHOLD,
    DEFAULT_MIN_FOV_FRACTION,
    crop_image_and_mask_to_fov,
)
from src.data.inventory import (
    _classify_path_role,
    _infer_lesion_class,
    _normalize_mask_sample_id,
)
from src.data.masks import (
    CANONICAL_LESION_CHANNELS,
    EXCLUDED_ANATOMICAL_CLASSES,
    MaskConstructionError,
    load_sample_image_and_mask,
)
from src.utils.config import load_yaml_config
from src.utils.paths import get_project_root, resolve_project_path

MANIFEST_SCHEMA_VERSION: str = "1.0"
SPLIT_SORT_ORDER: dict[str, int] = {
    "train": 0,
    "valid": 1,
    "val": 1,
    "test": 2,
}


class ManifestValidationError(ValueError):
    """Raised when dataset manifest construction encounters invalid or ambiguous data."""


def infer_split_from_relative_path(rel_path: Path) -> str:
    """Infer canonical split name (``'train'``, ``'valid'``, or ``'test'``) from a relative path."""
    for part in reversed(rel_path.parent.parts):
        lowered = part.lower().strip()
        if (
            lowered in {"train", "training", "a. training set"}
            or "training set" in lowered
            or lowered.startswith("train")
        ):
            return "train"
        if (
            lowered in {"valid", "val", "validation", "c. validation set"}
            or "validation set" in lowered
            or lowered.startswith(("valid", "val"))
        ):
            return "valid"
        if (
            lowered in {"test", "testing", "b. testing set"}
            or "testing set" in lowered
            or lowered.startswith("test")
        ):
            return "test"
    return "train"


def _to_portable_relative_path(path: Path, fallback_root: Path) -> str:
    """Convert ``path`` to a POSIX path relative to the project root (or ``fallback_root``)."""
    resolved = path.resolve()
    project_root = get_project_root().resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        pass
    try:
        return resolved.relative_to(fallback_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def build_dataset_manifest(
    dataset_name: str,
    dataset_root: str | Path,
    fov_threshold: int = DEFAULT_FOV_THRESHOLD,
    fov_margin: int = DEFAULT_FOV_MARGIN,
    min_fov_fraction: float = DEFAULT_MIN_FOV_FRACTION,
    mask_threshold: int = 0,
    preserve_boundary_lesions: bool = True,
    require_at_least_one_mask: bool = True,
) -> dict[str, Any]:
    """Construct and validate a complete segmentation manifest for a dataset directory.

    Args:
        dataset_name: Dataset identifier (e.g., ``"DDR"`` or ``"IDRiD"``).
        dataset_root: Root directory containing the raw dataset.
        fov_threshold: Intensity threshold for retinal FOV detection (default ``10``).
        fov_margin: Optional pixel margin around detected FOV bounds (default ``0``).
        min_fov_fraction: Minimum FOV bounding-box area fraction before fallback (default ``0.01``).
        mask_threshold: Binarization threshold for lesion masks (default ``0``).
        preserve_boundary_lesions: Expand FOV crop if needed so no positive lesion
            pixel falls outside the crop (default ``True``).
        require_at_least_one_mask: Raise :class:`ManifestValidationError` if any
            segmentation image has no matching lesion mask files on disk.

    Returns:
        JSON-serializable manifest dictionary with ``summary`` and ``records``.

    Raises:
        ManifestValidationError: If the dataset directory is missing/empty, or if
            duplicate sample IDs, unreadable files, unmatched masks, or dimension
            mismatches are found.
    """
    root_path = Path(dataset_root)
    if not root_path.exists() or not root_path.is_dir():
        raise ManifestValidationError(
            f"[{dataset_name}] Dataset root directory does not exist: {root_path}"
        )

    all_files = sorted(
        [p for p in root_path.rglob("*") if p.is_file() and not p.name.startswith(".")]
    )
    if not all_files:
        raise ManifestValidationError(
            f"[{dataset_name}] Dataset root directory is empty: {root_path}"
        )

    image_files: list[Path] = []
    mask_files: list[Path] = []
    excluded_grading_images = 0

    for file_path in all_files:
        rel_path = file_path.relative_to(root_path)
        role = _classify_path_role(rel_path)
        if role == "image":
            image_files.append(file_path)
        elif role == "mask":
            mask_files.append(file_path)
        elif role == "grading_image":
            excluded_grading_images += 1

    if not image_files:
        raise ManifestValidationError(
            f"[{dataset_name}] No segmentation images found under {root_path} "
            f"(excluded {excluded_grading_images} grading images)."
        )

    # Index segmentation images by sample_id and check for duplicates
    image_entry_by_id: dict[str, tuple[Path, str]] = {}
    for img_path in image_files:
        rel_path = img_path.relative_to(root_path)
        split = infer_split_from_relative_path(rel_path)
        sample_id = img_path.stem
        if sample_id in image_entry_by_id:
            prev_path, prev_split = image_entry_by_id[sample_id]
            raise ManifestValidationError(
                f"[{dataset_name}] Duplicate segmentation image sample_id '{sample_id}' "
                f"found in '{prev_path.relative_to(root_path).as_posix()}' (split={prev_split}) "
                f"and '{rel_path.as_posix()}' (split={split})."
            )
        image_entry_by_id[sample_id] = (img_path, split)

    known_image_stems = set(image_entry_by_id.keys())

    # Index masks by sample_id and canonical lesion class (excluding OD)
    masks_by_sample: dict[str, dict[str, Path]] = {sid: {} for sid in known_image_stems}
    excluded_od_masks = 0

    for m_path in mask_files:
        rel_path = m_path.relative_to(root_path)
        lesion_cls = _infer_lesion_class(rel_path)

        if lesion_cls in EXCLUDED_ANATOMICAL_CLASSES:
            excluded_od_masks += 1
            continue

        if lesion_cls not in CANONICAL_LESION_CHANNELS:
            raise ManifestValidationError(
                f"[{dataset_name}] Mask file '{rel_path.as_posix()}' resolved to unsupported "
                f"lesion class '{lesion_cls}'; expected one of {CANONICAL_LESION_CHANNELS}."
            )

        sample_id = _normalize_mask_sample_id(m_path.stem, known_image_stems)
        if sample_id not in image_entry_by_id:
            raise ManifestValidationError(
                f"[{dataset_name}] Unmatched mask file '{rel_path.as_posix()}' "
                f"(sample_id='{sample_id}') has no corresponding segmentation image."
            )

        img_path, img_split = image_entry_by_id[sample_id]
        mask_split = infer_split_from_relative_path(rel_path)
        if mask_split != img_split:
            raise ManifestValidationError(
                f"[{dataset_name}] Split mismatch for sample_id '{sample_id}': "
                f"image is in split '{img_split}' ({img_path.relative_to(root_path).as_posix()}) "
                f"but mask is in split '{mask_split}' ({rel_path.as_posix()})."
            )

        sample_masks = masks_by_sample[sample_id]
        if lesion_cls in sample_masks:
            prev_m = sample_masks[lesion_cls].relative_to(root_path).as_posix()
            raise ManifestValidationError(
                f"[{dataset_name}] Duplicate '{lesion_cls}' mask for sample_id '{sample_id}': "
                f"'{prev_m}' and '{rel_path.as_posix()}'."
            )
        sample_masks[lesion_cls] = m_path

    # Sort samples deterministically by (split_order, split, sample_id)
    sorted_sample_ids = sorted(
        known_image_stems,
        key=lambda sid: (
            SPLIT_SORT_ORDER.get(image_entry_by_id[sid][1], 99),
            image_entry_by_id[sid][1],
            sid,
        ),
    )

    records: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    present_counts = {c: 0 for c in CANONICAL_LESION_CHANNELS}
    missing_counts = {c: 0 for c in CANONICAL_LESION_CHANNELS}
    synthesized_counts = {c: 0 for c in CANONICAL_LESION_CHANNELS}
    empty_on_disk_counts = {c: 0 for c in CANONICAL_LESION_CHANNELS}
    positive_counts = {c: 0 for c in CANONICAL_LESION_CHANNELS}
    multichannel_mask_count = 0
    src_channel_dist: dict[str, int] = {}
    src_encoding_dist: dict[str, int] = {}
    overlapping_sample_count = 0

    cropped_sample_count = 0
    full_image_sample_count = 0
    fallback_used_count = 0
    boundary_expanded_count = 0
    lesion_pixels_outside_fov_total = 0
    area_ratios: list[float] = []

    for sample_id in sorted_sample_ids:
        img_path, split = image_entry_by_id[sample_id]
        sample_mask_map = masks_by_sample[sample_id]

        if require_at_least_one_mask and not sample_mask_map:
            raise ManifestValidationError(
                f"[{dataset_name}] Segmentation image '{img_path.relative_to(root_path).as_posix()}' "
                f"(sample_id='{sample_id}') has no matching lesion masks across {CANONICAL_LESION_CHANNELS}."
            )

        canonical_path_inputs: dict[str, Path | None] = {
            cls_name: sample_mask_map.get(cls_name)
            for cls_name in CANONICAL_LESION_CHANNELS
        }

        try:
            image_rgb, mask_res = load_sample_image_and_mask(
                image_path=img_path,
                mask_paths=canonical_path_inputs,
                threshold=mask_threshold,
            )
        except (FileNotFoundError, MaskConstructionError) as exc:
            raise ManifestValidationError(
                f"[{dataset_name}] Sample '{sample_id}' failed validation: {exc}"
            ) from exc

        _, _, fov_meta = crop_image_and_mask_to_fov(
            image=image_rgb,
            mask=mask_res.mask,
            threshold=fov_threshold,
            margin=fov_margin,
            min_fov_fraction=min_fov_fraction,
            preserve_boundary_lesions=preserve_boundary_lesions,
        )

        img_h, img_w = int(image_rgb.shape[0]), int(image_rgb.shape[1])
        img_c = int(image_rgb.shape[2])

        portable_img_path = _to_portable_relative_path(img_path, root_path.parent)
        portable_mask_paths: dict[str, str | None] = {
            cls_name: (
                _to_portable_relative_path(sample_mask_map[cls_name], root_path.parent)
                if cls_name in sample_mask_map
                else None
            )
            for cls_name in CANONICAL_LESION_CHANNELS
        }

        # Update summary counters
        split_counts[split] = split_counts.get(split, 0) + 1
        for cls_name in mask_res.present_classes:
            present_counts[cls_name] += 1
        for cls_name in mask_res.missing_classes:
            missing_counts[cls_name] += 1
        for cls_name in mask_res.synthesized_zero_classes:
            synthesized_counts[cls_name] += 1
        for cls_name in mask_res.empty_mask_classes:
            empty_on_disk_counts[cls_name] += 1
        for cls_name in mask_res.positive_classes:
            positive_counts[cls_name] += 1

        for cls_name in CANONICAL_LESION_CHANNELS:
            ch_cnt = mask_res.source_channel_counts[cls_name]
            if ch_cnt is not None:
                ch_key = str(ch_cnt)
                src_channel_dist[ch_key] = src_channel_dist.get(ch_key, 0) + 1
                if ch_cnt > 1:
                    multichannel_mask_count += 1
            enc = mask_res.source_unique_values[cls_name]
            if enc is not None:
                enc_key = str(enc)
                src_encoding_dist[enc_key] = src_encoding_dist.get(enc_key, 0) + 1

        if mask_res.overlapping_pixel_count > 0:
            overlapping_sample_count += 1

        orig_area = float(img_h * img_w)
        crop_area = float(fov_meta.cropped_height * fov_meta.cropped_width)
        area_ratios.append(crop_area / orig_area if orig_area > 0 else 1.0)

        if (fov_meta.cropped_height, fov_meta.cropped_width) < (img_h, img_w):
            cropped_sample_count += 1
        else:
            full_image_sample_count += 1

        if fov_meta.fallback_used:
            fallback_used_count += 1
        if fov_meta.boundary_expanded_for_mask:
            boundary_expanded_count += 1
        lesion_pixels_outside_fov_total += fov_meta.lesion_pixels_outside_fov

        record: dict[str, Any] = {
            "dataset_name": dataset_name,
            "sample_id": sample_id,
            "split": split,
            "image_path": portable_img_path,
            "mask_paths": portable_mask_paths,
            "channel_order": list(CANONICAL_LESION_CHANNELS),
            "present_lesion_masks": list(mask_res.present_classes),
            "missing_lesion_masks": list(mask_res.missing_classes),
            "synthesized_zero_channels": list(mask_res.synthesized_zero_classes),
            "empty_lesion_masks": list(mask_res.empty_mask_classes),
            "positive_lesion_masks": list(mask_res.positive_classes),
            "has_missing_masks": len(mask_res.missing_classes) > 0,
            "lesion_pixel_counts": dict(mask_res.foreground_pixel_counts),
            "source_mask_channels": dict(mask_res.source_channel_counts),
            "source_mask_encodings": dict(mask_res.source_unique_values),
            "overlapping_pixel_count": int(mask_res.overlapping_pixel_count),
            "image_dimensions": {
                "height": img_h,
                "width": img_w,
                "channels": img_c,
            },
            "canonical_mask_dimensions": [
                img_h,
                img_w,
                len(CANONICAL_LESION_CHANNELS),
            ],
            "fov_crop": fov_meta.to_dict(),
        }
        records.append(record)

    mean_retention = (
        round(sum(area_ratios) / len(area_ratios), 6) if area_ratios else 1.0
    )

    summary: dict[str, Any] = {
        "total_samples": len(records),
        "split_counts": split_counts,
        "present_mask_counts_by_class": present_counts,
        "missing_mask_counts_by_class": missing_counts,
        "synthesized_zero_channel_counts_by_class": synthesized_counts,
        "empty_on_disk_mask_counts_by_class": empty_on_disk_counts,
        "positive_mask_counts_by_class": positive_counts,
        "multichannel_source_mask_count": multichannel_mask_count,
        "source_mask_channel_distribution": dict(sorted(src_channel_dist.items())),
        "source_mask_encoding_distribution": dict(sorted(src_encoding_dist.items())),
        "samples_with_overlapping_lesions": overlapping_sample_count,
        "excluded_od_mask_count": excluded_od_masks,
        "excluded_grading_image_count": excluded_grading_images,
        "unreadable_sample_count": 0,
        "dimension_mismatch_count": 0,
        "fov_summary": {
            "cropped_sample_count": cropped_sample_count,
            "full_image_sample_count": full_image_sample_count,
            "fallback_used_count": fallback_used_count,
            "boundary_expanded_for_mask_count": boundary_expanded_count,
            "lesion_pixels_outside_fov_total": lesion_pixels_outside_fov_total,
            "mean_area_retention_ratio": mean_retention,
        },
    }

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "channel_order": list(CANONICAL_LESION_CHANNELS),
        "excluded_anatomical_classes": list(EXCLUDED_ANATOMICAL_CLASSES),
        "fov_config": {
            "threshold": fov_threshold,
            "margin": fov_margin,
            "min_fov_fraction": min_fov_fraction,
            "preserve_boundary_lesions": preserve_boundary_lesions,
        },
        "summary": summary,
        "records": records,
    }


def save_manifest_json(manifest: dict[str, Any], output_path: str | Path) -> Path:
    """Write a manifest dictionary deterministically to a JSON file."""
    out_p = resolve_project_path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2)
    out_p.write_text(payload + "\n", encoding="utf-8")
    return out_p


def load_manifest_json(manifest_path: str | Path) -> dict[str, Any]:
    """Load and parse a dataset manifest JSON file."""
    path = resolve_project_path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_configured_manifests(
    dataset_config_path: str | Path = "dataset.yaml",
    output_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Build and save segmentation manifests for all datasets configured in ``dataset.yaml``.

    Saves per-dataset manifests (``ddr_manifest.json``, ``idrid_manifest.json``)
    and a combined summary manifest (``combined_manifest.json``) under
    ``data/processed/manifests/`` (or ``output_dir`` when specified).
    """
    cfg = load_yaml_config(dataset_config_path, validate=True)
    paths_cfg = cfg.get("paths", {})
    dataset_names = cfg.get("dataset_names", ["DDR", "IDRiD"])

    fov_cfg = cfg.get("fov_cropping", {})
    fov_threshold = int(fov_cfg.get("intensity_threshold", DEFAULT_FOV_THRESHOLD))
    fov_margin = int(fov_cfg.get("margin_pixels", DEFAULT_FOV_MARGIN))
    min_fov_fraction = float(
        fov_cfg.get("min_fov_area_fraction", DEFAULT_MIN_FOV_FRACTION)
    )
    mask_threshold = int(cfg.get("mask_binarization_threshold", 0))

    if output_dir is None:
        rel_out = paths_cfg.get("manifests", "data/processed/manifests")
        manifest_dir = resolve_project_path(rel_out)
    else:
        manifest_dir = resolve_project_path(output_dir)

    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifests: dict[str, dict[str, Any]] = {}
    combined_records: list[dict[str, Any]] = []
    combined_summaries: dict[str, Any] = {}

    for name in dataset_names:
        path_key = f"raw_{name.lower()}"
        rel_raw = paths_cfg.get(path_key, f"data/raw/{name}")
        abs_raw = resolve_project_path(rel_raw)

        manifest = build_dataset_manifest(
            dataset_name=name,
            dataset_root=abs_raw,
            fov_threshold=fov_threshold,
            fov_margin=fov_margin,
            min_fov_fraction=min_fov_fraction,
            mask_threshold=mask_threshold,
        )
        file_path = manifest_dir / f"{name.lower()}_manifest.json"
        save_manifest_json(manifest, file_path)

        manifests[name] = manifest
        combined_records.extend(manifest["records"])
        combined_summaries[name] = manifest["summary"]

    combined_manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_name": "combined",
        "datasets": list(dataset_names),
        "channel_order": list(CANONICAL_LESION_CHANNELS),
        "excluded_anatomical_classes": list(EXCLUDED_ANATOMICAL_CLASSES),
        "fov_config": {
            "threshold": fov_threshold,
            "margin": fov_margin,
            "min_fov_fraction": min_fov_fraction,
            "preserve_boundary_lesions": True,
        },
        "total_samples": len(combined_records),
        "dataset_summaries": combined_summaries,
        "records": combined_records,
    }
    save_manifest_json(combined_manifest, manifest_dir / "combined_manifest.json")

    return manifests


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "SPLIT_SORT_ORDER",
    "ManifestValidationError",
    "build_configured_manifests",
    "build_dataset_manifest",
    "infer_split_from_relative_path",
    "load_manifest_json",
    "save_manifest_json",
]

