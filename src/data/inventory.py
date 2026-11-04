"""Dataset inventory, candidate locator, and annotation audit utilities for DDR and IDRiD."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.utils.config import load_yaml_config
from src.utils.paths import resolve_project_path

SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
)

OTHER_ANNOTATION_EXTENSIONS: tuple[str, ...] = (
    ".xml",
    ".json",
    ".csv",
    ".txt",
)

IGNORED_DOC_FILENAMES: frozenset[str] = frozenset(
    {
        "license.txt",
        "cc-by-4.0.txt",
        "readme.txt",
        "readme.md",
    }
)

SUPPORTED_ARCHIVE_EXTENSIONS: tuple[str, ...] = (
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
)

MASK_DIR_KEYWORDS: tuple[str, ...] = (
    "mask",
    "masks",
    "label",
    "labels",
    "groundtruth",
    "groundtruths",
    "gt",
    "annotation",
    "annotations",
)

GRADING_DIR_KEYWORDS: tuple[str, ...] = (
    "dr_grading",
    "disease grading",
    "b. disease grading",
)

DETECTION_DIR_KEYWORDS: tuple[str, ...] = (
    "lesion_detection",
    "c. localization",
)

LESION_ALIASES: dict[str, str] = {
    "ma": "MA",
    "microaneurysms": "MA",
    "1. microaneurysms": "MA",
    "he": "HE",
    "hemorrhages": "HE",
    "haemorrhages": "HE",
    "2. haemorrhages": "HE",
    "ex": "EX",
    "hard_exudates": "EX",
    "hard exudates": "EX",
    "3. hard exudates": "EX",
    "se": "SE",
    "soft_exudates": "SE",
    "soft exudates": "SE",
    "4. soft exudates": "SE",
    "od": "OD",
    "optic_disc": "OD",
    "5. optic disc": "OD",
}

_MASK_STEM_SUFFIX_RE = re.compile(
    r"^(?P<base>.+?)[_-](?P<suffix>MA|HE|EX|SE|OD|mask|gt|label)$",
    re.IGNORECASE,
)


def is_supported_image_file(path: Path) -> bool:
    """Return ``True`` if ``path`` has a supported raster image/mask extension."""
    return (
        path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def is_archive_file(path: Path) -> bool:
    """Return ``True`` if ``path`` is a recognized compressed archive file."""
    if not path.is_file() or path.name.startswith("."):
        return False
    lower_name = path.name.lower()
    if lower_name.endswith((".tar.gz", ".tar.bz2", ".tar.xz")):
        return True
    return path.suffix.lower() in SUPPORTED_ARCHIVE_EXTENSIONS


def get_archive_extraction_hint(archive_path: Path, target_dir: str | Path) -> str:
    """Return a manual extraction command hint appropriate for the archive format."""
    lower_name = archive_path.name.lower()
    target_str = str(target_dir)
    arch_str = str(archive_path)
    if lower_name.endswith(".zip"):
        return f'Expand-Archive -Path "{arch_str}" -DestinationPath "{target_str}"'
    if lower_name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
        return f'tar -xf "{arch_str}" -C "{target_str}"'
    if lower_name.endswith((".7z", ".rar")):
        return f'7z x "{arch_str}" -o"{target_str}"'
    return f'Extract "{arch_str}" into "{target_str}" using your archive utility.'


def _classify_path_role(rel_path: Path) -> str:
    """Classify a file path relative to the dataset root.

    Returns one of: ``'mask'``, ``'image'``, ``'grading_image'``, ``'archive'``,
    ``'other_annotation'``, or ``'ignored'``.
    """
    suffix = rel_path.suffix.lower()
    lower_name = rel_path.name.lower()
    if lower_name in IGNORED_DOC_FILENAMES:
        return "ignored"
    if lower_name.endswith((".tar.gz", ".tar.bz2", ".tar.xz")) or suffix in SUPPORTED_ARCHIVE_EXTENSIONS:
        return "archive"
    if suffix in OTHER_ANNOTATION_EXTENSIONS:
        return "other_annotation"
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        return "ignored"

    parent_parts_lower = [part.lower().strip() for part in rel_path.parent.parts]

    # Check if inside an auxiliary grading or detection task folder (e.g., DDR DR_grading)
    for lowered in parent_parts_lower:
        if lowered in GRADING_DIR_KEYWORDS or any(gk in lowered for gk in GRADING_DIR_KEYWORDS):
            return "grading_image"
        if lowered in DETECTION_DIR_KEYWORDS or any(dk in lowered for dk in DETECTION_DIR_KEYWORDS):
            return "ignored"

    for lowered in parent_parts_lower:
        if lowered in MASK_DIR_KEYWORDS or any(
            kw in lowered for kw in ("groundtruth", "mask", "label", "annotation")
        ):
            return "mask"

    if _MASK_STEM_SUFFIX_RE.match(rel_path.stem):
        for lowered in parent_parts_lower:
            if "image" in lowered:
                return "image"
        return "mask"

    return "image"


def _infer_lesion_class(rel_path: Path) -> str:
    """Infer the lesion class tag (e.g., ``'MA'``, ``'HE'``, ``'EX'``, ``'SE'``, ``'OD'``, or ``'generic'``)."""
    for part in reversed(rel_path.parent.parts):
        lowered = part.lower().strip()
        if lowered in LESION_ALIASES:
            return LESION_ALIASES[lowered]

    match = _MASK_STEM_SUFFIX_RE.match(rel_path.stem)
    if match:
        suffix = match.group("suffix").lower()
        if suffix in LESION_ALIASES:
            return LESION_ALIASES[suffix]

    return "generic"


def _detect_recognized_layout(rel_dirs: set[str]) -> str:
    """Identify whether observed relative directories match known DDR or IDRiD layouts."""
    lowered_dirs = " | ".join(d.lower() for d in rel_dirs)
    if "lesion_segmentation" in lowered_dirs or "dr_grading" in lowered_dirs:
        return "ddr_official"
    if (
        "original images" in lowered_dirs
        or "all segmentation groundtruths" in lowered_dirs
        or "a. segmentation" in lowered_dirs
    ):
        return "idrid_official"
    if rel_dirs:
        return "generic_image_mask"
    return "none"


def _normalize_mask_sample_id(mask_stem: str, known_image_stems: set[str]) -> str:
    """Resolve a mask filename stem to its canonical image sample ID."""
    if mask_stem in known_image_stems:
        return mask_stem

    match = _MASK_STEM_SUFFIX_RE.match(mask_stem)
    if match:
        candidate = match.group("base")
        if candidate in known_image_stems or not known_image_stems:
            return candidate
        return candidate

    return mask_stem


def _inspect_raster_file(path: Path, collect_unique_values: bool = False) -> dict[str, Any]:
    """Open a raster image or mask file and inspect its dimensions, channels, and pixel values."""
    if not collect_unique_values:
        try:
            with Image.open(path) as img:
                width, height = int(img.size[0]), int(img.size[1])
                bands = img.getbands()
                channels = len(bands) if bands else 1
                img.verify()
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
            return {
                "readable": False,
                "error": str(exc),
                "height": None,
                "width": None,
                "channels": None,
                "unique_values": [],
                "is_empty_mask": False,
            }

        if height <= 0 or width <= 0:
            return {
                "readable": False,
                "error": f"Invalid image dimensions: {height}x{width}",
                "height": None,
                "width": None,
                "channels": None,
                "unique_values": [],
                "is_empty_mask": False,
            }

        return {
            "readable": True,
            "error": None,
            "height": height,
            "width": width,
            "channels": channels,
            "unique_values": [],
            "is_empty_mask": False,
        }

    try:
        with Image.open(path) as img:
            arr = np.asarray(img)
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        return {
            "readable": False,
            "error": str(exc),
            "height": None,
            "width": None,
            "channels": None,
            "unique_values": [],
            "is_empty_mask": False,
        }

    if arr.ndim < 2:
        return {
            "readable": False,
            "error": f"Invalid array dimensions: {arr.shape}",
            "height": None,
            "width": None,
            "channels": None,
            "unique_values": [],
            "is_empty_mask": False,
        }

    height, width = int(arr.shape[0]), int(arr.shape[1])
    channels = 1 if arr.ndim == 2 else int(arr.shape[2])

    uniq = np.unique(arr)
    is_empty = bool(uniq.size == 1 and int(uniq[0]) == 0)
    if uniq.size <= 256:
        unique_values = [int(v) for v in uniq.tolist()]
    else:
        unique_values = [int(uniq.min()), int(uniq.max())]

    return {
        "readable": True,
        "error": None,
        "height": height,
        "width": width,
        "channels": channels,
        "unique_values": unique_values,
        "is_empty_mask": is_empty,
    }


def inspect_dataset_directory(
    dataset_name: str,
    dataset_root: str | Path,
) -> dict[str, Any]:
    """Perform a comprehensive filesystem and annotation audit of a dataset root directory.

    Args:
        dataset_name: Name of the dataset (e.g., ``"DDR"`` or ``"IDRiD"``).
        dataset_root: Path to the root directory of the dataset.

    Returns:
        Structured dictionary summarizing file counts, extensions, readability,
        dimensions, channel counts, mask unique values, pairing status, archive files,
        and anomalies.
    """
    root_path = Path(dataset_root)
    report: dict[str, Any] = {
        "dataset_name": dataset_name,
        "root_path": str(root_path),
        "root_exists": root_path.exists() and root_path.is_dir(),
        "status": "missing_directory",
        "recognized_layout": "none",
        "archive_files": [],
        "image_directories": [],
        "mask_directories": [],
        "grading_image_directories": [],
        "other_annotation_files": [],
        "image_count": 0,
        "grading_image_count": 0,
        "mask_count": 0,
        "empty_mask_count": 0,
        "non_empty_mask_count": 0,
        "readable_image_count": 0,
        "readable_mask_count": 0,
        "unreadable_images": [],
        "unreadable_masks": [],
        "image_extensions": [],
        "mask_extensions": [],
        "image_extension_counts": {},
        "mask_extension_counts": {},
        "image_dimensions": {},
        "mask_dimensions": {},
        "image_channel_counts": {},
        "mask_channel_counts": {},
        "mask_unique_values": [],
        "lesion_class_counts": {},
        "lesion_class_unique_values": {},
        "paired_sample_count": 0,
        "images_missing_masks": [],
        "unmatched_masks": [],
        "duplicate_image_ids": [],
        "duplicate_mask_entries": [],
        "dimension_mismatches": [],
    }

    if not report["root_exists"]:
        return report

    all_files = sorted(
        [
            p
            for p in root_path.rglob("*")
            if p.is_file() and not p.name.startswith(".")
        ]
    )
    if not all_files:
        report["status"] = "empty"
        return report

    image_files: list[Path] = []
    grading_image_files: list[Path] = []
    mask_files: list[Path] = []
    other_annotations: list[str] = []
    archives: list[str] = []

    for file_path in all_files:
        rel_path = file_path.relative_to(root_path)
        role = _classify_path_role(rel_path)
        if role == "image":
            image_files.append(file_path)
        elif role == "grading_image":
            grading_image_files.append(file_path)
        elif role == "mask":
            mask_files.append(file_path)
        elif role == "other_annotation":
            other_annotations.append(rel_path.as_posix())
        elif role == "archive":
            archives.append(rel_path.as_posix())

    # If a dataset contains ONLY grading images and no segmentation images, treat them as primary images
    if not image_files and grading_image_files and not mask_files:
        image_files = grading_image_files
        grading_image_files = []

    report["archive_files"] = sorted(archives)

    if not image_files and not mask_files and not other_annotations:
        report["status"] = "archives_only" if archives else "empty"
        return report

    report["status"] = "present"
    report["other_annotation_files"] = sorted(other_annotations)
    report["image_count"] = len(image_files)
    report["grading_image_count"] = len(grading_image_files)
    report["mask_count"] = len(mask_files)

    image_dirs = sorted(
        {
            p.relative_to(root_path).parent.as_posix()
            for p in image_files
        }
    )
    grading_dirs = sorted(
        {
            p.relative_to(root_path).parent.as_posix()
            for p in grading_image_files
        }
    )
    mask_dirs = sorted(
        {
            p.relative_to(root_path).parent.as_posix()
            for p in mask_files
        }
    )
    report["image_directories"] = image_dirs
    report["grading_image_directories"] = grading_dirs
    report["mask_directories"] = mask_dirs
    report["recognized_layout"] = _detect_recognized_layout(
        set(image_dirs) | set(mask_dirs) | set(grading_dirs)
    )

    # Track segmentation image metadata and sample IDs
    image_ext_counts: dict[str, int] = {}
    image_dims: dict[str, int] = {}
    image_channels: dict[int, int] = {}
    image_by_id: dict[str, list[Path]] = {}
    image_shape_by_id: dict[str, tuple[int, int]] = {}

    for img_path in image_files:
        rel_str = img_path.relative_to(root_path).as_posix()
        ext = img_path.suffix.lower()
        image_ext_counts[ext] = image_ext_counts.get(ext, 0) + 1

        sample_id = img_path.stem
        image_by_id.setdefault(sample_id, []).append(img_path)

        info = _inspect_raster_file(img_path, collect_unique_values=False)
        if not info["readable"]:
            report["unreadable_images"].append(rel_str)
            continue

        report["readable_image_count"] += 1
        dim_key = f"{info['height']}x{info['width']}"
        image_dims[dim_key] = image_dims.get(dim_key, 0) + 1
        ch = int(info["channels"])
        image_channels[ch] = image_channels.get(ch, 0) + 1
        image_shape_by_id[sample_id] = (int(info["height"]), int(info["width"]))

    duplicate_images = sorted(
        [sid for sid, paths in image_by_id.items() if len(paths) > 1]
    )
    report["duplicate_image_ids"] = duplicate_images
    report["image_extension_counts"] = dict(sorted(image_ext_counts.items()))
    report["image_extensions"] = sorted(image_ext_counts.keys())
    report["image_dimensions"] = dict(sorted(image_dims.items()))
    report["image_channel_counts"] = dict(sorted(image_channels.items()))

    # Track mask metadata, lesion classes, and pairing
    known_image_stems = set(image_by_id.keys())
    mask_ext_counts: dict[str, int] = {}
    mask_dims: dict[str, int] = {}
    mask_channels: dict[int, int] = {}
    global_unique_values: set[int] = set()
    lesion_counts: dict[str, int] = {}
    lesion_unique_vals: dict[str, set[int]] = {}
    masks_by_sample_id: dict[str, list[Path]] = {}
    mask_sample_class_seen: dict[tuple[str, str], list[str]] = {}
    empty_masks = 0
    non_empty_masks = 0

    for m_path in mask_files:
        rel_path = m_path.relative_to(root_path)
        rel_str = rel_path.as_posix()
        ext = m_path.suffix.lower()
        mask_ext_counts[ext] = mask_ext_counts.get(ext, 0) + 1

        lesion_cls = _infer_lesion_class(rel_path)
        lesion_counts[lesion_cls] = lesion_counts.get(lesion_cls, 0) + 1

        sample_id = _normalize_mask_sample_id(m_path.stem, known_image_stems)
        masks_by_sample_id.setdefault(sample_id, []).append(m_path)
        mask_sample_class_seen.setdefault((sample_id, lesion_cls), []).append(rel_str)

        info = _inspect_raster_file(m_path, collect_unique_values=True)
        if not info["readable"]:
            report["unreadable_masks"].append(rel_str)
            continue

        report["readable_mask_count"] += 1
        if info["is_empty_mask"]:
            empty_masks += 1
        else:
            non_empty_masks += 1

        dim_key = f"{info['height']}x{info['width']}"
        mask_dims[dim_key] = mask_dims.get(dim_key, 0) + 1
        ch = int(info["channels"])
        mask_channels[ch] = mask_channels.get(ch, 0) + 1

        for val in info["unique_values"]:
            global_unique_values.add(val)
            lesion_unique_vals.setdefault(lesion_cls, set()).add(val)

        if sample_id in image_shape_by_id:
            img_h, img_w = image_shape_by_id[sample_id]
            mask_h, mask_w = int(info["height"]), int(info["width"])
            if (img_h, img_w) != (mask_h, mask_w):
                img_rel = image_by_id[sample_id][0].relative_to(root_path).as_posix()
                report["dimension_mismatches"].append(
                    {
                        "sample_id": sample_id,
                        "image_path": img_rel,
                        "mask_path": rel_str,
                        "image_size": f"{img_h}x{img_w}",
                        "mask_size": f"{mask_h}x{mask_w}",
                    }
                )

    duplicate_masks = sorted(
        [
            f"{sid}:{lcls}"
            for (sid, lcls), entries in mask_sample_class_seen.items()
            if len(entries) > 1
        ]
    )
    report["empty_mask_count"] = empty_masks
    report["non_empty_mask_count"] = non_empty_masks
    report["duplicate_mask_entries"] = duplicate_masks
    report["mask_extension_counts"] = dict(sorted(mask_ext_counts.items()))
    report["mask_extensions"] = sorted(mask_ext_counts.keys())
    report["mask_dimensions"] = dict(sorted(mask_dims.items()))
    report["mask_channel_counts"] = dict(sorted(mask_channels.items()))
    report["mask_unique_values"] = sorted(global_unique_values)
    report["lesion_class_counts"] = dict(sorted(lesion_counts.items()))
    report["lesion_class_unique_values"] = {
        k: sorted(v) for k, v in sorted(lesion_unique_vals.items())
    }

    # Pairing analysis
    matched_sample_ids = sorted(known_image_stems & set(masks_by_sample_id.keys()))
    missing_mask_ids = sorted(known_image_stems - set(masks_by_sample_id.keys()))
    unmatched_mask_ids = sorted(set(masks_by_sample_id.keys()) - known_image_stems)

    report["paired_sample_count"] = len(matched_sample_ids)
    report["images_missing_masks"] = [
        image_by_id[sid][0].relative_to(root_path).as_posix()
        for sid in missing_mask_ids
    ]
    unmatched_mask_files: list[str] = []
    for sid in unmatched_mask_ids:
        for mp in masks_by_sample_id[sid]:
            unmatched_mask_files.append(mp.relative_to(root_path).as_posix())
    report["unmatched_masks"] = sorted(unmatched_mask_files)

    return report


def inspect_configured_datasets(
    dataset_config_path: str | Path = "dataset.yaml",
) -> dict[str, dict[str, Any]]:
    """Inspect all datasets configured in ``configs/dataset.yaml``."""
    cfg = load_yaml_config(dataset_config_path, validate=True)
    paths_cfg = cfg.get("paths", {})
    dataset_names = cfg.get("dataset_names", ["DDR", "IDRiD"])

    results: dict[str, dict[str, Any]] = {}
    for name in dataset_names:
        path_key = f"raw_{name.lower()}"
        rel_dir = paths_cfg.get(path_key, f"data/raw/{name}")
        abs_dir = resolve_project_path(rel_dir)
        results[name] = inspect_dataset_directory(name, abs_dir)

    return results


def _infer_dataset_affinity(path: Path) -> str:
    """Infer whether an archive or folder path likely belongs to DDR or IDRiD."""
    full_str = path.as_posix().lower()
    if "idrid" in full_str or "original images" in full_str or "segmentation groundtruths" in full_str:
        return "IDRiD"
    if "ddr" in full_str or "lesion_segmentation" in full_str or "dr_grading" in full_str:
        return "DDR"
    return "unknown"


def locate_dataset_candidates(
    search_dirs: Sequence[str | Path] | None = None,
    dataset_config_path: str | Path = "dataset.yaml",
    configured_reports: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Locate downloaded DDR/IDRiD archives or extracted folders without moving/modifying files.

    By default (when ``search_dirs`` is ``None``), inspects only the project's
    ``data/raw`` directory and configured dataset roots. Additional directories are
    scanned only if explicitly provided in ``search_dirs``.

    Args:
        search_dirs: Optional sequence of user-provided directories to scan for
            downloaded archives or extracted DDR/IDRiD folders.
        dataset_config_path: Dataset configuration file name or path.
        configured_reports: Optional pre-computed output of ``inspect_configured_datasets``
            to avoid redundant filesystem scans.

    Returns:
        Dictionary summarizing configured roots status, detected archives,
        candidate extracted directories, and the recommended next action.
    """
    if configured_reports is None:
        configured_reports = inspect_configured_datasets(dataset_config_path)

    configured_status = {
        name: rep["status"] for name, rep in configured_reports.items()
    }

    raw_base_dir = resolve_project_path("data/raw")
    detected_archives: list[dict[str, Any]] = []
    candidate_directories: list[dict[str, Any]] = []
    seen_archives: set[Path] = set()

    # Collect any archives already discovered inside configured dataset roots
    for name, rep in configured_reports.items():
        root_p = Path(rep["root_path"])
        for rel_arch in rep.get("archive_files", []):
            arch_p = (root_p / rel_arch).resolve()
            if arch_p.exists() and arch_p not in seen_archives:
                seen_archives.add(arch_p)
                detected_archives.append(
                    {
                        "path": str(arch_p),
                        "size_bytes": int(arch_p.stat().st_size),
                        "inferred_dataset": name,
                        "target_directory": str(root_p),
                        "extraction_hint": get_archive_extraction_hint(arch_p, root_p),
                    }
                )

    # Also check top-level files directly inside data/raw/
    if raw_base_dir.exists() and raw_base_dir.is_dir():
        for p in sorted(raw_base_dir.iterdir()):
            if is_archive_file(p):
                resolved_p = p.resolve()
                if resolved_p in seen_archives:
                    continue
                seen_archives.add(resolved_p)
                affinity = _infer_dataset_affinity(p)
                target_dir = (
                    resolve_project_path(f"data/raw/{affinity}")
                    if affinity in ("DDR", "IDRiD")
                    else raw_base_dir
                )
                detected_archives.append(
                    {
                        "path": str(p),
                        "size_bytes": int(p.stat().st_size),
                        "inferred_dataset": affinity,
                        "target_directory": str(target_dir),
                        "extraction_hint": get_archive_extraction_hint(p, target_dir),
                    }
                )

    if search_dirs:
        for sdir in search_dirs:
            spath = Path(sdir)
            if not (spath.exists() and spath.is_dir()):
                continue
            for p in sorted(spath.rglob("*")):
                if is_archive_file(p):
                    resolved_p = p.resolve()
                    if resolved_p in seen_archives:
                        continue
                    seen_archives.add(resolved_p)
                    affinity = _infer_dataset_affinity(p)
                    target_dir = (
                        resolve_project_path(f"data/raw/{affinity}")
                        if affinity in ("DDR", "IDRiD")
                        else raw_base_dir
                    )
                    detected_archives.append(
                        {
                            "path": str(p),
                            "size_bytes": int(p.stat().st_size),
                            "inferred_dataset": affinity,
                            "target_directory": str(target_dir),
                            "extraction_hint": get_archive_extraction_hint(p, target_dir),
                        }
                    )

            probe = inspect_dataset_directory(spath.name, spath)
            if probe["image_count"] > 0 or probe["mask_count"] > 0:
                affinity = _infer_dataset_affinity(spath)
                if affinity == "unknown" and probe["recognized_layout"] == "ddr_official":
                    affinity = "DDR"
                elif affinity == "unknown" and probe["recognized_layout"] == "idrid_official":
                    affinity = "IDRiD"
                target_dir = (
                    resolve_project_path(f"data/raw/{affinity}")
                    if affinity in ("DDR", "IDRiD")
                    else raw_base_dir
                )
                candidate_directories.append(
                    {
                        "path": str(spath),
                        "inferred_dataset": affinity,
                        "recognized_layout": probe["recognized_layout"],
                        "image_count": probe["image_count"],
                        "mask_count": probe["mask_count"],
                        "image_directories": probe["image_directories"],
                        "mask_directories": probe["mask_directories"],
                        "placement_target": str(target_dir),
                    }
                )

    all_present = all(st == "present" for st in configured_status.values())
    if all_present:
        next_action = "ready_for_audit_and_preprocessing"
    elif detected_archives:
        next_action = "extract_archives_to_configured_roots"
    elif candidate_directories:
        next_action = "place_candidate_directories_under_data_raw"
    else:
        next_action = "manual_acquisition_required"

    return {
        "configured_roots_status": configured_status,
        "detected_archives": detected_archives,
        "candidate_directories": candidate_directories,
        "recommended_next_action": next_action,
    }
