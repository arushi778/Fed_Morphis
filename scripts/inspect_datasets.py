"""CLI and library entrypoint for inspecting and locating DDR and IDRiD datasets.

Run with:
    .\\.venv311\\Scripts\\python.exe scripts\\inspect_datasets.py
    .\\.venv311\\Scripts\\python.exe scripts\\inspect_datasets.py --json-out logs\\dataset_inventory.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.inventory import (
    LESION_ALIASES,
    MASK_DIR_KEYWORDS,
    OTHER_ANNOTATION_EXTENSIONS,
    SUPPORTED_ARCHIVE_EXTENSIONS,
    SUPPORTED_IMAGE_EXTENSIONS,
    get_archive_extraction_hint,
    inspect_configured_datasets,
    inspect_dataset_directory,
    is_archive_file,
    is_supported_image_file,
    locate_dataset_candidates,
)
from src.utils.paths import resolve_project_path


def format_inventory_summary(
    results: dict[str, dict[str, Any]],
    locator_report: dict[str, Any] | None = None,
) -> str:
    """Format a multi-dataset inventory and locator dictionary into a readable console report."""
    lines: list[str] = [
        "=" * 76,
        "FedMorphis Phase 1: Dataset Inventory, Locator & Annotation Audit",
        "=" * 76,
    ]

    for name, info in results.items():
        lines.append(f"\nDataset: {name}")
        lines.append("-" * 76)
        lines.append(f"  Root Path                : {info['root_path']}")
        lines.append(f"  Root Exists              : {info['root_exists']}")
        lines.append(f"  Dataset Status           : {info['status'].upper()}")
        lines.append(f"  Recognized Layout        : {info.get('recognized_layout', 'none')}")

        if info.get("archive_files"):
            lines.append(f"  Archive Files Detected   : {info['archive_files']}")

        if info["status"] in ("missing_directory", "empty", "archives_only"):
            if info["status"] == "archives_only":
                lines.append(
                    "  Notice                   : Unextracted archive(s) found; extract into this directory."
                )
            else:
                lines.append(
                    "  Notice                   : No dataset image or annotation files found on disk."
                )
            continue

        lines.append(f"  Image Directories        : {info['image_directories']}")
        if info.get("grading_image_directories"):
            lines.append(f"  Grading Image Dirs (SSL) : {info['grading_image_directories']}")
        lines.append(f"  Mask Directories         : {info['mask_directories']}")
        lines.append(f"  Other Annotation Files   : {len(info['other_annotation_files'])}")
        lines.append(
            f"  Segmentation Images      : {info['image_count']} "
            f"(readable: {info['readable_image_count']}, unreadable: {len(info['unreadable_images'])})"
        )
        if info.get("grading_image_count", 0) > 0:
            lines.append(
                f"  Grading/SSL Images       : {info['grading_image_count']}"
            )
        lines.append(
            f"  Mask Files               : {info['mask_count']} "
            f"(readable: {info['readable_mask_count']}, unreadable: {len(info['unreadable_masks'])}, "
            f"non_empty: {info.get('non_empty_mask_count', 0)}, empty_all_zero: {info.get('empty_mask_count', 0)})"
        )
        lines.append(f"  Image Extensions         : {info['image_extension_counts']}")
        lines.append(f"  Mask Extensions          : {info['mask_extension_counts']}")
        lines.append(
            f"  Unique Image Dimensions  : {len(info['image_dimensions'])} distinct resolution(s)"
        )
        lines.append(f"  Image Channel Counts     : {info['image_channel_counts']}")
        lines.append(
            f"  Unique Mask Dimensions   : {len(info['mask_dimensions'])} distinct resolution(s)"
        )
        lines.append(f"  Mask Channel Counts      : {info['mask_channel_counts']}")
        lines.append(f"  Mask Unique Pixel Values : {info['mask_unique_values']}")
        lines.append(f"  Lesion Class File Counts : {info['lesion_class_counts']}")
        lines.append(f"  Lesion Class Pixel IDs   : {info['lesion_class_unique_values']}")
        lines.append(f"  Paired Image/Mask Samples: {info['paired_sample_count']}")
        lines.append(f"  Images Missing Masks     : {len(info['images_missing_masks'])}")
        lines.append(f"  Unmatched Masks          : {len(info['unmatched_masks'])}")
        lines.append(f"  Duplicate Image IDs      : {info['duplicate_image_ids']}")
        lines.append(f"  Duplicate Mask Entries   : {info['duplicate_mask_entries']}")
        lines.append(f"  Dimension Mismatches     : {len(info['dimension_mismatches'])}")

    if locator_report is not None:
        lines.append("\nDataset Locator & Acquisition Status")
        lines.append("-" * 76)
        lines.append(
            f"  Recommended Next Action  : {locator_report['recommended_next_action']}"
        )
        if locator_report["detected_archives"]:
            lines.append("  Detected Archives:")
            for arch in locator_report["detected_archives"]:
                lines.append(
                    f"    - {arch['path']} (dataset={arch['inferred_dataset']}, "
                    f"size={arch['size_bytes']} bytes)"
                )
                lines.append(f"      Hint: {arch['extraction_hint']}")
        if locator_report["candidate_directories"]:
            lines.append("  Candidate Extracted Directories:")
            for cand in locator_report["candidate_directories"]:
                lines.append(
                    f"    - {cand['path']} (dataset={cand['inferred_dataset']}, "
                    f"layout={cand['recognized_layout']}, images={cand['image_count']}, "
                    f"masks={cand['mask_count']})"
                )
                lines.append(f"      Target Placement: {cand['placement_target']}")

    lines.append("\n" + "=" * 76)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for inspecting and locating datasets."""
    parser = argparse.ArgumentParser(
        description="Inspect DDR and IDRiD dataset directories, locate archives, and audit annotations."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="dataset.yaml",
        help="Dataset configuration YAML file name or path (default: dataset.yaml).",
    )
    parser.add_argument(
        "--search-dir",
        action="append",
        default=None,
        help="Optional directory to scan for downloaded DDR/IDRiD archives or folders (can be specified multiple times).",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional file path to save the structured JSON inventory report.",
    )
    args = parser.parse_args(argv)

    results = inspect_configured_datasets(args.config)
    locator_report = locate_dataset_candidates(
        search_dirs=args.search_dir,
        dataset_config_path=args.config,
        configured_reports=results,
    )
    print(format_inventory_summary(results, locator_report=locator_report))

    if args.json_out:
        out_path = resolve_project_path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "datasets": results,
            "locator": locator_report,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved JSON inventory report to: {out_path}")

    return 0


__all__ = [
    "LESION_ALIASES",
    "MASK_DIR_KEYWORDS",
    "OTHER_ANNOTATION_EXTENSIONS",
    "SUPPORTED_ARCHIVE_EXTENSIONS",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "format_inventory_summary",
    "get_archive_extraction_hint",
    "inspect_configured_datasets",
    "inspect_dataset_directory",
    "is_archive_file",
    "is_supported_image_file",
    "locate_dataset_candidates",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
