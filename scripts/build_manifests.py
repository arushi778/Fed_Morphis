"""CLI entrypoint to build and validate canonical DDR and IDRiD segmentation manifests.

Run with:
    .\\.venv311\\Scripts\\python.exe scripts\\build_manifests.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.manifests import build_configured_manifests
from src.utils.paths import resolve_project_path


def format_manifest_summary(manifests: dict[str, dict[str, Any]]) -> str:
    """Format a human-readable validation report from generated dataset manifests."""
    lines: list[str] = [
        "=" * 78,
        "FedMorphis Phase 1: Canonical Mask, FOV Cropping & Dataset Manifest Report",
        "=" * 78,
    ]

    for name, manifest in manifests.items():
        summary = manifest["summary"]
        fov = summary["fov_summary"]
        lines.append(f"\nDataset: {name}")
        lines.append("-" * 78)
        lines.append(f"  Canonical Channel Order  : {manifest['channel_order']}")
        lines.append(f"  Excluded Structures      : {manifest['excluded_anatomical_classes']}")
        lines.append(f"  Total Segmentation Cases : {summary['total_samples']}")
        lines.append(f"  Split Counts             : {summary['split_counts']}")
        lines.append(
            f"  Present Masks by Class   : {summary['present_mask_counts_by_class']}"
        )
        lines.append(
            f"  Missing Masks by Class   : {summary['missing_mask_counts_by_class']}"
        )
        lines.append(
            f"  Synthesized Zero Ch.     : {summary['synthesized_zero_channel_counts_by_class']}"
        )
        lines.append(
            f"  Empty On-Disk Masks      : {summary['empty_on_disk_mask_counts_by_class']}"
        )
        lines.append(
            f"  Positive Masks by Class  : {summary['positive_mask_counts_by_class']}"
        )
        lines.append(
            f"  Multichannel Source Masks: {summary['multichannel_source_mask_count']}"
        )
        lines.append(
            f"  Source Channel Dist.     : {summary['source_mask_channel_distribution']}"
        )
        lines.append(
            f"  Source Encoding Dist.    : {summary['source_mask_encoding_distribution']}"
        )
        lines.append(
            f"  Samples w/ Class Overlap : {summary['samples_with_overlapping_lesions']}"
        )
        lines.append(
            f"  Excluded OD Masks        : {summary['excluded_od_mask_count']}"
        )
        lines.append(
            f"  Excluded Grading Images  : {summary['excluded_grading_image_count']}"
        )
        lines.append(
            f"  Unreadable / Mismatched  : {summary['unreadable_sample_count']} / {summary['dimension_mismatch_count']}"
        )
        lines.append(
            f"  FOV Cropped vs Full-Size : {fov['cropped_sample_count']} cropped, {fov['full_image_sample_count']} full-size "
            f"(fallback_used={fov['fallback_used_count']}, boundary_expanded={fov['boundary_expanded_for_mask_count']})"
        )
        lines.append(
            f"  Lesion Pixels Outside FOV: {fov['lesion_pixels_outside_fov_total']}"
        )
        lines.append(
            f"  Mean FOV Area Retention  : {fov['mean_area_retention_ratio']:.4f}"
        )

    lines.append("\n" + "=" * 78)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for building and validating dataset manifests."""
    parser = argparse.ArgumentParser(
        description="Build and validate canonical 4-channel segmentation manifests for DDR and IDRiD."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="dataset.yaml",
        help="Dataset configuration YAML file name or path (default: dataset.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory to write manifest JSON files (default: data/processed/manifests).",
    )
    args = parser.parse_args(argv)

    manifests = build_configured_manifests(
        dataset_config_path=args.config,
        output_dir=args.output_dir,
    )
    print(format_manifest_summary(manifests))

    out_dir = (
        resolve_project_path(args.output_dir)
        if args.output_dir
        else resolve_project_path("data/processed/manifests")
    )
    print(f"Saved manifests to: {out_dir}")
    return 0


__all__ = [
    "format_manifest_summary",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())

