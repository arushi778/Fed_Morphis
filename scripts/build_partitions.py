"""CLI entrypoint for building federated client partitions and running real-data smoke tests.

Run with:
    .\\.venv311\\Scripts\\python.exe scripts\\build_partitions.py --smoke-test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.dataset import (
    RetinalLesionSegmentationDataset,
    build_segmentation_dataloader,
)
from src.data.manifests import load_manifest_json
from src.data.masks import CANONICAL_LESION_CHANNELS
from src.data.partitioning import build_configured_partitions
from src.utils.paths import resolve_project_path
from src.utils.seed import set_seed


def run_real_data_smoke_test(num_samples_per_dataset: int = 4) -> dict[str, Any]:
    """Run a bounded real-data smoke test on DDR and IDRiD samples without training a model."""
    manifest_dir = resolve_project_path("data/processed/manifests")
    ddr_manifest = load_manifest_json(manifest_dir / "ddr_manifest.json")
    idrid_manifest = load_manifest_json(manifest_dir / "idrid_manifest.json")

    ddr_train = [
        r for r in ddr_manifest["records"] if r["split"] == "train"
    ][:num_samples_per_dataset]
    ddr_val = [
        r for r in ddr_manifest["records"] if r["split"] == "valid"
    ][:2]
    idrid_train = [
        r for r in idrid_manifest["records"] if r["split"] == "train"
    ][:num_samples_per_dataset]
    # Also include IDRiD_81 from test split (the 4-channel RGBA EX mask case) in eval check
    idrid_test = [
        r for r in idrid_manifest["records"] if r["split"] == "test"
    ][-2:]

    smoke_results: dict[str, Any] = {}

    for label, records, split_name in (
        ("DDR_train_subset", ddr_train, "train"),
        ("DDR_valid_subset", ddr_val, "valid"),
        ("IDRiD_train_subset", idrid_train, "train"),
        ("IDRiD_test_subset", idrid_test, "test"),
    ):
        set_seed(42)
        ds = RetinalLesionSegmentationDataset(
            records,
            split=split_name,
            image_size=(256, 256),
            return_metadata=True,
        )
        img0, mask0, meta0 = ds[0]

        set_seed(42)
        ds_repeat = RetinalLesionSegmentationDataset(
            records,
            split=split_name,
            image_size=(256, 256),
            return_metadata=True,
        )
        img0_rep, mask0_rep, _ = ds_repeat[0]

        reproducible = bool(
            torch.allclose(img0, img0_rep) and torch.equal(mask0, mask0_rep)
        )
        unique_mask_vals = sorted(float(v) for v in torch.unique(mask0).tolist())

        # DataLoader iteration check (batch_size=2, num_workers=0 for fast smoke test)
        ds_plain = RetinalLesionSegmentationDataset(
            records,
            split=split_name,
            image_size=(256, 256),
            return_metadata=False,
        )
        loader = build_segmentation_dataloader(
            ds_plain,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            seed=42,
        )
        batch_images, batch_masks = next(iter(loader))

        smoke_results[label] = {
            "num_records": len(ds),
            "split": split_name,
            "first_sample_id": meta0["sample_id"],
            "channel_order": meta0["channel_order"],
            "image_tensor_shape": list(img0.shape),
            "image_tensor_dtype": str(img0.dtype),
            "mask_tensor_shape": list(mask0.shape),
            "mask_tensor_dtype": str(mask0.dtype),
            "mask_unique_values": unique_mask_vals,
            "spatial_aligned": (img0.shape[1:] == mask0.shape[1:]),
            "reproducible_under_seed42": reproducible,
            "batch_image_shape": list(batch_images.shape),
            "batch_mask_shape": list(batch_masks.shape),
        }

        assert img0.shape == (3, 256, 256)
        assert mask0.shape == (4, 256, 256)
        assert img0.dtype == torch.float32
        assert mask0.dtype == torch.float32
        assert set(unique_mask_vals).issubset({0.0, 1.0})
        assert tuple(meta0["channel_order"]) == CANONICAL_LESION_CHANNELS
        assert reproducible is True

    return smoke_results


def format_partition_report(
    partitions: dict[str, dict[str, Any]],
    smoke_results: dict[str, Any] | None = None,
) -> str:
    """Format a human-readable summary of generated client partitions and smoke tests."""
    lines: list[str] = [
        "=" * 78,
        "FedMorphis Phase 1, Task 4: Federated Client Partitions & Smoke-Test Report",
        "=" * 78,
    ]

    if smoke_results is not None:
        lines.append("\nReal-Data Dataset & DataLoader Smoke Test")
        lines.append("-" * 78)
        for name, res in smoke_results.items():
            lines.append(
                f"  {name:<20}: samples={res['num_records']}, split={res['split']}, "
                f"img={res['image_tensor_shape']} ({res['image_tensor_dtype']}), "
                f"mask={res['mask_tensor_shape']} ({res['mask_tensor_dtype']}), "
                f"vals={res['mask_unique_values']}, batch={res['batch_image_shape']}, "
                f"reproducible={res['reproducible_under_seed42']}"
            )

    for key, part in partitions.items():
        ver = part["verification"]
        lines.append(f"\nPartition Set: {key} (strategy={part['strategy']}, seed={part['seed']})")
        lines.append("-" * 78)
        lines.append(
            f"  Source Datasets          : {part['source_datasets']} | "
            f"Total Train={part['total_train_samples']} | "
            f"Held-out Valid={part['held_out_validation_samples']} | "
            f"Held-out Test={part['held_out_test_samples']}"
        )
        lines.append(
            f"  Verification             : assigned_once={ver['all_train_samples_assigned_once']}, "
            f"disjoint={ver['clients_disjoint']}, lost={ver['lost_train_sample_count']}, "
            f"val_leak={ver['validation_leakage_count']}, test_leak={ver['test_leakage_count']}"
        )
        for cid, csum in part["client_summaries"].items():
            lines.append(
                f"    {cid:<10}: total={csum['num_samples']:<4} | "
                f"datasets={csum['dataset_counts']} | "
                f"positive_lesions={csum['positive_lesion_counts']}"
            )

    lines.append("\n" + "=" * 78)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for building partitions and running optional real-data smoke tests."""
    parser = argparse.ArgumentParser(
        description="Build reproducible federated client partitions and run optional real-data smoke tests."
    )
    parser.add_argument(
        "--federated-config",
        type=str,
        default="federated.yaml",
        help="Federated configuration YAML file name or path.",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="dataset.yaml",
        help="Dataset configuration YAML file name or path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory for partition JSON files (default: data/partitions).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a bounded real-data Dataset and DataLoader smoke test on DDR and IDRiD.",
    )
    args = parser.parse_args(argv)

    smoke_res = run_real_data_smoke_test() if args.smoke_test else None
    partitions = build_configured_partitions(
        federated_config_path=args.federated_config,
        dataset_config_path=args.dataset_config,
        output_dir=args.output_dir,
    )
    print(format_partition_report(partitions, smoke_results=smoke_res))
    return 0


__all__ = [
    "format_partition_report",
    "main",
    "run_real_data_smoke_test",
]


if __name__ == "__main__":
    sys.exit(main())

