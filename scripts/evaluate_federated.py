"""CLI entry point for evaluating saved global federated checkpoints on DDR and IDRiD splits.

Supports:
- Loading a saved global federated checkpoint (default:
  ``checkpoints/federated/best_checkpoint.pt``) and reconstructing the
  :class:`src.models.unet.ResNet50UNet` model via
  :func:`src.engine.evaluator.reconstruct_model_from_checkpoint`.
- Evaluating on ``test`` (default) or ``valid`` splits.
- Preserving the official DDR (225 test / 149 valid) and IDRiD (27 held-out test /
  10 deterministically carved valid) splits.
- Evaluating and reporting DDR and IDRiD metrics **separately** under
  ``report["datasets"]["DDR"]`` and ``report["datasets"]["IDRiD"]`` without pooling
  or combining samples across datasets.
- Reporting loss, macro Dice, macro IoU, and per-lesion (``MA``, ``HE``, ``EX``, ``SE``)
  Dice/IoU along with checkpoint round and configuration metadata.
- Saving structured JSON evaluation reports atomically.

Example commands:
    python scripts/evaluate_federated.py --checkpoint checkpoints/federated/best_checkpoint.pt --split test
    python scripts/evaluate_federated.py --checkpoint checkpoints/federated/best_checkpoint.pt --split valid
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import sys
from typing import Any
import uuid

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
from src.data.partitioning import load_partition_manifest
from src.data.transforms import build_segmentation_transforms
from src.engine.evaluator import (
    evaluate_checkpoint_on_manifest,
    evaluate_segmentation_dataloader,
    reconstruct_model_from_checkpoint,
)
from src.federated.data import (
    FederatedDataError,
    resolve_client_train_val_records,
    verify_client_split_integrity,
)
from src.models.losses import build_loss_from_config
from src.utils.config import (
    ConfigValidationError,
    load_yaml_config,
    validate_dataset_config,
    validate_federated_config,
    validate_model_config,
    validate_project_config,
)
from src.utils.device import get_device_info, resolve_device
from src.utils.logger import get_logger
from src.utils.paths import format_project_path, resolve_project_path
from src.utils.seed import set_seed

SUPPORTED_EVAL_SPLITS: dict[str, str] = {
    "test": "test",
    "valid": "valid",
    "val": "valid",
    "validation": "valid",
}

CANONICAL_DATASET_NAMES: dict[str, str] = {
    "ddr": "DDR",
    "idrid": "IDRiD",
}


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write ``payload`` as formatted JSON atomically using a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=False)
            fh.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for global federated checkpoint evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a saved global federated ResNet-50 U-Net checkpoint separately on DDR and IDRiD."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/federated/best_checkpoint.pt",
        help="Path to the saved global federated model checkpoint (.pt).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Evaluation split ('test' or 'valid'; default: 'test').",
    )
    parser.add_argument(
        "--datasets",
        "--dataset",
        dest="datasets",
        type=str,
        nargs="+",
        default=["ddr", "idrid"],
        help="Datasets to evaluate separately ('ddr', 'idrid', or 'both'; default: ddr idrid).",
    )
    parser.add_argument(
        "--config",
        "--federated-config",
        dest="config",
        type=str,
        default="configs/federated.yaml",
        help="Path to federated configuration YAML (default: 'configs/federated.yaml').",
    )
    parser.add_argument(
        "--project-config",
        type=str,
        default="configs/project.yaml",
        help="Fallback project configuration YAML (default: 'configs/project.yaml').",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="configs/dataset.yaml",
        help="Fallback dataset configuration YAML (default: 'configs/dataset.yaml').",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model.yaml",
        help="Fallback model configuration YAML (default: 'configs/model.yaml').",
    )
    parser.add_argument(
        "--ddr-manifest",
        type=str,
        default="data/processed/manifests/ddr_manifest.json",
        help="Path to DDR dataset manifest JSON.",
    )
    parser.add_argument(
        "--idrid-manifest",
        type=str,
        default="data/processed/manifests/idrid_manifest.json",
        help="Path to IDRiD dataset manifest JSON.",
    )
    parser.add_argument(
        "--manifest",
        "--source-manifest",
        dest="manifest",
        type=str,
        default=None,
        help="Optional combined or single-dataset manifest JSON path overriding default per-dataset manifests.",
    )
    parser.add_argument(
        "--partition-manifest",
        type=str,
        default=None,
        help="Optional partition manifest JSON path (used for 'valid' split resolution and test-leakage verification).",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Output JSON report path (defaults to logs/federated_fedavg_<split>_evaluation_report.json).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for evaluation batch size.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Optional override for square evaluation resolution (H = W, divisible by 32).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional override for binary probability threshold in (0, 1) (default: 0.5).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device ('cuda', 'cpu', or 'auto').",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _resolve_requested_datasets(raw_datasets: Sequence[str] | str) -> list[str]:
    """Normalize and validate requested dataset names into canonical ``['DDR', 'IDRiD']`` order."""
    tokens: list[str] = []
    if isinstance(raw_datasets, str):
        raw_list = [raw_datasets]
    else:
        raw_list = list(raw_datasets)

    for item in raw_list:
        for part in str(item).split(","):
            cleaned = part.strip().lower()
            if not cleaned:
                continue
            if cleaned in {"both", "all"}:
                tokens.extend(["ddr", "idrid"])
            elif cleaned in CANONICAL_DATASET_NAMES:
                tokens.append(cleaned)
            else:
                raise ConfigValidationError(
                    f"Unsupported dataset {part!r}; expected 'ddr', 'idrid', or 'both'."
                )

    if not tokens:
        raise ConfigValidationError("At least one dataset ('ddr' or 'idrid') must be specified.")

    ordered: list[str] = []
    for key in ("ddr", "idrid"):
        if key in tokens and CANONICAL_DATASET_NAMES[key] not in ordered:
            ordered.append(CANONICAL_DATASET_NAMES[key])
    return ordered


def _resolve_eval_split(raw_split: str) -> str:
    """Validate and normalize the requested evaluation split ('test' or 'valid')."""
    cleaned = str(raw_split).strip().lower()
    if cleaned not in SUPPORTED_EVAL_SPLITS:
        raise ConfigValidationError(
            f"Unsupported evaluation split {raw_split!r}; only 'test' and 'valid' are permitted "
            f"(training split evaluation is disallowed)."
        )
    return SUPPORTED_EVAL_SPLITS[cleaned]


def _extract_records_list(manifest_or_path: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Load a manifest JSON or normalize an in-memory record list."""
    if isinstance(manifest_or_path, (str, Path)):
        path = resolve_project_path(manifest_or_path)
        if not path.is_file():
            raise FileNotFoundError(f"Manifest file not found: {path}")
        loaded = load_manifest_json(path)
        return [dict(r) for r in loaded.get("records", [])]
    if isinstance(manifest_or_path, Mapping):
        return [dict(r) for r in manifest_or_path.get("records", [])]
    if isinstance(manifest_or_path, Sequence):
        return [dict(r) for r in manifest_or_path]
    raise TypeError(f"Unsupported manifest type: {type(manifest_or_path).__name__}")


def _filter_records_by_dataset(
    records: Sequence[Mapping[str, Any]],
    dataset_name: str,
) -> list[dict[str, Any]]:
    """Filter manifest records matching ``dataset_name`` case-insensitively."""
    target = dataset_name.strip().upper()
    out: list[dict[str, Any]] = []
    for r in records:
        ds = str(r.get("dataset_name", r.get("dataset", ""))).strip().upper()
        if ds == target:
            out.append(dict(r))
    return out


def _verify_partition_manifest_disjoint_from_test(
    partition_manifest_path: str | Path | Mapping[str, Any] | None,
    test_records: Sequence[Mapping[str, Any]],
    dataset_name: str,
) -> None:
    """Verify zero overlap between held-out test records and any client partition manifest entry."""
    if partition_manifest_path is None:
        return
    try:
        part_data = (
            dict(partition_manifest_path)
            if isinstance(partition_manifest_path, Mapping)
            else load_partition_manifest(resolve_project_path(partition_manifest_path))
        )
    except FileNotFoundError:
        return

    part_sample_keys: set[tuple[str, str]] = set()
    for c_entry in part_data.get("clients", {}).values():
        if isinstance(c_entry, Sequence):
            for r in c_entry:
                if isinstance(r, Mapping):
                    part_sample_keys.add(
                        (
                            str(r.get("dataset_name", "")).upper(),
                            str(r.get("sample_id", "")),
                        )
                    )

    if not part_sample_keys:
        return

    test_keys = {
        (
            str(r.get("dataset_name", dataset_name)).upper(),
            str(r.get("sample_id", "")),
        )
        for r in test_records
        if str(r.get("split", "")).lower() == "test"
    }
    leaked = sorted(test_keys & part_sample_keys)
    if leaked:
        raise FederatedDataError(
            f"Test split leakage detected for {dataset_name}: {len(leaked)} test sample(s) "
            f"appear in the client partition manifest: {leaked[:5]}"
        )



def evaluate_federated_checkpoint(
    checkpoint_path: str | Path,
    *,
    split: str = "test",
    datasets: Sequence[str] = ("ddr", "idrid"),
    federated_config: Mapping[str, Any] | str | Path = "configs/federated.yaml",
    dataset_config: Mapping[str, Any] | str | Path = "configs/dataset.yaml",
    model_config: Mapping[str, Any] | str | Path = "configs/model.yaml",
    project_config: Mapping[str, Any] | str | Path = "configs/project.yaml",
    ddr_manifest: str | Path = "data/processed/manifests/ddr_manifest.json",
    idrid_manifest: str | Path = "data/processed/manifests/idrid_manifest.json",
    manifest: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    partition_manifest: str | Path | Mapping[str, Any] | None = None,
    batch_size: int | None = None,
    image_size: int | None = None,
    metric_threshold: float | None = None,
    device: torch.device | str | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a global federated checkpoint separately on requested datasets and split.

    Enforces:
    - Separate evaluation of DDR and IDRiD (never combines test metrics across datasets).
    - Strict split hygiene (test samples never overlap client training or carved validation).
    - Full reconstruction of :class:`ResNet50UNet` via :func:`reconstruct_model_from_checkpoint`.
    """
    target_split = _resolve_eval_split(split)
    requested_datasets = _resolve_requested_datasets(datasets)

    fed_cfg = (
        load_yaml_config(resolve_project_path(federated_config), validate=True)
        if isinstance(federated_config, (str, Path))
        else dict(federated_config)
    )
    validate_federated_config(fed_cfg)

    ds_cfg = (
        load_yaml_config(resolve_project_path(dataset_config), validate=True)
        if isinstance(dataset_config, (str, Path))
        else dict(dataset_config)
    )
    validate_dataset_config(ds_cfg)

    mdl_cfg = (
        load_yaml_config(resolve_project_path(model_config), validate=True)
        if isinstance(model_config, (str, Path))
        else dict(model_config)
    )
    validate_model_config(mdl_cfg)

    proj_cfg = (
        load_yaml_config(resolve_project_path(project_config), validate=True)
        if isinstance(project_config, (str, Path))
        else dict(project_config)
    )
    validate_project_config(proj_cfg)

    if batch_size is not None and int(batch_size) < 1:
        raise ConfigValidationError(f"batch_size must be >= 1, got {batch_size}.")
    if image_size is not None and (int(image_size) < 32 or int(image_size) % 32 != 0):
        raise ConfigValidationError(
            f"image_size must be >= 32 and divisible by 32, got {image_size}."
        )
    if metric_threshold is not None and not (0.0 < float(metric_threshold) < 1.0):
        raise ConfigValidationError(
            f"threshold must be strictly between 0 and 1, got {metric_threshold}."
        )

    if device is None:
        target_device = resolve_device(str(proj_cfg.get("device", "auto")))
    elif isinstance(device, str):
        target_device = resolve_device(device)
    else:
        target_device = torch.device(device)

    ckpt_path = resolve_project_path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    # Reconstruct ResNet50UNet and load checkpoint payload
    model, checkpoint, saved_configs = reconstruct_model_from_checkpoint(
        checkpoint_path=ckpt_path,
        fallback_model_config=mdl_cfg,
        device=target_device,
        strict=True,
    )

    seed = int(
        checkpoint.get(
            "seed",
            fed_cfg.get("seed", proj_cfg.get("random_seed", 42)),
        )
    )
    deterministic = bool(proj_cfg.get("deterministic", True))
    set_seed(seed, deterministic=deterministic)

    ckpt_round = int(checkpoint.get("round", checkpoint.get("epoch", 0)))
    ckpt_best_round = int(
        checkpoint.get("best_round", checkpoint.get("best_epoch", ckpt_round))
    )
    ckpt_best_val = (
        float(checkpoint["best_metric_value"])
        if checkpoint.get("best_metric_value") is not None
        and checkpoint.get("best_metric_value") not in {float("-inf"), float("inf")}
        else None
    )

    # Resolve per-dataset evaluation records
    per_dataset_reports: dict[str, dict[str, Any]] = {}

    if target_split == "test":
        # Verify against partition manifest (if available) that no test sample is in any client partition
        part_sample_keys: set[tuple[str, str]] = set()
        raw_part_path = (
            partition_manifest
            if partition_manifest is not None
            else fed_cfg.get("partitioning", {}).get("manifest_path")
        )
        if raw_part_path is not None:
            try:
                part_data = (
                    dict(raw_part_path)
                    if isinstance(raw_part_path, Mapping)
                    else load_partition_manifest(resolve_project_path(raw_part_path))
                )
                for c_entry in part_data.get("clients", {}).values():
                    if isinstance(c_entry, Sequence):
                        for r in c_entry:
                            if isinstance(r, Mapping):
                                part_sample_keys.add(
                                    (
                                        str(r.get("dataset_name", "")).upper(),
                                        str(r.get("sample_id", "")),
                                    )
                                )
            except FileNotFoundError:
                pass

        for ds_name in requested_datasets:
            if manifest is not None:
                all_recs = _extract_records_list(manifest)
                ds_records = _filter_records_by_dataset(all_recs, ds_name)
                manifest_source_for_eval: Any = ds_records
                manifest_display = (
                    format_project_path(manifest)
                    if isinstance(manifest, (str, Path))
                    else f"in_memory_{ds_name.lower()}_records"
                )
            else:
                ds_manifest_path = (
                    resolve_project_path(ddr_manifest)
                    if ds_name == "DDR"
                    else resolve_project_path(idrid_manifest)
                )
                manifest_source_for_eval = ds_manifest_path
                manifest_display = format_project_path(ds_manifest_path)

            # Verify zero overlap between evaluated test sample_ids and partition training/carved-val sample_ids
            recs_checked = _extract_records_list(manifest_source_for_eval)
            _verify_partition_manifest_disjoint_from_test(
                partition_manifest_path=raw_part_path,
                test_records=recs_checked,
                dataset_name=ds_name,
            )

            ds_eval_report = evaluate_checkpoint_on_manifest(
                checkpoint_path=ckpt_path,
                manifest_or_records=manifest_source_for_eval,
                split="test",
                dataset_config=ds_cfg,
                model_config=mdl_cfg,
                project_config=proj_cfg,
                batch_size=batch_size,
                image_size=image_size,
                metric_threshold=metric_threshold,
                device=target_device,
                report_path=None,
                manifest_path_display=manifest_display,
                checkpoint_path_display=format_project_path(ckpt_path),
            )

            per_dataset_reports[ds_name] = {
                "dataset_name": ds_name,
                "split": "test",
                "manifest_path": manifest_display,
                "sample_count": int(ds_eval_report["dataset"]["sample_count"]),
                "num_batches": int(ds_eval_report["dataset"]["num_batches"]),
                "channel_order": list(CANONICAL_LESION_CHANNELS),
                "metrics": ds_eval_report["test_metrics"],
                "evaluation_duration_sec": ds_eval_report["evaluation_duration_sec"],
            }

    else:
        # target_split == "valid": evaluate each dataset's federated client validation split
        for ds_name in requested_datasets:
            _, val_records, split_info = resolve_client_train_val_records(
                client_id=ds_name,
                partition_manifest=partition_manifest,
                source_manifest=manifest,
                federated_config=fed_cfg,
                dataset_config=ds_cfg,
            )
            raw_part_disp = (
                partition_manifest
                if isinstance(partition_manifest, (str, Path))
                else fed_cfg.get("partitioning", {}).get(
                    "manifest_path",
                    "data/partitions/combined_cross_dataset_2clients_seed42.json",
                )
            )
            manifest_display = format_project_path(raw_part_disp)

            ds_eval_report = evaluate_checkpoint_on_manifest(
                checkpoint_path=ckpt_path,
                manifest_or_records=val_records,
                split="valid",
                dataset_config=ds_cfg,
                model_config=mdl_cfg,
                project_config=proj_cfg,
                batch_size=batch_size,
                image_size=image_size,
                metric_threshold=metric_threshold,
                device=target_device,
                report_path=None,
                manifest_path_display=manifest_display,
                checkpoint_path_display=format_project_path(ckpt_path),
            )

            per_dataset_reports[ds_name] = {
                "dataset_name": ds_name,
                "client_id": split_info["client_id"],
                "split": "valid",
                "val_split_mode": split_info["val_split_mode"],
                "val_source": split_info["val_source"],
                "manifest_path": manifest_display,
                "sample_count": int(ds_eval_report["dataset"]["sample_count"]),
                "num_batches": int(ds_eval_report["dataset"]["num_batches"]),
                "channel_order": list(CANONICAL_LESION_CHANNELS),
                "metrics": ds_eval_report["test_metrics"],
                "evaluation_duration_sec": ds_eval_report["evaluation_duration_sec"],
            }

    resolved_report_path: Path | None = None
    if report_path is not None:
        resolved_report_path = resolve_project_path(report_path)
    else:
        default_name = (
            "logs/federated_fedavg_test_evaluation_report.json"
            if target_split == "test"
            else "logs/federated_fedavg_val_evaluation_report.json"
        )
        resolved_report_path = resolve_project_path(default_name)

    full_report: dict[str, Any] = {
        "status": "completed",
        "framework": "flower",
        "strategy": str(fed_cfg.get("strategy", "FedAvg")),
        "split": target_split,
        "evaluated_datasets": list(requested_datasets),
        "separate_dataset_reporting": True,
        "checkpoint": {
            "path": format_project_path(ckpt_path),
            "round": ckpt_round,
            "epoch": ckpt_round,
            "best_round": ckpt_best_round,
            "best_epoch": ckpt_best_round,
            "monitor_metric": checkpoint.get("monitor_metric", "val_dice_macro"),
            "monitor_mode": checkpoint.get("monitor_mode", "max"),
            "best_validation_metric_value": ckpt_best_val,
            "seed": seed,
            "saved_config": saved_configs,
        },
        "model": model.get_model_summary(),
        "datasets": per_dataset_reports,
        "config": {
            "federated": dict(fed_cfg),
            "dataset": dict(ds_cfg),
            "model": dict(mdl_cfg),
            "project": dict(proj_cfg),
            "device": str(target_device),
        },
    }

    if resolved_report_path is not None:
        _atomic_write_json(resolved_report_path, full_report)
        full_report["report_path"] = format_project_path(resolved_report_path)

    return full_report


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for global federated checkpoint evaluation."""
    args = parse_args(argv)
    logger = get_logger("fedmorphis.evaluate_federated")
    device_info = get_device_info()

    report = evaluate_federated_checkpoint(
        checkpoint_path=args.checkpoint,
        split=args.split,
        datasets=args.datasets,
        federated_config=args.config,
        dataset_config=args.dataset_config,
        model_config=args.model_config,
        project_config=args.project_config,
        ddr_manifest=args.ddr_manifest,
        idrid_manifest=args.idrid_manifest,
        manifest=args.manifest,
        partition_manifest=args.partition_manifest,
        batch_size=args.batch_size,
        image_size=args.image_size,
        metric_threshold=args.threshold,
        device=args.device,
        report_path=args.report_path,
    )

    for ds_name, ds_block in report["datasets"].items():
        m = ds_block["metrics"]
        pc = m["per_class"]
        logger.info(
            "Dataset=%s (%s, n=%d, cuda=%s) | loss=%.4f | dice_macro=%.4f | iou_macro=%.4f "
            "(MA=%.4f, HE=%.4f, EX=%.4f, SE=%.4f)",
            ds_name,
            ds_block["split"],
            ds_block["sample_count"],
            device_info.get("cuda_available"),
            m["loss"] if m["loss"] is not None else float("nan"),
            m["dice_macro"],
            m["iou_macro"],
            pc["MA"]["dice"],
            pc["HE"]["dice"],
            pc["EX"]["dice"],
            pc["SE"]["dice"],
        )

    summary_stdout = {
        "status": report["status"],
        "split": report["split"],
        "evaluated_datasets": report["evaluated_datasets"],
        "checkpoint": {
            "path": report["checkpoint"]["path"],
            "round": report["checkpoint"]["round"],
            "best_round": report["checkpoint"]["best_round"],
            "best_validation_metric_value": report["checkpoint"]["best_validation_metric_value"],
        },
        "datasets": {
            ds_name: {
                "split": ds_block["split"],
                "sample_count": ds_block["sample_count"],
                "loss": ds_block["metrics"]["loss"],
                "dice_macro": ds_block["metrics"]["dice_macro"],
                "iou_macro": ds_block["metrics"]["iou_macro"],
                "per_class_dice": {
                    ch: ds_block["metrics"]["per_class"][ch]["dice"]
                    for ch in CANONICAL_LESION_CHANNELS
                },
                "per_class_iou": {
                    ch: ds_block["metrics"]["per_class"][ch]["iou"]
                    for ch in CANONICAL_LESION_CHANNELS
                },
            }
            for ds_name, ds_block in report["datasets"].items()
        },
        "report_path": report.get("report_path"),
    }
    print(json.dumps(summary_stdout, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
