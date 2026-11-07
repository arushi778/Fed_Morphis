"""CLI script for held-out test split evaluation of supervised segmentation checkpoints.

Loads a saved segmentation checkpoint (default: ``checkpoints/segmentation/best_checkpoint.pt``),
reconstructs the model from the saved configuration and architecture, evaluates strictly
on the held-out ``test`` split of the target manifest (default: ``ddr_manifest.json``),
and writes a structured JSON report (default: ``logs/supervised_baseline_ddr_test_report.json``).
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

import torch

from src.engine.evaluator import evaluate_checkpoint_on_manifest
from src.utils.config import (
    load_yaml_config,
    validate_dataset_config,
    validate_model_config,
    validate_project_config,
)
from src.utils.device import get_device_info, resolve_device
from src.utils.logger import get_logger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for held-out test evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a saved ResNet-50 U-Net checkpoint on the held-out test split."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/segmentation/best_checkpoint.pt",
        help="Path to the saved model checkpoint (.pt).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/processed/manifests/ddr_manifest.json",
        help="Path to the dataset manifest JSON.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate (default: 'test').",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="logs/supervised_baseline_ddr_test_report.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--project-config",
        type=str,
        default="configs/project.yaml",
        help="Fallback project configuration YAML.",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="configs/dataset.yaml",
        help="Fallback dataset configuration YAML.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model.yaml",
        help="Fallback model configuration YAML.",
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
        help="Optional override for square evaluation resolution (H = W).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional override for binary probability threshold (default: 0.5).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device ('cuda', 'cpu', or 'auto').",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run held-out test split evaluation and save the structured JSON report."""
    args = parse_args(argv)
    logger = get_logger("fedmorphis.evaluate_baseline")

    base_cfg = load_yaml_config(PROJECT_ROOT / args.project_config, validate=False)
    validate_project_config(base_cfg)
    data_cfg = load_yaml_config(PROJECT_ROOT / args.dataset_config, validate=False)
    validate_dataset_config(data_cfg)
    model_cfg = load_yaml_config(PROJECT_ROOT / args.model_config, validate=False)
    validate_model_config(model_cfg)

    requested_device = args.device or str(base_cfg.get("device", "auto"))
    device_info = get_device_info()
    device = torch.device(str(resolve_device(requested_device)))

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path

    report_path = Path(args.report_path)
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path

    logger.info(
        "Starting held-out '%s' split evaluation on device=%s (cuda_available=%s, checkpoint=%s, manifest=%s)",
        args.split,
        device,
        device_info.get("cuda_available"),
        args.checkpoint,
        args.manifest,
    )

    report = evaluate_checkpoint_on_manifest(
        checkpoint_path=checkpoint_path,
        manifest_or_records=manifest_path,
        split=args.split,
        dataset_config=data_cfg,
        model_config=model_cfg,
        project_config=base_cfg,
        batch_size=args.batch_size,
        image_size=args.image_size,
        metric_threshold=args.threshold,
        device=device,
        report_path=report_path,
        manifest_path_display=str(args.manifest).replace("\\", "/"),
        checkpoint_path_display=str(args.checkpoint).replace("\\", "/"),
    )

    test_metrics = report["test_metrics"]
    per_class = test_metrics["per_class"]
    logger.info(
        "Completed '%s' evaluation (%d samples) | loss=%.4f | dice_macro=%.4f | iou_macro=%.4f "
        "(MA=%.4f, HE=%.4f, EX=%.4f, SE=%.4f)",
        report["dataset"]["split"],
        report["dataset"]["sample_count"],
        test_metrics["loss"] if test_metrics["loss"] is not None else float("nan"),
        test_metrics["dice_macro"],
        test_metrics["iou_macro"],
        per_class["MA"]["dice"],
        per_class["HE"]["dice"],
        per_class["EX"]["dice"],
        per_class["SE"]["dice"],
    )

    summary_stdout: dict[str, Any] = {
        "status": report["status"],
        "checkpoint": report["checkpoint"],
        "dataset": report["dataset"],
        "evaluation_config": {
            "seed": report["evaluation_config"]["seed"],
            "device": report["evaluation_config"]["device"],
            "amp_enabled": report["evaluation_config"]["amp_enabled"],
            "image_size": report["evaluation_config"]["image_size"],
            "batch_size": report["evaluation_config"]["batch_size"],
            "metric_threshold": report["evaluation_config"]["metric_threshold"],
        },
        "test_loss": test_metrics["loss"],
        "test_dice_macro": test_metrics["dice_macro"],
        "test_dice_macro_present_only": test_metrics["dice_macro_present_only"],
        "test_iou_macro": test_metrics["iou_macro"],
        "test_iou_macro_present_only": test_metrics["iou_macro_present_only"],
        "per_class": per_class,
        "evaluation_duration_sec": report["evaluation_duration_sec"],
        "gpu_memory": report["gpu_memory"],
        "report_path": str(args.report_path).replace("\\", "/"),
    }
    print(json.dumps(summary_stdout, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

