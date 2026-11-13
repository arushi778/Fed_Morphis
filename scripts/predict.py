"""CLI inference tool for FedMorphis retinal lesion segmentation.

Usage:
    python scripts/predict.py --image data/raw/DDR/DDR-dataset/lesion_segmentation/test/image/007-0010-000.jpg --output-dir outputs/demo
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.inference import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_FROZEN_THRESHOLDS,
    LesionSegmentationPipeline,
)
from src.utils.logger import get_logger
from src.utils.paths import format_project_path, resolve_project_path

logger = get_logger("fedmorphis.predict_cli")


def main() -> None:
    parser = argparse.ArgumentParser(description="FedMorphis: Run retinal lesion segmentation inference.")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to retinal fundus image (.jpg, .png, .tif)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"Path to model checkpoint (default: {DEFAULT_CHECKPOINT_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/predictions",
        help="Directory to save output masks, overlay, and summary JSON",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference ('auto', 'cuda', 'cpu')",
    )
    args = parser.parse_args()

    img_path = resolve_project_path(args.image)
    if not img_path.is_file():
        logger.error("Image file not found: %s", img_path)
        sys.exit(1)

    pipeline = LesionSegmentationPipeline(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    sample_name = img_path.stem
    logger.info("Running inference on %s...", img_path.name)
    result = pipeline.predict(img_path, sample_name=sample_name)

    artifacts = pipeline.save_prediction_artifacts(
        result=result,
        output_dir=args.output_dir,
        base_name=sample_name,
    )

    print("\n" + "=" * 80)
    print(f"FEDMORPHIS INFERENCE COMPLETED: {sample_name}")
    print("=" * 80)
    print(f"Original Resolution: {result.original_dimensions[1]} x {result.original_dimensions[0]} pixels")
    print(f"Inference Latency:   {result.latency_ms:.2f} ms")
    print(f"Device:              {result.metadata['device']}")
    print(f"Model Checkpoint:    {result.metadata['checkpoint_name']}")
    print("-" * 80)
    print("LESION DETECTION SUMMARY:")
    for ch_code, cp in result.channels.items():
        status = "DETECTED" if cp.detected else "NOT DETECTED"
        print(f"  [{status:<12}] {ch_code} ({cp.channel_name:<16}): "
              f"Threshold={cp.threshold:.2f} | Positive Pixels={cp.positive_pixel_count:<6} | Area={cp.area_percentage:.4f}%")
    print("-" * 80)
    print("SAVED ARTIFACTS:")
    for k, p in artifacts.items():
        print(f"  {k:<15}: {p}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

