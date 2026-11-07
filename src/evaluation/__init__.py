"""Segmentation and cross-dataset evaluation metrics and reporting utilities."""

from src.models.metrics import (
    METRIC_NAMES,
    SegmentationMetricTracker,
    binarize_predictions,
    compute_confusion_counts,
    compute_segmentation_metrics,
    summarize_channel_counts,
)

__all__ = [
    "METRIC_NAMES",
    "SegmentationMetricTracker",
    "binarize_predictions",
    "compute_confusion_counts",
    "compute_segmentation_metrics",
    "summarize_channel_counts",
]
