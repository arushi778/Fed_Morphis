"""Segmentation and cross-dataset evaluation metrics and reporting utilities."""

from src.models.metrics import (
    METRIC_NAMES,
    PROTOCOL_NAME,
    AUPRTracker,
    ChannelPRResult,
    SegmentationMetricTracker,
    binarize_predictions,
    compute_channel_pr_curve,
    compute_confusion_counts,
    compute_multilabel_aupr,
    compute_segmentation_metrics,
    summarize_channel_counts,
)

__all__ = [
    "METRIC_NAMES",
    "PROTOCOL_NAME",
    "AUPRTracker",
    "ChannelPRResult",
    "SegmentationMetricTracker",
    "binarize_predictions",
    "compute_channel_pr_curve",
    "compute_confusion_counts",
    "compute_multilabel_aupr",
    "compute_segmentation_metrics",
    "summarize_channel_counts",
]
