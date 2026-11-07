"""Multilabel retinal lesion segmentation evaluation metrics.

Computes per-class (``MA``, ``HE``, ``EX``, ``SE``) and macro-averaged
metrics for:
- Dice score (F1)
- Intersection over Union (IoU / Jaccard)
- Precision
- Recall
- Sensitivity (True Positive Rate, identical to Recall)
- Specificity (True Negative Rate)

Empty-Mask Policy
-----------------
Retinal lesion datasets (DDR and IDRiD) frequently have samples or splits where
one or more lesion classes are absent (for example, Soft Exudates ``SE``). To
ensure deterministic, transparent evaluation without NaN values or silently
hidden undefined metrics, we enforce the following explicit convention per class
channel:

1. **Both ground truth and prediction empty** (``TP = 0, FP = 0, FN = 0``):
   - ``Dice = 1.0``, ``IoU = 1.0``, ``Precision = 1.0``, ``Recall = 1.0``,
     ``Sensitivity = 1.0``, ``Specificity = 1.0``.
   - Recorded as ``empty_handling = "both_empty_exact_match"`` with
     ``target_present = False`` and ``prediction_present = False``.
2. **Ground truth empty, prediction non-empty** (``TP = 0, FP > 0, FN = 0``):
   - False positive prediction on a lesion-free channel.
   - ``Dice = 0.0``, ``IoU = 0.0``, ``Precision = 0.0``, ``Recall = 0.0``,
     ``Sensitivity = 0.0``, ``Specificity = TN / (TN + FP)``.
   - Recorded as ``empty_handling = "target_empty_false_positive"`` with
     ``target_present = False`` and ``prediction_present = True``.
3. **Ground truth non-empty, prediction empty** (``TP = 0, FP = 0, FN > 0``):
   - Missed positive lesion channel.
   - ``Dice = 0.0``, ``IoU = 0.0``, ``Precision = 0.0``, ``Recall = 0.0``,
     ``Sensitivity = 0.0``, ``Specificity = 1.0`` (if ``TN > 0`` else ``0.0``).
   - Recorded as ``empty_handling = "prediction_empty_false_negative"`` with
     ``target_present = True`` and ``prediction_present = False``.
4. **Both ground truth and prediction non-empty**:
   - Standard confusion-matrix definitions with exact integer/float counts.
   - Recorded as ``empty_handling = "standard_non_empty"``.

In addition to ``macro`` (mean across all four canonical classes), results
always report ``macro_present_only`` (mean across classes where
``target_present is True``) alongside ``present_classes`` and ``absent_classes``
lists so absent lesion classes in any sample or validation subset are explicitly
visible.
"""

from __future__ import annotations

from typing import Any

import torch

from src.data.masks import CANONICAL_LESION_CHANNELS

CANONICAL_LESION_ORDER: tuple[str, ...] = tuple(CANONICAL_LESION_CHANNELS)
METRIC_NAMES: tuple[str, ...] = (
    "dice",
    "iou",
    "precision",
    "recall",
    "sensitivity",
    "specificity",
)


def _validate_metric_inputs(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> None:
    """Validate prediction and target tensors for metric computation."""
    if not isinstance(predictions, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError("predictions and targets must be torch.Tensor instances.")
    if predictions.ndim != 4 or targets.ndim != 4:
        raise ValueError(
            f"Expected 4D tensors (B, C, H, W), got predictions={tuple(predictions.shape)} "
            f"and targets={tuple(targets.shape)}."
        )
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: predictions {tuple(predictions.shape)} vs targets {tuple(targets.shape)}."
        )
    if not (0.0 < threshold < 1.0):
        raise ValueError(f"threshold must be strictly between 0 and 1, got {threshold}")


def binarize_predictions(
    predictions: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> torch.Tensor:
    """Convert raw logits or probabilities into binary ``{0, 1}`` masks."""
    if not (0.0 < threshold < 1.0):
        raise ValueError(f"threshold must be strictly between 0 and 1, got {threshold}")
    probs = torch.sigmoid(predictions.float()) if from_logits else predictions.float()
    return (probs >= threshold).to(dtype=torch.long)


def _compute_single_channel_metrics_from_counts(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
) -> dict[str, Any]:
    """Compute deterministic metrics and empty-mask metadata from confusion counts."""
    target_present = (tp + fn) > 0
    prediction_present = (tp + fp) > 0

    if not target_present and not prediction_present:
        return {
            "dice": 1.0,
            "iou": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "sensitivity": 1.0,
            "specificity": 1.0,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "target_present": False,
            "prediction_present": False,
            "empty_handling": "both_empty_exact_match",
        }

    if not target_present and prediction_present:
        spec = float(tn) / float(tn + fp) if (tn + fp) > 0 else 0.0
        return {
            "dice": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "sensitivity": 0.0,
            "specificity": spec,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "target_present": False,
            "prediction_present": True,
            "empty_handling": "target_empty_false_positive",
        }

    if target_present and not prediction_present:
        spec = float(tn) / float(tn + fp) if (tn + fp) > 0 else 1.0
        return {
            "dice": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "sensitivity": 0.0,
            "specificity": spec,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "target_present": True,
            "prediction_present": False,
            "empty_handling": "prediction_empty_false_negative",
        }

    # Both target and prediction are non-empty
    dice = (2.0 * float(tp)) / float(2 * tp + fp + fn)
    iou = float(tp) / float(tp + fp + fn)
    precision = float(tp) / float(tp + fp)
    recall = float(tp) / float(tp + fn)
    sensitivity = recall
    specificity = float(tn) / float(tn + fp) if (tn + fp) > 0 else 1.0

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "target_present": True,
        "prediction_present": True,
        "empty_handling": "standard_non_empty",
    }


def compute_confusion_counts(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-channel ``(tp, fp, fn, tn)`` integer tensors of shape ``(C,)``."""
    _validate_metric_inputs(predictions, targets, threshold)
    pred_bin = binarize_predictions(predictions, threshold=threshold, from_logits=from_logits)
    target_bin = (targets.float() >= 0.5).to(dtype=torch.long)

    dims = (0, 2, 3)
    tp = torch.sum((pred_bin == 1) & (target_bin == 1), dim=dims).cpu()
    fp = torch.sum((pred_bin == 1) & (target_bin == 0), dim=dims).cpu()
    fn = torch.sum((pred_bin == 0) & (target_bin == 1), dim=dims).cpu()
    tn = torch.sum((pred_bin == 0) & (target_bin == 0), dim=dims).cpu()
    return tp, fp, fn, tn


def summarize_channel_counts(
    tp: torch.Tensor,
    fp: torch.Tensor,
    fn: torch.Tensor,
    tn: torch.Tensor,
    channel_names: tuple[str, ...] = CANONICAL_LESION_ORDER,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Build full per-class and macro metric report from per-channel confusion counts."""
    num_channels = int(tp.numel())
    names = (
        channel_names
        if len(channel_names) == num_channels
        else tuple(f"channel_{i}" for i in range(num_channels))
    )

    per_class: dict[str, dict[str, Any]] = {}
    present_classes: list[str] = []
    absent_classes: list[str] = []

    for idx, name in enumerate(names):
        ch_metrics = _compute_single_channel_metrics_from_counts(
            tp=int(tp[idx].item()),
            fp=int(fp[idx].item()),
            fn=int(fn[idx].item()),
            tn=int(tn[idx].item()),
        )
        per_class[name] = ch_metrics
        if ch_metrics["target_present"]:
            present_classes.append(name)
        else:
            absent_classes.append(name)

    macro: dict[str, float] = {}
    macro_present_only: dict[str, float | None] = {}

    for metric_name in METRIC_NAMES:
        all_vals = [float(per_class[ch][metric_name]) for ch in names]
        macro[metric_name] = float(sum(all_vals) / len(all_vals)) if all_vals else 0.0

        present_vals = [
            float(per_class[ch][metric_name]) for ch in present_classes
        ]
        if present_vals:
            macro_present_only[metric_name] = float(
                sum(present_vals) / len(present_vals)
            )
        else:
            macro_present_only[metric_name] = None

    # Flat convenience keys for logging and checkpoint monitoring
    flat_metrics: dict[str, float] = {}
    for metric_name in METRIC_NAMES:
        flat_metrics[f"{metric_name}_macro"] = macro[metric_name]
        for ch in names:
            flat_metrics[f"{metric_name}_{ch}"] = float(per_class[ch][metric_name])

    return {
        "threshold": float(threshold),
        "channel_order": list(names),
        "present_classes": present_classes,
        "absent_classes": absent_classes,
        "per_class": per_class,
        "macro": macro,
        "macro_present_only": macro_present_only,
        "flat": flat_metrics,
    }


def compute_segmentation_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
    channel_names: tuple[str, ...] = CANONICAL_LESION_ORDER,
) -> dict[str, Any]:
    """Compute per-class and macro-averaged segmentation metrics for a batch or tensor."""
    tp, fp, fn, tn = compute_confusion_counts(
        predictions=predictions,
        targets=targets,
        threshold=threshold,
        from_logits=from_logits,
    )
    return summarize_channel_counts(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        channel_names=channel_names,
        threshold=threshold,
    )


class SegmentationMetricTracker:
    """Stateful accumulator for dataset-level multilabel segmentation metrics.

    Accumulates exact confusion counts ``(TP, FP, FN, TN)`` per lesion channel
    across batches and computes dataset-wide per-class and macro metrics at the
    end of an epoch.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        from_logits: bool = True,
        channel_names: tuple[str, ...] = CANONICAL_LESION_ORDER,
    ) -> None:
        if not (0.0 < threshold < 1.0):
            raise ValueError(f"threshold must be strictly between 0 and 1, got {threshold}")
        self.threshold = float(threshold)
        self.from_logits = bool(from_logits)
        self.channel_names = tuple(channel_names)
        self.num_channels = len(self.channel_names)
        self.reset()

    def reset(self) -> None:
        """Reset accumulated confusion counts and sample counter."""
        self.tp = torch.zeros(self.num_channels, dtype=torch.int64)
        self.fp = torch.zeros(self.num_channels, dtype=torch.int64)
        self.fn = torch.zeros(self.num_channels, dtype=torch.int64)
        self.tn = torch.zeros(self.num_channels, dtype=torch.int64)
        self._device_counts: list[torch.Tensor] | None = None
        self.num_samples = 0
        self.num_batches = 0

    @torch.no_grad()
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Accumulate confusion counts for a single batch without forcing GPU-CPU sync."""
        _validate_metric_inputs(predictions, targets, self.threshold)
        if predictions.shape[1] != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels, got {predictions.shape[1]}."
            )
        pred_bin = binarize_predictions(
            predictions.detach(),
            threshold=self.threshold,
            from_logits=self.from_logits,
        )
        target_bin = (targets.detach().float() >= 0.5).to(dtype=torch.long)

        dims = (0, 2, 3)
        b_tp = torch.sum((pred_bin == 1) & (target_bin == 1), dim=dims)
        b_fp = torch.sum((pred_bin == 1) & (target_bin == 0), dim=dims)
        b_fn = torch.sum((pred_bin == 0) & (target_bin == 1), dim=dims)
        b_tn = torch.sum((pred_bin == 0) & (target_bin == 0), dim=dims)

        if self._device_counts is None:
            self._device_counts = [b_tp, b_fp, b_fn, b_tn]
        else:
            self._device_counts[0] = self._device_counts[0] + b_tp
            self._device_counts[1] = self._device_counts[1] + b_fp
            self._device_counts[2] = self._device_counts[2] + b_fn
            self._device_counts[3] = self._device_counts[3] + b_tn

        self.num_samples += int(predictions.shape[0])
        self.num_batches += 1

    def compute(self) -> dict[str, Any]:
        """Compute summarized per-class and macro metrics over all accumulated batches."""
        if self._device_counts is not None:
            self.tp = self._device_counts[0].cpu()
            self.fp = self._device_counts[1].cpu()
            self.fn = self._device_counts[2].cpu()
            self.tn = self._device_counts[3].cpu()
            self._device_counts = None
        summary = summarize_channel_counts(
            tp=self.tp,
            fp=self.fp,
            fn=self.fn,
            tn=self.tn,
            channel_names=self.channel_names,
            threshold=self.threshold,
        )
        summary["num_samples"] = self.num_samples
        summary["num_batches"] = self.num_batches
        return summary
