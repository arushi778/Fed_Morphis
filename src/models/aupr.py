"""Area Under the Precision-Recall Curve (AUPR) and Average Precision (AP) evaluation.

Provides deterministic, memory-efficient precision-recall evaluation for multilabel
retinal lesion segmentation (Microaneurysms ``MA``, Haemorrhages ``HE``,
Hard Exudates ``EX``, Soft Exudates ``SE``).

Evaluation Protocol (FedMorphis Project Protocol v1):
----------------------------------------------------
1. **Background & Motivation**:
   Retinal lesion segmentation suffers from extreme class imbalance where lesions
   occupy only a tiny fraction (often < 0.1%) of total fundus pixels, while
   healthy background accounts for > 99.9%. Under heavy class imbalance, ROC-AUC
   is overly optimistic because the vast number of true negatives deflates the
   false positive rate denominator. Precision-Recall curves focus strictly on the
   minority positive class (the lesion tissue).

2. **Average Precision (AP) vs. Trapezoidal AUPR**:
   - **Average Precision (AP)**: Calculated via step-wise Riemann sum of precision
     weighted by the recall increase at each threshold:
         AP = sum_{n} (R_n - R_{n-1}) * P_n
     Implemented via ``sklearn.metrics.average_precision_score``.
   - **Trapezoidal AUPR (AUC-PR)**: Calculated via trapezoidal numerical integration
     of the empirical precision-recall curve points:
         AUPR_trapz = int_0^1 P(R) dR ~= sum_{k} ((P_k + P_{k+1}) / 2) * (R_{k+1} - R_k)
     Implemented via ``sklearn.metrics.auc(recall, precision)`` on monotonic recall.
   - **Distinction**: In highly non-linear or sparse PR spaces, linear trapezoidal
     interpolation can overestimate the area between distant recall points, while AP
     uses the horizontal step interpolation recommended by Davis & Goadrich (ICML 2006).
     Both metrics are reported separately with unambiguous keys.

3. **Aggregation Protocol**:
   - **Global Pooled (Micro-level, Primary Benchmark Protocol)**:
     Aggregates all predicted probabilities and binary targets across all images in
     the evaluation split into a unified distribution per channel. This evaluates
     overall dataset-level segmentation quality and avoids unstable or undefined
     image-level metrics when an individual fundus image lacks positive lesion pixels.
   - **Per-Image Mean (Macro per-image average)**:
     Evaluates AP and AUPR on each image individually, and averages across images
     where the lesion target is present.

4. **Absent Class & Empty-Mask Policy**:
   - If a lesion channel has zero ground-truth positive pixels across the evaluated set
     (target_present is False), recall is mathematically undefined (0 / 0).
   - Rather than returning silent 0.0 or raising unhandled exceptions, the evaluator:
     * Records ``target_present = False``.
     * Sets ``ap = None`` and ``aupr_trapezoidal = None``.
     * Sets ``empty_handling = "absent_target_class"``.
     * Categorizes the class into ``absent_classes`` (excluded from ``macro_present_only``).

5. **Endpoints & Undefined Precision**:
   - At threshold = 1.0 (or when no pixels are predicted positive, TP + FP = 0),
     precision is set to 1.0 following standard convention (starting at R=0, P=1.0).
   - For an uninformative flat prediction on a non-empty target, AP equals the
     base lesion prevalence P / (P + N).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve
import torch

from src.data.masks import CANONICAL_LESION_CHANNELS

PROTOCOL_NAME: str = "fedmorphis_pixel_level_aupr_v1"
CANONICAL_LESION_ORDER: tuple[str, ...] = tuple(CANONICAL_LESION_CHANNELS)

DEFAULT_OPERATING_THRESHOLDS: tuple[float, ...] = (
    0.10,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
)

DEFAULT_FROZEN_THRESHOLDS: dict[str, float] = {
    "MA": 0.25,
    "HE": 0.40,
    "EX": 0.35,
    "SE": 0.35,
}


@dataclass
class ChannelPRResult:
    """Detailed precision-recall evaluation result for a single lesion channel."""

    channel_name: str
    target_present: bool
    positive_pixels: int
    negative_pixels: int
    total_pixels: int
    prevalence: float
    ap: float | None
    aupr_trapezoidal: float | None
    empty_handling: str
    operating_thresholds: dict[str, dict[str, float]]
    curve_summary: dict[str, list[float]] | None
    raw_curve_points: int

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to JSON-serializable dictionary."""
        return asdict(self)


def _to_flat_numpy_arrays(
    y_true: np.ndarray | torch.Tensor,
    y_score: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor | None = None,
    _prevalidated: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert inputs to 1D uint8/float32 numpy arrays, applying optional mask."""
    if (
        _prevalidated
        and mask is None
        and isinstance(y_true, np.ndarray)
        and isinstance(y_score, np.ndarray)
        and y_true.ndim == 1
        and y_score.ndim == 1
        and y_true.shape == y_score.shape
        and y_true.dtype == np.uint8
        and y_score.dtype == np.float32
        and y_true.flags.c_contiguous
        and y_score.flags.c_contiguous
    ):
        return y_true, y_score

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_score, torch.Tensor):
        y_score = y_score.detach().cpu().numpy()

    if not isinstance(y_true, np.ndarray) or not isinstance(y_score, np.ndarray):
        raise TypeError("y_true and y_score must be numpy.ndarray or torch.Tensor instances.")

    if y_true.shape != y_score.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_score {y_score.shape}."
        )

    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        if not isinstance(mask, np.ndarray):
            raise TypeError("mask must be a numpy.ndarray or torch.Tensor instance.")
        if mask.shape != y_true.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match y_true shape {y_true.shape}."
            )
        valid = mask if mask.dtype == bool else mask.astype(bool)
        t_flat = y_true[valid].ravel()
        s_flat = y_score[valid].ravel()
    else:
        t_flat = y_true.ravel()
        s_flat = y_score.ravel()

    # Binarize targets as compact 1-byte uint8 ({0, 1}): >= 0.5 is positive
    if (
        t_flat.dtype == np.uint8
        and t_flat.flags.c_contiguous
        and (t_flat.size == 0 or int(np.max(t_flat)) <= 1)
    ):
        t_bin = t_flat
    else:
        t_bin = (t_flat >= 0.5).astype(np.uint8)

    if s_flat.dtype == np.float32 and s_flat.flags.c_contiguous:
        s_probs = s_flat
    else:
        s_probs = np.ascontiguousarray(s_flat, dtype=np.float32)

    # Check for NaNs or Infs
    if not np.all(np.isfinite(s_probs)):
        raise ValueError("y_score contains non-finite values (NaN or Inf).")

    return t_bin, s_probs


def sample_pr_curve_grid(
    recalls: np.ndarray,
    precisions: np.ndarray,
    num_grid_points: int = 101,
) -> dict[str, list[float]]:
    """Sample precision-recall curve onto a uniform recall grid for compact serialization.

    Uses monotonic precision envelope (standard maximum precision at recall >= r)
    interpolated over ``num_grid_points`` uniformly spaced in [0.0, 1.0].
    """
    if len(recalls) == 0 or len(precisions) == 0:
        return {"recall": [], "precision": []}

    # recalls returned by precision_recall_curve decrease from 1.0 to 0.0
    r_inc = recalls[::-1]
    p_inc = precisions[::-1]

    grid_r = np.linspace(0.0, 1.0, num_grid_points, dtype=np.float32)
    # Compute precision envelope: maximum precision achieved at recall >= r
    # On increasing r_inc, this corresponds to maximum precision from right to left
    p_envelope = np.maximum.accumulate(p_inc[::-1])[::-1]

    # Interpolate envelope onto grid
    grid_p = np.interp(grid_r, r_inc, p_envelope)

    return {
        "recall": [round(float(r), 4) for r in grid_r],
        "precision": [round(float(p), 4) for p in grid_p],
    }


def compute_operating_threshold_metrics(
    y_true_bin: np.ndarray,
    y_score_probs: np.ndarray,
    thresholds: Sequence[float] = DEFAULT_OPERATING_THRESHOLDS,
) -> dict[str, dict[str, float]]:
    """Compute empirical confusion, precision, recall, and F1 across specific operating thresholds."""
    results: dict[str, dict[str, float]] = {}
    pos_target = y_true_bin == 1
    neg_target = y_true_bin == 0
    num_pos = int(np.sum(pos_target))

    for th in thresholds:
        th_val = float(th)
        pred_pos = y_score_probs >= th_val

        tp = int(np.sum(pred_pos & pos_target))
        fp = int(np.sum(pred_pos & neg_target))
        fn = num_pos - tp

        prec = float(tp) / float(tp + fp) if (tp + fp) > 0 else 1.0
        rec = float(tp) / float(num_pos) if num_pos > 0 else 0.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0.0 else 0.0

        key = f"{th_val:.2f}"
        results[key] = {
            "threshold": th_val,
            "precision": float(round(prec, 6)),
            "recall": float(round(rec, 6)),
            "f1": float(round(f1, 6)),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return results


def compute_channel_pr_curve(
    y_true: np.ndarray | torch.Tensor,
    y_score: np.ndarray | torch.Tensor,
    channel_name: str = "lesion",
    mask: np.ndarray | torch.Tensor | None = None,
    operating_thresholds: Sequence[float] = DEFAULT_OPERATING_THRESHOLDS,
    sample_curve: bool = True,
    num_grid_points: int = 101,
    _prevalidated: bool = False,
) -> ChannelPRResult:
    """Compute Average Precision, Trapezoidal AUPR, and PR curve for a single channel.

    Args:
        y_true: Ground truth binary masks (any shape, values >= 0.5 treated as 1).
        y_score: Predicted continuous probability map in [0.0, 1.0].
        channel_name: Identifier for the channel (e.g. 'MA', 'HE').
        mask: Optional binary spatial mask (e.g. FOV). Non-zero pixels are evaluated.
        operating_thresholds: Thresholds to evaluate empirical precision/recall.
        sample_curve: If True, includes compact sampled PR curve data for reporting.
        num_grid_points: Number of recall grid points for curve sampling.
        _prevalidated: Internal fast-path flag when inputs are already validated 1D arrays.

    Returns:
        :class:`ChannelPRResult` with AP, Trapezoidal AUPR, and detailed curve metadata.
    """
    t_bin, s_probs = _to_flat_numpy_arrays(
        y_true, y_score, mask=mask, _prevalidated=_prevalidated
    )

    total_pixels = int(t_bin.size)
    positive_pixels = int(np.sum(t_bin))
    negative_pixels = total_pixels - positive_pixels
    prevalence = float(positive_pixels) / float(total_pixels) if total_pixels > 0 else 0.0

    # Case 1: Target has zero positive pixels across the evaluated set
    if positive_pixels == 0:
        return ChannelPRResult(
            channel_name=str(channel_name),
            target_present=False,
            positive_pixels=0,
            negative_pixels=negative_pixels,
            total_pixels=total_pixels,
            prevalence=0.0,
            ap=None,
            aupr_trapezoidal=None,
            empty_handling="absent_target_class",
            operating_thresholds={
                f"{float(th):.2f}": {
                    "threshold": float(th),
                    "precision": 0.0 if np.any(s_probs >= float(th)) else 1.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "tp": 0,
                    "fp": int(np.sum(s_probs >= float(th))),
                    "fn": 0,
                }
                for th in operating_thresholds
            },
            curve_summary=None,
            raw_curve_points=0,
        )

    # Case 2: Target is non-empty
    is_all_zero_pred = bool(s_probs.size > 0 and np.all(s_probs == 0.0))
    empty_handling_label = (
        "all_zero_prediction" if is_all_zero_pred else "standard_non_empty"
    )

    # Average Precision via step-wise interpolation
    ap = float(average_precision_score(t_bin, s_probs))

    # Precision-Recall curve
    prec, rec, thresholds = precision_recall_curve(t_bin, s_probs)

    # Trapezoidal AUPR: recall decreases from 1 to 0 in precision_recall_curve,
    # so we integrate over reversed (monotonically increasing) recall
    aupr_trapz = float(auc(rec[::-1], prec[::-1]))

    # Operating threshold metrics
    op_metrics = compute_operating_threshold_metrics(
        t_bin, s_probs, thresholds=operating_thresholds
    )

    # Sampled curve for lightweight serialization
    curve_summary = (
        sample_pr_curve_grid(rec, prec, num_grid_points=num_grid_points)
        if sample_curve
        else None
    )

    return ChannelPRResult(
        channel_name=str(channel_name),
        target_present=True,
        positive_pixels=positive_pixels,
        negative_pixels=negative_pixels,
        total_pixels=total_pixels,
        prevalence=prevalence,
        ap=float(round(ap, 6)),
        aupr_trapezoidal=float(round(aupr_trapz, 6)),
        empty_handling=empty_handling_label,
        operating_thresholds=op_metrics,
        curve_summary=curve_summary,
        raw_curve_points=int(len(prec)),
    )


def compute_multilabel_aupr(
    predictions: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor | None = None,
    channel_names: Sequence[str] = CANONICAL_LESION_ORDER,
    from_logits: bool = False,
    operating_thresholds: Sequence[float] = DEFAULT_OPERATING_THRESHOLDS,
    sample_curve: bool = True,
    num_grid_points: int = 101,
) -> dict[str, Any]:
    """Compute per-channel and macro AP / AUPR metrics for multilabel retinal lesion maps.

    Args:
        predictions: Predicted tensor/array of shape (C, ...) or (B, C, ...).
        targets: Ground truth binary tensor/array of matching shape.
        mask: Optional spatial mask of shape (...) or (B, 1, ...) or (1, ...).
        channel_names: Sequence of channel names in canonical order.
        from_logits: If True, applies sigmoid to predictions.
        operating_thresholds: Thresholds for operating point metrics.
        sample_curve: If True, includes sampled PR curves in per-class results.
        num_grid_points: Number of points for PR curve sampling.

    Returns:
        Structured dictionary containing:
        - ``channel_order``: List of channel names.
        - ``present_classes``: List of classes with non-empty targets.
        - ``absent_classes``: List of classes with zero positive pixels.
        - ``per_class``: Mapping of channel name to :class:`ChannelPRResult` dict.
        - ``macro``: Macro metrics across all channels (or None if classes absent).
        - ``macro_present_only``: Macro metrics across present channels.
        - ``protocol``: Protocol identifier string.
    """
    if isinstance(predictions, torch.Tensor):
        if from_logits:
            predictions = torch.sigmoid(predictions)
        predictions = predictions.detach().cpu().numpy()
    elif from_logits:
        predictions = 1.0 / (1.0 + np.exp(-predictions))

    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    if predictions.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}."
        )

    # Determine channel dimension:
    # If 4D (B, C, H, W): channel is dim 1
    # If 3D (C, H, W): channel is dim 0
    # If 2D (C, N): channel is dim 0
    if predictions.ndim == 4:
        num_channels = predictions.shape[1]
        c_axis = 1
    elif predictions.ndim in (2, 3):
        num_channels = predictions.shape[0]
        c_axis = 0
    else:
        raise ValueError(
            f"Expected predictions of ndim 2, 3, or 4; got shape {predictions.shape}."
        )

    if num_channels != len(channel_names):
        names = tuple(f"channel_{i}" for i in range(num_channels))
    else:
        names = tuple(channel_names)

    per_class: dict[str, dict[str, Any]] = {}
    present_classes: list[str] = []
    absent_classes: list[str] = []

    for c_idx, ch_name in enumerate(names):
        if c_axis == 1:
            p_ch = predictions[:, c_idx]
            t_ch = targets[:, c_idx]
        else:
            p_ch = predictions[c_idx]
            t_ch = targets[c_idx]

        m_ch = mask
        if m_ch is not None and m_ch.ndim == predictions.ndim:
            if c_axis == 1:
                m_ch = m_ch[:, min(c_idx, m_ch.shape[1] - 1)]
            else:
                m_ch = m_ch[min(c_idx, m_ch.shape[0] - 1)]

        ch_res = compute_channel_pr_curve(
            y_true=t_ch,
            y_score=p_ch,
            channel_name=ch_name,
            mask=m_ch,
            operating_thresholds=operating_thresholds,
            sample_curve=sample_curve,
            num_grid_points=num_grid_points,
        )

        res_dict = ch_res.to_dict()
        per_class[ch_name] = res_dict
        if ch_res.target_present:
            present_classes.append(ch_name)
        else:
            absent_classes.append(ch_name)

    # Compute macro-averages
    # 1. macro_present_only: mean across present classes
    present_aps = [
        float(per_class[ch]["ap"])
        for ch in present_classes
        if per_class[ch]["ap"] is not None
    ]
    present_auprs = [
        float(per_class[ch]["aupr_trapezoidal"])
        for ch in present_classes
        if per_class[ch]["aupr_trapezoidal"] is not None
    ]

    macro_present_only = {
        "ap": float(sum(present_aps) / len(present_aps)) if present_aps else None,
        "aupr_trapezoidal": (
            float(sum(present_auprs) / len(present_auprs)) if present_auprs else None
        ),
    }

    # 2. macro across all channels (None if any channel is absent)
    all_aps = [per_class[ch]["ap"] for ch in names]
    all_auprs = [per_class[ch]["aupr_trapezoidal"] for ch in names]

    if all(val is not None for val in all_aps) and all_aps:
        ap_macro: float | None = float(sum(all_aps) / len(all_aps))  # type: ignore[arg-type]
    else:
        ap_macro = None

    if all(val is not None for val in all_auprs) and all_auprs:
        aupr_macro: float | None = float(sum(all_auprs) / len(all_auprs))  # type: ignore[arg-type]
    else:
        aupr_macro = None

    macro = {
        "ap": ap_macro,
        "aupr_trapezoidal": aupr_macro,
    }

    # Flat convenience keys for logging
    flat: dict[str, Any] = {
        "ap_macro": ap_macro,
        "ap_macro_present_only": macro_present_only["ap"],
        "aupr_macro": aupr_macro,
        "aupr_macro_present_only": macro_present_only["aupr_trapezoidal"],
    }
    for ch in names:
        flat[f"ap_{ch}"] = per_class[ch]["ap"]
        flat[f"aupr_trapezoidal_{ch}"] = per_class[ch]["aupr_trapezoidal"]

    return {
        "protocol": PROTOCOL_NAME,
        "channel_order": list(names),
        "present_classes": present_classes,
        "absent_classes": absent_classes,
        "macro": macro,
        "macro_present_only": macro_present_only,
        "per_class": per_class,
        "flat": flat,
    }


class AUPRTracker:
    """Stateful accumulator for dataset-level AUPR and AP evaluation.

    Accumulates flattened predictions and ground-truth targets across batches
    or full images, and computes both global pooled (micro) and per-image (macro)
    AUPR metrics without maintaining bulky GPU tensors.
    """

    def __init__(
        self,
        channel_names: Sequence[str] = CANONICAL_LESION_ORDER,
        from_logits: bool = True,
        operating_thresholds: Sequence[float] = DEFAULT_OPERATING_THRESHOLDS,
        store_per_image: bool = True,
    ) -> None:
        self.channel_names = tuple(channel_names)
        self.num_channels = len(self.channel_names)
        self.from_logits = bool(from_logits)
        self.operating_thresholds = tuple(operating_thresholds)
        self.store_per_image = bool(store_per_image)
        self.reset()

    def reset(self) -> None:
        """Reset accumulated predictions, targets, and image records."""
        self._pooled_targets: dict[str, list[np.ndarray]] = {
            ch: [] for ch in self.channel_names
        }
        self._pooled_scores: dict[str, list[np.ndarray]] = {
            ch: [] for ch in self.channel_names
        }
        self.per_image_results: list[dict[str, Any]] = []
        self.num_samples = 0
        self.total_pixels_evaluated = 0

    @torch.no_grad()
    def update(
        self,
        predictions: np.ndarray | torch.Tensor,
        targets: np.ndarray | torch.Tensor,
        mask: np.ndarray | torch.Tensor | None = None,
        sample_ids: Sequence[str] | str | None = None,
    ) -> None:
        """Accumulate predictions and targets for a batch or single image.

        Args:
            predictions: Tensor/array of shape (B, C, H, W) or (C, H, W).
            targets: Tensor/array of matching shape.
            mask: Optional spatial valid mask (e.g. FOV).
            sample_ids: Optional sample ID(s) for per-image tracking.
        """
        if isinstance(predictions, torch.Tensor):
            if self.from_logits:
                predictions = torch.sigmoid(predictions)
            pred_np = predictions.detach().cpu().numpy()
        else:
            if self.from_logits:
                pred_np = 1.0 / (1.0 + np.exp(-predictions))
            else:
                pred_np = np.asarray(predictions, dtype=np.float32)

        if isinstance(targets, torch.Tensor):
            target_np = targets.detach().cpu().numpy()
        else:
            target_np = np.asarray(targets)

        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask_np = mask.detach().cpu().numpy()
            else:
                mask_np = np.asarray(mask)
        else:
            mask_np = None

        if pred_np.ndim == 3:
            # Single sample: (C, H, W) -> add batch dimension (1, C, H, W)
            pred_np = pred_np[None, ...]
            target_np = target_np[None, ...]
            if mask_np is not None and mask_np.ndim in (2, 3):
                mask_np = mask_np[None, ...]
        elif pred_np.ndim == 4 and mask_np is not None and mask_np.ndim == 2:
            mask_np = np.broadcast_to(
                mask_np[None, ...],
                (pred_np.shape[0], mask_np.shape[0], mask_np.shape[1]),
            )

        if pred_np.shape[1] != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels, got shape {pred_np.shape}."
            )

        batch_size = int(pred_np.shape[0])

        if sample_ids is not None:
            if isinstance(sample_ids, str):
                s_ids = [sample_ids]
            else:
                s_ids = [str(sid) for sid in sample_ids]
        else:
            s_ids = [f"sample_{self.num_samples + i}" for i in range(batch_size)]

        for b_idx in range(batch_size):
            p_img = pred_np[b_idx]  # (C, H, W)
            t_img = target_np[b_idx]  # (C, H, W)
            m_img = mask_np[b_idx] if mask_np is not None else None
            if m_img is not None and m_img.ndim == 2 and m_img.dtype != bool:
                m_img = m_img.astype(bool)

            img_pixels = 0
            img_channel_metrics: dict[str, Any] = {}

            for c_idx, ch in enumerate(self.channel_names):
                p_c = p_img[c_idx]
                t_c = t_img[c_idx]
                m_c = m_img[c_idx] if (m_img is not None and m_img.ndim == 3) else m_img

                t_flat, s_flat = _to_flat_numpy_arrays(t_c, p_c, mask=m_c)
                self._pooled_targets[ch].append(t_flat)
                self._pooled_scores[ch].append(s_flat)
                img_pixels += len(t_flat)

                if self.store_per_image:
                    ch_res = compute_channel_pr_curve(
                        y_true=t_flat,
                        y_score=s_flat,
                        channel_name=ch,
                        sample_curve=False,  # Keep per-image records compact
                        _prevalidated=True,
                    )
                    img_channel_metrics[ch] = ch_res.to_dict()

            if self.store_per_image:
                self.per_image_results.append(
                    {
                        "sample_id": s_ids[b_idx] if b_idx < len(s_ids) else f"sample_{self.num_samples}",
                        "channels": img_channel_metrics,
                    }
                )

            self.num_samples += 1
            self.total_pixels_evaluated += img_pixels // self.num_channels

    def compute(self) -> dict[str, Any]:
        """Compute dataset-level global pooled and per-image mean AP and AUPR metrics.

        Returns:
            Structured dictionary with:
            - ``protocol``: Protocol version string.
            - ``num_samples``: Total images evaluated.
            - ``total_pixels_evaluated``: Total pixels evaluated per channel.
            - ``global_pooled``: Dataset-level pooled AUPR and AP metrics.
            - ``per_image_mean``: Mean of per-image AP and AUPR metrics.
            - ``per_class``: Per-class global pooled metrics.
            - ``macro``: Macro metrics across all channels.
            - ``macro_present_only``: Macro metrics across channels with targets present.
        """
        if self.num_samples == 0:
            raise ValueError("AUPRTracker contains zero evaluated samples.")

        # Compute global pooled metrics
        per_class_pooled: dict[str, dict[str, Any]] = {}
        present_classes: list[str] = []
        absent_classes: list[str] = []

        for ch in self.channel_names:
            ch_targets = (
                np.concatenate(self._pooled_targets[ch])
                if self._pooled_targets[ch]
                else np.empty(0, dtype=np.uint8)
            )
            ch_scores = (
                np.concatenate(self._pooled_scores[ch])
                if self._pooled_scores[ch]
                else np.empty(0, dtype=np.float32)
            )

            ch_res = compute_channel_pr_curve(
                y_true=ch_targets,
                y_score=ch_scores,
                channel_name=ch,
                operating_thresholds=self.operating_thresholds,
                sample_curve=True,
                _prevalidated=True,
            )
            res_dict = ch_res.to_dict()
            per_class_pooled[ch] = res_dict

            if ch_res.target_present:
                present_classes.append(ch)
            else:
                absent_classes.append(ch)

        # Macro averages for global pooled
        present_aps = [
            float(per_class_pooled[ch]["ap"])
            for ch in present_classes
            if per_class_pooled[ch]["ap"] is not None
        ]
        present_auprs = [
            float(per_class_pooled[ch]["aupr_trapezoidal"])
            for ch in present_classes
            if per_class_pooled[ch]["aupr_trapezoidal"] is not None
        ]

        macro_present_only = {
            "ap": float(sum(present_aps) / len(present_aps)) if present_aps else None,
            "aupr_trapezoidal": (
                float(sum(present_auprs) / len(present_auprs)) if present_auprs else None
            ),
        }

        all_aps = [per_class_pooled[ch]["ap"] for ch in self.channel_names]
        all_auprs = [per_class_pooled[ch]["aupr_trapezoidal"] for ch in self.channel_names]

        ap_macro = (
            float(sum(all_aps) / len(all_aps))  # type: ignore[arg-type]
            if all(v is not None for v in all_aps) and all_aps
            else None
        )
        aupr_macro = (
            float(sum(all_auprs) / len(all_auprs))  # type: ignore[arg-type]
            if all(v is not None for v in all_auprs) and all_auprs
            else None
        )

        global_pooled_summary = {
            "channel_order": list(self.channel_names),
            "present_classes": present_classes,
            "absent_classes": absent_classes,
            "macro": {
                "ap": ap_macro,
                "aupr_trapezoidal": aupr_macro,
            },
            "macro_present_only": macro_present_only,
            "per_class": per_class_pooled,
        }

        # Per-image mean metrics (if tracked)
        per_image_mean: dict[str, Any] = {}
        if self.store_per_image and self.per_image_results:
            per_ch_means: dict[str, dict[str, float | None]] = {}
            for ch in self.channel_names:
                img_aps = [
                    res["channels"][ch]["ap"]
                    for res in self.per_image_results
                    if res["channels"][ch]["target_present"] and res["channels"][ch]["ap"] is not None
                ]
                img_auprs = [
                    res["channels"][ch]["aupr_trapezoidal"]
                    for res in self.per_image_results
                    if res["channels"][ch]["target_present"] and res["channels"][ch]["aupr_trapezoidal"] is not None
                ]
                per_ch_means[ch] = {
                    "ap_mean": float(sum(img_aps) / len(img_aps)) if img_aps else None,
                    "aupr_trapezoidal_mean": (
                        float(sum(img_auprs) / len(img_auprs)) if img_auprs else None
                    ),
                    "images_with_target": len(img_aps),
                    "total_images": len(self.per_image_results),
                }
            per_image_mean = {
                "per_class": per_ch_means,
            }

        return {
            "protocol": PROTOCOL_NAME,
            "num_samples": self.num_samples,
            "total_pixels_evaluated": self.total_pixels_evaluated,
            "channel_order": list(self.channel_names),
            "present_classes": present_classes,
            "absent_classes": absent_classes,
            "macro": global_pooled_summary["macro"],
            "macro_present_only": macro_present_only,
            "per_class": per_class_pooled,
            "global_pooled": global_pooled_summary,
            "per_image_mean": per_image_mean,
            "flat": {
                "ap_macro": ap_macro,
                "ap_macro_present_only": macro_present_only["ap"],
                "aupr_macro": aupr_macro,
                "aupr_macro_present_only": macro_present_only["aupr_trapezoidal"],
                **{
                    f"ap_{ch}": per_class_pooled[ch]["ap"]
                    for ch in self.channel_names
                },
                **{
                    f"aupr_trapezoidal_{ch}": per_class_pooled[ch]["aupr_trapezoidal"]
                    for ch in self.channel_names
                },
            },
        }
