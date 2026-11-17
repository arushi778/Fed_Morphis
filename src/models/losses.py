"""Multilabel retinal lesion segmentation loss functions.

Implements numerically stable, configurable loss functions for four independent
binary lesion channels ``[MA, HE, EX, SE]``:
- Binary Cross-Entropy with Logits (:class:`WeightedBCELoss`)
- Soft Dice and Balanced Dice (:class:`SoftDiceLoss`, :class:`BalancedDiceLoss`)
- Asymmetric Tversky and Focal Tversky (:class:`TverskyLoss`, :class:`FocalTverskyLoss`)
- Composite BCE + Dice/Tversky/Focal Tversky (:class:`BCEDiceLoss`, :class:`BCETverskyLoss`,
  :class:`BCEFocalTverskyLoss`)
- Training-only metadata-driven class weight derivation (:func:`derive_training_class_weights`)
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.masks import CANONICAL_LESION_CHANNELS
from src.utils.config import SUPPORTED_LOSS_TYPES, load_yaml_config
from src.utils.paths import resolve_project_path

CANONICAL_LESION_ORDER: tuple[str, ...] = tuple(CANONICAL_LESION_CHANNELS)


class TrainingWeightDerivationError(ValueError):
    """Raised when training-only class weight derivation encounters invalid split or metadata."""


def _is_valid_positive_number(value: Any) -> bool:
    """Return True if ``value`` is a finite positive real number (excluding bool)."""
    return (
        isinstance(value, (int, float, np.number))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _is_valid_non_negative_number(value: Any) -> bool:
    """Return True if ``value`` is a finite non-negative real number (excluding bool)."""
    return (
        isinstance(value, (int, float, np.number))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _validate_channel_weight_vector(
    weights: Sequence[float] | torch.Tensor | np.ndarray | None,
    field_name: str,
    expected_channels: int = 4,
    reshape_4d: bool = False,
) -> torch.Tensor | None:
    """Validate an optional per-channel positive weight vector for ``[MA, HE, EX, SE]``."""
    if weights is None:
        return None
    if isinstance(weights, (str, bytes, Mapping)):
        raise ValueError(
            f"{field_name} must be a sequence or 1D tensor of {expected_channels} positive numbers, "
            f"got {type(weights).__name__}."
        )
    if isinstance(weights, torch.Tensor):
        if weights.dtype == torch.bool:
            raise ValueError(f"{field_name} cannot be a boolean tensor.")
        flat_list = weights.detach().cpu().reshape(-1).tolist()
    elif isinstance(weights, np.ndarray):
        if weights.dtype == np.bool_:
            raise ValueError(f"{field_name} cannot be a boolean array.")
        flat_list = weights.reshape(-1).tolist()
    elif isinstance(weights, Sequence):
        flat_list = list(weights)
    else:
        raise ValueError(
            f"{field_name} must be a sequence or tensor of {expected_channels} positive numbers, "
            f"got {type(weights).__name__}."
        )

    if len(flat_list) != expected_channels:
        raise ValueError(
            f"{field_name} must have length {expected_channels} corresponding to "
            f"{list(CANONICAL_LESION_ORDER)}, got length {len(flat_list)} ({weights!r})."
        )

    validated_floats: list[float] = []
    for idx, val in enumerate(flat_list):
        if not _is_valid_positive_number(val):
            raise ValueError(
                f"{field_name}[{idx}] must be a finite positive number > 0, got {val!r}."
            )
        validated_floats.append(float(val))

    tensor = torch.tensor(validated_floats, dtype=torch.float32)
    if reshape_4d:
        return tensor.view(1, expected_channels, 1, 1)
    return tensor


def _validate_segmentation_tensors(
    logits: torch.Tensor,
    targets: torch.Tensor,
    expected_channels: int | None = None,
) -> None:
    """Validate logits and binary target masks for multilabel segmentation."""
    if not isinstance(logits, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError("Both logits and targets must be torch.Tensor instances.")
    if logits.ndim != 4:
        raise ValueError(
            f"Expected 4D logits tensor of shape (B, C, H, W), got {tuple(logits.shape)}."
        )
    if targets.ndim != 4:
        raise ValueError(
            f"Expected 4D targets tensor of shape (B, C, H, W), got {tuple(targets.shape)}."
        )
    if logits.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch between logits {tuple(logits.shape)} and targets {tuple(targets.shape)}."
        )
    if expected_channels is not None and logits.shape[1] != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} lesion channels, got {logits.shape[1]}."
        )
    if logits.shape[0] < 1:
        raise ValueError("Batch dimension B must be at least 1.")


def _apply_channel_weights(
    channel_loss: torch.Tensor,
    class_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Compute scalar mean (or class-weighted mean) across channels."""
    if class_weights is None:
        return channel_loss.mean()
    weights = class_weights.to(device=channel_loss.device, dtype=channel_loss.dtype).view(-1)
    if weights.numel() != channel_loss.numel():
        raise ValueError(
            f"Channel weight count ({weights.numel()}) does not match loss channel count ({channel_loss.numel()})."
        )
    return torch.mean(channel_loss * weights)


class SoftDiceLoss(nn.Module):
    """Per-channel Soft Dice Loss for multilabel binary segmentation.

    Computes ``1 - Dice`` independently per lesion channel after applying
    ``torch.sigmoid`` to raw logits. Uses Laplace smoothing ``smooth`` in both
    numerator and denominator so empty ground-truth and empty prediction
    channels are finite and never divide by zero.

    Args:
        smooth: Laplace smoothing constant (must be > 0).
        batch_dice: If True, aggregates intersection and denominator across the
            batch dimension ``(B, H, W)`` per channel before computing channel
            Dice. If False, computes Dice per ``(sample, channel)`` and averages
            across samples and channels.
        class_weights: Optional sequence/tensor of per-channel weights for ``[MA, HE, EX, SE]``.
        channel_names: Ordered tuple of channel names.
    """

    component_name: str = "dice"

    def __init__(
        self,
        smooth: float = 1e-5,
        batch_dice: bool = True,
        class_weights: Sequence[float] | torch.Tensor | None = None,
        channel_names: tuple[str, ...] = CANONICAL_LESION_ORDER,
    ) -> None:
        super().__init__()
        if not _is_valid_positive_number(smooth):
            raise ValueError(f"smooth must be a finite number > 0, got {smooth!r}")
        if not isinstance(batch_dice, bool):
            raise ValueError(f"batch_dice must be a boolean, got {batch_dice!r}")
        self.smooth = float(smooth)
        self.batch_dice = bool(batch_dice)
        self.channel_names = tuple(channel_names)

        cw_tensor = _validate_channel_weight_vector(
            class_weights,
            field_name="class_weights",
            expected_channels=len(self.channel_names),
            reshape_4d=False,
        )
        if cw_tensor is not None:
            self.register_buffer("class_weights", cw_tensor)
        else:
            self.class_weights = None

    def forward_per_channel(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-channel Soft Dice loss tensor of shape ``(C,)``."""
        _validate_segmentation_tensors(
            logits,
            targets,
            expected_channels=len(self.channel_names) if self.class_weights is not None else None,
        )
        probs = torch.sigmoid(logits.float())
        targets_float = targets.float().clamp(0.0, 1.0)

        if self.batch_dice:
            # Aggregate over (B, H, W) -> shape (C,)
            dims = (0, 2, 3)
            target_sum = torch.sum(targets_float, dim=dims)
            intersection = torch.sum(probs * targets_float, dim=dims)
            cardinality = torch.sum(probs * probs + targets_float * targets_float, dim=dims)
            dice_present = 1.0 - (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            empty_loss = torch.mean(probs * probs, dim=dims)
            return torch.where(target_sum > 0.0, dice_present, empty_loss)

        # Sum over spatial dimensions (H, W) -> shape (B, C)
        dims = (2, 3)
        target_sum = torch.sum(targets_float, dim=dims)
        intersection = torch.sum(probs * targets_float, dim=dims)
        cardinality = torch.sum(probs * probs + targets_float * targets_float, dim=dims)
        dice_present = 1.0 - (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        empty_loss = torch.mean(probs * probs, dim=dims)
        sample_channel_loss = torch.where(target_sum > 0.0, dice_present, empty_loss)
        return sample_channel_loss.mean(dim=0)

    def forward_tensors(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return 5-tuple ``(total_loss, bce_zero, dice_loss, bce_per_ch_zero, dice_per_ch)``."""
        dice_per_channel = self.forward_per_channel(logits, targets)
        dice_loss = _apply_channel_weights(dice_per_channel, self.class_weights)
        zero_scalar = torch.zeros_like(dice_loss)
        zero_per_ch = torch.zeros_like(dice_per_channel)
        return dice_loss, zero_scalar, dice_loss, zero_per_ch, dice_per_channel

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scalar mean (or class-weighted mean) Soft Dice loss across all channels."""
        channel_loss = self.forward_per_channel(logits, targets)
        return _apply_channel_weights(channel_loss, self.class_weights)


class BalancedDiceLoss(SoftDiceLoss):
    """Per-channel Balanced Soft Dice Loss with configurable per-class weights.

    Applies per-channel weights ``class_weights`` across ``[MA, HE, EX, SE]`` when
    aggregating channel Dice losses into a scalar objective.
    """

    component_name: str = "balanced_dice"

    def __init__(
        self,
        class_weights: Sequence[float] | torch.Tensor | None = None,
        smooth: float = 1e-5,
        batch_dice: bool = True,
        channel_names: tuple[str, ...] = CANONICAL_LESION_ORDER,
    ) -> None:
        super().__init__(
            smooth=smooth,
            batch_dice=batch_dice,
            class_weights=class_weights,
            channel_names=channel_names,
        )


class TverskyLoss(nn.Module):
    """Per-channel Tversky Loss for multilabel binary segmentation.

    Computes ``1 - TverskyIndex`` independently per lesion channel after applying
    ``torch.sigmoid`` to raw logits.

    Tversky Index formulation:
        ``TI = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)``

    When ``beta > alpha`` (e.g., ``alpha=0.3, beta=0.7``), false negatives (missed lesions)
    are penalized more heavily than false positives, boosting recall on small,
    sparse lesion classes like microaneurysms (MA).

    Args:
        alpha: Weight constant for false positives (must be in [0, 1]). Default 0.3.
        beta: Weight constant for false negatives (must be in [0, 1]). Default 0.7.
        smooth: Positive Laplace smoothing constant. Default 1e-5.
        batch_tversky: If True, aggregates TP, FP, FN across (B, H, W) per channel
            before computing Tversky index. If False, computes per sample and averages.
        class_weights: Optional sequence/tensor of per-channel weights for ``[MA, HE, EX, SE]``.
        channel_names: Ordered tuple of channel names.
    """

    component_name: str = "tversky"

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-5,
        batch_tversky: bool = True,
        class_weights: Sequence[float] | torch.Tensor | None = None,
        channel_names: tuple[str, ...] = CANONICAL_LESION_ORDER,
    ) -> None:
        super().__init__()
        if not _is_valid_non_negative_number(alpha) or float(alpha) > 1.0:
            raise ValueError(f"alpha must be a finite number in [0, 1], got {alpha!r}")
        if not _is_valid_non_negative_number(beta) or float(beta) > 1.0:
            raise ValueError(f"beta must be a finite number in [0, 1], got {beta!r}")
        if (float(alpha) + float(beta)) <= 0.0:
            raise ValueError("alpha + beta must be > 0.")
        if not _is_valid_positive_number(smooth):
            raise ValueError(f"smooth must be a finite number > 0, got {smooth!r}")
        if not isinstance(batch_tversky, bool):
            raise ValueError(f"batch_tversky must be a boolean, got {batch_tversky!r}")

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)
        self.batch_tversky = bool(batch_tversky)
        self.channel_names = tuple(channel_names)

        cw_tensor = _validate_channel_weight_vector(
            class_weights,
            field_name="class_weights",
            expected_channels=len(self.channel_names),
            reshape_4d=False,
        )
        if cw_tensor is not None:
            self.register_buffer("class_weights", cw_tensor)
        else:
            self.class_weights = None

    def forward_per_channel(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-channel Tversky loss tensor of shape ``(C,)``."""
        _validate_segmentation_tensors(
            logits,
            targets,
            expected_channels=len(self.channel_names) if self.class_weights is not None else None,
        )
        probs = torch.sigmoid(logits.float())
        targets_float = targets.float().clamp(0.0, 1.0)

        intersection = probs * targets_float
        fp = probs * (1.0 - targets_float)
        fn = (1.0 - probs) * targets_float

        if self.batch_tversky:
            dims = (0, 2, 3)
            target_sum = torch.sum(targets_float, dim=dims)
            tp_sum = torch.sum(intersection, dim=dims)
            fp_sum = torch.sum(fp, dim=dims)
            fn_sum = torch.sum(fn, dim=dims)

            denominator = tp_sum + self.alpha * fp_sum + self.beta * fn_sum + self.smooth
            tversky_present = 1.0 - (tp_sum + self.smooth) / denominator
            empty_loss = torch.mean(probs * probs, dim=dims)
            return torch.where(target_sum > 0.0, tversky_present, empty_loss)

        dims = (2, 3)
        target_sum = torch.sum(targets_float, dim=dims)
        tp_sum = torch.sum(intersection, dim=dims)
        fp_sum = torch.sum(fp, dim=dims)
        fn_sum = torch.sum(fn, dim=dims)

        denominator = tp_sum + self.alpha * fp_sum + self.beta * fn_sum + self.smooth
        tversky_present = 1.0 - (tp_sum + self.smooth) / denominator
        empty_loss = torch.mean(probs * probs, dim=dims)
        sample_channel_loss = torch.where(target_sum > 0.0, tversky_present, empty_loss)
        return sample_channel_loss.mean(dim=0)

    def forward_tensors(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return 5-tuple ``(total_loss, bce_zero, tversky_loss, bce_per_ch_zero, tversky_per_ch)``."""
        tversky_per_channel = self.forward_per_channel(logits, targets)
        tversky_loss = _apply_channel_weights(tversky_per_channel, self.class_weights)
        zero_scalar = torch.zeros_like(tversky_loss)
        zero_per_ch = torch.zeros_like(tversky_per_channel)
        return tversky_loss, zero_scalar, tversky_loss, zero_per_ch, tversky_per_channel

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scalar mean (or class-weighted mean) Tversky loss across all channels."""
        channel_loss = self.forward_per_channel(logits, targets)
        return _apply_channel_weights(channel_loss, self.class_weights)


class FocalTverskyLoss(TverskyLoss):
    """Per-channel Focal Tversky Loss for multilabel binary segmentation.

    For present-target channels (``target_sum > 0``):
        ``L_FT = (1 - TI)^gamma``
    where ``1 - TI = (alpha * FP + beta * FN) / (TP + alpha * FP + beta * FN + smooth)``.

    Numerical safeguards:
    - Computes ``one_minus_ti`` directly from the non-negative numerator
      ``alpha * FP + beta * FN`` divided by ``denominator``, avoiding catastrophic
      cancellation in ``1.0 - (TP + smooth) / denominator`` when ``TI`` is close to 1.
    - For ``gamma >= 1.0``, ``u^gamma`` and its derivative ``gamma * u^(gamma - 1)`` are
      smooth and finite for all ``u >= 0`` (including ``u = 0``), so no positive floor
      is imposed when ``gamma >= 1.0``.
    - For ``0 < gamma < 1.0``, ``d(u^gamma)/du = gamma * u^(gamma - 1)`` is singular at
      ``u = 0``. To preserve exact zero loss at ``u = 0`` while bounding the derivative
      with respect to ``u`` by ``gamma * eps^(gamma - 1)``, the shifted-power formulation
      ``(u + eps)^gamma - eps^gamma`` is applied using the explicitly documented
      numerical-stability parameter ``eps > 0`` (default ``1e-6``).
    - For absent-target channels (``target_sum == 0``), retains the un-exponentiated
      empty-target penalty ``L_empty = mean(p^2)``.

    Args:
        alpha: False-positive weight in ``[0, 1]``.
        beta: False-negative weight in ``[0, 1]``.
        gamma: Positive focal exponent ``> 0``.
        smooth: Positive denominator smoothing constant ``> 0``.
        eps: Positive numerical-stability shift ``> 0`` used in the shifted-power
            formulation ``(u + eps)^gamma - eps^gamma`` when ``0 < gamma < 1``.
        batch_tversky: Whether to aggregate over the batch axis.
        class_weights: Optional per-channel weights for ``[MA, HE, EX, SE]``.
        channel_names: Ordered tuple of channel names.
    """

    component_name: str = "focal_tversky"

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 4.0 / 3.0,
        smooth: float = 1e-5,
        batch_tversky: bool = True,
        class_weights: Sequence[float] | torch.Tensor | None = None,
        channel_names: tuple[str, ...] = CANONICAL_LESION_ORDER,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            alpha=alpha,
            beta=beta,
            smooth=smooth,
            batch_tversky=batch_tversky,
            class_weights=class_weights,
            channel_names=channel_names,
        )
        if not _is_valid_positive_number(gamma):
            raise ValueError(f"gamma must be a finite positive number > 0, got {gamma!r}")
        if not _is_valid_positive_number(eps):
            raise ValueError(f"eps must be a finite positive number > 0, got {eps!r}")
        self.gamma = float(gamma)
        self.eps = float(eps)

    def _apply_focal_exponent(self, one_minus_ti: torch.Tensor) -> torch.Tensor:
        """Apply focal exponent ``gamma`` to ``1 - TI`` with shifted-power stability for ``0 < gamma < 1``."""
        u = one_minus_ti.clamp(min=0.0, max=1.0)
        if self.gamma == 1.0:
            return u
        if self.gamma < 1.0:
            # Shifted-power formulation: (u + eps)^gamma - eps^gamma
            # Guarantees exact 0.0 at u == 0.0 and bounds d/du by gamma * eps^(gamma - 1).
            eps_t = torch.as_tensor(self.eps, device=u.device, dtype=u.dtype)
            return torch.pow(u + eps_t, self.gamma) - torch.pow(eps_t, self.gamma)
        return torch.pow(u, self.gamma)

    def forward_per_channel(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-channel Focal Tversky loss tensor of shape ``(C,)``."""
        _validate_segmentation_tensors(
            logits,
            targets,
            expected_channels=len(self.channel_names) if self.class_weights is not None else None,
        )
        probs = torch.sigmoid(logits.float())
        targets_float = targets.float().clamp(0.0, 1.0)

        intersection = probs * targets_float
        fp = probs * (1.0 - targets_float)
        fn = (1.0 - probs) * targets_float

        if self.batch_tversky:
            dims = (0, 2, 3)
            target_sum = torch.sum(targets_float, dim=dims)
            tp_sum = torch.sum(intersection, dim=dims)
            fp_sum = torch.sum(fp, dim=dims)
            fn_sum = torch.sum(fn, dim=dims)

            error_num = self.alpha * fp_sum + self.beta * fn_sum
            denominator = tp_sum + error_num + self.smooth
            one_minus_ti = error_num / denominator
            ftl_present = self._apply_focal_exponent(one_minus_ti)
            empty_loss = torch.mean(probs * probs, dim=dims)
            return torch.where(target_sum > 0.0, ftl_present, empty_loss)

        dims = (2, 3)
        target_sum = torch.sum(targets_float, dim=dims)
        tp_sum = torch.sum(intersection, dim=dims)
        fp_sum = torch.sum(fp, dim=dims)
        fn_sum = torch.sum(fn, dim=dims)

        error_num = self.alpha * fp_sum + self.beta * fn_sum
        denominator = tp_sum + error_num + self.smooth
        one_minus_ti = error_num / denominator
        ftl_present = self._apply_focal_exponent(one_minus_ti)
        empty_loss = torch.mean(probs * probs, dim=dims)
        sample_channel_loss = torch.where(target_sum > 0.0, ftl_present, empty_loss)
        return sample_channel_loss.mean(dim=0)


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross-Entropy with Logits + Soft Dice Loss.

    Designed for multilabel retinal lesion segmentation where the four classes
    ``[MA, HE, EX, SE]`` are independent binary channels and lesions are sparse
    or absent for specific classes within an image.

    Total loss:
        ``total_loss = bce_weight * bce_loss + dice_weight * dice_loss``

    Args:
        bce_weight: Non-negative weight for ``BCEWithLogitsLoss``.
        dice_weight: Non-negative weight for ``SoftDiceLoss``.
        smooth: Positive smoothing constant for ``SoftDiceLoss``.
        pos_weight: Optional per-channel positive class weights of length ``C`` for BCE.
        class_weights: Optional per-channel weights of length ``C`` for the overlap loss.
        batch_dice: Whether ``SoftDiceLoss`` aggregates over the batch axis.
        channel_names: Ordered tuple of channel names.
    """

    primary_component_name: str = "bce"
    secondary_component_name: str = "dice"

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-5,
        pos_weight: Sequence[float] | torch.Tensor | None = None,
        batch_dice: bool = True,
        class_weights: Sequence[float] | torch.Tensor | None = None,
        channel_names: tuple[str, ...] = CANONICAL_LESION_ORDER,
    ) -> None:
        super().__init__()
        if not _is_valid_non_negative_number(bce_weight):
            raise ValueError(f"bce_weight must be a finite number >= 0, got {bce_weight!r}")
        if not _is_valid_non_negative_number(dice_weight):
            raise ValueError(f"dice_weight must be a finite number >= 0, got {dice_weight!r}")
        if float(bce_weight) == 0.0 and float(dice_weight) == 0.0:
            raise ValueError("At least one of bce_weight or dice_weight must be > 0.")
        if not _is_valid_positive_number(smooth):
            raise ValueError(f"smooth must be a finite number > 0, got {smooth!r}")
        if not isinstance(batch_dice, bool):
            raise ValueError(f"batch_dice must be a boolean, got {batch_dice!r}")

        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)
        self.channel_names = tuple(channel_names)

        pw_tensor = _validate_channel_weight_vector(
            pos_weight,
            field_name="pos_weight",
            expected_channels=len(self.channel_names),
            reshape_4d=True,
        )
        if pw_tensor is not None:
            self.register_buffer("pos_weight", pw_tensor)
        else:
            self.pos_weight = None

        cw_tensor = _validate_channel_weight_vector(
            class_weights,
            field_name="class_weights",
            expected_channels=len(self.channel_names),
            reshape_4d=False,
        )
        if cw_tensor is not None:
            self.register_buffer("class_weights", cw_tensor)
        else:
            self.class_weights = None

        self.dice_loss_fn: nn.Module = SoftDiceLoss(
            smooth=self.smooth,
            batch_dice=batch_dice,
            class_weights=class_weights,
            channel_names=self.channel_names,
        )

    def forward_tensors(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute total loss and per-channel primary/secondary tensors without CPU synchronization.

        Returns:
            ``(total_loss, component_1, component_2, component_1_per_channel, component_2_per_channel)``
        """
        _validate_segmentation_tensors(logits, targets, expected_channels=len(self.channel_names))
        logits_fp32 = logits.float()
        targets_fp32 = targets.float().clamp(0.0, 1.0)

        bce_per_pixel = F.binary_cross_entropy_with_logits(
            logits_fp32,
            targets_fp32,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        bce_per_channel = bce_per_pixel.mean(dim=(0, 2, 3))
        bce_loss = bce_per_channel.mean()

        if self.dice_weight > 0.0:
            dice_per_channel = self.dice_loss_fn.forward_per_channel(
                logits_fp32, targets_fp32
            )
            dice_loss = _apply_channel_weights(dice_per_channel, self.class_weights)
        else:
            dice_per_channel = torch.zeros_like(bce_per_channel)
            dice_loss = torch.zeros_like(bce_loss)

        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return total_loss, bce_loss, dice_loss, bce_per_channel, dice_per_channel

    def compute_components(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, Any]:
        """Compute total loss along with detached primary, secondary, and per-channel components."""
        (
            total_loss,
            bce_loss,
            second_loss,
            bce_per_channel,
            second_per_channel,
        ) = self.forward_tensors(logits, targets)

        if self.class_weights is not None:
            cw = self.class_weights.to(
                device=second_per_channel.device, dtype=second_per_channel.dtype
            ).view(-1)
            weighted_second_per_ch = second_per_channel * cw
        else:
            weighted_second_per_ch = second_per_channel

        per_channel_total = (
            self.bce_weight * bce_per_channel + self.dice_weight * weighted_second_per_ch
        )

        num_channels = logits.shape[1]
        names = (
            self.channel_names
            if len(self.channel_names) == num_channels
            else tuple(f"channel_{i}" for i in range(num_channels))
        )

        sec_name = self.secondary_component_name
        result: dict[str, Any] = {
            "loss": total_loss,
            "primary_component_name": self.primary_component_name,
            "secondary_component_name": sec_name,
            "bce_loss": bce_loss.detach(),
            "dice_loss": second_loss.detach(),
            "overlap_loss": second_loss.detach(),
            f"{sec_name}_loss": second_loss.detach(),
            "per_channel_bce": {
                name: float(bce_per_channel[i].detach().item())
                for i, name in enumerate(names)
            },
            "per_channel_dice": {
                name: float(second_per_channel[i].detach().item())
                for i, name in enumerate(names)
            },
            f"per_channel_{sec_name}": {
                name: float(second_per_channel[i].detach().item())
                for i, name in enumerate(names)
            },
            "per_channel_loss": {
                name: float(per_channel_total[i].detach().item())
                for i, name in enumerate(names)
            },
        }
        return result

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scalar weighted composite loss."""
        return self.forward_tensors(logits, targets)[0]


class WeightedBCELoss(BCEDiceLoss):
    """Per-channel Weighted Binary Cross-Entropy with Logits Loss.

    Subclasses :class:`BCEDiceLoss` with ``bce_weight=1.0`` and ``dice_weight=0.0``
    so it preserves the 5-tuple :meth:`forward_tensors` contract for trainer and
    evaluator compatibility while computing pure (optionally ``pos_weight``-weighted)
    four-channel BCE.
    """

    primary_component_name: str = "bce"
    secondary_component_name: str = "none"

    def __init__(
        self,
        pos_weight: Sequence[float] | torch.Tensor | None = None,
        bce_weight: float = 1.0,
        channel_names: tuple[str, ...] = CANONICAL_LESION_ORDER,
    ) -> None:
        if not _is_valid_positive_number(bce_weight):
            raise ValueError(f"bce_weight must be a finite number > 0, got {bce_weight!r}")
        super().__init__(
            bce_weight=float(bce_weight),
            dice_weight=0.0,
            smooth=1e-5,
            pos_weight=pos_weight,
            batch_dice=True,
            class_weights=None,
            channel_names=channel_names,
        )


class BCETverskyLoss(BCEDiceLoss):
    """Combined Binary Cross-Entropy with Logits + Asymmetric Tversky Loss.

    Designed for multilabel retinal lesion segmentation to improve recall on small,
    sparse lesions like microaneurysms (MA) and subtle hemorrhages (HE).

    Total loss:
        ``total_loss = bce_weight * bce_loss + tversky_weight * tversky_loss``

    Inherits from :class:`BCEDiceLoss` to preserve full compatibility with
    :class:`SupervisedSegmentationTrainer` and evaluation harnesses.
    """

    primary_component_name: str = "bce"
    secondary_component_name: str = "tversky"

    def __init__(
        self,
        bce_weight: float = 0.5,
        tversky_weight: float = 0.5,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-5,
        pos_weight: Sequence[float] | torch.Tensor | None = None,
        batch_tversky: bool = True,
        class_weights: Sequence[float] | torch.Tensor | None = None,
        channel_names: tuple[str, ...] = CANONICAL_LESION_ORDER,
    ) -> None:
        super().__init__(
            bce_weight=bce_weight,
            dice_weight=tversky_weight,
            smooth=smooth,
            pos_weight=pos_weight,
            batch_dice=batch_tversky,
            class_weights=class_weights,
            channel_names=channel_names,
        )
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tversky_weight = float(tversky_weight)
        self.tversky_loss_fn = TverskyLoss(
            alpha=self.alpha,
            beta=self.beta,
            smooth=self.smooth,
            batch_tversky=batch_tversky,
            class_weights=class_weights,
            channel_names=self.channel_names,
        )
        # Point dice_loss_fn to tversky_loss_fn so inherited methods evaluate Tversky loss
        self.dice_loss_fn = self.tversky_loss_fn


class BCEFocalTverskyLoss(BCETverskyLoss):
    """Combined Binary Cross-Entropy with Logits + Focal Tversky Loss.

    Supports both unweighted BCE + Focal Tversky and Weighted BCE (via ``pos_weight``)
    + Focal Tversky (with optional per-channel ``class_weights``).

    Total loss:
        ``total_loss = bce_weight * bce_loss + focal_tversky_weight * focal_tversky_loss``
    """

    primary_component_name: str = "bce"
    secondary_component_name: str = "focal_tversky"

    def __init__(
        self,
        bce_weight: float = 0.5,
        focal_tversky_weight: float = 0.5,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 4.0 / 3.0,
        smooth: float = 1e-5,
        pos_weight: Sequence[float] | torch.Tensor | None = None,
        batch_tversky: bool = True,
        class_weights: Sequence[float] | torch.Tensor | None = None,
        channel_names: tuple[str, ...] = CANONICAL_LESION_ORDER,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            bce_weight=bce_weight,
            tversky_weight=focal_tversky_weight,
            alpha=alpha,
            beta=beta,
            smooth=smooth,
            pos_weight=pos_weight,
            batch_tversky=batch_tversky,
            class_weights=class_weights,
            channel_names=channel_names,
        )
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.focal_tversky_weight = float(focal_tversky_weight)
        self.focal_tversky_loss_fn = FocalTverskyLoss(
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            smooth=self.smooth,
            batch_tversky=batch_tversky,
            class_weights=class_weights,
            channel_names=self.channel_names,
            eps=self.eps,
        )
        self.tversky_loss_fn = self.focal_tversky_loss_fn
        self.dice_loss_fn = self.focal_tversky_loss_fn


def extract_loss_components(
    criterion: nn.Module,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, float, float, str, str]:
    """Evaluate ``criterion`` and extract ``(loss, bce_val, overlap_val, primary_name, secondary_name)``.

    Ensures that standalone overlap losses (:class:`SoftDiceLoss`, :class:`BalancedDiceLoss`,
    :class:`TverskyLoss`, :class:`FocalTverskyLoss`) as well as composite losses
    (:class:`BCEDiceLoss`, :class:`BCETverskyLoss`, :class:`BCEFocalTverskyLoss`,
    :class:`WeightedBCELoss`) report accurate component values and component names
    without ever silently reporting zero overlap loss for non-:class:`BCEDiceLoss` classes.
    """
    (
        loss,
        bce_val,
        overlap_val,
        primary_name,
        secondary_name,
        _,
        _,
    ) = extract_loss_components_with_per_channel(criterion, logits, targets)
    return loss, bce_val, overlap_val, primary_name, secondary_name


def extract_loss_components_with_per_channel(
    criterion: nn.Module,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[
    torch.Tensor,
    float,
    float,
    str,
    str,
    dict[str, float],
    dict[str, float],
]:
    """Evaluate ``criterion`` and extract scalar and per-channel loss components.

    Returns:
        7-tuple ``(loss, bce_val, overlap_val, primary_name, secondary_name,
        per_channel_bce, per_channel_overlap)`` where ``per_channel_bce`` and
        ``per_channel_overlap`` map canonical lesion names ``[MA, HE, EX, SE]``
        to detached float loss values.
    """
    channel_names = tuple(
        getattr(criterion, "channel_names", CANONICAL_LESION_ORDER)
    )
    num_channels = int(logits.shape[1]) if logits.ndim >= 2 else len(channel_names)
    resolved_names = (
        channel_names
        if len(channel_names) == num_channels
        else tuple(f"channel_{i}" for i in range(num_channels))
    )

    if isinstance(criterion, BCEDiceLoss) or (
        hasattr(criterion, "forward_tensors") and callable(criterion.forward_tensors)
    ):
        loss, bce_t, overlap_t, bce_per_ch_t, overlap_per_ch_t = (
            criterion.forward_tensors(logits, targets)
        )
        default_primary = "bce" if isinstance(criterion, BCEDiceLoss) else "none"
        default_secondary = (
            "dice"
            if isinstance(criterion, BCEDiceLoss)
            else getattr(criterion, "component_name", "overlap")
        )
        primary_name = getattr(criterion, "primary_component_name", default_primary)
        secondary_name = getattr(
            criterion, "secondary_component_name", default_secondary
        )

        bce_ch_list = bce_per_ch_t.detach().cpu().tolist()
        ov_ch_list = overlap_per_ch_t.detach().cpu().tolist()
        per_channel_bce = {
            str(name): float(bce_ch_list[i])
            for i, name in enumerate(resolved_names)
            if i < len(bce_ch_list)
        }
        per_channel_overlap = {
            str(name): float(ov_ch_list[i])
            for i, name in enumerate(resolved_names)
            if i < len(ov_ch_list)
        }
        return (
            loss,
            float(bce_t.detach().item()),
            float(overlap_t.detach().item()),
            str(primary_name),
            str(secondary_name),
            per_channel_bce,
            per_channel_overlap,
        )

    loss = criterion(logits, targets)
    loss_val = float(loss.detach().item())
    zero_per_ch = {str(name): 0.0 for name in resolved_names}
    return loss, loss_val, 0.0, "custom", "none", dict(zero_per_ch), dict(zero_per_ch)


def _extract_metadata_records_for_weights(
    manifest_or_records: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Extract manifest record dicts strictly from metadata without loading any images or masks."""
    if isinstance(manifest_or_records, (str, Path)):
        path = resolve_project_path(manifest_or_records)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Manifest file not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or "records" not in payload:
            raise TrainingWeightDerivationError(
                f"Manifest file '{path}' must be a JSON object containing a 'records' list."
            )
        raw_records = payload["records"]
    elif isinstance(manifest_or_records, Mapping):
        if "records" not in manifest_or_records:
            raise TrainingWeightDerivationError(
                "Manifest mapping is missing required 'records' key."
            )
        raw_records = manifest_or_records["records"]
    elif isinstance(manifest_or_records, Sequence) and not isinstance(
        manifest_or_records, (str, bytes)
    ):
        raw_records = manifest_or_records
    else:
        raise TrainingWeightDerivationError(
            f"Unsupported manifest_or_records type {type(manifest_or_records).__name__}."
        )

    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        raise TrainingWeightDerivationError(
            "Manifest 'records' must be a sequence of record dictionaries."
        )
    if len(raw_records) == 0:
        raise TrainingWeightDerivationError(
            "Cannot derive class weights from an empty records sequence."
        )

    records: list[dict[str, Any]] = []
    for idx, rec in enumerate(raw_records):
        if not isinstance(rec, Mapping):
            raise TrainingWeightDerivationError(
                f"Record at index {idx} must be a mapping, got {type(rec).__name__}."
            )
        records.append(dict(rec))
    return records


def derive_training_class_weights(
    manifest_or_records: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    mode: str = "sqrt_dampened",
    pixel_domain: str = "fov_crop_box",
    min_weight: float | None = 1.0,
    max_weight: float | None = 25.0,
    zero_positive_policy: str | None = None,
    normalize_mean: bool = False,
    channel_names: tuple[str, ...] = CANONICAL_LESION_ORDER,
) -> dict[str, Any]:
    """Derive deterministic positive-class weights strictly from training-split manifest metadata.

    Security & Methodological Guarantees:
    1. **Train-split isolation**: Every input record must explicitly carry ``split == "train"``.
       Any record with ``valid``, ``val``, ``validation``, ``test``, missing, empty, or
       ambiguous ``split`` metadata immediately raises :class:`TrainingWeightDerivationError`.
    2. **Zero disk I/O on images/masks**: Operates exclusively on precomputed manifest fields
       (``lesion_pixel_counts``, ``fov_crop``, ``image_dimensions``). Never opens or reads
       any image or mask file.
    3. **FOV bounding-box vs. circular retinal FOV accounting**:
       During training, :class:`RetinalLesionSegmentationDataset` crops each sample to its
       rectangular ``fov_crop`` bounding box ``[y_min:y_max, x_min:x_max]`` (or full image if
       ``fov_crop=False``) and feeds the rectangular tensor directly to the loss without a
       circular FOV mask. Consequently, the training loss includes the dark corner pixels
       inside the rectangular ``fov_crop`` bounding box. Because manifest metadata stores
       rectangular ``fov_crop`` bounds and ``image_dimensions`` (and does not store exact
       circular valid-FOV pixel counts), negative pixel counts computed here represent
       **rectangular training-tensor negative pixels** (either ``fov_crop_box`` or ``full_image``),
       NOT exact circular retinal-pixel ratios. This distinction and limitation are explicitly
       recorded in the returned metadata dictionary.
    4. **Normalization and clipping order (normalize-then-clip with clipping precedence)**:
       - First, raw unclipped weights ``raw_unclipped_weights_by_class`` are computed from
         training pixel counts (or ``zero_positive_policy`` for absent channels).
       - Second, if ``normalize_mean=True``, the raw weights across ``present_channels``
         (channels with ``> 0`` positive pixels) are rescaled so their pre-clip present-channel
         mean equals ``1.0`` (``normalized_unclipped_weights_by_class``). Zero-positive channels
         retain their configured ``zero_positive_policy`` weight and are excluded from the
         mean-1 normalization divisor.
       - Third, explicit ``min_weight`` and ``max_weight`` bounds (when not ``None``) are applied.
         **Explicit clipping bounds take precedence over ``normalize_mean=True``**.
       - Reporting fields unambiguously separate present-channel and all-channel means:
         - ``present_channels``: list of channels with ``> 0`` positive pixels.
         - ``zero_positive_channels``: list of channels with ``0`` positive pixels.
         - ``present_weights_mean`` (and ``present_channels_weights_mean``): post-clip mean across
           ``present_channels`` only (equals ``1.0`` when ``normalize_mean=True`` and no present
           channel is clipped).
         - ``weights_mean`` (and ``all_channels_weights_mean``): post-clip mean across all 4
           canonical channels ``[MA, HE, EX, SE]`` (can differ from ``1.0`` without clipping when
           ``zero_positive_channels`` is non-empty).
         - ``clipping_altered_mean``: ``True`` if and only if ``[min_weight, max_weight]`` clipping
           modified one or more channel weights (``len(clipped_channels) > 0``), never triggered
           merely by a zero-positive channel's policy weight.
         - Contradictory bounds that make mean ``1.0`` impossible (``min_weight > 1.0`` or
           ``max_weight < 1.0`` when ``normalize_mean=True``) raise :class:`ValueError`.

    Args:
        manifest_or_records: Path to manifest JSON, manifest dict, or sequence of ``train`` records.
        mode: Weighting formula:
            - ``"sqrt_dampened"`` (default): ``sqrt(neg_pixels / pos_pixels)``
            - ``"linear_ratio"``: ``neg_pixels / pos_pixels``
            - ``"log_dampened"``: ``log(1 + neg_pixels / pos_pixels)``
            - ``"effective_num"``: inverse square-root prevalence ``1 / sqrt(prevalence)``
        pixel_domain: Spatial domain used by the training loss:
            - ``"fov_crop_box"`` (default): rectangular FOV bounding box ``(y_max - y_min) * (x_max - x_min)``
              matching ``fov_crop=True`` training tensors.
            - ``"full_image"``: full uncropped image ``height * width`` matching ``fov_crop=False``.
        min_weight: Minimum clipped weight (must be > 0, or ``None`` for no lower bound). Default ``1.0``.
        max_weight: Maximum clipped weight (must be > 0 and >= ``min_weight``, or ``None`` for no upper bound).
            Default ``25.0``.
        zero_positive_policy: How to handle a channel with zero positive pixels across all
            provided training records:
            - ``"max_weight"`` (default): assign ``max_weight`` (requires non-None ``max_weight``)
              and record the channel in ``zero_positive_channels``.
            - ``"min_weight"``: assign ``min_weight`` (requires non-None ``min_weight``).
            - ``"one"``: assign ``1.0`` (clipped to ``[min_weight, max_weight]``).
            - ``"error"``: raise :class:`TrainingWeightDerivationError`.
        normalize_mean: If True, rescales the raw unclipped weights so their pre-clip mean across
            present channels is 1.0 before applying ``[min_weight, max_weight]`` clipping.
        channel_names: Canonical channel order ``("MA", "HE", "EX", "SE")``.

    Returns:
        Structured dictionary with ``weights`` (list of 4 floats in ``[MA, HE, EX, SE]`` order),
        ``weights_by_class``, ``raw_unclipped_weights_by_class``,
        ``normalized_unclipped_weights_by_class``, ``weights_mean``, ``clipping_altered_mean``,
        ``normalization_clipping_order``, ``weight_source="derived_from_train_metadata"``,
        per-class pixel counts, prevalence, absent-image counts, and explicit domain/limitation
        documentation.
    """
    supported_modes = {"sqrt_dampened", "linear_ratio", "log_dampened", "effective_num"}
    mode_norm = str(mode).strip().lower()
    if mode_norm not in supported_modes:
        raise ValueError(
            f"Unsupported weight derivation mode {mode!r}; expected one of {sorted(supported_modes)}."
        )

    supported_domains = {"fov_crop_box", "full_image"}
    domain_norm = str(pixel_domain).strip().lower()
    if domain_norm not in supported_domains:
        raise ValueError(
            f"Unsupported pixel_domain {pixel_domain!r}; expected one of {sorted(supported_domains)}."
        )

    zero_policy_explicit = zero_positive_policy is not None
    raw_zero_policy = "max_weight" if zero_positive_policy is None else zero_positive_policy
    if not isinstance(raw_zero_policy, str):
        raise ValueError(
            f"Unsupported zero_positive_policy {zero_positive_policy!r}; expected a string."
        )
    supported_zero_policies = {"max_weight", "min_weight", "one", "error"}
    zero_policy_norm = raw_zero_policy.strip().lower()
    if zero_policy_norm not in supported_zero_policies:
        raise ValueError(
            f"Unsupported zero_positive_policy {zero_positive_policy!r}; "
            f"expected one of {sorted(supported_zero_policies)}."
        )

    if not isinstance(normalize_mean, bool):
        raise ValueError(f"normalize_mean must be a boolean, got {normalize_mean!r}.")

    if min_weight is not None and not _is_valid_positive_number(min_weight):
        raise ValueError(f"min_weight must be a finite number > 0 or None, got {min_weight!r}.")
    if max_weight is not None and not _is_valid_positive_number(max_weight):
        raise ValueError(f"max_weight must be a finite number > 0 or None, got {max_weight!r}.")

    min_w: float | None = float(min_weight) if min_weight is not None else None
    max_w: float | None = float(max_weight) if max_weight is not None else None

    if min_w is not None and max_w is not None and max_w < min_w:
        raise ValueError(
            f"max_weight ({max_weight}) must be >= min_weight ({min_weight})."
        )

    if normalize_mean:
        if min_w is not None and min_w > 1.0:
            raise ValueError(
                f"Contradictory parameters: normalize_mean=True cannot be combined with "
                f"min_weight={min_w} > 1.0 because weights >= {min_w} cannot have mean 1.0."
            )
        if max_w is not None and max_w < 1.0:
            raise ValueError(
                f"Contradictory parameters: normalize_mean=True cannot be combined with "
                f"max_weight={max_w} < 1.0 because weights <= {max_w} cannot have mean 1.0."
            )

    if zero_policy_norm == "min_weight" and min_w is None:
        raise ValueError(
            "Contradictory parameters: zero_positive_policy='min_weight' requires a non-None min_weight."
        )
    if zero_policy_norm == "max_weight" and max_w is None and zero_policy_explicit:
        raise ValueError(
            "Contradictory parameters: zero_positive_policy='max_weight' requires a non-None max_weight."
        )

    channels = tuple(str(c) for c in channel_names)
    if tuple(channels) != CANONICAL_LESION_ORDER:
        raise ValueError(
            f"channel_names must match canonical order {list(CANONICAL_LESION_ORDER)}, got {list(channels)}."
        )

    records = _extract_metadata_records_for_weights(manifest_or_records)

    pos_counts = {c: 0 for c in channels}
    absent_image_counts = {c: 0 for c in channels}
    present_image_counts = {c: 0 for c in channels}
    total_domain_pixels = 0
    total_full_image_pixels = 0
    total_fov_box_pixels = 0

    for idx, rec in enumerate(records):
        sample_id = str(rec.get("sample_id", f"record_{idx}"))
        if "split" not in rec or rec["split"] is None:
            raise TrainingWeightDerivationError(
                f"Missing 'split' metadata on record '{sample_id}' (index {idx}). "
                "Class weight derivation requires explicit split=='train' on every record."
            )
        raw_split = rec["split"]
        if not isinstance(raw_split, str) or not raw_split.strip():
            raise TrainingWeightDerivationError(
                f"Empty or ambiguous 'split' metadata {raw_split!r} on record '{sample_id}' (index {idx})."
            )
        split_norm = raw_split.strip().lower()
        if split_norm != "train":
            raise TrainingWeightDerivationError(
                f"Forbidden non-training record '{sample_id}' with split={raw_split!r} at index {idx}. "
                "Class weights must be derived exclusively from 'train' split metadata."
            )

        img_dims = rec.get("image_dimensions")
        if not isinstance(img_dims, Mapping):
            raise TrainingWeightDerivationError(
                f"Record '{sample_id}' is missing required 'image_dimensions' mapping."
            )
        img_h = img_dims.get("height")
        img_w = img_dims.get("width")
        if (
            isinstance(img_h, bool)
            or isinstance(img_w, bool)
            or not isinstance(img_h, int)
            or not isinstance(img_w, int)
            or img_h <= 0
            or img_w <= 0
        ):
            raise TrainingWeightDerivationError(
                f"Record '{sample_id}' has invalid image_dimensions: height={img_h!r}, width={img_w!r}."
            )
        full_area = int(img_h * img_w)
        total_full_image_pixels += full_area

        fov_info = rec.get("fov_crop")
        if isinstance(fov_info, Mapping) and {
            "y_min",
            "y_max",
            "x_min",
            "x_max",
        }.issubset(fov_info.keys()):
            y0, y1 = int(fov_info["y_min"]), int(fov_info["y_max"])
            x0, x1 = int(fov_info["x_min"]), int(fov_info["x_max"])
            fov_box_area = int((y1 - y0) * (x1 - x0))
            if fov_box_area <= 0 or fov_box_area > full_area:
                raise TrainingWeightDerivationError(
                    f"Record '{sample_id}' has invalid fov_crop bounding box area {fov_box_area} "
                    f"(full_area={full_area})."
                )
        elif domain_norm == "fov_crop_box":
            raise TrainingWeightDerivationError(
                f"Record '{sample_id}' is missing required 'fov_crop' bounding-box metadata "
                "for pixel_domain='fov_crop_box'."
            )
        else:
            fov_box_area = full_area

        total_fov_box_pixels += fov_box_area
        sample_domain_area = fov_box_area if domain_norm == "fov_crop_box" else full_area
        total_domain_pixels += sample_domain_area

        lesion_counts = rec.get("lesion_pixel_counts")
        if not isinstance(lesion_counts, Mapping):
            raise TrainingWeightDerivationError(
                f"Record '{sample_id}' is missing required 'lesion_pixel_counts' mapping."
            )

        for c in channels:
            if c not in lesion_counts:
                raise TrainingWeightDerivationError(
                    f"Record '{sample_id}' is missing lesion_pixel_counts for channel '{c}'."
                )
            cnt_raw = lesion_counts[c]
            if isinstance(cnt_raw, bool) or not isinstance(cnt_raw, int) or cnt_raw < 0:
                raise TrainingWeightDerivationError(
                    f"Record '{sample_id}' has invalid lesion_pixel_counts[{c!r}]={cnt_raw!r}."
                )
            if cnt_raw > sample_domain_area:
                raise TrainingWeightDerivationError(
                    f"Record '{sample_id}' has lesion_pixel_counts[{c!r}]={cnt_raw} exceeding "
                    f"domain pixel area {sample_domain_area}."
                )
            pos_counts[c] += int(cnt_raw)
            if cnt_raw == 0:
                absent_image_counts[c] += 1
            else:
                present_image_counts[c] += 1

    raw_unclipped_weights: dict[str, float] = {}
    zero_positive_channels: list[str] = []
    neg_counts: dict[str, int] = {}
    prevalence_by_class: dict[str, float] = {}
    neg_to_pos_ratio_by_class: dict[str, float | None] = {}

    for c in channels:
        pos_c = pos_counts[c]
        neg_c = int(total_domain_pixels - pos_c)
        neg_counts[c] = neg_c
        prevalence_by_class[c] = float(pos_c) / float(total_domain_pixels)

        if pos_c == 0:
            zero_positive_channels.append(c)
            neg_to_pos_ratio_by_class[c] = None
            if zero_policy_norm == "error":
                raise TrainingWeightDerivationError(
                    f"Channel '{c}' has 0 positive pixels across all {len(records)} training records."
                )
            elif zero_policy_norm == "max_weight":
                if max_w is None:
                    raise ValueError(
                        f"Contradictory parameters: channel '{c}' has 0 positive pixels and "
                        "zero_positive_policy='max_weight', but max_weight is None."
                    )
                raw_unclipped_weights[c] = max_w
            elif zero_policy_norm == "min_weight":
                assert min_w is not None
                raw_unclipped_weights[c] = min_w
            else:  # "one"
                raw_unclipped_weights[c] = 1.0
        else:
            ratio = float(neg_c) / float(pos_c)
            neg_to_pos_ratio_by_class[c] = ratio
            if mode_norm == "sqrt_dampened":
                raw_unclipped_weights[c] = math.sqrt(ratio)
            elif mode_norm == "linear_ratio":
                raw_unclipped_weights[c] = ratio
            elif mode_norm == "log_dampened":
                raw_unclipped_weights[c] = math.log1p(ratio)
            else:  # "effective_num"
                prev = max(prevalence_by_class[c], 1e-12)
                raw_unclipped_weights[c] = 1.0 / math.sqrt(prev)

    present_channels: list[str] = [c for c in channels if c not in zero_positive_channels]
    normalized_unclipped_weights: dict[str, float] = dict(raw_unclipped_weights)
    if normalize_mean:
        present_vals = [raw_unclipped_weights[c] for c in present_channels]
        mean_val = (
            sum(present_vals) / float(len(present_vals))
            if present_vals
            else 1.0
        )
        if mean_val > 0.0:
            for c in present_channels:
                normalized_unclipped_weights[c] = raw_unclipped_weights[c] / mean_val

    clipped_weights_by_class: dict[str, float] = {}
    clipped_channels: list[str] = []
    for c in channels:
        cand_w = normalized_unclipped_weights[c]
        clipped_w = cand_w
        if min_w is not None:
            clipped_w = max(clipped_w, min_w)
        if max_w is not None:
            clipped_w = min(clipped_w, max_w)
        if abs(clipped_w - cand_w) > 1e-12:
            clipped_channels.append(c)
        clipped_weights_by_class[c] = round(float(clipped_w), 6)

    weights_list = [clipped_weights_by_class[c] for c in channels]
    weights_mean = round(float(sum(weights_list) / len(weights_list)), 6)
    present_weights_mean: float | None = (
        round(
            float(
                sum(clipped_weights_by_class[c] for c in present_channels)
                / len(present_channels)
            ),
            6,
        )
        if present_channels
        else None
    )
    pre_clip_present_weights_mean: float | None = (
        round(
            float(
                sum(normalized_unclipped_weights[c] for c in present_channels)
                / len(present_channels)
            ),
            6,
        )
        if present_channels
        else None
    )
    pre_clip_all_weights_mean = round(
        float(sum(normalized_unclipped_weights[c] for c in channels) / len(channels)),
        6,
    )
    clipping_altered_mean = bool(len(clipped_channels) > 0)

    return {
        "channel_order": list(channels),
        "present_channels": present_channels,
        "weights": weights_list,
        "weights_by_class": clipped_weights_by_class,
        "raw_unclipped_weights_by_class": {
            c: round(float(raw_unclipped_weights[c]), 6) for c in channels
        },
        "normalized_unclipped_weights_by_class": {
            c: round(float(normalized_unclipped_weights[c]), 6) for c in channels
        },
        "weights_mean": weights_mean,
        "all_channels_weights_mean": weights_mean,
        "present_weights_mean": present_weights_mean,
        "present_channels_weights_mean": present_weights_mean,
        "pre_clip_present_channels_weights_mean": pre_clip_present_weights_mean,
        "pre_clip_all_channels_weights_mean": pre_clip_all_weights_mean,
        "clipping_altered_mean": clipping_altered_mean,
        "normalization_clipping_order": "normalize_then_clip_with_clipping_precedence",
        "weight_source": "derived_from_train_metadata",
        "mode": mode_norm,
        "pixel_domain": domain_norm,
        "min_weight": min_w,
        "max_weight": max_w,
        "zero_positive_policy": zero_policy_norm,
        "normalize_mean": bool(normalize_mean),
        "num_train_records": len(records),
        "total_domain_pixels": int(total_domain_pixels),
        "total_fov_crop_box_pixels": int(total_fov_box_pixels),
        "total_full_image_pixels": int(total_full_image_pixels),
        "positive_pixel_counts": pos_counts,
        "negative_pixel_counts": neg_counts,
        "prevalence_by_class": {
            c: round(float(prevalence_by_class[c]), 8) for c in channels
        },
        "neg_to_pos_ratio_by_class": {
            c: (round(float(neg_to_pos_ratio_by_class[c]), 4) if neg_to_pos_ratio_by_class[c] is not None else None)
            for c in channels
        },
        "present_image_counts": present_image_counts,
        "absent_image_counts": absent_image_counts,
        "zero_positive_channels": zero_positive_channels,
        "clipped_channels": clipped_channels,
        "fov_accounting_note": (
            "Training loss operates on rectangular tensors (FOV-cropped bounding box when "
            "fov_crop=True, or full image when fov_crop=False) without applying a circular "
            "FOV mask inside the loss. Manifest metadata records rectangular fov_crop bounds "
            "and full image_dimensions, not exact circular retinal-mask pixel counts. "
            "Accordingly, negative_pixel_counts and derived weights reflect the rectangular "
            f"'{domain_norm}' training tensor domain rather than exact circular retinal-pixel ratios."
        ),
    }


def summarize_loss_configuration(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a structured summary of the configured loss and weight provenance for experiment reports.

    Explicitly distinguishes ``manual`` weights configured in YAML/CLI from
    ``derived_from_train_metadata`` weights and ``none`` (unweighted baseline).
    """
    loss_cfg = config.get("loss", {}) if isinstance(config, Mapping) else {}
    if not isinstance(loss_cfg, Mapping):
        loss_cfg = {}

    loss_type = str(loss_cfg.get("type", "bce_dice")).strip().lower()
    pos_weight = loss_cfg.get("pos_weight", None)
    class_weights = loss_cfg.get("class_weights", None)

    pos_weight_source = str(
        loss_cfg.get(
            "pos_weight_source",
            "manual" if pos_weight is not None else "none",
        )
    )
    class_weights_source = str(
        loss_cfg.get(
            "class_weights_source",
            "manual" if class_weights is not None else "none",
        )
    )

    raw_gamma = loss_cfg.get("gamma", None)
    raw_eps = loss_cfg.get("eps", None)

    return {
        "type": loss_type,
        "bce_weight": float(loss_cfg.get("bce_weight", 0.5)),
        "dice_weight": float(loss_cfg.get("dice_weight", 0.5)),
        "tversky_weight": float(
            loss_cfg.get("tversky_weight", loss_cfg.get("dice_weight", 0.5))
        ),
        "focal_tversky_weight": float(
            loss_cfg.get(
                "focal_tversky_weight",
                loss_cfg.get("tversky_weight", loss_cfg.get("dice_weight", 0.5)),
            )
        ),
        "alpha": float(loss_cfg.get("alpha", 0.3)),
        "beta": float(loss_cfg.get("beta", 0.7)),
        "gamma": float(4.0 / 3.0 if raw_gamma is None else raw_gamma),
        "eps": float(1e-6 if raw_eps is None else raw_eps),
        "smooth": float(loss_cfg.get("smooth", 1e-5)),
        "pos_weight": list(pos_weight) if pos_weight is not None else None,
        "pos_weight_source": pos_weight_source,
        "class_weights": list(class_weights) if class_weights is not None else None,
        "class_weights_source": class_weights_source,
    }


def build_loss_from_config(config: dict[str, Any]) -> nn.Module:
    """Construct a segmentation loss module from ``configs/model.yaml``."""
    loss_cfg = config.get("loss", {})
    if not isinstance(loss_cfg, Mapping):
        loss_cfg = {}

    loss_type = str(loss_cfg.get("type", "bce_dice")).strip().lower()
    if loss_type not in SUPPORTED_LOSS_TYPES:
        raise ValueError(
            f"Unsupported loss.type {loss_type!r}; supported types are {SUPPORTED_LOSS_TYPES}."
        )

    for w_field in ("bce_weight", "dice_weight", "tversky_weight", "focal_tversky_weight"):
        if w_field in loss_cfg and not _is_valid_non_negative_number(loss_cfg[w_field]):
            raise ValueError(
                f"loss.{w_field} must be a finite number >= 0, got {loss_cfg[w_field]!r}."
            )

    smooth = float(loss_cfg.get("smooth", 1e-5))
    raw_eps = loss_cfg.get("eps", None)
    if raw_eps is None:
        eps = 1e-6
    else:
        if not _is_valid_positive_number(raw_eps):
            raise ValueError(f"loss.eps must be a finite positive number > 0, got {raw_eps!r}.")
        eps = float(raw_eps)

    raw_gamma = loss_cfg.get("gamma", None)
    if raw_gamma is None:
        gamma = 4.0 / 3.0
    else:
        if not _is_valid_positive_number(raw_gamma):
            raise ValueError(f"loss.gamma must be a finite positive number > 0, got {raw_gamma!r}.")
        gamma = float(raw_gamma)

    pos_weight = loss_cfg.get("pos_weight", None)
    class_weights = loss_cfg.get("class_weights", None)

    channel_order = config.get("output_channel_order", list(CANONICAL_LESION_ORDER))
    channel_names = tuple(str(ch) for ch in channel_order)

    if loss_type in {"bce", "weighted_bce"}:
        bce_weight = float(loss_cfg.get("bce_weight", 1.0))
        return WeightedBCELoss(
            pos_weight=pos_weight,
            bce_weight=bce_weight,
            channel_names=channel_names,
        )

    if loss_type in {"dice", "soft_dice"}:
        if "dice_weight" in loss_cfg and not _is_valid_positive_number(loss_cfg["dice_weight"]):
            raise ValueError(
                f"loss.dice_weight must be a finite number > 0 for loss.type={loss_type!r}, "
                f"got {loss_cfg['dice_weight']!r}."
            )
        batch_dice = bool(loss_cfg.get("batch_dice", True))
        return SoftDiceLoss(
            smooth=smooth,
            batch_dice=batch_dice,
            class_weights=class_weights,
            channel_names=channel_names,
        )

    if loss_type == "balanced_dice":
        if "dice_weight" in loss_cfg and not _is_valid_positive_number(loss_cfg["dice_weight"]):
            raise ValueError(
                f"loss.dice_weight must be a finite number > 0 for loss.type='balanced_dice', "
                f"got {loss_cfg['dice_weight']!r}."
            )
        batch_dice = bool(loss_cfg.get("batch_dice", True))
        return BalancedDiceLoss(
            class_weights=class_weights,
            smooth=smooth,
            batch_dice=batch_dice,
            channel_names=channel_names,
        )

    if loss_type == "tversky":
        if "tversky_weight" in loss_cfg and not _is_valid_positive_number(loss_cfg["tversky_weight"]):
            raise ValueError(
                f"loss.tversky_weight must be a finite number > 0 for loss.type='tversky', "
                f"got {loss_cfg['tversky_weight']!r}."
            )
        alpha = float(loss_cfg.get("alpha", 0.3))
        beta = float(loss_cfg.get("beta", 0.7))
        batch_tversky = bool(loss_cfg.get("batch_tversky", loss_cfg.get("batch_dice", True)))
        return TverskyLoss(
            alpha=alpha,
            beta=beta,
            smooth=smooth,
            batch_tversky=batch_tversky,
            class_weights=class_weights,
            channel_names=channel_names,
        )

    if loss_type == "focal_tversky":
        if "focal_tversky_weight" in loss_cfg and not _is_valid_positive_number(
            loss_cfg["focal_tversky_weight"]
        ):
            raise ValueError(
                f"loss.focal_tversky_weight must be a finite number > 0 for loss.type='focal_tversky', "
                f"got {loss_cfg['focal_tversky_weight']!r}."
            )
        alpha = float(loss_cfg.get("alpha", 0.3))
        beta = float(loss_cfg.get("beta", 0.7))
        batch_tversky = bool(loss_cfg.get("batch_tversky", loss_cfg.get("batch_dice", True)))
        return FocalTverskyLoss(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            smooth=smooth,
            batch_tversky=batch_tversky,
            class_weights=class_weights,
            channel_names=channel_names,
            eps=eps,
        )

    if loss_type in {"bce_focal_tversky", "weighted_bce_focal_tversky"}:
        bce_weight = float(loss_cfg.get("bce_weight", 0.5))
        ft_weight = float(
            loss_cfg.get(
                "focal_tversky_weight",
                loss_cfg.get("tversky_weight", loss_cfg.get("dice_weight", 0.5)),
            )
        )
        alpha = float(loss_cfg.get("alpha", 0.3))
        beta = float(loss_cfg.get("beta", 0.7))
        batch_tversky = bool(loss_cfg.get("batch_tversky", loss_cfg.get("batch_dice", True)))
        return BCEFocalTverskyLoss(
            bce_weight=bce_weight,
            focal_tversky_weight=ft_weight,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            smooth=smooth,
            pos_weight=pos_weight,
            batch_tversky=batch_tversky,
            class_weights=class_weights,
            channel_names=channel_names,
            eps=eps,
        )

    if loss_type in {"bce_tversky", "weighted_bce_tversky"}:
        bce_weight = float(loss_cfg.get("bce_weight", 0.5))
        tversky_weight = float(loss_cfg.get("tversky_weight", loss_cfg.get("dice_weight", 0.5)))
        alpha = float(loss_cfg.get("alpha", 0.3))
        beta = float(loss_cfg.get("beta", 0.7))
        batch_tversky = bool(loss_cfg.get("batch_tversky", loss_cfg.get("batch_dice", True)))
        return BCETverskyLoss(
            bce_weight=bce_weight,
            tversky_weight=tversky_weight,
            alpha=alpha,
            beta=beta,
            smooth=smooth,
            pos_weight=pos_weight,
            batch_tversky=batch_tversky,
            class_weights=class_weights,
            channel_names=channel_names,
        )

    # Default: "bce_dice", "weighted_bce_dice", "bce_balanced_dice"
    bce_weight = float(loss_cfg.get("bce_weight", 0.5))
    dice_weight = float(loss_cfg.get("dice_weight", 0.5))
    batch_dice = bool(loss_cfg.get("batch_dice", True))
    criterion = BCEDiceLoss(
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        smooth=smooth,
        pos_weight=pos_weight,
        batch_dice=batch_dice,
        class_weights=class_weights,
        channel_names=channel_names,
    )
    if loss_type == "bce_balanced_dice":
        criterion.secondary_component_name = "balanced_dice"
        criterion.dice_loss_fn = BalancedDiceLoss(
            class_weights=class_weights,
            smooth=smooth,
            batch_dice=batch_dice,
            channel_names=channel_names,
        )
    return criterion


def _validate_contrastive_tensors(
    z1: torch.Tensor,
    z2: torch.Tensor,
) -> tuple[int, int]:
    """Validate two-view projection tensors ``(B, D)`` without GPU-CPU synchronization."""
    if not isinstance(z1, torch.Tensor) or not isinstance(z2, torch.Tensor):
        raise TypeError("Both z1 and z2 must be torch.Tensor instances.")
    if z1.ndim != 2 or z2.ndim != 2:
        raise ValueError(
            f"Expected 2D projection tensors of shape (B, D), got z1={tuple(z1.shape)} "
            f"and z2={tuple(z2.shape)}."
        )
    if z1.shape != z2.shape:
        raise ValueError(
            f"Shape mismatch between z1 {tuple(z1.shape)} and z2 {tuple(z2.shape)}."
        )
    batch_size, proj_dim = int(z1.shape[0]), int(z1.shape[1])
    if proj_dim < 1:
        raise ValueError(f"Projection dimension D must be >= 1, got {proj_dim}.")
    if batch_size < 1:
        raise ValueError(f"Batch size B must be >= 1, got {batch_size}.")
    if batch_size == 1:
        raise ValueError(
            "NTXentLoss requires batch_size >= 2 to form negative pairs when "
            "self-similarity is excluded (got batch_size=1)."
        )
    return batch_size, proj_dim


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross-Entropy (NT-Xent) loss for SimCLR.

    Given projected representations ``z1, z2`` of shape ``(B, D)`` from two
    independently augmented views of the same batch of ``B`` images:

    1. L2-normalizes ``z1`` and ``z2`` along feature dimension ``D`` (when
       ``normalize=True``).
    2. Concatenates both views into ``2B`` representations and computes the
       all-pairs cosine similarity matrix divided by ``temperature``.
    3. Excludes self-similarity ``(k, k)`` on the main diagonal so each of the
       ``2B`` anchors contrasts its single matching positive view against the
       ``2B - 2`` negative views in the batch.
    4. Computes symmetric cross-entropy across all ``2B`` anchors without any
       CPU-GPU synchronization.
    """

    def __init__(
        self,
        temperature: float = 0.5,
        normalize: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0.0
        ):
            raise ValueError(
                f"temperature must be a finite number > 0, got {temperature!r}."
            )
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps!r}.")

        self.temperature = float(temperature)
        self.normalize = bool(normalize)
        self.eps = float(eps)

    def forward_with_metrics(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute NT-Xent loss and detached on-device contrastive alignment metrics."""
        batch_size, _ = _validate_contrastive_tensors(z1, z2)

        z1_fp32 = z1.float()
        z2_fp32 = z2.float()
        if self.normalize:
            z1_fp32 = F.normalize(z1_fp32, p=2.0, dim=1, eps=self.eps)
            z2_fp32 = F.normalize(z2_fp32, p=2.0, dim=1, eps=self.eps)

        representations = torch.cat([z1_fp32, z2_fp32], dim=0)  # (2B, D)
        sim_matrix = torch.matmul(representations, representations.T)  # (2B, 2B)

        n = 2 * batch_size
        device = representations.device

        # Exclude self-similarity (main diagonal) cleanly by selecting the (2B - 1) off-diagonal columns per row
        off_diag_mask = ~torch.eye(n, dtype=torch.bool, device=device)
        sim_off_diag = sim_matrix.masked_select(off_diag_mask).view(n, n - 1)
        logits = sim_off_diag / self.temperature

        # In row i in [0, B-1], positive partner (i + B) is at column (i + B - 1) after removing diagonal column i.
        # In row i in [B, 2B-1], positive partner (i - B) is at column (i - B) since removed diagonal column i > i - B.
        pos_targets = torch.cat(
            [
                torch.arange(
                    batch_size - 1,
                    2 * batch_size - 1,
                    device=device,
                    dtype=torch.long,
                ),
                torch.arange(0, batch_size, device=device, dtype=torch.long),
            ],
            dim=0,
        )

        loss = F.cross_entropy(logits, pos_targets)

        with torch.no_grad():
            row_idx = torch.arange(n, device=device)
            pos_sim = sim_off_diag[row_idx, pos_targets].mean()
            neg_sum = sim_off_diag.sum(dim=1) - sim_off_diag[row_idx, pos_targets]
            neg_sim = (neg_sum / float(n - 2)).mean()
            top1_acc = (logits.argmax(dim=1) == pos_targets).float().mean()

        return {
            "loss": loss,
            "positive_similarity": pos_sim,
            "negative_similarity": neg_sim,
            "contrastive_accuracy": top1_acc,
            "logits": logits.detach(),
            "targets": pos_targets,
        }

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scalar NT-Xent contrastive loss for projected views ``z1`` and ``z2``."""
        return self.forward_with_metrics(z1, z2)["loss"]


def build_nt_xent_loss_from_config(
    config: Mapping[str, Any] | str | Path = "configs/ssl.yaml",
) -> NTXentLoss:
    """Construct an :class:`NTXentLoss` from ``configs/ssl.yaml`` or ``configs/model.yaml``."""
    if isinstance(config, (str, Path)):
        cfg = load_yaml_config(config, validate=False)
    else:
        cfg = dict(config)

    if isinstance(cfg.get("simclr"), Mapping):
        cfg = dict(cfg["simclr"])

    temperature = float(cfg.get("temperature", 0.5))
    normalize = bool(cfg.get("normalize_projections", True))
    return NTXentLoss(temperature=temperature, normalize=normalize)


__all__ = [
    "BCEDiceLoss",
    "BCEFocalTverskyLoss",
    "BCETverskyLoss",
    "BalancedDiceLoss",
    "CANONICAL_LESION_ORDER",
    "FocalTverskyLoss",
    "NTXentLoss",
    "SoftDiceLoss",
    "TrainingWeightDerivationError",
    "TverskyLoss",
    "WeightedBCELoss",
    "build_loss_from_config",
    "build_nt_xent_loss_from_config",
    "derive_training_class_weights",
    "extract_loss_components",
    "extract_loss_components_with_per_channel",
    "summarize_loss_configuration",
]

