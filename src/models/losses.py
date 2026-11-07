"""Multilabel retinal lesion segmentation loss functions.

Implements a numerically stable, configurable weighted combination of
Binary Cross-Entropy with Logits (BCEWithLogitsLoss) and per-channel
Soft Dice Loss for four independent binary lesion channels:
``[MA, HE, EX, SE]``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.masks import CANONICAL_LESION_CHANNELS
from src.utils.config import load_yaml_config

CANONICAL_LESION_ORDER: tuple[str, ...] = tuple(CANONICAL_LESION_CHANNELS)


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
            across samples and channels. Default is False so every sample's
            empty or positive channel behavior is preserved independently.
        channel_names: Ordered tuple of channel names.
    """

    def __init__(
        self,
        smooth: float = 1e-5,
        batch_dice: bool = True,
        channel_names: tuple[str, ...] = CANONICAL_LESION_ORDER,
    ) -> None:
        super().__init__()
        if smooth <= 0.0:
            raise ValueError(f"smooth must be > 0, got {smooth}")
        self.smooth = float(smooth)
        self.batch_dice = bool(batch_dice)
        self.channel_names = tuple(channel_names)

    def forward_per_channel(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-channel Soft Dice loss tensor of shape ``(C,)``."""
        _validate_segmentation_tensors(logits, targets)
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

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scalar mean Soft Dice loss across all channels."""
        channel_loss = self.forward_per_channel(logits, targets)
        return channel_loss.mean()


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
        pos_weight: Optional per-channel positive class weights of length ``C``.
        batch_dice: Whether ``SoftDiceLoss`` aggregates over the batch axis.
        channel_names: Ordered tuple of channel names.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-5,
        pos_weight: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        batch_dice: bool = True,
        channel_names: tuple[str, ...] = CANONICAL_LESION_ORDER,
    ) -> None:
        super().__init__()
        if bce_weight < 0.0:
            raise ValueError(f"bce_weight must be >= 0, got {bce_weight}")
        if dice_weight < 0.0:
            raise ValueError(f"dice_weight must be >= 0, got {dice_weight}")
        if bce_weight == 0.0 and dice_weight == 0.0:
            raise ValueError("At least one of bce_weight or dice_weight must be > 0.")
        if smooth <= 0.0:
            raise ValueError(f"smooth must be > 0, got {smooth}")

        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)
        self.channel_names = tuple(channel_names)

        if pos_weight is not None:
            pos_weight_tensor = torch.as_tensor(pos_weight, dtype=torch.float32)
            if pos_weight_tensor.ndim == 1:
                pos_weight_tensor = pos_weight_tensor.view(1, -1, 1, 1)
            self.register_buffer("pos_weight", pos_weight_tensor)
        else:
            self.pos_weight = None

        self.dice_loss_fn = SoftDiceLoss(
            smooth=self.smooth,
            batch_dice=batch_dice,
            channel_names=self.channel_names,
        )

    def forward_tensors(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute total loss and per-channel BCE/Dice tensors without CPU synchronization."""
        _validate_segmentation_tensors(logits, targets)
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

        dice_per_channel = self.dice_loss_fn.forward_per_channel(
            logits_fp32, targets_fp32
        )
        dice_loss = dice_per_channel.mean()

        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return total_loss, bce_loss, dice_loss, bce_per_channel, dice_per_channel

    def compute_components(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, Any]:
        """Compute total loss along with detached BCE, Dice, and per-channel components."""
        (
            total_loss,
            bce_loss,
            dice_loss,
            bce_per_channel,
            dice_per_channel,
        ) = self.forward_tensors(logits, targets)
        per_channel_total = (
            self.bce_weight * bce_per_channel + self.dice_weight * dice_per_channel
        )

        num_channels = logits.shape[1]
        names = (
            self.channel_names
            if len(self.channel_names) == num_channels
            else tuple(f"channel_{i}" for i in range(num_channels))
        )

        return {
            "loss": total_loss,
            "bce_loss": bce_loss.detach(),
            "dice_loss": dice_loss.detach(),
            "per_channel_bce": {
                name: float(bce_per_channel[i].detach().item())
                for i, name in enumerate(names)
            },
            "per_channel_dice": {
                name: float(dice_per_channel[i].detach().item())
                for i, name in enumerate(names)
            },
            "per_channel_loss": {
                name: float(per_channel_total[i].detach().item())
                for i, name in enumerate(names)
            },
        }

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scalar weighted BCE + Soft Dice loss."""
        return self.forward_tensors(logits, targets)[0]


def build_loss_from_config(config: dict[str, Any]) -> BCEDiceLoss:
    """Construct a :class:`BCEDiceLoss` from the ``model.yaml`` configuration dict."""
    loss_cfg = config.get("loss", {})
    if not isinstance(loss_cfg, dict):
        loss_cfg = {}

    bce_weight = float(loss_cfg.get("bce_weight", 0.5))
    dice_weight = float(loss_cfg.get("dice_weight", 0.5))
    smooth = float(loss_cfg.get("smooth", 1e-5))
    batch_dice = bool(loss_cfg.get("batch_dice", True))
    pos_weight = loss_cfg.get("pos_weight", None)

    channel_order = config.get("output_channel_order", list(CANONICAL_LESION_ORDER))
    channel_names = tuple(str(ch) for ch in channel_order)

    return BCEDiceLoss(
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        smooth=smooth,
        pos_weight=pos_weight,
        batch_dice=batch_dice,
        channel_names=channel_names,
    )


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
    "CANONICAL_LESION_ORDER",
    "NTXentLoss",
    "SoftDiceLoss",
    "build_loss_from_config",
    "build_nt_xent_loss_from_config",
]

