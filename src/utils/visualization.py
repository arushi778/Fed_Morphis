"""Qualitative segmentation visualization utilities for multi-lesion retinal fundus images.

Provides deterministic, self-contained rendering of comparative 4-panel figures:
1. Original retinal fundus image (un-normalized RGB).
2. Canonical 4-channel ground-truth lesion masks.
3. Canonical 4-channel predicted lesion masks.
4. Comparative overlay on the retinal background (ground-truth semi-transparent fill
   with predicted boundary contour outlines).

Features:
- Distinct, consistent color-coding across the canonical lesion channels:
  - MA (Microaneurysms): Vivid Red
  - HE (Hemorrhages): Lime Green
  - EX (Hard Exudates): Bright Yellow / Gold
  - SE (Soft Exudates): Bright Cyan / Turquoise
- Informative title banners displaying dataset name, sample ID, selection category,
  and sample-level macro and per-lesion Dice scores.
- Legend explaining color mappings and outline vs. fill semantics.
- Non-cherry-picked representative sample selection across performance percentiles.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.data.masks import CANONICAL_LESION_CHANNELS

# Canonical lesion channel display colors (RGB uint8)
CANONICAL_LESION_COLORS: dict[str, tuple[int, int, int]] = {
    "MA": (255, 50, 50),     # Vivid Red
    "HE": (50, 205, 50),     # Lime Green
    "EX": (255, 215, 0),     # Gold / Bright Yellow
    "SE": (0, 206, 209),     # Dark Turquoise / Vivid Cyan
}

DEFAULT_IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
DEFAULT_IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def unnormalize_image(
    image: torch.Tensor | np.ndarray,
    mean: Sequence[float] = DEFAULT_IMAGENET_MEAN,
    std: Sequence[float] = DEFAULT_IMAGENET_STD,
) -> np.ndarray:
    """Reverse ImageNet standardization and return an (H, W, 3) uint8 RGB image.

    Args:
        image: Array or tensor of shape (3, H, W) or (H, W, 3).
        mean: Channel means (length 3).
        std: Channel standard deviations (length 3).

    Returns:
        (H, W, 3) uint8 NumPy array in RGB order clipped to [0, 255].
    """
    if isinstance(image, torch.Tensor):
        img_np = image.detach().cpu().float().numpy()
    else:
        img_np = np.asarray(image, dtype=np.float32)

    if img_np.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape {img_np.shape}.")

    # Normalize to (3, H, W)
    if img_np.shape[0] == 3 and img_np.shape[2] != 3:
        pass
    elif img_np.shape[2] == 3:
        img_np = np.transpose(img_np, (2, 0, 1))
    else:
        raise ValueError(f"Image must have 3 channels, got shape {img_np.shape}.")

    mean_arr = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
    std_arr = np.array(std, dtype=np.float32).reshape(3, 1, 1)

    unnorm = (img_np * std_arr + mean_arr) * 255.0
    unnorm = np.clip(unnorm, 0.0, 255.0).astype(np.uint8)
    return np.transpose(unnorm, (1, 2, 0))


def create_colored_mask(
    mask_4ch: torch.Tensor | np.ndarray,
    colors: Mapping[str, tuple[int, int, int]] | None = None,
    channel_order: Sequence[str] = CANONICAL_LESION_CHANNELS,
    background_color: tuple[int, int, int] = (15, 15, 20),
) -> np.ndarray:
    """Render a 4-channel binary lesion mask into an RGB composite image.

    Args:
        mask_4ch: Binary mask of shape (4, H, W) or (H, W, 4).
        colors: Color mapping {channel_name: (R, G, B)}.
        channel_order: Canonical order of channels in ``mask_4ch``.
        background_color: RGB background for non-lesion pixels.

    Returns:
        (H, W, 3) uint8 NumPy array in RGB order.
    """
    if colors is None:
        colors = CANONICAL_LESION_COLORS

    if isinstance(mask_4ch, torch.Tensor):
        mask_np = mask_4ch.detach().cpu().numpy()
    else:
        mask_np = np.asarray(mask_4ch)

    if mask_np.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {mask_np.shape}.")

    if mask_np.shape[0] == len(channel_order) and mask_np.shape[2] != len(channel_order):
        pass
    elif mask_np.shape[2] == len(channel_order):
        mask_np = np.transpose(mask_np, (2, 0, 1))
    else:
        raise ValueError(
            f"Mask channel count mismatch: shape={mask_np.shape}, expected {len(channel_order)} channels."
        )

    _, h, w = mask_np.shape
    canvas = np.full((h, w, 3), fill_value=background_color, dtype=np.uint8)

    for ch_idx, ch_name in enumerate(channel_order):
        ch_bin = (mask_np[ch_idx] > 0).astype(np.uint8)
        if not np.any(ch_bin):
            continue
        color_rgb = np.array(colors.get(ch_name, (255, 255, 255)), dtype=np.uint8)
        # Apply color where mask is active
        canvas[ch_bin > 0] = color_rgb

    return canvas


def create_overlay(
    image_rgb: np.ndarray,
    gt_mask_4ch: torch.Tensor | np.ndarray,
    pred_mask_4ch: torch.Tensor | np.ndarray,
    colors: Mapping[str, tuple[int, int, int]] | None = None,
    channel_order: Sequence[str] = CANONICAL_LESION_CHANNELS,
    alpha: float = 0.35,
    contour_thickness: int = 2,
) -> np.ndarray:
    """Overlay ground truth (semi-transparent fill) and prediction (contour outline) on the RGB image.

    Ground truth regions are filled with a semi-transparent tint in the channel's
    canonical color. Predicted regions are outlined with a solid contour in the
    same color. This makes true positive agreements (filled region framed by outline),
    false positives (outline only), and false negatives (fill only) immediately
    distinguishable on the anatomical background.

    Args:
        image_rgb: (H, W, 3) uint8 RGB image.
        gt_mask_4ch: (4, H, W) or (H, W, 4) binary ground-truth mask.
        pred_mask_4ch: (4, H, W) or (H, W, 4) binary predicted mask.
        colors: Color mapping {channel_name: (R, G, B)}.
        channel_order: Canonical order of channels.
        alpha: Transparency of the ground-truth shaded fill.
        contour_thickness: Thickness of predicted boundary contour.

    Returns:
        (H, W, 3) uint8 RGB image with composite overlay.
    """
    if colors is None:
        colors = CANONICAL_LESION_COLORS

    if isinstance(gt_mask_4ch, torch.Tensor):
        gt_np = gt_mask_4ch.detach().cpu().numpy()
    else:
        gt_np = np.asarray(gt_mask_4ch)

    if isinstance(pred_mask_4ch, torch.Tensor):
        pred_np = pred_mask_4ch.detach().cpu().numpy()
    else:
        pred_np = np.asarray(pred_mask_4ch)

    if gt_np.shape[2] == len(channel_order):
        gt_np = np.transpose(gt_np, (2, 0, 1))
    if pred_np.shape[2] == len(channel_order):
        pred_np = np.transpose(pred_np, (2, 0, 1))

    h, w, _ = image_rgb.shape
    base = image_rgb.copy()

    # Step 1: Apply semi-transparent fill for Ground Truth masks
    gt_tint_layer = np.zeros_like(base)
    gt_active_any = np.zeros((h, w), dtype=bool)

    for ch_idx, ch_name in enumerate(channel_order):
        ch_gt = (gt_np[ch_idx] > 0).astype(np.uint8)
        if not np.any(ch_gt):
            continue
        color_rgb = np.array(colors.get(ch_name, (255, 255, 255)), dtype=np.uint8)
        gt_tint_layer[ch_gt > 0] = color_rgb
        gt_active_any |= (ch_gt > 0)

    if np.any(gt_active_any):
        blended = cv2.addWeighted(base, 1.0 - alpha, gt_tint_layer, alpha, 0)
        base[gt_active_any] = blended[gt_active_any]

    # Step 2: Draw solid boundary contours for Predicted masks
    for ch_idx, ch_name in enumerate(channel_order):
        ch_pred = (pred_np[ch_idx] > 0).astype(np.uint8)
        if not np.any(ch_pred):
            continue
        color_rgb = tuple(int(c) for c in colors.get(ch_name, (255, 255, 255)))
        contours, _ = cv2.findContours(
            ch_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(base, contours, -1, color_rgb, contour_thickness)

    return base


def render_segmentation_comparison_panel(
    image: torch.Tensor | np.ndarray,
    gt_mask: torch.Tensor | np.ndarray,
    pred_mask: torch.Tensor | np.ndarray,
    metadata: Mapping[str, Any],
    metrics: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
    colors: Mapping[str, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Render a professional 4-panel comparative visualization with header and legend.

    Panels (left to right):
    1. Original Retinal Image (un-normalized RGB).
    2. Ground Truth Lesion Masks (color-coded by channel).
    3. Predicted Lesion Masks (color-coded by channel).
    4. Comparative Overlay (GT Shaded Fill + Pred Contour Outlines on Image).

    Args:
        image: Normalized or un-normalized RGB image tensor or array.
        gt_mask: 4-channel binary ground-truth mask tensor or array.
        pred_mask: 4-channel binary predicted mask tensor or array.
        metadata: Sample metadata dict (sample_id, dataset_name, category, etc.).
        metrics: Optional per-sample metrics dict (macro_dice, per-class dice).
        output_path: Optional path to save the resulting PNG image.
        colors: Color mapping for lesion channels.

    Returns:
        Complete composite RGB image as uint8 NumPy array.
    """
    if colors is None:
        colors = CANONICAL_LESION_COLORS

    rgb_image = unnormalize_image(image)
    gt_colored = create_colored_mask(gt_mask, colors=colors)
    pred_colored = create_colored_mask(pred_mask, colors=colors)
    overlay = create_overlay(rgb_image, gt_mask, pred_mask, colors=colors)

    panel_h, panel_w, _ = rgb_image.shape

    # Panel titles
    panel_titles = [
        "1. Original Fundus Image",
        "2. Ground Truth Masks",
        "3. Predicted Masks",
        "4. Overlay (GT Fill + Pred Outline)",
    ]

    panels: list[np.ndarray] = [rgb_image, gt_colored, pred_colored, overlay]
    labeled_panels: list[np.ndarray] = []

    title_bar_h = 32
    for p_img, p_title in zip(panels, panel_titles):
        p_copy = p_img.copy()
        # Top banner for panel title
        title_bar = np.full((title_bar_h, panel_w, 3), fill_value=(25, 25, 30), dtype=np.uint8)
        cv2.putText(
            title_bar,
            p_title,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        combined_panel = np.vstack([title_bar, p_copy])
        labeled_panels.append(combined_panel)

    grid = np.hstack(labeled_panels)
    grid_h, grid_w, _ = grid.shape

    # Top overall header banner
    header_h = 70
    header = np.full((header_h, grid_w, 3), fill_value=(18, 18, 22), dtype=np.uint8)

    sample_id = str(metadata.get("sample_id", "unknown"))
    dataset_name = str(metadata.get("dataset_name", metadata.get("dataset", "Unknown")))
    category = str(metadata.get("category", "Representative Evaluation Sample"))

    line1 = f"FedMorphis Flower FedAvg (Round 17) Held-Out Test Evaluation | Dataset: {dataset_name} | Sample ID: {sample_id} | Category: {category}"
    cv2.putText(
        header,
        line1,
        (15, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    macro_dice_str = "N/A"
    pc_str = ""
    if metrics:
        md = metrics.get("dice_macro", metrics.get("macro_dice"))
        if md is not None:
            macro_dice_str = f"{float(md):.4f}"
        pc = metrics.get("per_class", {})
        pc_items = []
        for ch in CANONICAL_LESION_CHANNELS:
            ch_val = pc.get(ch, {}).get("dice") if isinstance(pc.get(ch), Mapping) else pc.get(ch)
            if ch_val is not None:
                pc_items.append(f"{ch}={float(ch_val):.4f}")
            else:
                pc_items.append(f"{ch}=N/A")
        pc_str = " | ".join(pc_items)

    line2 = f"Sample Macro Dice: {macro_dice_str} | Per-Lesion Dice: {pc_str}"
    cv2.putText(
        header,
        line2,
        (15, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 220, 255),
        1,
        cv2.LINE_AA,
    )

    # Bottom legend banner
    footer_h = 44
    footer = np.full((footer_h, grid_w, 3), fill_value=(18, 18, 22), dtype=np.uint8)

    # Draw color patches and channel names
    legend_x = 15
    legend_y = 26
    patch_size = 14

    cv2.putText(
        footer,
        "Lesion Channels:",
        (legend_x, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    legend_x += 135

    for ch in CANONICAL_LESION_CHANNELS:
        c_rgb = colors[ch]
        # Draw small square patch
        cv2.rectangle(
            footer,
            (legend_x, legend_y - patch_size + 2),
            (legend_x + patch_size, legend_y + 2),
            c_rgb,
            -1,
        )
        legend_x += patch_size + 6
        cv2.putText(
            footer,
            f"{ch}",
            (legend_x, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        legend_x += 48

    legend_x += 30
    cv2.putText(
        footer,
        "Representation: Ground Truth = Semi-Transparent Fill  |  Prediction = Solid Contour Outline",
        (legend_x, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    composite = np.vstack([header, grid, footer])

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        # cv2 expects BGR for imwrite
        composite_bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_p), composite_bgr)

    return composite


def select_representative_samples(
    scored_samples: Sequence[dict[str, Any]],
    num_samples: int = 4,
) -> list[dict[str, Any]]:
    """Select representative, non-cherry-picked test samples across performance percentiles.

    Selection strategy:
    1. Filter samples that have at least one lesion present (non-empty ground truth)
       so the segmentation task is non-trivial.
    2. Sort the samples by macro Dice.
    3. Select:
       - High Performer: ~85th-95th percentile
       - Median Performer: ~50th percentile (representative median case)
       - Challenging Case: ~20th-30th percentile (difficult boundaries or small lesions)
       - Multi-Lesion Case: Case with the maximum number of simultaneous active lesion channels.

    Args:
        scored_samples: Sequence of sample dicts containing 'sample_id', 'dice_macro',
                        'present_classes', and 'per_class'.
        num_samples: Number of samples to select (default 4).

    Returns:
        List of selected sample dicts with an added 'category' field.
    """
    if not scored_samples:
        return []

    # Prefer samples with lesions present
    with_lesions = [
        s for s in scored_samples if len(s.get("present_classes", [])) > 0
    ]
    pool = with_lesions if len(with_lesions) >= num_samples else list(scored_samples)

    # Sort ascending by macro Dice
    sorted_by_dice = sorted(
        pool, key=lambda s: float(s.get("dice_macro", 0.0))
    )
    n = len(sorted_by_dice)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    # 1. Challenging case (~20-25th percentile)
    idx_challenging = max(0, int(n * 0.25))
    s_challenging = dict(sorted_by_dice[idx_challenging])
    s_challenging["category"] = "Challenging / Low-Contrast Lesions"
    selected.append(s_challenging)
    selected_ids.add(str(s_challenging["sample_id"]))

    # 2. Median case (~50th percentile)
    idx_median = min(n - 1, max(0, int(n * 0.50)))
    s_median = dict(sorted_by_dice[idx_median])
    if str(s_median["sample_id"]) not in selected_ids:
        s_median["category"] = "Median / Representative Performance"
        selected.append(s_median)
        selected_ids.add(str(s_median["sample_id"]))

    # 3. High performer (~90th percentile)
    idx_high = min(n - 1, max(0, int(n * 0.90)))
    s_high = dict(sorted_by_dice[idx_high])
    if str(s_high["sample_id"]) not in selected_ids:
        s_high["category"] = "High Performance"
        selected.append(s_high)
        selected_ids.add(str(s_high["sample_id"]))

    # 4. Multi-lesion case (highest count of present lesions)
    sorted_by_presence = sorted(
        pool,
        key=lambda s: (
            len(s.get("present_classes", [])),
            float(s.get("dice_macro", 0.0)),
        ),
        reverse=True,
    )
    for candidate in sorted_by_presence:
        cid = str(candidate["sample_id"])
        if cid not in selected_ids:
            s_multi = dict(candidate)
            s_multi["category"] = "Complex Multi-Lesion Pathology"
            selected.append(s_multi)
            selected_ids.add(cid)
            break

    # If still fewer than num_samples, backfill with remaining distinct samples
    for item in sorted_by_dice:
        if len(selected) >= num_samples:
            break
        if str(item["sample_id"]) not in selected_ids:
            s_fill = dict(item)
            s_fill["category"] = "Representative Evaluation Sample"
            selected.append(s_fill)
            selected_ids.add(str(item["sample_id"]))

    return selected

