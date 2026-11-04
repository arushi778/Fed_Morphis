"""Canonical four-channel binary lesion mask construction for DDR and IDRiD.

Constructs independent binary lesion masks of shape ``(H, W, 4)`` in the
canonical channel order ``[MA, HE, EX, SE]`` with binary values ``{0, 1}``
(``np.uint8``).

Key design guarantees:
- All four lesion channels (``MA``, ``HE``, ``EX``, ``SE``) remain independent
  so multi-label spatial overlaps are preserved.
- ``OD`` (Optic Disc) is explicitly excluded from the canonical lesion tensor.
- Omitted lesion mask files (e.g., absent ``HE`` or ``SE`` masks in IDRiD) are
  represented as synthesized all-zero channels of matching ``(H, W)`` dimensions.
- Explicit all-zero mask files on disk (e.g., DDR absent-lesion ``.tif`` masks)
  are distinguished from missing mask files in the returned audit metadata.
- Single-channel grayscale ``(H, W)``, ``(H, W, 1)``, 3-channel RGB ``(H, W, 3)``,
  and 4-channel RGBA ``(H, W, 4)`` masks (including IDRiD's RGBA ``IDRiD_81_EX.tif``)
  are inspected per channel before binarization so opaque alpha channels never
  corrupt foreground extraction.
- Source pixel encodings ``{0, 1}`` and ``{0, 255}`` are deterministically mapped
  to ``{0, 1}`` via ``(channel > threshold).astype(np.uint8)`` with default
  ``threshold = 0``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

CANONICAL_LESION_CHANNELS: tuple[str, ...] = ("MA", "HE", "EX", "SE")
EXCLUDED_ANATOMICAL_CLASSES: tuple[str, ...] = ("OD",)
ALLOWED_SOURCE_CLASS_KEYS: frozenset[str] = frozenset(
    CANONICAL_LESION_CHANNELS + EXCLUDED_ANATOMICAL_CLASSES
)


class MaskConstructionError(ValueError):
    """Raised when an image or lesion mask is unreadable, mismatched, or ambiguous."""


@dataclass(frozen=True)
class SingleMaskInspection:
    """Detailed inspection result for a single loaded lesion mask."""

    binary_mask: np.ndarray
    source_shape: tuple[int, ...]
    source_channels: int
    source_unique_values: tuple[int, ...]
    foreground_pixels: int
    is_empty: bool


@dataclass(frozen=True)
class CanonicalMaskResult:
    """Canonical ``(H, W, 4)`` lesion mask tensor and associated channel audit metadata."""

    mask: np.ndarray
    channel_order: tuple[str, ...]
    image_shape: tuple[int, int]
    source_mask_paths: dict[str, str | None]
    present_classes: tuple[str, ...]
    missing_classes: tuple[str, ...]
    synthesized_zero_classes: tuple[str, ...]
    empty_mask_classes: tuple[str, ...]
    positive_classes: tuple[str, ...]
    foreground_pixel_counts: dict[str, int]
    source_channel_counts: dict[str, int | None]
    source_unique_values: dict[str, list[int] | None]
    overlapping_pixel_count: int

    def to_metadata_dict(self) -> dict[str, Any]:
        """Return JSON-serializable metadata describing the canonical mask."""
        h, w = self.image_shape
        return {
            "channel_order": list(self.channel_order),
            "canonical_mask_dimensions": [int(h), int(w), len(self.channel_order)],
            "present_lesion_masks": list(self.present_classes),
            "missing_lesion_masks": list(self.missing_classes),
            "synthesized_zero_channels": list(self.synthesized_zero_classes),
            "empty_lesion_masks": list(self.empty_mask_classes),
            "positive_lesion_masks": list(self.positive_classes),
            "has_missing_masks": len(self.missing_classes) > 0,
            "lesion_pixel_counts": dict(self.foreground_pixel_counts),
            "source_mask_channels": dict(self.source_channel_counts),
            "source_mask_encodings": dict(self.source_unique_values),
            "overlapping_pixel_count": int(self.overlapping_pixel_count),
        }


def _extract_2d_foreground_channel(
    raw_mask: np.ndarray,
    threshold: int = 0,
    source_label: str = "mask",
) -> tuple[np.ndarray, int, tuple[int, ...]]:
    """Extract a validated 2D array from a grayscale, RGB, or RGBA mask array.

    For multichannel RGB ``(H, W, 3)`` and RGBA ``(H, W, 4)`` arrays:
    - Inspects each color channel ``R, G, B`` independently to determine which
      channels carry non-zero annotations.
    - Verifies that all active color channels share the exact same binary spatial
      support ``(ch > threshold)``. If multiple color channels contain conflicting
      non-zero patterns, raises :class:`MaskConstructionError`.
    - For RGBA ``(H, W, 4)`` masks (such as IDRiD's ``IDRiD_81_EX.tif`` where
      ``R in {0, 255}``, ``G = 0``, ``B = 0``, and ``A = 255``), separates the
      alpha channel ``A`` from ``RGB`` so a constant opaque alpha channel (``255``)
      never turns background pixels into foreground.
    """
    if not isinstance(raw_mask, np.ndarray):
        raise MaskConstructionError(
            f"[{source_label}] Expected numpy.ndarray, got {type(raw_mask).__name__}."
        )

    if raw_mask.ndim == 2:
        h, w = raw_mask.shape
        if h <= 0 or w <= 0:
            raise MaskConstructionError(
                f"[{source_label}] Invalid 2D mask dimensions: {raw_mask.shape}."
            )
        uniq = tuple(int(v) for v in np.unique(raw_mask).tolist())
        return raw_mask, 1, uniq

    if raw_mask.ndim != 3:
        raise MaskConstructionError(
            f"[{source_label}] Unsupported mask array rank {raw_mask.ndim} "
            f"(shape={raw_mask.shape}); expected 2D (H, W) or 3D (H, W, C)."
        )

    h, w, c = raw_mask.shape
    if h <= 0 or w <= 0:
        raise MaskConstructionError(
            f"[{source_label}] Invalid 3D mask dimensions: {raw_mask.shape}."
        )

    if c == 1:
        ch2d = raw_mask[:, :, 0]
        uniq = tuple(int(v) for v in np.unique(ch2d).tolist())
        return ch2d, 1, uniq

    if c not in (3, 4):
        raise MaskConstructionError(
            f"[{source_label}] Unsupported mask channel count {c} "
            f"(shape={raw_mask.shape}); supported channel counts are 1, 3 (RGB), or 4 (RGBA)."
        )

    rgb = raw_mask[:, :, :3]
    rgb_bins = [rgb[:, :, idx] > threshold for idx in range(3)]
    active_rgb_indices = [idx for idx, b in enumerate(rgb_bins) if bool(np.any(b))]

    # Verify all active RGB channels share identical binary foreground support
    if len(active_rgb_indices) > 1:
        ref_idx = active_rgb_indices[0]
        ref_bin = rgb_bins[ref_idx]
        for other_idx in active_rgb_indices[1:]:
            if not np.array_equal(ref_bin, rgb_bins[other_idx]):
                raise MaskConstructionError(
                    f"[{source_label}] Ambiguous multichannel mask: color channels "
                    f"{ref_idx} and {other_idx} contain distinct non-zero spatial annotations."
                )

    if c == 4:
        alpha = raw_mask[:, :, 3]
        alpha_bin = alpha > threshold
        alpha_any = bool(np.any(alpha_bin))
        alpha_all = bool(np.all(alpha_bin))

        if active_rgb_indices:
            if not alpha_any:
                raise MaskConstructionError(
                    f"[{source_label}] Ambiguous RGBA mask: RGB channel(s) {active_rgb_indices} "
                    "contain foreground annotations but alpha channel is completely zero."
                )
            ref_rgb_bin = rgb_bins[active_rgb_indices[0]]
            if not alpha_all and not np.array_equal(alpha_bin, ref_rgb_bin):
                raise MaskConstructionError(
                    f"[{source_label}] Ambiguous RGBA mask: spatially varying alpha channel "
                    "conflicts with RGB foreground channel annotation."
                )
            selected_2d = rgb[:, :, active_rgb_indices[0]]
            uniq = tuple(int(v) for v in np.unique(selected_2d).tolist())
            return selected_2d, 4, uniq
        else:
            # No active RGB channels. If alpha is uniformly 255 (opaque empty RGBA mask)
            # or all 0, the lesion mask is empty. If alpha varies spatially while RGB is 0,
            # reject as ambiguous.
            if alpha_any and not alpha_all:
                raise MaskConstructionError(
                    f"[{source_label}] Ambiguous RGBA mask: RGB channels are all zero "
                    "while alpha channel varies spatially."
                )
            zeros_2d = np.zeros((h, w), dtype=raw_mask.dtype)
            return zeros_2d, 4, (0,)

    # c == 3 (RGB)
    if not active_rgb_indices:
        zeros_2d = np.zeros((h, w), dtype=raw_mask.dtype)
        return zeros_2d, 3, (0,)

    selected_2d = rgb[:, :, active_rgb_indices[0]]
    uniq = tuple(int(v) for v in np.unique(selected_2d).tolist())
    return selected_2d, 3, uniq


def inspect_and_binarize_mask_array(
    raw_mask: np.ndarray,
    threshold: int = 0,
    source_label: str = "mask",
    require_two_level_encoding: bool = True,
) -> SingleMaskInspection:
    """Convert a raw mask array to a binary ``{0, 1}`` ``uint8`` 2D mask with validation.

    Thresholding policy:
    - Extracts the validated 2D annotation plane from grayscale ``(H, W)``,
      ``(H, W, 1)``, RGB ``(H, W, 3)``, or RGBA ``(H, W, 4)`` arrays.
    - Rejects negative pixel values or non-binary continuous intensity distributions
      (>2 distinct values in the annotation plane when ``require_two_level_encoding=True``).
    - Maps ``pixel > threshold`` (default ``threshold = 0``) to ``1`` and ``<= threshold``
      to ``0`` as ``np.uint8``, normalizing both ``{0, 1}`` and ``{0, 255}`` source
      encodings deterministically to ``{0, 1}``.
    """
    source_shape = tuple(int(d) for d in raw_mask.shape) if isinstance(raw_mask, np.ndarray) else ()
    plane_2d, source_channels, uniq_vals = _extract_2d_foreground_channel(
        raw_mask, threshold=threshold, source_label=source_label
    )

    if any(v < 0 for v in uniq_vals):
        raise MaskConstructionError(
            f"[{source_label}] Mask contains negative pixel values: {uniq_vals}."
        )

    if require_two_level_encoding and len(uniq_vals) > 2:
        raise MaskConstructionError(
            f"[{source_label}] Ambiguous non-binary mask encoding with {len(uniq_vals)} "
            f"unique values {uniq_vals[:10]}; expected binary source encoding such as {{0, 1}} or {{0, 255}}."
        )

    binary = (plane_2d > threshold).astype(np.uint8)
    fg_pixels = int(binary.sum())
    return SingleMaskInspection(
        binary_mask=binary,
        source_shape=source_shape,
        source_channels=source_channels,
        source_unique_values=uniq_vals,
        foreground_pixels=fg_pixels,
        is_empty=(fg_pixels == 0),
    )


def convert_raw_mask_to_binary(
    raw_mask: np.ndarray,
    threshold: int = 0,
    source_label: str = "mask",
    require_two_level_encoding: bool = True,
) -> np.ndarray:
    """Convert a raw mask array (grayscale, RGB, or RGBA) to a 2D ``{0, 1}`` ``uint8`` array."""
    return inspect_and_binarize_mask_array(
        raw_mask=raw_mask,
        threshold=threshold,
        source_label=source_label,
        require_two_level_encoding=require_two_level_encoding,
    ).binary_mask


def inspect_single_lesion_mask(
    mask_path: str | Path,
    expected_shape: tuple[int, int] | None = None,
    threshold: int = 0,
) -> SingleMaskInspection:
    """Load, validate, and binarize a single lesion mask file from disk.

    Args:
        mask_path: Path to the lesion mask image file.
        expected_shape: Optional ``(height, width)`` of the paired source fundus image.
        threshold: Binarization threshold (default ``0``, mapping ``> 0`` to ``1``).

    Returns:
        :class:`SingleMaskInspection` containing the ``(H, W)`` ``uint8`` binary mask
        and source encoding metadata.

    Raises:
        FileNotFoundError: If ``mask_path`` does not exist.
        MaskConstructionError: If the file cannot be decoded, has an unsupported or
            ambiguous channel layout, or does not match ``expected_shape``.
    """
    path = Path(mask_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Lesion mask file not found: {path}")

    try:
        with Image.open(path) as img:
            raw_arr = np.asarray(img)
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise MaskConstructionError(
            f"Unreadable or corrupted lesion mask file '{path}': {exc}"
        ) from exc

    inspection = inspect_and_binarize_mask_array(
        raw_arr,
        threshold=threshold,
        source_label=str(path),
    )

    if expected_shape is not None:
        exp_h, exp_w = int(expected_shape[0]), int(expected_shape[1])
        mask_h, mask_w = inspection.binary_mask.shape
        if (mask_h, mask_w) != (exp_h, exp_w):
            raise MaskConstructionError(
                f"Mask dimension mismatch for '{path}': mask has shape ({mask_h}, {mask_w}) "
                f"but source image has shape ({exp_h}, {exp_w})."
            )

    return inspection


def load_single_lesion_mask(
    mask_path: str | Path,
    expected_shape: tuple[int, int] | None = None,
    threshold: int = 0,
) -> np.ndarray:
    """Load a single lesion mask file and return its 2D ``(H, W)`` binary ``{0, 1}`` ``uint8`` array."""
    return inspect_single_lesion_mask(
        mask_path=mask_path,
        expected_shape=expected_shape,
        threshold=threshold,
    ).binary_mask


def build_canonical_lesion_mask(
    image_shape: tuple[int, int],
    mask_paths: Mapping[str, str | Path | None],
    channel_order: Sequence[str] = CANONICAL_LESION_CHANNELS,
    threshold: int = 0,
) -> CanonicalMaskResult:
    """Construct a canonical ``(H, W, 4)`` binary lesion mask in ``[MA, HE, EX, SE]`` order.

    Args:
        image_shape: ``(height, width)`` of the source fundus image.
        mask_paths: Mapping from lesion class code (``"MA"``, ``"HE"``, ``"EX"``, ``"SE"``;
            any ``"OD"`` entry is ignored/excluded) to a file path or ``None`` when
            that lesion mask is omitted.
        channel_order: Must be ``("MA", "HE", "EX", "SE")``.
        threshold: Binarization threshold (default ``0``).

    Returns:
        :class:`CanonicalMaskResult` containing the ``(H, W, 4)`` ``np.uint8`` array
        with values in ``{0, 1}`` and detailed per-channel audit metadata.
    """
    if len(image_shape) < 2:
        raise MaskConstructionError(
            f"image_shape must provide (height, width), got {image_shape!r}."
        )
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise MaskConstructionError(
            f"image_shape dimensions must be positive, got ({height}, {width})."
        )

    order_tuple = tuple(str(ch).upper() for ch in channel_order)
    if order_tuple != CANONICAL_LESION_CHANNELS:
        raise MaskConstructionError(
            f"Canonical channel_order must be {CANONICAL_LESION_CHANNELS}, got {order_tuple}."
        )

    normalized_paths: dict[str, Path | None] = {}
    for raw_key, raw_val in mask_paths.items():
        key_upper = str(raw_key).strip().upper()
        if key_upper not in ALLOWED_SOURCE_CLASS_KEYS:
            raise MaskConstructionError(
                f"Unsupported lesion class key {raw_key!r} in mask_paths. "
                f"Expected keys from {CANONICAL_LESION_CHANNELS} (OD is automatically excluded)."
            )
        if key_upper in EXCLUDED_ANATOMICAL_CLASSES:
            # Explicitly exclude Optic Disc (OD) from the 4-channel lesion target
            continue
        normalized_paths[key_upper] = Path(raw_val) if raw_val is not None else None

    channels_list: list[np.ndarray] = []
    source_mask_paths_str: dict[str, str | None] = {}
    present_classes: list[str] = []
    missing_classes: list[str] = []
    synthesized_zero_classes: list[str] = []
    empty_mask_classes: list[str] = []
    positive_classes: list[str] = []
    fg_counts: dict[str, int] = {}
    src_channels: dict[str, int | None] = {}
    src_unique: dict[str, list[int] | None] = {}

    for cls_name in CANONICAL_LESION_CHANNELS:
        m_path = normalized_paths.get(cls_name)
        if m_path is None:
            zero_channel = np.zeros((height, width), dtype=np.uint8)
            channels_list.append(zero_channel)
            source_mask_paths_str[cls_name] = None
            missing_classes.append(cls_name)
            synthesized_zero_classes.append(cls_name)
            fg_counts[cls_name] = 0
            src_channels[cls_name] = None
            src_unique[cls_name] = None
        else:
            inspection = inspect_single_lesion_mask(
                m_path,
                expected_shape=(height, width),
                threshold=threshold,
            )
            channels_list.append(inspection.binary_mask)
            source_mask_paths_str[cls_name] = str(m_path)
            present_classes.append(cls_name)
            fg_counts[cls_name] = inspection.foreground_pixels
            src_channels[cls_name] = inspection.source_channels
            src_unique[cls_name] = list(inspection.source_unique_values)
            if inspection.is_empty:
                empty_mask_classes.append(cls_name)
            else:
                positive_classes.append(cls_name)

    canonical_mask = np.stack(channels_list, axis=-1).astype(np.uint8)
    overlap_pixels = int((canonical_mask.sum(axis=-1) > 1).sum())

    return CanonicalMaskResult(
        mask=canonical_mask,
        channel_order=CANONICAL_LESION_CHANNELS,
        image_shape=(height, width),
        source_mask_paths=source_mask_paths_str,
        present_classes=tuple(present_classes),
        missing_classes=tuple(missing_classes),
        synthesized_zero_classes=tuple(synthesized_zero_classes),
        empty_mask_classes=tuple(empty_mask_classes),
        positive_classes=tuple(positive_classes),
        foreground_pixel_counts=fg_counts,
        source_channel_counts=src_channels,
        source_unique_values=src_unique,
        overlapping_pixel_count=overlap_pixels,
    )


def load_rgb_image(image_path: str | Path) -> np.ndarray:
    """Load a source retinal fundus image as an RGB ``(H, W, 3)`` ``uint8`` NumPy array."""
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Source image file not found: {path}")

    try:
        with Image.open(path) as img:
            rgb_img = img.convert("RGB")
            arr = np.asarray(rgb_img, dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise MaskConstructionError(
            f"Unreadable or corrupted source image '{path}': {exc}"
        ) from exc

    if arr.ndim != 3 or arr.shape[2] != 3 or arr.shape[0] <= 0 or arr.shape[1] <= 0:
        raise MaskConstructionError(
            f"Invalid loaded RGB image shape {arr.shape} for '{path}'."
        )
    return arr


def load_sample_image_and_mask(
    image_path: str | Path,
    mask_paths: Mapping[str, str | Path | None],
    threshold: int = 0,
) -> tuple[np.ndarray, CanonicalMaskResult]:
    """Load an RGB fundus image ``(H, W, 3)`` and its canonical ``(H, W, 4)`` lesion mask."""
    image = load_rgb_image(image_path)
    h, w = int(image.shape[0]), int(image.shape[1])
    mask_result = build_canonical_lesion_mask(
        image_shape=(h, w),
        mask_paths=mask_paths,
        threshold=threshold,
    )
    return image, mask_result


__all__ = [
    "ALLOWED_SOURCE_CLASS_KEYS",
    "CANONICAL_LESION_CHANNELS",
    "CanonicalMaskResult",
    "EXCLUDED_ANATOMICAL_CLASSES",
    "MaskConstructionError",
    "SingleMaskInspection",
    "build_canonical_lesion_mask",
    "convert_raw_mask_to_binary",
    "inspect_and_binarize_mask_array",
    "inspect_single_lesion_mask",
    "load_rgb_image",
    "load_sample_image_and_mask",
    "load_single_lesion_mask",
]

