"""Retinal field-of-view (FOV) detection and synchronized image/mask cropping.

Detection Method & Thresholding Policy:
1. Per-pixel foreground intensity is computed as the maximum across the color
   channels ``image[..., :3].max(axis=-1)`` (or directly ``image`` for 2D arrays).
   Using the channel-wise maximum rather than a green/blue-weighted grayscale
   conversion preserves dim red-dominant retinal tissue near the circular fundus
   periphery.
2. Pixels with ``intensity > threshold`` (default ``threshold = 10`` on ``uint8``
   ``[0, 255]`` images, rejecting low-amplitude JPEG compression noise in black
   background corners) are marked as retinal foreground.
3. The tight axis-aligned bounding box ``[y_min:y_max, x_min:x_max]`` enclosing
   all foreground rows and columns is computed and optionally padded by ``margin``
   pixels (clamped to ``[0, H] x [0, W]``).

Fallback Behavior:
- If no pixel exceeds ``threshold`` (e.g., an all-black or under-exposed image),
  or if the detected bounding-box area is smaller than ``min_fov_fraction`` of
  the full image area (default ``0.01``), the detector deterministically falls
  back to the full image bounds ``(0, H, 0, W)`` with ``fallback_used=True`` and
  an explicit ``fallback_reason``.

Synchronized Image & Mask Cropping:
- The lesion mask is **never** cropped independently from its paired fundus image.
- Both image and mask are sliced using the exact same ``[y_min:y_max, x_min:x_max]``
  integer coordinates without any spatial interpolation, preserving binary ``{0, 1}``
  mask values and exact spatial alignment.
- When ``preserve_boundary_lesions=True`` and a mask is provided, if any positive
  lesion pixel lies outside the initial intensity-derived bounding box (e.g., on
  an unusually dark peripheral border), the shared bounding box is expanded to
  include all positive lesion coordinates before slicing both image and mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_FOV_THRESHOLD: int = 10
DEFAULT_FOV_MARGIN: int = 0
DEFAULT_MIN_FOV_FRACTION: float = 0.01


@dataclass(frozen=True)
class FOVCropMetadata:
    """Deterministic bounding-box coordinates and audit metadata for retinal FOV cropping."""

    y_min: int
    y_max: int
    x_min: int
    x_max: int
    original_height: int
    original_width: int
    cropped_height: int
    cropped_width: int
    threshold: int
    margin: int
    fallback_used: bool
    fallback_reason: str | None = None
    boundary_expanded_for_mask: bool = False
    lesion_pixels_outside_fov: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize FOV crop coordinates and metadata to a JSON-compatible dictionary."""
        return {
            "y_min": int(self.y_min),
            "y_max": int(self.y_max),
            "x_min": int(self.x_min),
            "x_max": int(self.x_max),
            "original_height": int(self.original_height),
            "original_width": int(self.original_width),
            "cropped_height": int(self.cropped_height),
            "cropped_width": int(self.cropped_width),
            "threshold": int(self.threshold),
            "margin": int(self.margin),
            "fallback_used": bool(self.fallback_used),
            "fallback_reason": self.fallback_reason,
            "boundary_expanded_for_mask": bool(self.boundary_expanded_for_mask),
            "lesion_pixels_outside_fov": int(self.lesion_pixels_outside_fov),
        }


def _validate_image_array(image: np.ndarray) -> tuple[int, int]:
    """Validate an input image array and return ``(height, width)``."""
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Expected image as numpy.ndarray, got {type(image).__name__}.")
    if image.ndim not in (2, 3):
        raise ValueError(
            f"Expected 2D (H, W) or 3D (H, W, C) image array, got shape {image.shape}."
        )
    h, w = int(image.shape[0]), int(image.shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"Image dimensions must be positive, got shape {image.shape}.")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError(
            f"Unsupported image channel count {image.shape[2]} for shape {image.shape}."
        )
    return h, w


def detect_retinal_fov(
    image: np.ndarray,
    threshold: int = DEFAULT_FOV_THRESHOLD,
    margin: int = DEFAULT_FOV_MARGIN,
    min_fov_fraction: float = DEFAULT_MIN_FOV_FRACTION,
) -> FOVCropMetadata:
    """Detect the retinal field-of-view bounding box from a fundus image.

    Args:
        image: 2D ``(H, W)`` or 3D ``(H, W, C)`` fundus image array.
        threshold: Intensity threshold above which pixels are considered retinal
            foreground (default ``10``). Must be ``>= 0``.
        margin: Optional non-negative pixel padding added around the detected
            bounding box and clamped to image bounds (default ``0``).
        min_fov_fraction: Minimum fraction of total image area ``[0.0, 1.0]``
            required for a valid FOV crop before triggering fallback to the full
            image (default ``0.01``).

    Returns:
        :class:`FOVCropMetadata` with deterministic bounding-box coordinates.
    """
    h, w = _validate_image_array(image)
    if threshold < 0:
        raise ValueError(f"FOV threshold must be >= 0, got {threshold}.")
    if margin < 0:
        raise ValueError(f"FOV margin must be >= 0, got {margin}.")
    if not (0.0 <= min_fov_fraction <= 1.0):
        raise ValueError(
            f"min_fov_fraction must be between 0.0 and 1.0, got {min_fov_fraction}."
        )

    if image.ndim == 2:
        intensity = image
    elif image.shape[2] == 1:
        intensity = image[:, :, 0]
    else:
        intensity = image[:, :, :3].max(axis=-1)

    fg_mask = intensity > threshold
    if not bool(np.any(fg_mask)):
        return FOVCropMetadata(
            y_min=0,
            y_max=h,
            x_min=0,
            x_max=w,
            original_height=h,
            original_width=w,
            cropped_height=h,
            cropped_width=w,
            threshold=threshold,
            margin=margin,
            fallback_used=True,
            fallback_reason="no_foreground_detected",
        )

    fg_rows = np.where(fg_mask.any(axis=1))[0]
    fg_cols = np.where(fg_mask.any(axis=0))[0]

    y_min = max(0, int(fg_rows[0]) - margin)
    y_max = min(h, int(fg_rows[-1]) + 1 + margin)
    x_min = max(0, int(fg_cols[0]) - margin)
    x_max = min(w, int(fg_cols[-1]) + 1 + margin)

    crop_h = y_max - y_min
    crop_w = x_max - x_min
    total_area = float(h * w)
    crop_area = float(crop_h * crop_w)

    if min_fov_fraction > 0.0 and crop_area < (min_fov_fraction * total_area):
        return FOVCropMetadata(
            y_min=0,
            y_max=h,
            x_min=0,
            x_max=w,
            original_height=h,
            original_width=w,
            cropped_height=h,
            cropped_width=w,
            threshold=threshold,
            margin=margin,
            fallback_used=True,
            fallback_reason="fov_area_below_min_fraction",
        )

    return FOVCropMetadata(
        y_min=y_min,
        y_max=y_max,
        x_min=x_min,
        x_max=x_max,
        original_height=h,
        original_width=w,
        cropped_height=crop_h,
        cropped_width=crop_w,
        threshold=threshold,
        margin=margin,
        fallback_used=False,
        fallback_reason=None,
    )


def crop_image_and_mask_to_fov(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    threshold: int = DEFAULT_FOV_THRESHOLD,
    margin: int = DEFAULT_FOV_MARGIN,
    min_fov_fraction: float = DEFAULT_MIN_FOV_FRACTION,
    preserve_boundary_lesions: bool = True,
    fov_box: FOVCropMetadata | None = None,
) -> tuple[np.ndarray, np.ndarray | None, FOVCropMetadata]:
    """Crop an image and its optional lesion mask using identical retinal FOV coordinates.

    Args:
        image: Source fundus image of shape ``(H, W, 3)`` or ``(H, W)``.
        mask: Optional lesion mask of shape ``(H, W, C)`` or ``(H, W)``. Must have
            the exact same spatial dimensions ``(H, W)`` as ``image``.
        threshold: Intensity threshold for FOV detection (default ``10``).
        margin: Optional boundary padding in pixels (default ``0``).
        min_fov_fraction: Minimum valid FOV area fraction (default ``0.01``).
        preserve_boundary_lesions: When ``True`` and ``mask`` is provided, ensures
            the shared crop coordinates encompass 100% of positive mask pixels even
            if a boundary lesion falls on a very dark peripheral pixel.
        fov_box: Optional pre-computed :class:`FOVCropMetadata` to apply directly.

    Returns:
        Tuple of ``(cropped_image, cropped_mask, fov_metadata)``.
         Cropping is performed by integer array slicing without interpolation.
    """
    h, w = _validate_image_array(image)

    if mask is not None:
        if not isinstance(mask, np.ndarray) or mask.ndim not in (2, 3):
            raise ValueError(
                f"Expected mask as 2D or 3D numpy.ndarray, got {type(mask).__name__}."
            )
        mask_h, mask_w = int(mask.shape[0]), int(mask.shape[1])
        if (mask_h, mask_w) != (h, w):
            raise ValueError(
                f"Spatial dimension mismatch between image ({h}, {w}) and mask ({mask_h}, {mask_w})."
            )

    box = (
        fov_box
        if fov_box is not None
        else detect_retinal_fov(
            image,
            threshold=threshold,
            margin=margin,
            min_fov_fraction=min_fov_fraction,
        )
    )

    if (box.original_height, box.original_width) != (h, w):
        raise ValueError(
            f"FOVCropMetadata original dimensions ({box.original_height}, {box.original_width}) "
            f"do not match image dimensions ({h}, {w})."
        )

    y_min, y_max, x_min, x_max = box.y_min, box.y_max, box.x_min, box.x_max
    boundary_expanded = False
    outside_pixels = 0

    if mask is not None:
        fg_2d = (mask > 0).any(axis=-1) if mask.ndim == 3 else (mask > 0)
        total_fg = int(fg_2d.sum())
        if total_fg > 0:
            inside_fg = int(fg_2d[y_min:y_max, x_min:x_max].sum())
            outside_pixels = total_fg - inside_fg
            if outside_pixels > 0 and preserve_boundary_lesions:
                m_rows = np.where(fg_2d.any(axis=1))[0]
                m_cols = np.where(fg_2d.any(axis=0))[0]
                y_min = min(y_min, int(m_rows[0]))
                y_max = max(y_max, int(m_rows[-1]) + 1)
                x_min = min(x_min, int(m_cols[0]))
                x_max = max(x_max, int(m_cols[-1]) + 1)
                boundary_expanded = True
                outside_pixels = 0

    final_box = FOVCropMetadata(
        y_min=y_min,
        y_max=y_max,
        x_min=x_min,
        x_max=x_max,
        original_height=h,
        original_width=w,
        cropped_height=y_max - y_min,
        cropped_width=x_max - x_min,
        threshold=box.threshold,
        margin=box.margin,
        fallback_used=box.fallback_used,
        fallback_reason=box.fallback_reason,
        boundary_expanded_for_mask=boundary_expanded,
        lesion_pixels_outside_fov=outside_pixels,
    )

    cropped_image = image[y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max] if mask is not None else None

    return cropped_image, cropped_mask, final_box


__all__ = [
    "DEFAULT_FOV_MARGIN",
    "DEFAULT_FOV_THRESHOLD",
    "DEFAULT_MIN_FOV_FRACTION",
    "FOVCropMetadata",
    "crop_image_and_mask_to_fov",
    "detect_retinal_fov",
]

