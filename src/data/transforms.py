"""Configurable Albumentations 2.0.8 segmentation transforms for retinal images and 4-channel masks.

Interpolation & Alignment Policy:
- Source RGB fundus images ``(H, W, 3)`` are resized using continuous interpolation
  (default ``cv2.INTER_LINEAR`` / bilinear).
- Canonical four-channel lesion masks ``(H, W, 4)`` in ``[MA, HE, EX, SE]`` order
  are resized strictly with nearest-neighbor interpolation (``cv2.INTER_NEAREST``)
  via Albumentations ``A.Resize(..., mask_interpolation=cv2.INTER_NEAREST)`` and
  explicitly thresholded ``(mask > 0).astype(np.uint8)`` so binary ``{0, 1}`` values
  are preserved without fractional blending.
- Tradeoff note on spatial downsampling: Nearest-neighbor mask interpolation preserves
  exact binary ``{0, 1}`` semantics and sharp lesion boundaries without requiring
  post-hoc probability threshold tuning; however, when downsampling high-resolution
  fundus images (e.g., ``2848x4288`` or ``3456x5184`` to ``512x512``), isolated
  1-2 pixel microaneurysms (``MA``) smaller than the downsampling stride can be
  undersampled. Using higher target resolutions (e.g., ``1024x1024``) or patch-based
  sampling mitigates this when fine-grained ``MA`` recall is prioritized.
- Spatial transforms (``Resize``, ``HorizontalFlip``, ``VerticalFlip``,
  ``RandomRotate90``) are applied jointly in a single ``A.Compose`` call so the
  RGB image and all 4 lesion channels receive the exact same geometric transformation.
- Photometric transforms (``ColorJitter``, ``Normalize``) are ``ImageOnlyTransform``
  operations and never modify the lesion mask.
- Validation and test transforms are strictly deterministic (``Resize`` + normalization
  only; no random crops, flips, rotations, or color jitter).
"""

from __future__ import annotations

import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np

from src.utils.config import load_yaml_config

IMAGE_INTERPOLATION_MAP: dict[str, int] = {
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos4": cv2.INTER_LANCZOS4,
}

DEFAULT_IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
DEFAULT_IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TransformConfigSummary:
    """Configuration summary for a constructed segmentation transform pipeline."""

    is_train: bool
    target_height: int
    target_width: int
    image_interpolation: str
    mask_interpolation: str
    augmentation_enabled: bool
    horizontal_flip_prob: float
    vertical_flip_prob: float
    random_rotate90_prob: float
    color_jitter_enabled: bool
    color_jitter_prob: float
    normalize_enabled: bool
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


class RetinalSegmentationTransform:
    """Joint image and 4-channel binary mask transform wrapper around Albumentations 2.0.8."""

    def __init__(
        self,
        height: int = 512,
        width: int = 512,
        is_train: bool = False,
        augmentation_enabled: bool = True,
        horizontal_flip_prob: float = 0.5,
        vertical_flip_prob: float = 0.5,
        random_rotate90_prob: float = 0.5,
        color_jitter_enabled: bool = True,
        color_jitter_prob: float = 0.5,
        brightness: float = 0.15,
        contrast: float = 0.15,
        saturation: float = 0.15,
        hue: float = 0.05,
        normalize_enabled: bool = True,
        mean: Sequence[float] = DEFAULT_IMAGENET_MEAN,
        std: Sequence[float] = DEFAULT_IMAGENET_STD,
        image_interpolation: str = "linear",
        mask_interpolation: str = "nearest",
    ) -> None:
        if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
            raise ValueError(f"Transform height must be a positive integer, got {height!r}.")
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError(f"Transform width must be a positive integer, got {width!r}.")

        interp_key = str(image_interpolation).strip().lower()
        if interp_key not in IMAGE_INTERPOLATION_MAP:
            raise ValueError(
                f"Unsupported image_interpolation {image_interpolation!r}; "
                f"expected one of {tuple(IMAGE_INTERPOLATION_MAP.keys())}."
            )
        mask_interp_key = str(mask_interpolation).strip().lower()
        if mask_interp_key != "nearest":
            raise ValueError(
                f"mask_interpolation must be 'nearest' to preserve binary lesion masks, "
                f"got {mask_interpolation!r}."
            )

        self.height = int(height)
        self.width = int(width)
        self.is_train = bool(is_train)
        self.augmentation_enabled = bool(is_train and augmentation_enabled)
        self.normalize_enabled = bool(normalize_enabled)
        self.mean = (float(mean[0]), float(mean[1]), float(mean[2]))
        self.std = (float(std[0]), float(std[1]), float(std[2]))
        self.image_interpolation = interp_key
        self.mask_interpolation = mask_interp_key

        cv_img_interp = IMAGE_INTERPOLATION_MAP[interp_key]
        ops: list[A.BasicTransform] = [
            A.Resize(
                height=self.height,
                width=self.width,
                interpolation=cv_img_interp,
                mask_interpolation=cv2.INTER_NEAREST,
                p=1.0,
            )
        ]

        if self.augmentation_enabled:
            if horizontal_flip_prob > 0.0:
                ops.append(A.HorizontalFlip(p=float(horizontal_flip_prob)))
            if vertical_flip_prob > 0.0:
                ops.append(A.VerticalFlip(p=float(vertical_flip_prob)))
            if random_rotate90_prob > 0.0:
                ops.append(A.RandomRotate90(p=float(random_rotate90_prob)))
            if color_jitter_enabled and color_jitter_prob > 0.0:
                ops.append(
                    A.ColorJitter(
                        brightness=float(brightness),
                        contrast=float(contrast),
                        saturation=float(saturation),
                        hue=float(hue),
                        p=float(color_jitter_prob),
                    )
                )

        if self.normalize_enabled:
            ops.append(
                A.Normalize(
                    mean=self.mean,
                    std=self.std,
                    max_pixel_value=255.0,
                    p=1.0,
                )
            )

        self.compose = A.Compose(ops)
        self.summary = TransformConfigSummary(
            is_train=self.is_train,
            target_height=self.height,
            target_width=self.width,
            image_interpolation=self.image_interpolation,
            mask_interpolation=self.mask_interpolation,
            augmentation_enabled=self.augmentation_enabled,
            horizontal_flip_prob=float(horizontal_flip_prob) if self.augmentation_enabled else 0.0,
            vertical_flip_prob=float(vertical_flip_prob) if self.augmentation_enabled else 0.0,
            random_rotate90_prob=float(random_rotate90_prob) if self.augmentation_enabled else 0.0,
            color_jitter_enabled=bool(self.augmentation_enabled and color_jitter_enabled),
            color_jitter_prob=float(color_jitter_prob) if (self.augmentation_enabled and color_jitter_enabled) else 0.0,
            normalize_enabled=self.normalize_enabled,
            mean=self.mean,
            std=self.std,
        )

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply joint transforms to ``image`` ``(H, W, 3)`` and ``mask`` ``(H, W, 4)``.

        Returns:
            Tuple ``(transformed_image, transformed_mask)`` where ``transformed_image``
            has shape ``(target_H, target_W, 3)`` (``np.float32``) and ``transformed_mask``
            has shape ``(target_H, target_W, 4)`` (``np.uint8`` with values in ``{0, 1}``).
        """
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected RGB image of shape (H, W, 3), got {getattr(image, 'shape', type(image))}."
            )
        if not isinstance(mask, np.ndarray) or mask.ndim != 3 or mask.shape[2] != 4:
            raise ValueError(
                f"Expected 4-channel lesion mask of shape (H, W, 4), got {getattr(mask, 'shape', type(mask))}."
            )
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Spatial mismatch before transform: image {image.shape[:2]} vs mask {mask.shape[:2]}."
            )

        if self.augmentation_enabled:
            # Draw step seed from NumPy's RNG so set_seed() and DataLoader seed_worker()
            # deterministically control Albumentations 2.0.8 internal generators.
            step_seed = int(np.random.randint(0, 2**31 - 1))
            self.compose.set_random_seed(step_seed)

        result = self.compose(image=image, mask=mask)
        aug_image = result["image"]
        aug_mask = result["mask"]

        if not self.normalize_enabled:
            aug_image = aug_image.astype(np.float32) / 255.0
        else:
            aug_image = aug_image.astype(np.float32, copy=False)

        # Explicitly enforce binary {0, 1} uint8 semantics after spatial transform
        binary_mask = (aug_mask > 0).astype(np.uint8)
        return aug_image, binary_mask


def build_segmentation_transforms(
    split: str = "train",
    dataset_config: Mapping[str, Any] | str | Path = "dataset.yaml",
    image_size: tuple[int, int] | None = None,
) -> RetinalSegmentationTransform:
    """Construct a configured :class:`RetinalSegmentationTransform` for the given split.

    Args:
        split: Split identifier (``"train"`` enables configured stochastic augmentations;
            ``"valid"``, ``"val"``, and ``"test"`` construct deterministic evaluation
            transforms).
        dataset_config: Loaded configuration dictionary or path to ``dataset.yaml``.
        image_size: Optional ``(height, width)`` override.

    Returns:
        Configured :class:`RetinalSegmentationTransform`.
    """
    if isinstance(dataset_config, (str, Path)):
        cfg = load_yaml_config(dataset_config, validate=True)
    else:
        cfg = dict(dataset_config)

    split_norm = str(split).strip().lower()
    if split_norm not in {"train", "valid", "val", "test"}:
        raise ValueError(
            f"Unsupported split {split!r}; expected 'train', 'valid', or 'test'."
        )
    is_train = split_norm == "train"

    if image_size is not None:
        height, width = int(image_size[0]), int(image_size[1])
    else:
        size_cfg = cfg.get("image_size", {"height": 512, "width": 512})
        if isinstance(size_cfg, Mapping):
            height = int(size_cfg["height"])
            width = int(size_cfg["width"])
        else:
            height = int(size_cfg)
            width = int(size_cfg)

    eval_cfg = cfg.get("evaluation_transforms", {})
    img_interp = str(eval_cfg.get("resize_interpolation", "linear"))
    mask_interp = str(eval_cfg.get("mask_interpolation", "nearest"))

    norm_cfg = cfg.get("normalization", {})
    norm_enabled = bool(norm_cfg.get("enabled", True))
    if "normalize" in eval_cfg and not is_train:
        norm_enabled = bool(eval_cfg["normalize"])
    mean = norm_cfg.get("mean", list(DEFAULT_IMAGENET_MEAN))
    std = norm_cfg.get("std", list(DEFAULT_IMAGENET_STD))

    aug_cfg = cfg.get("augmentation", {})
    aug_enabled = bool(aug_cfg.get("enabled", True))
    h_flip = float(aug_cfg.get("horizontal_flip_prob", 0.5))
    v_flip = float(aug_cfg.get("vertical_flip_prob", 0.5))
    rot90 = float(aug_cfg.get("random_rotate90_prob", 0.5))

    cj_cfg = aug_cfg.get("color_jitter", {})
    cj_enabled = bool(cj_cfg.get("enabled", True))
    cj_prob = float(cj_cfg.get("prob", 0.5))
    brightness = float(cj_cfg.get("brightness", 0.15))
    contrast = float(cj_cfg.get("contrast", 0.15))
    saturation = float(cj_cfg.get("saturation", 0.15))
    hue = float(cj_cfg.get("hue", 0.05))

    return RetinalSegmentationTransform(
        height=height,
        width=width,
        is_train=is_train,
        augmentation_enabled=aug_enabled,
        horizontal_flip_prob=h_flip,
        vertical_flip_prob=v_flip,
        random_rotate90_prob=rot90,
        color_jitter_enabled=cj_enabled,
        color_jitter_prob=cj_prob,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
        normalize_enabled=norm_enabled,
        mean=mean,
        std=std,
        image_interpolation=img_interp,
        mask_interpolation=mask_interp,
    )


__all__ = [
    "DEFAULT_IMAGENET_MEAN",
    "DEFAULT_IMAGENET_STD",
    "IMAGE_INTERPOLATION_MAP",
    "RetinalSegmentationTransform",
    "TransformConfigSummary",
    "build_segmentation_transforms",
]

