"""High-resolution retinal lesion patch extraction and lesion-aware sampling.

Provides deterministic, spatially aligned patch extraction and lesion-aware
sampling for high-resolution retinal fundus images and 4-channel binary masks
in canonical ``[MA, HE, EX, SE]`` order.

Key Capabilities:
1. Slices full FOV-cropped images and masks into aligned patches (e.g. 512x512, 384x384).
2. Preserves exact binary ``{0, 1}`` mask semantics and 4-channel ordering.
3. Categorizes patches by lesion presence using a strict rarity hierarchy
   (MA > SE > HE > EX > background).
4. Provides lesion-aware weighted sampling to counter extreme class imbalance
   while preserving a configurable background quota to avoid false-positive bias.
5. Implements ``PatchRetinalSegmentationDataset`` integrating with standard
   PyTorch ``DataLoader`` instances.
6. Strictly enforces split isolation: training patches may ONLY be sampled
   from permitted training records (e.g. the 44 IDRiD training records).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.fov import crop_image_and_mask_to_fov
from src.data.masks import CANONICAL_LESION_CHANNELS, load_sample_image_and_mask
from src.data.transforms import (
    DEFAULT_IMAGENET_MEAN,
    DEFAULT_IMAGENET_STD,
    build_segmentation_transforms,
)
from src.utils.config import load_yaml_config
from src.utils.logger import get_logger
from src.utils.paths import resolve_project_path
from src.utils.seed import MAX_UINT32_SEED

logger = get_logger("fedmorphis.data.patches")

CANONICAL_LESION_ORDER: tuple[str, ...] = tuple(CANONICAL_LESION_CHANNELS)

# Rarity hierarchy for multi-lesion patches: MA is rarest and most clinically vulnerable
# to downsampling, followed by SE, HE, EX.
LESION_RARITY_PRIORITY: tuple[str, ...] = ("MA", "SE", "HE", "EX")

DEFAULT_SAMPLING_WEIGHTS: dict[str, float] = {
    "MA": 5.0,
    "SE": 3.0,
    "HE": 2.0,
    "EX": 1.0,
    "background": 0.5,
}


@dataclass(frozen=True)
class PatchCoordinates:
    """Bounding box coordinates for a spatial patch within an image."""

    y_min: int
    y_max: int
    x_min: int
    x_max: int

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def width(self) -> int:
        return self.x_max - self.x_min


@dataclass(frozen=True)
class PatchMetadata:
    """Metadata for an extracted patch, including lesion counts and assigned category."""

    sample_id: str
    record_index: int
    coordinates: PatchCoordinates
    pixel_counts: dict[str, int]
    primary_category: str
    contains_lesion: bool
    is_padded: bool


def compute_patch_grid_coordinates(
    image_height: int,
    image_width: int,
    patch_size: int | tuple[int, int] = 512,
    stride: int | tuple[int, int] | None = None,
) -> list[PatchCoordinates]:
    """Calculate deterministic grid coordinates covering an image without gaps.

    Ensures that boundary patches are shifted inward to cover the trailing
    edges of the image if dimensions are not exact multiples of the stride.

    Args:
        image_height: Height of the image to tile.
        image_width: Width of the image to tile.
        patch_size: Square dimension or (patch_height, patch_width).
        stride: Step between patch origins. Defaults to patch_size // 2 (50% overlap).

    Returns:
        List of PatchCoordinates covering the entire image.
    """
    if isinstance(patch_size, (int, float)):
        p_h = p_w = int(patch_size)
    else:
        p_h, p_w = int(patch_size[0]), int(patch_size[1])

    if p_h <= 0 or p_w <= 0:
        raise ValueError(f"Patch dimensions must be positive, got ({p_h}, {p_w})")

    if stride is None:
        s_y, s_x = p_h // 2, p_w // 2
    elif isinstance(stride, (int, float)):
        s_y = s_x = int(stride)
    else:
        s_y, s_x = int(stride[0]), int(stride[1])

    if s_y <= 0 or s_x <= 0:
        raise ValueError(f"Stride must be positive, got ({s_y}, {s_x})")

    # If image is smaller than patch size, single patch coordinates spanning 0 to p
    if image_height <= p_h:
        y_starts = [0]
    else:
        y_starts = list(range(0, image_height - p_h + 1, s_y))
        if y_starts[-1] + p_h < image_height:
            y_starts.append(image_height - p_h)

    if image_width <= p_w:
        x_starts = [0]
    else:
        x_starts = list(range(0, image_width - p_w + 1, s_x))
        if x_starts[-1] + p_w < image_width:
            x_starts.append(image_width - p_w)

    coords: list[PatchCoordinates] = []
    for y in sorted(set(y_starts)):
        for x in sorted(set(x_starts)):
            coords.append(
                PatchCoordinates(
                    y_min=y,
                    y_max=min(y + p_h, max(image_height, p_h)),
                    x_min=x,
                    x_max=min(x + p_w, max(image_width, p_w)),
                )
            )
    return coords


def extract_patch(
    image: np.ndarray,
    mask: np.ndarray | None,
    coords: PatchCoordinates,
    target_size: tuple[int, int] | int = 512,
    image_pad_mode: str = "reflect",
) -> tuple[np.ndarray, np.ndarray | None, bool]:
    """Extract a patch from an image and optional mask with zero/reflect boundary padding.

    Args:
        image: RGB image array of shape (H, W, 3).
        mask: Optional 4-channel binary mask of shape (H, W, 4).
        coords: Bounding box coordinates.
        target_size: Desired patch (height, width).
        image_pad_mode: OpenCV / numpy padding mode for RGB image when padding is required.

    Returns:
        Tuple (image_patch, mask_patch, was_padded).
    """
    if isinstance(target_size, (int, float)):
        t_h = t_w = int(target_size)
    else:
        t_h, t_w = int(target_size[0]), int(target_size[1])

    h, w = image.shape[:2]
    y1, y2 = coords.y_min, coords.y_max
    x1, x2 = coords.x_min, coords.x_max

    # Valid slice within image bounds
    slice_y1 = max(0, y1)
    slice_y2 = min(h, y2)
    slice_x1 = max(0, x1)
    slice_x2 = min(w, x2)

    img_crop = image[slice_y1:slice_y2, slice_x1:slice_x2]
    mask_crop = mask[slice_y1:slice_y2, slice_x1:slice_x2] if mask is not None else None

    crop_h, crop_w = img_crop.shape[:2]
    needs_pad = (crop_h != t_h) or (crop_w != t_w)

    if not needs_pad:
        return img_crop, mask_crop, False

    # Pad image to target dimensions (pad_bottom, pad_right)
    pad_h = t_h - crop_h
    pad_w = t_w - crop_w
    if pad_h < 0 or pad_w < 0:
        raise ValueError(
            f"Cropped patch ({crop_h}, {crop_w}) exceeds target size ({t_h}, {t_w})"
        )

    pad_spec_img = ((0, pad_h), (0, pad_w), (0, 0))
    if image_pad_mode == "reflect" and crop_h > 1 and crop_w > 1:
        img_patch = np.pad(img_crop, pad_spec_img, mode="reflect")
    else:
        img_patch = np.pad(img_crop, pad_spec_img, mode="constant", constant_values=0)

    if mask_crop is not None:
        # Mask padding is ALWAYS zero so boundary padding never introduces false lesions
        pad_spec_mask = ((0, pad_h), (0, pad_w), (0, 0))
        mask_patch = np.pad(mask_crop, pad_spec_mask, mode="constant", constant_values=0)
    else:
        mask_patch = None

    return img_patch, mask_patch, True


def categorize_patch(
    mask_patch: np.ndarray,
    min_pixels_threshold: int = 1,
) -> tuple[str, dict[str, int], bool]:
    """Determine the primary lesion category and lesion presence for a 4-channel mask patch.

    Rarity Hierarchy:
        If multiple lesion classes coexist within a single patch, the patch is
        categorized according to the rarest lesion present:
        ``MA > SE > HE > EX``. If no lesion contains at least
        ``min_pixels_threshold`` positive pixels, it is labeled ``"background"``.

    Args:
        mask_patch: 4-channel binary mask of shape (H, W, 4) in [MA, HE, EX, SE] order.
        min_pixels_threshold: Minimum positive pixels to consider a lesion present.

    Returns:
        Tuple of (primary_category, pixel_counts_dict, contains_lesion_bool).
    """
    if mask_patch.ndim != 3 or mask_patch.shape[2] != len(CANONICAL_LESION_ORDER):
        raise ValueError(
            f"Expected mask of shape (H, W, {len(CANONICAL_LESION_ORDER)}), got {mask_patch.shape}"
        )

    counts: dict[str, int] = {}
    for idx, ch_name in enumerate(CANONICAL_LESION_ORDER):
        counts[ch_name] = int(np.count_nonzero(mask_patch[..., idx] > 0))

    has_any_lesion = any(c >= min_pixels_threshold for c in counts.values())

    if not has_any_lesion:
        return "background", counts, False

    # Assign category according to defined rarity hierarchy
    for rare_ch in LESION_RARITY_PRIORITY:
        if counts.get(rare_ch, 0) >= min_pixels_threshold:
            return rare_ch, counts, True

    # Fallback to class with maximum pixel count if outside priority list
    max_ch = max(counts.keys(), key=lambda k: counts[k])
    return max_ch, counts, True


class PatchRetinalSegmentationDataset(Dataset):
    """High-resolution patch dataset with lesion-aware sampling for retinal lesion segmentation.

    Features:
    - Extracts aligned (patch_size, patch_size) patches from full-resolution FOV images.
    - Preserves canonical 4-channel masks: [MA, HE, EX, SE].
    - Supports configurable lesion-aware sampling probabilities to balance rare lesions (MA).
    - Preserves a controlled proportion of background-dominant patches.
    - Compatible with standard PyTorch DataLoaders and worker seeding.
    - Strictly validates that only records with split == "train" are included.
    """

    def __init__(
        self,
        manifest_or_records: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]],
        split: str = "train",
        dataset_config: Mapping[str, Any] | str | Path = "dataset.yaml",
        patch_size: int = 512,
        stride: int | None = None,
        sampling_weights: Mapping[str, float] | None = None,
        background_ratio: float = 0.25,
        epoch_samples: int = 500,
        seed: int = 42,
        transform: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
        cache_fov_images: bool = True,
    ) -> None:
        """Initialize high-resolution patch dataset.

        Args:
            manifest_or_records: Manifest path or sequence of record dictionaries.
            split: Must be 'train' (validation/test must use full-resolution evaluation).
            dataset_config: Dataset configuration file path or dictionary.
            patch_size: Height and width of square patches (typically 512 or 384).
            stride: Stride between extracted candidate patches (default: patch_size // 2).
            sampling_weights: Hierarchical class sampling weights for lesion categories.
            background_ratio: Expected probabilistic quota for background-dominant patches
                (drawn via Bernoulli trial with probability p = background_ratio per sample
                when background candidates are available). Note: This is an expected probability
                per sample rather than a deterministic, exact slice count per epoch.
            epoch_samples: Number of patches drawn per training epoch.
            seed: Base random seed for sampling reproducibility.
            transform: Optional spatial/color augmentations applied per patch.
            cache_fov_images: Whether to cache full-resolution FOV crops in RAM.
        """
        super().__init__()
        from src.data.dataset import _extract_records_from_input, _validate_manifest_record

        norm_split = str(split).strip().lower()
        if norm_split != "train":
            raise ValueError(
                f"PatchRetinalSegmentationDataset is designed exclusively for training patch extraction. "
                f"Got split={split!r}. Validation and testing must evaluate complete full-resolution images."
            )
        self.split = norm_split

        all_records = _extract_records_from_input(manifest_or_records)
        train_records: list[dict[str, Any]] = []
        for idx, rec in enumerate(all_records):
            _validate_manifest_record(rec, idx)
            rec_split = str(rec.get("split", "")).strip().lower()
            if rec_split != "train":
                raise ValueError(
                    f"Data leakage protection violation: Record {rec.get('sample_id')} has "
                    f"split={rec_split!r}, but PatchRetinalSegmentationDataset only permits 'train' records."
                )
            train_records.append(dict(rec))

        if not train_records:
            raise ValueError("No training records provided to PatchRetinalSegmentationDataset.")

        self.records = train_records
        self.patch_size = int(patch_size)
        if self.patch_size not in (384, 512) and self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive (typically 384 or 512), got {patch_size}")

        self.stride = int(stride) if stride is not None else self.patch_size // 2
        self.background_ratio = float(background_ratio)
        if not (0.0 <= self.background_ratio <= 1.0):
            raise ValueError(f"background_ratio must be in [0.0, 1.0], got {background_ratio}")

        self.sampling_weights = dict(DEFAULT_SAMPLING_WEIGHTS)
        if sampling_weights is not None:
            for k, v in sampling_weights.items():
                if float(v) < 0.0:
                    raise ValueError(f"Sampling weight for {k} must be non-negative, got {v}")
                self.sampling_weights[k] = float(v)

        self.epoch_samples = int(epoch_samples)
        if self.epoch_samples <= 0:
            raise ValueError(f"epoch_samples must be positive, got {epoch_samples}")

        if isinstance(seed, bool) or not isinstance(seed, int) or not (0 <= seed <= MAX_UINT32_SEED):
            raise ValueError(f"seed must be in [0, {MAX_UINT32_SEED}], got {seed!r}")
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.cache_fov_images = bool(cache_fov_images)
        self._fov_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        if isinstance(dataset_config, (str, Path)):
            self.dataset_config = load_yaml_config(dataset_config, validate=False)
        else:
            self.dataset_config = dict(dataset_config)

        # Standard augmentation transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = build_segmentation_transforms(
                split="train",
                dataset_config=self.dataset_config,
                image_size=(self.patch_size, self.patch_size),
            )

        # Index patch grid and build category pools
        self._index_patch_pools()

    def _load_fov_sample(self, record_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Load and FOV-crop a training sample, caching in memory if enabled."""
        if self.cache_fov_images and record_idx in self._fov_cache:
            return self._fov_cache[record_idx]

        record = self.records[record_idx]
        img_path = resolve_project_path(record["image_path"])
        raw_mask_paths = record["mask_paths"]

        resolved_mask_paths: dict[str, Path | None] = {}
        for cls_name in CANONICAL_LESION_CHANNELS:
            m_val = raw_mask_paths.get(cls_name)
            resolved_mask_paths[cls_name] = (
                resolve_project_path(m_val) if m_val is not None else None
            )

        image_rgb, mask_res = load_sample_image_and_mask(
            image_path=img_path,
            mask_paths=resolved_mask_paths,
            threshold=0,
        )
        mask_arr = mask_res.mask  # (H, W, 4), uint8 {0, 1}

        # Synchronized FOV crop
        cropped_img, cropped_mask, _ = crop_image_and_mask_to_fov(
            image=image_rgb,
            mask=mask_arr,
            threshold=10,
            margin=0,
            preserve_boundary_lesions=True,
        )
        if self.cache_fov_images:
            self._fov_cache[record_idx] = (cropped_img, cropped_mask)
        return cropped_img, cropped_mask

    def _index_patch_pools(self) -> None:
        """Tile all training images into patch grids and assign each patch to a category pool."""
        logger.info(
            "Indexing high-resolution patches for %d training records (patch_size=%d, stride=%d)...",
            len(self.records),
            self.patch_size,
            self.stride,
        )
        self.pools: dict[str, list[PatchMetadata]] = {
            "MA": [],
            "SE": [],
            "HE": [],
            "EX": [],
            "background": [],
        }
        self.all_patches: list[PatchMetadata] = []

        total_patches = 0
        for rec_idx, rec in enumerate(self.records):
            img_fov, mask_fov = self._load_fov_sample(rec_idx)
            h, w = img_fov.shape[:2]

            coords_list = compute_patch_grid_coordinates(
                image_height=h,
                image_width=w,
                patch_size=self.patch_size,
                stride=self.stride,
            )

            for coords in coords_list:
                _, mask_patch, is_padded = extract_patch(
                    image=img_fov,
                    mask=mask_fov,
                    coords=coords,
                    target_size=self.patch_size,
                )
                assert mask_patch is not None
                cat, counts, has_lesion = categorize_patch(mask_patch, min_pixels_threshold=1)

                meta = PatchMetadata(
                    sample_id=str(rec["sample_id"]),
                    record_index=rec_idx,
                    coordinates=coords,
                    pixel_counts=counts,
                    primary_category=cat,
                    contains_lesion=has_lesion,
                    is_padded=is_padded,
                )
                self.pools[cat].append(meta)
                self.all_patches.append(meta)
                total_patches += 1

        logger.info(
            "Indexed %d candidate patches across %d records: MA=%d, SE=%d, HE=%d, EX=%d, BG=%d",
            total_patches,
            len(self.records),
            len(self.pools["MA"]),
            len(self.pools["SE"]),
            len(self.pools["HE"]),
            len(self.pools["EX"]),
            len(self.pools["background"]),
        )

    def sample_patch_metadata(self) -> PatchMetadata:
        """Sample a patch according to lesion-aware probabilities and probabilistic background quota.

        On each sample, a Bernoulli trial with probability background_ratio determines whether
        to sample from the background candidate pool (when background candidates exist).
        If not sampled as background (or if background candidates are absent), sampling selects
        a lesion class according to normalized sampling_weights.
        """
        # 1. Decide if sampling from background quota or lesion pool
        lesion_categories = [c for c in ("MA", "SE", "HE", "EX") if len(self.pools[c]) > 0]
        has_any_lesion_patches = len(lesion_categories) > 0
        has_background_patches = len(self.pools["background"]) > 0

        sample_background = (
            (self.rng.random() < self.background_ratio)
            or not has_any_lesion_patches
        ) and has_background_patches

        if sample_background:
            idx = int(self.rng.integers(0, len(self.pools["background"])))
            return self.pools["background"][idx]

        if not has_any_lesion_patches:
            # Fallback to any patch if no lesions exist at all
            idx = int(self.rng.integers(0, len(self.all_patches)))
            return self.all_patches[idx]

        # 2. Weighted choice among available lesion classes
        weights = [self.sampling_weights.get(c, 1.0) for c in lesion_categories]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]

        selected_cat = str(self.rng.choice(lesion_categories, p=probs))
        pool = self.pools[selected_cat]
        idx = int(self.rng.integers(0, len(pool)))
        return pool[idx]

    def __len__(self) -> int:
        return self.epoch_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch a lesion-aware patch with spatial augmentations and normalization.

        Returns:
            Tuple of:
            - image_tensor: (3, patch_size, patch_size) float32 normalized tensor.
            - mask_tensor: (4, patch_size, patch_size) float32 binary {0, 1} tensor.
        """
        meta = self.sample_patch_metadata()
        img_fov, mask_fov = self._load_fov_sample(meta.record_index)

        img_patch, mask_patch, _ = extract_patch(
            image=img_fov,
            mask=mask_fov,
            coords=meta.coordinates,
            target_size=self.patch_size,
        )
        assert mask_patch is not None

        # Apply spatial/photometric augmentations
        if callable(self.transform):
            try:
                res = self.transform(image=img_patch, mask=mask_patch)
                if isinstance(res, tuple):
                    img_aug, mask_aug = res
                elif isinstance(res, Mapping):
                    img_aug, mask_aug = res["image"], res["mask"]
                else:
                    img_aug, mask_aug = res[0], res[1]
            except TypeError:
                res = self.transform(img_patch, mask_patch)
                if isinstance(res, tuple):
                    img_aug, mask_aug = res
                elif isinstance(res, Mapping):
                    img_aug, mask_aug = res["image"], res["mask"]
                else:
                    img_aug, mask_aug = res[0], res[1]
        else:
            img_aug, mask_aug = img_patch, mask_patch

        # Convert to PyTorch tensors (C, H, W)
        if isinstance(img_aug, torch.Tensor):
            img_tensor = img_aug.float()
        else:
            img_tensor = torch.from_numpy(img_aug).permute(2, 0, 1).float()

        if isinstance(mask_aug, torch.Tensor):
            mask_tensor = (mask_aug.float() > 0.5).float()
            if mask_tensor.ndim == 3 and mask_tensor.shape[-1] == 4:
                mask_tensor = mask_tensor.permute(2, 0, 1)
        else:
            mask_arr = (mask_aug > 0).astype(np.float32)
            mask_tensor = torch.from_numpy(mask_arr).permute(2, 0, 1)

        return img_tensor, mask_tensor

