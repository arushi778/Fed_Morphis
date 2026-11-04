"""PyTorch Dataset and DataLoader construction for DDR and IDRiD lesion segmentation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.fov import (
    DEFAULT_FOV_MARGIN,
    DEFAULT_FOV_THRESHOLD,
    DEFAULT_MIN_FOV_FRACTION,
    FOVCropMetadata,
    crop_image_and_mask_to_fov,
)
from src.data.inventory import GRADING_DIR_KEYWORDS
from src.data.manifests import load_manifest_json
from src.data.masks import (
    CANONICAL_LESION_CHANNELS,
    load_sample_image_and_mask,
)
from src.data.transforms import (
    RetinalSegmentationTransform,
    build_segmentation_transforms,
)
from src.utils.config import load_yaml_config
from src.utils.paths import resolve_project_path
from src.utils.seed import MAX_UINT32_SEED, seed_worker


class DatasetSampleError(ValueError):
    """Raised when a manifest record or segmentation sample is invalid."""


def _extract_records_from_input(
    manifest_or_records: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize a manifest path, manifest dict, or list of record dicts into a list of records."""
    if isinstance(manifest_or_records, (str, Path)):
        manifest_dict = load_manifest_json(manifest_or_records)
        if not isinstance(manifest_dict, Mapping) or "records" not in manifest_dict:
            raise DatasetSampleError(
                f"Manifest file '{manifest_or_records}' does not contain a 'records' list."
            )
        raw_records = manifest_dict["records"]
    elif isinstance(manifest_or_records, Mapping):
        if "records" not in manifest_or_records:
            raise DatasetSampleError("Manifest mapping is missing required 'records' key.")
        raw_records = manifest_or_records["records"]
    elif isinstance(manifest_or_records, Sequence):
        raw_records = manifest_or_records
    else:
        raise DatasetSampleError(
            f"Unsupported manifest_or_records type {type(manifest_or_records).__name__}."
        )

    if not isinstance(raw_records, Sequence):
        raise DatasetSampleError("Manifest 'records' must be a sequence of record mappings.")

    return [dict(r) for r in raw_records]


def _validate_manifest_record(record: Mapping[str, Any], index: int) -> None:
    """Validate that a manifest record is a genuine segmentation record and not a grading sample."""
    required_keys = ("dataset_name", "sample_id", "split", "image_path", "mask_paths")
    missing = [k for k in required_keys if k not in record]
    if missing:
        raise DatasetSampleError(
            f"Manifest record at index {index} is missing required field(s): {missing}."
        )

    img_path_str = str(record["image_path"])
    if not img_path_str.strip():
        raise DatasetSampleError(
            f"Manifest record '{record.get('sample_id')}' has an empty image_path."
        )

    lowered_parts = [p.lower().strip() for p in Path(img_path_str).parts]
    for part in lowered_parts:
        if part in GRADING_DIR_KEYWORDS or any(gk in part for gk in GRADING_DIR_KEYWORDS):
            raise DatasetSampleError(
                f"Grading image '{img_path_str}' (sample_id='{record['sample_id']}') "
                "cannot be loaded as a segmentation sample."
            )

    mask_paths = record["mask_paths"]
    if not isinstance(mask_paths, Mapping):
        raise DatasetSampleError(
            f"Manifest record '{record['sample_id']}' must provide a 'mask_paths' mapping."
        )

    if "channel_order" in record:
        ch_order = tuple(str(c).upper() for c in record["channel_order"])
        if ch_order != CANONICAL_LESION_CHANNELS:
            raise DatasetSampleError(
                f"Manifest record '{record['sample_id']}' has invalid channel_order "
                f"{ch_order}; expected {CANONICAL_LESION_CHANNELS}."
            )


class RetinalLesionSegmentationDataset(Dataset):
    """PyTorch Dataset for DDR and IDRiD four-channel retinal lesion segmentation.

    Output Format:
    - Image tensor: ``(3, H, W)`` of dtype ``torch.float32`` (RGB channel order).
    - Mask tensor: ``(4, H, W)`` of dtype ``mask_dtype`` (default ``torch.float32``)
      with binary values ``{0, 1}`` and fixed channel order ``[MA, HE, EX, SE]``.
    - When ``return_metadata=True``, returns ``(image_tensor, mask_tensor, metadata_dict)``.
    """

    def __init__(
        self,
        manifest_or_records: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]],
        split: str | None = None,
        transform: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
        dataset_config: Mapping[str, Any] | str | Path = "dataset.yaml",
        image_size: tuple[int, int] | None = None,
        fov_crop: bool | None = None,
        return_metadata: bool = False,
        mask_dtype: torch.dtype = torch.float32,
        allow_empty: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(dataset_config, (str, Path)):
            self.config = load_yaml_config(dataset_config, validate=True)
        else:
            self.config = dict(dataset_config)

        all_records = _extract_records_from_input(manifest_or_records)

        if split is not None:
            split_norm = str(split).strip().lower()
            if split_norm == "val":
                split_norm = "valid"
            filtered_records = [
                r for r in all_records if str(r.get("split", "")).strip().lower() == split_norm
            ]
            self.split = split_norm
        else:
            filtered_records = all_records
            observed_splits = {str(r.get("split", "train")).strip().lower() for r in filtered_records}
            self.split = next(iter(observed_splits)) if len(observed_splits) == 1 else "train"

        for idx, rec in enumerate(filtered_records):
            _validate_manifest_record(rec, idx)

        if not filtered_records and not allow_empty:
            raise DatasetSampleError(
                f"No segmentation records found for split={split!r}."
            )

        self.records: list[dict[str, Any]] = filtered_records
        self.channel_order: tuple[str, ...] = CANONICAL_LESION_CHANNELS
        self.return_metadata = bool(return_metadata)
        self.mask_dtype = mask_dtype

        fov_cfg = self.config.get("fov_cropping", {})
        norm_cfg = self.config.get("normalization", {})
        if fov_crop is not None:
            self.fov_crop = bool(fov_crop)
        else:
            self.fov_crop = bool(
                fov_cfg.get("enabled", norm_cfg.get("fov_crop", True))
            )
        self.fov_threshold = int(
            fov_cfg.get("intensity_threshold", DEFAULT_FOV_THRESHOLD)
        )
        self.fov_margin = int(fov_cfg.get("margin_pixels", DEFAULT_FOV_MARGIN))
        self.min_fov_fraction = float(
            fov_cfg.get("min_fov_area_fraction", DEFAULT_MIN_FOV_FRACTION)
        )
        self.preserve_boundary_lesions = bool(
            fov_cfg.get("preserve_boundary_lesions", True)
        )
        self.mask_threshold = int(self.config.get("mask_binarization_threshold", 0))

        if transform is not None:
            self.transform = transform
        else:
            transform_split = self.split if self.split in {"train", "valid", "test"} else "test"
            self.transform = build_segmentation_transforms(
                split=transform_split,
                dataset_config=self.config,
                image_size=image_size,
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        if index < 0 or index >= len(self.records):
            raise IndexError(
                f"Dataset index {index} out of range for dataset of length {len(self.records)}."
            )

        record = self.records[index]
        sample_id = str(record["sample_id"])
        dataset_name = str(record["dataset_name"])
        rec_split = str(record["split"])

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
            threshold=self.mask_threshold,
        )
        mask_arr = mask_res.mask  # (H, W, 4), uint8 in {0, 1}, [MA, HE, EX, SE]

        if self.fov_crop:
            precomputed_fov: FOVCropMetadata | None = None
            fov_dict = record.get("fov_crop")
            if isinstance(fov_dict, Mapping) and {
                "y_min",
                "y_max",
                "x_min",
                "x_max",
                "original_height",
                "original_width",
            }.issubset(fov_dict.keys()):
                orig_h, orig_w = int(fov_dict["original_height"]), int(fov_dict["original_width"])
                if (orig_h, orig_w) == image_rgb.shape[:2]:
                    y0, y1 = int(fov_dict["y_min"]), int(fov_dict["y_max"])
                    x0, x1 = int(fov_dict["x_min"]), int(fov_dict["x_max"])
                    precomputed_fov = FOVCropMetadata(
                        y_min=y0,
                        y_max=y1,
                        x_min=x0,
                        x_max=x1,
                        original_height=orig_h,
                        original_width=orig_w,
                        cropped_height=y1 - y0,
                        cropped_width=x1 - x0,
                        threshold=int(fov_dict.get("threshold", self.fov_threshold)),
                        margin=int(fov_dict.get("margin", self.fov_margin)),
                        fallback_used=bool(fov_dict.get("fallback_used", False)),
                        fallback_reason=fov_dict.get("fallback_reason"),
                        boundary_expanded_for_mask=bool(
                            fov_dict.get("boundary_expanded_for_mask", False)
                        ),
                        lesion_pixels_outside_fov=int(
                            fov_dict.get("lesion_pixels_outside_fov", 0)
                        ),
                    )

            image_cropped, mask_cropped, fov_meta = crop_image_and_mask_to_fov(
                image=image_rgb,
                mask=mask_arr,
                threshold=self.fov_threshold,
                margin=self.fov_margin,
                min_fov_fraction=self.min_fov_fraction,
                preserve_boundary_lesions=self.preserve_boundary_lesions,
                fov_box=precomputed_fov,
            )
            assert mask_cropped is not None
        else:
            h, w = image_rgb.shape[:2]
            image_cropped = image_rgb
            mask_cropped = mask_arr
            fov_meta = FOVCropMetadata(
                y_min=0,
                y_max=h,
                x_min=0,
                x_max=w,
                original_height=h,
                original_width=w,
                cropped_height=h,
                cropped_width=w,
                threshold=self.fov_threshold,
                margin=self.fov_margin,
                fallback_used=False,
                fallback_reason=None,
            )

        transformed_image, transformed_mask = self.transform(
            image_cropped, mask_cropped
        )

        # Ensure strict binary {0, 1} semantics after transform
        binary_mask = (transformed_mask > 0).astype(np.uint8)

        # Convert (H, W, C) NumPy arrays to (C, H, W) CPU PyTorch tensors
        image_chw = np.ascontiguousarray(np.transpose(transformed_image, (2, 0, 1)))
        mask_chw = np.ascontiguousarray(np.transpose(binary_mask, (2, 0, 1)))

        image_tensor = torch.from_numpy(image_chw).to(dtype=torch.float32)
        mask_tensor = torch.from_numpy(mask_chw).to(dtype=self.mask_dtype)

        if not self.return_metadata:
            return image_tensor, mask_tensor

        metadata: dict[str, Any] = {
            "dataset_name": dataset_name,
            "sample_id": sample_id,
            "split": rec_split,
            "image_path": str(record["image_path"]),
            "channel_order": list(CANONICAL_LESION_CHANNELS),
            "missing_lesion_masks": list(mask_res.missing_classes),
            "synthesized_zero_channels": list(mask_res.synthesized_zero_classes),
            "positive_lesion_masks": list(mask_res.positive_classes),
            "original_shape": list(image_rgb.shape[:2]),
            "cropped_shape": [fov_meta.cropped_height, fov_meta.cropped_width],
            "fov_crop": fov_meta.to_dict(),
        }
        return image_tensor, mask_tensor, metadata


def segmentation_collate_fn(
    batch: Sequence[
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]
    ],
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, dict[str, list[Any]]]
):
    """Collate `(image, mask)` or `(image, mask, metadata)` tuples safely into batches."""
    first = batch[0]
    images = torch.stack([item[0] for item in batch], dim=0)
    masks = torch.stack([item[1] for item in batch], dim=0)
    if len(first) == 2:
        return images, masks

    metas = [item[2] for item in batch]  # type: ignore[misc]
    collated_meta: dict[str, list[Any]] = {
        key: [m.get(key) for m in metas] for key in metas[0].keys()
    }
    return images, masks, collated_meta


def build_segmentation_dataloader(
    dataset: Dataset,
    dataset_config: Mapping[str, Any] | str | Path = "dataset.yaml",
    batch_size: int | None = None,
    shuffle: bool | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    drop_last: bool | None = None,
    seed: int = 42,
) -> DataLoader:
    """Construct a reproducible PyTorch DataLoader without mutating global RNG state.

    Uses an isolated CPU :class:`torch.Generator` seeded with ``seed`` and
    :func:`src.utils.seed.seed_worker` as ``worker_init_fn`` so separate DataLoaders
    do not interfere with each other.
    """
    if isinstance(dataset_config, (str, Path)):
        cfg = load_yaml_config(dataset_config, validate=True)
    else:
        cfg = dict(dataset_config)

    if isinstance(seed, bool) or not isinstance(seed, int) or not (0 <= seed <= MAX_UINT32_SEED):
        raise ValueError(
            f"DataLoader seed must be an integer in [0, {MAX_UINT32_SEED}], got {seed!r}."
        )

    resolved_batch_size = int(
        batch_size if batch_size is not None else cfg.get("batch_size", 8)
    )
    if resolved_batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {resolved_batch_size}.")

    resolved_workers = int(
        num_workers if num_workers is not None else cfg.get("number_of_workers", 0)
    )
    if resolved_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {resolved_workers}.")

    resolved_pin_memory = bool(
        pin_memory if pin_memory is not None else cfg.get("pin_memory", False)
    )

    cfg_persistent = bool(
        persistent_workers
        if persistent_workers is not None
        else cfg.get("persistent_workers", False)
    )
    # PyTorch requires num_workers > 0 when persistent_workers=True
    resolved_persistent = bool(cfg_persistent and resolved_workers > 0)

    is_train_split = getattr(dataset, "split", "train") == "train"
    resolved_shuffle = bool(shuffle if shuffle is not None else is_train_split)
    resolved_drop_last = bool(
        drop_last
        if drop_last is not None
        else (cfg.get("drop_last_train", False) if is_train_split else False)
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    return DataLoader(
        dataset,
        batch_size=resolved_batch_size,
        shuffle=resolved_shuffle,
        num_workers=resolved_workers,
        pin_memory=resolved_pin_memory,
        persistent_workers=resolved_persistent,
        drop_last=resolved_drop_last,
        worker_init_fn=seed_worker,
        generator=generator,
        collate_fn=segmentation_collate_fn,
    )


__all__ = [
    "DatasetSampleError",
    "RetinalLesionSegmentationDataset",
    "RetinalSegmentationTransform",
    "build_segmentation_dataloader",
    "segmentation_collate_fn",
]

