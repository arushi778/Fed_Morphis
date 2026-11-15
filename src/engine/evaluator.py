"""Held-out evaluation engine for supervised and federated retinal lesion segmentation.

Provides deterministic, no-gradient evaluation of saved segmentation checkpoints
on held-out splits (such as the 225-image DDR test split) with strict split
verification so training/validation samples cannot accidentally mix with test
evaluation, and test evaluation never mutates model or checkpoint state.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import (
    RetinalLesionSegmentationDataset,
    build_segmentation_dataloader,
)
from src.data.fov import compute_retinal_fov_mask
from src.data.manifests import load_manifest_json
from src.data.masks import CANONICAL_LESION_CHANNELS
from src.data.transforms import build_segmentation_transforms
from src.models.aupr import (
    AUPRTracker,
    DEFAULT_FROZEN_THRESHOLDS,
    DEFAULT_OPERATING_THRESHOLDS,
    PROTOCOL_NAME,
    compute_channel_pr_curve,
    compute_multilabel_aupr,
)
from src.models.losses import BCEDiceLoss, build_loss_from_config, extract_loss_components
from src.models.metrics import SegmentationMetricTracker
from src.models.unet import ResNet50UNet, build_unet_from_config
from src.utils.config import load_yaml_config
from src.utils.logger import get_logger
from src.utils.paths import format_project_path
from src.utils.seed import set_seed

logger = get_logger("fedmorphis.evaluator")

CANONICAL_LESION_ORDER: tuple[str, ...] = tuple(CANONICAL_LESION_CHANNELS)


class SplitMismatchError(ValueError):
    """Raised when a dataset or batch contains records outside the expected evaluation split."""

    __test__ = False


def _normalize_split_name(split_name: str) -> str:
    """Normalize split aliases (e.g., 'val' -> 'valid')."""
    norm = str(split_name).strip().lower()
    if norm in {"val", "validation"}:
        return "valid"
    return norm


def _validate_subset_index_type(idx: Any) -> int:
    """Validate that a Subset index is a genuine integer (rejecting bool, float, str, etc.)."""
    if isinstance(idx, (bool, np.bool_)) or not isinstance(idx, (int, np.integer)):
        raise SplitMismatchError(
            f"Invalid Subset index {idx!r} (type {type(idx).__name__}): "
            "Subset indices must be integers, not booleans, floats, or strings."
        )
    return int(idx)


def _validate_subset_indices_sequence(raw_indices_obj: Any, target_split: str) -> list[int]:
    """Validate and extract a non-empty list of integer Subset indices."""
    if isinstance(raw_indices_obj, (str, bytes)) or not hasattr(raw_indices_obj, "__iter__"):
        raise SplitMismatchError(
            f"Invalid Subset.indices of type {type(raw_indices_obj).__name__} for split '{target_split}'."
        )
    validated = [_validate_subset_index_type(i) for i in raw_indices_obj]
    if len(validated) == 0:
        raise SplitMismatchError(
            f"Evaluation subset for split '{target_split}' contains 0 indices."
        )
    return validated


def verify_exact_split(
    dataloader: DataLoader[Any],
    expected_split: str = "test",
) -> int:
    """Verify that all active records in ``dataloader`` belong exclusively to ``expected_split``.

    Respects nested :class:`torch.utils.data.Subset` index mappings so only records
    selected by the active subset are validated. Supports ``.records``,
    ``.sample_metadata``, and ``.get_sample_metadata(idx)`` metadata-only interfaces.

    Returns the number of verified records found across the underlying dataset chain.
    Raises :class:`SplitMismatchError` if any active record belongs to a different split
    or if the dataset/subset contains zero records or invalid indices.
    """
    target_split = _normalize_split_name(expected_split)
    dataset: Any = getattr(dataloader, "dataset", None)
    visited: set[int] = set()
    active_indices: list[int] | None = None
    verified_count = 0

    while dataset is not None and id(dataset) not in visited:
        visited.add(id(dataset))

        if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
            raw_indices = _validate_subset_indices_sequence(
                getattr(dataset, "indices"), target_split
            )
            if active_indices is None:
                active_indices = raw_indices
            else:
                for i in active_indices:
                    if i < 0 or i >= len(raw_indices):
                        raise SplitMismatchError(
                            f"Subset index {i} is out of bounds for inner subset of length {len(raw_indices)}."
                        )
                active_indices = [raw_indices[i] for i in active_indices]

        for attr_name in ("records", "sample_metadata"):
            seq = getattr(dataset, attr_name, None)
            if isinstance(seq, Sequence) and not isinstance(seq, (str, bytes)):
                if len(seq) == 0:
                    raise SplitMismatchError(
                        f"Evaluation DataLoader for split '{target_split}' contains 0 records."
                    )
                if active_indices is not None:
                    for i in active_indices:
                        if i < 0 or i >= len(seq):
                            raise SplitMismatchError(
                                f"Subset index {i} is out of bounds for dataset of length {len(seq)}."
                            )
                    selected_records = [seq[i] for i in active_indices]
                else:
                    selected_records = list(seq)

                if len(selected_records) == 0:
                    raise SplitMismatchError(
                        f"Evaluation DataLoader subset for split '{target_split}' contains 0 records."
                    )
                for rec in selected_records:
                    raw_split = (
                        rec.get("split")
                        if isinstance(rec, Mapping)
                        else getattr(rec, "split", None)
                    )
                    if raw_split is None or not str(raw_split).strip():
                        sample_id = (
                            rec.get("sample_id")
                            if isinstance(rec, Mapping)
                            else getattr(rec, "sample_id", "unknown")
                        )
                        raise SplitMismatchError(
                            f"Split identity missing for sample '{sample_id}' during '{target_split}' evaluation."
                        )
                    rec_split = _normalize_split_name(str(raw_split))
                    if rec_split != target_split:
                        sample_id = (
                            rec.get("sample_id")
                            if isinstance(rec, Mapping)
                            else getattr(rec, "sample_id", "unknown")
                        )
                        raise SplitMismatchError(
                            f"Split mismatch during '{target_split}' evaluation: sample "
                            f"'{sample_id}' has split='{rec_split}', expected '{target_split}'."
                        )
                    verified_count += 1

        get_meta_fn = getattr(dataset, "get_sample_metadata", None)
        if callable(get_meta_fn):
            if not hasattr(dataset, "__len__"):
                raise SplitMismatchError(
                    "Split identity cannot be established: dataset with get_sample_metadata "
                    "must implement __len__."
                )
            ds_len = int(len(dataset))
            if ds_len <= 0:
                raise SplitMismatchError(
                    f"Evaluation DataLoader for split '{target_split}' contains 0 records."
                )
            if active_indices is not None:
                indices_to_check: list[int] = []
                for i in active_indices:
                    idx_int = _validate_subset_index_type(i)
                    if idx_int < 0 or idx_int >= ds_len:
                        raise SplitMismatchError(
                            f"Subset index {idx_int} is out of bounds for dataset of length {ds_len}."
                        )
                    indices_to_check.append(idx_int)
            else:
                indices_to_check = list(range(ds_len))

            for raw_idx in indices_to_check:
                rec = get_meta_fn(raw_idx)
                raw_split = (
                    rec.get("split")
                    if isinstance(rec, Mapping)
                    else getattr(rec, "split", None)
                )
                rec_split = _normalize_split_name(str(raw_split)) if raw_split is not None else ""
                if rec_split != target_split:
                    sample_id = (
                        rec.get("sample_id")
                        if isinstance(rec, Mapping)
                        else getattr(rec, "sample_id", f"sample_{raw_idx}")
                    )
                    raise SplitMismatchError(
                        f"Split mismatch during '{target_split}' evaluation: sample "
                        f"'{sample_id}' has split='{rec_split}', expected '{target_split}'."
                    )
                verified_count += 1

        dataset = getattr(dataset, "dataset", None)

    return verified_count


def _verify_batch_metadata_exact_split(
    metadata: dict[str, Any] | None,
    expected_split: str | None,
    forbid_test: bool = False,
    require_split_metadata: bool = False,
) -> list[str]:
    """Validate batch metadata split values and extract batch sample IDs if present."""
    sample_ids: list[str] = []
    if not isinstance(metadata, dict):
        if require_split_metadata:
            raise SplitMismatchError(
                "Split identity cannot be established: dataset has no upfront records "
                "and batch metadata is missing."
            )
        return sample_ids

    raw_ids = metadata.get("sample_id")
    if isinstance(raw_ids, (list, tuple)):
        sample_ids = [str(sid) for sid in raw_ids]
    elif isinstance(raw_ids, str):
        sample_ids = [raw_ids]

    splits = metadata.get("split")
    if require_split_metadata and splits is None:
        raise SplitMismatchError(
            "Split identity cannot be established: batch metadata is missing 'split'."
        )

    if splits is not None:
        split_values: list[Any]
        if isinstance(splits, (list, tuple)):
            split_values = list(splits)
        else:
            split_values = [splits]

        if require_split_metadata and len(split_values) == 0:
            raise SplitMismatchError(
                "Split identity cannot be established: batch metadata 'split' is empty."
            )

        target_split = (
            _normalize_split_name(expected_split)
            if expected_split is not None
            else None
        )
        for s_val in split_values:
            if s_val is None or not str(s_val).strip():
                if require_split_metadata or forbid_test:
                    raise SplitMismatchError(
                        "Batch metadata contains an empty or missing split value."
                    )
                continue
            norm_s = _normalize_split_name(str(s_val))
            if forbid_test and norm_s == "test":
                raise SplitMismatchError(
                    "Batch metadata contains forbidden split='test' during validation/training evaluation."
                )
            if target_split is not None and norm_s != target_split:
                raise SplitMismatchError(
                    f"Batch metadata contains split='{norm_s}', expected '{target_split}'."
                )

    return sample_ids


def _extract_batch_metadata(batch: Any) -> dict[str, Any] | None:
    """Extract metadata dictionary from a batch tuple/list or dict if present."""
    if isinstance(batch, dict):
        meta = batch.get("metadata")
        return meta if isinstance(meta, dict) else None
    if isinstance(batch, (tuple, list)) and len(batch) >= 3 and isinstance(batch[2], dict):
        return batch[2]
    return None


def _compute_batch_fov_mask(
    images: torch.Tensor,
    masks: torch.Tensor,
    metadata: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Compute a (B, H, W) boolean retinal FOV mask aligned with a batch of images and masks."""
    batch_size = int(images.shape[0])
    h, w = int(images.shape[-2]), int(images.shape[-1])

    if isinstance(metadata, Mapping) and metadata.get("fov_mask") is not None:
        raw_fov = metadata["fov_mask"]
        if isinstance(raw_fov, torch.Tensor):
            fov_np = raw_fov.detach().cpu().numpy().astype(bool)
        elif isinstance(raw_fov, (list, tuple)):
            fov_np = np.stack([np.asarray(m, dtype=bool) for m in raw_fov], axis=0)
        else:
            fov_np = np.asarray(raw_fov, dtype=bool)
        if fov_np.ndim == 2 and batch_size == 1:
            fov_np = fov_np[None, ...]
        if fov_np.shape == (batch_size, h, w):
            return fov_np

    images_np = images.detach().cpu().numpy()
    masks_np = masks.detach().cpu().numpy()
    fov_masks = [
        compute_retinal_fov_mask(images_np[b], mask=masks_np[b])
        for b in range(batch_size)
    ]
    return np.stack(fov_masks, axis=0)


def _unpack_eval_batch(
    batch: Any,
    expected_split: str | None = "test",
    forbid_test: bool = False,
    require_split_metadata: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Unpack a tuple or dict batch and verify split metadata."""
    if isinstance(batch, dict):
        images = batch["image"]
        masks = batch["mask"]
        metadata = batch.get("metadata")
    elif isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise ValueError(
                f"Expected batch tuple of length 2 or 3, got length {len(batch)}."
            )
        images = batch[0]
        masks = batch[1]
        metadata = batch[2] if len(batch) >= 3 and isinstance(batch[2], dict) else None
    else:
        raise TypeError(
            f"Unsupported batch type {type(batch).__name__}; expected tuple or dict."
        )

    sample_ids = _verify_batch_metadata_exact_split(
        metadata,
        expected_split=expected_split,
        forbid_test=forbid_test,
        require_split_metadata=require_split_metadata,
    )
    return images, masks, sample_ids


def reconstruct_model_from_checkpoint(
    checkpoint_path: str | Path,
    fallback_model_config: Mapping[str, Any] | str | Path | None = None,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> tuple[ResNet50UNet, dict[str, Any], dict[str, Any]]:
    """Load a saved segmentation checkpoint and reconstruct the :class:`ResNet50UNet`.

    Uses the model configuration stored inside ``checkpoint["config"]["model"]``
    when available, falling back to ``fallback_model_config`` if the checkpoint
    was saved with a minimal config dict.

    Returns:
        ``(model, checkpoint_dict, resolved_configs)`` with ``model`` moved to
        ``device`` and placed in ``eval()`` mode.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    target_device = torch.device(device)
    checkpoint = torch.load(path, map_location=target_device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Invalid checkpoint payload in '{path}': missing 'model_state_dict'."
        )

    saved_cfg = checkpoint.get("config", {})
    if not isinstance(saved_cfg, Mapping):
        saved_cfg = {}

    base_cfg: dict[str, Any] = (
        dict(saved_cfg["base"])
        if isinstance(saved_cfg.get("base"), Mapping)
        else {}
    )
    data_cfg: dict[str, Any] = (
        dict(saved_cfg["data"])
        if isinstance(saved_cfg.get("data"), Mapping)
        else {}
    )

    if isinstance(saved_cfg.get("model"), Mapping):
        model_cfg: dict[str, Any] = dict(saved_cfg["model"])
    elif "architecture" in saved_cfg or "encoder" in saved_cfg:
        model_cfg = dict(saved_cfg)
    elif fallback_model_config is not None:
        if isinstance(fallback_model_config, (str, Path)):
            model_cfg = load_yaml_config(fallback_model_config, validate=True)
        else:
            model_cfg = dict(fallback_model_config)
    else:
        model_cfg = {
            "architecture": "UNet",
            "encoder": "resnet50",
            "input_channels": 3,
            "output_channels": 4,
            "pretrained": False,
        }

    # Instantiate architecture without redundant ImageNet initialization before loading full state_dict
    model = build_unet_from_config(
        model_cfg,
        pretrained_override=False,
        strict_pretrained_override=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    model.pretrained_requested = bool(model_cfg.get("pretrained", False))
    model.pretrained_weights_loaded = True
    model.weights_status = f"loaded_checkpoint:{path.name}"
    model.to(target_device)
    model.eval()

    resolved_configs = {
        "base": base_cfg,
        "data": data_cfg,
        "model": model_cfg,
    }
    return model, checkpoint, resolved_configs


@torch.no_grad()
def evaluate_segmentation_dataloader(
    model: nn.Module,
    dataloader: DataLoader[Any],
    criterion: nn.Module | None = None,
    device: torch.device | str = "cpu",
    mixed_precision: bool = False,
    metric_threshold: float = 0.5,
    expected_split: str | None = "test",
    max_batches: int | None = None,
    compute_aupr: bool = False,
) -> dict[str, Any]:
    """Evaluate a segmentation model on ``dataloader`` without gradient updates.

    Enforces ``model.eval()``, ``torch.no_grad()``, and (when ``expected_split``
    is provided) verifies that every sample in ``dataloader`` belongs to
    ``expected_split``. When ``compute_aupr=True``, also tracks per-class
    and macro Average Precision (AP) and Trapezoidal AUPR.
    """
    if not (0.0 < metric_threshold < 1.0):
        raise ValueError(
            f"metric_threshold must be strictly between 0 and 1, got {metric_threshold}"
        )

    if expected_split is not None:
        verify_exact_split(dataloader, expected_split=expected_split)

    target_device = torch.device(device)
    model = model.to(target_device)
    model.eval()

    if criterion is not None:
        criterion = criterion.to(target_device)

    amp_enabled = bool(mixed_precision and target_device.type == "cuda")
    tracker = SegmentationMetricTracker(
        threshold=float(metric_threshold),
        from_logits=True,
        channel_names=CANONICAL_LESION_ORDER,
    )
    aupr_tracker = (
        AUPRTracker(
            channel_names=CANONICAL_LESION_ORDER,
            from_logits=True,
            store_per_image=False,
        )
        if compute_aupr
        else None
    )

    total_loss_sum = 0.0
    total_bce_sum = 0.0
    total_dice_loss_sum = 0.0
    primary_comp_name = (
        getattr(criterion, "primary_component_name", "bce")
        if criterion is not None
        else "none"
    )
    secondary_comp_name = (
        getattr(
            criterion,
            "secondary_component_name",
            getattr(criterion, "component_name", "dice"),
        )
        if criterion is not None
        else "none"
    )
    total_samples = 0
    batches_processed = 0
    evaluated_sample_ids: list[str] = []
    start_time = time.perf_counter()

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images_raw, masks_raw, batch_ids = _unpack_eval_batch(
            batch, expected_split=expected_split
        )
        evaluated_sample_ids.extend(batch_ids)

        images = images_raw.to(target_device, non_blocking=True)
        masks = masks_raw.to(target_device, non_blocking=True)
        batch_size = int(images.shape[0])

        with torch.amp.autocast(
            device_type=target_device.type,
            enabled=amp_enabled,
        ):
            logits = model(images)
            if criterion is not None:
                loss, bce_val, dice_val, primary_comp_name, secondary_comp_name = (
                    extract_loss_components(criterion, logits, masks)
                )
                total_loss_sum += float(loss.detach().item()) * batch_size
                total_bce_sum += bce_val * batch_size
                total_dice_loss_sum += dice_val * batch_size

        tracker.update(logits.detach(), masks.detach())
        if aupr_tracker is not None:
            batch_fov_mask = _compute_batch_fov_mask(
                images_raw, masks_raw, metadata=_extract_batch_metadata(batch)
            )
            aupr_tracker.update(
                logits.detach(),
                masks.detach(),
                mask=batch_fov_mask,
                sample_ids=batch_ids,
            )
        total_samples += batch_size
        batches_processed += 1

    if batches_processed == 0 or total_samples == 0:
        raise ValueError("Evaluation dataloader produced zero batches.")

    metrics_summary = tracker.compute()
    elapsed = time.perf_counter() - start_time

    macro = metrics_summary["macro"]
    macro_present = metrics_summary["macro_present_only"]

    mean_loss = float(total_loss_sum / total_samples) if criterion is not None else None
    mean_bce = float(total_bce_sum / total_samples) if criterion is not None else None
    mean_dice_loss = (
        float(total_dice_loss_sum / total_samples) if criterion is not None else None
    )

    result: dict[str, Any] = {
        "split": _normalize_split_name(expected_split) if expected_split else "unspecified",
        "num_samples": int(total_samples),
        "num_batches": int(batches_processed),
        "sample_ids": evaluated_sample_ids,
        "loss": mean_loss,
        "bce_loss": mean_bce,
        "dice_loss": mean_dice_loss,
        "overlap_loss": mean_dice_loss,
        "primary_loss_component": str(primary_comp_name) if criterion is not None else None,
        "secondary_loss_component": str(secondary_comp_name) if criterion is not None else None,
        "loss_components": (
            {
                "primary_name": str(primary_comp_name),
                "secondary_name": str(secondary_comp_name),
                "primary_loss": mean_bce,
                "secondary_loss": mean_dice_loss,
            }
            if criterion is not None
            else None
        ),
        "dice_macro": float(macro["dice"]),
        "dice_macro_present_only": (
            float(macro_present["dice"])
            if macro_present.get("dice") is not None
            else None
        ),
        "iou_macro": float(macro["iou"]),
        "iou_macro_present_only": (
            float(macro_present["iou"])
            if macro_present.get("iou") is not None
            else None
        ),
        "precision_macro": float(macro["precision"]),
        "precision_macro_present_only": (
            float(macro_present["precision"])
            if macro_present.get("precision") is not None
            else None
        ),
        "recall_macro": float(macro["recall"]),
        "recall_macro_present_only": (
            float(macro_present["recall"])
            if macro_present.get("recall") is not None
            else None
        ),
        "sensitivity_macro": float(macro["sensitivity"]),
        "sensitivity_macro_present_only": (
            float(macro_present["sensitivity"])
            if macro_present.get("sensitivity") is not None
            else None
        ),
        "specificity_macro": float(macro["specificity"]),
        "specificity_macro_present_only": (
            float(macro_present["specificity"])
            if macro_present.get("specificity") is not None
            else None
        ),
        "present_classes": list(metrics_summary["present_classes"]),
        "absent_classes": list(metrics_summary["absent_classes"]),
        "per_class": metrics_summary["per_class"],
        "macro": macro,
        "macro_present_only": macro_present,
        "metrics": metrics_summary,
        "elapsed_sec": round(elapsed, 4),
    }
    if criterion is not None and secondary_comp_name not in {"none", "custom"}:
        result[f"{secondary_comp_name}_loss"] = mean_dice_loss

    if aupr_tracker is not None:
        aupr_summary = aupr_tracker.compute()
        result["aupr"] = aupr_summary
        result["ap_macro"] = aupr_summary["macro"]["ap"]
        result["ap_macro_present_only"] = aupr_summary["macro_present_only"]["ap"]
        result["aupr_macro"] = aupr_summary["macro"]["aupr_trapezoidal"]
        result["aupr_macro_present_only"] = aupr_summary["macro_present_only"]["aupr_trapezoidal"]
        for ch in CANONICAL_LESION_ORDER:
            if ch in result["per_class"]:
                result["per_class"][ch]["ap"] = aupr_summary["per_class"][ch]["ap"]
                result["per_class"][ch]["aupr_trapezoidal"] = aupr_summary["per_class"][ch]["aupr_trapezoidal"]

    return result


def evaluate_checkpoint_on_manifest(
    checkpoint_path: str | Path,
    manifest_or_records: str | Path | Mapping[str, Any] | list[dict[str, Any]],
    split: str = "test",
    dataset_config: Mapping[str, Any] | str | Path | None = None,
    model_config: Mapping[str, Any] | str | Path | None = None,
    project_config: Mapping[str, Any] | str | Path | None = None,
    batch_size: int | None = None,
    image_size: int | None = None,
    metric_threshold: float | None = None,
    device: torch.device | str = "cpu",
    report_path: str | Path | None = None,
    manifest_path_display: str | None = None,
    checkpoint_path_display: str | None = None,
) -> dict[str, Any]:
    """Load a checkpoint, evaluate it strictly on ``split`` of ``manifest_or_records``, and build a structured report."""
    target_device = torch.device(device)
    model, checkpoint, saved_configs = reconstruct_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        fallback_model_config=model_config,
        device=target_device,
        strict=True,
    )

    # Resolve base, data, and model configs (preferring checkpoint config, falling back to passed configs)
    if saved_configs["base"]:
        base_cfg = saved_configs["base"]
    elif project_config is not None:
        base_cfg = (
            load_yaml_config(project_config, validate=False)
            if isinstance(project_config, (str, Path))
            else dict(project_config)
        )
    else:
        base_cfg = {}

    if saved_configs["data"]:
        data_cfg = saved_configs["data"]
    elif dataset_config is not None:
        data_cfg = (
            load_yaml_config(dataset_config, validate=False)
            if isinstance(dataset_config, (str, Path))
            else dict(dataset_config)
        )
    else:
        data_cfg = load_yaml_config("configs/dataset.yaml", validate=False)

    model_cfg = saved_configs["model"]
    train_cfg = model_cfg.get("training", {})
    if not isinstance(train_cfg, Mapping):
        train_cfg = {}

    seed = int(
        checkpoint.get(
            "seed",
            base_cfg.get("random_seed", base_cfg.get("seed", 42)),
        )
    )
    deterministic = bool(base_cfg.get("deterministic", True))
    set_seed(seed=seed, deterministic=deterministic)

    cfg_img_size = data_cfg.get("image_size", {"height": 512, "width": 512})
    if isinstance(cfg_img_size, Mapping):
        default_img_size = int(cfg_img_size.get("height", 512))
    elif isinstance(cfg_img_size, (list, tuple)):
        default_img_size = int(cfg_img_size[0])
    else:
        default_img_size = int(cfg_img_size)
    resolved_img_size = int(image_size or default_img_size)
    resolved_batch_size = int(batch_size or data_cfg.get("batch_size", 8))
    resolved_threshold = float(
        metric_threshold
        if metric_threshold is not None
        else train_cfg.get("metric_threshold", 0.5)
    )
    mixed_precision = bool(train_cfg.get("mixed_precision", True))
    amp_enabled = bool(mixed_precision and target_device.type == "cuda")

    target_split = _normalize_split_name(split)
    eval_transform = build_segmentation_transforms(
        split=target_split if target_split in {"valid", "test"} else "test",
        dataset_config=data_cfg,
        image_size=(resolved_img_size, resolved_img_size),
    )

    test_dataset = RetinalLesionSegmentationDataset(
        manifest_or_records=manifest_or_records,
        split=target_split,
        transform=eval_transform,
        dataset_config=data_cfg,
        image_size=(resolved_img_size, resolved_img_size),
        return_metadata=True,
        cache_in_memory=(len(manifest_or_records) > 16 if isinstance(manifest_or_records, list) else True),
    )
    if test_dataset.cache_in_memory and len(test_dataset) > 8:
        test_dataset.preload_cache(max_workers=8)

    test_loader = build_segmentation_dataloader(
        dataset=test_dataset,
        dataset_config=data_cfg,
        batch_size=resolved_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(target_device.type == "cuda"),
        drop_last=False,
        seed=seed,
    )

    if target_device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(target_device)

    criterion = build_loss_from_config(model_cfg)
    eval_summary = evaluate_segmentation_dataloader(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=target_device,
        mixed_precision=mixed_precision,
        metric_threshold=resolved_threshold,
        expected_split=target_split,
    )

    gpu_memory: dict[str, Any] = {
        "cuda_used": target_device.type == "cuda",
        "device_name": (
            torch.cuda.get_device_name(target_device)
            if target_device.type == "cuda"
            else "cpu"
        ),
        "max_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(target_device))
            if target_device.type == "cuda"
            else 0
        ),
        "max_memory_allocated_mb": (
            round(torch.cuda.max_memory_allocated(target_device) / (1024**2), 2)
            if target_device.type == "cuda"
            else 0.0
        ),
        "max_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(target_device))
            if target_device.type == "cuda"
            else 0
        ),
        "max_memory_reserved_mb": (
            round(torch.cuda.max_memory_reserved(target_device) / (1024**2), 2)
            if target_device.type == "cuda"
            else 0.0
        ),
    }

    ckpt_display = (
        format_project_path(checkpoint_path_display)
        if checkpoint_path_display is not None
        else format_project_path(checkpoint_path)
    )
    manifest_display = (
        format_project_path(manifest_path_display)
        if manifest_path_display is not None
        else (
            format_project_path(manifest_or_records)
            if isinstance(manifest_or_records, (str, Path))
            else "in_memory_records"
        )
    )

    report: dict[str, Any] = {
        "status": "completed",
        "phase": "Phase 2, Task 3 - Held-Out DDR Test Evaluation",
        "checkpoint": {
            "path": ckpt_display,
            "epoch": int(checkpoint.get("epoch", 0)),
            "best_epoch": int(checkpoint.get("best_epoch", checkpoint.get("epoch", 0))),
            "global_step": int(checkpoint.get("global_step", 0)),
            "monitor_metric": checkpoint.get("monitor_metric", "val_dice_macro"),
            "best_validation_metric_value": (
                float(checkpoint["best_metric_value"])
                if checkpoint.get("best_metric_value") is not None
                else None
            ),
        },
        "dataset": {
            "manifest_path": manifest_display,
            "split": target_split,
            "sample_count": int(eval_summary["num_samples"]),
            "num_batches": int(eval_summary["num_batches"]),
            "channel_order": list(CANONICAL_LESION_ORDER),
        },
        "evaluation_config": {
            "seed": seed,
            "deterministic": deterministic,
            "device": str(target_device),
            "amp_enabled": amp_enabled,
            "image_size": [resolved_img_size, resolved_img_size],
            "batch_size": resolved_batch_size,
            "fov_crop": bool(test_dataset.fov_crop),
            "metric_threshold": resolved_threshold,
            "evaluation_transforms": data_cfg.get("evaluation_transforms", {}),
            "normalization": data_cfg.get("normalization", {}),
            "loss_config": model_cfg.get("loss", {}),
        },
        "model": model.get_model_summary(),
        "test_metrics": {
            "loss": eval_summary["loss"],
            "bce_loss": eval_summary["bce_loss"],
            "dice_loss": eval_summary["dice_loss"],
            "dice_macro": eval_summary["dice_macro"],
            "dice_macro_present_only": eval_summary["dice_macro_present_only"],
            "iou_macro": eval_summary["iou_macro"],
            "iou_macro_present_only": eval_summary["iou_macro_present_only"],
            "precision_macro": eval_summary["precision_macro"],
            "precision_macro_present_only": eval_summary["precision_macro_present_only"],
            "recall_macro": eval_summary["recall_macro"],
            "recall_macro_present_only": eval_summary["recall_macro_present_only"],
            "sensitivity_macro": eval_summary["sensitivity_macro"],
            "sensitivity_macro_present_only": eval_summary["sensitivity_macro_present_only"],
            "specificity_macro": eval_summary["specificity_macro"],
            "specificity_macro_present_only": eval_summary["specificity_macro_present_only"],
            "present_classes": eval_summary["present_classes"],
            "absent_classes": eval_summary["absent_classes"],
            "macro": eval_summary["macro"],
            "macro_present_only": eval_summary["macro_present_only"],
            "per_class": eval_summary["per_class"],
        },
        "evaluation_duration_sec": eval_summary["elapsed_sec"],
        "gpu_memory": gpu_memory,
    }

    if report_path is not None:
        out_p = Path(report_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Saved held-out test evaluation report to %s", out_p)

    return report


def _unwrap_subset_chain(dataset: Any, idx: int) -> tuple[Any, int]:
    """Resolve a dataset index through any nested Subset wrapper chain."""
    cur_ds = dataset
    cur_idx = _validate_subset_index_type(idx)
    visited: set[int] = set()
    while (
        cur_ds is not None
        and id(cur_ds) not in visited
        and hasattr(cur_ds, "dataset")
        and hasattr(cur_ds, "indices")
    ):
        visited.add(id(cur_ds))
        indices = getattr(cur_ds, "indices")
        if cur_idx < 0 or cur_idx >= len(indices):
            raise SplitMismatchError(
                f"Subset index {cur_idx} is out of bounds for subset of length {len(indices)}."
            )
        cur_idx = _validate_subset_index_type(indices[cur_idx])
        cur_ds = getattr(cur_ds, "dataset")
    return cur_ds, cur_idx


def _verify_record_split_strict(
    rec: Any,
    target_split: str,
    fallback_id: str = "unknown",
) -> str:
    """Validate a single record or metadata mapping against target_split, unconditionally blocking 'test'."""
    if not isinstance(rec, Mapping):
        raw_split = getattr(rec, "split", None)
        raw_id = getattr(rec, "sample_id", fallback_id)
    else:
        raw_split = rec.get("split")
        raw_id = rec.get("sample_id", fallback_id)

    sample_id = str(raw_id) if raw_id is not None else fallback_id
    if raw_split is None or not str(raw_split).strip():
        raise SplitMismatchError(
            f"Split identity cannot be established for sample '{sample_id}': missing 'split' field."
        )

    rec_split = _normalize_split_name(str(raw_split))
    if rec_split == "test":
        raise SplitMismatchError(
            f"Split mismatch: test split sample '{sample_id}' is strictly prohibited in evaluate_aupr_dataset."
        )
    if rec_split != target_split:
        raise SplitMismatchError(
            f"Split mismatch during '{target_split}' evaluation: sample "
            f"'{sample_id}' has split='{rec_split}', expected '{target_split}'."
        )
    return sample_id


def _extract_metadata_only_record(base_ds: Any, base_idx: int) -> Any:
    """Extract a single sample's metadata from metadata-only interfaces without calling __getitem__."""
    idx_int = _validate_subset_index_type(base_idx)
    for attr_name in ("records", "sample_metadata"):
        seq = getattr(base_ds, attr_name, None)
        if isinstance(seq, Sequence) and not isinstance(seq, (str, bytes)):
            if idx_int < 0 or idx_int >= len(seq):
                raise SplitMismatchError(
                    f"Subset index {idx_int} is out of bounds for dataset of length {len(seq)}."
                )
            return seq[idx_int]
    get_meta_fn = getattr(base_ds, "get_sample_metadata", None)
    if callable(get_meta_fn):
        if not hasattr(base_ds, "__len__"):
            raise SplitMismatchError(
                "Split identity cannot be established: dataset with get_sample_metadata must implement __len__."
            )
        ds_len = int(len(base_ds))
        if idx_int < 0 or idx_int >= ds_len:
            raise SplitMismatchError(
                f"Subset index {idx_int} is out of bounds for dataset of length {ds_len}."
            )
        return get_meta_fn(idx_int)
    return None


def _verify_aupr_dataset_upfront(
    dataset_or_loader: Any,
    target_split: str,
) -> bool:
    """Validate all active dataset records upfront using strictly metadata-only interfaces.

    Never invokes ``__getitem__`` or loads image pixels. Walks :class:`DataLoader`
    and nested :class:`torch.utils.data.Subset` chains to inspect ``.records``,
    ``.sample_metadata``, and ``.get_sample_metadata(idx)`` at the active subset
    indices. Immediately raises :class:`SplitMismatchError` (failing closed) if no
    trustworthy metadata-only source exists or if any active sample is missing,
    inconsistent, or belongs to ``'test'`` or a split other than ``target_split``.
    """
    cur: Any = (
        getattr(dataset_or_loader, "dataset", None)
        if isinstance(dataset_or_loader, DataLoader)
        else dataset_or_loader
    )
    if cur is None:
        raise SplitMismatchError("Cannot validate dataset split: dataset is None.")

    if hasattr(cur, "__len__") and len(cur) == 0:
        raise SplitMismatchError(
            f"Evaluation dataset for split '{target_split}' contains 0 samples."
        )

    active_indices: list[int] | None = None
    visited: set[int] = set()
    verified_records = False

    while cur is not None and id(cur) not in visited:
        visited.add(id(cur))

        # Only enforce dataset-level .split when not filtered by an outer Subset
        if active_indices is None:
            ds_split = getattr(cur, "split", None)
            if ds_split is not None and str(ds_split).strip():
                norm_ds_split = _normalize_split_name(str(ds_split))
                if norm_ds_split == "test":
                    raise SplitMismatchError(
                        "Dataset with split='test' is strictly prohibited in evaluate_aupr_dataset."
                    )
                if norm_ds_split != target_split:
                    raise SplitMismatchError(
                        f"Dataset split='{norm_ds_split}' does not match expected split='{target_split}'."
                    )

        if hasattr(cur, "indices") and hasattr(cur, "dataset"):
            raw_indices = _validate_subset_indices_sequence(
                getattr(cur, "indices"), target_split
            )
            if active_indices is None:
                active_indices = raw_indices
            else:
                for i in active_indices:
                    if i < 0 or i >= len(raw_indices):
                        raise SplitMismatchError(
                            f"Subset index {i} is out of bounds for inner subset of length {len(raw_indices)}."
                        )
                active_indices = [raw_indices[i] for i in active_indices]
            if len(active_indices) == 0:
                raise SplitMismatchError(
                    f"Evaluation subset for split '{target_split}' contains 0 active samples."
                )

        for attr_name in ("records", "sample_metadata"):
            seq = getattr(cur, attr_name, None)
            if isinstance(seq, Sequence) and not isinstance(seq, (str, bytes)):
                if len(seq) == 0:
                    raise SplitMismatchError(
                        f"Evaluation dataset for split '{target_split}' contains 0 records."
                    )
                if active_indices is not None:
                    for i in active_indices:
                        if i < 0 or i >= len(seq):
                            raise SplitMismatchError(
                                f"Subset index {i} is out of bounds for dataset of length {len(seq)}."
                            )
                    selected_records = [seq[i] for i in active_indices]
                else:
                    selected_records = list(seq)
                if len(selected_records) == 0:
                    raise SplitMismatchError(
                        f"Evaluation subset for split '{target_split}' contains 0 records."
                    )
                for idx, rec in enumerate(selected_records):
                    _verify_record_split_strict(
                        rec, target_split, fallback_id=f"sample_{idx}"
                    )
                verified_records = True

        get_meta_fn = getattr(cur, "get_sample_metadata", None)
        if callable(get_meta_fn):
            if not hasattr(cur, "__len__"):
                raise SplitMismatchError(
                    "Split identity cannot be established: dataset with get_sample_metadata "
                    "must implement __len__."
                )
            total_len = int(len(cur))
            if total_len <= 0:
                raise SplitMismatchError(
                    f"Evaluation dataset for split '{target_split}' contains 0 samples."
                )
            if active_indices is not None:
                indices_to_check: list[int] = []
                for i in active_indices:
                    idx_int = _validate_subset_index_type(i)
                    if idx_int < 0 or idx_int >= total_len:
                        raise SplitMismatchError(
                            f"Subset index {idx_int} is out of bounds for dataset of length {total_len}."
                        )
                    indices_to_check.append(idx_int)
            else:
                indices_to_check = list(range(total_len))

            if len(indices_to_check) == 0:
                raise SplitMismatchError(
                    f"Evaluation subset for split '{target_split}' contains 0 samples."
                )
            for out_idx, raw_idx in enumerate(indices_to_check):
                meta_item = get_meta_fn(raw_idx)
                _verify_record_split_strict(
                    meta_item, target_split, fallback_id=f"sample_{out_idx}"
                )
            verified_records = True

        cur = getattr(cur, "dataset", None)

    if not verified_records:
        raise SplitMismatchError(
            "Split identity cannot be established safely without loading samples: "
            "dataset must expose a metadata-only split interface ('.records', "
            "'.sample_metadata', or '.get_sample_metadata(idx)')."
        )

    return True


@torch.no_grad()
def evaluate_aupr_dataset(
    model: nn.Module,
    dataset: Any,
    device: torch.device | str = "cpu",
    use_sliding_window: bool = False,
    sliding_window_inferencer: Any | None = None,
    expected_split: str | None = "valid",
    batch_size: int = 4,
    operating_thresholds: Sequence[float] = DEFAULT_OPERATING_THRESHOLDS,
    metric_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate a segmentation model on a validation dataset with full AUPR and AP analysis.

    Supports both standard global-resizing evaluation (via DataLoader) and
    high-resolution full-FOV sliding-window inference. Strictly enforces that
    no test-split samples are evaluated (raising SplitMismatchError) and restores
    the model's original training/evaluation mode upon completion or exception.

    FOV Handling Protocol Distinction:
    - AUPR / AP metrics (`aupr_tracker`) pass the 2D boolean retinal FOV mask
      (:func:`src.data.fov.compute_retinal_fov_mask`) via ``mask=...`` so dark
      exterior corners outside the circular retinal fundus within the rectangular
      FOV crop are excluded and never counted as true-negative background pixels.
    - Threshold-dependent legacy metrics (`seg_tracker`: Dice, IoU, Precision,
      Recall, Specificity) evaluate on the rectangular FOV-cropped tensor to
      preserve exact backward compatibility with existing baseline reports.

    Args:
        model: Trained ResNet50-UNet segmentation model.
        dataset: Dataset or DataLoader instance.
        device: Target execution device ('cpu' or 'cuda').
        use_sliding_window: If True, executes full-FOV sliding window inference on raw crops.
        sliding_window_inferencer: Optional pre-configured SlidingWindowInferencer.
        expected_split: Expected split name (must be non-empty and must not be 'test').
        batch_size: Minibatch size for DataLoader or sliding window patches.
        operating_thresholds: Threshold operating points for curve tracking.
        metric_threshold: Threshold for standard confusion matrix metrics (default 0.5).

    Returns:
        Structured evaluation report dictionary.
    """
    if expected_split is None or not str(expected_split).strip():
        raise SplitMismatchError(
            "evaluate_aupr_dataset requires an explicit non-empty expected_split "
            "(e.g., 'valid' or 'train'); expected_split=None is prohibited."
        )

    norm_split = _normalize_split_name(expected_split)
    if norm_split == "test":
        raise SplitMismatchError(
            "evaluate_aupr_dataset is restricted to validation/training evaluation; "
            "test split evaluation is strictly prohibited."
        )

    # Validate all inspectable dataset records/samples upfront before any model forward pass
    verified_upfront = _verify_aupr_dataset_upfront(dataset, target_split=norm_split)

    target_device = torch.device(device)
    was_training = bool(model.training)
    model = model.to(target_device)
    model.eval()

    try:
        aupr_tracker = AUPRTracker(
            channel_names=CANONICAL_LESION_ORDER,
            from_logits=False if use_sliding_window else True,
            operating_thresholds=operating_thresholds,
            store_per_image=True,
        )
        seg_tracker = SegmentationMetricTracker(
            threshold=float(metric_threshold),
            from_logits=False if use_sliding_window else True,
            channel_names=CANONICAL_LESION_ORDER,
        )

        start_time = time.perf_counter()
        evaluated_sample_ids: list[str] = []

        if use_sliding_window:
            if sliding_window_inferencer is None:
                from src.engine.sliding_window import SlidingWindowInferencer

                inferencer = SlidingWindowInferencer(
                    model=model,
                    device=target_device,
                    batch_size=batch_size,
                )
            else:
                inferencer = sliding_window_inferencer

            num_samples = len(dataset)
            for idx in range(num_samples):
                base_ds, base_idx = _unwrap_subset_chain(dataset, idx)
                rec = _extract_metadata_only_record(base_ds, base_idx)

                sample_id_from_rec: str | None = None
                if rec is not None:
                    sample_id_from_rec = _verify_record_split_strict(
                        rec, norm_split, fallback_id=f"sample_{idx}"
                    )

                item_meta: Mapping[str, Any] | None = None
                if hasattr(base_ds, "_load_and_crop_raw_index"):
                    image_rgb, mask_arr, raw_meta = base_ds._load_and_crop_raw_index(base_idx)
                    if isinstance(raw_meta, Mapping):
                        item_meta = raw_meta
                else:
                    item = dataset[idx]
                    if isinstance(item, Mapping):
                        img_t = item["image"]
                        mask_t = item["mask"]
                        if isinstance(item.get("metadata"), Mapping):
                            item_meta = item["metadata"]
                    elif isinstance(item, (tuple, list)) and len(item) >= 2:
                        img_t, mask_t = item[0], item[1]
                        if len(item) >= 3 and isinstance(item[2], Mapping):
                            item_meta = item[2]
                    else:
                        raise TypeError(f"Unsupported dataset item type: {type(item)}")

                    if isinstance(img_t, torch.Tensor):
                        img_np = img_t.detach().cpu().numpy()
                    else:
                        img_np = np.asarray(img_t)
                    if img_np.ndim == 3 and img_np.shape[0] == 3 and img_np.shape[2] != 3:
                        image_rgb = np.transpose(img_np, (1, 2, 0))
                    else:
                        image_rgb = img_np

                    if isinstance(mask_t, torch.Tensor):
                        mask_np = mask_t.detach().cpu().numpy()
                    else:
                        mask_np = np.asarray(mask_t)
                    mask_arr = mask_np

                if item_meta is not None and ("split" in item_meta or rec is None):
                    sample_id_from_meta = _verify_record_split_strict(
                        item_meta, norm_split, fallback_id=sample_id_from_rec or f"sample_{idx}"
                    )
                    sample_id = sample_id_from_rec or sample_id_from_meta
                elif sample_id_from_rec is not None:
                    sample_id = sample_id_from_rec
                else:
                    raise SplitMismatchError(
                        f"Split identity cannot be established for sample index {idx} in sliding-window mode."
                    )

                evaluated_sample_ids.append(sample_id)

                # Infer continuous probabilities: (4, H, W)
                probs_4ch = inferencer.predict_fov(image_rgb, return_numpy=True)
                # Ensure mask is (4, H, W)
                if mask_arr.ndim == 3 and mask_arr.shape[-1] == 4 and mask_arr.shape[0] != 4:
                    mask_4ch = np.transpose(mask_arr, (2, 0, 1))
                else:
                    mask_4ch = mask_arr

                # Compute valid retinal FOV mask (excluding dark exterior corners)
                if item_meta is not None and item_meta.get("fov_mask") is not None:
                    raw_fov = item_meta["fov_mask"]
                    fov_mask = (
                        raw_fov.detach().cpu().numpy().astype(bool)
                        if isinstance(raw_fov, torch.Tensor)
                        else np.asarray(raw_fov, dtype=bool)
                    )
                else:
                    fov_mask = compute_retinal_fov_mask(image_rgb, mask=mask_4ch)

                aupr_tracker.update(
                    probs_4ch, mask_4ch, mask=fov_mask, sample_ids=[sample_id]
                )
                # Keep full-resolution tensors on CPU for confusion-matrix accumulation
                seg_tracker.update(
                    torch.from_numpy(probs_4ch).unsqueeze(0),
                    torch.from_numpy(mask_4ch).unsqueeze(0),
                )
        else:
            # Standard DataLoader evaluation
            from src.data.dataset import build_segmentation_dataloader

            dl = (
                dataset
                if isinstance(dataset, DataLoader)
                else build_segmentation_dataloader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                )
            )
            verify_exact_split(dl, expected_split=norm_split)

            for batch in dl:
                batch_meta = _extract_batch_metadata(batch)
                images_raw, masks_raw, batch_ids = _unpack_eval_batch(
                    batch,
                    expected_split=norm_split,
                    forbid_test=True,
                    require_split_metadata=not verified_upfront,
                )
                evaluated_sample_ids.extend(batch_ids)
                images = images_raw.to(target_device)
                masks = masks_raw.to(target_device)

                logits = model(images)
                batch_fov_mask = _compute_batch_fov_mask(
                    images_raw, masks_raw, metadata=batch_meta
                )
                aupr_tracker.update(
                    logits.detach(),
                    masks.detach(),
                    mask=batch_fov_mask,
                    sample_ids=batch_ids,
                )
                seg_tracker.update(logits.detach(), masks.detach())

        elapsed = time.perf_counter() - start_time
        aupr_summary = aupr_tracker.compute()
        seg_summary = seg_tracker.compute()

        report = {
            "status": "completed",
            "protocol": PROTOCOL_NAME,
            "split": norm_split,
            "num_samples": len(evaluated_sample_ids),
            "sample_ids": evaluated_sample_ids,
            "use_sliding_window": bool(use_sliding_window),
            "threshold_dependent_metrics": seg_summary,
            "aupr_metrics": aupr_summary,
            "ap_macro": aupr_summary["macro"]["ap"],
            "ap_macro_present_only": aupr_summary["macro_present_only"]["ap"],
            "aupr_macro": aupr_summary["macro"]["aupr_trapezoidal"],
            "aupr_macro_present_only": aupr_summary["macro_present_only"]["aupr_trapezoidal"],
            "present_classes": aupr_summary["present_classes"],
            "absent_classes": aupr_summary["absent_classes"],
            "elapsed_sec": round(elapsed, 4),
        }
        return report
    finally:
        model.train(was_training)


__all__ = [
    "SplitMismatchError",
    "evaluate_aupr_dataset",
    "evaluate_checkpoint_on_manifest",
    "evaluate_segmentation_dataloader",
    "reconstruct_model_from_checkpoint",
    "verify_exact_split",
]
