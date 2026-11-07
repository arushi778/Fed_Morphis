"""Held-out evaluation engine for supervised and federated retinal lesion segmentation.

Provides deterministic, no-gradient evaluation of saved segmentation checkpoints
on held-out splits (such as the 225-image DDR test split) with strict split
verification so training/validation samples cannot accidentally mix with test
evaluation, and test evaluation never mutates model or checkpoint state.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import (
    RetinalLesionSegmentationDataset,
    build_segmentation_dataloader,
)
from src.data.manifests import load_manifest_json
from src.data.masks import CANONICAL_LESION_CHANNELS
from src.data.transforms import build_segmentation_transforms
from src.models.losses import BCEDiceLoss, build_loss_from_config
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


def verify_exact_split(
    dataloader: DataLoader[Any],
    expected_split: str = "test",
) -> int:
    """Verify that all records in ``dataloader`` belong exclusively to ``expected_split``.

    Returns the number of verified records found across the underlying dataset chain.
    Raises :class:`SplitMismatchError` if any record belongs to a different split
    or if the dataset contains zero records.
    """
    target_split = _normalize_split_name(expected_split)
    dataset: Any = getattr(dataloader, "dataset", None)
    visited: set[int] = set()
    verified_count = 0

    while dataset is not None and id(dataset) not in visited:
        visited.add(id(dataset))
        records = getattr(dataset, "records", None)
        if isinstance(records, list):
            if len(records) == 0:
                raise SplitMismatchError(
                    f"Evaluation DataLoader for split '{target_split}' contains 0 records."
                )
            for rec in records:
                raw_split = (
                    rec.get("split")
                    if isinstance(rec, Mapping)
                    else getattr(rec, "split", None)
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
        dataset = getattr(dataset, "dataset", None)

    return verified_count


def _verify_batch_metadata_exact_split(
    metadata: dict[str, Any] | None,
    expected_split: str | None,
) -> list[str]:
    """Validate batch metadata split values and extract batch sample IDs if present."""
    sample_ids: list[str] = []
    if not isinstance(metadata, dict):
        return sample_ids

    raw_ids = metadata.get("sample_id")
    if isinstance(raw_ids, (list, tuple)):
        sample_ids = [str(sid) for sid in raw_ids]
    elif isinstance(raw_ids, str):
        sample_ids = [raw_ids]

    if expected_split is not None:
        target_split = _normalize_split_name(expected_split)
        splits = metadata.get("split")
        if isinstance(splits, (list, tuple)):
            for s_val in splits:
                norm_s = _normalize_split_name(str(s_val))
                if norm_s != target_split:
                    raise SplitMismatchError(
                        f"Batch metadata contains split='{norm_s}', expected '{target_split}'."
                    )
        elif isinstance(splits, str):
            norm_s = _normalize_split_name(splits)
            if norm_s != target_split:
                raise SplitMismatchError(
                    f"Batch metadata contains split='{norm_s}', expected '{target_split}'."
                )

    return sample_ids


def _unpack_eval_batch(
    batch: Any,
    expected_split: str | None = "test",
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
        metadata, expected_split=expected_split
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
) -> dict[str, Any]:
    """Evaluate a segmentation model on ``dataloader`` without gradient updates.

    Enforces ``model.eval()``, ``torch.no_grad()``, and (when ``expected_split``
    is provided) verifies that every sample in ``dataloader`` belongs to
    ``expected_split``.
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

    total_loss_sum = 0.0
    total_bce_sum = 0.0
    total_dice_loss_sum = 0.0
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
                if isinstance(criterion, BCEDiceLoss):
                    loss, bce_t, dice_t, _, _ = criterion.forward_tensors(
                        logits, masks
                    )
                    bce_val = float(bce_t.detach().item())
                    dice_val = float(dice_t.detach().item())
                else:
                    loss = criterion(logits, masks)
                    bce_val = float(loss.detach().item())
                    dice_val = 0.0
                total_loss_sum += float(loss.detach().item()) * batch_size
                total_bce_sum += bce_val * batch_size
                total_dice_loss_sum += dice_val * batch_size

        tracker.update(logits.detach(), masks.detach())
        total_samples += batch_size
        batches_processed += 1

    if batches_processed == 0 or total_samples == 0:
        raise ValueError("Evaluation dataloader produced zero batches.")

    metrics_summary = tracker.compute()
    elapsed = time.perf_counter() - start_time

    macro = metrics_summary["macro"]
    macro_present = metrics_summary["macro_present_only"]

    result: dict[str, Any] = {
        "split": _normalize_split_name(expected_split) if expected_split else "unspecified",
        "num_samples": int(total_samples),
        "num_batches": int(batches_processed),
        "sample_ids": evaluated_sample_ids,
        "loss": float(total_loss_sum / total_samples) if criterion is not None else None,
        "bce_loss": float(total_bce_sum / total_samples) if criterion is not None else None,
        "dice_loss": (
            float(total_dice_loss_sum / total_samples) if criterion is not None else None
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


__all__ = [
    "SplitMismatchError",
    "evaluate_checkpoint_on_manifest",
    "evaluate_segmentation_dataloader",
    "reconstruct_model_from_checkpoint",
    "verify_exact_split",
]
