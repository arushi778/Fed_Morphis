"""Supervised training and validation engine for retinal lesion segmentation.

Provides :class:`SupervisedSegmentationTrainer` with:
- Configurable optimizer (``AdamW``, ``Adam``, ``SGD``) and optional scheduler
- Mixed precision training (``torch.amp``) on CUDA
- Optional gradient norm clipping
- Per-epoch and per-class validation metric tracking (``[MA, HE, EX, SE]``)
- Deterministic checkpoint saving (``latest_checkpoint.pt`` and ``best_checkpoint.pt``)
- Full state restoration for resuming interrupted runs
- Strict validation that test-split samples are never accessed during training
  or validation model selection
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from src.data.masks import CANONICAL_LESION_CHANNELS
from src.models.losses import BCEDiceLoss
from src.models.metrics import SegmentationMetricTracker
from src.models.unet import PretrainedWeightsError
from src.utils.logger import get_logger
from src.utils.paths import format_project_path, resolve_project_path
from src.utils.seed import capture_rng_state, restore_rng_state

logger = get_logger("fedmorphis.trainer")

CANONICAL_LESION_ORDER: tuple[str, ...] = tuple(CANONICAL_LESION_CHANNELS)


class TestSplitLeakageError(ValueError):
    """Raised when test-split samples are passed into training or model selection."""

    __test__ = False


def verify_no_test_split_leakage(
    dataloader: DataLoader[Any],
    stage_name: str,
) -> None:
    """Ensure a DataLoader does not expose forbidden split records for ``stage_name``.

    - For ``stage_name == "train"``, rejects ``valid``, ``val``, ``validation``, and ``test``
      records (never train on validation or test samples).
    - For ``stage_name == "validation"``, rejects ``test`` records (never use test data
      for model selection).
    """
    forbidden_splits = (
        {"test", "valid", "val", "validation"}
        if stage_name.strip().lower() == "train"
        else {"test"}
    )
    dataset: Any = getattr(dataloader, "dataset", None)
    visited: set[int] = set()
    while dataset is not None and id(dataset) not in visited:
        visited.add(id(dataset))
        records = getattr(dataset, "records", None)
        if isinstance(records, list):
            for rec in records:
                split_val = (
                    rec.get("split")
                    if isinstance(rec, dict)
                    else getattr(rec, "split", None)
                )
                norm_split = str(split_val).strip().lower()
                if norm_split in forbidden_splits:
                    sample_id = (
                        rec.get("sample_id")
                        if isinstance(rec, dict)
                        else getattr(rec, "sample_id", "unknown")
                    )
                    raise TestSplitLeakageError(
                        f"Forbidden '{norm_split}'-split sample '{sample_id}' detected in "
                        f"'{stage_name}' DataLoader. Test data must never be used "
                        "for model selection, and validation/test samples must never be "
                        "used for training."
                    )
        dataset = getattr(dataset, "dataset", None)


def _verify_batch_metadata_no_test_split(
    metadata: dict[str, Any] | None,
    stage_name: str,
) -> None:
    """Check batch metadata at runtime to prevent split leakage."""
    forbidden_splits = (
        {"test", "valid", "val", "validation"}
        if stage_name.strip().lower() == "train"
        else {"test"}
    )
    if isinstance(metadata, dict):
        splits = metadata.get("split")
        if isinstance(splits, (list, tuple)):
            for split_val in splits:
                norm_split = str(split_val).strip().lower()
                if norm_split in forbidden_splits:
                    raise TestSplitLeakageError(
                        f"Forbidden '{norm_split}'-split sample detected in batch metadata during '{stage_name}'."
                    )
        elif isinstance(splits, str) and splits.strip().lower() in forbidden_splits:
            raise TestSplitLeakageError(
                f"Forbidden '{splits.strip().lower()}'-split sample detected in batch metadata during '{stage_name}'."
            )


def _unpack_segmentation_batch(
    batch: Any,
    stage_name: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any] | None]:
    """Unpack either a ``(image, mask[, metadata])`` tuple or ``{'image', 'mask'}`` dict batch."""
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

    _verify_batch_metadata_no_test_split(metadata, stage_name=stage_name)
    return images, masks, metadata


def build_optimizer_from_config(
    model: nn.Module,
    config: dict[str, Any],
) -> Optimizer:
    """Construct an optimizer from the ``training`` block of ``model.yaml``."""
    train_cfg = config.get("training", {})
    if not isinstance(train_cfg, dict):
        train_cfg = {}

    opt_name = str(train_cfg.get("optimizer", "adamw")).strip().lower()
    lr = float(train_cfg.get("learning_rate", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("Model has no trainable parameters requiring gradients.")

    if opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if opt_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if opt_name == "sgd":
        momentum = float(train_cfg.get("momentum", 0.9))
        return torch.optim.SGD(
            params, lr=lr, weight_decay=weight_decay, momentum=momentum
        )

    raise ValueError(
        f"Unsupported optimizer '{opt_name}'. Supported: ['adamw', 'adam', 'sgd']."
    )


class SupervisedSegmentationTrainer:
    """Reusable supervised training and validation engine for U-Net segmentation.

    Args:
        model: Segmentation model mapping ``(B, 3, H, W)`` to ``(B, 4, H, W)`` logits.
        criterion: Loss module (e.g., :class:`BCEDiceLoss`).
        optimizer: PyTorch optimizer.
        device: Target ``torch.device`` (``cpu`` or ``cuda``).
        scheduler: Optional learning rate scheduler stepped once per epoch.
        checkpoint_dir: Directory where ``latest_checkpoint.pt`` and
            ``best_checkpoint.pt`` are saved.
        mixed_precision: Enable ``torch.amp`` mixed precision when ``device.type == "cuda"``.
        gradient_clip_max_norm: Optional max norm for ``clip_grad_norm_``.
        metric_threshold: Probability threshold in ``(0, 1)`` for binary predictions.
        monitor_metric: Metric key monitored for best-checkpoint selection
            (default: ``"val_dice_macro"``).
        monitor_mode: ``"max"`` or ``"min"`` for ``monitor_metric``.
        seed: Global random seed recorded in checkpoints.
        config: Full configuration snapshot recorded in checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device | str = "cpu",
        scheduler: LRScheduler | None = None,
        checkpoint_dir: str | Path = "checkpoints/segmentation",
        mixed_precision: bool = False,
        gradient_clip_max_norm: float | None = 1.0,
        metric_threshold: float = 0.5,
        monitor_metric: str = "val_dice_macro",
        monitor_mode: str = "max",
        seed: int = 42,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.device = torch.device(device)
        if getattr(model, "pretrained_requested", False) and not getattr(
            model, "pretrained_weights_loaded", False
        ):
            raise PretrainedWeightsError(
                f"Model requested pretrained encoder weights "
                f"(source={getattr(model, 'encoder_weights_source', 'unknown')!r}), "
                f"but pretrained_weights_loaded is False "
                f"(status={getattr(model, 'weights_status', 'unknown')!r}). "
                "Refusing to silently fall back to random initialization during training."
            )
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir)

        self.amp_enabled = bool(mixed_precision and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler(
            device="cuda",
            enabled=self.amp_enabled,
        )

        if gradient_clip_max_norm is not None and gradient_clip_max_norm <= 0.0:
            raise ValueError(
                f"gradient_clip_max_norm must be > 0 or None, got {gradient_clip_max_norm}"
            )
        self.gradient_clip_max_norm = (
            float(gradient_clip_max_norm)
            if gradient_clip_max_norm is not None
            else None
        )

        if not (0.0 < metric_threshold < 1.0):
            raise ValueError(
                f"metric_threshold must be strictly between 0 and 1, got {metric_threshold}"
            )
        self.metric_threshold = float(metric_threshold)

        if monitor_mode not in {"max", "min"}:
            raise ValueError(
                f"monitor_mode must be 'max' or 'min', got '{monitor_mode}'."
            )
        self.monitor_metric = str(monitor_metric)
        self.monitor_mode = str(monitor_mode)
        self.seed = int(seed)
        self.config = dict(config) if config is not None else {}

        self.start_epoch: int = 1
        self.current_epoch: int = 0
        self.global_step: int = 0
        self.best_metric_value: float = (
            float("-inf") if self.monitor_mode == "max" else float("inf")
        )
        self.best_epoch: int = 0
        self.history: list[dict[str, Any]] = []
        self._active_dataloader_generator: torch.Generator | None = None
        self._pending_rng_state: dict[str, Any] | None = None
        self._resumed_without_rng_state: bool = False

    def _is_improvement(self, candidate: float) -> bool:
        """Return True if ``candidate`` improves upon ``self.best_metric_value``."""
        if self.monitor_mode == "max":
            return candidate > self.best_metric_value
        return candidate < self.best_metric_value

    def _resolve_monitored_metric(
        self,
        epoch_record: dict[str, Any],
    ) -> float:
        """Extract the scalar monitored metric from an epoch summary dictionary."""
        if self.monitor_metric in epoch_record:
            return float(epoch_record[self.monitor_metric])

        val_block = epoch_record.get("val", {})
        if isinstance(val_block, dict):
            # Support both "val_dice_macro" and "dice_macro" or "loss"
            stripped = (
                self.monitor_metric[4:]
                if self.monitor_metric.startswith("val_")
                else self.monitor_metric
            )
            if stripped in val_block:
                return float(val_block[stripped])
            flat_metrics = val_block.get("metrics", {}).get("flat", {})
            if stripped in flat_metrics:
                return float(flat_metrics[stripped])
            macro_metrics = val_block.get("metrics", {}).get("macro", {})
            if stripped in macro_metrics:
                return float(macro_metrics[stripped])

        raise KeyError(
            f"Monitored metric '{self.monitor_metric}' not found in epoch summary."
        )

    def train_one_epoch(
        self,
        train_loader: DataLoader[Any],
        epoch: int = 1,
        max_batches: int | None = None,
        *,
        progress_prefix: str | None = None,
        log_interval: int | None = None,
        progress_logger: logging.Logger | None = None,
    ) -> dict[str, Any]:
        """Run one training epoch and return loss and metric summary."""
        verify_no_test_split_leakage(train_loader, stage_name="train")
        self.model.train()

        tracker = SegmentationMetricTracker(
            threshold=self.metric_threshold,
            from_logits=True,
            channel_names=CANONICAL_LESION_ORDER,
        )

        total_loss_sum = 0.0
        total_bce_sum = 0.0
        total_dice_loss_sum = 0.0
        total_samples = 0
        batches_processed = 0
        grad_norm_last = 0.0
        start_time = time.perf_counter()

        try:
            loader_len = int(len(train_loader))
        except TypeError:
            loader_len = 0
        total_expected_batches = (
            min(loader_len, int(max_batches))
            if (max_batches is not None and loader_len > 0)
            else loader_len
        )
        effective_log_interval = (
            int(log_interval)
            if (log_interval is not None and log_interval > 0)
            else max(1, total_expected_batches // 4) if total_expected_batches > 0 else 10
        )
        active_logger = progress_logger if progress_logger is not None else logger

        for batch_idx, batch in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images_raw, masks_raw, _ = _unpack_segmentation_batch(
                batch, stage_name="train"
            )
            images = images_raw.to(self.device, non_blocking=True)
            masks = masks_raw.to(self.device, non_blocking=True)
            batch_size = int(images.shape[0])

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.amp_enabled,
            ):
                logits = self.model(images)
                if isinstance(self.criterion, BCEDiceLoss):
                    loss, bce_t, dice_t, _, _ = self.criterion.forward_tensors(
                        logits, masks
                    )
                    bce_val = float(bce_t.detach().item())
                    dice_val = float(dice_t.detach().item())
                else:
                    loss = self.criterion(logits, masks)
                    bce_val = float(loss.detach().item())
                    dice_val = 0.0

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss encountered at epoch {epoch}, "
                    f"batch {batch_idx}: {loss.item()}"
                )

            if self.amp_enabled:
                self.scaler.scale(loss).backward()
                if self.gradient_clip_max_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.gradient_clip_max_norm,
                    )
                    grad_norm_last = float(grad_norm.item())
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.gradient_clip_max_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.gradient_clip_max_norm,
                    )
                    grad_norm_last = float(grad_norm.item())
                self.optimizer.step()

            tracker.update(logits.detach(), masks.detach())

            loss_scalar = float(loss.detach().item())
            total_loss_sum += loss_scalar * batch_size
            total_bce_sum += bce_val * batch_size
            total_dice_loss_sum += dice_val * batch_size
            total_samples += batch_size
            batches_processed += 1
            self.global_step += 1

            if progress_prefix is not None:
                should_log_batch = (
                    batches_processed == 1
                    or (batches_processed % effective_log_interval == 0)
                    or (
                        total_expected_batches > 0
                        and batches_processed == total_expected_batches
                    )
                )
                if should_log_batch:
                    running_loss = total_loss_sum / max(1, total_samples)
                    elapsed_batch = time.perf_counter() - start_time
                    batch_denom = (
                        f"{batches_processed}/{total_expected_batches}"
                        if total_expected_batches > 0
                        else f"{batches_processed}"
                    )
                    pct_str = (
                        f" ({100.0 * batches_processed / total_expected_batches:5.1f}%)"
                        if total_expected_batches > 0
                        else ""
                    )
                    gpu_mem_str = ""
                    if self.device.type == "cuda" and torch.cuda.is_available():
                        alloc_mb = torch.cuda.memory_allocated(self.device) / (1024**2)
                        peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024**2)
                        gpu_mem_str = f" | gpu_mem={alloc_mb:.1f}MB (peak={peak_mb:.1f}MB)"
                    active_logger.info(
                        "%s | batch %s%s | samples=%d | batch_loss=%.4f | running_loss=%.4f | elapsed=%.2fs%s",
                        progress_prefix,
                        batch_denom,
                        pct_str,
                        total_samples,
                        loss_scalar,
                        running_loss,
                        elapsed_batch,
                        gpu_mem_str,
                    )

        if batches_processed == 0 or total_samples == 0:
            raise ValueError("train_loader produced zero batches during train_one_epoch.")

        metrics_summary = tracker.compute()
        elapsed = time.perf_counter() - start_time
        mean_loss = total_loss_sum / total_samples
        mean_bce = total_bce_sum / total_samples
        mean_dice_loss = total_dice_loss_sum / total_samples

        return {
            "epoch": int(epoch),
            "loss": float(mean_loss),
            "bce_loss": float(mean_bce),
            "dice_loss": float(mean_dice_loss),
            "dice_macro": float(metrics_summary["macro"]["dice"]),
            "iou_macro": float(metrics_summary["macro"]["iou"]),
            "grad_norm_last": float(grad_norm_last),
            "num_samples": int(total_samples),
            "num_batches": int(batches_processed),
            "elapsed_sec": round(elapsed, 4),
            "metrics": metrics_summary,
        }

    @torch.no_grad()
    def validate_one_epoch(
        self,
        val_loader: DataLoader[Any],
        epoch: int = 1,
        max_batches: int | None = None,
    ) -> dict[str, Any]:
        """Run one validation epoch without gradient updates."""
        verify_no_test_split_leakage(val_loader, stage_name="validation")
        self.model.eval()

        tracker = SegmentationMetricTracker(
            threshold=self.metric_threshold,
            from_logits=True,
            channel_names=CANONICAL_LESION_ORDER,
        )

        total_loss_sum = 0.0
        total_bce_sum = 0.0
        total_dice_loss_sum = 0.0
        total_samples = 0
        batches_processed = 0
        start_time = time.perf_counter()

        for batch_idx, batch in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images_raw, masks_raw, _ = _unpack_segmentation_batch(
                batch, stage_name="validation"
            )
            images = images_raw.to(self.device, non_blocking=True)
            masks = masks_raw.to(self.device, non_blocking=True)
            batch_size = int(images.shape[0])

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.amp_enabled,
            ):
                logits = self.model(images)
                if isinstance(self.criterion, BCEDiceLoss):
                    loss, bce_t, dice_t, _, _ = self.criterion.forward_tensors(
                        logits, masks
                    )
                    bce_val = float(bce_t.detach().item())
                    dice_val = float(dice_t.detach().item())
                else:
                    loss = self.criterion(logits, masks)
                    bce_val = float(loss.detach().item())
                    dice_val = 0.0

            tracker.update(logits.detach(), masks.detach())

            total_loss_sum += float(loss.detach().item()) * batch_size
            total_bce_sum += bce_val * batch_size
            total_dice_loss_sum += dice_val * batch_size
            total_samples += batch_size
            batches_processed += 1

        if batches_processed == 0 or total_samples == 0:
            raise ValueError("val_loader produced zero batches during validate_one_epoch.")

        metrics_summary = tracker.compute()
        elapsed = time.perf_counter() - start_time
        mean_loss = total_loss_sum / total_samples
        mean_bce = total_bce_sum / total_samples
        mean_dice_loss = total_dice_loss_sum / total_samples

        return {
            "epoch": int(epoch),
            "loss": float(mean_loss),
            "bce_loss": float(mean_bce),
            "dice_loss": float(mean_dice_loss),
            "dice_macro": float(metrics_summary["macro"]["dice"]),
            "iou_macro": float(metrics_summary["macro"]["iou"]),
            "num_samples": int(total_samples),
            "num_batches": int(batches_processed),
            "elapsed_sec": round(elapsed, 4),
            "metrics": metrics_summary,
        }

    def save_checkpoint(
        self,
        checkpoint_path: str | Path,
        epoch: int,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Serialize model, optimizer, scheduler, scaler, and training state."""
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "epoch": int(epoch),
            "global_step": int(self.global_step),
            "best_metric_value": float(self.best_metric_value),
            "best_epoch": int(self.best_epoch),
            "monitor_metric": self.monitor_metric,
            "monitor_mode": self.monitor_mode,
            "seed": int(self.seed),
            "config": self.config,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "scaler_state_dict": (
                self.scaler.state_dict() if self.amp_enabled else None
            ),
            "rng_state": capture_rng_state(
                dataloader_generator=self._active_dataloader_generator
            ),
            "history": self.history,
            "extra_metadata": extra_metadata or {},
        }
        torch.save(payload, path)
        return path

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        strict_model: bool = True,
        dataloader: DataLoader[Any] | None = None,
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        """Restore trainer state from a saved checkpoint file."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError(f"Invalid checkpoint payload in {path}")

        self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict_model)

        if load_optimizer and checkpoint.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if (
            load_scheduler
            and self.scheduler is not None
            and checkpoint.get("scheduler_state_dict") is not None
        ):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.amp_enabled and checkpoint.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.current_epoch = int(checkpoint.get("epoch", 0))
        self.start_epoch = self.current_epoch + 1
        self.global_step = int(checkpoint.get("global_step", 0))
        self.best_metric_value = float(
            checkpoint.get(
                "best_metric_value",
                float("-inf") if self.monitor_mode == "max" else float("inf"),
            )
        )
        self.best_epoch = int(checkpoint.get("best_epoch", 0))
        if isinstance(checkpoint.get("history"), list):
            self.history = list(checkpoint["history"])

        saved_rng = checkpoint.get("rng_state")
        if restore_rng and isinstance(saved_rng, Mapping) and len(saved_rng) > 0:
            dl_gen = (
                getattr(dataloader, "generator", None)
                if dataloader is not None
                else self._active_dataloader_generator
            )
            restore_rng_state(saved_rng, dataloader_generator=dl_gen)
            self._pending_rng_state = dict(saved_rng)
            self._resumed_without_rng_state = False
        else:
            self._pending_rng_state = None
            self._resumed_without_rng_state = bool(restore_rng and self.current_epoch > 0)

        logger.info(
            "Restored checkpoint from %s (epoch=%d, global_step=%d, best_%s=%.6f)",
            path,
            self.current_epoch,
            self.global_step,
            self.monitor_metric,
            self.best_metric_value,
        )
        return checkpoint

    def fit(
        self,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        num_epochs: int,
        max_train_batches: int | None = None,
        max_val_batches: int | None = None,
        save_checkpoints: bool = True,
    ) -> dict[str, Any]:
        """Run supervised training and validation across ``num_epochs``."""
        if num_epochs < 1:
            raise ValueError(f"num_epochs must be >= 1, got {num_epochs}")

        verify_no_test_split_leakage(train_loader, stage_name="train")
        verify_no_test_split_leakage(val_loader, stage_name="validation")

        self._active_dataloader_generator = getattr(train_loader, "generator", None)
        if self._pending_rng_state is not None:
            restore_rng_state(
                self._pending_rng_state,
                dataloader_generator=self._active_dataloader_generator,
            )
            self._pending_rng_state = None
        elif self._resumed_without_rng_state and self.start_epoch > 1:
            sampler = getattr(train_loader, "sampler", None)
            if sampler is not None and self._active_dataloader_generator is not None:
                for _ in range(1, self.start_epoch):
                    _ = list(sampler)
            self._resumed_without_rng_state = False

        latest_ckpt_path = self.checkpoint_dir / "latest_checkpoint.pt"
        best_ckpt_path = self.checkpoint_dir / "best_checkpoint.pt"

        for epoch in range(self.start_epoch, num_epochs + 1):
            self.current_epoch = epoch
            epoch_lr = float(self.optimizer.param_groups[0]["lr"])
            train_summary = self.train_one_epoch(
                train_loader=train_loader,
                epoch=epoch,
                max_batches=max_train_batches,
            )
            val_summary = self.validate_one_epoch(
                val_loader=val_loader,
                epoch=epoch,
                max_batches=max_val_batches,
            )

            if self.scheduler is not None:
                self.scheduler.step()

            next_lr = float(self.optimizer.param_groups[0]["lr"])
            train_present_only = (
                train_summary.get("metrics", {}).get("macro_present_only", {}) or {}
            )
            val_present_only = (
                val_summary.get("metrics", {}).get("macro_present_only", {}) or {}
            )
            epoch_record: dict[str, Any] = {
                "epoch": int(epoch),
                "learning_rate": epoch_lr,
                "next_learning_rate": next_lr,
                "train_loss": float(train_summary["loss"]),
                "train_dice_macro": float(train_summary["dice_macro"]),
                "train_dice_macro_present_only": (
                    float(train_present_only["dice"])
                    if train_present_only.get("dice") is not None
                    else float(train_summary["dice_macro"])
                ),
                "val_loss": float(val_summary["loss"]),
                "val_dice_macro": float(val_summary["dice_macro"]),
                "val_dice_macro_present_only": (
                    float(val_present_only["dice"])
                    if val_present_only.get("dice") is not None
                    else float(val_summary["dice_macro"])
                ),
                "val_iou_macro": float(val_summary["iou_macro"]),
                "val_iou_macro_present_only": (
                    float(val_present_only["iou"])
                    if val_present_only.get("iou") is not None
                    else float(val_summary["iou_macro"])
                ),
                "train": train_summary,
                "val": val_summary,
            }
            for ch in CANONICAL_LESION_ORDER:
                epoch_record[f"val_dice_{ch}"] = float(
                    val_summary["metrics"]["per_class"][ch]["dice"]
                )
                epoch_record[f"val_iou_{ch}"] = float(
                    val_summary["metrics"]["per_class"][ch]["iou"]
                )

            monitored_val = self._resolve_monitored_metric(epoch_record)
            is_best = self._is_improvement(monitored_val)
            if is_best:
                self.best_metric_value = float(monitored_val)
                self.best_epoch = int(epoch)
            epoch_record["is_best"] = bool(is_best)
            self.history.append(epoch_record)

            if save_checkpoints:
                self.save_checkpoint(
                    latest_ckpt_path,
                    epoch=epoch,
                    extra_metadata={"monitored_value": monitored_val, "is_best": is_best},
                )
                if is_best:
                    self.save_checkpoint(
                        best_ckpt_path,
                        epoch=epoch,
                        extra_metadata={"monitored_value": monitored_val, "is_best": True},
                    )

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_dice_macro=%.4f "
                "(MA=%.4f, HE=%.4f, EX=%.4f, SE=%.4f)%s",
                epoch,
                num_epochs,
                epoch_record["train_loss"],
                epoch_record["val_loss"],
                epoch_record["val_dice_macro"],
                epoch_record["val_dice_MA"],
                epoch_record["val_dice_HE"],
                epoch_record["val_dice_EX"],
                epoch_record["val_dice_SE"],
                " [BEST]" if is_best else "",
            )

        return {
            "epochs_completed": int(self.current_epoch),
            "global_step": int(self.global_step),
            "best_metric_name": self.monitor_metric,
            "best_metric_value": float(self.best_metric_value),
            "best_epoch": int(self.best_epoch),
            "latest_checkpoint": (
                format_project_path(latest_ckpt_path) if save_checkpoints else None
            ),
            "best_checkpoint": (
                format_project_path(best_ckpt_path) if save_checkpoints else None
            ),
            "history": self.history,
        }


def _compute_metric_delta(candidate_val: float, baseline_val: float) -> dict[str, float | None]:
    """Compute absolute difference and relative percentage change vs. baseline."""
    c_val = float(candidate_val)
    b_val = float(baseline_val)
    abs_diff = c_val - b_val
    rel_pct = ((c_val - b_val) / b_val) * 100.0 if abs(b_val) > 1e-12 else None
    return {
        "baseline": b_val,
        "candidate": c_val,
        "absolute_diff": abs_diff,
        "relative_pct_change": rel_pct,
    }


def build_validation_comparison_report(
    candidate_report: Mapping[str, Any],
    baseline_report_or_path: Mapping[str, Any] | str | Path = "logs/supervised_baseline_ddr_report.json",
) -> dict[str, Any]:
    """Build a structured validation comparison report between a candidate run and the supervised baseline.

    Note:
        This comparison uses strictly the validation split (model-selection split)
        and never accesses or establishes held-out test-set performance.
    """
    baseline_path_str: str | None = None
    if isinstance(baseline_report_or_path, (str, Path)):
        resolved_p = resolve_project_path(baseline_report_or_path)
        if not resolved_p.is_file():
            raise FileNotFoundError(
                f"Baseline report JSON not found at '{resolved_p}'."
            )
        baseline_path_str = format_project_path(resolved_p)
        baseline_data: Mapping[str, Any] = json.loads(
            resolved_p.read_text(encoding="utf-8")
        )
    elif isinstance(baseline_report_or_path, Mapping):
        baseline_data = baseline_report_or_path
    else:
        raise TypeError(
            f"Expected Mapping, str, or Path for baseline_report_or_path, got {type(baseline_report_or_path).__name__}."
        )

    cand_best = candidate_report.get("best_validation_metrics", {})
    base_best = baseline_data.get("best_validation_metrics", {})
    if not isinstance(cand_best, Mapping) or not isinstance(base_best, Mapping):
        raise ValueError(
            "Both candidate_report and baseline_report must contain a 'best_validation_metrics' mapping."
        )

    cand_macro_dice = float(
        cand_best.get(
            "val_dice_macro", candidate_report.get("best_metric_value", 0.0)
        )
    )
    base_macro_dice = float(
        base_best.get("val_dice_macro", baseline_data.get("best_metric_value", 0.0))
    )
    cand_macro_iou = float(cand_best.get("val_iou_macro", 0.0))
    base_macro_iou = float(base_best.get("val_iou_macro", 0.0))
    cand_val_loss = float(cand_best.get("val_loss", 0.0))
    base_val_loss = float(base_best.get("val_loss", 0.0))

    cand_per_class = cand_best.get("per_class", {})
    base_per_class = base_best.get("per_class", {})

    per_class_dice: dict[str, dict[str, float | None]] = {}
    per_class_iou: dict[str, dict[str, float | None]] = {}
    for ch in CANONICAL_LESION_ORDER:
        c_ch = cand_per_class.get(ch, {}) if isinstance(cand_per_class, Mapping) else {}
        b_ch = base_per_class.get(ch, {}) if isinstance(base_per_class, Mapping) else {}
        per_class_dice[ch] = _compute_metric_delta(
            float(c_ch.get("dice", 0.0)),
            float(b_ch.get("dice", 0.0)),
        )
        per_class_iou[ch] = _compute_metric_delta(
            float(c_ch.get("iou", 0.0)),
            float(b_ch.get("iou", 0.0)),
        )

    cand_model = candidate_report.get("model", {})
    base_model = baseline_data.get("model", {})

    return {
        "baseline_report_path": baseline_path_str,
        "baseline_best_epoch": int(
            baseline_data.get("best_epoch", base_best.get("epoch", 0))
        ),
        "candidate_best_epoch": int(
            candidate_report.get("best_epoch", cand_best.get("epoch", 0))
        ),
        "baseline_weights_status": (
            base_model.get("weights_status")
            if isinstance(base_model, Mapping)
            else None
        ),
        "candidate_weights_status": (
            cand_model.get("weights_status")
            if isinstance(cand_model, Mapping)
            else None
        ),
        "macro_metrics": {
            "val_dice_macro": _compute_metric_delta(cand_macro_dice, base_macro_dice),
            "val_iou_macro": _compute_metric_delta(cand_macro_iou, base_macro_iou),
            "val_loss": _compute_metric_delta(cand_val_loss, base_val_loss),
        },
        "per_class_dice": per_class_dice,
        "per_class_iou": per_class_iou,
        "improved_over_baseline_dice": bool(cand_macro_dice > base_macro_dice),
        "improved_over_baseline_iou": bool(cand_macro_iou > base_macro_iou),
        "test_split_evaluated": False,
        "validation_scope_note": (
            "Validation results reflect model-selection performance on the DDR validation split "
            "(149 images) only and do not establish held-out DDR test-set performance."
        ),
    }


__all__ = [
    "CANONICAL_LESION_ORDER",
    "SupervisedSegmentationTrainer",
    "TestSplitLeakageError",
    "build_optimizer_from_config",
    "build_validation_comparison_report",
    "verify_no_test_split_leakage",
]
