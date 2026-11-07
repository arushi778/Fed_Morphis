"""Self-supervised SimCLR pretraining engine for retinal fundus representations.

Provides :class:`SimCLRTrainer` with:
- Configurable optimizer (``AdamW``, ``Adam``, ``SGD``) and LR scheduler
  (``cosine_annealing`` with linear warmup, ``step``, or ``none``)
- Mixed precision pretraining (``torch.amp``) on CUDA
- Optional gradient norm clipping
- Per-epoch contrastive loss, positive/negative similarity, top-1 contrastive
  retrieval accuracy, epoch duration, and peak GPU memory tracking
- Deterministic checkpoint saving (``latest_ssl_checkpoint.pt``,
  ``best_ssl_checkpoint.pt``, and ``best_simclr_resnet50.pt``) including both
  ``model_state_dict`` and ``encoder_state_dict`` for direct downstream U-Net transfer
- Full state restoration for resuming interrupted runs
- Strict verification that validation/test samples are never used in SSL pretraining
  and that requested pretrained encoder weights never silently fall back to random
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
    StepLR,
)
from torch.utils.data import DataLoader

from src.engine.trainer import (
    TestSplitLeakageError,
    _verify_batch_metadata_no_test_split,
    verify_no_test_split_leakage,
)
from src.models.losses import NTXentLoss
from src.models.simclr import SimCLRModel
from src.models.unet import PretrainedWeightsError
from src.utils.logger import get_logger
from src.utils.paths import format_project_path
from src.utils.seed import capture_rng_state, restore_rng_state

logger = get_logger("fedmorphis.simclr_trainer")


def _unpack_simclr_batch(
    batch: Any,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any] | None]:
    """Unpack either a ``(view1, view2[, metadata])`` tuple or ``{'view1', 'view2'}`` dict batch."""
    if isinstance(batch, dict):
        if "view1" in batch and "view2" in batch:
            view1 = batch["view1"]
            view2 = batch["view2"]
        else:
            raise KeyError("Expected 'view1' and 'view2' keys in SimCLR batch dict.")
        metadata = batch.get("metadata")
        if "mask" in batch or "masks" in batch:
            raise ValueError(
                "SimCLR pretraining batch must not contain segmentation masks."
            )
    elif isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise ValueError(
                f"Expected SimCLR batch tuple of length 2 or 3, got length {len(batch)}."
            )
        view1 = batch[0]
        view2 = batch[1]
        if len(batch) >= 3:
            if isinstance(batch[2], dict):
                metadata = batch[2]
            else:
                raise ValueError(
                    "Third element of SimCLR batch tuple must be a metadata dict "
                    "(lesion masks must never be passed to SimCLR pretraining)."
                )
        else:
            metadata = None
    else:
        raise TypeError(
            f"Unsupported SimCLR batch type {type(batch).__name__}; expected tuple or dict."
        )

    _verify_batch_metadata_no_test_split(metadata, stage_name="train")
    return view1, view2, metadata


def build_ssl_optimizer_from_config(
    model: nn.Module,
    ssl_config: Mapping[str, Any],
) -> Optimizer:
    """Construct an optimizer for SimCLR pretraining from ``ssl.yaml``."""
    cfg = dict(ssl_config.get("simclr", ssl_config))
    opt_name = str(cfg.get("optimizer", "adamw")).strip().lower()
    lr = float(cfg.get("learning_rate", 3e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-4))

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("SimCLR model has no trainable parameters requiring gradients.")

    if opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if opt_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if opt_name == "sgd":
        momentum = float(cfg.get("momentum", 0.9))
        return torch.optim.SGD(
            params, lr=lr, weight_decay=weight_decay, momentum=momentum
        )

    raise ValueError(
        f"Unsupported SSL optimizer '{opt_name}'. Supported: ['adamw', 'adam', 'sgd']."
    )


def build_ssl_scheduler_from_config(
    optimizer: Optimizer,
    ssl_config: Mapping[str, Any],
    total_epochs: int | None = None,
) -> LRScheduler | None:
    """Construct an optional learning-rate scheduler from ``ssl.yaml``."""
    cfg = dict(ssl_config.get("simclr", ssl_config))
    sched_name = str(cfg.get("lr_scheduler", "cosine_annealing")).strip().lower()
    if sched_name in {"none", "null", ""}:
        return None

    epochs = int(total_epochs if total_epochs is not None else cfg.get("epochs", 50))
    warmup_epochs = max(0, int(cfg.get("warmup_epochs", 0)))
    min_lr = float(cfg.get("min_learning_rate", 1e-6))

    if sched_name == "cosine_annealing":
        if warmup_epochs > 0 and epochs > warmup_epochs:
            warmup_sched = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_sched = CosineAnnealingLR(
                optimizer,
                T_max=max(1, epochs - warmup_epochs),
                eta_min=min_lr,
            )
            return SequentialLR(
                optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[warmup_epochs],
            )
        return CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=min_lr)

    if sched_name == "step":
        step_size = max(1, int(cfg.get("step_size", max(1, epochs // 3))))
        gamma = float(cfg.get("gamma", 0.5))
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    raise ValueError(
        f"Unsupported SSL lr_scheduler '{sched_name}'. "
        "Supported: ['cosine_annealing', 'step', 'none']."
    )


class SimCLRTrainer:
    """Self-supervised SimCLR pretraining engine.

    Args:
        model: :class:`SimCLRModel` instance.
        criterion: :class:`NTXentLoss` instance.
        optimizer: PyTorch optimizer.
        device: Target ``torch.device`` (``cpu`` or ``cuda``).
        scheduler: Optional epoch-level learning rate scheduler.
        checkpoint_dir: Directory where ``latest_ssl_checkpoint.pt``,
            ``best_ssl_checkpoint.pt``, and ``best_simclr_resnet50.pt`` are saved.
        mixed_precision: Enable ``torch.amp`` mixed precision when ``device.type == "cuda"``.
        gradient_clip_max_norm: Optional max norm for ``clip_grad_norm_``.
        seed: Global random seed recorded in checkpoints.
        config: Full configuration snapshot recorded in checkpoints.
    """

    def __init__(
        self,
        model: SimCLRModel,
        criterion: NTXentLoss,
        optimizer: Optimizer,
        device: torch.device | str = "cpu",
        scheduler: LRScheduler | None = None,
        checkpoint_dir: str | Path = "checkpoints/ssl",
        mixed_precision: bool = False,
        gradient_clip_max_norm: float | None = 1.0,
        seed: int = 42,
        config: dict[str, Any] | None = None,
    ) -> None:
        if getattr(model, "pretrained_requested", False) and not getattr(
            model, "pretrained_weights_loaded", False
        ):
            raise PretrainedWeightsError(
                f"SimCLRModel requested pretrained encoder weights "
                f"(source={getattr(model, 'encoder_weights_source', 'unknown')!r}) "
                f"but weights were not loaded (status={getattr(model, 'weights_status', 'unknown')!r}). "
                "Refusing to silently fall back to random initialization."
            )

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir)

        self.amp_enabled = bool(mixed_precision and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler(
            device="cuda",
            init_scale=4096.0,
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

        self.monitor_metric = "train_loss"
        self.monitor_mode = "min"
        self.seed = int(seed)
        self.config = dict(config) if config is not None else {}

        self.start_epoch: int = 1
        self.current_epoch: int = 0
        self.global_step: int = 0
        self.best_metric_value: float = float("inf")
        self.best_epoch: int = 0
        self.history: list[dict[str, Any]] = []
        self._active_dataloader_generator: torch.Generator | None = None
        self._pending_rng_state: dict[str, Any] | None = None
        self._resumed_without_rng_state: bool = False

    def train_one_epoch(
        self,
        train_loader: DataLoader[Any],
        epoch: int = 1,
        max_batches: int | None = None,
    ) -> dict[str, Any]:
        """Run one SimCLR contrastive pretraining epoch."""
        verify_no_test_split_leakage(train_loader, stage_name="train")
        self.model.train()

        total_loss_sum = 0.0
        total_pos_sim_sum = 0.0
        total_neg_sim_sum = 0.0
        total_acc_sum = 0.0
        total_samples = 0
        batches_processed = 0
        optimizer_steps = 0
        grad_norm_last = 0.0
        start_time = time.perf_counter()

        for batch_idx, batch in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            view1_raw, view2_raw, _ = _unpack_simclr_batch(batch)
            view1 = view1_raw.to(self.device, non_blocking=True)
            view2 = view2_raw.to(self.device, non_blocking=True)
            batch_size = int(view1.shape[0])

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.amp_enabled,
            ):
                views_out = self.model.forward_views(view1, view2)

            # Compute L2-normalized NT-Xent contrastive loss in FP32 for numerical stability
            loss_dict = self.criterion.forward_with_metrics(
                views_out["z1"].float(), views_out["z2"].float()
            )
            loss = loss_dict["loss"]

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite SimCLR contrastive loss at epoch {epoch}, "
                    f"batch {batch_idx}: {loss.item()}"
                )

            if self.amp_enabled:
                scale_before = self.scaler.get_scale()
                self.scaler.scale(loss).backward()
                if self.gradient_clip_max_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.gradient_clip_max_norm,
                    )
                    gn_val = float(grad_norm.item())
                    if torch.isfinite(grad_norm):
                        grad_norm_last = gn_val
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scaler.get_scale() >= scale_before:
                    optimizer_steps += 1
            else:
                loss.backward()
                if self.gradient_clip_max_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.gradient_clip_max_norm,
                    )
                    grad_norm_last = float(grad_norm.item())
                self.optimizer.step()
                optimizer_steps += 1

            loss_val = float(loss.detach().item())
            pos_sim_val = float(loss_dict["positive_similarity"].detach().item())
            neg_sim_val = float(loss_dict["negative_similarity"].detach().item())
            acc_val = float(loss_dict["contrastive_accuracy"].detach().item())

            total_loss_sum += loss_val * batch_size
            total_pos_sim_sum += pos_sim_val * batch_size
            total_neg_sim_sum += neg_sim_val * batch_size
            total_acc_sum += acc_val * batch_size
            total_samples += batch_size
            batches_processed += 1
            self.global_step += 1

        if batches_processed == 0 or total_samples == 0:
            raise ValueError(
                "train_loader produced zero batches during SimCLR train_one_epoch."
            )

        elapsed = time.perf_counter() - start_time
        gpu_alloc_mb = (
            round(torch.cuda.max_memory_allocated(self.device) / (1024**2), 2)
            if self.device.type == "cuda"
            else 0.0
        )
        gpu_res_mb = (
            round(torch.cuda.max_memory_reserved(self.device) / (1024**2), 2)
            if self.device.type == "cuda"
            else 0.0
        )

        return {
            "epoch": int(epoch),
            "loss": float(total_loss_sum / total_samples),
            "positive_similarity": float(total_pos_sim_sum / total_samples),
            "negative_similarity": float(total_neg_sim_sum / total_samples),
            "contrastive_accuracy": float(total_acc_sum / total_samples),
            "grad_norm_last": float(grad_norm_last),
            "num_samples": int(total_samples),
            "num_batches": int(batches_processed),
            "optimizer_steps": int(optimizer_steps),
            "elapsed_sec": round(elapsed, 4),
            "gpu_max_memory_allocated_mb": gpu_alloc_mb,
            "gpu_max_memory_reserved_mb": gpu_res_mb,
        }

    def save_checkpoint(
        self,
        checkpoint_path: str | Path,
        epoch: int,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Serialize SimCLR model, exported ResNet-50 encoder weights, optimizer, scheduler, scaler, and state."""
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
            "encoder_state_dict": self.model.export_encoder_state_dict(),
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
        """Restore SimCLRTrainer state from a saved SSL checkpoint file."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"SSL checkpoint file not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError(f"Invalid SSL checkpoint payload in {path}")

        self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict_model)
        self.model.pretrained_weights_loaded = True
        self.model.weights_status = f"resumed_ssl_checkpoint:{path.name}"

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
            checkpoint.get("best_metric_value", float("inf"))
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
            "Restored SSL checkpoint from %s (epoch=%d, global_step=%d, best_loss=%.6f)",
            path,
            self.current_epoch,
            self.global_step,
            self.best_metric_value,
        )
        return checkpoint

    def fit(
        self,
        train_loader: DataLoader[Any],
        num_epochs: int,
        max_train_batches: int | None = None,
        save_checkpoints: bool = True,
    ) -> dict[str, Any]:
        """Run SimCLR contrastive pretraining across ``num_epochs``."""
        if num_epochs < 1:
            raise ValueError(f"num_epochs must be >= 1, got {num_epochs}")

        verify_no_test_split_leakage(train_loader, stage_name="train")

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

        latest_ckpt_path = self.checkpoint_dir / "latest_ssl_checkpoint.pt"
        best_ckpt_path = self.checkpoint_dir / "best_ssl_checkpoint.pt"
        canonical_best_path = self.checkpoint_dir / "best_simclr_resnet50.pt"
        encoder_only_path = self.checkpoint_dir / "best_simclr_encoder.pt"

        for epoch in range(self.start_epoch, num_epochs + 1):
            self.current_epoch = epoch
            epoch_lr = float(self.optimizer.param_groups[0]["lr"])
            train_summary = self.train_one_epoch(
                train_loader=train_loader,
                epoch=epoch,
                max_batches=max_train_batches,
            )

            if self.scheduler is not None and train_summary.get("optimizer_steps", 1) > 0:
                self.scheduler.step()

            next_lr = float(self.optimizer.param_groups[0]["lr"])
            epoch_loss = float(train_summary["loss"])
            is_best = epoch_loss < self.best_metric_value
            if is_best:
                self.best_metric_value = epoch_loss
                self.best_epoch = int(epoch)

            epoch_record: dict[str, Any] = {
                "epoch": int(epoch),
                "learning_rate": epoch_lr,
                "next_learning_rate": next_lr,
                "train_loss": epoch_loss,
                "positive_similarity": float(train_summary["positive_similarity"]),
                "negative_similarity": float(train_summary["negative_similarity"]),
                "contrastive_accuracy": float(train_summary["contrastive_accuracy"]),
                "grad_norm_last": float(train_summary["grad_norm_last"]),
                "num_samples": int(train_summary["num_samples"]),
                "num_batches": int(train_summary["num_batches"]),
                "elapsed_sec": float(train_summary["elapsed_sec"]),
                "gpu_max_memory_allocated_mb": float(
                    train_summary["gpu_max_memory_allocated_mb"]
                ),
                "gpu_max_memory_reserved_mb": float(
                    train_summary["gpu_max_memory_reserved_mb"]
                ),
                "is_best": bool(is_best),
            }
            self.history.append(epoch_record)

            if save_checkpoints:
                self.save_checkpoint(
                    latest_ckpt_path,
                    epoch=epoch,
                    extra_metadata={"train_loss": epoch_loss, "is_best": is_best},
                )
                if is_best:
                    self.save_checkpoint(
                        best_ckpt_path,
                        epoch=epoch,
                        extra_metadata={"train_loss": epoch_loss, "is_best": True},
                    )
                    self.save_checkpoint(
                        canonical_best_path,
                        epoch=epoch,
                        extra_metadata={"train_loss": epoch_loss, "is_best": True},
                    )
                    torch.save(
                        {
                            "epoch": int(epoch),
                            "global_step": int(self.global_step),
                            "best_train_loss": float(epoch_loss),
                            "seed": int(self.seed),
                            "backbone": "resnet50",
                            "encoder_state_dict": self.model.export_encoder_state_dict(),
                        },
                        encoder_only_path,
                    )

            logger.info(
                "SimCLR Epoch %d/%d | loss=%.4f | pos_sim=%.4f | neg_sim=%.4f | "
                "acc=%.4f | lr=%.6f | time=%.2fs%s",
                epoch,
                num_epochs,
                epoch_loss,
                epoch_record["positive_similarity"],
                epoch_record["negative_similarity"],
                epoch_record["contrastive_accuracy"],
                epoch_lr,
                epoch_record["elapsed_sec"],
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
            "canonical_ssl_checkpoint": (
                format_project_path(canonical_best_path) if save_checkpoints else None
            ),
            "encoder_only_checkpoint": (
                format_project_path(encoder_only_path) if save_checkpoints else None
            ),
            "history": self.history,
        }


__all__ = [
    "SimCLRTrainer",
    "build_ssl_optimizer_from_config",
    "build_ssl_scheduler_from_config",
]
