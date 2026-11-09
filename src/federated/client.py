"""Flower NumPyClient and parameter serialization for federated retinal lesion segmentation.

Provides:
- Lossless, validated conversion between PyTorch ``model.state_dict()`` and
  Flower ``NDArrays`` / :class:`flwr.common.Parameters`, preserving exact key order,
  tensor shapes, floating-point dtypes, and integer buffers (such as BatchNorm
  ``num_batches_tracked`` of dtype ``int64``).
- :class:`FedMorphisNumPyClient` (subclassing :class:`flwr.client.NumPyClient`)
  wrapping :class:`src.federated.data.ClientDataBundle`,
  :class:`src.models.unet.ResNet50UNet`, and
  :class:`src.engine.trainer.SupervisedSegmentationTrainer`.
- :func:`build_federated_client` factory function for deterministic client construction.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
import time
from typing import Any

from flwr.client import Client, NumPyClient
from flwr.common import (
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.masks import CANONICAL_LESION_CHANNELS
from src.engine.evaluator import verify_exact_split
from src.engine.trainer import (
    SupervisedSegmentationTrainer,
    build_optimizer_from_config,
    verify_no_test_split_leakage,
)
from src.federated.data import (
    ClientDataBundle,
    FederatedDataError,
    build_federated_client_dataloaders,
)
from src.models.losses import build_loss_from_config
from src.models.unet import build_unet_from_config
from src.utils.config import load_yaml_config
from src.utils.device import resolve_device
from src.utils.logger import get_logger
from src.utils.seed import MAX_UINT32_SEED, set_seed

logger = get_logger("fedmorphis.federated.client")


class ParameterSerializationError(ValueError):
    """Raised when model parameter serialization or deserialization validation fails."""


_TORCH_TO_NUMPY_DTYPE: dict[torch.dtype, np.dtype[Any]] = {
    torch.float16: np.dtype(np.float16),
    torch.float32: np.dtype(np.float32),
    torch.float64: np.dtype(np.float64),
    torch.uint8: np.dtype(np.uint8),
    torch.int8: np.dtype(np.int8),
    torch.int16: np.dtype(np.int16),
    torch.int32: np.dtype(np.int32),
    torch.int64: np.dtype(np.int64),
    torch.bool: np.dtype(np.bool_),
}


def _expected_state_items(
    model: nn.Module,
    include_buffers: bool = True,
) -> list[tuple[str, torch.Tensor]]:
    """Return ordered ``(key, tensor)`` pairs from ``model`` for parameter serialization."""
    if not isinstance(model, nn.Module):
        raise ParameterSerializationError(
            f"Expected torch.nn.Module instance, got {type(model).__name__}."
        )
    if include_buffers:
        items = list(model.state_dict().items())
    else:
        items = [(k, p.detach()) for k, p in model.named_parameters()]

    if not items:
        raise ParameterSerializationError(
            "Model state dictionary is empty; cannot serialize parameters."
        )
    return items


def get_model_state_keys(
    model: nn.Module,
    include_buffers: bool = True,
) -> tuple[str, ...]:
    """Return the canonical ordered tuple of state_dict keys serialized for ``model``."""
    return tuple(k for k, _ in _expected_state_items(model, include_buffers=include_buffers))


def get_model_parameters(
    model: nn.Module,
    include_buffers: bool = True,
) -> list[np.ndarray]:
    """Serialize ``model`` parameters (and optionally buffers) into an ordered list of NumPy arrays.

    Preserves exact key ordering, tensor shapes, and dtypes (including ``int64``
    BatchNorm ``num_batches_tracked`` buffers). Returns independent copies so
    in-place NumPy modifications cannot mutate the live PyTorch model.
    """
    items = _expected_state_items(model, include_buffers=include_buffers)
    arrays: list[np.ndarray] = []
    for key, tensor in items:
        if tensor.dtype not in _TORCH_TO_NUMPY_DTYPE:
            raise ParameterSerializationError(
                f"Unsupported tensor dtype {tensor.dtype} for state key '{key}'."
            )
        arr = tensor.detach().cpu().numpy().copy()
        arrays.append(arr)
    return arrays


def flower_parameters_to_ndarrays(
    parameters: NDArrays | Parameters,
) -> list[np.ndarray]:
    """Normalize either a Flower :class:`Parameters` object or sequence of NumPy arrays."""
    if isinstance(parameters, Parameters):
        try:
            return list(parameters_to_ndarrays(parameters))
        except Exception as exc:
            raise ParameterSerializationError(
                f"Failed to deserialize Flower Parameters object: {exc}"
            ) from exc

    if isinstance(parameters, Sequence) and not isinstance(parameters, (str, bytes)):
        return list(parameters)

    raise ParameterSerializationError(
        f"Expected Flower Parameters or sequence of numpy.ndarray, got {type(parameters).__name__}."
    )


def model_to_flower_parameters(
    model: nn.Module,
    include_buffers: bool = True,
) -> Parameters:
    """Serialize ``model`` state into a Flower :class:`Parameters` protobuf object."""
    ndarrays = get_model_parameters(model, include_buffers=include_buffers)
    return ndarrays_to_parameters(ndarrays)


def validate_parameter_ndarrays(
    model: nn.Module,
    parameters: NDArrays | Parameters,
    include_buffers: bool = True,
) -> list[np.ndarray]:
    """Validate that ``parameters`` match ``model`` in count, order, shape, dtype, and finiteness.

    Raises:
        ParameterSerializationError: If the array count, shapes, or dtypes mismatch,
            if an integer buffer (e.g., ``num_batches_tracked``) was converted to a
            floating-point or negative value, or if any floating-point array contains
            NaN or Inf values.
    """
    arrays = flower_parameters_to_ndarrays(parameters)
    expected_items = _expected_state_items(model, include_buffers=include_buffers)

    if len(arrays) != len(expected_items):
        raise ParameterSerializationError(
            f"Parameter count mismatch: model expects {len(expected_items)} tensor(s) "
            f"(include_buffers={include_buffers}), but received {len(arrays)} array(s)."
        )

    validated: list[np.ndarray] = []
    for idx, ((key, ref_tensor), arr) in enumerate(zip(expected_items, arrays)):
        if not isinstance(arr, np.ndarray):
            raise ParameterSerializationError(
                f"Parameter at index {idx} ('{key}') must be a numpy.ndarray, "
                f"got {type(arr).__name__}."
            )

        expected_shape = tuple(ref_tensor.shape)
        actual_shape = tuple(arr.shape)
        if actual_shape != expected_shape:
            raise ParameterSerializationError(
                f"Shape mismatch at index {idx} ('{key}'): expected {expected_shape}, "
                f"got {actual_shape}."
            )

        expected_np_dtype = _TORCH_TO_NUMPY_DTYPE.get(ref_tensor.dtype)
        if expected_np_dtype is None:
            raise ParameterSerializationError(
                f"Unsupported reference tensor dtype {ref_tensor.dtype} at '{key}'."
            )

        if arr.dtype != expected_np_dtype:
            raise ParameterSerializationError(
                f"Dtype mismatch at index {idx} ('{key}'): expected {expected_np_dtype}, "
                f"got {arr.dtype}."
            )

        if np.issubdtype(arr.dtype, np.floating):
            if not bool(np.all(np.isfinite(arr))):
                raise ParameterSerializationError(
                    f"Non-finite (NaN or Inf) value(s) detected in floating-point "
                    f"parameter at index {idx} ('{key}')."
                )
        elif np.issubdtype(arr.dtype, np.integer):
            if key.endswith("num_batches_tracked") and bool(np.any(arr < 0)):
                raise ParameterSerializationError(
                    f"Negative value detected in BatchNorm counter '{key}': {arr!r}."
                )

        validated.append(arr)

    return validated


def set_model_parameters(
    model: nn.Module,
    parameters: NDArrays | Parameters,
    include_buffers: bool = True,
) -> None:
    """Validate and load ``parameters`` into ``model`` atomically without discarding state.

    When ``include_buffers=True`` (default), all ``state_dict`` entries (trainable
    weights, BatchNorm ``running_mean`` / ``running_var``, and integer
    ``num_batches_tracked`` counters) are validated and restored.
    When ``include_buffers=False``, only ``named_parameters()`` are updated while the
    client's existing local buffers are preserved intact.
    """
    validated_arrays = validate_parameter_ndarrays(
        model=model,
        parameters=parameters,
        include_buffers=include_buffers,
    )
    expected_items = _expected_state_items(model, include_buffers=include_buffers)

    current_state = model.state_dict()
    new_state: OrderedDict[str, torch.Tensor] = OrderedDict()

    if include_buffers:
        for (key, ref_tensor), arr in zip(expected_items, validated_arrays):
            tensor = torch.from_numpy(np.ascontiguousarray(arr).copy()).to(
                device=ref_tensor.device,
                dtype=ref_tensor.dtype,
            )
            new_state[key] = tensor
    else:
        incoming_param_map: dict[str, torch.Tensor] = {}
        for (key, ref_tensor), arr in zip(expected_items, validated_arrays):
            incoming_param_map[key] = torch.from_numpy(
                np.ascontiguousarray(arr).copy()
            ).to(
                device=ref_tensor.device,
                dtype=ref_tensor.dtype,
            )
        for key, existing_tensor in current_state.items():
            if key in incoming_param_map:
                new_state[key] = incoming_param_map[key]
            else:
                new_state[key] = existing_tensor.detach().clone()

    model.load_state_dict(new_state, strict=True)


def _flatten_trainer_metrics(
    summary: Mapping[str, Any],
    prefix: str,
    client_id: str,
    client_index: int,
    dataset_label: str,
    server_round: int,
) -> dict[str, Scalar]:
    """Convert a trainer/evaluator epoch summary into a flat Flower ``dict[str, Scalar]``."""
    loss_val = float(summary.get("loss", 0.0))
    bce_val = float(summary.get("bce_loss", 0.0))
    dice_loss_val = float(summary.get("dice_loss", 0.0))
    dice_macro_val = float(summary.get("dice_macro", 0.0))
    iou_macro_val = float(summary.get("iou_macro", 0.0))
    num_samples = int(summary.get("num_samples", 0))
    num_batches = int(summary.get("num_batches", 0))

    out: dict[str, Scalar] = {
        "client_id": str(client_id),
        "client_index": int(client_index),
        "dataset": str(dataset_label),
        "server_round": int(server_round),
        "num_samples": num_samples,
        "num_batches": num_batches,
        "loss": loss_val,
        "bce_loss": bce_val,
        "dice_loss": dice_loss_val,
        "dice": dice_macro_val,
        "iou": iou_macro_val,
        "dice_macro": dice_macro_val,
        "iou_macro": iou_macro_val,
        f"{prefix}_loss": loss_val,
        f"{prefix}_bce_loss": bce_val,
        f"{prefix}_dice_loss": dice_loss_val,
        f"{prefix}_dice_macro": dice_macro_val,
        f"{prefix}_iou_macro": iou_macro_val,
    }

    metrics_block = summary.get("metrics", {})
    if isinstance(metrics_block, Mapping):
        macro_present = metrics_block.get("macro_present_only", {})
        if isinstance(macro_present, Mapping):
            dp = macro_present.get("dice")
            ip = macro_present.get("iou")
            if dp is not None:
                out["dice_macro_present_only"] = float(dp)
                out[f"{prefix}_dice_macro_present_only"] = float(dp)
            if ip is not None:
                out["iou_macro_present_only"] = float(ip)
                out[f"{prefix}_iou_macro_present_only"] = float(ip)

        per_class = metrics_block.get("per_class", {})
        if isinstance(per_class, Mapping):
            for ch in CANONICAL_LESION_CHANNELS:
                ch_dict = per_class.get(ch)
                if isinstance(ch_dict, Mapping):
                    ch_dice = float(ch_dict.get("dice", 0.0))
                    ch_iou = float(ch_dict.get("iou", 0.0))
                    out[f"dice_{ch}"] = ch_dice
                    out[f"iou_{ch}"] = ch_iou
                    out[f"{prefix}_dice_{ch}"] = ch_dice
                    out[f"{prefix}_iou_{ch}"] = ch_iou

    return out


class FedMorphisNumPyClient(NumPyClient):
    """Flower :class:`NumPyClient` wrapping :class:`SupervisedSegmentationTrainer`.

    Supports:
    - Receiving global model parameters and validating/loading them into the client's
      segmentation model (`ResNet50UNet` by default).
    - Local training over `local_epochs` epochs via
      :meth:`SupervisedSegmentationTrainer.train_one_epoch`.
    - Local deterministic validation on the client's validation DataLoader via
      :meth:`SupervisedSegmentationTrainer.validate_one_epoch`.
    - Returning updated parameters as NumPy arrays, exact sample counts, and
      per-class + macro segmentation metrics (`[MA, HE, EX, SE]`).
    """

    def __init__(
        self,
        client_id: str,
        client_index: int,
        model: nn.Module,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        dataset_names: Sequence[str] = ("DDR",),
        data_bundle: ClientDataBundle | None = None,
        federated_config: Mapping[str, Any] | str | Path = "federated.yaml",
        model_config: Mapping[str, Any] | str | Path = "model.yaml",
        device: torch.device | str = "cpu",
        local_epochs: int | None = None,
        learning_rate: float | None = None,
        include_buffers: bool | None = None,
        mixed_precision: bool | None = None,
        seed: int | None = None,
        deterministic: bool = True,
        checkpoint_dir: str | Path | None = None,
        max_train_batches: int | None = None,
        max_val_batches: int | None = None,
    ) -> None:
        super().__init__()
        self.federated_config = (
            load_yaml_config(federated_config, validate=True)
            if isinstance(federated_config, (str, Path))
            else dict(federated_config)
        )
        self.model_config = (
            load_yaml_config(model_config, validate=True)
            if isinstance(model_config, (str, Path))
            else dict(model_config)
        )

        # Enforce split hygiene on both loaders at construction time
        verify_no_test_split_leakage(train_loader, stage_name="train")
        verify_no_test_split_leakage(val_loader, stage_name="validation")
        verify_exact_split(val_loader, expected_split="valid")

        self.client_id = str(client_id)
        self.client_index = int(client_index)
        self.dataset_names = tuple(str(d) for d in dataset_names)
        self.dataset_label = ",".join(self.dataset_names) if self.dataset_names else "unknown"
        self.data_bundle = data_bundle
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = resolve_device(str(device)) if isinstance(device, str) else torch.device(device)
        self.seed = int(
            seed
            if seed is not None
            else self.federated_config.get(
                "seed",
                self.federated_config.get("partitioning", {}).get("seed", 42),
            )
        )
        self.deterministic = bool(deterministic)

        self.local_epochs = int(
            local_epochs
            if local_epochs is not None
            else self.federated_config.get("local_epochs", 3)
        )
        if self.local_epochs < 1:
            raise ValueError(f"local_epochs must be >= 1, got {self.local_epochs}.")

        train_block = dict(self.model_config.get("training", {}))
        self.learning_rate = float(
            learning_rate
            if learning_rate is not None
            else self.federated_config.get(
                "local_learning_rate",
                train_block.get("learning_rate", 1e-4),
            )
        )
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}.")

        self.include_buffers = bool(
            include_buffers
            if include_buffers is not None
            else self.federated_config.get("share_batchnorm_buffers", True)
        )
        self.mixed_precision = bool(
            mixed_precision
            if mixed_precision is not None
            else train_block.get("mixed_precision", False)
        )
        self.max_train_batches = max_train_batches
        self.max_val_batches = max_val_batches

        resolved_ckpt_dir = (
            Path(checkpoint_dir)
            if checkpoint_dir is not None
            else Path(
                self.federated_config.get(
                    "checkpoint_directory", "checkpoints/federated"
                )
            )
            / self.client_id
        )

        self.model = model.to(self.device)
        self.criterion = build_loss_from_config(self.model_config).to(self.device)
        self.optimizer = self._build_local_optimizer(self.learning_rate)

        grad_clip = train_block.get("gradient_clip_max_norm", 1.0)
        metric_thresh = float(train_block.get("metric_threshold", 0.5))
        monitor_metric = str(train_block.get("monitor_metric", "val_dice_macro"))
        monitor_mode = str(train_block.get("monitor_mode", "max"))

        self.trainer = SupervisedSegmentationTrainer(
            model=self.model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=self.device,
            scheduler=None,
            checkpoint_dir=resolved_ckpt_dir,
            mixed_precision=self.mixed_precision,
            gradient_clip_max_norm=grad_clip,
            metric_threshold=metric_thresh,
            monitor_metric=monitor_metric,
            monitor_mode=monitor_mode,
            seed=self.seed,
            config={
                "federated": self.federated_config,
                "model": self.model_config,
                "client_id": self.client_id,
            },
        )

    def _build_local_optimizer(self, lr: float) -> torch.optim.Optimizer:
        """Build a fresh local optimizer configured with learning rate ``lr``."""
        cfg_copy = dict(self.model_config)
        tr_copy = dict(cfg_copy.get("training", {}))
        tr_copy["learning_rate"] = float(lr)
        cfg_copy["training"] = tr_copy
        return build_optimizer_from_config(self.model, cfg_copy)

    @property
    def client_display_name(self) -> str:
        """Canonical human-readable client attribution label (e.g. ``Client 0 [DDR]``)."""
        return f"Client {self.client_index} [{self.dataset_label}]"

    def get_properties(self, config: dict[str, Scalar]) -> dict[str, Scalar]:
        """Return static client properties for Flower server inspection."""
        return {
            "client_id": self.client_id,
            "client_index": self.client_index,
            "dataset": self.dataset_label,
            "num_train_samples": len(self.train_loader.dataset),  # type: ignore[arg-type]
            "num_val_samples": len(self.val_loader.dataset),  # type: ignore[arg-type]
        }

    def get_parameters(self, config: dict[str, Scalar] | None = None) -> NDArrays:
        """Return the current model state as a list of NumPy arrays."""
        return get_model_parameters(self.model, include_buffers=self.include_buffers)

    def set_parameters(self, parameters: NDArrays | Parameters) -> None:
        """Validate and load incoming global parameters into ``self.model``."""
        set_model_parameters(
            self.model,
            parameters=parameters,
            include_buffers=self.include_buffers,
        )

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Load global parameters, train locally for ``local_epochs``, and return updates."""
        cfg = config or {}
        server_round = int(cfg.get("server_round", cfg.get("round", 1)))
        local_epochs = int(cfg.get("local_epochs", self.local_epochs))
        if local_epochs < 1:
            raise ValueError(f"local_epochs must be >= 1, got {local_epochs}.")

        lr = float(cfg.get("learning_rate", cfg.get("lr", self.learning_rate)))
        if lr <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {lr}.")

        max_batches_raw = cfg.get("max_train_batches", cfg.get("max_batches", self.max_train_batches))
        max_batches = int(max_batches_raw) if max_batches_raw is not None else None

        fit_start = time.perf_counter()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        try:
            train_dataset_size = len(self.train_loader.dataset)  # type: ignore[arg-type]
            train_batch_count = len(self.train_loader)
        except TypeError:
            train_dataset_size = -1
            train_batch_count = -1

        logger.info(
            "[CLIENT TRAIN START] %s | Round %d | local_epochs=%d | lr=%.6g | train_samples=%d | batches=%d | device=%s",
            self.client_display_name,
            server_round,
            local_epochs,
            lr,
            train_dataset_size,
            train_batch_count,
            self.device,
        )

        try:
            # Deterministic per-client, per-round seeding
            round_seed = int(
                (self.seed + self.client_index * 1000 + server_round) % (MAX_UINT32_SEED + 1)
            )
            set_seed(round_seed, deterministic=self.deterministic)
            loader_gen = getattr(self.train_loader, "generator", None)
            if isinstance(loader_gen, torch.Generator):
                loader_gen.manual_seed(round_seed)

            # Load global weights and reset local optimizer for this communication round
            self.set_parameters(parameters)
            self.optimizer = self._build_local_optimizer(lr)
            self.trainer.optimizer = self.optimizer

            epoch_summaries: list[dict[str, Any]] = []
            for ep in range(1, local_epochs + 1):
                progress_prefix = (
                    f"[CLIENT TRAIN] {self.client_display_name} | Round {server_round} | Epoch {ep}/{local_epochs}"
                )
                summary = self.trainer.train_one_epoch(
                    train_loader=self.train_loader,
                    epoch=ep,
                    max_batches=max_batches,
                    progress_prefix=progress_prefix,
                    progress_logger=logger,
                )
                epoch_summaries.append(summary)
                logger.info(
                    "[CLIENT EPOCH DONE] %s | Round %d | Epoch %d/%d | loss=%.4f | bce=%.4f | dice_loss=%.4f | train_dice_macro=%.4f | samples=%d | duration=%.2fs",
                    self.client_display_name,
                    server_round,
                    ep,
                    local_epochs,
                    float(summary["loss"]),
                    float(summary["bce_loss"]),
                    float(summary["dice_loss"]),
                    float(summary["dice_macro"]),
                    int(summary["num_samples"]),
                    float(summary["elapsed_sec"]),
                )

            last_summary = epoch_summaries[-1]
            num_examples = int(last_summary["num_samples"])
            mean_loss_across_epochs = float(
                sum(float(s["loss"]) for s in epoch_summaries) / len(epoch_summaries)
            )
            fit_duration = time.perf_counter() - fit_start

            metrics = _flatten_trainer_metrics(
                summary=last_summary,
                prefix="train",
                client_id=self.client_id,
                client_index=self.client_index,
                dataset_label=self.dataset_label,
                server_round=server_round,
            )
            metrics["local_epochs"] = int(local_epochs)
            metrics["learning_rate"] = float(lr)
            metrics["train_loss_mean"] = mean_loss_across_epochs
            metrics["fit_duration_sec"] = round(fit_duration, 4)

            gpu_mem_suffix = ""
            if self.device.type == "cuda" and torch.cuda.is_available():
                alloc_mb = round(float(torch.cuda.memory_allocated(self.device) / (1024**2)), 2)
                peak_mb = round(float(torch.cuda.max_memory_allocated(self.device) / (1024**2)), 2)
                reserved_mb = round(float(torch.cuda.memory_reserved(self.device) / (1024**2)), 2)
                metrics["gpu_mem_allocated_mb"] = alloc_mb
                metrics["gpu_mem_peak_mb"] = peak_mb
                metrics["gpu_mem_reserved_mb"] = reserved_mb
                gpu_mem_suffix = f" | gpu_mem_alloc={alloc_mb:.1f}MB | gpu_mem_peak={peak_mb:.1f}MB | gpu_mem_reserved={reserved_mb:.1f}MB"

            logger.info(
                "[CLIENT TRAIN COMPLETE] %s | Round %d | duration=%.2fs | samples=%d | final_loss=%.4f | mean_loss=%.4f | train_dice_macro=%.4f%s",
                self.client_display_name,
                server_round,
                fit_duration,
                num_examples,
                float(last_summary["loss"]),
                mean_loss_across_epochs,
                float(last_summary["dice_macro"]),
                gpu_mem_suffix,
            )

            updated_params = self.get_parameters(config={})
            return updated_params, num_examples, metrics
        except Exception:
            logger.exception(
                "[CLIENT TRAIN ERROR] %s failed during local training in Round %d",
                self.client_display_name,
                server_round,
            )
            raise

    def evaluate(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        """Load global parameters and evaluate deterministically on the validation split."""
        cfg = config or {}
        server_round = int(cfg.get("server_round", cfg.get("round", 1)))
        max_batches_raw = cfg.get("max_val_batches", cfg.get("max_batches", self.max_val_batches))
        max_batches = int(max_batches_raw) if max_batches_raw is not None else None

        eval_start = time.perf_counter()
        try:
            self.set_parameters(parameters)

            val_summary = self.trainer.validate_one_epoch(
                val_loader=self.val_loader,
                epoch=server_round,
                max_batches=max_batches,
            )
            eval_duration = time.perf_counter() - eval_start

            val_loss = float(val_summary["loss"])
            num_examples = int(val_summary["num_samples"])
            metrics = _flatten_trainer_metrics(
                summary=val_summary,
                prefix="val",
                client_id=self.client_id,
                client_index=self.client_index,
                dataset_label=self.dataset_label,
                server_round=server_round,
            )
            metrics["eval_duration_sec"] = round(eval_duration, 4)

            per_lesion_str = (
                f"MA={float(metrics.get('val_dice_MA', 0.0)):.4f}, "
                f"HE={float(metrics.get('val_dice_HE', 0.0)):.4f}, "
                f"EX={float(metrics.get('val_dice_EX', 0.0)):.4f}, "
                f"SE={float(metrics.get('val_dice_SE', 0.0)):.4f}"
            )
            logger.info(
                "[CLIENT VALIDATION] %s | Round %d | val_loss=%.4f | val_dice_macro=%.4f | val_iou_macro=%.4f | lesion_dice=[%s] | samples=%d | duration=%.2fs",
                self.client_display_name,
                server_round,
                val_loss,
                float(metrics.get("val_dice_macro", 0.0)),
                float(metrics.get("val_iou_macro", 0.0)),
                per_lesion_str,
                num_examples,
                eval_duration,
            )
            return val_loss, num_examples, metrics
        except Exception:
            logger.exception(
                "[CLIENT VALIDATION ERROR] %s failed during validation in Round %d",
                self.client_display_name,
                server_round,
            )
            raise


def build_federated_client(
    client_id: int | str,
    federated_config: Mapping[str, Any] | str | Path = "federated.yaml",
    dataset_config: Mapping[str, Any] | str | Path = "dataset.yaml",
    model_config: Mapping[str, Any] | str | Path = "model.yaml",
    project_config: Mapping[str, Any] | str | Path = "project.yaml",
    partition_manifest: str | Path | Mapping[str, Any] | None = None,
    source_manifest: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    data_bundle: ClientDataBundle | None = None,
    model: nn.Module | None = None,
    pretrained: bool = False,
    strict_pretrained: bool = False,
    device: torch.device | str | None = None,
    local_epochs: int | None = None,
    batch_size: int | None = None,
    val_batch_size: int | None = None,
    num_workers: int | None = None,
    learning_rate: float | None = None,
    include_buffers: bool | None = None,
    mixed_precision: bool | None = None,
    seed: int | None = None,
    image_size: tuple[int, int] | None = None,
    fov_crop: bool | None = None,
    cache_in_memory: bool = False,
    verify_files_exist: bool = False,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    as_flower_client: bool = False,
) -> FedMorphisNumPyClient | Client:
    """Factory function constructing a configured :class:`FedMorphisNumPyClient`.

    Args:
        client_id: Client identifier (``0``, ``1``, ``"client_0"``, ``"client_1"``,
            ``"DDR"``, or ``"IDRiD"``).
        federated_config: Path or mapping for ``federated.yaml``.
        dataset_config: Path or mapping for ``dataset.yaml``.
        model_config: Path or mapping for ``model.yaml``.
        project_config: Path or mapping for ``project.yaml``.
        partition_manifest: Optional path or dict for the partition manifest.
        source_manifest: Optional path, dict, or record list for full dataset records.
        data_bundle: Optional pre-constructed :class:`ClientDataBundle`.
        model: Optional pre-constructed segmentation ``nn.Module``. When ``None``,
            builds :class:`ResNet50UNet` via :func:`build_unet_from_config`.
        pretrained: Whether to load encoder pretrained weights on client model
            instantiation (defaults to ``False`` because global weights are broadcast
            by the Flower server).
        strict_pretrained: Whether to enforce strict pretrained weight loading when
            ``pretrained=True``.
        device: Target device (``"cpu"``, ``"cuda"``, or ``"auto"``). Defaults to
            ``project.yaml:device``.
        local_epochs: Optional override for local epochs per round.
        batch_size: Optional override for local training batch size.
        val_batch_size: Optional override for local validation batch size.
        num_workers: Optional override for DataLoader worker processes.
        learning_rate: Optional override for local optimizer learning rate.
        include_buffers: Optional override for ``share_batchnorm_buffers``.
        mixed_precision: Optional override for AMP mixed precision.
        seed: Optional override for base random seed.
        image_size: Optional ``(height, width)`` override.
        fov_crop: Optional FOV cropping override.
        cache_in_memory: Whether to cache dataset samples in RAM.
        verify_files_exist: Whether to verify all image/mask files exist immediately.
        max_train_batches: Optional cap on batches per training epoch.
        max_val_batches: Optional cap on batches per validation epoch.
        as_flower_client: When ``True``, returns ``numpy_client.to_client()``
            (:class:`flwr.client.Client`); when ``False`` (default), returns
            :class:`FedMorphisNumPyClient`.

    Returns:
        Configured :class:`FedMorphisNumPyClient` (or :class:`flwr.client.Client`).
    """
    fed_cfg = (
        load_yaml_config(federated_config, validate=True)
        if isinstance(federated_config, (str, Path))
        else dict(federated_config)
    )
    ds_cfg = (
        load_yaml_config(dataset_config, validate=True)
        if isinstance(dataset_config, (str, Path))
        else dict(dataset_config)
    )
    mdl_cfg = (
        load_yaml_config(model_config, validate=True)
        if isinstance(model_config, (str, Path))
        else dict(model_config)
    )
    proj_cfg = (
        load_yaml_config(project_config, validate=True)
        if isinstance(project_config, (str, Path))
        else dict(project_config)
    )

    resolved_seed = int(
        seed
        if seed is not None
        else fed_cfg.get(
            "seed",
            fed_cfg.get("partitioning", {}).get("seed", proj_cfg.get("random_seed", 42)),
        )
    )
    deterministic = bool(proj_cfg.get("deterministic", True))
    set_seed(resolved_seed, deterministic=deterministic)

    if device is None:
        resolved_device = resolve_device(str(proj_cfg.get("device", "auto")))
    elif isinstance(device, str):
        resolved_device = resolve_device(device)
    else:
        resolved_device = torch.device(device)

    if data_bundle is None:
        bundle = build_federated_client_dataloaders(
            client_id=client_id,
            partition_manifest=partition_manifest,
            source_manifest=source_manifest,
            federated_config=fed_cfg,
            dataset_config=ds_cfg,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            seed=resolved_seed,
            image_size=image_size,
            fov_crop=fov_crop,
            cache_in_memory=cache_in_memory,
            verify_files_exist=verify_files_exist,
        )
    else:
        bundle = data_bundle

    if model is None:
        seg_model: nn.Module = build_unet_from_config(
            model_config=mdl_cfg,
            pretrained_override=pretrained,
            strict_pretrained_override=strict_pretrained,
        )
    else:
        seg_model = model

    numpy_client = FedMorphisNumPyClient(
        client_id=bundle.client_id,
        client_index=bundle.client_index,
        model=seg_model,
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        dataset_names=bundle.dataset_names,
        data_bundle=bundle,
        federated_config=fed_cfg,
        model_config=mdl_cfg,
        device=resolved_device,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        include_buffers=include_buffers,
        mixed_precision=mixed_precision,
        seed=resolved_seed,
        deterministic=deterministic,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
    )

    if as_flower_client:
        return numpy_client.to_client()
    return numpy_client


__all__ = [
    "FedMorphisNumPyClient",
    "ParameterSerializationError",
    "build_federated_client",
    "flower_parameters_to_ndarrays",
    "get_model_parameters",
    "get_model_state_keys",
    "model_to_flower_parameters",
    "set_model_parameters",
    "validate_parameter_ndarrays",
]
