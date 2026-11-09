"""Custom Flower FedAvg aggregation strategy, global model initialization, and atomic checkpointing.

Provides:
- :func:`initialize_global_model`: Initializes the global :class:`ResNet50UNet` segmentation
  model from a full segmentation checkpoint, a SimCLR/encoder checkpoint, ImageNet
  weights, or deterministic random weights.
- :func:`aggregate_parameters_with_policy`: Sample-weighted parameter and floating-point
  buffer averaging with an explicit integer buffer policy (``"max"`` by default for
  BatchNorm ``num_batches_tracked`` of dtype ``int64``) and strict shape/dtype/finiteness
  validation.
- :func:`aggregate_client_metrics`: Sample-weighted, unweighted client-macro, and
  per-client metric aggregation for training and validation rounds.
- :func:`save_global_checkpoint` and :func:`load_global_checkpoint`: Atomic global
  checkpoint serialization compatible with
  :func:`src.engine.evaluator.reconstruct_model_from_checkpoint`.
- :class:`FedMorphisFedAvg`: Subclass of :class:`flwr.server.strategy.FedAvg`
  implementing validated parameter aggregation, metric aggregation, round history
  tracking, and atomic ``latest_checkpoint.pt`` / ``best_checkpoint.pt`` persistence.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
import logging
import math
import os
from pathlib import Path
import time
from typing import Any
import uuid

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
import numpy as np
import torch
import torch.nn as nn

from src.data.masks import CANONICAL_LESION_CHANNELS
from src.engine.evaluator import reconstruct_model_from_checkpoint
from src.federated.client import (
    ParameterSerializationError,
    _expected_state_items,
    flower_parameters_to_ndarrays,
    get_model_parameters,
    model_to_flower_parameters,
    set_model_parameters,
    validate_parameter_ndarrays,
)
from src.models.unet import (
    PretrainedWeightsError,
    ResNet50UNet,
    build_unet_from_config,
)
from src.utils.config import load_yaml_config
from src.utils.device import resolve_device
from src.utils.logger import get_logger
from src.utils.paths import format_project_path, resolve_project_path
from src.utils.seed import set_seed

logger = get_logger("fedmorphis.federated.strategy")


class FederatedStrategyError(ParameterSerializationError):
    """Raised when federated parameter aggregation, client validation, or strategy execution fails."""


SUPPORTED_INTEGER_BUFFER_POLICIES: tuple[str, ...] = ("max", "sum", "first")

_METADATA_SCALAR_KEYS: frozenset[str] = frozenset(
    {
        "client_id",
        "client_index",
        "dataset",
        "server_round",
        "round",
        "num_samples",
        "num_batches",
        "local_epochs",
        "learning_rate",
    }
)


def _is_finite_number(value: Any) -> bool:
    """Return ``True`` if ``value`` is an int or float (excluding bool) that is finite."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return False
    return bool(math.isfinite(float(value)))


def initialize_global_model(
    federated_config: Mapping[str, Any] | str | Path = "federated.yaml",
    model_config: Mapping[str, Any] | str | Path = "model.yaml",
    project_config: Mapping[str, Any] | str | Path = "project.yaml",
    *,
    pretrained_override: bool | None = None,
    strict_pretrained_override: bool | None = None,
    encoder_weights_source_override: str | None = None,
    ssl_checkpoint_path_override: str | Path | None = None,
    initial_checkpoint_path_override: str | Path | None = None,
    seed: int | None = None,
    device: torch.device | str = "cpu",
) -> tuple[ResNet50UNet, dict[str, Any]]:
    """Initialize the global :class:`ResNet50UNet` according to federated configuration.

    Initialization precedence:
    1. Explicit full segmentation checkpoint (``initialization.initial_checkpoint_path``
       or ``initial_checkpoint_path_override``).
    2. SimCLR / encoder checkpoint (``encoder_weights_source`` in
       ``{"ssl_checkpoint", "supervised_checkpoint"}`` with ``ssl_checkpoint_path``).
    3. ImageNet-pretrained encoder (``pretrained=True`` and
       ``encoder_weights_source="imagenet"``).
    4. Deterministic random initialization (``pretrained=False`` or
       ``encoder_weights_source="random"``).

    Raises:
        FileNotFoundError: If an explicitly configured full checkpoint or encoder
            checkpoint file does not exist.
        PretrainedWeightsError: If checkpoint loading or validation fails.
        ValueError: If checkpoint payload or architecture configuration is invalid.
    """
    fed_cfg = (
        load_yaml_config(federated_config, validate=True)
        if isinstance(federated_config, (str, Path))
        else dict(federated_config)
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

    target_device = (
        resolve_device(device) if isinstance(device, str) else torch.device(device)
    )

    init_block = dict(fed_cfg.get("initialization", {}))
    initial_ckpt_raw = (
        initial_checkpoint_path_override
        if initial_checkpoint_path_override is not None
        else init_block.get("initial_checkpoint_path")
    )
    has_initial_ckpt = (
        initial_ckpt_raw is not None and str(initial_ckpt_raw).strip() != ""
    )

    weights_source = str(
        encoder_weights_source_override
        if encoder_weights_source_override is not None
        else init_block.get(
            "encoder_weights_source",
            mdl_cfg.get("encoder_weights_source", "imagenet"),
        )
    ).strip().lower()

    pretrained = bool(
        pretrained_override
        if pretrained_override is not None
        else init_block.get("pretrained", mdl_cfg.get("pretrained", True))
    )
    strict_pretrained = bool(
        strict_pretrained_override
        if strict_pretrained_override is not None
        else init_block.get("strict_pretrained", mdl_cfg.get("strict_pretrained", False))
    )
    ssl_ckpt_raw = (
        ssl_checkpoint_path_override
        if ssl_checkpoint_path_override is not None
        else init_block.get(
            "ssl_checkpoint_path",
            mdl_cfg.get("ssl_checkpoint_path") or mdl_cfg.get("pretrained_checkpoint_path"),
        )
    )

    # 1. Full segmentation checkpoint initialization
    if has_initial_ckpt:
        ckpt_path = resolve_project_path(str(initial_ckpt_raw))
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"Initial global segmentation checkpoint not found: {ckpt_path}"
            )
        model = build_unet_from_config(
            mdl_cfg,
            pretrained_override=False,
            strict_pretrained_override=False,
        )
        try:
            raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise PretrainedWeightsError(
                f"Failed to read initial segmentation checkpoint '{ckpt_path}': {exc}"
            ) from exc

        if not isinstance(raw_ckpt, Mapping) or len(raw_ckpt) == 0:
            raise PretrainedWeightsError(
                f"Expected non-empty mapping in initial checkpoint '{ckpt_path}', "
                f"got {type(raw_ckpt).__name__}."
            )

        if "model_state_dict" in raw_ckpt and isinstance(raw_ckpt["model_state_dict"], Mapping):
            state_dict = raw_ckpt["model_state_dict"]
        elif "state_dict" in raw_ckpt and isinstance(raw_ckpt["state_dict"], Mapping):
            state_dict = raw_ckpt["state_dict"]
        else:
            state_dict = raw_ckpt

        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as exc:
            raise PretrainedWeightsError(
                f"Incompatible state_dict in initial segmentation checkpoint '{ckpt_path}': {exc}"
            ) from exc

        model.pretrained_requested = True
        model.pretrained_weights_loaded = True
        model.encoder_weights_source = "full_checkpoint"
        model.encoder_checkpoint_path = format_project_path(ckpt_path)
        model.weights_status = f"loaded_full_checkpoint:{ckpt_path.name}"
        model.to(target_device)

        init_info = {
            "initialization_mode": "full_checkpoint",
            "checkpoint_path": format_project_path(ckpt_path),
            "encoder_weights_source": model.encoder_weights_source,
            "pretrained_weights_loaded": bool(model.pretrained_weights_loaded),
            "weights_status": model.weights_status,
            "seed": resolved_seed,
        }
        return model, init_info

    # 2. SimCLR / encoder checkpoint initialization
    if pretrained and weights_source in {"ssl_checkpoint", "supervised_checkpoint"}:
        if ssl_ckpt_raw is None or str(ssl_ckpt_raw).strip() == "":
            raise PretrainedWeightsError(
                f"encoder_weights_source={weights_source!r} requires a non-empty ssl_checkpoint_path."
            )
        ckpt_path = resolve_project_path(str(ssl_ckpt_raw))
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"Pretrained encoder checkpoint not found: {ckpt_path}"
            )

        model = build_unet_from_config(
            mdl_cfg,
            pretrained_override=False,
            strict_pretrained_override=False,
        )
        try:
            load_report = model.load_encoder_from_checkpoint(ckpt_path, strict=True)
        except Exception as exc:
            raise PretrainedWeightsError(
                f"Failed to initialize global encoder from checkpoint '{ckpt_path}': {exc}"
            ) from exc

        model.encoder_weights_source = weights_source
        model.pretrained_requested = True
        model.to(target_device)

        init_info = {
            "initialization_mode": weights_source,
            "checkpoint_path": load_report["checkpoint_path"],
            "loaded_encoder_parameter_count": load_report["loaded_parameter_count"],
            "encoder_weights_source": model.encoder_weights_source,
            "pretrained_weights_loaded": bool(model.pretrained_weights_loaded),
            "weights_status": model.weights_status,
            "seed": resolved_seed,
        }
        return model, init_info

    # 3. Random initialization
    if not pretrained or weights_source == "random":
        merged_cfg = dict(mdl_cfg)
        merged_cfg["encoder_weights_source"] = "random"
        model = build_unet_from_config(
            merged_cfg,
            pretrained_override=False,
            strict_pretrained_override=False,
        )
        model.to(target_device)
        init_info = {
            "initialization_mode": "random",
            "checkpoint_path": None,
            "encoder_weights_source": "random",
            "pretrained_weights_loaded": False,
            "weights_status": model.weights_status,
            "seed": resolved_seed,
        }
        return model, init_info

    # 4. ImageNet-pretrained encoder initialization
    merged_cfg = dict(mdl_cfg)
    merged_cfg["encoder_weights_source"] = weights_source
    model = build_unet_from_config(
        merged_cfg,
        pretrained_override=True,
        strict_pretrained_override=strict_pretrained,
    )
    if not model.pretrained_weights_loaded:
        raise PretrainedWeightsError(
            f"Global model requested pretrained encoder weights (source={weights_source!r}), "
            f"but pretrained_weights_loaded is False (status={model.weights_status!r}). "
            "Refusing to silently fall back to random initialization."
        )
    model.to(target_device)
    init_info = {
        "initialization_mode": "imagenet",
        "checkpoint_path": model.encoder_checkpoint_path,
        "encoder_weights_source": model.encoder_weights_source,
        "pretrained_weights_loaded": bool(model.pretrained_weights_loaded),
        "weights_status": model.weights_status,
        "seed": resolved_seed,
    }
    return model, init_info


def aggregate_parameters_with_policy(
    results: Sequence[tuple[NDArrays | Parameters, int]],
    reference_model: nn.Module,
    *,
    include_buffers: bool = True,
    integer_buffer_policy: str = "max",
) -> list[np.ndarray]:
    """Aggregate client parameter updates with sample-weighted averaging and integer buffer policy.

    Floating-point parameters and floating-point BatchNorm buffers (``running_mean``,
    ``running_var``) are aggregated via sample-weighted averaging:
    ``w = sum_k (n_k / N) * w_k``.

    Integer buffers (specifically BatchNorm ``num_batches_tracked`` of dtype ``int64``)
    are aggregated using ``integer_buffer_policy`` (default ``"max"`` across participating
    clients), preserving the exact ``np.int64`` dtype without floating-point conversion
    or rounding.

    Raises:
        FederatedStrategyError: If ``results`` is empty, any client has ``num_examples <= 0``,
            any client's array count/shapes/dtypes mismatch ``reference_model``, or any
            input or aggregated floating-point value is non-finite (``NaN`` or ``Inf``).
    """
    policy = str(integer_buffer_policy).strip().lower()
    if policy not in SUPPORTED_INTEGER_BUFFER_POLICIES:
        raise FederatedStrategyError(
            f"Unsupported integer_buffer_policy={integer_buffer_policy!r}; "
            f"expected one of {SUPPORTED_INTEGER_BUFFER_POLICIES}."
        )

    if not results:
        raise FederatedStrategyError(
            "Cannot aggregate empty client parameter results list."
        )

    expected_items = _expected_state_items(
        reference_model,
        include_buffers=include_buffers,
    )

    validated_client_updates: list[tuple[list[np.ndarray], int]] = []
    total_examples = 0

    for client_idx, (raw_params, num_examples) in enumerate(results):
        if isinstance(num_examples, bool) or not isinstance(num_examples, (int, np.integer)):
            raise FederatedStrategyError(
                f"Client update at index {client_idx} has non-integer num_examples={num_examples!r}."
            )
        n_ex = int(num_examples)
        if n_ex <= 0:
            raise FederatedStrategyError(
                f"Client update at index {client_idx} has non-positive num_examples={n_ex}."
            )

        try:
            validated_arrays = validate_parameter_ndarrays(
                model=reference_model,
                parameters=raw_params,
                include_buffers=include_buffers,
            )
        except ParameterSerializationError as exc:
            raise FederatedStrategyError(
                f"Client update at index {client_idx} failed parameter validation: {exc}"
            ) from exc

        validated_client_updates.append((validated_arrays, n_ex))
        total_examples += n_ex

    if total_examples <= 0:
        raise FederatedStrategyError(
            f"Total training examples across clients must be > 0, got {total_examples}."
        )

    weights = [float(n_ex) / float(total_examples) for _, n_ex in validated_client_updates]
    aggregated: list[np.ndarray] = []

    for param_idx, (key, _) in enumerate(expected_items):
        client_arrays = [client_arrs[param_idx] for client_arrs, _ in validated_client_updates]
        target_dtype = client_arrays[0].dtype
        target_shape = client_arrays[0].shape

        if np.issubdtype(target_dtype, np.floating):
            accum = np.zeros(target_shape, dtype=np.float64)
            for arr, weight in zip(client_arrays, weights):
                accum += arr.astype(np.float64, copy=False) * weight

            if not bool(np.all(np.isfinite(accum))):
                raise FederatedStrategyError(
                    f"Non-finite aggregated floating-point value detected at index "
                    f"{param_idx} ('{key}')."
                )
            agg_arr = accum.astype(target_dtype)
            if not bool(np.all(np.isfinite(agg_arr))):
                raise FederatedStrategyError(
                    f"Aggregated parameter overflowed after casting to {target_dtype} "
                    f"at index {param_idx} ('{key}')."
                )
            aggregated.append(agg_arr)

        elif np.issubdtype(target_dtype, np.integer):
            if policy == "max":
                reduced = np.maximum.reduce(client_arrays)
            elif policy == "sum":
                reduced = np.add.reduce(client_arrays)
            else:  # policy == "first"
                reduced = client_arrays[0].copy()

            agg_int = np.asarray(reduced, dtype=target_dtype).reshape(target_shape).copy()
            if key.endswith("num_batches_tracked") and bool(np.any(agg_int < 0)):
                raise FederatedStrategyError(
                    f"Negative aggregated BatchNorm counter at '{key}': {agg_int!r}."
                )
            aggregated.append(agg_int)

        elif target_dtype == np.dtype(np.bool_):
            agg_bool = np.asarray(client_arrays[0], dtype=target_dtype).reshape(target_shape).copy()
            aggregated.append(agg_bool)

        else:
            raise FederatedStrategyError(
                f"Unsupported parameter dtype {target_dtype} at index {param_idx} ('{key}')."
            )

    # Final post-aggregation verification against the reference model
    return validate_parameter_ndarrays(
        model=reference_model,
        parameters=aggregated,
        include_buffers=include_buffers,
    )


def aggregate_client_metrics(
    metrics_results: Sequence[tuple[int, Mapping[str, Scalar]]],
    *,
    stage: str = "val",
    strict_finite: bool = True,
) -> dict[str, Scalar]:
    """Aggregate per-client Flower metric dictionaries into weighted, macro, and per-client metrics.

    Computes:
    - Sample-weighted average across participating clients for every numeric metric key.
    - Unweighted macro average across participating clients (``macro_client_<key>``).
    - Per-client metric entries (``<client_id>_<dataset>_<key>`` and ``<client_id>_<key>``).
    """
    if not metrics_results:
        return {}

    valid_entries: list[tuple[int, Mapping[str, Scalar]]] = []
    total_samples = 0

    for idx, (num_examples, metrics) in enumerate(metrics_results):
        n_ex = int(num_examples)
        if n_ex <= 0:
            if strict_finite:
                raise FederatedStrategyError(
                    f"Client {idx} reported non-positive sample count {n_ex} during {stage} metric aggregation."
                )
            continue
        if not isinstance(metrics, Mapping):
            continue
        valid_entries.append((n_ex, metrics))
        total_samples += n_ex

    if not valid_entries or total_samples <= 0:
        return {}

    # Sort deterministically by (client_index, client_id) when present
    def _sort_key(item: tuple[int, Mapping[str, Scalar]]) -> tuple[int, str]:
        m = item[1]
        c_idx = int(m.get("client_index", 9999)) if isinstance(m.get("client_index"), (int, float)) else 9999
        c_id = str(m.get("client_id", ""))
        return (c_idx, c_id)

    sorted_entries = sorted(valid_entries, key=_sort_key)

    aggregated: dict[str, Scalar] = {
        "num_clients": int(len(sorted_entries)),
        "num_samples_total": int(total_samples),
        f"{stage}_num_clients": int(len(sorted_entries)),
        f"{stage}_num_samples_total": int(total_samples),
    }

    # Collect numeric keys eligible for aggregation
    numeric_keys: list[str] = []
    seen_keys: set[str] = set()
    for _, client_metrics in sorted_entries:
        for k, v in client_metrics.items():
            if k in _METADATA_SCALAR_KEYS or k in seen_keys:
                continue
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float, np.number)):
                seen_keys.add(k)
                numeric_keys.append(k)

    # Record per-client metrics and validate finiteness
    client_ids_participated: list[str] = []
    for idx, (n_ex, client_metrics) in enumerate(sorted_entries):
        raw_cid = str(client_metrics.get("client_id", f"client_{idx}")).strip()
        raw_ds = str(client_metrics.get("dataset", "")).strip()
        client_ids_participated.append(raw_cid)

        prefix_short = raw_cid
        prefix_full = f"{raw_cid}_{raw_ds}" if raw_ds else raw_cid

        aggregated[f"{prefix_short}_num_samples"] = int(n_ex)
        if raw_ds:
            aggregated[f"{prefix_short}_dataset"] = raw_ds
            aggregated[f"{prefix_full}_num_samples"] = int(n_ex)

        for k in numeric_keys:
            if k not in client_metrics:
                continue
            val = client_metrics[k]
            if not _is_finite_number(val):
                if strict_finite:
                    raise FederatedStrategyError(
                        f"Non-finite {stage} metric '{k}'={val!r} reported by client '{raw_cid}'."
                    )
                continue
            fval = float(val)
            aggregated[f"{prefix_short}_{k}"] = fval
            if raw_ds:
                aggregated[f"{prefix_full}_{k}"] = fval

    aggregated["participating_clients"] = ",".join(client_ids_participated)

    # Compute sample-weighted average and unweighted macro across clients
    for k in numeric_keys:
        weighted_sum = 0.0
        weight_total = 0
        unweighted_vals: list[float] = []

        for n_ex, client_metrics in sorted_entries:
            if k not in client_metrics:
                continue
            val = client_metrics[k]
            if not _is_finite_number(val):
                continue
            fval = float(val)
            weighted_sum += fval * float(n_ex)
            weight_total += n_ex
            unweighted_vals.append(fval)

        if weight_total > 0 and unweighted_vals:
            weighted_avg = float(weighted_sum / float(weight_total))
            macro_client_avg = float(sum(unweighted_vals) / len(unweighted_vals))
            aggregated[k] = weighted_avg
            aggregated[f"macro_client_{k}"] = macro_client_avg

    # Ensure stage-prefixed canonical keys exist
    if "dice_macro" in aggregated and f"{stage}_dice_macro" not in aggregated:
        aggregated[f"{stage}_dice_macro"] = float(aggregated["dice_macro"])
    if "iou_macro" in aggregated and f"{stage}_iou_macro" not in aggregated:
        aggregated[f"{stage}_iou_macro"] = float(aggregated["iou_macro"])
    if "loss" in aggregated and f"{stage}_loss" not in aggregated:
        aggregated[f"{stage}_loss"] = float(aggregated["loss"])

    for ch in CANONICAL_LESION_CHANNELS:
        d_key = f"dice_{ch}"
        i_key = f"iou_{ch}"
        if d_key in aggregated and f"{stage}_{d_key}" not in aggregated:
            aggregated[f"{stage}_{d_key}"] = float(aggregated[d_key])
        if i_key in aggregated and f"{stage}_{i_key}" not in aggregated:
            aggregated[f"{stage}_{i_key}"] = float(aggregated[i_key])

    return aggregated


def save_global_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    server_round: int,
    *,
    metrics: Mapping[str, Any] | None = None,
    best_metric_value: float = float("-inf"),
    best_round: int = 0,
    monitor_metric: str = "val_dice_macro",
    monitor_mode: str = "max",
    seed: int = 42,
    federated_config: Mapping[str, Any] | None = None,
    model_config: Mapping[str, Any] | None = None,
    dataset_config: Mapping[str, Any] | None = None,
    project_config: Mapping[str, Any] | None = None,
    round_history: Sequence[Mapping[str, Any]] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically serialize a global federated model checkpoint.

    Writes to a process-unique temporary file in the target directory and atomically
    replaces ``checkpoint_path`` via :func:`os.replace`. The resulting payload is
    directly compatible with :func:`src.engine.evaluator.reconstruct_model_from_checkpoint`.
    """
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cpu_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict(
        (k, v.detach().cpu().clone()) for k, v in model.state_dict().items()
    )
    model_summary = (
        model.get_model_summary()
        if hasattr(model, "get_model_summary") and callable(model.get_model_summary)
        else {}
    )

    payload: dict[str, Any] = {
        "round": int(server_round),
        "epoch": int(server_round),
        "global_step": int(server_round),
        "best_metric_value": float(best_metric_value),
        "best_round": int(best_round),
        "best_epoch": int(best_round),
        "monitor_metric": str(monitor_metric),
        "monitor_mode": str(monitor_mode),
        "seed": int(seed),
        "config": {
            "federated": dict(federated_config or {}),
            "model": dict(model_config or {}),
            "data": dict(dataset_config or {}),
            "base": dict(project_config or {}),
        },
        "model_state_dict": cpu_state_dict,
        "model_summary": model_summary,
        "metrics": dict(metrics or {}),
        "history": [dict(entry) for entry in (round_history or [])],
        "extra_metadata": dict(extra_metadata or {}),
    }

    tmp_path = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise

    return path


def load_global_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module | None = None,
    *,
    fallback_model_config: Mapping[str, Any] | str | Path | None = None,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a saved global federated checkpoint into ``model`` or reconstruct a new :class:`ResNet50UNet`."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Global checkpoint file not found: {path}")

    if model is None:
        reconstructed_model, ckpt_dict, _ = reconstruct_model_from_checkpoint(
            checkpoint_path=path,
            fallback_model_config=fallback_model_config,
            device=device,
            strict=strict,
        )
        return reconstructed_model, ckpt_dict

    target_device = (
        resolve_device(device) if isinstance(device, str) else torch.device(device)
    )
    checkpoint = torch.load(path, map_location=target_device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Invalid checkpoint payload in '{path}': missing 'model_state_dict'."
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    model.to(target_device)
    model.eval()
    return model, checkpoint


class FedMorphisFedAvg(FedAvg):
    """Flower :class:`FedAvg` strategy tailored for FedMorphis retinal lesion segmentation.

    Features:
    - Sample-weighted parameter and floating-point BatchNorm buffer aggregation.
    - Explicit integer buffer aggregation policy (``"max"`` by default) for ``int64``
      BatchNorm ``num_batches_tracked`` counters.
    - Configurable ``share_batchnorm_buffers`` (``include_buffers``).
    - Strict shape, dtype, and finiteness validation of all client updates before
      updating global parameters.
    - Sample-weighted, unweighted client-macro, and per-client metric aggregation.
    - Atomic global checkpointing (``latest_checkpoint.pt`` and ``best_checkpoint.pt``
      monitored on validation Dice when finite and improved).
    """

    def __init__(
        self,
        *,
        global_model: nn.Module,
        federated_config: Mapping[str, Any] | str | Path = "federated.yaml",
        model_config: Mapping[str, Any] | str | Path = "model.yaml",
        dataset_config: Mapping[str, Any] | str | Path = "dataset.yaml",
        project_config: Mapping[str, Any] | str | Path = "project.yaml",
        initial_parameters: Parameters | None = None,
        fraction_fit: float | None = None,
        fraction_evaluate: float | None = None,
        min_fit_clients: int | None = None,
        min_evaluate_clients: int | None = None,
        min_available_clients: int | None = None,
        accept_failures: bool = False,
        include_buffers: bool | None = None,
        integer_buffer_policy: str = "max",
        local_epochs: int | None = None,
        learning_rate: float | None = None,
        max_train_batches: int | None = None,
        max_val_batches: int | None = None,
        checkpoint_dir: str | Path | None = None,
        save_checkpoints: bool = True,
        save_every_round: int | None = None,
        monitor_metric: str | None = None,
        monitor_mode: str | None = None,
        seed: int | None = None,
        num_rounds: int | None = None,
        evaluate_fn: (
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                tuple[float, dict[str, Scalar]] | None,
            ]
            | None
        ) = None,
        on_fit_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        on_evaluate_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        fit_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        evaluate_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        initialization_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(global_model, nn.Module):
            raise TypeError(
                f"global_model must be a torch.nn.Module, got {type(global_model).__name__}."
            )

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
        self.dataset_config = (
            load_yaml_config(dataset_config, validate=True)
            if isinstance(dataset_config, (str, Path))
            else dict(dataset_config)
        )
        self.project_config = (
            load_yaml_config(project_config, validate=True)
            if isinstance(project_config, (str, Path))
            else dict(project_config)
        )

        self.global_model = global_model
        self.initialization_metadata = dict(initialization_metadata or {})

        self.include_buffers = bool(
            include_buffers
            if include_buffers is not None
            else self.federated_config.get("share_batchnorm_buffers", True)
        )
        self.integer_buffer_policy = str(integer_buffer_policy).strip().lower()
        if self.integer_buffer_policy not in SUPPORTED_INTEGER_BUFFER_POLICIES:
            raise FederatedStrategyError(
                f"Unsupported integer_buffer_policy={integer_buffer_policy!r}; "
                f"expected one of {SUPPORTED_INTEGER_BUFFER_POLICIES}."
            )

        self.seed = int(
            seed
            if seed is not None
            else self.federated_config.get(
                "seed",
                self.federated_config.get("partitioning", {}).get(
                    "seed", self.project_config.get("random_seed", 42)
                ),
            )
        )

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

        self.max_train_batches = max_train_batches
        self.max_val_batches = max_val_batches

        resolved_fraction_fit = float(
            fraction_fit
            if fraction_fit is not None
            else self.federated_config.get("client_fraction", 1.0)
        )
        resolved_fraction_eval = float(
            fraction_evaluate
            if fraction_evaluate is not None
            else self.federated_config.get("client_fraction", 1.0)
        )
        num_clients_cfg = int(self.federated_config.get("number_of_clients", 2))
        resolved_min_fit = int(
            min_fit_clients
            if min_fit_clients is not None
            else self.federated_config.get("min_fit_clients", num_clients_cfg)
        )
        resolved_min_eval = int(
            min_evaluate_clients
            if min_evaluate_clients is not None
            else self.federated_config.get("min_evaluate_clients", num_clients_cfg)
        )
        resolved_min_avail = int(
            min_available_clients
            if min_available_clients is not None
            else self.federated_config.get("min_available_clients", num_clients_cfg)
        )

        if initial_parameters is None:
            resolved_initial_params = model_to_flower_parameters(
                self.global_model,
                include_buffers=self.include_buffers,
            )
        else:
            # Validate and synchronize global_model with supplied initial_parameters
            set_model_parameters(
                self.global_model,
                initial_parameters,
                include_buffers=self.include_buffers,
            )
            resolved_initial_params = model_to_flower_parameters(
                self.global_model,
                include_buffers=self.include_buffers,
            )

        self.current_parameters: Parameters = resolved_initial_params

        ckpt_cfg = dict(self.federated_config.get("checkpointing", {}))
        self.save_checkpoints = bool(save_checkpoints)
        raw_ckpt_dir = (
            checkpoint_dir
            if checkpoint_dir is not None
            else ckpt_cfg.get(
                "save_dir",
                self.federated_config.get("checkpoint_directory", "checkpoints/federated"),
            )
        )
        self.checkpoint_dir: Path | None = (
            resolve_project_path(raw_ckpt_dir) if raw_ckpt_dir is not None else None
        )
        self.save_every_round = int(
            save_every_round
            if save_every_round is not None
            else ckpt_cfg.get("save_every_round", 1)
        )
        self.monitor_metric = str(
            monitor_metric
            if monitor_metric is not None
            else ckpt_cfg.get("monitor_metric", "val_dice_macro")
        )
        self.monitor_mode = str(
            monitor_mode
            if monitor_mode is not None
            else ckpt_cfg.get("monitor_mode", "max")
        ).lower()
        if self.monitor_mode not in {"max", "min"}:
            raise ValueError(
                f"monitor_mode must be 'max' or 'min', got {self.monitor_mode!r}."
            )

        raw_num_rounds = (
            num_rounds
            if num_rounds is not None
            else self.federated_config.get(
                "communication_rounds",
                self.federated_config.get("rounds"),
            )
        )
        self.num_rounds: int | None = int(raw_num_rounds) if raw_num_rounds is not None else None

        self.best_metric_value: float = (
            float("-inf") if self.monitor_mode == "max" else float("inf")
        )
        self.best_round: int = 0
        self.latest_checkpoint_path: Path | None = None
        self.best_checkpoint_path: Path | None = None

        self._round_start_times: dict[int, float] = {}
        self._round_fit_metrics: dict[int, dict[str, Scalar]] = {}
        self._round_eval_metrics: dict[int, dict[str, Scalar]] = {}
        self.round_history: list[dict[str, Any]] = []

        super().__init__(
            fraction_fit=resolved_fraction_fit,
            fraction_evaluate=resolved_fraction_eval,
            min_fit_clients=resolved_min_fit,
            min_evaluate_clients=resolved_min_eval,
            min_available_clients=resolved_min_avail,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=bool(accept_failures),
            initial_parameters=resolved_initial_params,
            fit_metrics_aggregation_fn=(
                fit_metrics_aggregation_fn
                if fit_metrics_aggregation_fn is not None
                else lambda m: aggregate_client_metrics(m, stage="train")
            ),
            evaluate_metrics_aggregation_fn=(
                evaluate_metrics_aggregation_fn
                if evaluate_metrics_aggregation_fn is not None
                else lambda m: aggregate_client_metrics(m, stage="val")
            ),
            inplace=False,
        )

    def _format_round_label(self, server_round: int) -> str:
        """Return ``'X / Y'`` when ``num_rounds`` is known, else ``'X'``."""
        if self.num_rounds is not None and self.num_rounds > 0:
            return f"{int(server_round)} / {self.num_rounds}"
        return f"{int(server_round)}"

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        """Return initial global parameters and retain a reference on ``self.current_parameters``."""
        _ = client_manager
        if self.initial_parameters is not None:
            self.current_parameters = self.initial_parameters
        return self.current_parameters

    def _default_fit_config(self, server_round: int) -> dict[str, Scalar]:
        """Build default per-round fit configuration dictionary sent to clients."""
        cfg: dict[str, Scalar] = {
            "server_round": int(server_round),
            "round": int(server_round),
            "local_epochs": int(self.local_epochs),
            "learning_rate": float(self.learning_rate),
        }
        if self.max_train_batches is not None:
            cfg["max_train_batches"] = int(self.max_train_batches)
        return cfg

    def _default_evaluate_config(self, server_round: int) -> dict[str, Scalar]:
        """Build default per-round evaluate configuration dictionary sent to clients."""
        cfg: dict[str, Scalar] = {
            "server_round": int(server_round),
            "round": int(server_round),
        }
        if self.max_val_batches is not None:
            cfg["max_val_batches"] = int(self.max_val_batches)
        return cfg

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Sample clients and construct deterministic :class:`FitIns` for ``server_round``."""
        self._round_start_times[int(server_round)] = time.perf_counter()
        round_label = self._format_round_label(server_round)

        config = self._default_fit_config(server_round)
        if self.on_fit_config_fn is not None:
            custom_cfg = self.on_fit_config_fn(server_round)
            if isinstance(custom_cfg, Mapping):
                config.update(custom_cfg)

        fit_ins = FitIns(parameters, config)
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )
        clients_sorted = sorted(clients, key=lambda c: str(c.cid))

        logger.info("========== FEDERATED ROUND %s ==========", round_label)
        logger.info(
            "[ROUND START] Round %s | participating_clients=%d | client_ids=%s | local_epochs=%d | lr=%.6g",
            round_label,
            len(clients_sorted),
            ",".join(str(c.cid) for c in clients_sorted),
            self.local_epochs,
            self.learning_rate,
        )
        logger.info(
            "[SERVER] Waiting for %d client training update(s) in Round %s...",
            len(clients_sorted),
            round_label,
        )
        return [(client, fit_ins) for client in clients_sorted]

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Sample clients and construct deterministic :class:`EvaluateIns` for ``server_round``."""
        if self.fraction_evaluate == 0.0:
            return []

        config = self._default_evaluate_config(server_round)
        if self.on_evaluate_config_fn is not None:
            custom_cfg = self.on_evaluate_config_fn(server_round)
            if isinstance(custom_cfg, Mapping):
                config.update(custom_cfg)

        evaluate_ins = EvaluateIns(parameters, config)
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )
        clients_sorted = sorted(clients, key=lambda c: str(c.cid))
        round_label = self._format_round_label(server_round)
        logger.info(
            "[SERVER] Dispatching global model validation to %d client(s) in Round %s (%s)...",
            len(clients_sorted),
            round_label,
            ",".join(str(c.cid) for c in clients_sorted),
        )
        return [(client, evaluate_ins) for client in clients_sorted]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Validate and aggregate client fit updates and training metrics for ``server_round``."""
        round_label = self._format_round_label(server_round)
        logger.info(
            "[SERVER AGGREGATION] Received %d client training update(s) in Round %s (failures=%d).",
            len(results),
            round_label,
            len(failures),
        )

        if failures and not self.accept_failures:
            logger.error(
                "[SERVER AGGREGATION ERROR] Round %s fit failed with %d client failure(s): %r",
                round_label,
                len(failures),
                failures,
            )
            raise FederatedStrategyError(
                f"Round {server_round} fit failed with {len(failures)} client failure(s) "
                f"and accept_failures=False: {failures!r}"
            )

        if not results:
            if not self.accept_failures:
                raise FederatedStrategyError(
                    f"Round {server_round} received zero client fit results."
                )
            return None, {}

        if len(results) < self.min_fit_clients and not self.accept_failures:
            raise FederatedStrategyError(
                f"Round {server_round} received {len(results)} fit result(s), "
                f"fewer than min_fit_clients={self.min_fit_clients}."
            )

        sorted_results = sorted(results, key=lambda item: str(item[0].cid))

        valid_param_results: list[tuple[Parameters, int]] = []
        valid_metric_results: list[tuple[int, Mapping[str, Scalar]]] = []

        for client_proxy, fit_res in sorted_results:
            if not self.accept_failures:
                valid_param_results.append((fit_res.parameters, fit_res.num_examples))
                valid_metric_results.append((fit_res.num_examples, fit_res.metrics))
            else:
                try:
                    if int(fit_res.num_examples) <= 0:
                        raise FederatedStrategyError(
                            f"Client '{client_proxy.cid}' returned num_examples={fit_res.num_examples}."
                        )
                    validate_parameter_ndarrays(
                        model=self.global_model,
                        parameters=fit_res.parameters,
                        include_buffers=self.include_buffers,
                    )
                    valid_param_results.append((fit_res.parameters, fit_res.num_examples))
                    valid_metric_results.append((fit_res.num_examples, fit_res.metrics))
                except Exception as exc:
                    logger.warning(
                        "Skipping invalid fit update from client '%s' in round %d (accept_failures=True): %s",
                        client_proxy.cid,
                        server_round,
                        exc,
                    )

        if not valid_param_results:
            if not self.accept_failures:
                raise FederatedStrategyError(
                    f"Round {server_round} has no valid client updates to aggregate."
                )
            return None, {}

        total_fit_samples = sum(int(n_ex) for _, n_ex in valid_param_results)
        weight_descriptions: list[str] = []
        for idx, (n_ex, m_dict) in enumerate(valid_metric_results):
            c_idx = m_dict.get("client_index", idx)
            c_ds = m_dict.get("dataset", m_dict.get("client_id", f"client_{idx}"))
            w_frac = float(n_ex) / float(max(1, total_fit_samples))
            weight_descriptions.append(
                f"Client {c_idx} [{c_ds}]: {n_ex}/{total_fit_samples} ({w_frac * 100.0:.2f}%)"
            )
        logger.info(
            "[SERVER AGGREGATION] Starting FedAvg parameter aggregation (Round %s) | total_samples=%d | weights=[%s] | share_bn_buffers=%s | int_buffer_policy=%s",
            round_label,
            total_fit_samples,
            ", ".join(weight_descriptions),
            self.include_buffers,
            self.integer_buffer_policy,
        )

        aggregated_ndarrays = aggregate_parameters_with_policy(
            results=valid_param_results,
            reference_model=self.global_model,
            include_buffers=self.include_buffers,
            integer_buffer_policy=self.integer_buffer_policy,
        )

        # Update global model and current parameters atomically
        set_model_parameters(
            self.global_model,
            aggregated_ndarrays,
            include_buffers=self.include_buffers,
        )
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        self.current_parameters = parameters_aggregated

        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn is not None and valid_metric_results:
            metrics_aggregated = dict(self.fit_metrics_aggregation_fn(valid_metric_results))
        metrics_aggregated["server_round"] = int(server_round)
        metrics_aggregated["num_failures"] = int(len(failures))

        self._round_fit_metrics[int(server_round)] = dict(metrics_aggregated)
        logger.info(
            "[SERVER AGGREGATION] Completed FedAvg parameter aggregation (Round %s) | global_train_loss=%.4f | global_train_dice_macro=%.4f",
            round_label,
            float(metrics_aggregated.get("train_loss", metrics_aggregated.get("loss", 0.0))),
            float(metrics_aggregated.get("train_dice_macro", metrics_aggregated.get("dice_macro", 0.0))),
        )

        # If distributed evaluation is disabled, finalize round record and latest checkpoint here
        if self.fraction_evaluate == 0.0:
            self._finalize_round(
                server_round=int(server_round),
                val_loss=None,
                val_metrics={},
            )

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, Scalar]]:
        """Aggregate distributed client validation results and trigger round checkpointing."""
        round_label = self._format_round_label(server_round)
        if failures and not self.accept_failures:
            logger.error(
                "[SERVER EVALUATION ERROR] Round %s evaluation failed with %d client failure(s): %r",
                round_label,
                len(failures),
                failures,
            )
            raise FederatedStrategyError(
                f"Round {server_round} evaluation failed with {len(failures)} client failure(s) "
                f"and accept_failures=False: {failures!r}"
            )

        if not results:
            if not self.accept_failures:
                raise FederatedStrategyError(
                    f"Round {server_round} received zero client evaluation results."
                )
            return None, {}

        if len(results) < self.min_evaluate_clients and not self.accept_failures:
            raise FederatedStrategyError(
                f"Round {server_round} received {len(results)} evaluation result(s), "
                f"fewer than min_evaluate_clients={self.min_evaluate_clients}."
            )

        sorted_results = sorted(results, key=lambda item: str(item[0].cid))

        valid_loss_entries: list[tuple[int, float]] = []
        valid_metric_entries: list[tuple[int, Mapping[str, Scalar]]] = []

        for client_proxy, eval_res in sorted_results:
            n_ex = int(eval_res.num_examples)
            loss_val = float(eval_res.loss)
            if n_ex <= 0 or not _is_finite_number(loss_val):
                if not self.accept_failures:
                    raise FederatedStrategyError(
                        f"Client '{client_proxy.cid}' returned invalid evaluation result "
                        f"(num_examples={n_ex}, loss={loss_val!r})."
                    )
                logger.warning(
                    "Skipping invalid evaluation result from client '%s' in round %d.",
                    client_proxy.cid,
                    server_round,
                )
                continue

            valid_loss_entries.append((n_ex, loss_val))
            valid_metric_entries.append((n_ex, eval_res.metrics))

        if not valid_loss_entries:
            if not self.accept_failures:
                raise FederatedStrategyError(
                    f"Round {server_round} has no valid evaluation results."
                )
            return None, {}

        total_val_samples = sum(n_ex for n_ex, _ in valid_loss_entries)
        loss_aggregated = float(
            sum(n_ex * l_val for n_ex, l_val in valid_loss_entries) / float(total_val_samples)
        )

        metrics_aggregated: dict[str, Scalar] = {}
        if self.evaluate_metrics_aggregation_fn is not None and valid_metric_entries:
            metrics_aggregated = dict(
                self.evaluate_metrics_aggregation_fn(valid_metric_entries)
            )
        metrics_aggregated["server_round"] = int(server_round)
        metrics_aggregated["val_loss"] = float(loss_aggregated)
        metrics_aggregated["loss"] = float(loss_aggregated)
        metrics_aggregated["num_failures"] = int(len(failures))

        self._round_eval_metrics[int(server_round)] = dict(metrics_aggregated)
        self._finalize_round(
            server_round=int(server_round),
            val_loss=loss_aggregated,
            val_metrics=metrics_aggregated,
        )

        return loss_aggregated, metrics_aggregated

    def _is_metric_improved(self, candidate: float) -> bool:
        """Return ``True`` if ``candidate`` is finite and strictly improves ``self.best_metric_value``."""
        if not _is_finite_number(candidate):
            return False
        if self.monitor_mode == "max":
            return candidate > self.best_metric_value
        return candidate < self.best_metric_value

    def _resolve_monitored_metric(
        self,
        val_metrics: Mapping[str, Scalar],
        val_loss: float | None,
    ) -> float | None:
        """Extract the monitored validation metric if present and finite."""
        target = self.monitor_metric
        candidates = [target]
        if target.startswith("val_"):
            candidates.append(target.removeprefix("val_"))
        else:
            candidates.append(f"val_{target}")

        for key in candidates:
            if key in val_metrics:
                raw_val = val_metrics[key]
                if _is_finite_number(raw_val):
                    return float(raw_val)
                return None

        if target in {"val_loss", "loss"} and val_loss is not None and _is_finite_number(val_loss):
            return float(val_loss)

        return None

    def _finalize_round(
        self,
        server_round: int,
        val_loss: float | None,
        val_metrics: Mapping[str, Scalar],
    ) -> dict[str, Any]:
        """Record round summary and persist ``latest_checkpoint.pt`` and ``best_checkpoint.pt``."""
        round_label = self._format_round_label(server_round)
        round_start_t = self._round_start_times.get(int(server_round))
        round_duration_sec = (
            round(time.perf_counter() - round_start_t, 4)
            if round_start_t is not None
            else 0.0
        )

        fit_metrics = dict(self._round_fit_metrics.get(int(server_round), {}))
        eval_metrics = dict(val_metrics)

        if eval_metrics or val_loss is not None:
            per_lesion_global = (
                f"MA={float(eval_metrics.get('val_dice_MA', eval_metrics.get('dice_MA', 0.0))):.4f}, "
                f"HE={float(eval_metrics.get('val_dice_HE', eval_metrics.get('dice_HE', 0.0))):.4f}, "
                f"EX={float(eval_metrics.get('val_dice_EX', eval_metrics.get('dice_EX', 0.0))):.4f}, "
                f"SE={float(eval_metrics.get('val_dice_SE', eval_metrics.get('dice_SE', 0.0))):.4f}"
            )
            logger.info(
                "[GLOBAL VALIDATION] Round %s | global_val_loss=%.4f | global_val_dice_macro=%.4f | client_macro_val_dice=%.4f | global_val_iou_macro=%.4f | lesion_dice=[%s] | total_val_samples=%d",
                round_label,
                float(val_loss if val_loss is not None else eval_metrics.get("val_loss", 0.0)),
                float(eval_metrics.get("val_dice_macro", eval_metrics.get("dice_macro", 0.0))),
                float(eval_metrics.get("macro_client_val_dice_macro", eval_metrics.get("macro_client_dice_macro", 0.0))),
                float(eval_metrics.get("val_iou_macro", eval_metrics.get("iou_macro", 0.0))),
                per_lesion_global,
                int(eval_metrics.get("val_num_samples_total", eval_metrics.get("num_samples_total", 0))),
            )

        combined_metrics: dict[str, Any] = {}
        for k, v in fit_metrics.items():
            combined_metrics[k if k.startswith("train_") else f"train_{k}"] = v
        for k, v in eval_metrics.items():
            combined_metrics[k] = v
            if not k.startswith("val_") and k not in _METADATA_SCALAR_KEYS:
                combined_metrics[f"val_{k}"] = v
        if val_loss is not None:
            combined_metrics["val_loss"] = float(val_loss)

        monitored_val = self._resolve_monitored_metric(eval_metrics, val_loss)
        is_best = False
        if monitored_val is not None and self._is_metric_improved(monitored_val):
            self.best_metric_value = float(monitored_val)
            self.best_round = int(server_round)
            is_best = True

        round_record: dict[str, Any] = {
            "round": int(server_round),
            "epoch": int(server_round),
            "round_duration_sec": float(round_duration_sec),
            "train": fit_metrics,
            "val": eval_metrics,
            "val_loss": float(val_loss) if val_loss is not None and _is_finite_number(val_loss) else None,
            "monitored_metric": self.monitor_metric,
            "monitored_value": monitored_val,
            "is_best": bool(is_best),
            "best_metric_value": (
                float(self.best_metric_value)
                if _is_finite_number(self.best_metric_value)
                else None
            ),
            "best_round": int(self.best_round),
        }
        self.round_history.append(round_record)

        if self.save_checkpoints and self.checkpoint_dir is not None:
            if self.save_every_round > 0 and (int(server_round) % self.save_every_round == 0):
                latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
                self.latest_checkpoint_path = save_global_checkpoint(
                    checkpoint_path=latest_path,
                    model=self.global_model,
                    server_round=int(server_round),
                    metrics=combined_metrics,
                    best_metric_value=self.best_metric_value,
                    best_round=self.best_round,
                    monitor_metric=self.monitor_metric,
                    monitor_mode=self.monitor_mode,
                    seed=self.seed,
                    federated_config=self.federated_config,
                    model_config=self.model_config,
                    dataset_config=self.dataset_config,
                    project_config=self.project_config,
                    round_history=self.round_history,
                    extra_metadata={
                        "checkpoint_type": "latest",
                        "initialization": self.initialization_metadata,
                    },
                )
                logger.info(
                    "[CHECKPOINT] Saved latest global checkpoint for Round %s -> %s",
                    round_label,
                    format_project_path(self.latest_checkpoint_path),
                )

            if is_best:
                best_path = self.checkpoint_dir / "best_checkpoint.pt"
                self.best_checkpoint_path = save_global_checkpoint(
                    checkpoint_path=best_path,
                    model=self.global_model,
                    server_round=int(server_round),
                    metrics=combined_metrics,
                    best_metric_value=self.best_metric_value,
                    best_round=self.best_round,
                    monitor_metric=self.monitor_metric,
                    monitor_mode=self.monitor_mode,
                    seed=self.seed,
                    federated_config=self.federated_config,
                    model_config=self.model_config,
                    dataset_config=self.dataset_config,
                    project_config=self.project_config,
                    round_history=self.round_history,
                    extra_metadata={
                        "checkpoint_type": "best",
                        "initialization": self.initialization_metadata,
                    },
                )
                logger.info(
                    "[CHECKPOINT] New best %s=%.4f at Round %s! Updated best global checkpoint -> %s",
                    self.monitor_metric,
                    self.best_metric_value,
                    round_label,
                    format_project_path(self.best_checkpoint_path),
                )
            elif monitored_val is not None:
                logger.info(
                    "[CHECKPOINT] Best %s remains %.4f (from Round %d); best_checkpoint.pt unchanged.",
                    self.monitor_metric,
                    self.best_metric_value,
                    self.best_round,
                )

        logger.info(
            "[ROUND COMPLETE] Round %s finished in %.2fs | best_%s=%s (Round %d)",
            round_label,
            round_duration_sec,
            self.monitor_metric,
            f"{self.best_metric_value:.4f}" if _is_finite_number(self.best_metric_value) else "N/A",
            self.best_round,
        )
        return round_record

    def get_global_parameters_ndarrays(self) -> list[np.ndarray]:
        """Return the current global model parameters as an ordered list of NumPy arrays."""
        return get_model_parameters(
            self.global_model,
            include_buffers=self.include_buffers,
        )


def build_fedavg_strategy(
    federated_config: Mapping[str, Any] | str | Path = "federated.yaml",
    model_config: Mapping[str, Any] | str | Path = "model.yaml",
    dataset_config: Mapping[str, Any] | str | Path = "dataset.yaml",
    project_config: Mapping[str, Any] | str | Path = "project.yaml",
    *,
    global_model: nn.Module | None = None,
    initial_parameters: Parameters | None = None,
    checkpoint_dir: str | Path | None = None,
    save_checkpoints: bool = True,
    accept_failures: bool = False,
    include_buffers: bool | None = None,
    integer_buffer_policy: str = "max",
    local_epochs: int | None = None,
    learning_rate: float | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    seed: int | None = None,
    num_rounds: int | None = None,
    device: torch.device | str = "cpu",
) -> tuple[FedMorphisFedAvg, nn.Module]:
    """Construct a configured :class:`FedMorphisFedAvg` strategy and global model."""
    fed_cfg = (
        load_yaml_config(federated_config, validate=True)
        if isinstance(federated_config, (str, Path))
        else dict(federated_config)
    )
    mdl_cfg = (
        load_yaml_config(model_config, validate=True)
        if isinstance(model_config, (str, Path))
        else dict(model_config)
    )
    ds_cfg = (
        load_yaml_config(dataset_config, validate=True)
        if isinstance(dataset_config, (str, Path))
        else dict(dataset_config)
    )
    proj_cfg = (
        load_yaml_config(project_config, validate=True)
        if isinstance(project_config, (str, Path))
        else dict(project_config)
    )

    init_info: dict[str, Any] = {}
    if global_model is None:
        global_model, init_info = initialize_global_model(
            federated_config=fed_cfg,
            model_config=mdl_cfg,
            project_config=proj_cfg,
            seed=seed,
            device=device,
        )

    strategy = FedMorphisFedAvg(
        global_model=global_model,
        federated_config=fed_cfg,
        model_config=mdl_cfg,
        dataset_config=ds_cfg,
        project_config=proj_cfg,
        initial_parameters=initial_parameters,
        accept_failures=accept_failures,
        include_buffers=include_buffers,
        integer_buffer_policy=integer_buffer_policy,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        checkpoint_dir=checkpoint_dir,
        save_checkpoints=save_checkpoints,
        seed=seed,
        num_rounds=num_rounds,
        initialization_metadata=init_info,
    )
    return strategy, global_model


__all__ = [
    "FedMorphisFedAvg",
    "FederatedStrategyError",
    "SUPPORTED_INTEGER_BUFFER_POLICIES",
    "aggregate_client_metrics",
    "aggregate_parameters_with_policy",
    "build_fedavg_strategy",
    "initialize_global_model",
    "load_global_checkpoint",
    "save_global_checkpoint",
]
