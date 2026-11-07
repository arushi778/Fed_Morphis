"""YAML configuration loading, deep-merging, and validation utilities for FedMorphis."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import yaml

from src.utils.device import SUPPORTED_DEVICES
from src.utils.paths import get_configs_dir
from src.utils.seed import MAX_UINT32_SEED

CONFIG_FILES: tuple[str, ...] = (
    "project.yaml",
    "dataset.yaml",
    "model.yaml",
    "ssl.yaml",
    "federated.yaml",
    "experiment.yaml",
)

SUPPORTED_FEDERATED_STRATEGIES: tuple[str, ...] = ("FedAvg", "FedProx")
SUPPORTED_PARTITION_STRATEGIES: tuple[str, ...] = (
    "iid",
    "dirichlet",
    "non_iid",
    "cross_dataset",
    "by_dataset",
)
SUPPORTED_IMAGE_INTERPOLATIONS: tuple[str, ...] = (
    "linear",
    "cubic",
    "area",
    "lanczos4",
)
SUPPORTED_ENCODER_WEIGHT_SOURCES: tuple[str, ...] = (
    "imagenet",
    "supervised_checkpoint",
    "ssl_checkpoint",
    "random",
)
SUPPORTED_OPTIMIZERS: tuple[str, ...] = ("adam", "adamw", "sgd")
SUPPORTED_MONITOR_MODES: tuple[str, ...] = ("max", "min")
CANONICAL_MODEL_CHANNELS: tuple[str, ...] = ("MA", "HE", "EX", "SE")


class ConfigValidationError(ValueError):
    """Raised when a YAML configuration file or dictionary fails validation."""


def _is_int(value: Any) -> bool:
    """Return True if value is an integer and not a boolean."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    """Return True if value is a finite int or float and not a boolean."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_mapping(config: Any, context: str) -> Mapping[str, Any]:
    """Ensure ``config`` is a dictionary/mapping."""
    if not isinstance(config, Mapping):
        raise ConfigValidationError(
            f"[{context}] Expected configuration mapping/dict, got {type(config).__name__}."
        )
    return config


def _require_fields(
    config: Mapping[str, Any], required: tuple[str, ...], context: str
) -> None:
    """Ensure all required keys are present in ``config``."""
    missing = [key for key in required if key not in config]
    if missing:
        missing_str = ", ".join(missing)
        raise ConfigValidationError(
            f"[{context}] Missing required configuration field(s): {missing_str}."
        )


def _require_non_empty_str(value: Any, field_name: str, context: str) -> str:
    """Ensure ``value`` is a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(
            f"[{context}] Field '{field_name}' must be a non-empty string, got {value!r}."
        )
    return value


def _require_bool(value: Any, field_name: str, context: str) -> bool:
    """Ensure ``value`` is a boolean."""
    if not isinstance(value, bool):
        raise ConfigValidationError(
            f"[{context}] Field '{field_name}' must be a boolean, got {type(value).__name__} ({value!r})."
        )
    return value


def _require_int(
    value: Any,
    field_name: str,
    context: str,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    """Ensure ``value`` is an integer within optional bounds."""
    if not _is_int(value):
        raise ConfigValidationError(
            f"[{context}] Field '{field_name}' must be an integer, got {type(value).__name__} ({value!r})."
        )
    if min_value is not None and value < min_value:
        raise ConfigValidationError(
            f"[{context}] Field '{field_name}' must be >= {min_value}, got {value}."
        )
    if max_value is not None and value > max_value:
        raise ConfigValidationError(
            f"[{context}] Field '{field_name}' must be <= {max_value}, got {value}."
        )
    return int(value)


def _require_number(
    value: Any,
    field_name: str,
    context: str,
    min_value: float | None = None,
    max_value: float | None = None,
    strict_min: bool = False,
) -> float:
    """Ensure ``value`` is a numeric (int or float) within optional bounds."""
    if not _is_number(value):
        raise ConfigValidationError(
            f"[{context}] Field '{field_name}' must be a number, got {type(value).__name__} ({value!r})."
        )
    num = float(value)
    if min_value is not None:
        if strict_min and num <= min_value:
            raise ConfigValidationError(
                f"[{context}] Field '{field_name}' must be > {min_value}, got {value}."
            )
        if not strict_min and num < min_value:
            raise ConfigValidationError(
                f"[{context}] Field '{field_name}' must be >= {min_value}, got {value}."
            )
    if max_value is not None and num > max_value:
        raise ConfigValidationError(
            f"[{context}] Field '{field_name}' must be <= {max_value}, got {value}."
        )
    return num


def validate_project_config(config: Mapping[str, Any]) -> None:
    """Validate ``project.yaml`` configuration fields."""
    ctx = "project.yaml"
    cfg = _require_mapping(config, ctx)
    _require_fields(
        cfg,
        ("project_name", "random_seed", "device", "output_directory"),
        ctx,
    )

    _require_non_empty_str(cfg["project_name"], "project_name", ctx)
    _require_int(
        cfg["random_seed"],
        "random_seed",
        ctx,
        min_value=0,
        max_value=MAX_UINT32_SEED,
    )

    if "deterministic" in cfg:
        _require_bool(cfg["deterministic"], "deterministic", ctx)

    device_val = _require_non_empty_str(cfg["device"], "device", ctx)
    if device_val.strip().lower() not in SUPPORTED_DEVICES:
        raise ConfigValidationError(
            f"[{ctx}] Field 'device' must be one of {SUPPORTED_DEVICES}, got {device_val!r}."
        )

    for dir_field in (
        "output_directory",
        "logs_directory",
        "checkpoints_directory",
        "data_directory",
    ):
        if dir_field in cfg:
            _require_non_empty_str(cfg[dir_field], dir_field, ctx)


def validate_dataset_config(config: Mapping[str, Any]) -> None:
    """Validate ``dataset.yaml`` configuration fields."""
    ctx = "dataset.yaml"
    cfg = _require_mapping(config, ctx)
    _require_fields(
        cfg,
        (
            "dataset_names",
            "image_size",
            "batch_size",
            "number_of_workers",
            "normalization",
            "augmentation",
            "train_validation_test_split",
        ),
        ctx,
    )

    dataset_names = cfg["dataset_names"]
    if not isinstance(dataset_names, list) or len(dataset_names) == 0:
        raise ConfigValidationError(
            f"[{ctx}] Field 'dataset_names' must be a non-empty list of strings."
        )
    for idx, name in enumerate(dataset_names):
        _require_non_empty_str(name, f"dataset_names[{idx}]", ctx)

    image_size = cfg["image_size"]
    if isinstance(image_size, Mapping):
        _require_fields(image_size, ("height", "width"), f"{ctx}:image_size")
        _require_int(image_size["height"], "image_size.height", ctx, min_value=1)
        _require_int(image_size["width"], "image_size.width", ctx, min_value=1)
    elif _is_int(image_size):
        _require_int(image_size, "image_size", ctx, min_value=1)
    else:
        raise ConfigValidationError(
            f"[{ctx}] Field 'image_size' must be a mapping with 'height'/'width' or a positive integer."
        )

    _require_int(cfg["batch_size"], "batch_size", ctx, min_value=1)
    _require_int(cfg["number_of_workers"], "number_of_workers", ctx, min_value=0)

    if "pin_memory" in cfg:
        _require_bool(cfg["pin_memory"], "pin_memory", ctx)
    if "persistent_workers" in cfg:
        _require_bool(cfg["persistent_workers"], "persistent_workers", ctx)
    if "drop_last_train" in cfg:
        _require_bool(cfg["drop_last_train"], "drop_last_train", ctx)

    if "evaluation_transforms" in cfg:
        eval_tf = _require_mapping(
            cfg["evaluation_transforms"], f"{ctx}:evaluation_transforms"
        )
        if "resize_interpolation" in eval_tf:
            interp = _require_non_empty_str(
                eval_tf["resize_interpolation"],
                "evaluation_transforms.resize_interpolation",
                ctx,
            ).lower()
            if interp not in SUPPORTED_IMAGE_INTERPOLATIONS:
                raise ConfigValidationError(
                    f"[{ctx}] Field 'evaluation_transforms.resize_interpolation' must be one of "
                    f"{SUPPORTED_IMAGE_INTERPOLATIONS}, got {interp!r}."
                )
        if "mask_interpolation" in eval_tf:
            m_interp = _require_non_empty_str(
                eval_tf["mask_interpolation"],
                "evaluation_transforms.mask_interpolation",
                ctx,
            ).lower()
            if m_interp != "nearest":
                raise ConfigValidationError(
                    f"[{ctx}] Field 'evaluation_transforms.mask_interpolation' must be 'nearest' "
                    f"to preserve binary lesion masks, got {m_interp!r}."
                )
        if "normalize" in eval_tf:
            _require_bool(eval_tf["normalize"], "evaluation_transforms.normalize", ctx)

    norm = _require_mapping(cfg["normalization"], f"{ctx}:normalization")
    if "enabled" in norm:
        _require_bool(norm["enabled"], "normalization.enabled", ctx)

    aug = _require_mapping(cfg["augmentation"], f"{ctx}:augmentation")
    if "enabled" in aug:
        _require_bool(aug["enabled"], "augmentation.enabled", ctx)
    for prob_key in (
        "horizontal_flip_prob",
        "vertical_flip_prob",
        "random_rotate90_prob",
    ):
        if prob_key in aug:
            _require_number(
                aug[prob_key],
                f"augmentation.{prob_key}",
                ctx,
                min_value=0.0,
                max_value=1.0,
            )

    splits = _require_mapping(
        cfg["train_validation_test_split"], f"{ctx}:train_validation_test_split"
    )
    _require_fields(
        splits,
        ("train", "validation", "test"),
        f"{ctx}:train_validation_test_split",
    )
    total_split = 0.0
    for split_key in ("train", "validation", "test"):
        ratio = _require_number(
            splits[split_key],
            f"train_validation_test_split.{split_key}",
            ctx,
            min_value=0.0,
            max_value=1.0,
            strict_min=True,
        )
        total_split += ratio
    if abs(total_split - 1.0) > 1e-5:
        raise ConfigValidationError(
            f"[{ctx}] 'train_validation_test_split' ratios must sum to 1.0, got {total_split:.4f}."
        )


def validate_model_config(config: Mapping[str, Any]) -> None:
    """Validate ``model.yaml`` configuration fields."""
    ctx = "model.yaml"
    cfg = _require_mapping(config, ctx)
    _require_fields(
        cfg,
        (
            "architecture",
            "encoder",
            "pretrained",
            "number_of_classes",
            "input_channels",
        ),
        ctx,
    )

    _require_non_empty_str(cfg["architecture"], "architecture", ctx)
    _require_non_empty_str(cfg["encoder"], "encoder", ctx)
    _require_bool(cfg["pretrained"], "pretrained", ctx)
    _require_int(cfg["number_of_classes"], "number_of_classes", ctx, min_value=2)
    _require_int(cfg["input_channels"], "input_channels", ctx, min_value=1)

    if "encoder_weights_source" in cfg:
        src_val = _require_non_empty_str(
            cfg["encoder_weights_source"], "encoder_weights_source", ctx
        )
        if src_val not in SUPPORTED_ENCODER_WEIGHT_SOURCES:
            raise ConfigValidationError(
                f"[{ctx}] Field 'encoder_weights_source' must be one of "
                f"{SUPPORTED_ENCODER_WEIGHT_SOURCES}, got {src_val!r}."
            )

    if "strict_pretrained" in cfg:
        _require_bool(cfg["strict_pretrained"], "strict_pretrained", ctx)

    if "ssl_checkpoint_path" in cfg and cfg["ssl_checkpoint_path"] is not None:
        _require_non_empty_str(cfg["ssl_checkpoint_path"], "ssl_checkpoint_path", ctx)

    if "output_channels" in cfg:
        out_ch = _require_int(cfg["output_channels"], "output_channels", ctx, min_value=1)
        if out_ch != 4:
            raise ConfigValidationError(
                f"[{ctx}] Field 'output_channels' must be 4 for [MA, HE, EX, SE], got {out_ch}."
            )

    if "output_channel_order" in cfg:
        order = cfg["output_channel_order"]
        if not isinstance(order, list) or tuple(order) != CANONICAL_MODEL_CHANNELS:
            raise ConfigValidationError(
                f"[{ctx}] Field 'output_channel_order' must be {list(CANONICAL_MODEL_CHANNELS)}, "
                f"got {order!r}."
            )

    if "loss" in cfg:
        loss_cfg = _require_mapping(cfg["loss"], f"{ctx}:loss")
        bce_w = float(
            _require_number(
                loss_cfg.get("bce_weight", 0.5),
                "loss.bce_weight",
                ctx,
                min_value=0.0,
            )
        )
        dice_w = float(
            _require_number(
                loss_cfg.get("dice_weight", 0.5),
                "loss.dice_weight",
                ctx,
                min_value=0.0,
            )
        )
        if (bce_w + dice_w) <= 0.0:
            raise ConfigValidationError(
                f"[{ctx}] 'loss.bce_weight' + 'loss.dice_weight' must be > 0."
            )
        if "smooth" in loss_cfg:
            _require_number(
                loss_cfg["smooth"], "loss.smooth", ctx, min_value=0.0, strict_min=True
            )

    if "training" in cfg:
        tr_cfg = _require_mapping(cfg["training"], f"{ctx}:training")
        if "optimizer" in tr_cfg:
            opt_name = _require_non_empty_str(
                tr_cfg["optimizer"], "training.optimizer", ctx
            ).lower()
            if opt_name not in SUPPORTED_OPTIMIZERS:
                raise ConfigValidationError(
                    f"[{ctx}] Field 'training.optimizer' must be one of "
                    f"{SUPPORTED_OPTIMIZERS}, got {tr_cfg['optimizer']!r}."
                )
        if "learning_rate" in tr_cfg:
            _require_number(
                tr_cfg["learning_rate"],
                "training.learning_rate",
                ctx,
                min_value=0.0,
                strict_min=True,
            )
        if "weight_decay" in tr_cfg:
            _require_number(
                tr_cfg["weight_decay"],
                "training.weight_decay",
                ctx,
                min_value=0.0,
            )
        if "epochs" in tr_cfg:
            _require_int(tr_cfg["epochs"], "training.epochs", ctx, min_value=1)
        if "mixed_precision" in tr_cfg:
            _require_bool(
                tr_cfg["mixed_precision"], "training.mixed_precision", ctx
            )
        if (
            "gradient_clip_max_norm" in tr_cfg
            and tr_cfg["gradient_clip_max_norm"] is not None
        ):
            _require_number(
                tr_cfg["gradient_clip_max_norm"],
                "training.gradient_clip_max_norm",
                ctx,
                min_value=0.0,
                strict_min=True,
            )
        if "metric_threshold" in tr_cfg:
            _require_number(
                tr_cfg["metric_threshold"],
                "training.metric_threshold",
                ctx,
                min_value=0.0,
                max_value=1.0,
                strict_min=True,
            )
            if float(tr_cfg["metric_threshold"]) >= 1.0:
                raise ConfigValidationError(
                    f"[{ctx}] Field 'training.metric_threshold' must be in (0.0, 1.0), "
                    f"got {tr_cfg['metric_threshold']}."
                )
        if "checkpoint_directory" in tr_cfg:
            _require_non_empty_str(
                tr_cfg["checkpoint_directory"],
                "training.checkpoint_directory",
                ctx,
            )
        if "monitor_metric" in tr_cfg:
            _require_non_empty_str(
                tr_cfg["monitor_metric"], "training.monitor_metric", ctx
            )
        if "monitor_mode" in tr_cfg:
            m_mode = _require_non_empty_str(
                tr_cfg["monitor_mode"], "training.monitor_mode", ctx
            ).lower()
            if m_mode not in SUPPORTED_MONITOR_MODES:
                raise ConfigValidationError(
                    f"[{ctx}] Field 'training.monitor_mode' must be one of "
                    f"{SUPPORTED_MONITOR_MODES}, got {tr_cfg['monitor_mode']!r}."
                )

    if "simclr" in cfg:
        validate_ssl_config(cfg["simclr"])


def validate_ssl_config(config: Mapping[str, Any]) -> None:
    """Validate ``ssl.yaml`` (or ``model.yaml:simclr``) configuration fields."""
    ctx = "ssl.yaml"
    cfg = _require_mapping(config, ctx)
    _require_fields(
        cfg,
        (
            "method",
            "projection_dimension",
            "temperature",
            "learning_rate",
            "epochs",
        ),
        ctx,
    )

    method_val = _require_non_empty_str(cfg["method"], "method", ctx)
    if method_val.strip().lower() != "simclr":
        raise ConfigValidationError(
            f"[{ctx}] Field 'method' must be 'SimCLR', got {method_val!r}."
        )

    _require_int(cfg["projection_dimension"], "projection_dimension", ctx, min_value=1)
    if "projection_hidden_dimension" in cfg:
        _require_int(
            cfg["projection_hidden_dimension"],
            "projection_hidden_dimension",
            ctx,
            min_value=1,
        )
    if "projection_use_batchnorm" in cfg:
        _require_bool(
            cfg["projection_use_batchnorm"], "projection_use_batchnorm", ctx
        )

    _require_number(
        cfg["temperature"], "temperature", ctx, min_value=0.0, strict_min=True
    )
    if "normalize_projections" in cfg:
        _require_bool(cfg["normalize_projections"], "normalize_projections", ctx)

    _require_number(
        cfg["learning_rate"], "learning_rate", ctx, min_value=0.0, strict_min=True
    )
    if "weight_decay" in cfg:
        _require_number(cfg["weight_decay"], "weight_decay", ctx, min_value=0.0)
    if "optimizer" in cfg:
        opt_name = _require_non_empty_str(cfg["optimizer"], "optimizer", ctx).lower()
        if opt_name not in SUPPORTED_OPTIMIZERS:
            raise ConfigValidationError(
                f"[{ctx}] Field 'optimizer' must be one of {SUPPORTED_OPTIMIZERS}, "
                f"got {cfg['optimizer']!r}."
            )

    _require_int(cfg["epochs"], "epochs", ctx, min_value=1)

    if "batch_size" in cfg:
        _require_int(cfg["batch_size"], "batch_size", ctx, min_value=2)

    if "random_seed" in cfg:
        _require_int(
            cfg["random_seed"],
            "random_seed",
            ctx,
            min_value=0,
            max_value=MAX_UINT32_SEED,
        )

    if "pretrained" in cfg:
        _require_bool(cfg["pretrained"], "pretrained", ctx)
    if "strict_pretrained" in cfg:
        _require_bool(cfg["strict_pretrained"], "strict_pretrained", ctx)

    if "ssl_checkpoint_path" in cfg and cfg["ssl_checkpoint_path"] is not None:
        _require_non_empty_str(cfg["ssl_checkpoint_path"], "ssl_checkpoint_path", ctx)
    if "encoder_weights_source" in cfg:
        src_val = _require_non_empty_str(
            cfg["encoder_weights_source"], "encoder_weights_source", ctx
        ).lower()
        if src_val not in SUPPORTED_ENCODER_WEIGHT_SOURCES:
            raise ConfigValidationError(
                f"[{ctx}] Field 'encoder_weights_source' must be one of "
                f"{SUPPORTED_ENCODER_WEIGHT_SOURCES}, got {cfg['encoder_weights_source']!r}."
            )

    if "augmentations" in cfg:
        aug = _require_mapping(cfg["augmentations"], f"{ctx}:augmentations")
        if "image_size" in aug:
            _require_int(aug["image_size"], "augmentations.image_size", ctx, min_value=32)
        if "random_resized_crop_scale" in aug:
            scale = aug["random_resized_crop_scale"]
            if (
                not isinstance(scale, (list, tuple))
                or len(scale) != 2
                or not _is_number(scale[0])
                or not _is_number(scale[1])
                or float(scale[0]) <= 0.0
                or float(scale[1]) > 1.0
                or float(scale[0]) > float(scale[1])
            ):
                raise ConfigValidationError(
                    f"[{ctx}] Field 'augmentations.random_resized_crop_scale' must be "
                    f"[min_scale, max_scale] with 0 < min_scale <= max_scale <= 1.0, got {scale!r}."
                )
        for prob_key in (
            "horizontal_flip_prob",
            "vertical_flip_prob",
            "random_rotate90_prob",
            "color_jitter_prob",
            "gaussian_blur_prob",
        ):
            if prob_key in aug:
                _require_number(
                    aug[prob_key],
                    f"augmentations.{prob_key}",
                    ctx,
                    min_value=0.0,
                    max_value=1.0,
                )


def validate_federated_config(config: Mapping[str, Any]) -> None:
    """Validate ``federated.yaml`` configuration fields."""
    ctx = "federated.yaml"
    cfg = _require_mapping(config, ctx)
    _require_fields(
        cfg,
        (
            "strategy",
            "number_of_clients",
            "communication_rounds",
            "local_epochs",
            "client_fraction",
            "aggregation_method",
        ),
        ctx,
    )

    strategy = _require_non_empty_str(cfg["strategy"], "strategy", ctx)
    if strategy not in SUPPORTED_FEDERATED_STRATEGIES:
        raise ConfigValidationError(
            f"[{ctx}] Field 'strategy' must be one of {SUPPORTED_FEDERATED_STRATEGIES}, "
            f"got {strategy!r}."
        )

    _require_int(cfg["number_of_clients"], "number_of_clients", ctx, min_value=1)
    _require_int(cfg["communication_rounds"], "communication_rounds", ctx, min_value=1)
    _require_int(cfg["local_epochs"], "local_epochs", ctx, min_value=1)
    _require_number(
        cfg["client_fraction"],
        "client_fraction",
        ctx,
        min_value=0.0,
        max_value=1.0,
        strict_min=True,
    )
    _require_non_empty_str(cfg["aggregation_method"], "aggregation_method", ctx)

    if "fedprox_mu" in cfg:
        _require_number(cfg["fedprox_mu"], "fedprox_mu", ctx, min_value=0.0)

    if "partitioning" in cfg:
        part = _require_mapping(cfg["partitioning"], f"{ctx}:partitioning")
        scheme_raw = part.get("scheme", part.get("strategy"))
        if scheme_raw is not None:
            scheme = _require_non_empty_str(
                scheme_raw, "partitioning.scheme", ctx
            ).lower()
            if scheme not in SUPPORTED_PARTITION_STRATEGIES:
                raise ConfigValidationError(
                    f"[{ctx}] Field 'partitioning.scheme' must be one of "
                    f"{SUPPORTED_PARTITION_STRATEGIES}, got {scheme_raw!r}."
                )
        if "dirichlet_alpha" in part:
            _require_number(
                part["dirichlet_alpha"],
                "partitioning.dirichlet_alpha",
                ctx,
                min_value=0.0,
                strict_min=True,
            )
        if "seed" in part:
            _require_int(
                part["seed"],
                "partitioning.seed",
                ctx,
                min_value=0,
                max_value=MAX_UINT32_SEED,
            )
        if "min_samples_per_client" in part:
            _require_int(
                part["min_samples_per_client"],
                "partitioning.min_samples_per_client",
                ctx,
                min_value=1,
            )


def validate_experiment_config(config: Mapping[str, Any]) -> None:
    """Validate ``experiment.yaml`` configuration fields."""
    ctx = "experiment.yaml"
    cfg = _require_mapping(config, ctx)
    _require_fields(
        cfg,
        (
            "experiment_name",
            "tracking_enabled",
            "checkpoint_directory",
            "evaluation_metrics",
        ),
        ctx,
    )

    _require_non_empty_str(cfg["experiment_name"], "experiment_name", ctx)
    _require_bool(cfg["tracking_enabled"], "tracking_enabled", ctx)
    _require_non_empty_str(cfg["checkpoint_directory"], "checkpoint_directory", ctx)

    metrics = cfg["evaluation_metrics"]
    if not isinstance(metrics, list) or len(metrics) == 0:
        raise ConfigValidationError(
            f"[{ctx}] Field 'evaluation_metrics' must be a non-empty list of strings."
        )
    for idx, metric in enumerate(metrics):
        _require_non_empty_str(metric, f"evaluation_metrics[{idx}]", ctx)


CONFIG_VALIDATORS: dict[str, Callable[[Mapping[str, Any]], None]] = {
    "project": validate_project_config,
    "dataset": validate_dataset_config,
    "model": validate_model_config,
    "ssl": validate_ssl_config,
    "federated": validate_federated_config,
    "experiment": validate_experiment_config,
}


def validate_config(config: Mapping[str, Any], config_name: str) -> dict[str, Any]:
    """Validate a configuration dictionary against its named section schema.

    Args:
        config: Configuration dictionary to validate.
        config_name: Section name or YAML filename (e.g., ``"project"`` or ``"project.yaml"``).

    Returns:
        Deep copy of the validated configuration dictionary.

    Raises:
        ConfigValidationError: If ``config_name`` is unknown or ``config`` is invalid.
    """
    if not isinstance(config_name, str) or not config_name.strip():
        raise ConfigValidationError("Configuration name must be a non-empty string.")

    stem = Path(config_name.strip()).stem.lower()
    validator = CONFIG_VALIDATORS.get(stem)
    if validator is None:
        supported = ", ".join(sorted(CONFIG_VALIDATORS.keys()))
        raise ConfigValidationError(
            f"Unknown configuration section {config_name!r}. Supported sections: {supported}."
        )

    validator(config)
    return copy.deepcopy(dict(config))


def deep_merge_configs(
    base: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Recursively deep-merge ``overrides`` into ``base`` without mutating inputs.

    Nested dictionaries are merged recursively so unrelated keys in ``base`` are
    preserved while matching leaf keys are updated from ``overrides``. Non-mapping
    values (scalars, lists) in ``overrides`` replace the corresponding value in ``base``.

    Args:
        base: Base configuration mapping.
        overrides: Override mapping to apply on top of ``base``.

    Returns:
        A new deep-copied dictionary containing the merged configuration.

    Raises:
        ConfigValidationError: If either ``base`` or ``overrides`` is not a mapping.
    """
    if not isinstance(base, Mapping):
        raise ConfigValidationError(
            f"Base configuration for merge must be a mapping/dict, got {type(base).__name__}."
        )
    if not isinstance(overrides, Mapping):
        raise ConfigValidationError(
            f"Override configuration for merge must be a mapping/dict, got {type(overrides).__name__}."
        )

    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, override_value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(override_value, Mapping)
        ):
            merged[key] = deep_merge_configs(merged[key], override_value)
        else:
            merged[key] = copy.deepcopy(override_value)

    return merged


def load_yaml_config(
    config_path: str | Path, validate: bool = False
) -> dict[str, Any]:
    """Load a YAML configuration file safely and return its contents as a dictionary.

    Args:
        config_path: Filename inside ``configs/`` (e.g., ``"project.yaml"``) or a
            ``Path`` to a YAML file.
        validate: When ``True`` and the file stem matches a known FedMorphis section,
            automatically run schema validation on the loaded dictionary.

    Returns:
        Parsed dictionary from the YAML file.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ConfigValidationError: If the file contains malformed YAML, is not a mapping,
            or fails validation when ``validate=True``.
    """
    path = Path(config_path)
    if not path.is_absolute() and not path.exists():
        path = get_configs_dir() / path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigValidationError(
            f"Malformed YAML in configuration file {path}: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ConfigValidationError(
            f"Expected YAML mapping in {path}, got {type(data).__name__}"
        )

    if validate:
        stem = path.stem.lower()
        if stem in CONFIG_VALIDATORS:
            validate_config(data, stem)

    return data


def load_all_configs(validate: bool = True) -> dict[str, dict[str, Any]]:
    """Load (and optionally validate) all standard FedMorphis configuration files."""
    configs: dict[str, dict[str, Any]] = {}
    for filename in CONFIG_FILES:
        stem = Path(filename).stem
        configs[stem] = load_yaml_config(filename, validate=validate)
    return configs


def validate_all_configs(
    configs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Validate all 6 FedMorphis configuration sections.

    Args:
        configs: Optional dictionary mapping section names (``"project"``, ``"dataset"``,
            etc.) to configuration dictionaries. If ``None``, loads all configs from
            ``configs/*.yaml`` and validates them.

    Returns:
        Deep copy of the validated multi-section configuration dictionary.

    Raises:
        ConfigValidationError: If any required section is missing or fails validation.
    """
    if configs is None:
        return load_all_configs(validate=True)

    _require_mapping(configs, "all_configs")
    expected_sections = tuple(Path(f).stem for f in CONFIG_FILES)
    _require_fields(configs, expected_sections, "all_configs")

    validated: dict[str, dict[str, Any]] = {}
    for section in expected_sections:
        validated[section] = validate_config(configs[section], section)
    return validated
