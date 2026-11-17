"""Supervised U-Net + ResNet-50 retinal lesion segmentation baseline CLI.

Supports:
- Full supervised training and validation from YAML configs
- ``--dry-run`` mode (single forward/backward + validation verification step)
- ``--smoke-test`` mode (real-data smoke test on DDR and IDRiD samples with
  CUDA memory profiling on the target GPU)
- ``--resume`` from a saved checkpoint
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.dataset import (
    RetinalLesionSegmentationDataset,
    build_segmentation_dataloader,
)
from src.data.manifests import load_manifest_json
from src.data.transforms import build_segmentation_transforms
from src.engine.evaluator import evaluate_aupr_dataset
from src.engine.trainer import (
    SupervisedSegmentationTrainer,
    build_optimizer_from_config,
    build_validation_comparison_report,
    inspect_dataloader_batch_distribution,
)
from src.federated.data import (
    carve_client_validation_records,
    verify_client_split_integrity,
)
from src.models.losses import build_loss_from_config, summarize_loss_configuration
from src.models.unet import build_unet_from_config
from src.utils.config import (
    load_yaml_config,
    validate_dataset_config,
    validate_model_config,
    validate_project_config,
)
from src.utils.device import get_device_info, resolve_device
from src.utils.logger import get_logger
from src.utils.paths import format_project_path
from src.utils.seed import set_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for supervised baseline training and smoke testing."""
    parser = argparse.ArgumentParser(
        description="Train or smoke-test the supervised ResNet-50 U-Net baseline."
    )
    parser.add_argument(
        "--project-config",
        type=str,
        default="configs/project.yaml",
        help="Path to project configuration YAML.",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="configs/dataset.yaml",
        help="Path to dataset configuration YAML.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model.yaml",
        help="Path to model configuration YAML.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/processed/manifests/ddr_manifest.json",
        help="Path to dataset manifest JSON for training/validation.",
    )
    parser.add_argument(
        "--datasets",
        "--dataset",
        dest="datasets",
        type=str,
        nargs="*",
        default=None,
        help="Optional dataset filter (e.g. '--dataset ddr').",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override square image resolution (H = W).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override global random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override execution device ('cuda', 'cpu', or 'auto').",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained encoder weights.",
    )
    parser.add_argument(
        "--encoder-weights-source",
        type=str,
        default=None,
        help="Override encoder initialization source ('imagenet', 'ssl_checkpoint', 'random').",
    )
    parser.add_argument(
        "--ssl-checkpoint-path",
        "--pretrained-checkpoint",
        dest="ssl_checkpoint_path",
        type=str,
        default=None,
        help="Path to SimCLR/SSL encoder checkpoint when encoder_weights_source='ssl_checkpoint'.",
    )
    parser.add_argument(
        "--baseline-report",
        type=str,
        default=None,
        help="Optional path to supervised baseline report JSON to attach validation comparison.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override checkpoint output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Isolated experiment output directory for checkpoints, effective config, and reports.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Path to initial model checkpoint (.pt) loaded before epoch 1 (mutually exclusive with --resume).",
    )
    parser.add_argument(
        "--create-init-checkpoint",
        type=str,
        default=None,
        help="Create a deterministic initial model checkpoint at this path and exit without training.",
    )
    parser.add_argument(
        "--val-split-mode",
        type=str,
        default="official_and_carved",
        choices=["official_and_carved", "official_only", "carved_only"],
        help="Validation split resolution mode (default: official_and_carved).",
    )
    parser.add_argument(
        "--val-carve-ratio",
        type=float,
        default=0.2,
        help="Validation carve ratio when a dataset has no official validation split (default: 0.2).",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Deterministic seed for validation carving (default: 42).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional path to checkpoint (.pt) to resume from.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a single train/val batch to verify pipeline wiring without full training.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a bounded real-data smoke test on DDR and IDRiD samples and report GPU memory.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Output JSON path for experiment, smoke-test, or dry-run report.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable in-memory dataset caching of FOV-cropped (H, W) arrays.",
    )
    return parser.parse_args(argv)


def _resolve_cli_path(path_like: str | Path) -> Path:
    """Resolve a relative CLI path against ``PROJECT_ROOT`` while preserving absolute paths."""
    p = Path(path_like)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def compute_file_sha256(path: str | Path) -> str:
    """Compute hex SHA-256 digest of a file on disk."""
    file_path = Path(path)
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_state_dict_sha256(state_dict: dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 digest over sorted parameter/buffer tensors."""
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        hasher.update(key.encode("utf-8"))
        val = state_dict[key]
        if isinstance(val, torch.Tensor):
            t_cpu = val.detach().cpu().contiguous()
            hasher.update(str(tuple(t_cpu.shape)).encode("utf-8"))
            hasher.update(str(t_cpu.dtype).encode("utf-8"))
            hasher.update(t_cpu.numpy().tobytes())
        else:
            hasher.update(repr(val).encode("utf-8"))
    return hasher.hexdigest()


def create_initial_model_checkpoint(
    model_cfg: dict[str, Any],
    output_path: str | Path,
    *,
    seed: int = 42,
    deterministic: bool = True,
    strict_pretrained_override: bool | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create and persist a deterministic initial model checkpoint without training."""
    target_path = _resolve_cli_path(output_path)
    if target_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing initialization checkpoint: {target_path}. "
            "Remove it or pass overwrite=True explicitly."
        )

    set_seed(seed=int(seed), deterministic=bool(deterministic))
    cfg_snapshot = copy.deepcopy(model_cfg)
    model = build_unet_from_config(
        cfg_snapshot,
        strict_pretrained_override=strict_pretrained_override,
    )
    state_dict = model.state_dict()
    state_sha256 = compute_state_dict_sha256(state_dict)

    payload: dict[str, Any] = {
        "format": "fedmorphis_initial_checkpoint_v1",
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_summary": model.get_model_summary(),
        "model_config": cfg_snapshot,
        "model_state_dict": state_dict,
        "model_state_sha256": state_sha256,
    }
    target_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target_path)
    file_sha256 = compute_file_sha256(target_path)

    return {
        "checkpoint_path": format_project_path(target_path),
        "absolute_path": str(target_path),
        "file_sha256": file_sha256,
        "model_state_sha256": state_sha256,
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "model_summary": payload["model_summary"],
    }


def load_initial_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Validate and load an initial model checkpoint into ``model`` before epoch 1."""
    ckpt_path = _resolve_cli_path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Initial checkpoint file not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        ckpt_state = payload["model_state_dict"]
        recorded_state_sha = payload.get("model_state_sha256")
    elif isinstance(payload, dict) and payload and all(
        isinstance(v, torch.Tensor) for v in payload.values()
    ):
        ckpt_state = payload
        recorded_state_sha = None
    else:
        raise ValueError(
            f"Invalid initial checkpoint payload in {ckpt_path}: "
            "expected a dict containing 'model_state_dict' or a raw tensor state_dict."
        )

    if not isinstance(ckpt_state, dict) or len(ckpt_state) == 0:
        raise ValueError(f"Empty or malformed model_state_dict in {ckpt_path}.")

    model_state = model.state_dict()
    ckpt_keys = set(ckpt_state.keys())
    model_keys = set(model_state.keys())
    missing_keys = sorted(model_keys - ckpt_keys)
    unexpected_keys = sorted(ckpt_keys - model_keys)
    if missing_keys or unexpected_keys:
        raise ValueError(
            f"Initial checkpoint architecture mismatch for {ckpt_path}: "
            f"missing_keys={missing_keys[:5]} (total {len(missing_keys)}), "
            f"unexpected_keys={unexpected_keys[:5]} (total {len(unexpected_keys)})."
        )

    for key in sorted(model_keys):
        ckpt_tensor = ckpt_state[key]
        model_tensor = model_state[key]
        if not isinstance(ckpt_tensor, torch.Tensor):
            raise ValueError(
                f"Initial checkpoint entry '{key}' is not a torch.Tensor "
                f"(got {type(ckpt_tensor).__name__})."
            )
        if tuple(ckpt_tensor.shape) != tuple(model_tensor.shape):
            raise ValueError(
                f"Initial checkpoint shape mismatch for '{key}': "
                f"checkpoint has {tuple(ckpt_tensor.shape)}, "
                f"model expects {tuple(model_tensor.shape)}."
            )
        if ckpt_tensor.is_floating_point() and not torch.isfinite(ckpt_tensor).all():
            raise ValueError(
                f"Initial checkpoint tensor '{key}' contains non-finite values."
            )

    model.load_state_dict(ckpt_state, strict=True)
    loaded_state_sha = compute_state_dict_sha256(model.state_dict())
    if recorded_state_sha is not None and str(recorded_state_sha) != loaded_state_sha:
        raise ValueError(
            f"Initial checkpoint state SHA-256 mismatch in {ckpt_path}: "
            f"recorded={recorded_state_sha}, loaded={loaded_state_sha}."
        )

    return {
        "init_checkpoint_path": format_project_path(ckpt_path),
        "init_checkpoint_sha256": compute_file_sha256(ckpt_path),
        "initial_model_state_sha256": loaded_state_sha,
    }


def verify_output_directory_safe(
    ckpt_dir: Path,
    report_path: Path | None,
    *,
    is_resume: bool,
    is_dry_run: bool,
) -> None:
    """Prevent silently overwriting an existing or completed experiment unless resuming."""
    if is_resume or is_dry_run:
        return

    conflicts: list[str] = []
    for fname in ("best_checkpoint.pt", "latest_checkpoint.pt"):
        candidate = ckpt_dir / fname
        if candidate.exists():
            conflicts.append(str(candidate))
    if report_path is not None and report_path.exists():
        conflicts.append(str(report_path))

    if conflicts:
        raise FileExistsError(
            "Refusing to overwrite existing experiment artifacts without --resume: "
            + ", ".join(conflicts)
        )


def resolve_supervised_train_val_records(
    records: list[dict[str, Any]],
    allowed_datasets: list[str] | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    *,
    val_split_mode: str = "official_and_carved",
    val_carve_ratio: float = 0.2,
    split_seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Resolve strictly non-test train and validation records per dataset.

    Each dataset present after filtering is resolved independently:
    - If a dataset has official ``valid`` records and ``val_split_mode`` allows
      official validation (``official_and_carved`` or ``official_only``), its
      official ``train`` and ``valid`` records are used.
    - If a dataset has no official ``valid`` records (such as IDRiD), its
      validation split is deterministically carved from its own ``train``
      records via :func:`src.federated.data.carve_client_validation_records`
      with ``seed=split_seed`` and ``val_carve_ratio``.
    - Held-out ``test`` records are never included in train or validation.
    """
    if val_split_mode not in {"official_and_carved", "official_only", "carved_only"}:
        raise ValueError(
            f"Unsupported val_split_mode {val_split_mode!r}; "
            "expected 'official_and_carved', 'official_only', or 'carved_only'."
        )

    filtered = list(records)
    if allowed_datasets:
        allowed_norm = {d.strip().lower() for d in allowed_datasets if d.strip()}
        if allowed_norm:
            filtered = [
                r
                for r in records
                if str(r.get("dataset_name", "")).strip().lower() in allowed_norm
            ]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in filtered:
        ds_name = str(r.get("dataset_name", "UNKNOWN")).strip() or "UNKNOWN"
        grouped.setdefault(ds_name, []).append(dict(r))

    if not grouped:
        raise ValueError("Manifest contains no matching records after dataset filtering.")

    combined_train: list[dict[str, Any]] = []
    combined_val: list[dict[str, Any]] = []
    all_held_out_test: list[dict[str, Any]] = []
    per_dataset_summary: dict[str, Any] = {}
    carved_val_ids_by_dataset: dict[str, list[str]] = {}

    for ds_name in sorted(grouped.keys()):
        ds_records = grouped[ds_name]
        ds_train = [
            dict(r) for r in ds_records if str(r.get("split", "")).strip().lower() == "train"
        ]
        ds_official_val = [
            dict(r)
            for r in ds_records
            if str(r.get("split", "")).strip().lower() in {"valid", "val"}
        ]
        ds_test = [
            dict(r) for r in ds_records if str(r.get("split", "")).strip().lower() == "test"
        ]
        all_held_out_test.extend(ds_test)

        if not ds_train:
            raise ValueError(f"Dataset '{ds_name}' contains no 'train' split records.")

        if ds_official_val and val_split_mode in {"official_and_carved", "official_only"}:
            resolved_train = ds_train
            resolved_val = ds_official_val
            for rec in resolved_val:
                rec.setdefault("val_source", "official_valid")
            val_source = "official_valid"
        else:
            if val_split_mode == "official_only":
                raise ValueError(
                    f"Dataset '{ds_name}' has no official 'valid' records and "
                    "val_split_mode='official_only' forbids carving."
                )
            resolved_train, resolved_val = carve_client_validation_records(
                ds_train,
                val_carve_ratio=val_carve_ratio,
                seed=split_seed,
            )
            val_source = "carved_from_train"
            carved_val_ids_by_dataset[ds_name] = [
                str(r["sample_id"]) for r in resolved_val
            ]

        verify_client_split_integrity(
            client_id=ds_name,
            train_records=resolved_train,
            val_records=resolved_val,
            held_out_test_records=ds_test,
        )

        combined_train.extend(resolved_train)
        combined_val.extend(resolved_val)
        per_dataset_summary[ds_name] = {
            "train_samples": len(resolved_train),
            "validation_samples": len(resolved_val),
            "test_samples_held_out": len(ds_test),
            "val_source": val_source,
        }

    combined_train.sort(
        key=lambda r: (str(r.get("dataset_name", "")), str(r.get("sample_id", "")))
    )
    combined_val.sort(
        key=lambda r: (str(r.get("dataset_name", "")), str(r.get("sample_id", "")))
    )

    if max_train_samples is not None:
        combined_train = combined_train[:max_train_samples]
    if max_val_samples is not None:
        combined_val = combined_val[:max_val_samples]

    verify_client_split_integrity(
        client_id="combined",
        train_records=combined_train,
        val_records=combined_val,
        held_out_test_records=all_held_out_test,
    )

    train_ids_sha = hashlib.sha256(
        json.dumps(
            [f"{r['dataset_name']}:{r['sample_id']}" for r in combined_train]
        ).encode("utf-8")
    ).hexdigest()
    val_ids_sha = hashlib.sha256(
        json.dumps(
            [f"{r['dataset_name']}:{r['sample_id']}" for r in combined_val]
        ).encode("utf-8")
    ).hexdigest()

    split_info: dict[str, Any] = {
        "val_split_mode": val_split_mode,
        "val_carve_ratio": float(val_carve_ratio),
        "split_seed": int(split_seed),
        "per_dataset": per_dataset_summary,
        "carved_validation_sample_ids": carved_val_ids_by_dataset,
        "train_samples": len(combined_train),
        "validation_samples": len(combined_val),
        "test_samples_held_out": len(all_held_out_test),
        "train_sample_ids_sha256": train_ids_sha,
        "val_sample_ids_sha256": val_ids_sha,
    }
    return combined_train, combined_val, split_info


def _select_train_val_records(
    records: list[dict[str, Any]],
    allowed_datasets: list[str] | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    *,
    val_split_mode: str = "official_and_carved",
    val_carve_ratio: float = 0.2,
    split_seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select strictly non-test train and validation records per dataset."""
    train_pool, val_pool, _ = resolve_supervised_train_val_records(
        records=records,
        allowed_datasets=allowed_datasets,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        val_split_mode=val_split_mode,
        val_carve_ratio=val_carve_ratio,
        split_seed=split_seed,
    )
    return train_pool, val_pool


def run_real_data_smoke_test(
    base_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    device: torch.device,
    batch_size: int = 2,
    image_size: int = 512,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Execute a bounded real-data smoke test on DDR and IDRiD samples."""
    ddr_manifest_path = PROJECT_ROOT / "data/processed/manifests/ddr_manifest.json"
    idrid_manifest_path = PROJECT_ROOT / "data/processed/manifests/idrid_manifest.json"

    ddr_manifest = load_manifest_json(ddr_manifest_path)
    idrid_manifest = load_manifest_json(idrid_manifest_path)

    ddr_train, ddr_val = _select_train_val_records(
        ddr_manifest["records"],
        max_train_samples=batch_size,
        max_val_samples=batch_size,
    )
    idrid_train, idrid_val = _select_train_val_records(
        idrid_manifest["records"],
        max_train_samples=batch_size,
        max_val_samples=batch_size,
    )

    # Combine 1 DDR + 1 IDRiD sample (or batch_size total) for real cross-dataset smoke validation
    smoke_train_records = ddr_train[:1] + idrid_train[:1]
    smoke_val_records = ddr_val[:1] + idrid_val[:1]

    train_tf = build_segmentation_transforms(
        split="train",
        dataset_config=data_cfg,
        image_size=(image_size, image_size),
    )
    eval_tf = build_segmentation_transforms(
        split="valid",
        dataset_config=data_cfg,
        image_size=(image_size, image_size),
    )

    train_ds = RetinalLesionSegmentationDataset(
        manifest_or_records=smoke_train_records,
        dataset_config=data_cfg,
        transform=train_tf,
        image_size=(image_size, image_size),
        return_metadata=True,
    )
    val_ds = RetinalLesionSegmentationDataset(
        manifest_or_records=smoke_val_records,
        dataset_config=data_cfg,
        transform=eval_tf,
        image_size=(image_size, image_size),
        return_metadata=True,
    )

    seed = int(base_cfg.get("random_seed", base_cfg.get("seed", 42)))
    train_loader = build_segmentation_dataloader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        seed=seed,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = build_segmentation_dataloader(
        dataset=val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        seed=seed,
        pin_memory=(device.type == "cuda"),
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model = build_unet_from_config(model_cfg)
    criterion = build_loss_from_config(model_cfg)
    optimizer = build_optimizer_from_config(model, model_cfg)

    train_cfg = model_cfg.get("training", {})
    ckpt_dir = PROJECT_ROOT / "checkpoints/segmentation/smoke_test"

    trainer = SupervisedSegmentationTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=ckpt_dir,
        mixed_precision=bool(train_cfg.get("mixed_precision", True)),
        gradient_clip_max_norm=train_cfg.get("gradient_clip_max_norm", 1.0),
        metric_threshold=float(train_cfg.get("metric_threshold", 0.5)),
        monitor_metric=str(train_cfg.get("monitor_metric", "val_dice_macro")),
        monitor_mode=str(train_cfg.get("monitor_mode", "max")),
        seed=seed,
        config={"base": base_cfg, "data": data_cfg, "model": model_cfg},
    )

    fit_result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=1,
        max_train_batches=1,
        max_val_batches=1,
        save_checkpoints=True,
    )

    gpu_memory: dict[str, Any] = {
        "cuda_used": device.type == "cuda",
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "max_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "max_memory_allocated_mb": (
            round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)
            if device.type == "cuda"
            else 0.0
        ),
        "max_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device))
            if device.type == "cuda"
            else 0
        ),
        "max_memory_reserved_mb": (
            round(torch.cuda.max_memory_reserved(device) / (1024**2), 2)
            if device.type == "cuda"
            else 0.0
        ),
    }

    epoch_summary = fit_result["history"][0]
    report: dict[str, Any] = {
        "status": "passed",
        "mode": "smoke_test",
        "device": str(device),
        "amp_enabled": trainer.amp_enabled,
        "batch_size": batch_size,
        "image_size": [image_size, image_size],
        "model": model.get_model_summary(),
        "train_sample_ids": [r["sample_id"] for r in smoke_train_records],
        "val_sample_ids": [r["sample_id"] for r in smoke_val_records],
        "train_loss": epoch_summary["train_loss"],
        "val_loss": epoch_summary["val_loss"],
        "val_dice_macro": epoch_summary["val_dice_macro"],
        "val_iou_macro": epoch_summary["val_iou_macro"],
        "val_per_class": epoch_summary["val"]["metrics"]["per_class"],
        "gpu_memory": gpu_memory,
        "latest_checkpoint": fit_result["latest_checkpoint"],
        "best_checkpoint": fit_result["best_checkpoint"],
    }

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def main(argv: list[str] | None = None) -> int:
    """Entry point for supervised U-Net baseline training or smoke testing."""
    args = parse_args(argv)
    logger = get_logger("fedmorphis.train_baseline")

    base_cfg = load_yaml_config(_resolve_cli_path(args.project_config), validate=False)
    validate_project_config(base_cfg)
    data_cfg = load_yaml_config(_resolve_cli_path(args.dataset_config), validate=False)
    validate_dataset_config(data_cfg)
    model_cfg = load_yaml_config(_resolve_cli_path(args.model_config), validate=False)
    validate_model_config(model_cfg)

    if args.no_pretrained:
        model_cfg["pretrained"] = False
        model_cfg["encoder_weights_source"] = "random"
    if args.encoder_weights_source is not None:
        model_cfg["encoder_weights_source"] = str(args.encoder_weights_source).strip().lower()
    if args.ssl_checkpoint_path is not None:
        model_cfg["ssl_checkpoint_path"] = format_project_path(args.ssl_checkpoint_path)
        if args.encoder_weights_source is None:
            model_cfg["encoder_weights_source"] = "ssl_checkpoint"
    validate_model_config(model_cfg)

    seed = int(
        args.seed
        if args.seed is not None
        else base_cfg.get("random_seed", base_cfg.get("seed", 42))
    )
    deterministic = bool(base_cfg.get("deterministic", True))

    if args.init_checkpoint and args.resume:
        raise ValueError(
            "--init-checkpoint and --resume are mutually exclusive. "
            "Use --init-checkpoint only when starting a fresh experiment from shared initial weights."
        )

    strict_pretrained = bool(model_cfg.get("pretrained", True) and not args.no_pretrained)

    if args.create_init_checkpoint:
        init_meta = create_initial_model_checkpoint(
            model_cfg=model_cfg,
            output_path=args.create_init_checkpoint,
            seed=seed,
            deterministic=deterministic,
            strict_pretrained_override=strict_pretrained,
            overwrite=False,
        )
        print(json.dumps({"status": "created_init_checkpoint", **init_meta}, indent=2))
        return 0

    set_seed(seed=seed, deterministic=deterministic)

    requested_device = args.device or str(base_cfg.get("device", "auto"))
    device_info = get_device_info()
    device = torch.device(str(resolve_device(requested_device)))

    batch_size = int(args.batch_size or data_cfg.get("batch_size", 4))
    cfg_img_size = data_cfg.get("image_size", {"height": 512, "width": 512})
    if isinstance(cfg_img_size, dict):
        default_img_size = int(cfg_img_size.get("height", 512))
    elif isinstance(cfg_img_size, (list, tuple)):
        default_img_size = int(cfg_img_size[0])
    else:
        default_img_size = int(cfg_img_size)
    image_size = int(args.image_size or default_img_size)

    output_dir = _resolve_cli_path(args.output_dir) if args.output_dir else None
    train_cfg = dict(model_cfg.get("training", {}))
    if args.checkpoint_dir is not None:
        ckpt_dir = _resolve_cli_path(args.checkpoint_dir)
        train_cfg["checkpoint_directory"] = format_project_path(ckpt_dir)
    elif output_dir is not None:
        ckpt_dir = output_dir
        train_cfg["checkpoint_directory"] = format_project_path(ckpt_dir)
    else:
        ckpt_dir = _resolve_cli_path(
            str(train_cfg.get("checkpoint_directory", "checkpoints/segmentation"))
        )

    if args.epochs is not None:
        train_cfg["epochs"] = int(args.epochs)
    model_cfg["training"] = train_cfg

    if args.report_path:
        out_report_path: Path | None = _resolve_cli_path(args.report_path)
    elif output_dir is not None:
        out_report_path = output_dir / "experiment_report.json"
    elif args.dry_run:
        out_report_path = None
    else:
        out_report_path = PROJECT_ROOT / "logs/supervised_baseline_ddr_report.json"

    if not args.smoke_test:
        verify_output_directory_safe(
            ckpt_dir=ckpt_dir,
            report_path=out_report_path,
            is_resume=bool(args.resume),
            is_dry_run=bool(args.dry_run),
        )

    logger.info(
        "Initialized baseline runner on device=%s (cuda_available=%s, seed=%d, batch_size=%d, image_size=%d)",
        device,
        device_info.get("cuda_available"),
        seed,
        batch_size,
        image_size,
    )

    if args.smoke_test:
        smoke_batch_size = int(args.batch_size or 2)
        smoke_img_size = int(args.image_size or 512)
        smoke_report_path = (
            _resolve_cli_path(args.report_path)
            if args.report_path
            else PROJECT_ROOT / "logs/unet_baseline_smoke_test.json"
        )
        report = run_real_data_smoke_test(
            base_cfg=base_cfg,
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            device=device,
            batch_size=smoke_batch_size,
            image_size=smoke_img_size,
            report_path=smoke_report_path,
        )
        print(json.dumps(report, indent=2))
        return 0

    manifest_path = _resolve_cli_path(args.manifest)
    manifest = load_manifest_json(manifest_path)
    all_records = manifest["records"]
    train_records, val_records, split_info = resolve_supervised_train_val_records(
        all_records,
        allowed_datasets=args.datasets,
        val_split_mode=args.val_split_mode,
        val_carve_ratio=float(args.val_carve_ratio),
        split_seed=int(args.split_seed),
    )
    held_out_test_count = int(split_info["test_samples_held_out"])
    datasets_included = sorted(
        list(
            {
                str(r.get("dataset_name", "DDR"))
                for r in (train_records + val_records)
            }
        )
    )

    train_tf = build_segmentation_transforms(
        split="train",
        dataset_config=data_cfg,
        image_size=(image_size, image_size),
    )
    eval_tf = build_segmentation_transforms(
        split="valid",
        dataset_config=data_cfg,
        image_size=(image_size, image_size),
    )

    use_cache = not bool(args.no_cache or args.dry_run)
    train_ds = RetinalLesionSegmentationDataset(
        manifest_or_records=train_records,
        dataset_config=data_cfg,
        transform=train_tf,
        image_size=(image_size, image_size),
        return_metadata=True,
        cache_in_memory=use_cache,
    )
    val_ds = RetinalLesionSegmentationDataset(
        manifest_or_records=val_records,
        dataset_config=data_cfg,
        transform=eval_tf,
        image_size=(image_size, image_size),
        return_metadata=True,
        cache_in_memory=use_cache,
    )

    if use_cache:
        t_preload = time.perf_counter()
        logger.info(
            "Pre-loading and FOV-cropping %d train + %d val samples into memory...",
            len(train_ds),
            len(val_ds),
        )
        train_ds.preload_cache(max_workers=8)
        val_ds.preload_cache(max_workers=8)
        logger.info(
            "Pre-loaded %d total samples in %.2fs",
            len(train_ds) + len(val_ds),
            time.perf_counter() - t_preload,
        )

    # When cache_in_memory is enabled on Windows, num_workers=0 keeps the cache in the main process
    num_workers = (
        0
        if (args.dry_run or use_cache)
        else int(data_cfg.get("number_of_workers", data_cfg.get("num_workers", 0)))
    )
    train_loader = build_segmentation_dataloader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=not args.dry_run,
        num_workers=num_workers,
        seed=seed,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = build_segmentation_dataloader(
        dataset=val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
        pin_memory=(device.type == "cuda"),
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    train_batch_dist = inspect_dataloader_batch_distribution(
        len(train_ds), batch_size=batch_size, drop_last=False
    )
    val_batch_dist = inspect_dataloader_batch_distribution(
        len(val_ds), batch_size=batch_size, drop_last=False
    )
    if train_batch_dist["has_singleton_tail_batch"] and bool(
        model_cfg.get("use_batchnorm", True)
    ):
        logger.warning(
            "Training dataset size %d with batch_size=%d produces a singleton tail batch "
            "(tail_batch_size=1) with BatchNorm enabled.",
            len(train_ds),
            batch_size,
        )

    model = build_unet_from_config(
        model_cfg,
        strict_pretrained_override=strict_pretrained,
    )
    init_checkpoint_info: dict[str, Any] = {
        "init_checkpoint_path": None,
        "init_checkpoint_sha256": None,
        "initial_model_state_sha256": compute_state_dict_sha256(model.state_dict()),
    }
    if args.init_checkpoint:
        init_checkpoint_info = load_initial_model_checkpoint(model, args.init_checkpoint)
        logger.info(
            "Loaded initial model weights from %s (sha256=%s)",
            init_checkpoint_info["init_checkpoint_path"],
            init_checkpoint_info["init_checkpoint_sha256"],
        )

    criterion = build_loss_from_config(model_cfg)
    optimizer = build_optimizer_from_config(model, model_cfg)

    trainer = SupervisedSegmentationTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=ckpt_dir,
        mixed_precision=bool(train_cfg.get("mixed_precision", True)),
        gradient_clip_max_norm=train_cfg.get("gradient_clip_max_norm", 1.0),
        metric_threshold=float(train_cfg.get("metric_threshold", 0.5)),
        monitor_metric=str(train_cfg.get("monitor_metric", "val_dice_macro")),
        monitor_mode=str(train_cfg.get("monitor_mode", "max")),
        seed=seed,
        config={"base": base_cfg, "data": data_cfg, "model": model_cfg},
    )

    if args.resume:
        trainer.load_checkpoint(_resolve_cli_path(args.resume))

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        effective_cfg_payload = {
            "project": base_cfg,
            "dataset": data_cfg,
            "model": model_cfg,
            "runtime": {
                "seed": seed,
                "deterministic": deterministic,
                "batch_size": batch_size,
                "image_size": [image_size, image_size],
                "manifest": format_project_path(args.manifest),
                "datasets": args.datasets,
                "split_resolution": split_info,
                "init_checkpoint": init_checkpoint_info,
            },
        }
        (output_dir / "effective_config.yaml").write_text(
            yaml.safe_dump(effective_cfg_payload, sort_keys=False),
            encoding="utf-8",
        )

    num_epochs = 1 if args.dry_run else int(args.epochs or train_cfg.get("epochs", 50))
    max_batches = 1 if args.dry_run else None

    t0 = time.perf_counter()
    result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        max_train_batches=max_batches,
        max_val_batches=max_batches,
        save_checkpoints=not args.dry_run,
    )
    session_duration_sec = round(time.perf_counter() - t0, 2)
    cumulative_epoch_sec = round(
        sum(
            float(h["train"]["elapsed_sec"]) + float(h["val"]["elapsed_sec"])
            for h in result["history"]
        ),
        2,
    )
    total_duration_sec = max(session_duration_sec, cumulative_epoch_sec)

    # Evaluate validation AUPR/AP using the selected best checkpoint
    best_ckpt_rel = result.get("best_checkpoint")
    if best_ckpt_rel is not None:
        best_ckpt_abs = _resolve_cli_path(best_ckpt_rel)
        if best_ckpt_abs.is_file():
            trainer.load_checkpoint(
                best_ckpt_abs,
                load_optimizer=False,
                load_scheduler=False,
                strict_model=True,
                restore_rng=False,
            )

    model.eval()
    pre_eval_state_sha = compute_state_dict_sha256(model.state_dict())
    validation_aupr_report = evaluate_aupr_dataset(
        model=model,
        dataset=val_loader,
        device=device,
        use_sliding_window=False,
        expected_split="valid",
        batch_size=batch_size,
        metric_threshold=trainer.metric_threshold,
    )
    post_eval_state_sha = compute_state_dict_sha256(model.state_dict())
    if pre_eval_state_sha != post_eval_state_sha:
        raise RuntimeError(
            "Validation AUPR evaluation unexpectedly mutated model weights."
        )
    if model.training:
        raise RuntimeError(
            "Model remained in training mode after validation AUPR evaluation."
        )

    if output_dir is not None:
        (output_dir / "validation_aupr_report.json").write_text(
            json.dumps(validation_aupr_report, indent=2),
            encoding="utf-8",
        )

    gpu_memory: dict[str, Any] = {
        "cuda_used": device.type == "cuda",
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "max_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "max_memory_allocated_mb": (
            round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)
            if device.type == "cuda"
            else 0.0
        ),
        "max_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device))
            if device.type == "cuda"
            else 0
        ),
        "max_memory_reserved_mb": (
            round(torch.cuda.max_memory_reserved(device) / (1024**2), 2)
            if device.type == "cuda"
            else 0.0
        ),
    }

    best_epoch_idx = int(result["best_epoch"])
    best_epoch_record = next(
        (h for h in result["history"] if int(h["epoch"]) == best_epoch_idx),
        result["history"][-1],
    )
    final_epoch_record = result["history"][-1]

    compact_history: list[dict[str, Any]] = []
    for h in result["history"]:
        compact_history.append(
            {
                "epoch": h["epoch"],
                "learning_rate": h["learning_rate"],
                "primary_loss_component": h.get("primary_loss_component"),
                "secondary_loss_component": h.get("secondary_loss_component"),
                "overlap_component_name": h.get("overlap_component_name"),
                "train_loss": h["train_loss"],
                "train_bce_loss": h.get("train_bce_loss"),
                "train_overlap_loss": h.get("train_overlap_loss"),
                "train_per_channel_bce": h.get("train_per_channel_bce"),
                "train_per_channel_overlap": h.get("train_per_channel_overlap"),
                "train_dice_macro": h["train_dice_macro"],
                "train_dice_macro_present_only": h.get("train_dice_macro_present_only"),
                "val_loss": h["val_loss"],
                "val_bce_loss": h.get("val_bce_loss"),
                "val_overlap_loss": h.get("val_overlap_loss"),
                "val_per_channel_bce": h.get("val_per_channel_bce"),
                "val_per_channel_overlap": h.get("val_per_channel_overlap"),
                "val_dice_macro": h["val_dice_macro"],
                "val_dice_macro_present_only": h.get("val_dice_macro_present_only"),
                "val_iou_macro": h["val_iou_macro"],
                "val_iou_macro_present_only": h.get("val_iou_macro_present_only"),
                "val_dice_per_class": {
                    ch: h[f"val_dice_{ch}"] for ch in ["MA", "HE", "EX", "SE"]
                },
                "val_iou_per_class": {
                    ch: h[f"val_iou_{ch}"] for ch in ["MA", "HE", "EX", "SE"]
                },
                "is_best": h["is_best"],
                "train_elapsed_sec": h["train"]["elapsed_sec"],
                "val_elapsed_sec": h["val"]["elapsed_sec"],
            }
        )

    experiment_report: dict[str, Any] = {
        "status": "completed",
        "dry_run": bool(args.dry_run),
        "seed": seed,
        "deterministic": deterministic,
        "device": str(device),
        "amp_enabled": trainer.amp_enabled,
        "output_directory": format_project_path(output_dir) if output_dir else None,
        "initialization": init_checkpoint_info,
        "dataset": {
            "manifest_path": format_project_path(args.manifest),
            "datasets_included": datasets_included,
            "train_samples": len(train_records),
            "validation_samples": len(val_records),
            "test_samples_held_out": held_out_test_count,
            "test_split_evaluated": False,
            "image_size": [image_size, image_size],
            "batch_size": batch_size,
            "cache_in_memory": use_cache,
            "split_resolution": split_info,
            "train_batch_distribution": train_batch_dist,
            "val_batch_distribution": val_batch_dist,
        },
        "model": model.get_model_summary(),
        "loss_config": summarize_loss_configuration(model_cfg),
        "training_config": train_cfg,
        "epochs_completed": int(result["epochs_completed"]),
        "global_step": int(result["global_step"]),
        "best_epoch": best_epoch_idx,
        "best_metric_name": result["best_metric_name"],
        "best_metric_value": float(result["best_metric_value"]),
        "best_validation_metrics": {
            "epoch": best_epoch_idx,
            "val_loss": best_epoch_record["val_loss"],
            "val_bce_loss": best_epoch_record.get("val_bce_loss"),
            "val_overlap_loss": best_epoch_record.get("val_overlap_loss"),
            "overlap_component_name": best_epoch_record.get("overlap_component_name"),
            "val_per_channel_bce": best_epoch_record.get("val_per_channel_bce"),
            "val_per_channel_overlap": best_epoch_record.get("val_per_channel_overlap"),
            "val_dice_macro": best_epoch_record["val_dice_macro"],
            "val_dice_macro_present_only": best_epoch_record.get(
                "val_dice_macro_present_only"
            ),
            "val_iou_macro": best_epoch_record["val_iou_macro"],
            "val_iou_macro_present_only": best_epoch_record.get(
                "val_iou_macro_present_only"
            ),
            "present_classes": best_epoch_record["val"]["metrics"]["present_classes"],
            "absent_classes": best_epoch_record["val"]["metrics"]["absent_classes"],
            "per_class": best_epoch_record["val"]["metrics"]["per_class"],
            "aupr": validation_aupr_report,
        },
        "validation_aupr": validation_aupr_report,
        "final_epoch_metrics": {
            "epoch": final_epoch_record["epoch"],
            "train_loss": final_epoch_record["train_loss"],
            "train_bce_loss": final_epoch_record.get("train_bce_loss"),
            "train_overlap_loss": final_epoch_record.get("train_overlap_loss"),
            "overlap_component_name": final_epoch_record.get("overlap_component_name"),
            "val_loss": final_epoch_record["val_loss"],
            "val_bce_loss": final_epoch_record.get("val_bce_loss"),
            "val_overlap_loss": final_epoch_record.get("val_overlap_loss"),
            "val_dice_macro": final_epoch_record["val_dice_macro"],
            "val_dice_macro_present_only": final_epoch_record.get(
                "val_dice_macro_present_only"
            ),
            "val_iou_macro": final_epoch_record["val_iou_macro"],
        },
        "training_duration_sec": total_duration_sec,
        "training_duration_minutes": round(total_duration_sec / 60.0, 2),
        "gpu_memory": gpu_memory,
        "checkpoints": {
            "latest_checkpoint": format_project_path(result["latest_checkpoint"]),
            "best_checkpoint": format_project_path(result["best_checkpoint"]),
        },
        "warnings_or_deviations": [],
        "epoch_history": compact_history,
    }

    if args.baseline_report:
        experiment_report["baseline_validation_comparison"] = (
            build_validation_comparison_report(
                candidate_report=experiment_report,
                baseline_report_or_path=_resolve_cli_path(args.baseline_report),
            )
        )

    if out_report_path is not None:
        out_report_path.parent.mkdir(parents=True, exist_ok=True)
        out_report_path.write_text(
            json.dumps(experiment_report, indent=2), encoding="utf-8"
        )
        logger.info("Saved supervised baseline experiment report to %s", out_report_path)

    print(
        json.dumps(
            {
                "status": "completed",
                "dry_run": bool(args.dry_run),
                "pretrained_weights_loaded": model.pretrained_weights_loaded,
                "weights_status": model.weights_status,
                "epochs_completed": result["epochs_completed"],
                "best_metric_name": result["best_metric_name"],
                "best_metric_value": result["best_metric_value"],
                "best_epoch": result["best_epoch"],
                "validation_aupr_macro": validation_aupr_report.get("aupr_macro"),
                "training_duration_sec": total_duration_sec,
                "gpu_memory": gpu_memory,
                "report_path": format_project_path(out_report_path) if out_report_path else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
