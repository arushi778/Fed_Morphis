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
import json
import sys
import time
from pathlib import Path
from typing import Any

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
from src.engine.trainer import (
    SupervisedSegmentationTrainer,
    build_optimizer_from_config,
    build_validation_comparison_report,
)
from src.models.losses import build_loss_from_config
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


def _select_train_val_records(
    records: list[dict[str, Any]],
    allowed_datasets: list[str] | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select strictly non-test train and validation records from a manifest.

    If the manifest has an explicit ``valid`` split (such as DDR), uses ``train``
    and ``valid``. If the manifest only has ``train`` and ``test`` (such as IDRiD),
    deterministically splits ``train`` into train/validation subsets without
    ever touching ``test``.
    """
    filtered = records
    if allowed_datasets:
        allowed_norm = {d.strip().lower() for d in allowed_datasets if d.strip()}
        if allowed_norm:
            filtered = [
                r
                for r in records
                if str(r.get("dataset_name", "")).strip().lower() in allowed_norm
            ]

    train_pool = [r for r in filtered if str(r.get("split")).lower() == "train"]
    val_pool = [r for r in filtered if str(r.get("split")).lower() == "valid"]

    if not train_pool:
        raise ValueError("Manifest contains no 'train' split records.")

    if not val_pool:
        # Deterministically carve validation records from train_pool (never test)
        val_count = max(2, len(train_pool) // 5)
        val_pool = [dict(r, split="valid") for r in train_pool[-val_count:]]
        train_pool = train_pool[:-val_count]

    if max_train_samples is not None:
        train_pool = train_pool[:max_train_samples]
    if max_val_samples is not None:
        val_pool = val_pool[:max_val_samples]

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

    base_cfg = load_yaml_config(PROJECT_ROOT / args.project_config, validate=False)
    validate_project_config(base_cfg)
    data_cfg = load_yaml_config(PROJECT_ROOT / args.dataset_config, validate=False)
    validate_dataset_config(data_cfg)
    model_cfg = load_yaml_config(PROJECT_ROOT / args.model_config, validate=False)
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
            PROJECT_ROOT / args.report_path
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

    manifest_path = PROJECT_ROOT / args.manifest
    manifest = load_manifest_json(manifest_path)
    all_records = manifest["records"]
    train_records, val_records = _select_train_val_records(
        all_records,
        allowed_datasets=args.datasets,
    )
    held_out_test_count = sum(
        1 for r in all_records if str(r.get("split")).strip().lower() == "test"
    )
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

    strict_pretrained = bool(model_cfg.get("pretrained", True) and not args.no_pretrained)
    model = build_unet_from_config(
        model_cfg,
        strict_pretrained_override=strict_pretrained,
    )
    criterion = build_loss_from_config(model_cfg)
    optimizer = build_optimizer_from_config(model, model_cfg)

    train_cfg = dict(model_cfg.get("training", {}))
    if args.checkpoint_dir is not None:
        train_cfg["checkpoint_directory"] = format_project_path(args.checkpoint_dir)
    if args.epochs is not None:
        train_cfg["epochs"] = int(args.epochs)
    model_cfg["training"] = train_cfg
    ckpt_dir = Path(
        args.checkpoint_dir
        or PROJECT_ROOT / str(train_cfg.get("checkpoint_directory", "checkpoints/segmentation"))
    )

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
        trainer.load_checkpoint(args.resume)

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
                "train_loss": h["train_loss"],
                "train_dice_macro": h["train_dice_macro"],
                "train_dice_macro_present_only": h.get("train_dice_macro_present_only"),
                "val_loss": h["val_loss"],
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
        },
        "model": model.get_model_summary(),
        "loss_config": model_cfg.get("loss", {}),
        "training_config": train_cfg,
        "epochs_completed": int(result["epochs_completed"]),
        "global_step": int(result["global_step"]),
        "best_epoch": best_epoch_idx,
        "best_metric_name": result["best_metric_name"],
        "best_metric_value": float(result["best_metric_value"]),
        "best_validation_metrics": {
            "epoch": best_epoch_idx,
            "val_loss": best_epoch_record["val_loss"],
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
        },
        "final_epoch_metrics": {
            "epoch": final_epoch_record["epoch"],
            "train_loss": final_epoch_record["train_loss"],
            "val_loss": final_epoch_record["val_loss"],
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
                baseline_report_or_path=PROJECT_ROOT / args.baseline_report,
            )
        )

    out_report_path = (
        PROJECT_ROOT / args.report_path
        if args.report_path
        else (
            None
            if args.dry_run
            else PROJECT_ROOT / "logs/supervised_baseline_ddr_report.json"
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
