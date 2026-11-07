"""SimCLR self-supervised contrastive pretraining CLI for retinal fundus images.

Supports:
- Full SimCLR pretraining on approved unlabeled ``train``-split retinal fundus
  images from ``ddr_manifest.json`` (383 train images), ``idrid_manifest.json``
  (54 train images), or ``combined_manifest.json`` (437 train images)
- ``--dry-run`` mode (single batch verification step)
- ``--smoke-test`` mode (bounded GPU smoke test verifying two-view generation,
  forward pass, NT-Xent loss, backward pass, optimizer step, checkpoint saving,
  and CUDA memory usage)
- ``--resume`` from a saved SSL checkpoint
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
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import (
    SimCLRUnlabeledDataset,
    build_simclr_dataloader,
    select_ssl_training_records,
    simclr_collate_fn,
)
from src.data.manifests import load_manifest_json
from src.data.transforms import build_simclr_transforms
from src.engine.simclr_trainer import (
    SimCLRTrainer,
    build_ssl_optimizer_from_config,
    build_ssl_scheduler_from_config,
)
from src.models.losses import build_nt_xent_loss_from_config
from src.models.simclr import build_simclr_from_config
from src.utils.config import (
    load_yaml_config,
    validate_dataset_config,
    validate_project_config,
    validate_ssl_config,
)
from src.utils.device import get_device_info, resolve_device
from src.utils.logger import get_logger
from src.utils.paths import format_project_path
from src.utils.seed import set_seed


class _SyntheticTwoViewDataset(Dataset[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]):
    """Synthetic two-view dataset used for fast bounded GPU/CPU smoke verification."""

    def __init__(
        self,
        num_samples: int = 4,
        image_size: int = 128,
        seed: int = 42,
    ) -> None:
        self.num_samples = int(num_samples)
        self.image_size = int(image_size)
        self.split = "train"
        gen = torch.Generator().manual_seed(int(seed))
        self.views1 = torch.randn(
            num_samples, 3, image_size, image_size, generator=gen
        )
        self.views2 = torch.randn(
            num_samples, 3, image_size, image_size, generator=gen
        )
        self.records = [
            {
                "dataset_name": "SYNTHETIC_SSL",
                "sample_id": f"SYN_SSL_TRAIN_{i:02d}",
                "split": "train",
            }
            for i in range(num_samples)
        ]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        return (
            self.views1[index],
            self.views2[index],
            dict(self.records[index]),
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for SimCLR self-supervised pretraining."""
    parser = argparse.ArgumentParser(
        description="Pretrain or smoke-test the SimCLR ResNet-50 encoder on retinal fundus images."
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
        "--ssl-config",
        type=str,
        default="configs/ssl.yaml",
        help="Path to SimCLR SSL configuration YAML.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/processed/manifests/ddr_manifest.json",
        help="Path to dataset manifest JSON (only 'train' split images are used).",
    )
    parser.add_argument(
        "--datasets",
        "--dataset",
        dest="datasets",
        type=str,
        nargs="*",
        default=None,
        help="Optional dataset filter (e.g. '--dataset ddr' or '--datasets DDR IDRiD').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override SimCLR pretraining batch size (must be >= 2).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of pretraining epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override optimizer learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Override optimizer weight decay.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        help="Override optimizer ('adamw', 'adam', or 'sgd').",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        help="Override LR scheduler ('cosine_annealing', 'step', or 'none').",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=None,
        help="Override warmup epochs for cosine annealing scheduler.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override NT-Xent contrastive temperature.",
    )
    parser.add_argument(
        "--projection-dim",
        type=int,
        default=None,
        help="Override SimCLR projection head output dimension.",
    )
    parser.add_argument(
        "--projection-hidden-dim",
        type=int,
        default=None,
        help="Override SimCLR projection head hidden dimension.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override square view resolution (H = W).",
    )
    parser.add_argument(
        "--crop-scale",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN_SCALE", "MAX_SCALE"),
        help="Override RandomResizedCrop scale bounds (e.g. '--crop-scale 0.5 1.0').",
    )
    parser.add_argument(
        "--color-jitter-prob",
        type=float,
        default=None,
        help="Override color jitter probability in [0.0, 1.0].",
    )
    parser.add_argument(
        "--gaussian-blur-prob",
        type=float,
        default=None,
        help="Override Gaussian blur probability in [0.0, 1.0].",
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
        help="Disable pretrained encoder initialization (start from random weights).",
    )
    parser.add_argument(
        "--encoder-weights-source",
        type=str,
        default=None,
        help="Override encoder initialization source ('imagenet', 'supervised_checkpoint', 'ssl_checkpoint', 'random').",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path when initializing encoder from a checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override SSL checkpoint output directory.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional path to SSL checkpoint (.pt) to resume from.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a single batch on the configured manifest to verify pipeline wiring.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a bounded GPU/CPU smoke test with small batch and limited image size.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run a full-resolution (512x512) real-data preflight across progressive batch sizes.",
    )
    parser.add_argument(
        "--preflight-batch-sizes",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16],
        help="Candidate batch sizes to test in --preflight mode (default: 2 4 8 16).",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable in-memory caching of FOV-cropped RGB images.",
    )
    return parser.parse_args(argv)


def _apply_cli_overrides(
    ssl_cfg: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Apply command-line overrides to the loaded ``ssl.yaml`` dictionary and re-validate."""
    cfg = dict(ssl_cfg)
    aug = dict(cfg.get("augmentations", {}))

    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)
    if args.learning_rate is not None:
        cfg["learning_rate"] = float(args.learning_rate)
    if args.weight_decay is not None:
        cfg["weight_decay"] = float(args.weight_decay)
    if args.optimizer is not None:
        cfg["optimizer"] = str(args.optimizer)
    if args.scheduler is not None:
        cfg["lr_scheduler"] = str(args.scheduler)
    if args.warmup_epochs is not None:
        cfg["warmup_epochs"] = int(args.warmup_epochs)
    if args.temperature is not None:
        cfg["temperature"] = float(args.temperature)
    if args.projection_dim is not None:
        cfg["projection_dimension"] = int(args.projection_dim)
    if args.projection_hidden_dim is not None:
        cfg["projection_hidden_dimension"] = int(args.projection_hidden_dim)
    if args.seed is not None:
        cfg["random_seed"] = int(args.seed)
    if args.no_pretrained:
        cfg["pretrained"] = False
        cfg["encoder_weights_source"] = "random"
    if args.encoder_weights_source is not None:
        cfg["encoder_weights_source"] = str(args.encoder_weights_source)
    if args.pretrained_checkpoint is not None:
        cfg["pretrained_checkpoint_path"] = format_project_path(args.pretrained_checkpoint)
    if args.checkpoint_dir is not None:
        cfg["checkpoint_directory"] = format_project_path(args.checkpoint_dir)

    if args.image_size is not None:
        aug["image_size"] = int(args.image_size)
    if args.crop_scale is not None:
        aug["random_resized_crop_scale"] = [
            float(args.crop_scale[0]),
            float(args.crop_scale[1]),
        ]
    if args.color_jitter_prob is not None:
        aug["color_jitter_prob"] = float(args.color_jitter_prob)
    if args.gaussian_blur_prob is not None:
        aug["gaussian_blur_prob"] = float(args.gaussian_blur_prob)

    cfg["augmentations"] = aug
    validate_ssl_config(cfg)
    return cfg


def run_simclr_smoke_test(
    base_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    ssl_cfg: dict[str, Any],
    device: torch.device,
    batch_size: int = 4,
    image_size: int = 128,
    checkpoint_dir: Path | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Run a bounded GPU/CPU SimCLR smoke test verifying forward, loss, backward, optimizer step, and checkpointing."""
    seed = int(ssl_cfg.get("random_seed", base_cfg.get("random_seed", 42)))
    set_seed(seed=seed, deterministic=bool(base_cfg.get("deterministic", True)))

    smoke_ds = _SyntheticTwoViewDataset(
        num_samples=batch_size,
        image_size=image_size,
        seed=seed,
    )
    smoke_loader = DataLoader(
        smoke_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=simclr_collate_fn,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    strict_pretrained = bool(ssl_cfg.get("pretrained", True))
    model = build_simclr_from_config(
        ssl_cfg,
        strict_pretrained_override=strict_pretrained,
    )
    criterion = build_nt_xent_loss_from_config(ssl_cfg)
    optimizer = build_ssl_optimizer_from_config(model, ssl_cfg)
    scheduler = build_ssl_scheduler_from_config(optimizer, ssl_cfg, total_epochs=1)

    ckpt_dir = (
        checkpoint_dir
        if checkpoint_dir is not None
        else PROJECT_ROOT / "checkpoints/ssl/smoke_test"
    )
    trainer = SimCLRTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=ckpt_dir,
        mixed_precision=bool(ssl_cfg.get("mixed_precision", True)),
        gradient_clip_max_norm=ssl_cfg.get("gradient_clip_max_norm", 1.0),
        seed=seed,
        config={"base": base_cfg, "data": data_cfg, "ssl": ssl_cfg},
    )

    fit_result = trainer.fit(
        train_loader=smoke_loader,
        num_epochs=1,
        max_train_batches=1,
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

    epoch_rec = fit_result["history"][0]
    report: dict[str, Any] = {
        "status": "passed",
        "mode": "simclr_smoke_test",
        "device": str(device),
        "amp_enabled": trainer.amp_enabled,
        "batch_size": batch_size,
        "image_size": [image_size, image_size],
        "model": model.get_model_summary(),
        "train_loss": epoch_rec["train_loss"],
        "positive_similarity": epoch_rec["positive_similarity"],
        "negative_similarity": epoch_rec["negative_similarity"],
        "contrastive_accuracy": epoch_rec["contrastive_accuracy"],
        "grad_norm_last": epoch_rec["grad_norm_last"],
        "elapsed_sec": epoch_rec["elapsed_sec"],
        "gpu_memory": gpu_memory,
        "latest_checkpoint": fit_result["latest_checkpoint"],
        "best_checkpoint": fit_result["best_checkpoint"],
    }

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def run_simclr_real_data_preflight(
    base_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    ssl_cfg: dict[str, Any],
    device: torch.device,
    manifest_path: Path,
    allowed_datasets: list[str] | None = None,
    candidate_batch_sizes: list[int] | None = None,
    image_size: int = 512,
    checkpoint_dir: Path | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Execute a full-resolution (512x512) real-data SimCLR preflight on approved train-split images.

    Starts from conservative batch sizes (e.g. ``[2, 4, 8, 16]``) and for each batch size:
    1. Loads real approved ``train``-split retinal fundus images (never ``valid`` or ``test``,
       and never loading lesion masks).
    2. Runs two-view augmentation at ``(B, 3, 512, 512)``, forward pass, NT-Xent loss,
       backward pass, optimizer step, and checkpoint save.
    3. Records peak allocated and reserved GPU memory.
    4. Verifies all model parameters and buffers remain finite.
    5. Verifies the saved checkpoint reloads cleanly into a fresh :class:`SimCLRTrainer`
       and :class:`SimCLRModel` with bit-for-bit identical weights.
    """
    seed = int(ssl_cfg.get("random_seed", base_cfg.get("random_seed", 42)))
    deterministic = bool(base_cfg.get("deterministic", True))
    batch_sizes = candidate_batch_sizes or [2, 4, 8, 16]

    manifest = load_manifest_json(manifest_path)
    all_records = manifest.get("records", [])
    approved_train_records = select_ssl_training_records(
        manifest_or_records=all_records,
        allowed_datasets=allowed_datasets,
        filter_train_split=True,
    )

    max_needed = max(batch_sizes)
    preflight_pool = approved_train_records[:max_needed]
    if len(preflight_pool) < min(batch_sizes):
        raise ValueError(
            f"Not enough approved train records ({len(preflight_pool)}) for minimum "
            f"preflight batch size ({min(batch_sizes)})."
        )

    simclr_tf = build_simclr_transforms(
        ssl_config=ssl_cfg,
        dataset_config=data_cfg,
        image_size=(image_size, image_size),
    )

    # Pre-load the bounded pool of real train images once (FOV-cropped, no masks)
    pool_ds = SimCLRUnlabeledDataset(
        manifest_or_records=preflight_pool,
        allowed_datasets=allowed_datasets,
        transform=simclr_tf,
        ssl_config=ssl_cfg,
        dataset_config=data_cfg,
        image_size=(image_size, image_size),
        return_metadata=True,
        filter_train_split=False,
        cache_in_memory=True,
    )
    pool_ds.preload_cache(max_workers=4)

    ckpt_base_dir = (
        checkpoint_dir
        if checkpoint_dir is not None
        else PROJECT_ROOT / "checkpoints/ssl/preflight"
    )

    batch_trials: list[dict[str, Any]] = []
    largest_verified_batch_size: int | None = None

    for bs in batch_sizes:
        if bs > len(preflight_pool):
            break

        set_seed(seed=seed, deterministic=deterministic)
        subset_records = preflight_pool[:bs]
        trial_ds = SimCLRUnlabeledDataset(
            manifest_or_records=subset_records,
            allowed_datasets=allowed_datasets,
            transform=simclr_tf,
            ssl_config=ssl_cfg,
            dataset_config=data_cfg,
            image_size=(image_size, image_size),
            return_metadata=True,
            filter_train_split=False,
            cache_in_memory=True,
        )
        # Share pre-loaded RGB arrays so disk I/O is not repeated per trial
        for idx in range(bs):
            trial_ds._image_cache[idx] = pool_ds._image_cache[idx]

        trial_loader = build_simclr_dataloader(
            dataset=trial_ds,
            ssl_config=ssl_cfg,
            dataset_config=data_cfg,
            batch_size=bs,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
            seed=seed,
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        trial_ckpt_dir = ckpt_base_dir / f"bs_{bs}"
        try:
            strict_pretrained = bool(ssl_cfg.get("pretrained", True))
            model = build_simclr_from_config(
                ssl_cfg,
                strict_pretrained_override=strict_pretrained,
            )
            criterion = build_nt_xent_loss_from_config(ssl_cfg)
            optimizer = build_ssl_optimizer_from_config(model, ssl_cfg)
            scheduler = build_ssl_scheduler_from_config(
                optimizer, ssl_cfg, total_epochs=1
            )

            trainer = SimCLRTrainer(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scheduler=scheduler,
                checkpoint_dir=trial_ckpt_dir,
                mixed_precision=bool(ssl_cfg.get("mixed_precision", True)),
                gradient_clip_max_norm=ssl_cfg.get("gradient_clip_max_norm", 1.0),
                seed=seed,
                config={"base": base_cfg, "data": data_cfg, "ssl": ssl_cfg},
            )

            fit_res = trainer.fit(
                train_loader=trial_loader,
                num_epochs=1,
                max_train_batches=1,
                save_checkpoints=True,
            )

            # Verify all model weights and buffers remain finite after backward + optimizer step
            params_finite = all(
                bool(torch.isfinite(p).all().item()) for p in model.parameters()
            )
            buffers_finite = all(
                bool(torch.isfinite(b).all().item())
                for b in model.buffers()
                if b.is_floating_point()
            )
            weights_finite = bool(params_finite and buffers_finite)
            if not weights_finite:
                raise RuntimeError(
                    f"Non-finite model parameters or buffers detected after step at batch_size={bs}."
                )

            # Verify saved checkpoint reloads cleanly and matches model state_dict bit-for-bit
            saved_latest = Path(str(fit_res["latest_checkpoint"]))
            saved_best = Path(str(fit_res["best_checkpoint"]))
            if not (saved_latest.is_file() and saved_best.is_file()):
                raise FileNotFoundError(
                    f"Expected checkpoint files not created in {trial_ckpt_dir}."
                )

            reload_model = build_simclr_from_config(
                ssl_cfg,
                pretrained_override=False,
                strict_pretrained_override=False,
            )
            reload_opt = build_ssl_optimizer_from_config(reload_model, ssl_cfg)
            reload_sched = build_ssl_scheduler_from_config(
                reload_opt, ssl_cfg, total_epochs=1
            )
            reload_trainer = SimCLRTrainer(
                model=reload_model,
                criterion=criterion,
                optimizer=reload_opt,
                device=device,
                scheduler=reload_sched,
                checkpoint_dir=trial_ckpt_dir,
                mixed_precision=bool(ssl_cfg.get("mixed_precision", True)),
                gradient_clip_max_norm=ssl_cfg.get("gradient_clip_max_norm", 1.0),
                seed=seed,
                config={"base": base_cfg, "data": data_cfg, "ssl": ssl_cfg},
            )
            loaded_payload = reload_trainer.load_checkpoint(saved_best)

            state_matches = all(
                torch.equal(v1, v2)
                for (_, v1), (_, v2) in zip(
                    model.state_dict().items(), reload_model.state_dict().items()
                )
            )
            encoder_export_valid = (
                "encoder_state_dict" in loaded_payload
                and len(loaded_payload["encoder_state_dict"]) > 0
            )
            checkpoint_reloaded = bool(state_matches and encoder_export_valid)
            if not checkpoint_reloaded:
                raise RuntimeError(
                    f"Reloaded checkpoint verification failed for batch_size={bs}."
                )

            alloc_bytes = (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            )
            res_bytes = (
                int(torch.cuda.max_memory_reserved(device))
                if device.type == "cuda"
                else 0
            )
            alloc_mb = round(alloc_bytes / (1024**2), 2)
            res_mb = round(res_bytes / (1024**2), 2)

            ep = fit_res["history"][0]
            batch_trials.append(
                {
                    "batch_size": bs,
                    "two_view_forward_tensors": [2 * bs, 3, image_size, image_size],
                    "status": "passed",
                    "sample_ids": [r["sample_id"] for r in subset_records],
                    "splits_verified": list({r["split"] for r in subset_records}),
                    "masks_loaded": False,
                    "train_loss": ep["train_loss"],
                    "positive_similarity": ep["positive_similarity"],
                    "negative_similarity": ep["negative_similarity"],
                    "contrastive_accuracy": ep["contrastive_accuracy"],
                    "grad_norm_last": ep["grad_norm_last"],
                    "elapsed_sec": ep["elapsed_sec"],
                    "weights_finite": weights_finite,
                    "checkpoint_reloaded": checkpoint_reloaded,
                    "latest_checkpoint": format_project_path(saved_latest),
                    "best_checkpoint": format_project_path(saved_best),
                    "max_memory_allocated_bytes": alloc_bytes,
                    "max_memory_allocated_mb": alloc_mb,
                    "max_memory_reserved_bytes": res_bytes,
                    "max_memory_reserved_mb": res_mb,
                }
            )
            largest_verified_batch_size = bs

            # Free trial objects before next batch size test
            del trainer, reload_trainer, model, reload_model, optimizer, reload_opt
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except RuntimeError as exc:
            is_oom = "out of memory" in str(exc).lower()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            batch_trials.append(
                {
                    "batch_size": bs,
                    "status": "oom" if is_oom else "failed",
                    "error": str(exc),
                }
            )
            if is_oom:
                break
            raise

    report: dict[str, Any] = {
        "status": "passed" if largest_verified_batch_size is not None else "failed",
        "mode": "simclr_full_resolution_preflight",
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "amp_enabled": bool(
            ssl_cfg.get("mixed_precision", True) and device.type == "cuda"
        ),
        "manifest_path": format_project_path(manifest_path),
        "approved_train_pool_count": len(approved_train_records),
        "validation_or_test_loaded": False,
        "masks_loaded": False,
        "image_size": [image_size, image_size],
        "candidate_batch_sizes": batch_sizes,
        "largest_verified_batch_size": largest_verified_batch_size,
        "trials": batch_trials,
    }

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def main(argv: list[str] | None = None) -> int:
    """Entry point for SimCLR self-supervised pretraining or smoke verification."""
    args = parse_args(argv)
    logger = get_logger("fedmorphis.pretrain_simclr")

    base_cfg = load_yaml_config(PROJECT_ROOT / args.project_config, validate=False)
    validate_project_config(base_cfg)
    data_cfg = load_yaml_config(PROJECT_ROOT / args.dataset_config, validate=False)
    validate_dataset_config(data_cfg)
    raw_ssl_cfg = load_yaml_config(PROJECT_ROOT / args.ssl_config, validate=False)
    ssl_cfg = _apply_cli_overrides(raw_ssl_cfg, args)

    seed = int(ssl_cfg.get("random_seed", base_cfg.get("random_seed", 42)))
    deterministic = bool(base_cfg.get("deterministic", True))
    set_seed(seed=seed, deterministic=deterministic)

    requested_device = args.device or str(base_cfg.get("device", "auto"))
    device_info = get_device_info()
    device = torch.device(str(resolve_device(requested_device)))

    if args.smoke_test:
        smoke_bs = int(args.batch_size or 4)
        smoke_img_sz = int(args.image_size or 128)
        smoke_ckpt_dir = (
            PROJECT_ROOT / args.checkpoint_dir
            if args.checkpoint_dir
            else PROJECT_ROOT / "checkpoints/ssl/smoke_test"
        )
        smoke_report_path = (
            PROJECT_ROOT / args.report_path
            if args.report_path
            else PROJECT_ROOT / "logs/simclr_smoke_test.json"
        )
        report = run_simclr_smoke_test(
            base_cfg=base_cfg,
            data_cfg=data_cfg,
            ssl_cfg=ssl_cfg,
            device=device,
            batch_size=smoke_bs,
            image_size=smoke_img_sz,
            checkpoint_dir=smoke_ckpt_dir,
            report_path=smoke_report_path,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.preflight:
        aug_cfg = ssl_cfg.get("augmentations", {})
        preflight_img_sz = int(args.image_size or aug_cfg.get("image_size", 512))
        preflight_ckpt_dir = (
            PROJECT_ROOT / args.checkpoint_dir
            if args.checkpoint_dir
            else PROJECT_ROOT / "checkpoints/ssl/preflight"
        )
        preflight_report_path = (
            PROJECT_ROOT / args.report_path
            if args.report_path
            else PROJECT_ROOT / "logs/simclr_preflight_report.json"
        )
        candidate_bs = (
            [int(args.batch_size)]
            if args.batch_size is not None
            else [int(b) for b in args.preflight_batch_sizes]
        )
        report = run_simclr_real_data_preflight(
            base_cfg=base_cfg,
            data_cfg=data_cfg,
            ssl_cfg=ssl_cfg,
            device=device,
            manifest_path=PROJECT_ROOT / args.manifest,
            allowed_datasets=args.datasets,
            candidate_batch_sizes=candidate_bs,
            image_size=preflight_img_sz,
            checkpoint_dir=preflight_ckpt_dir,
            report_path=preflight_report_path,
        )
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "passed" else 1

    manifest_path = PROJECT_ROOT / args.manifest
    manifest = load_manifest_json(manifest_path)
    all_records = manifest.get("records", [])

    ssl_records = select_ssl_training_records(
        manifest_or_records=all_records,
        allowed_datasets=args.datasets,
        filter_train_split=True,
    )
    excluded_val_count = sum(
        1
        for r in all_records
        if str(r.get("split", "")).strip().lower() in {"valid", "val", "validation"}
    )
    excluded_test_count = sum(
        1 for r in all_records if str(r.get("split", "")).strip().lower() == "test"
    )

    aug_cfg = ssl_cfg.get("augmentations", {})
    image_size = int(aug_cfg.get("image_size", 512))
    batch_size = int(ssl_cfg.get("batch_size", 16))
    num_epochs = 1 if args.dry_run else int(ssl_cfg.get("epochs", 50))
    use_cache = not bool(args.no_cache or args.dry_run)

    simclr_tf = build_simclr_transforms(
        ssl_config=ssl_cfg,
        dataset_config=data_cfg,
        image_size=(image_size, image_size),
    )
    ssl_dataset = SimCLRUnlabeledDataset(
        manifest_or_records=ssl_records,
        allowed_datasets=args.datasets,
        transform=simclr_tf,
        ssl_config=ssl_cfg,
        dataset_config=data_cfg,
        image_size=(image_size, image_size),
        return_metadata=True,
        filter_train_split=False,
        cache_in_memory=use_cache,
    )

    if use_cache:
        t_pre = time.perf_counter()
        logger.info(
            "Pre-loading and FOV-cropping %d unlabeled SSL training images into memory...",
            len(ssl_dataset),
        )
        ssl_dataset.preload_cache(max_workers=8)
        logger.info(
            "Pre-loaded %d SSL images in %.2fs",
            len(ssl_dataset),
            time.perf_counter() - t_pre,
        )

    num_workers = (
        0
        if (args.dry_run or use_cache)
        else int(data_cfg.get("number_of_workers", 0))
    )
    train_loader = build_simclr_dataloader(
        dataset=ssl_dataset,
        ssl_config=ssl_cfg,
        dataset_config=data_cfg,
        batch_size=batch_size,
        shuffle=not args.dry_run,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        seed=seed,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    strict_pretrained = bool(ssl_cfg.get("pretrained", True) and not args.no_pretrained)
    model = build_simclr_from_config(
        ssl_cfg,
        strict_pretrained_override=strict_pretrained,
    )
    criterion = build_nt_xent_loss_from_config(ssl_cfg)
    optimizer = build_ssl_optimizer_from_config(model, ssl_cfg)
    scheduler = build_ssl_scheduler_from_config(
        optimizer, ssl_cfg, total_epochs=num_epochs
    )

    ckpt_dir = Path(
        args.checkpoint_dir
        or PROJECT_ROOT / str(ssl_cfg.get("checkpoint_directory", "checkpoints/ssl"))
    )
    trainer = SimCLRTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=ckpt_dir,
        mixed_precision=bool(ssl_cfg.get("mixed_precision", True)),
        gradient_clip_max_norm=ssl_cfg.get("gradient_clip_max_norm", 1.0),
        seed=seed,
        config={"base": base_cfg, "data": data_cfg, "ssl": ssl_cfg},
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    t0 = time.perf_counter()
    fit_result = trainer.fit(
        train_loader=train_loader,
        num_epochs=num_epochs,
        max_train_batches=1 if args.dry_run else None,
        save_checkpoints=not args.dry_run,
    )
    total_duration_sec = round(time.perf_counter() - t0, 2)

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

    report: dict[str, Any] = {
        "status": "completed",
        "dry_run": bool(args.dry_run),
        "seed": seed,
        "deterministic": deterministic,
        "device": str(device),
        "amp_enabled": trainer.amp_enabled,
        "dataset": {
            "manifest_path": format_project_path(manifest_path),
            "datasets_included": sorted(
                list({str(r["dataset_name"]) for r in ssl_records})
            ),
            "unlabeled_train_samples": len(ssl_records),
            "validation_samples_excluded": excluded_val_count,
            "test_samples_excluded": excluded_test_count,
            "masks_used": False,
            "image_size": [image_size, image_size],
            "batch_size": batch_size,
        },
        "model": model.get_model_summary(),
        "ssl_config": ssl_cfg,
        "epochs_completed": fit_result["epochs_completed"],
        "global_step": fit_result["global_step"],
        "best_epoch": fit_result["best_epoch"],
        "best_train_loss": fit_result["best_metric_value"],
        "training_duration_sec": total_duration_sec,
        "training_duration_minutes": round(total_duration_sec / 60.0, 2),
        "gpu_memory": gpu_memory,
        "checkpoints": {
            "latest_checkpoint": format_project_path(fit_result["latest_checkpoint"]),
            "best_checkpoint": format_project_path(fit_result["best_checkpoint"]),
            "canonical_ssl_checkpoint": format_project_path(
                fit_result["canonical_ssl_checkpoint"]
            ),
            "encoder_only_checkpoint": format_project_path(
                fit_result.get("encoder_only_checkpoint")
            ),
        },
        "epoch_history": fit_result["history"],
    }

    out_report_path = (
        PROJECT_ROOT / args.report_path
        if args.report_path
        else (
            None
            if args.dry_run
            else PROJECT_ROOT / "logs/simclr_pretraining_report.json"
        )
    )
    if out_report_path is not None:
        out_report_path.parent.mkdir(parents=True, exist_ok=True)
        out_report_path.write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        logger.info("Saved SimCLR pretraining report to %s", out_report_path)

    print(
        json.dumps(
            {
                "status": "completed",
                "dry_run": bool(args.dry_run),
                "unlabeled_train_samples": len(ssl_records),
                "epochs_completed": fit_result["epochs_completed"],
                "best_epoch": fit_result["best_epoch"],
                "best_train_loss": fit_result["best_metric_value"],
                "training_duration_sec": total_duration_sec,
                "gpu_memory": gpu_memory,
                "report_path": format_project_path(out_report_path)
                if out_report_path
                else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

