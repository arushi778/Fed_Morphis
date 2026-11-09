"""CLI entry point for Flower federated retinal lesion segmentation training.

Supports:
- Loading ``configs/federated.yaml`` (and companion ``project.yaml``, ``dataset.yaml``,
  ``model.yaml``, ``experiment.yaml``) with full schema and override validation.
- Configurable communication rounds, local epochs, client fractions, batch size,
  learning rate, seed, device, and global initialization source (``imagenet``,
  ``ssl_checkpoint``, ``random``, or full segmentation checkpoint).
- Executing two-client cross-dataset FedAvg training (Client 0 = DDR, Client 1 = IDRiD).
- Saving atomic global checkpoints (``latest_checkpoint.pt``, ``best_checkpoint.pt``)
  and a structured JSON experiment report.

Example command:
    python scripts/train_federated.py --config configs/federated.yaml
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.federated.server import (
    FederatedServerRunResult,
    run_federated_training,
)
from src.utils.config import (
    ConfigValidationError,
    load_yaml_config,
    validate_dataset_config,
    validate_experiment_config,
    validate_federated_config,
    validate_model_config,
    validate_project_config,
)
from src.utils.device import get_device_info, resolve_device
from src.utils.logger import configure_federated_logging, get_logger
from src.utils.paths import resolve_project_path
from src.utils.seed import set_seed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for Flower federated segmentation training."""
    parser = argparse.ArgumentParser(
        description="Run Flower FedAvg federated retinal lesion segmentation training."
    )
    parser.add_argument(
        "--config",
        "--federated-config",
        dest="config",
        type=str,
        default="configs/federated.yaml",
        help="Path to federated configuration YAML (default: 'configs/federated.yaml').",
    )
    parser.add_argument(
        "--project-config",
        type=str,
        default="configs/project.yaml",
        help="Path to project configuration YAML (default: 'configs/project.yaml').",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="configs/dataset.yaml",
        help="Path to dataset configuration YAML (default: 'configs/dataset.yaml').",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model.yaml",
        help="Path to model configuration YAML (default: 'configs/model.yaml').",
    )
    parser.add_argument(
        "--experiment-config",
        type=str,
        default="configs/experiment.yaml",
        help="Path to experiment configuration YAML (default: 'configs/experiment.yaml').",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional override for the experiment name recorded in the report.",
    )
    parser.add_argument(
        "--partition-manifest",
        type=str,
        default=None,
        help="Optional override path to the federated client partition manifest JSON.",
    )
    parser.add_argument(
        "--source-manifest",
        "--manifest",
        dest="source_manifest",
        type=str,
        default=None,
        help="Optional override path to the combined dataset source manifest JSON.",
    )
    parser.add_argument(
        "--rounds",
        "--num-rounds",
        dest="rounds",
        type=int,
        default=None,
        help="Override number of federated communication rounds.",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=None,
        help="Override number of local client training epochs per round.",
    )
    parser.add_argument(
        "--client-fraction",
        "--fraction-fit",
        dest="client_fraction",
        type=float,
        default=None,
        help="Override fraction of available clients sampled per round in (0.0, 1.0].",
    )
    parser.add_argument(
        "--min-fit-clients",
        type=int,
        default=None,
        help="Override minimum number of clients required for training per round.",
    )
    parser.add_argument(
        "--min-evaluate-clients",
        type=int,
        default=None,
        help="Override minimum number of clients required for validation per round.",
    )
    parser.add_argument(
        "--min-available-clients",
        type=int,
        default=None,
        help="Override minimum number of available clients required by the server.",
    )
    parser.add_argument(
        "--batch-size",
        "--local-batch-size",
        dest="batch_size",
        type=int,
        default=None,
        help="Override local training batch size.",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=None,
        help="Override local validation batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override number of DataLoader worker processes per client.",
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        dest="learning_rate",
        type=float,
        default=None,
        help="Override local client optimizer learning rate.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override square image resolution (H = W, divisible by 32).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override global and partitioning random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override execution device ('cuda', 'cpu', or 'auto').",
    )
    parser.add_argument(
        "--encoder-weights-source",
        type=str,
        default=None,
        help="Override global encoder initialization ('imagenet', 'ssl_checkpoint', 'random').",
    )
    parser.add_argument(
        "--ssl-checkpoint-path",
        "--pretrained-checkpoint",
        dest="ssl_checkpoint_path",
        type=str,
        default=None,
        help="Override path to SimCLR encoder checkpoint when encoder_weights_source='ssl_checkpoint'.",
    )
    parser.add_argument(
        "--initial-checkpoint-path",
        type=str,
        default=None,
        help="Optional full segmentation checkpoint (.pt) to initialize the global model from.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Initialize global model from random weights (disables pretrained encoder weights).",
    )
    bn_group = parser.add_mutually_exclusive_group()
    bn_group.add_argument(
        "--share-batchnorm-buffers",
        dest="share_batchnorm_buffers",
        action="store_true",
        default=None,
        help="Aggregate BatchNorm running statistics and counters across clients.",
    )
    bn_group.add_argument(
        "--no-share-batchnorm-buffers",
        dest="share_batchnorm_buffers",
        action="store_false",
        default=None,
        help="Keep BatchNorm running statistics local to each client (FedBN-style).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override output directory for global checkpoints.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Override output JSON path for the federated training report.",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Optional output path for the persistent federated training log file.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level for real-time console and persistent file output (default: 'INFO').",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional cap on training batches per local epoch (for bounded smoke runs).",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Optional cap on validation batches per evaluation round.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable in-memory caching of FOV-cropped dataset samples.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def load_and_validate_federated_cli_configs(
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    """Load YAML configs, apply CLI overrides, and validate all configurations before training."""
    fed_cfg = copy.deepcopy(
        load_yaml_config(resolve_project_path(args.config), validate=False)
    )
    proj_cfg = copy.deepcopy(
        load_yaml_config(resolve_project_path(args.project_config), validate=False)
    )
    ds_cfg = copy.deepcopy(
        load_yaml_config(resolve_project_path(args.dataset_config), validate=False)
    )
    mdl_cfg = copy.deepcopy(
        load_yaml_config(resolve_project_path(args.model_config), validate=False)
    )
    exp_cfg = copy.deepcopy(
        load_yaml_config(resolve_project_path(args.experiment_config), validate=False)
    )

    # Apply CLI overrides to federated config
    if args.rounds is not None:
        fed_cfg["rounds"] = int(args.rounds)
        fed_cfg["communication_rounds"] = int(args.rounds)

    if args.local_epochs is not None:
        fed_cfg["local_epochs"] = int(args.local_epochs)

    if args.client_fraction is not None:
        fed_cfg["client_fraction"] = float(args.client_fraction)

    if args.min_fit_clients is not None:
        fed_cfg["min_fit_clients"] = int(args.min_fit_clients)

    if args.min_evaluate_clients is not None:
        fed_cfg["min_evaluate_clients"] = int(args.min_evaluate_clients)

    if args.min_available_clients is not None:
        fed_cfg["min_available_clients"] = int(args.min_available_clients)

    if args.batch_size is not None:
        fed_cfg["local_batch_size"] = int(args.batch_size)
        ds_cfg["batch_size"] = int(args.batch_size)

    if args.val_batch_size is not None:
        if int(args.val_batch_size) < 1:
            raise ConfigValidationError(
                f"val_batch_size must be >= 1, got {args.val_batch_size}."
            )

    if args.num_workers is not None:
        fed_cfg["num_workers"] = int(args.num_workers)
        ds_cfg["number_of_workers"] = int(args.num_workers)

    if args.learning_rate is not None:
        fed_cfg["local_learning_rate"] = float(args.learning_rate)
        train_block = dict(mdl_cfg.get("training", {}))
        train_block["learning_rate"] = float(args.learning_rate)
        mdl_cfg["training"] = train_block

    if args.image_size is not None:
        img_sz = int(args.image_size)
        if img_sz < 32 or img_sz % 32 != 0:
            raise ConfigValidationError(
                f"image_size must be >= 32 and divisible by 32, got {img_sz}."
            )
        ds_cfg["image_size"] = {"height": img_sz, "width": img_sz}

    if args.seed is not None:
        fed_cfg["seed"] = int(args.seed)
        part_block = dict(fed_cfg.get("partitioning", {}))
        part_block["seed"] = int(args.seed)
        fed_cfg["partitioning"] = part_block
        proj_cfg["random_seed"] = int(args.seed)

    if args.device is not None:
        proj_cfg["device"] = str(args.device)

    if args.partition_manifest is not None:
        part_block = dict(fed_cfg.get("partitioning", {}))
        part_block["manifest_path"] = str(args.partition_manifest)
        fed_cfg["partitioning"] = part_block

    init_block = dict(fed_cfg.get("initialization", {}))
    if args.no_pretrained:
        init_block["pretrained"] = False
        init_block["encoder_weights_source"] = "random"
        mdl_cfg["pretrained"] = False
        mdl_cfg["encoder_weights_source"] = "random"

    if args.encoder_weights_source is not None:
        src_val = str(args.encoder_weights_source).strip()
        init_block["encoder_weights_source"] = src_val
        mdl_cfg["encoder_weights_source"] = src_val
        if src_val == "random":
            init_block["pretrained"] = False
            mdl_cfg["pretrained"] = False
        else:
            init_block["pretrained"] = True
            mdl_cfg["pretrained"] = True

    if args.ssl_checkpoint_path is not None:
        init_block["ssl_checkpoint_path"] = str(args.ssl_checkpoint_path)
        mdl_cfg["ssl_checkpoint_path"] = str(args.ssl_checkpoint_path)

    if args.initial_checkpoint_path is not None:
        init_block["initial_checkpoint_path"] = str(args.initial_checkpoint_path)

    fed_cfg["initialization"] = init_block

    if args.share_batchnorm_buffers is not None:
        fed_cfg["share_batchnorm_buffers"] = bool(args.share_batchnorm_buffers)

    if args.checkpoint_dir is not None:
        fed_cfg["checkpoint_directory"] = str(args.checkpoint_dir)
        ckpt_block = dict(fed_cfg.get("checkpointing", {}))
        ckpt_block["save_dir"] = str(args.checkpoint_dir)
        fed_cfg["checkpointing"] = ckpt_block

    if args.report_path is not None:
        fed_cfg["report_path"] = str(args.report_path)
        log_block = dict(fed_cfg.get("logging", {}))
        log_block["report_path"] = str(args.report_path)
        fed_cfg["logging"] = log_block

    if getattr(args, "log_path", None) is not None:
        fed_cfg["log_path"] = str(args.log_path)
        log_block = dict(fed_cfg.get("logging", {}))
        log_block["log_path"] = str(args.log_path)
        fed_cfg["logging"] = log_block

    if getattr(args, "log_level", None) is not None:
        log_block = dict(fed_cfg.get("logging", {}))
        log_block["log_level"] = str(args.log_level)
        fed_cfg["logging"] = log_block

    if args.experiment_name is not None:
        exp_name = str(args.experiment_name).strip()
        if not exp_name:
            raise ConfigValidationError("experiment_name cannot be empty.")
        exp_cfg["experiment_name"] = exp_name
        fed_cfg["experiment_name"] = exp_name

    if args.max_train_batches is not None and int(args.max_train_batches) < 1:
        raise ConfigValidationError(
            f"max_train_batches must be >= 1, got {args.max_train_batches}."
        )
    if args.max_val_batches is not None and int(args.max_val_batches) < 1:
        raise ConfigValidationError(
            f"max_val_batches must be >= 1, got {args.max_val_batches}."
        )

    # Validate all merged configuration sections before training starts
    validate_project_config(proj_cfg)
    validate_dataset_config(ds_cfg)
    validate_model_config(mdl_cfg)
    validate_federated_config(fed_cfg)
    validate_experiment_config(exp_cfg)

    return {
        "federated": fed_cfg,
        "project": proj_cfg,
        "dataset": ds_cfg,
        "model": mdl_cfg,
        "experiment": exp_cfg,
    }


def run_from_args(args: argparse.Namespace) -> FederatedServerRunResult:
    """Validate configuration overrides and execute federated training."""
    configs = load_and_validate_federated_cli_configs(args)

    fed_cfg = configs["federated"]
    proj_cfg = configs["project"]
    ds_cfg = configs["dataset"]
    mdl_cfg = configs["model"]
    exp_cfg = configs["experiment"]

    log_path_arg = getattr(args, "log_path", None) or fed_cfg.get(
        "logging", {}
    ).get("log_path", fed_cfg.get("log_path"))
    log_level_arg = str(
        getattr(args, "log_level", None)
        or fed_cfg.get("logging", {}).get("log_level", "INFO")
    ).upper()
    if log_path_arg is not None:
        configure_federated_logging(
            level=log_level_arg,
            log_path=resolve_project_path(log_path_arg),
            console=True,
        )
    logger = get_logger("fedmorphis.train_federated")

    seed = int(fed_cfg.get("seed", proj_cfg.get("random_seed", 42)))
    deterministic = bool(proj_cfg.get("deterministic", True))
    set_seed(seed, deterministic=deterministic)

    device_info = get_device_info()
    target_device = resolve_device(str(proj_cfg.get("device", "auto")))

    img_cfg = ds_cfg.get("image_size", {"height": 512, "width": 512})
    if isinstance(img_cfg, Mapping):
        image_size = (int(img_cfg.get("height", 512)), int(img_cfg.get("width", 512)))
    else:
        image_size = (512, 512)

    logger.info(
        "Starting Flower FedAvg federated training (rounds=%d, local_epochs=%d, clients=%d, seed=%d, device=%s, cuda_available=%s)",
        int(fed_cfg.get("rounds", 20)),
        int(fed_cfg.get("local_epochs", 3)),
        int(fed_cfg.get("number_of_clients", 2)),
        seed,
        target_device,
        device_info.get("cuda_available"),
    )

    result = run_federated_training(
        federated_config=fed_cfg,
        dataset_config=ds_cfg,
        model_config=mdl_cfg,
        project_config=proj_cfg,
        experiment_config=exp_cfg,
        experiment_name=args.experiment_name,
        num_rounds=int(fed_cfg.get("rounds", 20)),
        partition_manifest=args.partition_manifest,
        source_manifest=args.source_manifest,
        checkpoint_dir=args.checkpoint_dir,
        save_checkpoints=True,
        report_path=args.report_path,
        save_report=True,
        log_path=log_path_arg,
        log_level=log_level_arg,
        accept_failures=False,
        include_buffers=bool(fed_cfg.get("share_batchnorm_buffers", True)),
        local_epochs=int(fed_cfg.get("local_epochs", 3)),
        batch_size=int(fed_cfg.get("local_batch_size", 8)),
        val_batch_size=args.val_batch_size,
        num_workers=int(fed_cfg.get("num_workers", 0)),
        learning_rate=float(fed_cfg.get("local_learning_rate", 1e-4)),
        seed=seed,
        device=target_device,
        image_size=image_size,
        cache_in_memory=(not bool(args.no_cache)),
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        max_workers=1,
    )

    report_dict = result.to_report_dict()
    logger.info(
        "Completed federated training (%d rounds in %.2fs) | best_round=%d | best_%s=%s | latest_ckpt=%s | best_ckpt=%s",
        result.num_rounds,
        result.elapsed_sec,
        result.best_round,
        result.monitor_metric,
        f"{result.best_metric_value:.4f}" if result.best_metric_value is not None else "N/A",
        report_dict["latest_checkpoint_path"],
        report_dict["best_checkpoint_path"],
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for Flower federated segmentation training."""
    args = parse_args(argv)
    result = run_from_args(args)
    report_dict = result.to_report_dict()

    summary_stdout = {
        "status": report_dict["status"],
        "experiment_name": report_dict["experiment_name"],
        "strategy": report_dict["strategy"],
        "number_of_clients": report_dict["number_of_clients"],
        "num_rounds": report_dict["num_rounds"],
        "rounds_completed": report_dict["rounds_completed"],
        "local_epochs": report_dict["local_epochs"],
        "seed": report_dict["seed"],
        "device": report_dict["device"],
        "global_initialization_source": report_dict["global_initialization_source"],
        "client_sample_counts": report_dict["client_sample_counts"],
        "monitor_metric": report_dict["monitor_metric"],
        "best_round": report_dict["best_round"],
        "best_metric_value": report_dict["best_metric_value"],
        "best_checkpoint_path": report_dict["best_checkpoint_path"],
        "latest_checkpoint_path": report_dict["latest_checkpoint_path"],
        "report_path": (
            str(result.report_path.as_posix()) if result.report_path is not None else None
        ),
        "log_path": report_dict.get("log_path"),
        "training_duration_sec": report_dict["training_duration_sec"],
    }
    print(json.dumps(summary_stdout, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
