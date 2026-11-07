"""Training and validation engines for FedMorphis."""

from src.engine.evaluator import (
    SplitMismatchError,
    evaluate_checkpoint_on_manifest,
    evaluate_segmentation_dataloader,
    reconstruct_model_from_checkpoint,
    verify_exact_split,
)
from src.engine.simclr_trainer import (
    SimCLRTrainer,
    build_ssl_optimizer_from_config,
    build_ssl_scheduler_from_config,
)
from src.engine.trainer import (
    SupervisedSegmentationTrainer,
    TestSplitLeakageError,
    build_optimizer_from_config,
    build_validation_comparison_report,
    verify_no_test_split_leakage,
)

__all__ = [
    "SimCLRTrainer",
    "SplitMismatchError",
    "SupervisedSegmentationTrainer",
    "TestSplitLeakageError",
    "build_optimizer_from_config",
    "build_ssl_optimizer_from_config",
    "build_ssl_scheduler_from_config",
    "build_validation_comparison_report",
    "evaluate_checkpoint_on_manifest",
    "evaluate_segmentation_dataloader",
    "reconstruct_model_from_checkpoint",
    "verify_exact_split",
    "verify_no_test_split_leakage",
]


