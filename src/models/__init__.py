"""Neural network backbones, ResNet-50 U-Net and SimCLR architectures, losses, and metrics."""

from src.models.losses import (
    BCEDiceLoss,
    NTXentLoss,
    SoftDiceLoss,
    build_loss_from_config,
    build_nt_xent_loss_from_config,
)
from src.models.metrics import (
    METRIC_NAMES,
    SegmentationMetricTracker,
    binarize_predictions,
    compute_confusion_counts,
    compute_segmentation_metrics,
    summarize_channel_counts,
)
from src.models.simclr import (
    DEFAULT_PROJECTION_DIM,
    DEFAULT_PROJECTION_HIDDEN_DIM,
    RESNET50_FEATURE_DIM,
    ResNet50Encoder,
    SimCLRModel,
    SimCLRProjectionHead,
    SimCLRResNet50,
    build_simclr_from_config,
)
from src.models.unet import (
    CANONICAL_LESION_ORDER,
    ConvBnReLU,
    PretrainedWeightsError,
    ResNet50UNet,
    UNetDecoderBlock,
    build_unet_from_config,
)

__all__ = [
    "BCEDiceLoss",
    "CANONICAL_LESION_ORDER",
    "ConvBnReLU",
    "DEFAULT_PROJECTION_DIM",
    "DEFAULT_PROJECTION_HIDDEN_DIM",
    "METRIC_NAMES",
    "NTXentLoss",
    "PretrainedWeightsError",
    "RESNET50_FEATURE_DIM",
    "ResNet50Encoder",
    "ResNet50UNet",
    "SegmentationMetricTracker",
    "SimCLRModel",
    "SimCLRProjectionHead",
    "SimCLRResNet50",
    "SoftDiceLoss",
    "UNetDecoderBlock",
    "binarize_predictions",
    "build_loss_from_config",
    "build_nt_xent_loss_from_config",
    "build_simclr_from_config",
    "build_unet_from_config",
    "compute_confusion_counts",
    "compute_segmentation_metrics",
    "summarize_channel_counts",
]
