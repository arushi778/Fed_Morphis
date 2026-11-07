"""SimCLR self-supervised representation learning model with a ResNet-50 encoder.

Architectural Specification:
- **Input**: Augmented RGB retinal fundus image tensor of shape ``(B, 3, H, W)``
  (``torch.float32``).
- **Encoder** (:class:`ResNet50Encoder`): ``torchvision.models.resnet50`` backbone
  with the classification layer (``fc``) removed, mapping ``(B, 3, H, W)`` to global
  pooled feature representations ``h`` of shape ``(B, 2048)``.
- **Projection Head** (:class:`SimCLRProjectionHead`): Configurable two-layer MLP
  ``Linear(2048, hidden_dim) -> [optional BatchNorm1d] -> ReLU -> Linear(hidden_dim, projection_dim)``
  mapping ``h`` ``(B, 2048)`` to contrastive projection vectors ``z`` of shape
  ``(B, projection_dim)``.
- **Outputs**:
  - ``forward(x)`` returns ``(h, z)`` where ``h`` has shape ``(B, 2048)`` and
    ``z`` has shape ``(B, projection_dim)``.
  - ``forward_views(view1, view2)`` returns ``{"h1": h1, "z1": z1, "h2": h2, "z2": z2}``.
- **Weight Loading & Transfer Compatibility**:
  - Supports ImageNet pretrained initialization, random initialization, loading
    encoder weights from a completed supervised ``ResNet50UNet`` checkpoint
    (e.g., ``checkpoints/segmentation/best_checkpoint.pt``), or loading from a
    saved SimCLR checkpoint.
  - Completely independent of the U-Net segmentation decoder while maintaining
    exact ResNet-50 encoder parameter compatibility with :class:`src.models.unet.ResNet50UNet`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

from src.models.losses import NTXentLoss, build_nt_xent_loss_from_config
from src.models.unet import PretrainedWeightsError, ResNet50UNet
from src.utils.config import load_yaml_config
from src.utils.logger import get_logger
from src.utils.paths import format_project_path, resolve_project_path

logger = get_logger("fedmorphis.models.simclr")

RESNET50_FEATURE_DIM: int = 2048
DEFAULT_PROJECTION_DIM: int = 128
DEFAULT_PROJECTION_HIDDEN_DIM: int = 2048
SUPPORTED_SIMCLR_WEIGHT_SOURCES: tuple[str, ...] = (
    "imagenet",
    "supervised_checkpoint",
    "ssl_checkpoint",
    "random",
)


def _normalize_encoder_state_dict(raw_state_dict: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Normalize encoder parameter keys from TorchVision, ResNet50UNet, or SimCLR checkpoints.

    Maps keys such as:
    - ``encoder_conv1.weight`` (from :class:`ResNet50UNet`) -> ``conv1.weight``
    - ``encoder_layer1.0.conv1.weight`` (from :class:`ResNet50UNet`) -> ``layer1.0.conv1.weight``
    - ``encoder.conv1.weight`` / ``backbone.conv1.weight`` -> ``conv1.weight``
    and filters out classification head (``fc.*``), U-Net decoder (``decoder*``),
    segmentation head (``segmentation_head.*``), and SimCLR projection head
    (``projection_head.*``) parameters.
    """
    valid_prefixes = ("conv1.", "bn1.", "layer1.", "layer2.", "layer3.", "layer4.")
    cleaned: dict[str, torch.Tensor] = {}

    for key, val in raw_state_dict.items():
        if not isinstance(val, torch.Tensor):
            continue
        k = str(key)
        if k.startswith("module."):
            k = k.removeprefix("module.")
        if k.startswith("encoder."):
            k = k.removeprefix("encoder.")
        elif k.startswith("backbone."):
            k = k.removeprefix("backbone.")

        # Map ResNet50UNet encoder_* submodule prefixes to canonical ResNet-50 keys
        if k.startswith("encoder_conv1."):
            k = "conv1." + k.removeprefix("encoder_conv1.")
        elif k.startswith("encoder_bn1."):
            k = "bn1." + k.removeprefix("encoder_bn1.")
        elif k.startswith("encoder_layer1."):
            k = "layer1." + k.removeprefix("encoder_layer1.")
        elif k.startswith("encoder_layer2."):
            k = "layer2." + k.removeprefix("encoder_layer2.")
        elif k.startswith("encoder_layer3."):
            k = "layer3." + k.removeprefix("encoder_layer3.")
        elif k.startswith("encoder_layer4."):
            k = "layer4." + k.removeprefix("encoder_layer4.")

        if k.startswith(valid_prefixes):
            cleaned[k] = val

    return cleaned


class ResNet50Encoder(nn.Module):
    """ResNet-50 feature extractor without the classification head.

    Maps input images ``(B, C, H, W)`` to global average-pooled feature vectors
    ``h`` of shape ``(B, 2048)``.
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}.")

        self.in_channels = int(in_channels)
        self.out_dim = RESNET50_FEATURE_DIM

        backbone = resnet50(weights=None)
        if self.in_channels != 3:
            old_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                self.in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None),
            )

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

    def forward_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract multi-scale ResNet-50 feature maps ``(s0, s1, s2, s3, bottleneck)``."""
        s0 = self.relu(self.bn1(self.conv1(x)))  # (B, 64, H/2, W/2)
        p0 = self.maxpool(s0)  # (B, 64, H/4, W/4)
        s1 = self.layer1(p0)  # (B, 256, H/4, W/4)
        s2 = self.layer2(s1)  # (B, 512, H/8, W/8)
        s3 = self.layer3(s2)  # (B, 1024, H/16, W/16)
        bottleneck = self.layer4(s3)  # (B, 2048, H/32, W/32)
        return s0, s1, s2, s3, bottleneck

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pooled encoder representation ``h`` of shape ``(B, 2048)``."""
        _, _, _, _, bottleneck = self.forward_features(x)
        pooled = self.avgpool(bottleneck)
        return torch.flatten(pooled, start_dim=1)


class SimCLRProjectionHead(nn.Module):
    """Two-layer MLP projection head for SimCLR contrastive learning.

    Architecture:
        ``Linear(in_dim, hidden_dim) -> [BatchNorm1d] -> ReLU -> Linear(hidden_dim, out_dim)``
    """

    def __init__(
        self,
        in_dim: int = RESNET50_FEATURE_DIM,
        hidden_dim: int = DEFAULT_PROJECTION_HIDDEN_DIM,
        out_dim: int = DEFAULT_PROJECTION_DIM,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(in_dim, bool) or not isinstance(in_dim, int) or in_dim < 1:
            raise ValueError(f"in_dim must be an integer >= 1, got {in_dim!r}.")
        if (
            isinstance(hidden_dim, bool)
            or not isinstance(hidden_dim, int)
            or hidden_dim < 1
        ):
            raise ValueError(
                f"hidden_dim must be an integer >= 1, got {hidden_dim!r}."
            )
        if isinstance(out_dim, bool) or not isinstance(out_dim, int) or out_dim < 1:
            raise ValueError(f"out_dim must be an integer >= 1, got {out_dim!r}.")

        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.use_batchnorm = bool(use_batchnorm)

        layers: list[nn.Module] = [
            nn.Linear(self.in_dim, self.hidden_dim, bias=not self.use_batchnorm),
        ]
        if self.use_batchnorm:
            layers.append(nn.BatchNorm1d(self.hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(self.hidden_dim, self.out_dim, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project encoder representations ``(B, in_dim)`` to ``(B, out_dim)``."""
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            raise ValueError(
                f"Expected 2D representation tensor of shape (B, {self.in_dim}), "
                f"got {getattr(x, 'shape', type(x))}."
            )
        if x.shape[1] != self.in_dim:
            raise ValueError(
                f"Expected representation dimension {self.in_dim}, got {x.shape[1]}."
            )
        return self.net(x)


class SimCLRModel(nn.Module):
    """SimCLR self-supervised architecture with a ResNet-50 encoder and 2-layer MLP projection head."""

    def __init__(
        self,
        in_channels: int = 3,
        projection_dim: int = DEFAULT_PROJECTION_DIM,
        projection_hidden_dim: int = DEFAULT_PROJECTION_HIDDEN_DIM,
        projection_use_batchnorm: bool = False,
        pretrained: bool = False,
        strict_pretrained: bool = True,
        encoder_weights_source: str = "imagenet",
        pretrained_checkpoint_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}.")
        if (
            isinstance(projection_dim, bool)
            or not isinstance(projection_dim, int)
            or projection_dim < 1
        ):
            raise ValueError(
                f"projection_dim must be an integer >= 1, got {projection_dim!r}."
            )
        if (
            isinstance(projection_hidden_dim, bool)
            or not isinstance(projection_hidden_dim, int)
            or projection_hidden_dim < 1
        ):
            raise ValueError(
                f"projection_hidden_dim must be an integer >= 1, got {projection_hidden_dim!r}."
            )

        weights_src = str(encoder_weights_source).strip().lower()
        if weights_src not in SUPPORTED_SIMCLR_WEIGHT_SOURCES:
            raise ValueError(
                f"Unsupported encoder_weights_source {encoder_weights_source!r}; "
                f"expected one of {SUPPORTED_SIMCLR_WEIGHT_SOURCES}."
            )

        self.in_channels = int(in_channels)
        self.encoder_dim = RESNET50_FEATURE_DIM
        self.projection_dim = int(projection_dim)
        self.projection_hidden_dim = int(projection_hidden_dim)
        self.projection_use_batchnorm = bool(projection_use_batchnorm)
        self.pretrained_requested = bool(pretrained)
        self.strict_pretrained = bool(strict_pretrained)
        self.encoder_weights_source = weights_src
        self.pretrained_weights_loaded: bool = False
        self.weights_status: str = "random_initialization"

        self.encoder = ResNet50Encoder(in_channels=self.in_channels)
        self.projection_head = SimCLRProjectionHead(
            in_dim=self.encoder_dim,
            hidden_dim=self.projection_hidden_dim,
            out_dim=self.projection_dim,
            use_batchnorm=self.projection_use_batchnorm,
        )

        self._initialize_encoder_weights(
            pretrained=self.pretrained_requested,
            strict_pretrained=self.strict_pretrained,
            weights_source=self.encoder_weights_source,
            checkpoint_path=pretrained_checkpoint_path,
        )

    def _initialize_encoder_weights(
        self,
        pretrained: bool,
        strict_pretrained: bool,
        weights_source: str,
        checkpoint_path: str | Path | None,
    ) -> None:
        """Load encoder weights according to ``weights_source`` and ``pretrained``."""
        if not pretrained or weights_source == "random":
            self.pretrained_weights_loaded = False
            self.weights_status = "random_initialization"
            return

        if checkpoint_path is not None or weights_source in {
            "supervised_checkpoint",
            "ssl_checkpoint",
        }:
            if checkpoint_path is None:
                raise PretrainedWeightsError(
                    f"encoder_weights_source={weights_source!r} requires a valid pretrained_checkpoint_path."
                )
            ckpt_p = resolve_project_path(checkpoint_path)
            if not ckpt_p.exists():
                msg = f"Pretrained encoder checkpoint not found at '{ckpt_p}'."
                if strict_pretrained:
                    raise PretrainedWeightsError(msg)
                logger.warning("%s Falling back to random initialization.", msg)
                self.pretrained_weights_loaded = False
                self.weights_status = f"fallback_random_missing_checkpoint:{ckpt_p.name}"
                return

            try:
                self.load_encoder_from_checkpoint(ckpt_p, strict=True)
                return
            except Exception as exc:
                if strict_pretrained:
                    raise PretrainedWeightsError(
                        f"Failed to load encoder weights from checkpoint '{ckpt_p}': {exc}"
                    ) from exc
                logger.warning(
                    "Failed to load checkpoint '%s' (%s); falling back to random initialization.",
                    ckpt_p,
                    exc,
                )
                self.pretrained_weights_loaded = False
                self.weights_status = (
                    f"fallback_random_checkpoint_error:{type(exc).__name__}"
                )
                return

        # weights_source == "imagenet"
        try:
            hub_ckpt_dir = Path(torch.hub.get_dir()) / "checkpoints"
            cached_candidates = (
                sorted(hub_ckpt_dir.glob("resnet50*.pth"))
                if hub_ckpt_dir.exists()
                else []
            )
            if cached_candidates:
                raw_sd = torch.load(
                    cached_candidates[0], map_location="cpu", weights_only=True
                )
                cleaned_sd = _normalize_encoder_state_dict(raw_sd)
                if self.in_channels != 3:
                    cleaned_sd.pop("conv1.weight", None)
                    cleaned_sd.pop("conv1.bias", None)
                    self.encoder.load_state_dict(cleaned_sd, strict=False)
                else:
                    self.encoder.load_state_dict(cleaned_sd, strict=True)
                self.pretrained_weights_loaded = True
                self.weights_status = (
                    f"loaded_imagenet_cached:{cached_candidates[0].name}"
                )
                return

            weights_enum = ResNet50_Weights.DEFAULT
            raw_sd = weights_enum.get_state_dict(progress=False, check_hash=True)
            cleaned_sd = _normalize_encoder_state_dict(raw_sd)
            if self.in_channels != 3:
                cleaned_sd.pop("conv1.weight", None)
                cleaned_sd.pop("conv1.bias", None)
                self.encoder.load_state_dict(cleaned_sd, strict=False)
            else:
                self.encoder.load_state_dict(cleaned_sd, strict=True)
            self.pretrained_weights_loaded = True
            self.weights_status = "loaded_imagenet_torchvision"
        except Exception as exc:
            msg = (
                f"Unable to load ImageNet pretrained ResNet-50 weights "
                f"(no cached weights in '{Path(torch.hub.get_dir()) / 'checkpoints'}' "
                f"and download failed: {exc})."
            )
            if strict_pretrained:
                raise PretrainedWeightsError(msg) from exc
            logger.warning(
                "%s Falling back to random initialization (pretrained_weights_loaded=False).",
                msg,
            )
            self.pretrained_weights_loaded = False
            self.weights_status = (
                f"fallback_random_imagenet_unavailable:{type(exc).__name__}"
            )

    def load_encoder_from_checkpoint(
        self,
        checkpoint_path: str | Path,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Load ResNet-50 encoder weights from a supervised U-Net or SimCLR checkpoint.

        Automatically maps :class:`src.models.unet.ResNet50UNet` keys
        (``encoder_conv1.*``, ``encoder_bn1.*``, ``encoder_layer1.*``..``encoder_layer4.*``)
        or SimCLR keys (``encoder.*``) to :class:`ResNet50Encoder` and ignores
        decoder/segmentation_head/projection_head weights.
        """
        ckpt_p = resolve_project_path(checkpoint_path)
        if not ckpt_p.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_p}")

        raw_ckpt = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        raw_sd = ResNet50UNet._extract_state_dict_from_checkpoint(raw_ckpt, ckpt_p)

        cleaned_sd = _normalize_encoder_state_dict(raw_sd)
        if not cleaned_sd:
            raise ValueError(
                f"No compatible ResNet-50 encoder parameters found in checkpoint '{ckpt_p}'."
            )

        missing, unexpected = self.encoder.load_state_dict(cleaned_sd, strict=strict)
        self.pretrained_weights_loaded = True
        is_supervised = any(str(k).startswith("encoder_layer") for k in raw_sd.keys())
        source_tag = (
            "loaded_supervised_checkpoint"
            if is_supervised
            else "loaded_checkpoint"
        )
        self.weights_status = f"{source_tag}:{ckpt_p.name}"

        return {
            "checkpoint_path": format_project_path(ckpt_p),
            "loaded_parameter_count": len(cleaned_sd),
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "weights_status": self.weights_status,
        }

    def export_encoder_state_dict(self) -> dict[str, torch.Tensor]:
        """Return canonical ResNet-50 encoder state_dict compatible with :class:`ResNet50UNet`."""
        return {k: v.detach().clone() for k, v in self.encoder.state_dict().items()}

    def _validate_image_input(self, x: torch.Tensor) -> None:
        """Validate 4D image batch ``(B, C, H, W)`` before encoding."""
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError(
                f"Expected 4D input tensor (B, C, H, W), got {getattr(x, 'shape', type(x))}."
            )
        b, c, h, w = x.shape
        if b < 1:
            raise ValueError(f"Batch size B must be >= 1, got {b}.")
        if c != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {c} channels (shape={tuple(x.shape)})."
            )
        if h < 32 or w < 32:
            raise ValueError(
                f"Input spatial dimensions (H={h}, W={w}) must be at least 32x32 for ResNet-50 encoding."
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Compute 2048-d encoder representation ``h`` without applying the projection head."""
        self._validate_image_input(x)
        return self.encoder(x)

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """Apply the 2-layer MLP projection head to encoder representations ``h``."""
        return self.projection_head(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(h, z)`` where ``h`` is ``(B, 2048)`` and ``z`` is ``(B, projection_dim)``."""
        h = self.encode(x)
        z = self.project(h)
        return h, z

    def forward_views(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Encode and project two augmented views ``(view1, view2)`` for SimCLR pretraining."""
        h1, z1 = self.forward(view1)
        h2, z2 = self.forward(view2)
        return {
            "h1": h1,
            "z1": z1,
            "h2": h2,
            "z2": z2,
        }

    def get_model_summary(self) -> dict[str, Any]:
        """Return a structured summary of the SimCLR architecture and parameter counts."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        proj_params = sum(p.numel() for p in self.projection_head.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "architecture": "SimCLR",
            "backbone": "resnet50",
            "in_channels": self.in_channels,
            "encoder_dim": self.encoder_dim,
            "projection_hidden_dim": self.projection_hidden_dim,
            "projection_dim": self.projection_dim,
            "projection_use_batchnorm": self.projection_use_batchnorm,
            "pretrained_requested": self.pretrained_requested,
            "pretrained_weights_loaded": self.pretrained_weights_loaded,
            "weights_status": self.weights_status,
            "encoder_parameters": int(encoder_params),
            "projection_head_parameters": int(proj_params),
            "total_parameters": int(total_params),
            "trainable_parameters": int(trainable_params),
        }


SimCLRResNet50 = SimCLRModel


def build_simclr_from_config(
    config: Mapping[str, Any] | str | Path = "configs/ssl.yaml",
    pretrained_override: bool | None = None,
    strict_pretrained_override: bool | None = None,
) -> SimCLRModel:
    """Instantiate a :class:`SimCLRModel` from ``configs/ssl.yaml`` or ``configs/model.yaml``."""
    if isinstance(config, (str, Path)):
        cfg = load_yaml_config(config, validate=True)
    else:
        cfg = dict(config)

    if isinstance(cfg.get("simclr"), Mapping):
        cfg = dict(cfg["simclr"])

    backbone = str(cfg.get("backbone", cfg.get("encoder", "resnet50"))).strip().lower()
    if backbone != "resnet50":
        raise ValueError(
            f"Unsupported SimCLR backbone {backbone!r}; expected 'resnet50'."
        )

    in_channels = int(cfg.get("input_channels", 3))
    proj_dim = int(cfg.get("projection_dimension", DEFAULT_PROJECTION_DIM))
    proj_hidden_dim = int(
        cfg.get("projection_hidden_dimension", DEFAULT_PROJECTION_HIDDEN_DIM)
    )
    proj_use_bn = bool(cfg.get("projection_use_batchnorm", False))

    pretrained = (
        bool(pretrained_override)
        if pretrained_override is not None
        else bool(cfg.get("pretrained", True))
    )
    strict_pretrained = (
        bool(strict_pretrained_override)
        if strict_pretrained_override is not None
        else bool(cfg.get("strict_pretrained", False))
    )
    weights_src = str(cfg.get("encoder_weights_source", "imagenet")).strip().lower()

    ckpt_path: str | Path | None = None
    if weights_src == "supervised_checkpoint":
        ckpt_path = cfg.get(
            "supervised_checkpoint_path",
            cfg.get("pretrained_checkpoint_path", "checkpoints/segmentation/best_checkpoint.pt"),
        )
    elif weights_src == "ssl_checkpoint":
        ckpt_path = cfg.get(
            "ssl_checkpoint_path",
            cfg.get("pretrained_checkpoint_path"),
        )
    elif cfg.get("pretrained_checkpoint_path") is not None:
        ckpt_path = cfg.get("pretrained_checkpoint_path")

    return SimCLRModel(
        in_channels=in_channels,
        projection_dim=proj_dim,
        projection_hidden_dim=proj_hidden_dim,
        projection_use_batchnorm=proj_use_bn,
        pretrained=pretrained,
        strict_pretrained=strict_pretrained,
        encoder_weights_source=weights_src,
        pretrained_checkpoint_path=ckpt_path,
    )


__all__ = [
    "DEFAULT_PROJECTION_DIM",
    "DEFAULT_PROJECTION_HIDDEN_DIM",
    "NTXentLoss",
    "RESNET50_FEATURE_DIM",
    "ResNet50Encoder",
    "SUPPORTED_SIMCLR_WEIGHT_SOURCES",
    "SimCLRModel",
    "SimCLRProjectionHead",
    "SimCLRResNet50",
    "build_nt_xent_loss_from_config",
    "build_simclr_from_config",
]

