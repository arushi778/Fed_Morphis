"""U-Net segmentation architecture with a ResNet-50 encoder for retinal lesion segmentation.

Architectural Specification:
- **Input**: RGB fundus image tensor of shape ``(B, 3, H, W)`` (``torch.float32``).
- **Encoder**: ``torchvision.models.resnet50`` hierarchical feature extractor:
  - ``stem`` (``conv1 -> bn1 -> relu``): ``(B, 64, H/2, W/2)``
  - ``layer1`` (after ``maxpool``): ``(B, 256, H/4, W/4)``
  - ``layer2``: ``(B, 512, H/8, W/8)``
  - ``layer3``: ``(B, 1024, H/16, W/16)``
  - ``layer4`` (bottleneck): ``(B, 2048, H/32, W/32)``
- **Decoder**: Five-stage U-Net upsampling pathway with skip connections from
  ``layer3``, ``layer2``, ``layer1``, and ``stem``, followed by a full-resolution
  refinement stage ``(256 -> 128 -> 64 -> 32 -> 16)``.
- **Output Head**: ``1x1`` convolution mapping decoder features to ``4`` independent
  lesion channels in canonical ``[MA, HE, EX, SE]`` order:
  - ``forward(x)`` returns **raw logits** of shape ``(B, 4, H, W)`` (no sigmoid).
  - ``predict_probabilities(x)`` applies channel-wise ``torch.sigmoid(logits)`` for
    evaluation metrics or inference.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50

from src.data.masks import CANONICAL_LESION_CHANNELS
from src.utils.config import load_yaml_config
from src.utils.logger import get_logger
from src.utils.paths import format_project_path, resolve_project_path

logger = get_logger("fedmorphis.models.unet")

CANONICAL_LESION_ORDER: tuple[str, ...] = tuple(CANONICAL_LESION_CHANNELS)
DEFAULT_DECODER_CHANNELS: tuple[int, int, int, int, int] = (256, 128, 64, 32, 16)


class PretrainedWeightsError(RuntimeError):
    """Raised when requested pretrained encoder weights cannot be loaded."""


class ConvBnReLU(nn.Module):
    """Two-layer ``3x3 Conv2d -> BatchNorm2d -> ReLU`` block used in U-Net decoder stages."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        bias = not use_batchnorm
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias)
        )
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetDecoderBlock(nn.Module):
    """Single U-Net upsampling stage combining bilinear upsampling, skip concatenation, and ConvBnReLU."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.conv = ConvBnReLU(
            in_channels=in_channels + skip_channels,
            out_channels=out_channels,
            use_batchnorm=use_batchnorm,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor | None = None,
        target_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if skip is not None:
            spatial_size = (int(skip.shape[-2]), int(skip.shape[-1]))
        elif target_size is not None:
            spatial_size = (int(target_size[0]), int(target_size[1]))
        else:
            spatial_size = (int(x.shape[-2]) * 2, int(x.shape[-1]) * 2)

        upsampled = F.interpolate(
            x, size=spatial_size, mode="bilinear", align_corners=False
        )
        if skip is not None:
            upsampled = torch.cat([upsampled, skip], dim=1)
        return self.conv(upsampled)


class ResNet50UNet(nn.Module):
    """U-Net with a ResNet-50 encoder producing 4-channel ``[MA, HE, EX, SE]`` logits."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 4,
        pretrained: bool = False,
        strict_pretrained: bool = True,
        encoder_weights_source: str = "imagenet",
        pretrained_checkpoint_path: str | Path | None = None,
        decoder_channels: Sequence[int] = DEFAULT_DECODER_CHANNELS,
        use_batchnorm: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}.")
        if out_channels != len(CANONICAL_LESION_CHANNELS):
            raise ValueError(
                f"out_channels must be {len(CANONICAL_LESION_CHANNELS)} for "
                f"canonical channels {CANONICAL_LESION_CHANNELS}, got {out_channels}."
            )
        if len(decoder_channels) != 5 or any(int(c) < 1 for c in decoder_channels):
            raise ValueError(
                f"decoder_channels must be a 5-element sequence of positive integers, "
                f"got {decoder_channels!r}."
            )
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}.")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.channel_order: tuple[str, ...] = CANONICAL_LESION_CHANNELS
        self.decoder_channels = tuple(int(c) for c in decoder_channels)
        self.pretrained_requested = bool(pretrained)
        self.strict_pretrained = bool(strict_pretrained)
        self.encoder_weights_source = str(encoder_weights_source).strip().lower()
        self.pretrained_weights_loaded: bool = False
        self.weights_status: str = "random_initialization"
        self.encoder_checkpoint_path: str | None = (
            format_project_path(pretrained_checkpoint_path)
            if pretrained_checkpoint_path is not None
            else None
        )
        self.loaded_encoder_parameter_count: int = 0

        backbone = self._init_resnet50_backbone(
            pretrained=self.pretrained_requested,
            strict_pretrained=self.strict_pretrained,
            weights_source=self.encoder_weights_source,
            checkpoint_path=pretrained_checkpoint_path,
        )

        self.encoder_conv1 = backbone.conv1
        self.encoder_bn1 = backbone.bn1
        self.encoder_relu = backbone.relu
        self.encoder_maxpool = backbone.maxpool
        self.encoder_layer1 = backbone.layer1  # 256 ch, H/4
        self.encoder_layer2 = backbone.layer2  # 512 ch, H/8
        self.encoder_layer3 = backbone.layer3  # 1024 ch, H/16
        self.encoder_layer4 = backbone.layer4  # 2048 ch, H/32

        d4, d3, d2, d1, d0 = self.decoder_channels
        self.decoder4 = UNetDecoderBlock(
            in_channels=2048,
            skip_channels=1024,
            out_channels=d4,
            use_batchnorm=use_batchnorm,
        )
        self.decoder3 = UNetDecoderBlock(
            in_channels=d4,
            skip_channels=512,
            out_channels=d3,
            use_batchnorm=use_batchnorm,
        )
        self.decoder2 = UNetDecoderBlock(
            in_channels=d3,
            skip_channels=256,
            out_channels=d2,
            use_batchnorm=use_batchnorm,
        )
        self.decoder1 = UNetDecoderBlock(
            in_channels=d2,
            skip_channels=64,
            out_channels=d1,
            use_batchnorm=use_batchnorm,
        )
        self.decoder0 = UNetDecoderBlock(
            in_channels=d1,
            skip_channels=0,
            out_channels=d0,
            use_batchnorm=use_batchnorm,
        )

        self.dropout = nn.Dropout2d(p=float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        self.segmentation_head = nn.Conv2d(d0, self.out_channels, kernel_size=1)
        # Initialize 1x1 segmentation head with sparse foreground prior (pi = 0.01)
        # so initial background probabilities start near 0.01 instead of 0.50.
        prior_prob = 0.01
        nn.init.normal_(self.segmentation_head.weight, mean=0.0, std=0.01)
        if self.segmentation_head.bias is not None:
            nn.init.constant_(
                self.segmentation_head.bias,
                -math.log((1.0 - prior_prob) / prior_prob),
            )

    @staticmethod
    def _normalize_encoder_state_dict(raw_state_dict: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Normalize ResNet-50 encoder keys from SimCLR, U-Net, or TorchVision checkpoints."""
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

    @staticmethod
    def _extract_state_dict_from_checkpoint(
        raw_ckpt: Any,
        ckpt_path: Path,
    ) -> Mapping[str, Any]:
        """Extract a non-empty state_dict mapping from a loaded checkpoint dictionary."""
        if not isinstance(raw_ckpt, Mapping):
            raise ValueError(
                f"Expected checkpoint mapping in '{ckpt_path}', got {type(raw_ckpt).__name__}."
            )
        if len(raw_ckpt) == 0:
            raise ValueError(f"Checkpoint '{ckpt_path}' is an empty dictionary.")

        for key in ("encoder_state_dict", "model_state_dict", "state_dict"):
            if key in raw_ckpt and raw_ckpt[key] is not None:
                candidate = raw_ckpt[key]
                if not isinstance(candidate, Mapping):
                    raise ValueError(
                        f"Checkpoint '{ckpt_path}' field '{key}' must be a mapping, "
                        f"got {type(candidate).__name__}."
                    )
                if len(candidate) == 0:
                    raise ValueError(
                        f"Checkpoint '{ckpt_path}' contains an empty '{key}' dictionary."
                    )
                return candidate

        return raw_ckpt

    def _init_resnet50_backbone(
        self,
        pretrained: bool,
        strict_pretrained: bool,
        weights_source: str,
        checkpoint_path: str | Path | None,
    ) -> nn.Module:
        """Instantiate the ResNet-50 encoder and load pretrained weights transparently."""
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

        if not pretrained or weights_source == "random":
            self.pretrained_weights_loaded = False
            self.weights_status = "random_initialization"
            self.loaded_encoder_parameter_count = 0
            return backbone

        if checkpoint_path is not None or weights_source in {
            "ssl_checkpoint",
            "supervised_checkpoint",
        }:
            if checkpoint_path is None:
                raise PretrainedWeightsError(
                    f"encoder_weights_source={weights_source!r} requires a valid pretrained_checkpoint_path."
                )
            ckpt_p = resolve_project_path(checkpoint_path)
            self.encoder_checkpoint_path = format_project_path(ckpt_p)
            if not ckpt_p.exists():
                msg = f"Pretrained encoder checkpoint not found at '{ckpt_p}'."
                if strict_pretrained:
                    raise PretrainedWeightsError(msg)
                logger.warning("%s Falling back to random initialization.", msg)
                self.pretrained_weights_loaded = False
                self.weights_status = f"fallback_random_missing_checkpoint:{ckpt_p.name}"
                return backbone

            try:
                raw_ckpt = torch.load(ckpt_p, map_location="cpu", weights_only=False)
                state_dict = self._extract_state_dict_from_checkpoint(raw_ckpt, ckpt_p)
                cleaned_sd = self._normalize_encoder_state_dict(state_dict)
                if not cleaned_sd:
                    raise ValueError(
                        f"No compatible ResNet-50 encoder parameters found in '{ckpt_p}'."
                    )

                expected_encoder_keys = {
                    k
                    for k in backbone.state_dict().keys()
                    if not k.startswith("fc.")
                }
                missing_encoder_keys = sorted(expected_encoder_keys - set(cleaned_sd.keys()))
                if missing_encoder_keys:
                    raise ValueError(
                        f"Checkpoint '{ckpt_p}' is missing {len(missing_encoder_keys)} expected "
                        f"ResNet-50 encoder keys (e.g. {missing_encoder_keys[:5]})."
                    )

                missing, unexpected = backbone.load_state_dict(cleaned_sd, strict=False)
                non_fc_missing = [k for k in missing if not k.startswith("fc.")]
                if non_fc_missing or unexpected:
                    raise ValueError(
                        f"Encoder state_dict mismatch when loading '{ckpt_p}': "
                        f"missing={non_fc_missing[:5]}, unexpected={list(unexpected)[:5]}"
                    )
                self.pretrained_weights_loaded = True
                self.loaded_encoder_parameter_count = len(cleaned_sd)
                self.weights_status = f"loaded_checkpoint:{ckpt_p.name}"
                return backbone
            except Exception as exc:
                if strict_pretrained:
                    raise PretrainedWeightsError(
                        f"Failed to load pretrained checkpoint '{ckpt_p}': {exc}"
                    ) from exc
                logger.warning(
                    "Failed to load checkpoint '%s' (%s); falling back to random initialization.",
                    ckpt_p,
                    exc,
                )
                self.pretrained_weights_loaded = False
                self.loaded_encoder_parameter_count = 0
                self.weights_status = f"fallback_random_checkpoint_error:{type(exc).__name__}"
                return backbone

        # weights_source == "imagenet"
        try:
            # First check if a cached ResNet-50 weight file exists in torch.hub checkpoints
            hub_ckpt_dir = Path(torch.hub.get_dir()) / "checkpoints"
            cached_candidates = (
                sorted(hub_ckpt_dir.glob("resnet50*.pth"))
                if hub_ckpt_dir.exists()
                else []
            )
            if cached_candidates:
                sd = torch.load(
                    cached_candidates[0], map_location="cpu", weights_only=True
                )
                if self.in_channels != 3:
                    cleaned_sd = self._normalize_encoder_state_dict(sd)
                    cleaned_sd.pop("conv1.weight", None)
                    cleaned_sd.pop("conv1.bias", None)
                    backbone.load_state_dict(cleaned_sd, strict=False)
                    self.loaded_encoder_parameter_count = len(cleaned_sd)
                else:
                    backbone.load_state_dict(sd, strict=True)
                    self.loaded_encoder_parameter_count = len(
                        [k for k in sd.keys() if not str(k).startswith("fc.")]
                    )
                self.pretrained_weights_loaded = True
                self.weights_status = f"loaded_imagenet_cached:{cached_candidates[0].name}"
                return backbone

            # Attempt standard TorchVision weights load
            weights_enum = ResNet50_Weights.DEFAULT
            sd = weights_enum.get_state_dict(progress=False, check_hash=True)
            if self.in_channels != 3:
                cleaned_sd = self._normalize_encoder_state_dict(sd)
                cleaned_sd.pop("conv1.weight", None)
                cleaned_sd.pop("conv1.bias", None)
                backbone.load_state_dict(cleaned_sd, strict=False)
                self.loaded_encoder_parameter_count = len(cleaned_sd)
            else:
                backbone.load_state_dict(sd, strict=True)
                self.loaded_encoder_parameter_count = len(
                    [k for k in sd.keys() if not str(k).startswith("fc.")]
                )
            self.pretrained_weights_loaded = True
            self.weights_status = "loaded_imagenet_torchvision"
            return backbone
        except Exception as exc:
            msg = (
                f"Unable to load ImageNet pretrained ResNet-50 weights "
                f"(no cached weights in '{Path(torch.hub.get_dir()) / 'checkpoints'}' "
                f"and download failed: {exc})."
            )
            if strict_pretrained:
                raise PretrainedWeightsError(msg) from exc
            logger.warning("%s Falling back to random initialization (pretrained_weights_loaded=False).", msg)
            self.pretrained_weights_loaded = False
            self.loaded_encoder_parameter_count = 0
            self.weights_status = f"fallback_random_imagenet_unavailable:{type(exc).__name__}"
            return backbone

    def export_encoder_state_dict(self) -> dict[str, torch.Tensor]:
        """Export the ResNet-50 encoder parameters using canonical ResNet-50 key names."""
        raw_unet_sd = self.state_dict()
        return {
            k: v.detach().clone()
            for k, v in self._normalize_encoder_state_dict(raw_unet_sd).items()
        }

    def load_encoder_from_checkpoint(
        self,
        checkpoint_path: str | Path,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Load ResNet-50 encoder weights from a SimCLR or U-Net checkpoint into this model's encoder."""
        ckpt_p = resolve_project_path(checkpoint_path)
        if not ckpt_p.is_file():
            raise FileNotFoundError(f"Encoder checkpoint file not found: {ckpt_p}")

        raw_ckpt = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        state_dict = self._extract_state_dict_from_checkpoint(raw_ckpt, ckpt_p)

        cleaned_sd = self._normalize_encoder_state_dict(state_dict)
        if not cleaned_sd:
            raise ValueError(
                f"No compatible ResNet-50 encoder parameters found in '{ckpt_p}'."
            )

        current_enc_keys = set(self.export_encoder_state_dict().keys())
        missing_keys = sorted(current_enc_keys - set(cleaned_sd.keys()))
        unexpected_keys = sorted(set(cleaned_sd.keys()) - current_enc_keys)
        if strict and (missing_keys or unexpected_keys):
            raise ValueError(
                f"Encoder state_dict mismatch for '{ckpt_p}': "
                f"missing={missing_keys[:5]}, unexpected={unexpected_keys[:5]}"
            )

        unet_mapped_sd: dict[str, torch.Tensor] = {}
        for k, v in cleaned_sd.items():
            if k.startswith("conv1."):
                unet_mapped_sd["encoder_conv1." + k.removeprefix("conv1.")] = v
            elif k.startswith("bn1."):
                unet_mapped_sd["encoder_bn1." + k.removeprefix("bn1.")] = v
            elif k.startswith("layer1."):
                unet_mapped_sd["encoder_layer1." + k.removeprefix("layer1.")] = v
            elif k.startswith("layer2."):
                unet_mapped_sd["encoder_layer2." + k.removeprefix("layer2.")] = v
            elif k.startswith("layer3."):
                unet_mapped_sd["encoder_layer3." + k.removeprefix("layer3.")] = v
            elif k.startswith("layer4."):
                unet_mapped_sd["encoder_layer4." + k.removeprefix("layer4.")] = v

        self.load_state_dict(unet_mapped_sd, strict=False)
        self.encoder_weights_source = "ssl_checkpoint"
        self.encoder_checkpoint_path = format_project_path(ckpt_p)
        self.pretrained_weights_loaded = True
        self.loaded_encoder_parameter_count = len(unet_mapped_sd)
        self.weights_status = f"loaded_checkpoint:{ckpt_p.name}"
        return {
            "checkpoint_path": format_project_path(ckpt_p),
            "loaded_parameter_count": len(unet_mapped_sd),
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "weights_status": self.weights_status,
        }

    def forward_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract multi-scale ResNet-50 encoder feature maps ``(s0, s1, s2, s3, bottleneck)``."""
        s0 = self.encoder_relu(self.encoder_bn1(self.encoder_conv1(x)))  # (B, 64, H/2, W/2)
        p0 = self.encoder_maxpool(s0)  # (B, 64, H/4, W/4)
        s1 = self.encoder_layer1(p0)  # (B, 256, H/4, W/4)
        s2 = self.encoder_layer2(s1)  # (B, 512, H/8, W/8)
        s3 = self.encoder_layer3(s2)  # (B, 1024, H/16, W/16)
        bottleneck = self.encoder_layer4(s3)  # (B, 2048, H/32, W/32)
        return s0, s1, s2, s3, bottleneck

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw segmentation logits of shape ``(B, 4, H, W)`` in ``[MA, HE, EX, SE]`` order."""
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError(
                f"Expected 4D input tensor (B, C, H, W), got {getattr(x, 'shape', type(x))}."
            )
        b, c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {c} channels (shape={tuple(x.shape)})."
            )
        if h < 32 or w < 32:
            raise ValueError(
                f"Input spatial dimensions (H={h}, W={w}) must be at least 32x32 for 5-stage ResNet-50 downsampling."
            )
        if h % 32 != 0 or w % 32 != 0:
            raise ValueError(
                f"Input spatial dimensions (H={h}, W={w}) must be divisible by 32 for 5-stage ResNet-50 downsampling."
            )

        s0, s1, s2, s3, bottleneck = self.forward_features(x)
        d4 = self.decoder4(bottleneck, skip=s3)
        d3 = self.decoder3(d4, skip=s2)
        d2 = self.decoder2(d3, skip=s1)
        d1 = self.decoder1(d2, skip=s0)
        d0 = self.decoder0(d1, skip=None, target_size=(h, w))
        logits = self.segmentation_head(self.dropout(d0))
        return logits

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Return independent per-channel lesion probabilities ``torch.sigmoid(self.forward(x))``."""
        return torch.sigmoid(self.forward(x))

    def predict_masks(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return thresholded binary lesion predictions ``(B, 4, H, W)`` in ``{0.0, 1.0}``."""
        probs = self.predict_probabilities(x)
        return (probs >= float(threshold)).to(dtype=torch.float32)

    def get_model_summary(self) -> dict[str, Any]:
        """Return a structured dictionary of architecture and parameter metadata."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(
            p.numel()
            for mod in (
                self.encoder_conv1,
                self.encoder_bn1,
                self.encoder_layer1,
                self.encoder_layer2,
                self.encoder_layer3,
                self.encoder_layer4,
            )
            for p in mod.parameters()
        )
        return {
            "architecture": "UNet",
            "encoder": "resnet50",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "output_channel_order": list(self.channel_order),
            "decoder_channels": list(self.decoder_channels),
            "encoder_weights_source": self.encoder_weights_source,
            "encoder_checkpoint_path": self.encoder_checkpoint_path,
            "pretrained_requested": self.pretrained_requested,
            "pretrained_weights_loaded": self.pretrained_weights_loaded,
            "loaded_encoder_parameter_count": self.loaded_encoder_parameter_count,
            "weights_status": self.weights_status,
            "encoder_parameters": int(encoder_params),
            "decoder_and_head_parameters": int(total_params - encoder_params),
            "total_parameters": int(total_params),
            "trainable_parameters": int(trainable_params),
        }


def build_unet_from_config(
    model_config: Mapping[str, Any] | str | Path = "model.yaml",
    pretrained_override: bool | None = None,
    strict_pretrained_override: bool | None = None,
) -> ResNet50UNet:
    """Instantiate a :class:`ResNet50UNet` from ``configs/model.yaml``."""
    if isinstance(model_config, (str, Path)):
        cfg = load_yaml_config(model_config, validate=True)
    else:
        cfg = dict(model_config)

    arch = str(cfg.get("architecture", "UNet")).strip()
    enc = str(cfg.get("encoder", "resnet50")).strip().lower()
    if arch.lower() != "unet":
        raise ValueError(f"Unsupported architecture {arch!r}; expected 'UNet'.")
    if enc != "resnet50":
        raise ValueError(f"Unsupported encoder {enc!r}; expected 'resnet50'.")

    in_ch = int(cfg.get("input_channels", 3))
    out_ch = int(cfg.get("output_channels", cfg.get("number_of_classes", 4)))
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
    weights_src = str(cfg.get("encoder_weights_source", "imagenet"))
    ssl_ckpt = cfg.get("ssl_checkpoint_path") or cfg.get("pretrained_checkpoint_path")

    dec_cfg = cfg.get("decoder", {})
    dec_channels = dec_cfg.get("channels", list(DEFAULT_DECODER_CHANNELS))
    use_bn = bool(dec_cfg.get("use_batchnorm", True))
    dropout = float(dec_cfg.get("dropout", 0.1))

    return ResNet50UNet(
        in_channels=in_ch,
        out_channels=out_ch,
        pretrained=pretrained,
        strict_pretrained=strict_pretrained,
        encoder_weights_source=weights_src,
        pretrained_checkpoint_path=(ssl_ckpt if weights_src == "ssl_checkpoint" else None),
        decoder_channels=dec_channels,
        use_batchnorm=use_bn,
        dropout=dropout,
    )


__all__ = [
    "CANONICAL_LESION_ORDER",
    "ConvBnReLU",
    "DEFAULT_DECODER_CHANNELS",
    "PretrainedWeightsError",
    "ResNet50UNet",
    "UNetDecoderBlock",
    "build_unet_from_config",
]
