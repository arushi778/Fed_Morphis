"""Production inference workflow for FedMorphis retinal lesion segmentation.

Provides deterministic, self-contained end-to-end inference from retinal fundus
images to multi-channel lesion segmentation masks and composite overlays using
the selected top-performing Flower FedAvg global model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.data.masks import CANONICAL_LESION_CHANNELS
from src.engine.evaluator import reconstruct_model_from_checkpoint
from src.utils.device import resolve_device
from src.utils.logger import get_logger
from src.utils.paths import format_project_path, resolve_project_path

logger = get_logger("fedmorphis.inference")

CANONICAL_LESION_ORDER: tuple[str, ...] = tuple(CANONICAL_LESION_CHANNELS)

DEFAULT_CHECKPOINT_PATH: str = "checkpoints/federated/fedavg_full_20rounds/best_checkpoint.pt"

# Frozen optimal thresholds established in Task 14 and verified in Task 15
DEFAULT_FROZEN_THRESHOLDS: dict[str, float] = {
    "MA": 0.25,
    "HE": 0.40,
    "EX": 0.35,
    "SE": 0.35,
}

CANONICAL_LESION_NAMES: dict[str, str] = {
    "MA": "Microaneurysms",
    "HE": "Hemorrhages",
    "EX": "Hard Exudates",
    "SE": "Soft Exudates",
}

CANONICAL_LESION_COLORS: dict[str, tuple[int, int, int]] = {
    "MA": (255, 50, 50),     # Vivid Red
    "HE": (50, 205, 50),     # Lime Green
    "EX": (255, 215, 0),     # Gold / Bright Yellow
    "SE": (0, 206, 209),     # Dark Turquoise / Vivid Cyan
}

DEFAULT_IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
DEFAULT_IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class ChannelPrediction:
    """Per-lesion channel prediction outputs and statistics."""

    channel_code: str
    channel_name: str
    threshold: float
    color_rgb: tuple[int, int, int]
    positive_pixel_count: int
    total_pixel_count: int
    area_percentage: float
    detected: bool
    mask_orig: np.ndarray  # (H, W) uint8 {0, 255}
    colored_mask_orig: np.ndarray  # (H, W, 3) uint8


@dataclass
class InferenceResult:
    """Complete structured output from a single image inference execution."""

    original_image_rgb: np.ndarray  # (H, W, 3) uint8
    original_dimensions: tuple[int, int]  # (H, W)
    preprocessed_image_rgb: np.ndarray  # (512, 512, 3) uint8
    probabilities_512: torch.Tensor  # (4, 512, 512) float32
    probabilities_orig: np.ndarray  # (4, H, W) float32
    binary_masks_orig: np.ndarray  # (4, H, W) uint8 {0, 1}
    combined_overlay_orig: np.ndarray  # (H, W, 3) uint8
    channels: dict[str, ChannelPrediction]
    latency_ms: float
    metadata: dict[str, Any]

    def summary_dict(self) -> dict[str, Any]:
        """Convert prediction statistics and metadata to JSON-serializable dictionary."""
        return {
            "metadata": self.metadata,
            "original_dimensions": {
                "height": self.original_dimensions[0],
                "width": self.original_dimensions[1],
            },
            "latency_ms": round(self.latency_ms, 2),
            "channels": {
                ch: {
                    "channel_name": cp.channel_name,
                    "threshold_applied": cp.threshold,
                    "positive_pixels": cp.positive_pixel_count,
                    "total_pixels": cp.total_pixel_count,
                    "area_percentage": round(cp.area_percentage, 4),
                    "detected": cp.detected,
                }
                for ch, cp in self.channels.items()
            },
        }


class LesionSegmentationPipeline:
    """End-to-end inference pipeline for multi-lesion retinal fundus segmentation.

    Loads the frozen federated ResNet-50 U-Net model and executes deterministic,
    reproducible segmentation with channel-specific probability thresholding.
    """

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        thresholds: Mapping[str, float] | None = None,
        device: torch.device | str = "auto",
        image_size: tuple[int, int] = (512, 512),
    ) -> None:
        self.device = resolve_device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_path = resolve_project_path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {self.checkpoint_path}")

        self.thresholds = dict(DEFAULT_FROZEN_THRESHOLDS)
        if thresholds is not None:
            for ch, val in thresholds.items():
                if ch in self.thresholds:
                    if not (0.0 < float(val) < 1.0):
                        raise ValueError(f"Threshold for {ch} must be strictly between 0 and 1, got {val}")
                    self.thresholds[ch] = float(val)

        self.image_size = tuple(image_size)
        logger.info(
            "Loading LesionSegmentationPipeline: checkpoint=%s, device=%s, thresholds=%s",
            format_project_path(self.checkpoint_path),
            self.device,
            self.thresholds,
        )

        self.model, self.checkpoint_dict, self.resolved_configs = reconstruct_model_from_checkpoint(
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            strict=True,
        )
        self.model.eval()

        # Cache normalization parameters
        self.mean_tensor = torch.tensor(DEFAULT_IMAGENET_MEAN, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std_tensor = torch.tensor(DEFAULT_IMAGENET_STD, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    @staticmethod
    def validate_and_load_image(
        image_input: str | Path | bytes | Image.Image | np.ndarray,
    ) -> np.ndarray:
        """Validate and convert input into a canonical (H, W, 3) uint8 RGB NumPy array."""
        if isinstance(image_input, (str, Path)):
            p = resolve_project_path(image_input)
            if not p.is_file():
                raise FileNotFoundError(f"Image file does not exist: {p}")
            pil_img = Image.open(p)
            rgb_img = pil_img.convert("RGB")
            arr = np.array(rgb_img, dtype=np.uint8)
        elif isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input))
            rgb_img = pil_img.convert("RGB")
            arr = np.array(rgb_img, dtype=np.uint8)
        elif isinstance(image_input, Image.Image):
            rgb_img = image_input.convert("RGB")
            arr = np.array(rgb_img, dtype=np.uint8)
        elif isinstance(image_input, np.ndarray):
            arr = np.ascontiguousarray(image_input)
            if arr.ndim == 2:
                # Grayscale to RGB
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif arr.ndim == 3 and arr.shape[2] == 1:
                arr = cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2RGB)
            elif arr.ndim == 3 and arr.shape[2] == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
            elif arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(f"Invalid image array shape: {arr.shape}; expected (H, W, 3).")
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.clip(0, 255).astype(np.uint8)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input).__name__}")

        h, w, _ = arr.shape
        if h < 32 or w < 32:
            raise ValueError(f"Image dimensions too small ({h}x{w}); minimum supported resolution is 32x32.")

        return arr

    def preprocess(
        self,
        image_rgb: np.ndarray,
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Resize image to model resolution and apply ImageNet standardization.

        Returns:
            - Normalized torch.Tensor of shape (1, 3, 512, 512) on device.
            - Resized uint8 RGB image array of shape (512, 512, 3).
        """
        target_h, target_w = self.image_size
        resized_rgb = cv2.resize(image_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Convert to float tensor [0, 1]
        t = torch.from_numpy(resized_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device).float() / 255.0
        normalized = (t - self.mean_tensor) / self.std_tensor
        return normalized, resized_rgb

    @torch.no_grad()
    def predict(
        self,
        image_input: str | Path | bytes | Image.Image | np.ndarray,
        sample_name: str | None = None,
    ) -> InferenceResult:
        """Run complete end-to-end segmentation inference on a retinal fundus image."""
        start_time = time.perf_counter()

        orig_rgb = self.validate_and_load_image(image_input)
        h_orig, w_orig, _ = orig_rgb.shape

        norm_tensor, resized_rgb = self.preprocess(orig_rgb)

        use_amp = self.device.type == "cuda"
        with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
            logits = self.model(norm_tensor)
            probs_512 = torch.sigmoid(logits.float()).squeeze(0)  # (4, 512, 512)

        probs_512_cpu = probs_512.cpu()

        # Bilinear interpolation for smooth probabilities at original resolution
        probs_4d = probs_512.unsqueeze(0)
        probs_orig_t = F.interpolate(
            probs_4d,
            size=(h_orig, w_orig),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        probs_orig = probs_orig_t.cpu().numpy()

        # Binary masks at original resolution using nearest-neighbor/thresholding
        binary_masks_orig = np.zeros((4, h_orig, w_orig), dtype=np.uint8)
        channel_predictions: dict[str, ChannelPrediction] = {}

        total_pixels = h_orig * w_orig

        for ch_idx, ch_code in enumerate(CANONICAL_LESION_ORDER):
            th = self.thresholds[ch_code]
            mask_bin = (probs_orig[ch_idx] >= th).astype(np.uint8)
            binary_masks_orig[ch_idx] = mask_bin

            pos_pixels = int(np.sum(mask_bin))
            area_pct = (float(pos_pixels) / float(total_pixels)) * 100.0 if total_pixels > 0 else 0.0

            # Render colored channel mask on black canvas
            ch_colored = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
            ch_color = CANONICAL_LESION_COLORS[ch_code]
            ch_colored[mask_bin > 0] = ch_color

            channel_predictions[ch_code] = ChannelPrediction(
                channel_code=ch_code,
                channel_name=CANONICAL_LESION_NAMES[ch_code],
                threshold=th,
                color_rgb=ch_color,
                positive_pixel_count=pos_pixels,
                total_pixel_count=total_pixels,
                area_percentage=area_pct,
                detected=pos_pixels > 0,
                mask_orig=(mask_bin * 255).astype(np.uint8),
                colored_mask_orig=ch_colored,
            )

        # Generate combined overlay on original fundus image
        overlay_orig = self.create_combined_overlay(orig_rgb, binary_masks_orig)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        metadata = {
            "model_architecture": "ResNet50UNet",
            "checkpoint_name": self.checkpoint_path.name,
            "checkpoint_path": format_project_path(self.checkpoint_path),
            "federated_round": self.checkpoint_dict.get("round", 17),
            "sample_name": sample_name or getattr(image_input, "name", "input_image"),
            "device": str(self.device),
            "thresholds_applied": self.thresholds,
            "disclaimer": (
                "Academic demonstration prototype for research purposes only. "
                "Not clinically validated or approved for diagnostic use."
            ),
        }

        return InferenceResult(
            original_image_rgb=orig_rgb,
            original_dimensions=(h_orig, w_orig),
            preprocessed_image_rgb=resized_rgb,
            probabilities_512=probs_512_cpu,
            probabilities_orig=probs_orig,
            binary_masks_orig=binary_masks_orig,
            combined_overlay_orig=overlay_orig,
            channels=channel_predictions,
            latency_ms=latency_ms,
            metadata=metadata,
        )

    @staticmethod
    def create_combined_overlay(
        image_rgb: np.ndarray,
        binary_masks_4ch: np.ndarray,
        alpha: float = 0.35,
        contour_thickness: int = 2,
    ) -> np.ndarray:
        """Create a composite image with semi-transparent fills and solid contour outlines."""
        h, w, _ = image_rgb.shape
        base = image_rgb.copy()

        # Step 1: Tint layer
        tint_layer = np.zeros_like(base)
        active_any = np.zeros((h, w), dtype=bool)

        for ch_idx, ch_code in enumerate(CANONICAL_LESION_ORDER):
            ch_bin = binary_masks_4ch[ch_idx]
            if not np.any(ch_bin):
                continue
            color = np.array(CANONICAL_LESION_COLORS[ch_code], dtype=np.uint8)
            tint_layer[ch_bin > 0] = color
            active_any |= (ch_bin > 0)

        if np.any(active_any):
            blended = cv2.addWeighted(base, 1.0 - alpha, tint_layer, alpha, 0)
            base[active_any] = blended[active_any]

        # Step 2: Solid contour outlines
        for ch_idx, ch_code in enumerate(CANONICAL_LESION_ORDER):
            ch_bin = binary_masks_4ch[ch_idx]
            if not np.any(ch_bin):
                continue
            color = tuple(int(c) for c in CANONICAL_LESION_COLORS[ch_code])
            contours, _ = cv2.findContours(
                ch_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(base, contours, -1, color, contour_thickness)

        return base

    def save_prediction_artifacts(
        self,
        result: InferenceResult,
        output_dir: str | Path,
        base_name: str = "prediction",
    ) -> dict[str, str]:
        """Save inference masks, overlay, and metadata to disk."""
        out_p = resolve_project_path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        paths: dict[str, str] = {}

        # 1. Original RGB
        orig_path = out_p / f"{base_name}_original.png"
        cv2.imwrite(str(orig_path), cv2.cvtColor(result.original_image_rgb, cv2.COLOR_RGB2BGR))
        paths["original"] = format_project_path(orig_path)

        # 2. Combined overlay
        overlay_path = out_p / f"{base_name}_overlay.png"
        cv2.imwrite(str(overlay_path), cv2.cvtColor(result.combined_overlay_orig, cv2.COLOR_RGB2BGR))
        paths["overlay"] = format_project_path(overlay_path)

        # 3. Individual channel masks
        for ch_code, cp in result.channels.items():
            mask_p = out_p / f"{base_name}_mask_{ch_code}.png"
            colored_p = out_p / f"{base_name}_colored_{ch_code}.png"
            cv2.imwrite(str(mask_p), cp.mask_orig)
            cv2.imwrite(str(colored_p), cv2.cvtColor(cp.colored_mask_orig, cv2.COLOR_RGB2BGR))
            paths[f"mask_{ch_code}"] = format_project_path(mask_p)
            paths[f"colored_{ch_code}"] = format_project_path(colored_p)

        # 4. Summary JSON
        summary_p = out_p / f"{base_name}_summary.json"
        with open(summary_p, "w", encoding="utf-8") as f:
            json.dump(result.summary_dict(), f, indent=2)
        paths["summary_json"] = format_project_path(summary_p)

        return paths

