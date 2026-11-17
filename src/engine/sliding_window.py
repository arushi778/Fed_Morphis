"""Full-resolution sliding-window inference with apodized blending.

Provides deterministic, memory-efficient sliding-window inference for high-resolution
retinal fundus images, reconstructing 4-channel continuous probability maps
in canonical ``[MA, HE, EX, SE]`` order without boundary seam artifacts.

Key Features:
1. Slices full FOV images into overlapping patches with configurable overlap (e.g. 50%).
2. Evaluates model in manageable GPU minibatches (e.g. batch_size=4) to prevent VRAM spikes.
3. Blends predictions across overlapping regions using 2D Gaussian or Hann windows.
4. Normalizes accumulated probabilities by total spatial blending weights.
5. Guarantees complete spatial coverage with zero uncovered pixels.
6. Reconstructs exact full-resolution 4-channel probability maps ``(4, H, W)``.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from src.data.patches import PatchCoordinates, compute_patch_grid_coordinates, extract_patch
from src.data.transforms import DEFAULT_IMAGENET_MEAN, DEFAULT_IMAGENET_STD
from src.utils.device import resolve_device
from src.utils.logger import get_logger

logger = get_logger("fedmorphis.engine.sliding_window")


def create_gaussian_window(
    patch_size: int | tuple[int, int],
    sigma_scale: float = 0.125,
    min_weight: float = 1e-4,
) -> np.ndarray:
    """Generate a 2D Gaussian apodization blending window.

    Downweights patch boundaries while preserving high center confidence,
    eliminating rectangular boundary artifacts during overlapping tile stitching.

    Args:
        patch_size: Square patch size or (height, width).
        sigma_scale: Standard deviation scale relative to patch size (default 0.125).
        min_weight: Minimum floor for window weights to prevent zero divisions.

    Returns:
        2D float32 numpy array of shape (patch_height, patch_width).
    """
    if isinstance(patch_size, (int, float)):
        p_h = p_w = int(patch_size)
    else:
        p_h, p_w = int(patch_size[0]), int(patch_size[1])

    y = np.arange(p_h, dtype=np.float32)
    x = np.arange(p_w, dtype=np.float32)
    cy = (p_h - 1.0) / 2.0
    cx = (p_w - 1.0) / 2.0
    sigma_y = max(1.0, p_h * float(sigma_scale))
    sigma_x = max(1.0, p_w * float(sigma_scale))

    dist_y = ((y - cy) / sigma_y) ** 2
    dist_x = ((x - cx) / sigma_x) ** 2

    # Outer addition -> 2D squared distance
    dist_2d = dist_y[:, None] + dist_x[None, :]
    window = np.exp(-0.5 * dist_2d)

    # Normalize max to 1.0 and clamp min
    window = window / np.max(window)
    window = np.clip(window, min_weight, 1.0)
    return window.astype(np.float32)


def create_hann_window(
    patch_size: int | tuple[int, int],
    min_weight: float = 1e-4,
) -> np.ndarray:
    """Generate a 2D Hann blending window.

    Args:
        patch_size: Square patch size or (height, width).
        min_weight: Floor for boundary weights.

    Returns:
        2D float32 numpy array of shape (patch_height, patch_width).
    """
    if isinstance(patch_size, (int, float)):
        p_h = p_w = int(patch_size)
    else:
        p_h, p_w = int(patch_size[0]), int(patch_size[1])

    w_y = np.hanning(p_h).astype(np.float32)
    w_x = np.hanning(p_w).astype(np.float32)
    window = np.outer(w_y, w_x)
    window = np.clip(window, min_weight, 1.0)
    return (window / np.max(window)).astype(np.float32)


class SlidingWindowInferencer:
    """Sliding-window inference engine with apodized window blending for high-resolution images.

    Executes overlapping patch inference, blends logits/probabilities with
    Gaussian or Hann window weighting, and reconstructs full-resolution 4-channel
    probability maps matching the original FOV dimensions.
    """

    def __init__(
        self,
        model: nn.Module,
        patch_size: int | tuple[int, int] = 512,
        overlap: float = 0.50,
        stride: int | tuple[int, int] | None = None,
        blending: Literal["gaussian", "hann", "uniform"] = "gaussian",
        batch_size: int = 4,
        device: torch.device | str = "auto",
        mean: Sequence[float] = DEFAULT_IMAGENET_MEAN,
        std: Sequence[float] = DEFAULT_IMAGENET_STD,
    ) -> None:
        self.model = model
        self.device = resolve_device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

        if isinstance(patch_size, (int, float)):
            self.patch_size = (int(patch_size), int(patch_size))
        else:
            self.patch_size = (int(patch_size[0]), int(patch_size[1]))

        self.overlap = float(overlap)
        if not (0.0 <= self.overlap < 1.0):
            raise ValueError(f"overlap must be in [0.0, 1.0), got {overlap}")

        if stride is not None:
            if isinstance(stride, (int, float)):
                self.stride = (int(stride), int(stride))
            else:
                self.stride = (int(stride[0]), int(stride[1]))
        else:
            s_h = max(1, int(self.patch_size[0] * (1.0 - self.overlap)))
            s_w = max(1, int(self.patch_size[1] * (1.0 - self.overlap)))
            self.stride = (s_h, s_w)

        self.blending = str(blending).lower().strip()
        if self.blending == "gaussian":
            self.window_np = create_gaussian_window(self.patch_size)
        elif self.blending == "hann":
            self.window_np = create_hann_window(self.patch_size)
        elif self.blending == "uniform":
            self.window_np = np.ones(self.patch_size, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported blending mode {blending!r}; choose 'gaussian', 'hann', or 'uniform'")

        self.batch_size = max(1, int(batch_size))
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def _normalize_image_rgb(self, image: np.ndarray) -> np.ndarray:
        """Ensure image is float32 normalized by ImageNet statistics."""
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        return (img - self.mean) / self.std

    @torch.inference_mode()
    def predict_fov(
        self,
        image_rgb: np.ndarray,
        return_numpy: bool = True,
    ) -> np.ndarray | torch.Tensor:
        """Run sliding-window inference on a full FOV-cropped RGB image.

        Args:
            image_rgb: RGB image array of shape (H, W, 3).
            return_numpy: If True, returns (4, H, W) float32 numpy array.
                If False, returns (4, H, W) torch.Tensor on self.device.

        Returns:
            Reconstructed 4-channel probability map of shape (4, H, W) in [0.0, 1.0].
        """
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image of shape (H, W, 3), got {image_rgb.shape}")

        orig_h, orig_w = image_rgb.shape[:2]
        p_h, p_w = self.patch_size

        # Handle image dimensions smaller than patch size via padding
        pad_bottom = max(0, p_h - orig_h)
        pad_right = max(0, p_w - orig_w)
        if pad_bottom > 0 or pad_right > 0:
            padded_image = np.pad(
                image_rgb,
                ((0, pad_bottom), (0, pad_right), (0, 0)),
                mode="reflect" if orig_h > 1 and orig_w > 1 else "constant",
            )
        else:
            padded_image = image_rgb

        work_h, work_w = padded_image.shape[:2]

        # Calculate grid coordinates
        grid_coords = compute_patch_grid_coordinates(
            image_height=work_h,
            image_width=work_w,
            patch_size=self.patch_size,
            stride=self.stride,
        )

        norm_image = self._normalize_image_rgb(padded_image)

        # Accumulator buffers (C=4, H, W)
        prob_accum = np.zeros((4, work_h, work_w), dtype=np.float32)
        weight_accum = np.zeros((1, work_h, work_w), dtype=np.float32)

        was_training = bool(self.model.training)
        self.model.eval()
        self.model.to(self.device)

        try:
            window_4ch = np.repeat(self.window_np[None, :, :], 4, axis=0)  # (4, P_H, P_W)
            window_1ch = self.window_np[None, :, :]  # (1, P_H, P_W)

            # Process in minibatches
            for batch_start in range(0, len(grid_coords), self.batch_size):
                batch_slice = grid_coords[batch_start : batch_start + self.batch_size]
                batch_tensors: list[torch.Tensor] = []

                for coords in batch_slice:
                    y1, y2, x1, x2 = coords.y_min, coords.y_max, coords.x_min, coords.x_max
                    patch_norm = norm_image[y1:y2, x1:x2]
                    # (H, W, 3) -> (3, H, W)
                    tensor = torch.from_numpy(patch_norm).permute(2, 0, 1).float()
                    batch_tensors.append(tensor)

                batch_input = torch.stack(batch_tensors, dim=0).to(self.device)  # (B, 3, P_H, P_W)
                logits = self.model(batch_input)  # (B, 4, P_H, P_W)
                probs = torch.sigmoid(logits).cpu().numpy()  # (B, 4, P_H, P_W)

                for i, coords in enumerate(batch_slice):
                    y1, y2, x1, x2 = coords.y_min, coords.y_max, coords.x_min, coords.x_max
                    prob_accum[:, y1:y2, x1:x2] += probs[i] * window_4ch
                    weight_accum[:, y1:y2, x1:x2] += window_1ch

            # Verify no pixel was left uncovered
            min_weight = np.min(weight_accum)
            if min_weight <= 0.0:
                raise RuntimeError(
                    f"Sliding window coverage error: Minimum accumulation weight is {min_weight} <= 0. "
                    "Some pixels were not covered by the sliding window grid."
                )

            # Normalize by accumulated weights
            reconstructed = prob_accum / weight_accum
            reconstructed = np.clip(reconstructed, 0.0, 1.0)

            # Crop back to original dimensions if padding was applied
            final_probs = reconstructed[:, :orig_h, :orig_w]

            if return_numpy:
                return final_probs
            return torch.from_numpy(final_probs).to(self.device)
        finally:
            if was_training:
                self.model.train()

