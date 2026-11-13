"""Unit tests for FedMorphis inference pipeline and demonstration endpoints."""

from __future__ import annotations

import asyncio
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from src.engine.inference import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_FROZEN_THRESHOLDS,
    CANONICAL_LESION_ORDER,
    InferenceResult,
    LesionSegmentationPipeline,
)
from scripts.run_demo import (
    index_page,
    get_sample_list,
    get_training_history_endpoint,
    get_benchmark_endpoint,
    run_predict,
)


@pytest.fixture(scope="module")
def pipeline():
    """Shared pipeline instance on CPU for fast deterministic testing."""
    return LesionSegmentationPipeline(
        checkpoint_path=DEFAULT_CHECKPOINT_PATH,
        thresholds=DEFAULT_FROZEN_THRESHOLDS,
        device="cpu",
    )


def test_pipeline_initialization(pipeline):
    """Verify that model is loaded and thresholds match frozen specifications."""
    assert pipeline.model is not None
    assert pipeline.thresholds == DEFAULT_FROZEN_THRESHOLDS
    assert pipeline.thresholds["MA"] == 0.25
    assert pipeline.thresholds["HE"] == 0.40
    assert pipeline.thresholds["EX"] == 0.35
    assert pipeline.thresholds["SE"] == 0.35


def test_input_validation_valid_formats(pipeline):
    """Verify image loading from numpy array, PIL Image, and dimensions."""
    # 1. Standard RGB array
    arr = np.zeros((100, 120, 3), dtype=np.uint8)
    loaded = pipeline.validate_and_load_image(arr)
    assert loaded.shape == (100, 120, 3)
    assert loaded.dtype == np.uint8

    # 2. Grayscale 2D array
    gray = np.zeros((80, 90), dtype=np.uint8)
    loaded_gray = pipeline.validate_and_load_image(gray)
    assert loaded_gray.shape == (80, 90, 3)

    # 3. PIL Image
    pil = Image.new("RGB", (64, 64), color=(100, 150, 200))
    loaded_pil = pipeline.validate_and_load_image(pil)
    assert loaded_pil.shape == (64, 64, 3)


def test_input_validation_rejects_invalid(pipeline):
    """Verify that pipeline rejects tiny or malformed inputs with descriptive errors."""
    tiny_arr = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="dimensions too small"):
        pipeline.validate_and_load_image(tiny_arr)

    with pytest.raises(FileNotFoundError):
        pipeline.validate_and_load_image("non_existent_retinal_image.jpg")


def test_predict_produces_four_channels_and_preserves_dimensions(pipeline):
    """Verify 4-channel canonical output and preservation of original dimensions."""
    h_orig, w_orig = 128, 160
    synthetic_fundus = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
    # Draw simple circular pattern simulating retinal disc
    synthetic_fundus[30:90, 40:120] = [180, 80, 30]

    result = pipeline.predict(synthetic_fundus, sample_name="synthetic_test")

    assert isinstance(result, InferenceResult)
    assert result.original_dimensions == (h_orig, w_orig)
    assert result.original_image_rgb.shape == (h_orig, w_orig, 3)
    assert result.combined_overlay_orig.shape == (h_orig, w_orig, 3)

    # Verify 4 channels in canonical order
    assert tuple(result.channels.keys()) == CANONICAL_LESION_ORDER
    assert result.binary_masks_orig.shape == (4, h_orig, w_orig)
    assert result.probabilities_orig.shape == (4, h_orig, w_orig)

    for ch in CANONICAL_LESION_ORDER:
        cp = result.channels[ch]
        assert cp.mask_orig.shape == (h_orig, w_orig)
        assert cp.colored_mask_orig.shape == (h_orig, w_orig, 3)
        assert cp.threshold == DEFAULT_FROZEN_THRESHOLDS[ch]
        assert cp.positive_pixel_count == int(np.sum(result.binary_masks_orig[CANONICAL_LESION_ORDER.index(ch)]))


def test_create_combined_overlay_rendering():
    """Verify overlay blending logic."""
    img = np.full((50, 50, 3), 100, dtype=np.uint8)
    masks = np.zeros((4, 50, 50), dtype=np.uint8)
    masks[0, 10:20, 10:20] = 1  # MA lesion

    overlay = LesionSegmentationPipeline.create_combined_overlay(img, masks)
    assert overlay.shape == (50, 50, 3)
    assert overlay.dtype == np.uint8
    # Pixels inside lesion should have modified color
    assert not np.array_equal(overlay[15, 15], img[15, 15])
    # Pixels outside should remain unchanged
    assert np.array_equal(overlay[0, 0], img[0, 0])


def test_demo_route_handlers():
    """Verify demonstration route handler functions return expected data structures."""
    # 1. Main UI page
    html = index_page()
    assert "FedMorphis" in html
    assert "Flower 1.43.0" in html
    assert "Demonstration Disclaimer" in html

    # 2. Sample list
    samples_data = get_sample_list()
    assert "samples" in samples_data
    assert "IDRiD_Sample_1" in samples_data["samples"]
    assert "DDR_Sample_1" in samples_data["samples"]

    # 3. Training history
    hist = get_training_history_endpoint()
    assert hist.get("best_round") == 17
    assert len(hist.get("per_round_val_metrics", [])) == 20

    # 4. Benchmark data
    bench = get_benchmark_endpoint()
    assert bench.get("total_test_samples") == 252
    assert "channel_tuned" in bench
    assert bench["channel_tuned"]["sample_weighted"]["macro"]["dice"] == pytest.approx(0.4760, abs=1e-3)


@pytest.mark.anyio
async def test_demo_predict_route_sample():
    """Verify predict endpoint handles sample selection cleanly."""
    res = await run_predict(sample_key="DDR_Sample_1", file=None)
    assert res["status"] == "success"
    assert "images" in res
    assert "original_b64" in res["images"]
    assert "overlay_b64" in res["images"]
    assert "channels" in res
    assert "MA" in res["channels"]
    assert "HE" in res["channels"]
