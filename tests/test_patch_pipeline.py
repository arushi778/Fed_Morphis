"""Comprehensive unit tests for high-resolution patch pipeline and sliding-window inference.

Tests:
1. Patch extraction: dimensions, alignment, 4-channel preservation, binary masks, boundary padding.
2. Lesion-aware sampling: deterministic sampling, rare lesion prioritization, multi-lesion hierarchy, background quota, empty masks.
3. Sliding-window inference: reconstructed dimensions, complete coverage, constant map reconstruction, apodized blending, small image padding.
4. Data leakage protection: validation and test split records rejected from training patch dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.data.patches import (
    CANONICAL_LESION_ORDER,
    DEFAULT_SAMPLING_WEIGHTS,
    LESION_RARITY_PRIORITY,
    PatchCoordinates,
    PatchMetadata,
    PatchRetinalSegmentationDataset,
    categorize_patch,
    compute_patch_grid_coordinates,
    extract_patch,
)
from src.engine.sliding_window import (
    SlidingWindowInferencer,
    create_gaussian_window,
    create_hann_window,
)


# =====================================================================
# 1. Patch Extraction Tests
# =====================================================================

def test_compute_patch_grid_coordinates_coverage():
    """Verify compute_patch_grid_coordinates covers image without gaps and snaps to borders."""
    h, w = 1000, 1500
    patch_size = 512
    stride = 256
    coords = compute_patch_grid_coordinates(h, w, patch_size=patch_size, stride=stride)

    assert len(coords) > 0
    # First patch starts at (0, 0)
    assert coords[0].y_min == 0 and coords[0].x_min == 0

    # Ensure rightmost and bottommost coordinates cover borders
    max_y = max(c.y_max for c in coords)
    max_x = max(c.x_max for c in coords)
    assert max_y == h
    assert max_x == w

    # Verify every coordinate has exact patch size
    for c in coords:
        assert c.height == patch_size
        assert c.width == patch_size


def test_extract_patch_dimensions_and_alignment():
    """Verify image and 4-channel mask patch extraction maintains exact spatial alignment."""
    h, w = 600, 800
    img = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w, 4), dtype=np.uint8)

    # Place a unique marker lesion at (150:180, 200:240) in MA channel
    img[150:180, 200:240, 0] = 200  # Red channel
    mask[150:180, 200:240, 0] = 1   # MA channel

    coords = PatchCoordinates(y_min=100, y_max=612, x_min=100, x_max=612)
    img_patch, mask_patch, was_padded = extract_patch(img, mask, coords, target_size=512)

    assert img_patch.shape == (512, 512, 3)
    assert mask_patch.shape == (512, 512, 4)
    assert was_padded is True  # y_max 612 > h 600 -> required padding

    # Check that lesion coordinates within patch (150-100=50:80, 200-100=100:140) match exactly
    assert np.all(mask_patch[50:80, 100:140, 0] == 1)
    assert np.all(img_patch[50:80, 100:140, 0] == 200)

    # Background in patch outside lesion must be strictly 0
    assert np.count_nonzero(mask_patch[..., 0]) == 30 * 40
    assert np.count_nonzero(mask_patch[..., 1:]) == 0  # Other channels empty


def test_extract_patches_different_sizes_384_and_512():
    """Verify patch extraction behaves correctly for both 384x384 and 512x512 target sizes."""
    img = np.ones((700, 700, 3), dtype=np.uint8) * 128
    mask = np.ones((700, 700, 4), dtype=np.uint8)

    for p_size in (384, 512):
        coords = PatchCoordinates(y_min=50, y_max=50 + p_size, x_min=50, x_max=50 + p_size)
        img_patch, mask_patch, padded = extract_patch(img, mask, coords, target_size=p_size)
        assert img_patch.shape == (p_size, p_size, 3)
        assert mask_patch.shape == (p_size, p_size, 4)
        assert padded is False


def test_boundary_padding_zero_masks_and_binary_preservation():
    """Verify padding at image boundaries preserves strict binary {0, 1} values in mask."""
    img = np.full((100, 120, 3), 150, dtype=np.uint8)
    mask = np.ones((100, 120, 4), dtype=np.uint8)

    # Patch extends significantly beyond image bounds
    coords = PatchCoordinates(y_min=0, y_max=384, x_min=0, x_max=384)
    img_patch, mask_patch, padded = extract_patch(img, mask, coords, target_size=384)

    assert padded is True
    assert img_patch.shape == (384, 384, 3)
    assert mask_patch.shape == (384, 384, 4)

    # Padded mask regions must be strictly 0
    assert np.all(mask_patch[100:, :, :] == 0)
    assert np.all(mask_patch[:, 120:, :] == 0)

    # Mask values must strictly be {0, 1}
    unique_vals = np.unique(mask_patch)
    assert set(unique_vals).issubset({0, 1})


# =====================================================================
# 2. Lesion-Aware Sampling & Categorization Tests
# =====================================================================

def test_categorize_patch_rarity_hierarchy():
    """Verify that multi-lesion patches are categorized by rarest lesion (MA > SE > HE > EX)."""
    h, w = 64, 64
    mask = np.zeros((h, w, 4), dtype=np.uint8)

    # Case 1: Pure EX (Hard Exudates)
    mask[10:20, 10:20, 2] = 1
    cat, counts, has_lesion = categorize_patch(mask)
    assert cat == "EX"
    assert counts["EX"] == 100
    assert has_lesion is True

    # Case 2: Add HE (Hemorrhages) -> HE has higher priority than EX
    mask[25:35, 25:35, 1] = 1
    cat, counts, _ = categorize_patch(mask)
    assert cat == "HE"

    # Case 3: Add SE (Soft Exudates) -> SE has higher priority than HE and EX
    mask[40:45, 40:45, 3] = 1
    cat, counts, _ = categorize_patch(mask)
    assert cat == "SE"

    # Case 4: Add single MA pixel -> MA has highest priority
    mask[50, 50, 0] = 1
    cat, counts, _ = categorize_patch(mask)
    assert cat == "MA"
    assert counts["MA"] == 1


def test_categorize_patch_background_on_empty_mask():
    """Verify empty mask patch is categorized as 'background' with has_lesion=False."""
    mask = np.zeros((64, 64, 4), dtype=np.uint8)
    cat, counts, has_lesion = categorize_patch(mask)
    assert cat == "background"
    assert has_lesion is False
    assert all(c == 0 for c in counts.values())


# =====================================================================
# 3. Patch Dataset & Leakage Protection Tests
# =====================================================================

def _create_synthetic_record(tmp_path: Path, sample_id: str, split: str, has_ma: bool = False) -> dict[str, Any]:
    """Helper to generate a mock retinal sample on disk."""
    import cv2

    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    img_p = img_dir / f"{sample_id}.jpg"
    img = np.full((300, 300, 3), 100, dtype=np.uint8)
    cv2.imwrite(str(img_p), img)

    mask_paths = {}
    for ch in CANONICAL_LESION_ORDER:
        m_p = mask_dir / f"{sample_id}_{ch}.tif"
        m = np.zeros((300, 300), dtype=np.uint8)
        if ch == "MA" and has_ma:
            m[50:60, 50:60] = 255  # Add lesion
        cv2.imwrite(str(m_p), m)
        mask_paths[ch] = str(m_p)

    return {
        "dataset_name": "IDRiD",
        "sample_id": sample_id,
        "split": split,
        "image_path": str(img_p),
        "mask_paths": mask_paths,
        "channel_order": list(CANONICAL_LESION_ORDER),
    }


def test_data_leakage_rejection_validation_and_test(tmp_path: Path, monkeypatch):
    """Verify PatchRetinalSegmentationDataset strictly rejects validation and test records."""
    monkeypatch.chdir(tmp_path)

    val_record = _create_synthetic_record(tmp_path, "IDRiD_val", split="valid")
    test_record = _create_synthetic_record(tmp_path, "IDRiD_test", split="test")
    train_record = _create_synthetic_record(tmp_path, "IDRiD_train", split="train")

    # 1. Initializing with split="valid" or split="test" must raise ValueError
    with pytest.raises(ValueError, match="exclusively for training"):
        PatchRetinalSegmentationDataset(
            manifest_or_records=[val_record],
            split="valid",
            patch_size=384,
        )

    # 2. Including a validation or test record in training dataset must trigger leakage violation
    with pytest.raises(ValueError, match="Data leakage protection violation"):
        PatchRetinalSegmentationDataset(
            manifest_or_records=[train_record, val_record],
            split="train",
            patch_size=384,
        )

    with pytest.raises(ValueError, match="Data leakage protection violation"):
        PatchRetinalSegmentationDataset(
            manifest_or_records=[train_record, test_record],
            split="train",
            patch_size=384,
        )


def test_patch_dataset_pytorch_dataloader_integration(tmp_path: Path, monkeypatch):
    """Verify PatchRetinalSegmentationDataset integrates seamlessly with PyTorch DataLoader."""
    monkeypatch.chdir(tmp_path)

    rec1 = _create_synthetic_record(tmp_path, "IDRiD_train_1", split="train", has_ma=True)
    rec2 = _create_synthetic_record(tmp_path, "IDRiD_train_2", split="train", has_ma=False)

    ds = PatchRetinalSegmentationDataset(
        manifest_or_records=[rec1, rec2],
        split="train",
        patch_size=384,
        stride=192,
        background_ratio=0.30,
        epoch_samples=16,
        seed=42,
        cache_fov_images=True,
    )

    assert len(ds) == 16

    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
    batches = list(loader)

    assert len(batches) == 4
    for imgs, masks in batches:
        assert imgs.shape == (4, 3, 384, 384)
        assert masks.shape == (4, 4, 384, 384)
        assert imgs.dtype == torch.float32
        assert masks.dtype == torch.float32
        # Mask values must be binary {0.0, 1.0}
        unique_m = torch.unique(masks).tolist()
        assert set(unique_m).issubset({0.0, 1.0})


def test_deterministic_sampling_under_fixed_seed(tmp_path: Path, monkeypatch):
    """Verify that specifying seed produces identical patch sample sequences."""
    monkeypatch.chdir(tmp_path)

    rec1 = _create_synthetic_record(tmp_path, "IDRiD_train_1", split="train", has_ma=True)
    rec2 = _create_synthetic_record(tmp_path, "IDRiD_train_2", split="train", has_ma=False)

    ds1 = PatchRetinalSegmentationDataset(
        manifest_or_records=[rec1, rec2],
        split="train",
        patch_size=384,
        epoch_samples=10,
        seed=12345,
    )
    samples1 = [ds1.sample_patch_metadata().coordinates for _ in range(10)]

    ds2 = PatchRetinalSegmentationDataset(
        manifest_or_records=[rec1, rec2],
        split="train",
        patch_size=384,
        epoch_samples=10,
        seed=12345,
    )
    samples2 = [ds2.sample_patch_metadata().coordinates for _ in range(10)]

    assert samples1 == samples2


# =====================================================================
# 4. Sliding-Window Inference Engine Tests
# =====================================================================

class _MockConstantModel(nn.Module):
    """Dummy model returning constant logit values for all pixels."""

    def __init__(self, const_logit: float = 0.0):
        super().__init__()
        self.const_logit = float(const_logit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        return torch.full((b, 4, h, w), self.const_logit, dtype=torch.float32, device=x.device)


def test_sliding_window_reconstructed_dimensions_and_no_uncovered_pixels():
    """Verify that reconstructed probability map matches exact original dimensions."""
    h, w = 750, 1100
    dummy_img = np.full((h, w, 3), 120, dtype=np.uint8)

    model = _MockConstantModel(const_logit=0.0)  # sigmoid(0) = 0.50
    inferencer = SlidingWindowInferencer(
        model=model,
        patch_size=512,
        overlap=0.50,
        blending="gaussian",
        batch_size=2,
        device="cpu",
    )

    reconstructed = inferencer.predict_fov(dummy_img, return_numpy=True)

    assert isinstance(reconstructed, np.ndarray)
    assert reconstructed.shape == (4, h, w)
    assert reconstructed.dtype == np.float32
    # Verify no uncovered pixels or NaNs
    assert not np.isnan(reconstructed).any()
    assert np.all(reconstructed >= 0.0)
    assert np.all(reconstructed <= 1.0)


def test_sliding_window_constant_probability_reconstruction():
    """Verify that blending preserves flat probabilities across overlapping tiles."""
    h, w = 600, 800
    dummy_img = np.full((h, w, 3), 80, dtype=np.uint8)

    # Logit = 2.197224 -> sigmoid = 0.90
    model = _MockConstantModel(const_logit=2.197224577)
    inferencer = SlidingWindowInferencer(
        model=model,
        patch_size=384,
        overlap=0.50,
        blending="gaussian",
        batch_size=4,
        device="cpu",
    )

    probs = inferencer.predict_fov(dummy_img, return_numpy=True)

    # Because weights sum to 1.0 after normalization, constant probabilities must reconstruct to ~0.90
    assert np.allclose(probs, 0.90, atol=1e-3)


def test_sliding_window_handles_image_smaller_than_patch():
    """Verify sliding window handles images with height or width smaller than patch size."""
    h, w = 250, 300  # Smaller than 512
    dummy_img = np.full((h, w, 3), 90, dtype=np.uint8)

    model = _MockConstantModel(const_logit=0.0)
    inferencer = SlidingWindowInferencer(
        model=model,
        patch_size=512,
        overlap=0.50,
        blending="hann",
        batch_size=1,
        device="cpu",
    )

    reconstructed = inferencer.predict_fov(dummy_img, return_numpy=True)

    assert reconstructed.shape == (4, h, w)
    assert not np.isnan(reconstructed).any()
    assert np.allclose(reconstructed, 0.50, atol=1e-3)


def test_gaussian_and_hann_window_generation():
    """Verify mathematical properties of 2D Gaussian and Hann blending windows."""
    for p_size in (384, 512):
        g_win = create_gaussian_window(p_size)
        assert g_win.shape == (p_size, p_size)
        assert g_win.max() == pytest.approx(1.0)
        assert g_win.min() > 0.0
        # Peak at center
        cy, cx = p_size // 2, p_size // 2
        assert g_win[cy, cx] == pytest.approx(1.0)
        assert g_win[0, 0] < g_win[cy, cx]

        h_win = create_hann_window(p_size)
        assert h_win.shape == (p_size, p_size)
        assert h_win.max() == pytest.approx(1.0)
        assert h_win[cy, cx] == pytest.approx(1.0)


def test_real_idrid_patch_dataset_and_sliding_window():
    """Verify patch extraction and sliding-window inference on real IDRiD training records."""
    from src.federated.data import resolve_client_train_val_records
    from src.models.unet import ResNet50UNet

    train_recs, val_recs, _ = resolve_client_train_val_records(
        client_id="client_1",
        partition_manifest="data/partitions/combined_cross_dataset_2clients_seed42.json",
        source_manifest="data/processed/manifests/combined_manifest.json",
        val_split_mode="official_and_carved",
        val_carve_ratio=0.2,
        seed=42,
    )
    assert len(train_recs) == 44
    assert len(val_recs) == 10

    # Test patch dataset on first 2 real training records for fast execution
    ds = PatchRetinalSegmentationDataset(
        manifest_or_records=train_recs[:2],
        split="train",
        patch_size=512,
        stride=512,
        epoch_samples=4,
        seed=42,
        cache_fov_images=False,
    )
    assert len(ds) == 4
    img_patch, mask_patch = ds[0]
    assert img_patch.shape == (3, 512, 512)
    assert mask_patch.shape == (4, 512, 512)

    # Test sliding window inference on real IDRiD validation sample
    from src.data.masks import load_sample_image_and_mask
    from src.data.fov import crop_image_and_mask_to_fov
    from src.utils.paths import resolve_project_path

    v_rec = val_recs[0]
    img_path = resolve_project_path(v_rec["image_path"])
    mask_paths = {
        ch: resolve_project_path(v_rec["mask_paths"][ch])
        if v_rec["mask_paths"].get(ch) is not None
        else None
        for ch in CANONICAL_LESION_ORDER
    }

    img_rgb, mask_res = load_sample_image_and_mask(img_path, mask_paths, threshold=0)
    cropped_img, cropped_mask, _ = crop_image_and_mask_to_fov(img_rgb, mask_res.mask)

    model = ResNet50UNet(pretrained=False)
    inferencer = SlidingWindowInferencer(
        model=model,
        patch_size=512,
        overlap=0.50,
        blending="gaussian",
        batch_size=2,
        device="cpu",
    )

    pred_probs = inferencer.predict_fov(cropped_img, return_numpy=True)
    assert pred_probs.shape == (4, cropped_img.shape[0], cropped_img.shape[1])
    assert not np.isnan(pred_probs).any()
    assert (pred_probs >= 0.0).all() and (pred_probs <= 1.0).all()


def test_sampling_frequency_and_duplicate_sampling_behaviour(tmp_path: Path, monkeypatch):
    """Verify duplicate sampling behaviour with replacement and empirical background ratio."""
    monkeypatch.chdir(tmp_path)

    rec1 = _create_synthetic_record(tmp_path, "IDRiD_train_1", split="train", has_ma=True)
    rec2 = _create_synthetic_record(tmp_path, "IDRiD_train_2", split="train", has_ma=False)

    ds = PatchRetinalSegmentationDataset(
        manifest_or_records=[rec1, rec2],
        split="train",
        patch_size=384,
        epoch_samples=200,
        background_ratio=0.30,
        seed=999,
    )
    # Total candidate patches is small (~2-4), but epoch_samples is 200.
    # Sampling with replacement must yield 200 valid items without crashing.
    categories = [ds.sample_patch_metadata().primary_category for _ in range(200)]
    assert len(categories) == 200
    bg_count = categories.count("background")
    empirical_bg_ratio = bg_count / 200.0
    # Expected ~0.30; allow statistical margin [0.20, 0.42]
    assert 0.20 <= empirical_bg_ratio <= 0.42


def test_sliding_window_hann_blending_constant_reconstruction():
    """Verify that Hann blending window also reconstructs flat probabilities accurately."""
    h, w = 600, 800
    dummy_img = np.full((h, w, 3), 80, dtype=np.uint8)

    # Logit = 0.0 -> sigmoid = 0.50
    model = _MockConstantModel(const_logit=0.0)
    inferencer = SlidingWindowInferencer(
        model=model,
        patch_size=384,
        overlap=0.50,
        blending="hann",
        batch_size=2,
        device="cpu",
    )

    probs = inferencer.predict_fov(dummy_img, return_numpy=True)
    assert probs.shape == (4, h, w)
    assert not np.isnan(probs).any()
    assert np.allclose(probs, 0.50, atol=1e-3)


class _MockStateTrackingModel(nn.Module):
    """Mock model that records its training mode during forward pass and can simulate exceptions."""

    def __init__(self, should_fail: bool = False):
        super().__init__()
        self.should_fail = should_fail
        self.training_states_during_forward: list[bool] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.training_states_during_forward.append(bool(self.training))
        if self.should_fail:
            raise RuntimeError("Simulated failure inside forward pass.")
        b, _, h, w = x.shape
        return torch.zeros((b, 4, h, w), dtype=torch.float32, device=x.device)


def test_sliding_window_restores_initial_training_mode():
    """Verify inferencer runs forward pass in eval mode but restores training mode afterwards."""
    model = _MockStateTrackingModel(should_fail=False)
    model.train()
    assert model.training is True

    inferencer = SlidingWindowInferencer(
        model=model,
        patch_size=384,
        batch_size=1,
        device="cpu",
    )
    _ = inferencer.predict_fov(np.zeros((400, 400, 3), dtype=np.uint8))

    # Reconstructed in eval mode during inference
    assert len(model.training_states_during_forward) > 0
    assert all(state is False for state in model.training_states_during_forward)
    # Restored to original train mode after predict_fov finishes
    assert model.training is True


def test_sliding_window_preserves_initial_eval_mode():
    """Verify inferencer preserves evaluation mode if model was initially in eval mode."""
    model = _MockStateTrackingModel(should_fail=False)
    model.eval()
    assert model.training is False

    inferencer = SlidingWindowInferencer(
        model=model,
        patch_size=384,
        batch_size=1,
        device="cpu",
    )
    _ = inferencer.predict_fov(np.zeros((400, 400, 3), dtype=np.uint8))

    assert len(model.training_states_during_forward) > 0
    assert all(state is False for state in model.training_states_during_forward)
    # Model remains in eval mode
    assert model.training is False


def test_sliding_window_restores_state_on_inference_exception():
    """Verify model training state is restored even if an exception occurs during inference."""
    # Case A: Model initially in training mode
    model_train = _MockStateTrackingModel(should_fail=True)
    model_train.train()
    assert model_train.training is True

    inferencer_train = SlidingWindowInferencer(
        model=model_train,
        patch_size=384,
        batch_size=1,
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="Simulated failure inside forward pass"):
        _ = inferencer_train.predict_fov(np.zeros((400, 400, 3), dtype=np.uint8))
    # Must be safely restored to training mode
    assert model_train.training is True

    # Case B: Model initially in evaluation mode
    model_eval = _MockStateTrackingModel(should_fail=True)
    model_eval.eval()
    assert model_eval.training is False

    inferencer_eval = SlidingWindowInferencer(
        model=model_eval,
        patch_size=384,
        batch_size=1,
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="Simulated failure inside forward pass"):
        _ = inferencer_eval.predict_fov(np.zeros((400, 400, 3), dtype=np.uint8))
    # Must remain in evaluation mode
    assert model_eval.training is False


def test_idrid_44_records_strict_split_isolation():
    """Verify that IDRiD Client 1 training records are strictly disjoint from validation and test."""
    from src.federated.data import resolve_client_train_val_records
    from src.data.manifests import load_manifest_json

    train_recs, val_recs, info = resolve_client_train_val_records(
        client_id="client_1",
        partition_manifest="data/partitions/combined_cross_dataset_2clients_seed42.json",
        source_manifest="data/processed/manifests/combined_manifest.json",
        val_split_mode="official_and_carved",
        val_carve_ratio=0.2,
        seed=42,
    )

    train_ids = {r["sample_id"] for r in train_recs}
    val_ids = {r["sample_id"] for r in val_recs}
    assert len(train_ids) == 44
    assert len(val_ids) == 10
    # Zero intersection between train and carved val
    assert len(train_ids.intersection(val_ids)) == 0

    # Official IDRiD test records
    idrid_manifest = load_manifest_json("data/processed/manifests/idrid_manifest.json")
    test_ids = {r["sample_id"] for r in idrid_manifest["records"] if r["split"] == "test"}
    assert len(test_ids) == 27

    # Zero intersection with official test set
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0



