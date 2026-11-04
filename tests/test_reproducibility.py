"""Unit tests for reproducibility and random seed utilities (src/utils/seed.py)."""

from __future__ import annotations

import os
import random
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.utils.config import load_yaml_config
from src.utils.device import is_pytorch_available
from src.utils.seed import MAX_UINT32_SEED, seed_worker, set_seed


def test_set_seed_accepts_valid_integer() -> None:
    """Verify set_seed accepts valid integers and updates PYTHONHASHSEED."""
    applied = set_seed(42, deterministic=True)
    assert applied == 42
    assert os.environ.get("PYTHONHASHSEED") == "42"

    assert set_seed(0, deterministic=False) == 0
    assert set_seed(MAX_UINT32_SEED, deterministic=True) == MAX_UINT32_SEED


@pytest.mark.parametrize("invalid_seed", [-1, MAX_UINT32_SEED + 1])
def test_set_seed_rejects_out_of_range_integers(invalid_seed: int) -> None:
    """Verify out-of-range integer seeds raise ValueError."""
    with pytest.raises(ValueError, match="between 0 and"):
        set_seed(invalid_seed)


@pytest.mark.parametrize("invalid_type", ["42", 42.0, None, True, False, [42]])
def test_set_seed_rejects_non_integer_types(invalid_type: object) -> None:
    """Verify non-integer seeds (including booleans) raise TypeError."""
    with pytest.raises(TypeError, match="Random seed must be an integer"):
        set_seed(invalid_type)  # type: ignore[arg-type]


def test_set_seed_rejects_non_boolean_deterministic_flag() -> None:
    """Verify non-boolean deterministic flag raises TypeError."""
    with pytest.raises(TypeError, match="'deterministic' must be a boolean"):
        set_seed(42, deterministic="yes")  # type: ignore[arg-type]


def test_python_random_operations_are_reproducible() -> None:
    """Verify repeated Python random operations produce identical sequences."""
    set_seed(1337)
    seq_a = [random.random() for _ in range(10)]
    ints_a = [random.randint(0, 10000) for _ in range(10)]

    set_seed(1337)
    seq_b = [random.random() for _ in range(10)]
    ints_b = [random.randint(0, 10000) for _ in range(10)]

    assert seq_a == seq_b
    assert ints_a == ints_b

    set_seed(9999)
    seq_c = [random.random() for _ in range(10)]
    assert seq_a != seq_c


def test_numpy_random_operations_are_reproducible() -> None:
    """Verify repeated NumPy random operations produce identical arrays."""
    set_seed(2026)
    arr_a = np.random.rand(4, 4)
    norm_a = np.random.normal(loc=0.0, scale=1.0, size=8)

    set_seed(2026)
    arr_b = np.random.rand(4, 4)
    norm_b = np.random.normal(loc=0.0, scale=1.0, size=8)

    np.testing.assert_array_equal(arr_a, arr_b)
    np.testing.assert_array_equal(norm_a, norm_b)

    set_seed(7)
    arr_c = np.random.rand(4, 4)
    assert not np.array_equal(arr_a, arr_c)


def test_set_seed_without_pytorch_installed() -> None:
    """Verify set_seed works gracefully when PyTorch is not installed."""
    with patch("src.utils.seed._get_torch", return_value=None):
        result = set_seed(101, deterministic=True)
        assert result == 101


def test_set_seed_configures_mock_pytorch_and_cuda_deterministic() -> None:
    """Verify set_seed configures PyTorch, CUDA seeds, and cuDNN deterministic flags."""
    mock_cudnn = SimpleNamespace(deterministic=False, benchmark=True)
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.backends = SimpleNamespace(cudnn=mock_cudnn)

    with patch("src.utils.seed._get_torch", return_value=mock_torch):
        set_seed(77, deterministic=True)

        mock_torch.manual_seed.assert_called_once_with(77)
        mock_torch.cuda.manual_seed.assert_called_once_with(77)
        mock_torch.cuda.manual_seed_all.assert_called_once_with(77)
        mock_torch.use_deterministic_algorithms.assert_called_once_with(
            True, warn_only=True
        )
        assert mock_cudnn.deterministic is True
        assert mock_cudnn.benchmark is False

        set_seed(77, deterministic=False)
        assert mock_cudnn.deterministic is False
        assert mock_cudnn.benchmark is True


def test_seed_worker_does_not_crash_and_is_reproducible() -> None:
    """Verify seed_worker runs cleanly and produces reproducible worker RNG streams."""
    set_seed(42)
    seed_worker(0)
    py_val_w0_a = random.random()
    np_val_w0_a = float(np.random.rand())

    set_seed(42)
    seed_worker(0)
    py_val_w0_b = random.random()
    np_val_w0_b = float(np.random.rand())

    assert py_val_w0_a == py_val_w0_b
    assert np_val_w0_a == np_val_w0_b

    # Different worker IDs get distinct streams
    seed_worker(1)
    py_val_w1 = random.random()
    assert py_val_w0_a != py_val_w1


def test_seed_worker_with_mock_pytorch_initial_seed() -> None:
    """Verify seed_worker derives its seed from torch.initial_seed() when available."""
    mock_torch = MagicMock()
    mock_torch.initial_seed.return_value = (2**32) + 555

    with patch("src.utils.seed._get_torch", return_value=mock_torch):
        seed_worker(2)
        mock_torch.initial_seed.assert_called_once()
        val_a = random.random()

        seed_worker(2)
        val_b = random.random()
        assert val_a == val_b


@pytest.mark.parametrize("bad_worker_id", ["0", None, True, 1.5])
def test_seed_worker_rejects_non_integer(bad_worker_id: object) -> None:
    """Verify seed_worker raises TypeError for non-integer worker IDs."""
    with pytest.raises(TypeError, match="'worker_id' must be an integer"):
        seed_worker(bad_worker_id)  # type: ignore[arg-type]


def test_seed_worker_rejects_negative_worker_id() -> None:
    """Verify seed_worker raises ValueError for negative worker IDs."""
    with pytest.raises(ValueError, match="non-negative"):
        seed_worker(-1)


def test_project_yaml_reproducibility_config_integration() -> None:
    """Verify configs/project.yaml contains random_seed and deterministic and works with set_seed."""
    project_cfg = load_yaml_config("project.yaml")
    assert "random_seed" in project_cfg
    assert "deterministic" in project_cfg
    assert project_cfg["random_seed"] == 42
    assert project_cfg["deterministic"] is True

    applied = set_seed(
        seed=project_cfg["random_seed"],
        deterministic=project_cfg["deterministic"],
    )
    assert applied == 42


@pytest.mark.skipif(not is_pytorch_available(), reason="PyTorch is not installed")
def test_pytorch_tensor_reproducibility_when_installed() -> None:
    """Verify PyTorch random tensor generation is reproducible when PyTorch is installed."""
    import torch

    set_seed(31415)
    tensor_a = torch.rand(4, 4)

    set_seed(31415)
    tensor_b = torch.rand(4, 4)

    assert torch.equal(tensor_a, tensor_b)

